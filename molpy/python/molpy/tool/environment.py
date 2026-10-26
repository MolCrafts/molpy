"""Process-environment isolation for the external programs molpy runs.

Isolation is configured **explicitly** with ``env`` + ``env_manager`` — in
molpy's configuration (:mod:`molpy.config`), which every engine and AmberTools program
resolves its :class:`EnvironmentSpec` from. There is no auto-detection of
manager type from paths or tool layout.

Supported managers
------------------
* ``None`` + ``env is None`` — system environment (current ``PATH``).
* ``"conda"`` — ``conda run -n <name>`` or ``conda run -p <prefix>``, with
  the ``conda`` named by ``conda.executable`` (default: ``conda`` on
  ``PATH``).
* ``"venv"`` — inject ``<prefix>/bin`` (or ``Scripts`` on Windows) into
  ``PATH``.  Covers standard venv, virtualenv, and uv-created environments
  (same layout).

Path contract
-------------
Public entry points accept ``str | Path`` (path-like).  On resolve, any
value that is a path (or looks like one) is converted to
:class:`~pathlib.Path` and stored/used as ``Path`` only.  Conda *env
names* (no path separators) remain plain ``str``.  Subprocess argv
receives ``str(path)`` at the OS boundary only.

This module is the single owner of env-isolation logic used by
:class:`~molpy.tool.tool.Tool`, so by every engine and AmberTools program.
"""

from __future__ import annotations

import os
import shutil
from dataclasses import dataclass, field
from pathlib import Path
from typing import Literal, Mapping

import mollog

from molpy.tool.process import ToolError, run_process

ManagerKind = Literal["conda", "venv"]

_SUPPORTED = "'conda', 'venv'"

#: Seconds a ``conda run … which`` lookup may take (conda starts slowly).
RESOLVE_TIMEOUT = 60.0

#: Where a lookup logs when its caller passes no logger.
_LOGGER = "molpy.tool.environment"


def _looks_like_path(value: str) -> bool:
    """Best-effort: path separators or relative/absolute prefixes."""
    return (
        (os.sep in value)
        or ("/" in value)
        or ("\\" in value)
        or value.startswith((".", "~"))
        or (len(value) >= 2 and value[1] == ":")  # Windows drive letter
    )


def _as_path(value: str | Path) -> Path:
    """Convert path-like input to :class:`Path` (expand ``~``)."""
    if isinstance(value, Path):
        return value.expanduser()
    return Path(value).expanduser()


def _bin_dir(prefix: Path) -> Path:
    return prefix / ("Scripts" if os.name == "nt" else "bin")


@dataclass(frozen=True)
class EnvironmentSpec:
    """Validated subprocess environment isolation.

    Construct via :meth:`resolve` (or :meth:`system`).  After resolve:

    * system — both fields ``None``
    * conda name — ``env: str``, ``env_manager: "conda"``
    * conda prefix / venv — ``env: Path``, ``env_manager: "conda"|"venv"``

    ``conda_executable`` is the ``conda`` that ``conda run`` invokes.
    :meth:`resolve_executable` remembers each answer for the life of the
    instance (each engine and AmberTools program resolves its own).
    """

    env: str | Path | None = None
    env_manager: ManagerKind | None = None
    conda_executable: str = "conda"
    _found: dict[str, str | None] = field(
        default_factory=dict, init=False, repr=False, compare=False, hash=False
    )

    # -- construction -------------------------------------------------------

    @classmethod
    def system(cls) -> EnvironmentSpec:
        """No isolation — use the current process environment."""
        return cls()

    @classmethod
    def resolve(
        cls,
        env: str | Path | None = None,
        env_manager: str | None = None,
        *,
        conda_executable: str = "conda",
    ) -> EnvironmentSpec:
        """Validate and normalise user ``env`` / ``env_manager`` input.

        Both must be set together, or both omitted for the system
        environment.  Manager type is never inferred from ``env``.
        Path-like values are stored as :class:`~pathlib.Path`.

        Args:
            env: Conda env name / prefix, or venv prefix (``str | Path``).
            env_manager: ``"conda"`` or ``"venv"``.
            conda_executable: The ``conda`` that ``conda run`` invokes.

        Returns:
            A frozen :class:`EnvironmentSpec`.

        Raises:
            ValueError: If exactly one of the pair is set, or the manager
                string is unsupported.
        """
        if env is None and env_manager is None:
            return cls.system()

        if env is None or env_manager is None:
            raise ValueError(
                "Environment configuration is incomplete: set both env and "
                "env_manager, or set neither for the system environment.  "
                f"Got env={env!r}, env_manager={env_manager!r}."
            )

        manager = env_manager.strip().lower()
        if manager == "venv":
            kind: ManagerKind = "venv"
        elif manager == "conda":
            kind = "conda"
        else:
            raise ValueError(
                f"Unsupported env_manager {env_manager!r}. "
                f"Supported values: {_SUPPORTED}."
            )

        # venv always needs a filesystem prefix → Path
        if kind == "venv":
            return cls(env=_as_path(env), env_manager=kind)

        # conda: Path / path-like str → Path prefix; bare name stays str
        if isinstance(env, Path) or _looks_like_path(str(env)):
            return cls(
                env=_as_path(env), env_manager=kind, conda_executable=conda_executable
            )
        return cls(env=str(env), env_manager=kind, conda_executable=conda_executable)

    # -- queries ------------------------------------------------------------

    @property
    def is_system(self) -> bool:
        """True when no isolation is configured."""
        return self.env_manager is None

    def _prefix_path(self) -> Path:
        """Filesystem prefix for venv / conda -p (must be a :class:`Path`)."""
        assert self.env is not None
        if not isinstance(self.env, Path):
            raise TypeError(
                f"EnvironmentSpec.env must be Path for manager {self.env_manager!r}, "
                f"got {type(self.env).__name__}: {self.env!r}"
            )
        return self.env

    # -- subprocess helpers -------------------------------------------------

    def command_prefix(self, *, no_capture_output: bool = False) -> list[str]:
        """Return an argv prefix for the configured manager, or ``[]``.

        For conda this is ``[conda, run, (-n|-p), <env>]``.  For venv /
        system the prefix is empty (isolation is via :meth:`merge_environ`).
        Path prefixes are emitted as ``str(path)`` for the OS.

        Args:
            no_capture_output: When True, insert
                ``--no-capture-output`` after ``run`` (useful for engines
                that stream logs).
        """
        if self.env_manager != "conda":
            return []

        assert self.env is not None
        prefix = [self.conda_executable, "run"]
        if no_capture_output:
            prefix.append("--no-capture-output")

        if isinstance(self.env, Path):
            return [*prefix, "-p", str(self.env)]
        return [*prefix, "-n", self.env]

    def merge_environ(
        self,
        base: Mapping[str, str] | None = None,
        extra: Mapping[str, str] | None = None,
    ) -> dict[str, str]:
        """Build a subprocess environment dict.

        Merge order (later wins): *base* (default ``os.environ``) →
        venv ``PATH`` / ``VIRTUAL_ENV`` injection when applicable →
        *extra*.
        """
        merged = dict(base if base is not None else os.environ)

        if self.env_manager == "venv":
            prefix = self._prefix_path()
            bin_dir = _bin_dir(prefix)
            existing = merged.get("PATH", "")
            merged["PATH"] = str(bin_dir) + (os.pathsep + existing if existing else "")
            merged["VIRTUAL_ENV"] = str(prefix)

        if extra:
            merged.update(extra)
        return merged

    def resolve_executable(
        self, exe: str | Path, *, logger: mollog.Logger | None = None
    ) -> str | None:
        """Best-effort absolute path for *exe* inside this environment.

        *exe* may be path-like at the public boundary; filesystem paths are
        converted to :class:`~pathlib.Path` immediately.  A conda env is
        asked with ``conda run … which`` through
        :func:`~molpy.tool.process.run_process` (logged at DEBUG to
        *logger*, killed after :data:`RESOLVE_TIMEOUT` seconds).  Each answer
        is remembered by this instance.

        Args:
            exe: A program name or a path.
            logger: Where the conda lookup logs; ``None`` is
                ``molpy.tool.environment``.

        Returns:
            A ``str`` suitable for subprocess argv, or ``None`` if not found
            (or the lookup could not run).
        """
        key = str(exe)
        if key not in self._found:
            self._found[key] = self._lookup(exe, logger)
        return self._found[key]

    def _lookup(self, exe: str | Path, logger: mollog.Logger | None) -> str | None:
        # Explicit path (Path object or path-like string) — no PATH search.
        if isinstance(exe, Path) or (isinstance(exe, str) and _looks_like_path(exe)):
            exe_path = _as_path(exe)
            return str(exe_path.resolve()) if exe_path.is_file() else None

        name = str(exe)

        if self.is_system:
            return shutil.which(name)

        if self.env_manager == "venv":
            prefix = self._prefix_path()
            bin_dir = _bin_dir(prefix)
            candidates = [bin_dir / name]
            if os.name == "nt" and not name.lower().endswith(".exe"):
                candidates.append(bin_dir / f"{name}.exe")
            for candidate in candidates:
                if candidate.is_file():
                    return str(candidate.resolve())
            return shutil.which(name, path=self.merge_environ().get("PATH"))

        # conda: ask the env via ``conda run … which``
        log = logger if logger is not None else mollog.get_logger(_LOGGER)
        try:
            proc = run_process(
                log,
                [*self.command_prefix(), "which", name],
                tool=f"conda env {self.env}",
                cwd=None,
                env=None,
                capture_output=True,
                check=False,
                timeout=RESOLVE_TIMEOUT,
                quiet=True,
                lookup=name,
            )
        except ToolError as exc:  # conda missing or hung: say so, report not found
            log.warning("executable lookup failed", lookup=name, error=str(exc))
            return None
        if proc.returncode != 0:
            return None
        lines = (proc.stdout or "").strip().splitlines()
        return lines[0] if lines else None

    def __repr__(self) -> str:
        if self.is_system:
            return "EnvironmentSpec(system)"
        return f"EnvironmentSpec(env={self.env!r}, env_manager={self.env_manager!r})"
