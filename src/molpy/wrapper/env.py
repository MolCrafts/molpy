"""Process-environment isolation for external tool wrappers.

Users configure isolation **explicitly** with ``env`` + ``env_manager``.
There is no auto-detection of manager type from paths or tool layout.

Supported managers
------------------
* ``None`` + ``env is None`` — system environment (current ``PATH``).
* ``"conda"`` — ``conda run -n <name>`` or ``conda run -p <prefix>``.
* ``"venv"`` / ``"pip"`` / ``"virtualenv"`` — inject ``<prefix>/bin``
  (or ``Scripts`` on Windows) into ``PATH``.  Covers standard venv,
  virtualenv, and uv-created environments (same layout).

This module is the single owner of env-isolation logic used by
:class:`~molpy.wrapper.base.Wrapper` and higher-level facades that
construct wrappers.
"""

from __future__ import annotations

import os
import shutil
import subprocess
from dataclasses import dataclass
from pathlib import Path
from typing import Literal, Mapping

ManagerKind = Literal["conda", "venv"]

_VENV_ALIASES = frozenset({"venv", "pip", "virtualenv"})
_SUPPORTED = "'conda', 'venv' (aliases: 'pip', 'virtualenv')"


def _looks_like_path(value: str) -> bool:
    """Best-effort: path separators or relative/absolute prefixes."""
    return (
        (os.sep in value)
        or ("/" in value)
        or ("\\" in value)
        or value.startswith((".", "~", "/"))
    )


def _bin_dir(prefix: Path) -> Path:
    return prefix / ("Scripts" if os.name == "nt" else "bin")


def _conda_exe() -> str:
    """Return a conda executable path, or the bare name ``"conda"``."""
    conda_exe = os.environ.get("CONDA_EXE")
    if conda_exe:
        return conda_exe
    found = shutil.which("conda")
    return found if found is not None else "conda"


@dataclass(frozen=True, slots=True)
class EnvSpec:
    """Validated subprocess environment isolation.

    Construct via :meth:`resolve` (or :meth:`system`).  Fields after
    resolve are either both ``None`` (system) or a concrete manager kind
    plus a non-``None`` ``env``.
    """

    env: str | Path | None = None
    env_manager: ManagerKind | None = None

    # -- construction -------------------------------------------------------

    @classmethod
    def system(cls) -> EnvSpec:
        """No isolation — use the current process environment."""
        return cls()

    @classmethod
    def resolve(
        cls,
        env: str | Path | None = None,
        env_manager: str | None = None,
    ) -> EnvSpec:
        """Validate and normalise user ``env`` / ``env_manager`` input.

        Both must be set together, or both omitted for the system
        environment.  Manager type is never inferred from ``env``.

        Args:
            env: Conda env name / prefix, or venv prefix path.
            env_manager: ``"conda"``, ``"venv"``, ``"pip"``, or
                ``"virtualenv"``.

        Returns:
            A frozen :class:`EnvSpec`.

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
        if manager in _VENV_ALIASES:
            kind: ManagerKind = "venv"
        elif manager == "conda":
            kind = "conda"
        else:
            raise ValueError(
                f"Unsupported env_manager {env_manager!r}. "
                f"Supported values: {_SUPPORTED}."
            )
        return cls(env=env, env_manager=kind)

    # -- queries ------------------------------------------------------------

    @property
    def is_system(self) -> bool:
        """True when no isolation is configured."""
        return self.env_manager is None

    # -- subprocess helpers -------------------------------------------------

    def command_prefix(self, *, no_capture_output: bool = False) -> list[str]:
        """Return an argv prefix for the configured manager, or ``[]``.

        For conda this is ``[conda, run, (-n|-p), <env>]``.  For venv /
        system the prefix is empty (isolation is via :meth:`merge_environ`).

        Args:
            no_capture_output: When True, insert
                ``--no-capture-output`` after ``run`` (useful for engines
                that stream logs).
        """
        if self.env_manager != "conda":
            return []

        assert self.env is not None
        conda = _conda_exe()
        prefix = [conda, "run"]
        if no_capture_output:
            prefix.append("--no-capture-output")

        if isinstance(self.env, Path):
            return [*prefix, "-p", str(self.env)]

        env_str = str(self.env)
        if _looks_like_path(env_str):
            return [*prefix, "-p", env_str]
        return [*prefix, "-n", env_str]

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
            assert self.env is not None
            prefix = self.env if isinstance(self.env, Path) else Path(str(self.env))
            bin_dir = _bin_dir(prefix)
            existing = merged.get("PATH", "")
            merged["PATH"] = str(bin_dir) + (os.pathsep + existing if existing else "")
            merged["VIRTUAL_ENV"] = str(prefix)

        if extra:
            merged.update(extra)
        return merged

    def resolve_executable(self, exe: str) -> str | None:
        """Best-effort absolute path for *exe* inside this environment.

        Returns ``None`` if the executable cannot be found.
        """
        exe_path = Path(exe)
        if exe_path.is_file():
            return str(exe_path)

        if self.is_system:
            return shutil.which(exe)

        if self.env_manager == "venv":
            assert self.env is not None
            prefix = self.env if isinstance(self.env, Path) else Path(str(self.env))
            bin_dir = _bin_dir(prefix)
            candidates = [bin_dir / exe]
            if os.name == "nt" and not exe.lower().endswith(".exe"):
                candidates.append(bin_dir / f"{exe}.exe")
            for candidate in candidates:
                if candidate.is_file():
                    return str(candidate)
            return shutil.which(exe, path=self.merge_environ().get("PATH"))

        # conda: ask the env via ``conda run … which``
        cmd_prefix = self.command_prefix()
        try:
            proc = subprocess.run(
                [*cmd_prefix, "which", exe],
                capture_output=True,
                text=True,
                check=False,
                env=self.merge_environ(),
            )
        except OSError:
            return None
        if proc.returncode != 0:
            return None
        lines = (proc.stdout or "").strip().splitlines()
        return lines[0] if lines else None

    def __repr__(self) -> str:
        if self.is_system:
            return "EnvSpec(system)"
        return f"EnvSpec(env={self.env!r}, env_manager={self.env_manager!r})"
