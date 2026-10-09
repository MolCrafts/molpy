"""
Engine base classes for molecular simulation engines.

Provides :class:`Engine`, an abstract base for running external computational
chemistry programs (LAMMPS, CP2K, OpenMM, …).  Each concrete engine handles
command construction, file management, and subprocess execution for its
specific program.

The two supported usage modes are:

1. **Generate-only** — write input files to disk without executing anything::

       paths = engine.generate_inputs(frame, ff, config, "./output")

2. **Execute** — write files *and* run the engine subprocess::

       result = engine.run(script, workdir="./calc")

MPI and job-scheduler launchers, the executable, its conda / venv
environment, environment variables and time limit are the engine's
``engine.<name>`` settings in molpy's configuration (:mod:`molpy.config`)::

    # molpy.toml
    [engine.lammps]
    launcher = ["mpirun", "-np", "16"]

Every subprocess an engine starts is logged to ``molpy.engine.<name>``
(:mod:`mollog`).
"""

from __future__ import annotations

import subprocess
import tempfile
from abc import ABC, abstractmethod
from collections.abc import Sequence
from pathlib import Path
from typing import TYPE_CHECKING, Any, ClassVar

import mollog

from molpy.config import load_config, tool_settings
from molpy.wrapper import EnvironmentSpec
from molpy.wrapper._process import run_process

from ._script import Script

if TYPE_CHECKING:
    from molcfg import Config


class Engine(ABC):
    """Abstract base class for computational chemistry engines.

    Concrete subclasses name their configuration table in :attr:`tool`
    (``"lammps"`` reads ``[engine.lammps]``, falling back to ``[engine]``) and
    implement :meth:`_execute` and :meth:`_get_default_extension`. The base
    class resolves the settings, normalises scripts, manages the working
    directory, prefixes commands (environment wrapper + launcher) and logs
    every subprocess.

    Args:
        config: molpy's configuration (:func:`molpy.config.load_config`);
            ``None`` loads it (package defaults, user and project files).
        workdir: Default working directory.  ``None`` creates a temporary
            directory on each :meth:`run` call.
        check_executable: Verify the executable is available at construction
            time (system ``PATH`` or the configured env).  Set to ``False``
            when only writing inputs, or when the binary is only available
            on a remote node.

    Attributes:
        settings: The resolved :class:`~molpy.config.ToolSettings`.
        executable: Path or command to the engine binary.
        work_dir: Default working directory; ``None`` means a temporary
            directory is created on each :meth:`run` call.
        launcher: MPI / scheduler prefix inserted before the executable,
            e.g. ``["mpirun", "-np", "16"]`` (empty for none).
        env_vars: Extra environment variables forwarded to the subprocess.
        timeout: Seconds before a subprocess is killed (``None``: no limit).
        environment: The :class:`~molpy.wrapper.EnvironmentSpec` the engine
            runs in.
        logger: The engine's logger, ``molpy.engine.<tool>``.
        scripts: Scripts registered by the last :meth:`run` call (or ``[]``
            before the first call).
        input_script: Primary input script resolved by the last :meth:`run`
            call (or ``None`` before the first call).

    Raises:
        FileNotFoundError: If *check_executable* is ``True`` and the
            executable is not found.
        ValueError: If the configured environment is incomplete or its
            manager unsupported.

    Example:
        >>> from molpy.config import load_config
        >>> from molpy.engine import LammpsEngine, Script
        >>>
        >>> script = Script.from_text(
        ...     name="input",
        ...     text="units real\\natom_style full\\n",
        ...     language="other",
        ... )
        >>> config = load_config({"engine": {"lammps": {"executable": "lmp"}}})
        >>> engine = LammpsEngine(config=config, check_executable=False)
        >>> result = engine.run(script, workdir="./calc", check=False)
        >>> print(result.returncode)
        0
    """

    tool: ClassVar[str]

    def __init__(
        self,
        *,
        config: Config | None = None,
        workdir: str | Path | None = None,
        check_executable: bool = True,
    ) -> None:
        self.settings = tool_settings(
            config if config is not None else load_config(), f"engine.{self.tool}"
        )
        self.environment = EnvironmentSpec.resolve(
            self.settings.env,
            self.settings.env_manager,
            conda_executable=self.settings.conda_executable,
        )
        self.executable: str = (
            self.settings.executable
            if self.settings.executable is not None
            else self._default_executable()
        )
        self.work_dir = Path(workdir) if workdir is not None else None
        self.launcher: list[str] = list(self.settings.launcher)
        self.env_vars: dict[str, str] = dict(self.settings.env_vars)
        self.timeout = self.settings.timeout
        self.logger = mollog.get_logger(f"molpy.engine.{self.tool}")

        # Initialised here so attribute access is always valid.
        self.scripts: list[Script] = []
        self.input_script: Script | None = None

        if check_executable:
            self.check_executable()

    # ------------------------------------------------------------------
    # Abstract interface
    # ------------------------------------------------------------------

    @property
    @abstractmethod
    def name(self) -> str:
        """Human-readable engine name (e.g. ``"LAMMPS"``).

        Returns:
            A short, stable identifier used for ``__repr__``.
        """

    @abstractmethod
    def _get_default_extension(self) -> str:
        """File extension used when saving an unnamed script to disk.

        Returns:
            Extension string including the leading dot (e.g. ``".lmp"``).
        """

    @abstractmethod
    def _execute(
        self,
        run_dir: Path,
        capture_output: bool = False,
        check: bool = True,
        **kwargs: Any,
    ) -> subprocess.CompletedProcess:
        """Run the engine subprocess.

        Called by :meth:`run` after scripts have been written to *run_dir*.
        Subclasses build the concrete command with
        :meth:`_build_full_command` (environment wrapper + launcher +
        executable + engine flags) and start it with :meth:`_run_process`.

        Args:
            run_dir: Directory where input files have been written; use as
                ``cwd`` for the subprocess.
            capture_output: Capture stdout/stderr into
                ``CompletedProcess.stdout`` / ``.stderr``.
            check: Raise :exc:`subprocess.CalledProcessError` on non-zero exit.
            **kwargs: Additional engine-specific keyword arguments.

        Returns:
            :class:`subprocess.CompletedProcess` with execution results.

        Raises:
            RuntimeError: If no input script is found in *run_dir*.
            subprocess.CalledProcessError: If *check* is ``True`` and the
                process exits with a non-zero code.
            subprocess.TimeoutExpired: If the configured ``timeout`` is
                exceeded.
        """

    def _default_executable(self) -> str:
        """The executable when the configuration names none.

        Raises:
            ValueError: This engine has no fallback; configure
                ``engine.<tool>.executable``.
        """
        raise ValueError(f"engine.{self.tool}.executable is not configured")

    # ------------------------------------------------------------------
    # Public methods
    # ------------------------------------------------------------------

    def check_executable(self) -> None:
        """Verify the executable is available in the configured environment.

        Uses system ``PATH`` when no isolation is set; otherwise resolves
        inside the configured conda / venv via :class:`~molpy.wrapper.EnvironmentSpec`.

        Raises:
            FileNotFoundError: If the executable cannot be found.
        """
        if self.environment.resolve_executable(self.executable) is None:
            raise FileNotFoundError(
                f"Executable '{self.executable}' not found in the configured "
                "environment.  Install the engine and put it on PATH, or set "
                f"engine.{self.tool}.executable / env / env_manager in molpy's "
                "configuration."
            )

    def run(
        self,
        scripts: "Script | str | Path | Sequence[Script] | None" = None,
        *,
        workdir: str | Path | None = None,
        capture_output: bool = False,
        check: bool = True,
        **kwargs: Any,
    ) -> subprocess.CompletedProcess:
        """Write scripts to disk and execute the engine.

        Accepts scripts as :class:`~molpy.engine.Script` objects, raw
        strings, :class:`~pathlib.Path` objects, or a list thereof.  If
        *workdir* is given it is used for this call only — ``self.work_dir``
        is **not** modified.

        Args:
            scripts: Input script(s) to run.  If ``None``, previously
                registered scripts (from the last call) are re-used.
            workdir: Working directory for this run.  Overrides
                ``self.work_dir`` for the duration of the call only.
            capture_output: Capture stdout/stderr.
            check: Raise on non-zero exit code.
            **kwargs: Forwarded to :meth:`_execute`.

        Returns:
            :class:`subprocess.CompletedProcess` with execution results.

        Raises:
            ValueError: If no scripts are provided and none were registered
                previously.
        """
        # Resolve run directory (does NOT write back to self.work_dir)
        run_dir = Path(workdir) if workdir is not None else self.work_dir
        if run_dir is None:
            run_dir = Path(tempfile.mkdtemp())
        run_dir.mkdir(parents=True, exist_ok=True)

        # Normalise scripts argument
        if scripts is not None:
            if isinstance(scripts, str):
                normalised: list[Script] = [Script.from_text("input", scripts)]
            elif isinstance(scripts, Path):
                normalised = [Script.from_path(scripts)]
            elif isinstance(scripts, Script):
                normalised = [scripts]
            else:
                normalised = list(scripts)

            if not normalised:
                raise ValueError("At least one script is required.")

            self.scripts = normalised
        elif not self.scripts:
            raise ValueError(
                "At least one script is required.  Pass scripts to run() or "
                "call generate_inputs() first."
            )

        # Write scripts to run_dir
        for script in self.scripts:
            if script.path is not None:
                script_path = run_dir / script.path.name
            else:
                ext = self._get_default_extension()
                script_path = run_dir / f"{script.name}{ext}"
            script.save(script_path)

        self.input_script = self._find_input_script()

        return self._execute(
            run_dir,
            capture_output=capture_output,
            check=check,
            **kwargs,
        )

    # ------------------------------------------------------------------
    # Protected helpers
    # ------------------------------------------------------------------

    def _build_full_command(self, engine_args: list[str]) -> list[str]:
        """Build the complete command list for :func:`subprocess.run`.

        The order is::

            [env_wrapper...] [launcher...] executable [engine_args...]

        where *env_wrapper* comes from :meth:`EnvironmentSpec.command_prefix`
        (conda uses ``conda run --no-capture-output``; venv has an empty
        prefix and injects ``PATH`` via :meth:`_merged_environment`) and
        *launcher* is the configured ``launcher``.

        Args:
            engine_args: Engine-specific flags that follow the executable,
                e.g. ``["-in", "input.lmp", "-log", "log.lammps"]``.

        Returns:
            Full command list suitable for :func:`subprocess.run`.
        """
        cmd = self.environment.command_prefix(no_capture_output=True)
        cmd += self.launcher
        cmd += [self.executable] + engine_args
        return cmd

    def _run_process(
        self,
        command: list[str],
        run_dir: Path,
        *,
        capture_output: bool,
        check: bool,
        **fields: Any,
    ) -> subprocess.CompletedProcess:
        """Start *command* in *run_dir* with the engine's environment and timeout.

        The run is logged to :attr:`logger` (``process started`` /
        ``finished`` / ``failed`` / ``timed out``), each record carrying the
        command, the directory and *fields* (e.g. ``step="mdrun"``).
        """
        return run_process(
            self.logger,
            command,
            cwd=run_dir,
            env=self._merged_environment(),
            capture_output=capture_output,
            check=check,
            timeout=self.timeout,
            **fields,
        )

    def _find_input_script(self) -> Script | None:
        """Return the primary input script from :attr:`scripts`.

        Prefers a script tagged ``"input"``; falls back to the first script.

        Returns:
            The primary :class:`~molpy.engine.Script`, or ``None`` if
            :attr:`scripts` is empty.
        """
        for script in self.scripts:
            if "input" in script.tags:
                return script
        return self.scripts[0] if self.scripts else None

    def _merged_environment(
        self, extra: dict[str, str] | None = None
    ) -> dict[str, str] | None:
        """Build the environment dict for :func:`subprocess.run`.

        Delegates to :class:`~molpy.wrapper.EnvironmentSpec` (venv ``PATH`` /
        ``VIRTUAL_ENV`` injection, then :attr:`env_vars`, then *extra*).

        Returns ``None`` when isolation is off and both :attr:`env_vars` and
        *extra* are empty so the subprocess inherits the parent environment
        without an unnecessary full copy.

        Args:
            extra: Additional variables to merge on top of :attr:`env_vars`.

        Returns:
            Merged environment dict, or ``None`` if nothing to override.
        """
        spec = self.environment
        if spec.is_system and not self.env_vars and not extra:
            return None
        overlay = dict(self.env_vars)
        if extra:
            overlay.update(extra)
        return spec.merge_environ(extra=overlay)

    def __repr__(self) -> str:
        parts = [f"executable='{self.executable}'"]
        if self.work_dir is not None:
            parts.append(f"workdir='{self.work_dir}'")
        if self.launcher:
            parts.append(f"launcher={self.launcher!r}")
        if not self.environment.is_system:
            parts.append(f"environment={self.environment!r}")
        return f"<{self.__class__.__name__}({', '.join(parts)})>"
