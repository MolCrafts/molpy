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

MPI and job-scheduler launchers are supported via the ``launcher`` parameter::

    engine = LAMMPSEngine("lmp", launcher=["mpirun", "-np", "16"])
    engine = LAMMPSEngine("lmp", launcher=["srun", "--ntasks", "16"])
"""

import subprocess
import tempfile
from abc import ABC, abstractmethod
from collections.abc import Sequence
from pathlib import Path
from typing import Any

from molpy.core.script import Script
from molpy.wrapper.env import EnvSpec


class Engine(ABC):
    """Abstract base class for computational chemistry engines.

    Concrete subclasses implement :meth:`_execute` and
    :meth:`_get_default_extension`.  The base class handles script
    normalization, working-directory management, and command prefixing
    (launcher + environment wrapper).

    Environment isolation uses the shared
    :class:`~molpy.wrapper.env.EnvSpec` contract (same as wrappers): omit
    both ``env`` and ``env_manager`` for the system ``PATH``, or set both
    explicitly (``"conda"``, ``"venv"`` / ``"pip"`` / ``"virtualenv"``).

    Attributes:
        executable: Path or command to the engine binary.
        work_dir: Default working directory; ``None`` means a temporary
            directory is created on each :meth:`run` call.
        launcher: Optional MPI / scheduler prefix inserted before the
            executable, e.g. ``["mpirun", "-np", "16"]`` or
            ``["srun", "--ntasks", "16"]``.
        env_vars: Extra environment variables forwarded to the subprocess.
        env: Conda env name / prefix, or venv prefix, for isolation.
        env_manager: Environment manager (``"conda"``, ``"venv"``, …).
        scripts: Scripts registered by the last :meth:`run` call (or ``[]``
            before the first call).
        input_script: Primary input script resolved by the last :meth:`run`
            call (or ``None`` before the first call).

    Example:
        >>> from molpy.core.script import Script
        >>> from molpy.engine import LAMMPSEngine
        >>>
        >>> script = Script.from_text(
        ...     name="input",
        ...     text="units real\\natom_style full\\n",
        ...     language="other",
        ... )
        >>> engine = LAMMPSEngine(executable="lmp", check_executable=False)
        >>> result = engine.run(script, workdir="./calc", check=False)
        >>> print(result.returncode)
        0
    """

    def __init__(
        self,
        executable: str,
        *,
        workdir: str | Path | None = None,
        launcher: list[str] | None = None,
        env_vars: dict[str, str] | None = None,
        env: str | Path | None = None,
        env_manager: str | None = None,
        check_executable: bool = True,
    ) -> None:
        """Initialise the engine.

        Args:
            executable: Path or command to the engine binary (e.g. ``"lmp"``).
            workdir: Default working directory.  ``None`` creates a temporary
                directory on each :meth:`run` call.
            launcher: MPI or scheduler prefix prepended before the executable,
                e.g. ``["mpirun", "-np", "16"]`` or ``["srun", "--ntasks", "8"]``.
            env_vars: Extra environment variables set for the subprocess.
            env: Conda env name / prefix, or venv prefix.  Must be provided
                together with *env_manager* (see
                :class:`~molpy.wrapper.env.EnvSpec`).
            env_manager: ``"conda"``, ``"venv"``, ``"pip"``, or
                ``"virtualenv"``.  Conda isolation uses
                ``conda run --no-capture-output``; venv injects ``PATH``.
            check_executable: Verify the executable is available at construction
                time (system ``PATH`` or the configured env).  Set to
                ``False`` in tests or when the binary is only available on a
                remote node.

        Raises:
            FileNotFoundError: If *check_executable* is ``True`` and the
                executable is not found.
            ValueError: If exactly one of *env* / *env_manager* is provided,
                or *env_manager* is unsupported.
        """
        spec = EnvSpec.resolve(env, env_manager)
        self.executable = executable
        self.work_dir = Path(workdir) if workdir is not None else None
        self.launcher = launcher
        self.env_vars: dict[str, str] = env_vars or {}
        self.env = spec.env
        self.env_manager = spec.env_manager

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
            A short, stable identifier used for logging and ``__repr__``.
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
        timeout: float | None = None,
        **kwargs: Any,
    ) -> subprocess.CompletedProcess:
        """Run the engine subprocess.

        Called by :meth:`run` after scripts have been written to *run_dir*.
        Subclasses build the concrete command and call :func:`subprocess.run`.
        Use :meth:`_build_full_command` to obtain the correctly prefixed
        command list (launcher + env wrapper + executable + engine flags).

        Args:
            run_dir: Directory where input files have been written; use as
                ``cwd`` for the subprocess.
            capture_output: Capture stdout/stderr into
                ``CompletedProcess.stdout`` / ``.stderr``.
            check: Raise :exc:`subprocess.CalledProcessError` on non-zero exit.
            timeout: Timeout in seconds; raises
                :exc:`subprocess.TimeoutExpired` when exceeded.
            **kwargs: Additional engine-specific keyword arguments.

        Returns:
            :class:`subprocess.CompletedProcess` with execution results.

        Raises:
            RuntimeError: If no input script is found in *run_dir*.
            subprocess.CalledProcessError: If *check* is ``True`` and the
                process exits with a non-zero code.
            subprocess.TimeoutExpired: If *timeout* is exceeded.
        """

    # ------------------------------------------------------------------
    # Public methods
    # ------------------------------------------------------------------

    def process_env(self) -> EnvSpec:
        """Return the validated :class:`~molpy.wrapper.env.EnvSpec` for this engine."""
        return EnvSpec.resolve(self.env, self.env_manager)

    def check_executable(self) -> None:
        """Verify the executable is available in the configured environment.

        Uses system ``PATH`` when no isolation is set; otherwise resolves
        inside the configured conda / venv via :class:`~molpy.wrapper.env.EnvSpec`.

        Raises:
            FileNotFoundError: If the executable cannot be found.
        """
        if self.process_env().resolve_executable(self.executable) is None:
            raise FileNotFoundError(
                f"Executable '{self.executable}' not found in the configured "
                "environment.  Install the engine, put it on PATH, set "
                "env/env_manager, or provide the full path."
            )

    def run(
        self,
        scripts: "Script | str | Path | Sequence[Script] | None" = None,
        *,
        workdir: str | Path | None = None,
        capture_output: bool = False,
        check: bool = True,
        timeout: float | None = None,
        **kwargs: Any,
    ) -> subprocess.CompletedProcess:
        """Write scripts to disk and execute the engine.

        Accepts scripts as :class:`~molpy.core.script.Script` objects, raw
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
            timeout: Timeout in seconds.
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
            timeout=timeout,
            **kwargs,
        )

    # ------------------------------------------------------------------
    # Protected helpers
    # ------------------------------------------------------------------

    def _build_full_command(self, engine_args: list[str]) -> list[str]:
        """Build the complete command list for :func:`subprocess.run`.

        The order is::

            [env_wrapper...] [launcher...] executable [engine_args...]

        where *env_wrapper* comes from :meth:`EnvSpec.command_prefix`
        (conda uses ``conda run --no-capture-output``; venv has an empty
        prefix and injects ``PATH`` via :meth:`_merged_env`).

        Args:
            engine_args: Engine-specific flags that follow the executable,
                e.g. ``["-in", "input.lmp", "-log", "log.lammps"]``.

        Returns:
            Full command list suitable for :func:`subprocess.run`.
        """
        cmd = self.process_env().command_prefix(no_capture_output=True)
        cmd += self.launcher or []
        cmd += [self.executable] + engine_args
        return cmd

    def _find_input_script(self) -> Script | None:
        """Return the primary input script from :attr:`scripts`.

        Prefers a script tagged ``"input"``; falls back to the first script.

        Returns:
            The primary :class:`~molpy.core.script.Script`, or ``None`` if
            :attr:`scripts` is empty.
        """
        for script in self.scripts:
            if "input" in script.tags:
                return script
        return self.scripts[0] if self.scripts else None

    def _merged_env(self, extra: dict[str, str] | None = None) -> dict[str, str] | None:
        """Build the environment dict for :func:`subprocess.run`.

        Delegates to :class:`~molpy.wrapper.env.EnvSpec` (venv ``PATH`` /
        ``VIRTUAL_ENV`` injection, then :attr:`env_vars`, then *extra*).

        Returns ``None`` when isolation is off and both :attr:`env_vars` and
        *extra* are empty so the subprocess inherits the parent environment
        without an unnecessary full copy.

        Args:
            extra: Additional variables to merge on top of :attr:`env_vars`.

        Returns:
            Merged environment dict, or ``None`` if nothing to override.
        """
        spec = self.process_env()
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
        if self.env is not None:
            parts.append(f"env='{self.env}'")
        if self.env_manager is not None:
            parts.append(f"env_manager='{self.env_manager}'")
        return f"<{self.__class__.__name__}({', '.join(parts)})>"
