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

import os
import shutil
import subprocess
import tempfile
from abc import ABC, abstractmethod
from collections.abc import Sequence
from pathlib import Path
from typing import Any

from molpy.core.script import Script


class Engine(ABC):
    """Abstract base class for computational chemistry engines.

    Concrete subclasses implement :meth:`_execute` and
    :meth:`_get_default_extension`.  The base class handles script
    normalization, working-directory management, and command prefixing
    (launcher + environment wrapper).

    Attributes:
        executable: Path or command to the engine binary.
        work_dir: Default working directory; ``None`` means a temporary
            directory is created on each :meth:`run` call.
        launcher: Optional MPI / scheduler prefix inserted before the
            executable, e.g. ``["mpirun", "-np", "16"]`` or
            ``["srun", "--ntasks", "16"]``.
        env_vars: Extra environment variables forwarded to the subprocess.
        env: Conda / virtual-environment name to activate before execution.
        env_manager: Environment manager; currently ``"conda"`` is supported.
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
        env: str | None = None,
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
            env: Conda / virtual-environment name to activate.  Must be
                provided together with *env_manager*.
            env_manager: Environment manager type.  ``"conda"`` is currently
                supported; activation uses ``conda run -n <env>``.
            check_executable: Verify the executable is on PATH at construction
                time.  Set to ``False`` in tests or when the binary is only
                available on a remote node.

        Raises:
            FileNotFoundError: If *check_executable* is ``True`` and the
                executable is not found.
            ValueError: If exactly one of *env* / *env_manager* is provided.
        """
        if (env is None) != (env_manager is None):
            raise ValueError(
                "Both 'env' and 'env_manager' must be set together, or both omitted "
                "(environment configuration is incomplete).  "
                "Got env=%r, env_manager=%r." % (env, env_manager)
            )

        self.executable = executable
        self.work_dir = Path(workdir) if workdir is not None else None
        self.launcher = launcher
        self.env_vars: dict[str, str] = env_vars or {}
        self.env = env
        self.env_manager = env_manager

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

    def check_executable(self) -> None:
        """Verify the executable is available on PATH.

        Raises:
            FileNotFoundError: If the executable cannot be found.
        """
        if not shutil.which(self.executable):
            raise FileNotFoundError(
                f"Executable '{self.executable}' not found in PATH. "
                "Install the engine or provide the full path."
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

        where *env_wrapper* is only present when :attr:`env_manager` is set.
        Currently ``"conda"`` activates the environment via
        ``conda run --no-capture-output -n <env>``.

        Args:
            engine_args: Engine-specific flags that follow the executable,
                e.g. ``["-in", "input.lmp", "-log", "log.lammps"]``.

        Returns:
            Full command list suitable for :func:`subprocess.run`.
        """
        cmd: list[str] = []
        if self.env_manager == "conda":
            cmd = ["conda", "run", "--no-capture-output", "-n", self.env]
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

        Merge order (later entries win): ``os.environ`` → :attr:`env_vars`
        → *extra*.

        Returns ``None`` when both :attr:`env_vars` and *extra* are empty so
        that the subprocess inherits the parent environment without an
        unnecessary full copy.

        Args:
            extra: Additional variables to merge on top of :attr:`env_vars`.

        Returns:
            Merged environment dict, or ``None`` if nothing to override.
        """
        if not self.env_vars and not extra:
            return None
        merged = dict(os.environ)
        merged.update(self.env_vars)
        if extra:
            merged.update(extra)
        return merged

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
