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
(:mod:`mollog`), and every failure is a :class:`~molpy.wrapper.ToolError`.
The settings are resolved by the :class:`~molpy.wrapper.Tool` core every
engine and wrapper shares.
"""

from __future__ import annotations

import subprocess
import tempfile
from abc import ABC, abstractmethod
from collections.abc import Sequence
from pathlib import Path
from typing import TYPE_CHECKING, Any, ClassVar

from molpy.engine._script import Script
from molpy.wrapper import Tool

if TYPE_CHECKING:
    from molcfg import Config


class Engine(Tool, ABC):
    """Abstract base class for computational chemistry engines.

    Concrete subclasses name their configuration table in :attr:`tool`
    (``"lammps"`` reads ``[engine.lammps]``, falling back to ``[engine]``) and
    implement :meth:`_execute` and :meth:`_get_default_extension`. The
    :class:`~molpy.wrapper.Tool` core resolves the settings, environment,
    executable, launcher, environment variables, time limit and logger; this
    class normalises scripts and manages the working directory.

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
        scripts: Scripts registered by the last :meth:`run` call (or ``[]``
            before the first call).
        input_script: Primary input script resolved by the last :meth:`run`
            call (or ``None`` before the first call).

    Raises:
        ToolError: If *check_executable* is ``True`` and the executable is
            not found.
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

    group: ClassVar[str] = "engine"

    def __init__(
        self,
        *,
        config: Config | None = None,
        workdir: str | Path | None = None,
        check_executable: bool = True,
    ) -> None:
        super().__init__(workdir, config=config)

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
        Subclasses build the concrete command with :meth:`command`
        (environment wrapper + launcher + executable + engine flags) and
        start it with :meth:`run_command`, naming the program's log file so a
        failure carries its tail.

        Args:
            run_dir: Directory where input files have been written; use as
                ``cwd`` for the subprocess.
            capture_output: Capture stdout/stderr into
                ``CompletedProcess.stdout`` / ``.stderr``.
            check: Raise :class:`~molpy.wrapper.ToolError` on non-zero exit.
            **kwargs: Additional engine-specific keyword arguments.

        Returns:
            :class:`subprocess.CompletedProcess` with execution results.

        Raises:
            RuntimeError: If no input script is found in *run_dir*.
            ToolError: The engine could not start, timed out, or (with
                *check*) exited non-zero.
        """

    # ------------------------------------------------------------------
    # Public methods
    # ------------------------------------------------------------------

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
        *workdir* is given it is used for this call only — ``self.workdir``
        is **not** modified.

        Args:
            scripts: Input script(s) to run.  If ``None``, previously
                registered scripts (from the last call) are re-used.
            workdir: Working directory for this run.  Overrides
                ``self.workdir`` for the duration of the call only.
            capture_output: Capture stdout/stderr.
            check: Raise on non-zero exit code.
            **kwargs: Forwarded to :meth:`_execute`.

        Returns:
            :class:`subprocess.CompletedProcess` with execution results.

        Raises:
            ValueError: If no scripts are provided and none were registered
                previously.
        """
        # Resolve run directory (does NOT write back to self.workdir)
        run_dir = Path(workdir) if workdir is not None else self.workdir
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
