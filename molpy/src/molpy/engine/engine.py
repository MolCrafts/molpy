"""
Engine base classes for molecular simulation engines.

Provides :class:`Engine`, an abstract base for running external computational
chemistry programs (LAMMPS, CP2K, OpenMM, …).  Each concrete engine handles
command construction, file management, and subprocess execution for its
specific program.

The two supported usage modes are:

1. **Generate-only** — write input files to disk without executing anything
   (LAMMPS, GROMACS and OpenMM; ``paths["input"]`` is the runnable input)::

       paths = engine.generate_inputs(frame, ff, "./output")

2. **Execute** — write the scripts *and* run the engine subprocess::

       result = engine.run(script, workdir="./calc")

MPI and job-scheduler launchers, the executable, its conda / venv
environment, environment variables and time limit are the engine's
``engine.<name>`` settings in molpy's configuration (:mod:`molpy.config`),
resolved by the :class:`~molpy.tool.tool.Tool` core every external program
shares::

    # molpy.toml
    [engine.lammps]
    launcher = ["mpirun", "-np", "16"]

Every subprocess an engine starts is logged to ``molpy.engine.<name>``
(:mod:`mollog`), and every failure is a :class:`molpy.tool.ToolError`.
"""

from __future__ import annotations

import subprocess
import tempfile
from abc import ABC, abstractmethod
from collections.abc import Sequence
from pathlib import Path
from typing import TYPE_CHECKING, ClassVar

from molpy.engine.script import Script
from molpy.tool.tool import Tool

if TYPE_CHECKING:
    from molcfg import Config


class Engine(Tool, ABC):
    """Abstract base class for computational chemistry engines.

    Concrete subclasses name their configuration table in :attr:`tool`
    (``"lammps"`` reads ``[engine.lammps]``, falling back to ``[engine]``) and
    implement :meth:`_execute` and :meth:`_get_default_extension`. The
    :class:`~molpy.tool.tool.Tool` core resolves the settings, environment,
    executable, launcher, environment variables, time limit and logger; this
    class writes the scripts into the run directory.

    Args:
        config: molpy's configuration (:func:`molpy.config.load_config`);
            ``None`` loads it (package defaults, user and project files).
        workdir: Default working directory.  ``None`` creates a temporary
            directory on each :meth:`run` call.
        check_executable: Verify the executable is available at construction
            time (system ``PATH`` or the configured env).  Set to ``False``
            when only writing inputs, or when the binary is only available
            on a remote node.

    Raises:
        ToolError: If *check_executable* is ``True`` and the executable is
            not found.
        ValueError: If the configured environment is incomplete or its
            manager unsupported.

    Example:
        >>> from molpy.config import load_config
        >>> from molpy.engine import LammpsEngine, Script
        >>>
        >>> script = Script.from_text("input", "units real\\natom_style full\\n")
        >>> config = load_config({"engine": {"lammps": {"executable": "lmp"}}})
        >>> engine = LammpsEngine(config=config, check_executable=False)
        >>> result = engine.run(script, workdir="./calc")
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
        if check_executable:
            self.check_executable()

    # ------------------------------------------------------------------
    # Abstract interface
    # ------------------------------------------------------------------

    @abstractmethod
    def _get_default_extension(self) -> str:
        """File extension used when writing an unpathed script.

        Returns:
            Extension string including the leading dot (e.g. ``".lmp"``).
        """

    @abstractmethod
    def _execute(
        self,
        run_dir: Path,
        input_file: Path,
        *,
        capture_output: bool,
        check: bool,
    ) -> subprocess.CompletedProcess[str]:
        """Run the engine subprocess on *input_file*.

        Called by :meth:`run` after the scripts have been written to
        *run_dir*. Subclasses build the command with :meth:`command`
        (environment prefix + launcher + executable + engine flags) and
        start it with :meth:`run_command`, naming the program's log file so a
        failure carries its tail.

        Args:
            run_dir: Directory the scripts were written to; the ``cwd``.
            input_file: The primary script, as written in *run_dir*.
            capture_output: Capture stdout/stderr.
            check: Raise :class:`molpy.tool.ToolError` on a non-zero exit.

        Returns:
            :class:`subprocess.CompletedProcess` with execution results.

        Raises:
            ToolError: The engine could not start, timed out, or (with
                *check*) exited non-zero.
        """

    # ------------------------------------------------------------------
    # Public methods
    # ------------------------------------------------------------------

    def run(
        self,
        scripts: Script | str | Path | Sequence[Script | Path],
        *,
        workdir: str | Path | None = None,
        capture_output: bool = False,
        check: bool = True,
    ) -> subprocess.CompletedProcess[str]:
        """Write *scripts* into the run directory and execute the engine.

        Accepts a :class:`~molpy.engine.Script`, a raw string (named
        ``input``), a :class:`~pathlib.Path` to a script file, or a list of
        scripts and paths. The first script is the engine's input; the rest
        are written beside it. A script with a path is written under its file
        name, any other as ``<name><extension>``. Neither the scripts nor the
        engine are changed: every call names what it runs.

        Args:
            scripts: The script(s) to run; the first is the input.
            workdir: Working directory for this run; ``None`` is
                :attr:`workdir`, or a new temporary directory when that is
                ``None`` too.
            capture_output: Capture stdout/stderr.
            check: Raise :class:`molpy.tool.ToolError` on a non-zero exit.

        Returns:
            :class:`subprocess.CompletedProcess` with execution results.

        Raises:
            ValueError: If *scripts* is an empty list.
            ToolError: The engine could not start, timed out, or (with
                *check*) exited non-zero.
        """
        run_dir, input_file = self._write_scripts(scripts, workdir)
        return self._execute(
            run_dir, input_file, capture_output=capture_output, check=check
        )

    def _write_scripts(
        self,
        scripts: Script | str | Path | Sequence[Script | Path],
        workdir: str | Path | None,
    ) -> tuple[Path, Path]:
        """Write *scripts* into the run directory, as :meth:`run` does.

        Returns:
            The run directory and the input (the first script) written in it.

        Raises:
            ValueError: If *scripts* is an empty list.
        """
        normalised = _as_scripts(scripts)
        run_dir = Path(workdir) if workdir is not None else self.workdir
        if run_dir is None:
            run_dir = Path(tempfile.mkdtemp())
        run_dir.mkdir(parents=True, exist_ok=True)

        written = [
            script.save(
                run_dir
                / (
                    script.path.name
                    if script.path is not None
                    else f"{script.name}{self._get_default_extension()}"
                )
            )
            for script in normalised
        ]
        return run_dir, written[0]


def _as_scripts(scripts: Script | str | Path | Sequence[Script | Path]) -> list[Script]:
    """*scripts* as a non-empty list of :class:`Script`."""
    if isinstance(scripts, (Script, str, Path)):
        scripts = [scripts]
    normalised = [
        Script.from_text("input", item)
        if isinstance(item, str)
        else Script.from_path(item)
        if isinstance(item, Path)
        else item
        for item in scripts
    ]
    if not normalised:
        raise ValueError("At least one script is required.")
    return normalised
