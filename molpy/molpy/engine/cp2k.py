"""CP2K quantum chemistry / molecular dynamics engine.

Wraps the `CP2K <https://www.cp2k.org>`_ program.  It generates no inputs:
you write the CP2K input (a :class:`~molpy.engine.Script`), and the engine
writes it to the working directory and runs::

    [launcher...] cp2k.psmp -i <input> -o cp2k.out

Standard CP2K output (log) is redirected to *cp2k.out* via the ``-o`` flag;
stdout is therefore empty, which avoids pipe-buffer deadlocks when the caller
captures output.

The executable, MPI / scheduler launcher, environment and time limit are the
``[engine.cp2k]`` settings of molpy's configuration (:mod:`molpy.config`)::

    [engine.cp2k]
    executable = "cp2k.popt"
    launcher = ["srun", "--ntasks=32"]

Reference:
    Kühne, T. D. et al. (2020). CP2K: An electronic structure and molecular
    dynamics software package. *J. Chem. Phys.* **152**, 194103.
    https://doi.org/10.1063/5.0007045
"""

import subprocess
from pathlib import Path

from molpy.engine.engine import Engine


class Cp2kEngine(Engine):
    """CP2K quantum chemistry / molecular dynamics engine.

    Runs CP2K input scripts.  The executable is ``engine.cp2k.executable``:
    ``cp2k.psmp`` (MPI + OpenMP build) by default, or e.g. ``cp2k.popt``
    (MPI only).  Settings: ``[engine.cp2k]``; logger: ``molpy.engine.cp2k``.

    A minimal CP2K input must contain at least ``&GLOBAL``, ``&FORCE_EVAL``,
    and ``&MOTION`` (or ``&ENERGY``) sections.

    Example:
        >>> from molpy.engine import Script
        >>> from molpy.engine import Cp2kEngine
        >>>
        >>> inp = (
        ...     "&GLOBAL\\n"
        ...     "  PROJECT water\\n"
        ...     "  RUN_TYPE ENERGY\\n"
        ...     "&END GLOBAL\\n"
        ...     "&FORCE_EVAL\\n"
        ...     "  METHOD Quickstep\\n"
        ...     "&END FORCE_EVAL\\n"
        ... )
        >>> script = Script.from_text("input", inp)
        >>> engine = Cp2kEngine(check_executable=False)
        >>> result = engine.run(script, workdir="./calc")
        >>> print(result.returncode)
        0
    """

    tool = "cp2k"

    def _get_default_extension(self) -> str:
        """Return ``".inp"`` — the conventional CP2K input extension.

        Returns:
            ``".inp"``
        """
        return ".inp"

    def _execute(
        self,
        run_dir: Path,
        input_file: Path,
        *,
        capture_output: bool,
        check: bool,
    ) -> subprocess.CompletedProcess[str]:
        """Run CP2K on *input_file* in *run_dir*.

        Builds the command::

            [launcher...] cp2k.psmp -i <input_file> -o cp2k.out

        The ``-o cp2k.out`` flag redirects CP2K's log output to a file,
        keeping stdout empty and preventing pipe-buffer deadlocks; a failure
        carries the tail of ``cp2k.out``.

        Raises:
            ToolError: CP2K could not start, timed out, or (with *check*)
                exited non-zero.
        """
        command = self.command(["-i", input_file.name, "-o", "cp2k.out"])
        return self.run_command(
            command,
            cwd=run_dir,
            capture_output=capture_output,
            check=check,
            log=run_dir / "cp2k.out",
        )
