"""CP2K quantum chemistry / molecular dynamics engine.

Wraps the `CP2K <https://www.cp2k.org>`_ program.  The engine writes an
input script to the working directory and runs::

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
from typing import Any

from ._engine import Engine


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
        >>> script = Script.from_text(name="input", text=inp, language="other")
        >>> engine = Cp2kEngine(check_executable=False)
        >>> result = engine.run(script, workdir="./calc", check=False)
        >>> print(result.returncode)
        0
    """

    tool = "cp2k"

    @property
    def name(self) -> str:
        """Return ``"CP2K"``.

        Returns:
            Engine identifier string.
        """
        return "CP2K"

    def _get_default_extension(self) -> str:
        """Return ``".inp"`` — the conventional CP2K input extension.

        Returns:
            ``".inp"``
        """
        return ".inp"

    def _execute(
        self,
        run_dir: Path,
        capture_output: bool = False,
        check: bool = True,
        **kwargs: Any,
    ) -> subprocess.CompletedProcess:
        """Run CP2K in *run_dir*.

        Builds the command::

            [launcher...] cp2k.psmp -i <input_file> -o cp2k.out

        The ``-o cp2k.out`` flag redirects CP2K's log output to a file,
        keeping stdout empty and preventing pipe-buffer deadlocks.

        Args:
            run_dir: Directory containing the input files; used as ``cwd``.
            capture_output: Capture stdout/stderr.
            check: Raise :exc:`subprocess.CalledProcessError` on failure.
            **kwargs: Ignored (reserved for future use).

        Returns:
            :class:`subprocess.CompletedProcess`.

        Raises:
            RuntimeError: If no input script has been registered.
            subprocess.CalledProcessError: If *check* is ``True`` and CP2K
                exits with a non-zero code.
            subprocess.TimeoutExpired: If the configured ``timeout`` is
                exceeded.
        """
        if self.input_script is None or self.input_script.path is None:
            raise RuntimeError("No input script found.  Pass a script to run() first.")

        input_file = self.input_script.path.name
        command = self._build_full_command(["-i", input_file, "-o", "cp2k.out"])

        return self._run_process(
            command, run_dir, capture_output=capture_output, check=check
        )
