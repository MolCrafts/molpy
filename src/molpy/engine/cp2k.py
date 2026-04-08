"""CP2K quantum chemistry / molecular dynamics engine.

Wraps the `CP2K <https://www.cp2k.org>`_ program.  The engine writes an
input script to the working directory and runs::

    [launcher...] cp2k.psmp -i <input> -o cp2k.out

Standard CP2K output (log) is redirected to *cp2k.out* via the ``-o`` flag;
stdout is therefore empty, which avoids pipe-buffer deadlocks when the caller
captures output.

MPI and scheduler launchers are configured on the :class:`~molpy.engine.base.Engine`
base class::

    engine = CP2KEngine("cp2k.psmp", launcher=["mpirun", "-np", "32"])
    engine = CP2KEngine("cp2k.psmp", launcher=["srun", "--ntasks=32"])

Reference:
    Kühne, T. D. et al. (2020). CP2K: An electronic structure and molecular
    dynamics software package. *J. Chem. Phys.* **152**, 194103.
    https://doi.org/10.1063/5.0007045
"""

import subprocess
from pathlib import Path
from typing import Any

from .base import Engine


class CP2KEngine(Engine):
    """CP2K quantum chemistry / molecular dynamics engine.

    Runs CP2K input scripts.  The typical executable name is ``cp2k.psmp``
    (MPI + OpenMP build) or ``cp2k.popt`` (MPI only).

    A minimal CP2K input must contain at least ``&GLOBAL``, ``&FORCE_EVAL``,
    and ``&MOTION`` (or ``&ENERGY``) sections.

    Example:
        >>> from molpy.core.script import Script
        >>> from molpy.engine import CP2KEngine
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
        >>> engine = CP2KEngine(executable="cp2k.psmp", check_executable=False)
        >>> result = engine.run(script, workdir="./calc", check=False)
        >>> print(result.returncode)
        0

        MPI execution::

            engine = CP2KEngine("cp2k.psmp", launcher=["mpirun", "-np", "32"])
            result = engine.run(script, workdir="./calc")
    """

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
        timeout: float | None = None,
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
            timeout: Timeout in seconds.
            **kwargs: Ignored (reserved for future use).

        Returns:
            :class:`subprocess.CompletedProcess`.

        Raises:
            RuntimeError: If no input script has been registered.
            subprocess.CalledProcessError: If *check* is ``True`` and CP2K
                exits with a non-zero code.
            subprocess.TimeoutExpired: If *timeout* is exceeded.
        """
        if self.input_script is None or self.input_script.path is None:
            raise RuntimeError("No input script found.  Pass a script to run() first.")

        input_file = self.input_script.path.name
        command = self._build_full_command(["-i", input_file, "-o", "cp2k.out"])

        return subprocess.run(
            command,
            cwd=run_dir,
            capture_output=capture_output,
            text=True,
            check=check,
            timeout=timeout,
            env=self._merged_env(),
        )
