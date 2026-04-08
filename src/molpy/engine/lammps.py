"""LAMMPS molecular dynamics engine.

Wraps the `LAMMPS <https://www.lammps.org>`_ molecular dynamics code.
The engine writes an input script to the working directory and runs::

    [launcher...] lmp -in <input> -log log.lammps -screen none

The ``-screen none`` flag suppresses duplicate stdout output; all
per-timestep data is written exclusively to *log.lammps*.

MPI and scheduler launchers are configured on the :class:`~molpy.engine.base.Engine`
base class::

    engine = LAMMPSEngine("lmp", launcher=["mpirun", "-np", "16"])
    engine = LAMMPSEngine("lmp", launcher=["srun", "--ntasks=16"])

Reference:
    Thompson, A. P. et al. (2022). LAMMPS — A flexible simulation tool for
    particle-based materials modeling. *Comput. Phys. Commun.* **271**, 108171.
    https://doi.org/10.1016/j.cpc.2021.108171
"""

import subprocess
from pathlib import Path
from typing import Any

from .base import Engine


class LAMMPSEngine(Engine):
    """LAMMPS molecular dynamics engine.

    Runs LAMMPS input scripts.  The engine binary is typically named ``lmp``,
    ``lmp_serial``, or ``lmp_mpi`` depending on the build.

    Example:
        >>> from molpy.core.script import Script
        >>> from molpy.engine import LAMMPSEngine
        >>>
        >>> script = Script.from_text(
        ...     name="input",
        ...     text="units real\\natom_style full\\nrun 0\\n",
        ...     language="other",
        ... )
        >>> engine = LAMMPSEngine(executable="lmp", check_executable=False)
        >>> result = engine.run(script, workdir="./calc", check=False)
        >>> print(result.returncode)
        0

        MPI execution::

            engine = LAMMPSEngine("lmp", launcher=["mpirun", "-np", "16"])
            result = engine.run(script, workdir="./calc")
    """

    @property
    def name(self) -> str:
        """Return ``"LAMMPS"``.

        Returns:
            Engine identifier string.
        """
        return "LAMMPS"

    def _get_default_extension(self) -> str:
        """Return ``".lmp"`` — the conventional LAMMPS input extension.

        Returns:
            ``".lmp"``
        """
        return ".lmp"

    def _execute(
        self,
        run_dir: Path,
        capture_output: bool = False,
        check: bool = True,
        timeout: float | None = None,
        **kwargs: Any,
    ) -> subprocess.CompletedProcess:
        """Run LAMMPS in *run_dir*.

        Builds the command::

            [launcher...] lmp -in <input_file> -log log.lammps -screen none

        ``-screen none`` prevents LAMMPS from writing timestep data to stdout
        (it still goes to *log.lammps*), avoiding pipe-buffer deadlocks when
        the caller captures output.

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
            subprocess.CalledProcessError: If *check* is ``True`` and LAMMPS
                exits with a non-zero code.
            subprocess.TimeoutExpired: If *timeout* is exceeded.
        """
        if self.input_script is None or self.input_script.path is None:
            raise RuntimeError("No input script found.  Pass a script to run() first.")

        input_file = self.input_script.path.name
        command = self._build_full_command(
            ["-in", input_file, "-log", "log.lammps", "-screen", "none"]
        )

        return subprocess.run(
            command,
            cwd=run_dir,
            capture_output=capture_output,
            text=True,
            check=check,
            timeout=timeout,
            env=self._merged_env(),
        )
