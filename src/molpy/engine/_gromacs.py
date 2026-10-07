"""GROMACS molecular dynamics engine.

Wraps `GROMACS <https://www.gromacs.org>`_. :meth:`GromacsEngine.generate_inputs`
writes a ready-to-run input set — coordinates (``.gro``), the whole topology
(``.top``: directives, one ``[ moleculetype ]`` per molecule, ``[ system ]``
and ``[ molecules ]``, from ``write_gromacs_top_system``) and the energy-minimisation and
NVT ``.mdp`` templates. :meth:`GromacsEngine.run` takes one ``.mdp`` as its
input script and runs::

    gmx grompp -f <mdp> -c <prefix>.gro -p <prefix>.top -o <stem>.tpr
    [launcher...] gmx mdrun -deffnm <stem>

Only ``mdrun`` runs under the launcher; ``grompp`` is a serial preprocessor.

Reference:
    Abraham, M. J. et al. (2015). GROMACS: High performance molecular
    simulations through multi-level parallelism from laptops to
    supercomputers. *SoftwareX* **1–2**, 19–25.
    https://doi.org/10.1016/j.softx.2015.06.001
"""

from __future__ import annotations

import subprocess
from pathlib import Path
from typing import TYPE_CHECKING, Any

from molrs.io import write_gro, write_gromacs_top_system

from ._engine import Engine

if TYPE_CHECKING:
    from molrs.ff.forcefield import ForceField
    from molrs.core import Frame


class GromacsEngine(Engine):
    """GROMACS molecular dynamics engine.

    The executable is the ``gmx`` driver (``gmx``, ``gmx_mpi``, …). An input
    script is an ``.mdp`` file; :meth:`generate_inputs` writes the structure
    and topology it is run against.

    Example::

        engine = GromacsEngine("gmx", check_executable=False)
        paths = engine.generate_inputs(frame, ff, "./md")
        engine.run(Script.from_path(paths["em"]), workdir="./md")
    """

    def __init__(
        self,
        executable: str = "gmx",
        *,
        prefix: str = "system",
        check_executable: bool = True,
        **kwargs: Any,
    ) -> None:
        """Initialise the GROMACS engine.

        Args:
            executable: The ``gmx`` driver binary.
            prefix: Stem of the ``.gro`` / ``.top`` pair :meth:`run` passes to
                ``grompp`` (what :meth:`generate_inputs` writes by default).
            check_executable: Verify the executable is on ``PATH``.
            **kwargs: Forwarded to :class:`~molpy.engine.Engine`.
        """
        super().__init__(executable, check_executable=check_executable, **kwargs)
        self.prefix = prefix

    @property
    def name(self) -> str:
        """Return ``"GROMACS"``."""
        return "GROMACS"

    def _get_default_extension(self) -> str:
        """Return ``".mdp"`` — a GROMACS run-parameter file."""
        return ".mdp"

    def _execute(
        self,
        run_dir: Path,
        capture_output: bool = False,
        check: bool = True,
        timeout: float | None = None,
        **kwargs: Any,
    ) -> subprocess.CompletedProcess:
        """Run ``grompp`` then ``mdrun`` for the input ``.mdp`` in *run_dir*.

        Raises:
            RuntimeError: If no input script has been registered.
            subprocess.CalledProcessError: If *check* is ``True`` and a step
                exits non-zero.
            subprocess.TimeoutExpired: If *timeout* is exceeded.
        """
        if self.input_script is None or self.input_script.path is None:
            raise RuntimeError("No input script found.  Pass an .mdp to run() first.")
        mdp = self.input_script.path.name
        stem = Path(mdp).stem
        env = self._merged_environment()
        grompp = self.process_environment().command_prefix(no_capture_output=True) + [
            self.executable,
            "grompp",
            "-f",
            mdp,
            "-c",
            f"{self.prefix}.gro",
            "-p",
            f"{self.prefix}.top",
            "-o",
            f"{stem}.tpr",
        ]
        done = subprocess.run(
            grompp,
            cwd=run_dir,
            capture_output=capture_output,
            text=True,
            check=check,
            timeout=timeout,
            env=env,
            encoding="utf-8",
        )
        if done.returncode != 0:
            return done
        return subprocess.run(
            self._build_full_command(["mdrun", "-deffnm", stem]),
            cwd=run_dir,
            capture_output=capture_output,
            text=True,
            check=check,
            timeout=timeout,
            env=env,
            encoding="utf-8",
        )

    def generate_inputs(
        self,
        frame: Frame,
        forcefield: ForceField,
        output_dir: str | Path,
        *,
        temperature: float = 300.0,
    ) -> dict[str, Path]:
        """Write a ready-to-run GROMACS input set.

        Files written (with the engine's ``prefix``, ``"system"`` by default):
        ``system.gro`` (initial structure), ``system.top`` (the whole
        topology: the force field's directives, one ``[ moleculetype ]`` per
        molecule with each row's parameters, ``[ system ]`` and
        ``[ molecules ]`` — what ``grompp -p`` reads), ``em.mdp``
        (steepest-descent minimisation) and ``nvt.mdp`` (V-rescale NVT at
        *temperature* K).

        Args:
            frame: The typed structure (``Atomistic.to_frame()`` for a graph)
                with its angles and dihedrals (``generate_topology``): GROMACS
                excludes every pair within three bonds, and those rows are
                how the pair list knows them. Its atoms' ``mol_id`` splits it
                into molecules, and its box is the ``.gro`` box.
            forcefield: The force field the frame's types name.
            output_dir: Directory for the files (created if absent).
            temperature: Reference and initial-velocity temperature (K).

        Returns:
            ``{"gro", "top", "em", "nvt"}`` mapped to the written paths.
        """
        out = Path(output_dir)
        out.mkdir(parents=True, exist_ok=True)
        paths = {
            "gro": out / f"{self.prefix}.gro",
            "top": out / f"{self.prefix}.top",
            "em": out / "em.mdp",
            "nvt": out / "nvt.mdp",
        }
        write_gro(paths["gro"], frame)
        write_gromacs_top_system(paths["top"], forcefield, frame)
        paths["em"].write_text(_EM_MDP, encoding="utf-8")
        paths["nvt"].write_text(
            _NVT_MDP.format(temperature=temperature), encoding="utf-8"
        )
        return paths


_EM_MDP = """\
; MolPy-generated energy minimisation
integrator      = steep
emtol           = 1000.0
emstep          = 0.01
nsteps          = 50000

nstlist         = 10
cutoff-scheme   = Verlet
rlist           = 1.2
coulombtype     = PME
rcoulomb        = 1.2
rvdw            = 1.2
pbc             = xyz
"""


_NVT_MDP = """\
; MolPy-generated NVT equilibration
integrator      = md
nsteps          = 50000
dt              = 0.002

nstxout         = 5000
nstvout         = 5000
nstenergy       = 5000
nstlog          = 5000

continuation    = no
constraint_algorithm = lincs
constraints     = h-bonds
lincs_iter      = 1
lincs_order     = 4

cutoff-scheme   = Verlet
nstlist         = 10
rlist           = 1.2
coulombtype     = PME
rcoulomb        = 1.2
rvdw            = 1.2

tcoupl          = V-rescale
tc-grps         = System
tau_t           = 0.1
ref_t           = {temperature}

pcoupl          = no
gen_vel         = yes
gen_temp        = {temperature}
gen_seed        = -1

pbc             = xyz
"""
