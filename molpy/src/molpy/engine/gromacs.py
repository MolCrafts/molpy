"""GROMACS molecular dynamics engine.

Wraps `GROMACS <https://www.gromacs.org>`_. :meth:`GromacsEngine.generate_inputs`
writes a ready-to-run input set — coordinates (``.gro``), the whole topology
(``.top``: directives, one ``[ moleculetype ]`` per molecule, ``[ system ]``
and ``[ molecules ]``, from ``write_top_system``) and the energy-minimisation and
NVT ``.mdp`` templates. :meth:`GromacsEngine.run` takes one ``.mdp`` as its
input script and runs::

    gmx grompp -f <mdp> -c <structure> -p <prefix>.top -o <stem>.tpr
    [launcher...] gmx mdrun -deffnm <stem>

``<structure>`` is ``<prefix>.gro`` for the first stage; a later stage starts
from the structure the stage before it wrote (``mdrun -deffnm em`` writes
``em.gro``), named by ``run(..., structure="em.gro")``::

    engine.run(Script.from_path(paths["input"]), workdir=d)  # em.mdp
    engine.run(Script.from_path(paths["nvt"]), workdir=d, structure="em.gro")

Only ``mdrun`` runs under the launcher; ``grompp`` is a serial preprocessor.
The executable, launcher, environment and time limit (per step) are the
``[engine.gromacs]`` settings of molpy's configuration (:mod:`molpy.config`);
each step is logged to ``molpy.engine.gromacs`` with ``step="grompp"`` or
``step="mdrun"``.

Reference:
    Abraham, M. J. et al. (2015). GROMACS: High performance molecular
    simulations through multi-level parallelism from laptops to
    supercomputers. *SoftwareX* **1–2**, 19–25.
    https://doi.org/10.1016/j.softx.2015.06.001
"""

from __future__ import annotations

import subprocess
from collections.abc import Sequence
from pathlib import Path
from typing import TYPE_CHECKING

from molrs.io import write_gro
from molrs.io.gromacs import GromacsTopForcefieldWriter

from molpy.engine.engine import Engine
from molrs.core import Element
from molpy.engine.nonbonded import is_periodic, nonbonded_of, refuse_box_free

if TYPE_CHECKING:
    from molcfg import Config
    from molrs.core import Frame
    from molrs.ff.forcefield import ForceField

    from molpy.engine.script import Script

#: The .mdp's cutoff (nm) and electrostatics where the force field states none.
_DEFAULT_CUTOFF_NM = 1.2
_DEFAULT_COULOMBTYPE = "PME"
#: GROMACS's ``coulombtype`` for the force field's kind of Coulomb style.
_COULOMBTYPE = {"long": "PME", "cut": "Cut-off"}


class GromacsEngine(Engine):
    """GROMACS molecular dynamics engine.

    The executable is the ``gmx`` driver, ``engine.gromacs.executable``
    (``gmx`` by default; ``gmx_mpi``, …). An input script is an ``.mdp``
    file; :meth:`generate_inputs` writes the structure and topology it is run
    against.

    Example::

        engine = GromacsEngine(check_executable=False)
        paths = engine.generate_inputs(frame, ff, "./md")
        engine.run(Script.from_path(paths["input"]), workdir="./md")
        # NVT from the minimised structure EM wrote:
        engine.run(Script.from_path(paths["nvt"]), workdir="./md", structure="em.gro")
    """

    tool = "gromacs"

    def __init__(
        self,
        *,
        prefix: str = "system",
        config: Config | None = None,
        workdir: str | Path | None = None,
        check_executable: bool = True,
    ) -> None:
        """Initialise the GROMACS engine.

        Args:
            prefix: Stem of the ``.top`` :meth:`run` passes to ``grompp``,
                and of the ``.gro`` it starts a first stage from (what
                :meth:`generate_inputs` writes).
            config: molpy's configuration; ``None`` loads it (see
                :class:`~molpy.engine.Engine`).
            workdir: Default working directory.
            check_executable: Verify the executable is available.
        """
        super().__init__(
            config=config, workdir=workdir, check_executable=check_executable
        )
        self.prefix = prefix

    def _get_default_extension(self) -> str:
        """Return ``".mdp"`` — a GROMACS run-parameter file."""
        return ".mdp"

    def run(
        self,
        scripts: Script | str | Path | Sequence[Script | Path],
        *,
        workdir: str | Path | None = None,
        structure: str | Path | None = None,
        capture_output: bool = False,
        check: bool = True,
    ) -> subprocess.CompletedProcess[str]:
        """``grompp`` then ``mdrun`` the ``.mdp`` *scripts* starts with.

        As :meth:`Engine.run <molpy.engine.Engine.run>`, plus the structure
        the stage starts from.

        Args:
            scripts: The ``.mdp`` (first) and any files written beside it.
            workdir: Working directory for this run (see
                :meth:`Engine.run <molpy.engine.Engine.run>`).
            structure: The coordinates ``grompp -c`` reads, relative to the
                run directory: ``None`` is ``<prefix>.gro`` (a first stage);
                a later stage names the previous stage's output, e.g.
                ``"em.gro"`` after ``em.mdp``.
            capture_output: Capture stdout/stderr.
            check: Raise :class:`molpy.tool.ToolError` on a non-zero exit.

        Raises:
            ValueError: If *scripts* is an empty list.
            ToolError: A step could not start, timed out, or (with *check*)
                exited non-zero.
        """
        run_dir, input_file = self._write_scripts(scripts, workdir)
        return self._grompp_mdrun(
            run_dir,
            input_file,
            structure=str(structure) if structure is not None else None,
            capture_output=capture_output,
            check=check,
        )

    def _execute(
        self,
        run_dir: Path,
        input_file: Path,
        *,
        capture_output: bool,
        check: bool,
    ) -> subprocess.CompletedProcess[str]:
        """``grompp`` then ``mdrun`` *input_file* from ``<prefix>.gro``."""
        return self._grompp_mdrun(
            run_dir,
            input_file,
            structure=None,
            capture_output=capture_output,
            check=check,
        )

    def _grompp_mdrun(
        self,
        run_dir: Path,
        input_file: Path,
        *,
        structure: str | None,
        capture_output: bool,
        check: bool,
    ) -> subprocess.CompletedProcess[str]:
        """Run ``grompp`` then ``mdrun`` for the ``.mdp`` *input_file*.

        ``grompp`` reads *structure* (``<prefix>.gro`` when ``None``) and
        runs without the launcher; a failed ``mdrun`` carries the tail of
        ``<stem>.log``.

        Raises:
            ToolError: A step could not start, timed out, or (with *check*)
                exited non-zero.
        """
        stem = input_file.stem
        grompp = self.command(
            [
                "grompp",
                "-f",
                input_file.name,
                "-c",
                structure if structure is not None else f"{self.prefix}.gro",
                "-p",
                f"{self.prefix}.top",
                "-o",
                f"{stem}.tpr",
            ],
            launch=False,
        )
        done = self.run_command(
            grompp,
            cwd=run_dir,
            capture_output=capture_output,
            check=check,
            step="grompp",
        )
        if done.returncode != 0:
            return done
        return self.run_command(
            self.command(["mdrun", "-deffnm", stem]),
            cwd=run_dir,
            capture_output=capture_output,
            check=check,
            log=run_dir / f"{stem}.log",
            step="mdrun",
        )

    def generate_inputs(
        self,
        frame: Frame,
        forcefield: ForceField,
        output_dir: str | Path,
        *,
        temperature: float = 300.0,
        coulombtype: str | None = None,
        rcoulomb: float | None = None,
        rvdw: float | None = None,
    ) -> dict[str, Path]:
        """Write a ready-to-run GROMACS input set.

        Files written (with the engine's ``prefix``, ``"system"`` by default):
        ``system.gro`` (initial structure), ``system.top`` (the whole
        topology: the force field's directives, one ``[ moleculetype ]`` per
        molecule with each row's parameters, ``[ system ]`` and
        ``[ molecules ]`` — what ``grompp -p`` reads), ``em.mdp``
        (steepest-descent minimisation) and ``nvt.mdp`` (V-rescale NVT at
        *temperature* K).

        The ``.mdp`` nonbonded settings are the force field's own:
        ``coulombtype`` is ``PME`` for a long-range Coulomb style
        (``coul/long/pme``) and ``Cut-off`` for a truncated one, ``rcoulomb``
        is its Coulomb style's ``cutoff`` and ``rvdw`` its other pair styles'
        (each defaulting to the other when only one is stated), and
        ``rlist`` the larger. Only what the force field leaves unstated falls
        back to ``PME`` and 1.2 nm; the keyword arguments override.

        Args:
            frame: The typed structure (``Atomistic.to_frame()`` for a graph)
                with its angles and dihedrals (``generate_topology``): GROMACS
                excludes every pair within three bonds, and those rows are
                how the pair list knows them. Its atoms' ``mol_id`` splits it
                into molecules, and its periodic box is the ``.gro`` box.
            forcefield: The force field the frame's types name.
            output_dir: Directory for the files (created if absent).
            temperature: Reference and initial-velocity temperature (K).
            coulombtype: The ``.mdp`` ``coulombtype``; ``None`` is the force
                field's.
            rcoulomb: Coulomb cutoff (nm); ``None`` is the force field's.
            rvdw: Van der Waals cutoff (nm); ``None`` is the force field's.

        Returns:
            ``{"input", "gro", "top", "nvt"}`` mapped to the written paths:
            ``"input"`` is ``em.mdp``, the first stage :meth:`run` takes, and
            ``"nvt"`` the stage after it.

        Raises:
            ValueError: If the frame has no periodic box, before any file is
                written: GROMACS's Verlet cutoff scheme (its only one since
                2020) simulates periodic systems only, so a vacuum system
                needs a box larger than the molecule plus twice the cutoff.
        """
        if not is_periodic(frame):
            raise refuse_box_free("GROMACS", "the Verlet cutoff scheme (pbc = xyz)")
        frame = frame.copy()
        Element.fill_symbols(frame)
        out = Path(output_dir)
        out.mkdir(parents=True, exist_ok=True)
        paths = {
            "input": out / "em.mdp",
            "gro": out / f"{self.prefix}.gro",
            "top": out / f"{self.prefix}.top",
            "nvt": out / "nvt.mdp",
        }
        nonbonded = _nonbonded_mdp(forcefield, coulombtype, rcoulomb, rvdw)
        write_gro(paths["gro"], frame)
        GromacsTopForcefieldWriter().write_system(paths["top"], forcefield, frame)
        paths["input"].write_text(_EM_MDP.format(**nonbonded), encoding="utf-8")
        paths["nvt"].write_text(
            _NVT_MDP.format(temperature=temperature, **nonbonded), encoding="utf-8"
        )
        self.logger.info(
            "inputs written",
            output_dir=str(out),
            files={key: path.name for key, path in paths.items()},
            temperature=temperature,
            **nonbonded,
        )
        return paths


def _nonbonded_mdp(
    forcefield: ForceField,
    coulombtype: str | None,
    rcoulomb: float | None,
    rvdw: float | None,
) -> dict[str, str | float]:
    """The ``.mdp`` nonbonded settings: the overrides, else the force field's."""
    stated = nonbonded_of(forcefield)
    coulomb_nm, vdw_nm = stated.coulomb_cutoff_nm, stated.vdw_cutoff_nm
    if rcoulomb is None:
        rcoulomb = next(
            (r for r in (coulomb_nm, vdw_nm) if r is not None), _DEFAULT_CUTOFF_NM
        )
    if rvdw is None:
        rvdw = next(
            (r for r in (vdw_nm, coulomb_nm) if r is not None), _DEFAULT_CUTOFF_NM
        )
    if coulombtype is None:
        coulombtype = _COULOMBTYPE.get(stated.coulomb or "", _DEFAULT_COULOMBTYPE)
    return {
        "coulombtype": coulombtype,
        "rcoulomb": rcoulomb,
        "rvdw": rvdw,
        "rlist": max(rcoulomb, rvdw),
    }


_EM_MDP = """\
; MolPy-generated energy minimisation
integrator      = steep
emtol           = 1000.0
emstep          = 0.01
nsteps          = 50000

nstlist         = 10
cutoff-scheme   = Verlet
rlist           = {rlist:g}
coulombtype     = {coulombtype}
rcoulomb        = {rcoulomb:g}
rvdw            = {rvdw:g}
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
rlist           = {rlist:g}
coulombtype     = {coulombtype}
rcoulomb        = {rcoulomb:g}
rvdw            = {rvdw:g}

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
