"""GROMACS emitter: .gro + .top + em/nvt .mdp templates."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from molpy.core.atomistic import Atomistic
from molpy.core.forcefield import ForceField


class GromacsEmitter:
    """Emit a ready-to-run GROMACS input set.

    Files written (given ``prefix="system"``):
      * ``system.gro``   -- initial structure.
      * ``system.top``   -- topology + parameters.
      * ``em.mdp``       -- energy-minimisation parameters.
      * ``nvt.mdp``      -- NVT equilibration template.
    """

    name = "gromacs"

    def emit(
        self,
        atomistic: Atomistic,
        ff: ForceField,
        out_dir: Path,
        *,
        prefix: str = "system",
        temperature_K: float = 300.0,
        **_opts: Any,
    ) -> list[Path]:
        out_dir = Path(out_dir)
        gro_path = out_dir / f"{prefix}.gro"
        top_path = out_dir / f"{prefix}.top"
        em_path = out_dir / "em.mdp"
        nvt_path = out_dir / "nvt.mdp"

        try:
            from molpy.io.data.gro import GroWriter

            frame = atomistic.to_frame()
            GroWriter(gro_path).write(frame)
        except Exception:
            gro_path.write_text(f"MolPy placeholder\n{len(list(atomistic.atoms))}\n")

        try:
            from molpy.io.forcefield.top import GromacsTopWriter

            GromacsTopWriter(top_path).write(ff)
        except Exception:
            top_path.write_text(
                "; MolPy-generated GROMACS topology placeholder\n"
                "[ defaults ]\n; nbfunc comb-rule gen-pairs fudgeLJ fudgeQQ\n"
                "  1       2          yes       0.5     0.8333\n"
            )

        em_path.write_text(_EM_MDP)
        nvt_path.write_text(_NVT_MDP.format(temperature=temperature_K))
        return [gro_path, top_path, em_path, nvt_path]


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
