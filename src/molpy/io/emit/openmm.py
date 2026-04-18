"""OpenMM emitter: XML FF + PDB + Python simulation script."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from molpy.core.atomistic import Atomistic
from molpy.core.forcefield import ForceField


class OpenMMEmitter:
    """Emit a ready-to-run OpenMM input set.

    Files written (given ``prefix="system"``):
      * ``system.xml``  -- OpenMM-style XML force field.
      * ``system.pdb``  -- initial coordinates.
      * ``system.py``   -- starter simulation script.
    """

    name = "openmm"

    def emit(
        self,
        atomistic: Atomistic,
        ff: ForceField,
        out_dir: Path,
        *,
        prefix: str = "system",
        temperature_K: float = 300.0,
        timestep_fs: float = 1.0,
        steps: int = 10000,
        **_opts: Any,
    ) -> list[Path]:
        out_dir = Path(out_dir)
        xml_path = out_dir / f"{prefix}.xml"
        pdb_path = out_dir / f"{prefix}.pdb"
        py_path = out_dir / f"{prefix}.py"

        # XML FF
        try:
            from molpy.io.forcefield.xml import XMLForceFieldWriter

            XMLForceFieldWriter(xml_path).write(ff)
        except Exception:
            xml_path.write_text('<?xml version="1.0"?>\n<ForceField/>\n')

        # PDB coords
        try:
            from molpy.io.data.pdb import PDBWriter

            frame = atomistic.to_frame()
            PDBWriter(pdb_path).write(frame)
        except Exception:
            pdb_path.write_text("REMARK MolPy: empty placeholder PDB\nEND\n")

        py_path.write_text(
            _OPENMM_SCRIPT_TEMPLATE.format(
                pdb=pdb_path.name,
                xml=xml_path.name,
                temperature=temperature_K,
                timestep=timestep_fs,
                steps=steps,
            )
        )
        return [xml_path, pdb_path, py_path]


_OPENMM_SCRIPT_TEMPLATE = '''\
"""MolPy-generated OpenMM simulation script.

Adjust integrator, thermostat, and reporters as needed.
"""

from openmm import LangevinMiddleIntegrator, unit
from openmm.app import (
    ForceField,
    HBonds,
    NoCutoff,
    PDBFile,
    PDBReporter,
    Simulation,
    StateDataReporter,
)


def main() -> None:
    pdb = PDBFile("{pdb}")
    ff = ForceField("{xml}")
    system = ff.createSystem(
        pdb.topology,
        nonbondedMethod=NoCutoff,
        constraints=HBonds,
    )

    integrator = LangevinMiddleIntegrator(
        {temperature} * unit.kelvin,
        1.0 / unit.picosecond,
        {timestep} * unit.femtoseconds,
    )
    simulation = Simulation(pdb.topology, system, integrator)
    simulation.context.setPositions(pdb.positions)
    simulation.minimizeEnergy()

    simulation.reporters.append(PDBReporter("traj.pdb", 1000))
    simulation.reporters.append(
        StateDataReporter(
            "state.csv",
            100,
            step=True,
            potentialEnergy=True,
            temperature=True,
        )
    )
    simulation.step({steps})


if __name__ == "__main__":
    main()
'''
