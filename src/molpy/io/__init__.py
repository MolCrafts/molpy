import molpy as mp
import numpy as np
from pathlib import Path

read_txt = np.readtxt


def read_lammps_data(file: Path, system: mp.System = mp.System()) -> mp.System:
    """Read LAMMPS data file and return a molpy System object."""
    from .data.lammps import LammpsDataReader

    reader = LammpsDataReader(file)
    return reader.read(system)


def read_lammps_molecule(file: Path, system: mp.System = mp.System()) -> mp.System:
    """Read LAMMPS molecule file and return a molpy System object."""
    from .data.lammps import LammpsMoleculeReader

    reader = LammpsMoleculeReader(file)
    return reader.read(system)


def read_amber(
    file: Path, inpcrd: Path | None = None, system: mp.System = mp.System()
) -> mp.ForceField:
    """Read AMBER force field file and return a molpy ForceField object."""
    from .forcefield.amber import AmberForceFieldReader

    reader = AmberForceFieldReader(file)
    system = reader.read(system)
    if inpcrd is not None:
        from .data.amber import AmberInpcrdReader

        reader = AmberInpcrdReader(inpcrd)
        system = reader.read(system)
    return system
