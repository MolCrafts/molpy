import molpy as mp
import numpy as np
from pathlib import Path

read_txt = np.loadtxt


def read_lammps_data(file: Path, system: mp.System | None = None) -> mp.System:
    """Read LAMMPS data file and return a molpy System object."""
    from .data.lammps import LammpsDataReader
    if system is None:
        system = mp.System()
    reader = LammpsDataReader(file)
    return reader.read(system)

def read_lammps_forcefield(file: Path, system: mp.System | None = None) -> mp.System:
    """Read LAMMPS force field file and return a molpy System object."""
    from .forcefield.lammps import LAMMPSForceFieldReader
    reader = LAMMPSForceFieldReader(file)
    if system is None:
        system = mp.System()
    return reader.read(system)


def read_lammps_molecule(file: Path, system: mp.System | None = None) -> mp.System:
    """Read LAMMPS molecule file and return a molpy System object."""
    from .data.lammps import LammpsMoleculeReader

    reader = LammpsMoleculeReader(file)
    if system is None:
        system = mp.System()
    return reader.read(system)

def read_lammps(data: Path, input_: Path | None = None, system: mp.System | None = None) -> mp.System:
    """Read LAMMPS data and force field files and return a molpy System object."""
    if system is None:
        system = mp.System()
    if input_ is not None:
        system = read_lammps_forcefield(input_, system)
    system = read_lammps_data(data, system)
    return system

def write_lammps_data(system: mp.System, file: Path) -> None:
    """Write a molpy System object to a LAMMPS data file."""
    from .data.lammps import LammpsDataWriter
    writer = LammpsDataWriter(file)
    writer.write(system)

def write_pdb(system: mp.System, file: Path) -> None:
    """Write a molpy System object to a PDB file."""
    from .data.pdb import PDBWriter
    writer = PDBWriter(file)
    writer.write(system)

def write_lammps_molecule(data: mp.System, file: Path) -> None:

    from .data.lammps import LammpsMoleculeWriter
    writer = LammpsMoleculeWriter(file)
    writer.write(data)

def read_amber(
    prmtop: Path, inpcrd: Path | None = None, system: mp.System | None = None
) -> mp.ForceField:
    """Read AMBER force field prmtop and return a molpy ForceField object."""
    from .forcefield.amber import AmberPrmtopReader
    reader = AmberPrmtopReader(prmtop)
    if system is None:
        system = mp.System()
    system = reader.read(system)
    if inpcrd is not None:
        from .data.amber import AmberInpcrdReader

        reader = AmberInpcrdReader(inpcrd)
        system = reader.read(system)
    return system
