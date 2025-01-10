import molpy as mp
import numpy as np
from pathlib import Path

from .utils import to_system

read_txt = np.loadtxt


def read_lammps_data(file: Path, system: mp.System | None = None) -> mp.System:
    """Read LAMMPS data file and return a molpy System object."""
    from .data.lammps import LammpsDataReader
    if system is None:
        system = mp.System()
    reader = LammpsDataReader(file)
    return reader.read(system)

def read_lammps_forcefield(script: Path, data: Path, system: mp.System | None = None) -> mp.System:
    """Read LAMMPS force field file and return a molpy System object."""
    from .forcefield.lammps import LAMMPSForceFieldReader
    reader = LAMMPSForceFieldReader(script, data)
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

def read_lammps(data: Path, script: Path | None = None, system: mp.System | None = None) -> mp.System:
    """Read LAMMPS data and force field files and return a molpy System object. If data file is provided, only read model;
    If input file is provided, read force field.
    """
    if system is None:
        system = mp.System()
    if script is not None:  # read defination first
        system = read_lammps_forcefield(script, data, system)
    system = read_lammps_data(data, system)
    return system

def read_pdb(file: Path) -> mp.System:
    """Read a PDB file and return a molpy System object."""
    from .data.pdb import PDBReader
    reader = PDBReader(file)
    return reader.read()

def read_amber(
    prmtop: Path, inpcrd: Path | None = None, system: mp.System | None = None
) -> mp.ForceField:
    """Read AMBER force field prmtop and return a molpy ForceField object."""
    from .forcefield.amber import AmberPrmtopReader
    prmtop = Path(prmtop)
    inpcrd = Path(inpcrd) if inpcrd is not None else None
    reader = AmberPrmtopReader(prmtop)
    if system is None:
        system = mp.System()
    system.frame.name = prmtop.stem
    system = reader.read(system)
    if inpcrd is not None:
        from .data.amber import AmberInpcrdReader

        reader = AmberInpcrdReader(inpcrd)
        system = reader.read(system)
    return system

def read_mol2(file: Path, system: mp.System | None = None) -> mp.System:
    """Read a mol2 file and return a molpy System object."""
    from .data.mol2 import Mol2Reader
    if system is None:
        system = mp.System()
    reader = Mol2Reader(file)
    return reader.read(system)

def read_xml_forcefield(file: Path, system: mp.System | None = None) -> mp.System:
    """Read an XML force field file and return a molpy System object."""
    from .forcefield.xml import XMLForceFieldReader
    if system is None:
        system = mp.System()
    reader = XMLForceFieldReader(file)
    return reader.read(system)

@to_system
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

@to_system
def write_lammps_molecule(data: mp.System, file: Path) -> None:

    from .data.lammps import LammpsMoleculeWriter
    writer = LammpsMoleculeWriter(file)
    writer.write(data)

def write_lammps_forcefield(system: mp.System, input_: Path | None = None) -> None:
    """Write a molpy System object to a LAMMPS force field file."""
    from .forcefield.lammps import LAMMPSForceFieldWriter
    writer = LAMMPSForceFieldWriter(input_)
    writer.write(system)

def write_lammps(system: mp.System, data: Path, input_: Path | None = None) -> None:
    """Write a molpy System object to LAMMPS data and force field files."""
    write_lammps_data(system, data)
    write_lammps_forcefield(system, input_)
