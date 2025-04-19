import molpy as mp
import numpy as np
from pathlib import Path
from . import data
from . import forcefield
from . import log
from . import trajectory

from .utils import to_system, ZipReader

read_txt = np.loadtxt


def read_lammps_data(file: Path, atom_style: str, frame: mp.Frame | None = None) -> mp.Frame:
    """Read LAMMPS data file and return a molpy System object."""
    from .data.lammps import LammpsDataReader
    reader = LammpsDataReader(file, atom_style, frame=frame)
    return reader.read()

def read_lammps_forcefield(scripts: Path|list[Path], system: mp.ForceField | None = None) -> mp.ForceField:
    """Read LAMMPS force field file and return a molpy ForceField object."""
    from .forcefield.lammps import LAMMPSForceFieldReader
    reader = LAMMPSForceFieldReader(scripts, system=system)
    return reader.read()

def read_lammps_molecule(file: Path, system: mp.System | None = None) -> mp.System:
    """Read LAMMPS molecule file and return a molpy System object."""
    from .data.lammps import LammpsMoleculeReader
    reader = LammpsMoleculeReader(file, system=system)
    return reader.read()

def read_lammps(data: Path, scripts: Path | list[Path] | None = None, system: mp.System | None = None) -> mp.System:
    """Read LAMMPS data and force field files and return a molpy System object. If data file is provided, only read model;
    If input file is provided, read force field.
    """
    if scripts is not None:  # read defination first
        system = read_lammps_forcefield(scripts, system)
    system = read_lammps_data(data, system)
    return system

def read_pdb(file: Path, frame: mp.Frame | None = None) -> mp.Frame:
    """Read a PDB file and return a molpy Frame object."""
    from .data.pdb import PDBReader
    reader = PDBReader(file)
    if frame is None:
        frame = mp.Frame()
    return reader.read(frame)

def read_amber(
    prmtop: Path, inpcrd: Path | None = None, system: mp.System | None = None
) -> mp.ForceField:
    """Read AMBER force field prmtop and return a molpy ForceField object."""
    from .forcefield.amber import AmberPrmtopReader
    prmtop = Path(prmtop)
    inpcrd = Path(inpcrd) if inpcrd is not None else None
    reader = AmberPrmtopReader(prmtop, system)
    system = reader.read()
    if inpcrd is not None:
        from .data.amber import AmberInpcrdReader

        reader = AmberInpcrdReader(inpcrd, system)
        system = reader.read()
    return system

def read_mol2(file: Path, system: mp.System | None = None) -> mp.System:
    """Read a mol2 file and return a molpy System object."""
    from .data.mol2 import Mol2Reader
    reader = Mol2Reader(file)
    return reader.read(system)

def read_xml_forcefield(file: Path, system: mp.System | None = None) -> mp.System:
    """Read an XML force field file and return a molpy System object."""
    from .forcefield.xml import XMLForceFieldReader
    if system.forcefield is None:
        system.forcefield = mp.ForceField()

    builtin = Path(__file__).parent / f'forcefield/xml/{file}.xml'
    if builtin.exists():
        file = builtin

    reader = XMLForceFieldReader(file, system)
    return reader.read()

@to_system
def write_lammps_data(system: mp.System, file: Path) -> None:
    """Write a molpy System object to a LAMMPS data file."""
    from .data.lammps import LammpsDataWriter
    writer = LammpsDataWriter(file)
    writer.write(system)

def write_pdb(file: Path, frame: mp.System, ) -> None:
    """Write a molpy System object to a PDB file."""
    from .data.pdb import PDBWriter
    writer = PDBWriter(file)
    writer.write(frame)

@to_system
def write_lammps_molecule(data: mp.System, file: Path) -> None:

    from .data.lammps import LammpsMoleculeWriter
    writer = LammpsMoleculeWriter(file)
    writer.write(data)

def write_lammps_forcefield(system: mp.System, script: Path | None = None) -> None:
    """Write a molpy System object to a LAMMPS force field file."""
    from .forcefield.lammps import LAMMPSForceFieldWriter
    writer = LAMMPSForceFieldWriter(script)
    writer.write(system)

def write_lammps(system: mp.System, data: Path, script: Path | None = None) -> None:
    """Write a molpy System object to LAMMPS data and force field files."""
    write_lammps_data(system, data)
    write_lammps_forcefield(system, script)
