import molpy as mp
import numpy as np
from pathlib import Path
from typing import List
from . import data
from . import forcefield
from . import log
from . import trajectory

from .utils import to_system, ZipReader, to_frame, FrameLike

read_txt = np.loadtxt



def read_lammps_data(file: Path, atom_style: str, frame: mp.Frame | None = None) -> mp.Frame:
    """Read LAMMPS data file and return a molpy System object."""
    from .data.lammps import LammpsDataReader
    reader = LammpsDataReader(file, atom_style)
    return reader.read(frame=frame)

def read_lammps_forcefield(scripts: Path|list[Path], frame: mp.ForceField | None = None) -> mp.ForceField:
    """Read LAMMPS force field file and return a molpy ForceField object."""
    from .forcefield.lammps import LAMMPSForceFieldReader
    reader = LAMMPSForceFieldReader(scripts)
    return reader.read(frame=frame)

def read_lammps_molecule(file: Path, frame: mp.Frame | None = None) -> mp.Frame:
    """Read LAMMPS molecule file and return a molpy System object."""
    from .data.lammps import LammpsMoleculeReader
    reader = LammpsMoleculeReader(file, )
    return reader.read(frame=frame)

def read_lammps(data: Path, scripts: Path | list[Path] | None = None, frame: mp.Frame | None = None) -> mp.Frame:
    """Read LAMMPS data and force field files and return a molpy System object. If data file is provided, only read model;
    If input file is provided, read force field.
    """
    if scripts is not None:  # read defination first
        frame = read_lammps_forcefield(scripts, frame)
    frame = read_lammps_data(data, frame)
    return frame

def read_pdb(file: Path, frame: mp.Frame | None = None) -> mp.Frame:
    """Read a PDB file and return a molpy Frame object."""
    from .data.pdb import PDBReader
    reader = PDBReader(file)
    if frame is None:
        frame = mp.Frame()
    return reader.read(frame)

def read_amber(
    prmtop: Path, inpcrd: Path | None = None, frame: mp.Frame | None = None
) -> mp.Frame:
    """Read AMBER force field prmtop and return a molpy ForceField object."""
    from .forcefield.amber import AmberPrmtopReader
    prmtop = Path(prmtop)
    inpcrd = Path(inpcrd) if inpcrd is not None else None
    reader = AmberPrmtopReader(prmtop)
    if frame is None:
        frame = mp.Frame()
    frame = reader.read(frame)
    if inpcrd is not None:
        from .data.amber import AmberInpcrdReader

        reader = AmberInpcrdReader(inpcrd)
        frame = reader.read(frame)
    return frame

def read_amber_ac(file: Path, frame: mp.Frame | None = None) -> mp.Frame:

    """Read an AC file and return a molpy System object."""
    from .data.ac import AcReader
    reader = AcReader(file)
    if frame is None:
        frame = mp.Frame()
    return reader.read(frame)


def read_mol2(file: Path, frame: mp.Frame | None = None) -> mp.Frame:

    """Read a mol2 file and return a molpy System object."""
    from .data.mol2 import Mol2Reader
    reader = Mol2Reader(file)
    return reader.read(frame)

def read_xml_forcefield(file: Path, frame: mp.Frame | None = None) -> mp.Frame:
    """Read an XML force field file and return a molpy System object."""
    from .forcefield.xml import XMLForceFieldReader
    if frame.forcefield is None:
        frame.forcefield = mp.ForceField()

    builtin = Path(__file__).parent / f'forcefield/xml/{file}.xml'
    if builtin.exists():
        file = builtin

    reader = XMLForceFieldReader(file)
    return reader.read(frame)


def read_gro(file: Path, frame: mp.Frame | None = None) -> mp.Frame:
    """Read a GROMACS gro file and return a molpy System object."""
    from .data.gro import GroReader
    reader = GroReader(file)
    return reader.read(frame)

def read_top(file: Path, frame: mp.Frame | None = None) -> mp.Frame:
    """Read a GROMACS top file and return a molpy System object."""
    from .data.top import TopReader
    reader = TopReader(file)
    return reader.read(frame)


def write_lammps_data(file:Path, frame: mp.Frame) -> None:
    """Write a molpy System object to a LAMMPS data file."""
    from .data.lammps import LammpsDataWriter
    writer = LammpsDataWriter(file)
    writer.write(frame)

def write_pdb(file: Path, frame: mp.Frame) -> None:
    """Write a molpy System object to a PDB file."""
    from .data.pdb import PDBWriter
    writer = PDBWriter(file)
    writer.write(frame)

def write_lammps_molecule(file: Path, frame: mp.Frame) -> None:

    from .data.lammps import LammpsMoleculeWriter
    writer = LammpsMoleculeWriter(file)
    writer.write(frame)

def write_lammps_forcefield(file: Path, forcefield: mp.ForceField) -> None:
    """Write a molpy System object to a LAMMPS force field file."""
    from .forcefield.lammps import LAMMPSForceFieldWriter
    writer = LAMMPSForceFieldWriter(file)
    writer.write(forcefield)

def write_lammps(workdir: Path, frame: mp.Frame) -> None:
    """Write a molpy System object to LAMMPS data and force field files."""
    if not workdir.exists():
        workdir.mkdir(parents=True, exist_ok=True)
    file_path = workdir / workdir.stem
    write_lammps_data(file_path.with_suffix(".data"), frame)
    write_lammps_forcefield(file_path.with_suffix(".ff"), frame.forcefield)

def read_top(file: Path, forcefield: mp.ForceField | None = None) -> mp.ForceField:
    """Read a GROMACS top file and return a molpy ForceField object."""
    from .forcefield.top import GromacsTopReader
    reader = GromacsTopReader(file)
    return reader.read(forcefield)

# Trajectory functions
def read_lammps_trajectory(file: Path) -> 'trajectory.lammps.LammpsTrajectoryReader':
    """Read LAMMPS trajectory file and return a trajectory reader."""
    from .trajectory.lammps import LammpsTrajectoryReader
    return LammpsTrajectoryReader(file)

def write_lammps_trajectory(file: Path, frames: List[mp.Frame], atom_style: str = "full") -> None:
    """Write frames to a LAMMPS trajectory file."""
    from .trajectory.lammps import LammpsTrajectoryWriter
    with LammpsTrajectoryWriter(file, atom_style) as writer:
        for i, frame in enumerate(frames):
            timestep = getattr(frame, 'timestep', i)
            writer.write_frame(frame, timestep)