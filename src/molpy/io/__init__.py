import molpy as mp
import numpy as np
from pathlib import Path

def read_txt(
    fpath: str | Path,
    delimiter: str = " ",
    skiprows: int = 0,
    usecols: list[int] | None = None,
) -> np.ndarray:
    data = np.readtxt(fpath, delimiter, skiprows, usecols)
    return data


def read_forcefield(*fpath: tuple[str | Path], format: str = "") -> mp.ForceField:
    if format == "lammps":
        from .forcefield.lammps import LAMMPSForceFieldReader

        return LAMMPSForceFieldReader(fpath, None).read()


def read_log(fpath: str | Path, format: str = ""):
    assert format in ["lammps"]
    if format == "lammps":
        from .log.lammps import LAMMPSLog

        return LAMMPSLog(fpath)

def read_frame(fpath: str | Path, format: str = "") -> mp.Frame:

    if format == "lammps":
        from .data.lammps import LammpsDataReader

        return LammpsDataReader(fpath)
    
    elif format == "pdb":
        from .data.pdb import PDBReader

        return PDBReader(fpath)
    
def write_frame(frame: mp.Frame, fpath: str | Path, format: str = ""):
    if format == "lammps":
        from .data.lammps import LammpsDataSaver

        return LammpsDataSaver(frame, fpath)
    elif format == "pdb":
        from .data.pdb import PDBWriter

        return PDBWriter(frame, fpath)
    
def builder(fpath: str | Path, format: str = "") -> mp.Segment:
    frame = read_frame(fpath, format).read()
    seg = mp.Segment()
    for prop in frame["atoms"]:
        seg.add_atom(
            mp.Atom(
                prop
            )
        )
    atoms = seg.atoms
    for prop in frame["bonds"]:
        i = prop['i']
        j = prop['j']
        seg.add_bond(
            mp.Bond(
                atoms[i-1],
                atoms[j-1]
            ))
        
    return seg