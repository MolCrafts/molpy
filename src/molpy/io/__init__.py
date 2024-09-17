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

def read_data(fpath: str | Path, format: str = "") -> mp.System:

    if format == "lammps":
        from .data.lammps import LammpsDataReader

        return LammpsDataReader(fpath).read()
    
    elif format == "pdb":
        from .data.pdb import PDBReader

        return PDBReader(fpath).read()
    
def write_frame(frame: mp.Frame, fpath: str | Path, format: str = ""):
    if format == "lammps":
        from .data.lammps import LammpsDataSaver

        return LammpsDataSaver(frame, fpath)
    elif format == "pdb":
        from .data.pdb import PDBWriter

        return PDBWriter(frame, fpath)
    
def read_struct(fpath: str | Path, format: str = "") -> mp.Struct:
    if format == "lammps":
        from .data.lammps import LammpsDataReader

        frame = LammpsDataReader(fpath).read()

    elif format == "pdb":
        from .data.pdb import PDBReader

        frame = PDBReader(fpath).read()

    struct = mp.Struct()
    for props in frame['atoms']:
        atom_id = props.pop('id')
        struct.add_atom(mp.Atom(atom_id, **props))

    for props in frame['bonds']:
        i, j = props.pop('i'), props.pop('j')
        itom = struct.get_atom(lambda atom: atom.id == i)
        jtom = struct.get_atom(lambda atom: atom.id == j)
        struct.add_bond(mp.Bond(itom, jtom, **props))
    return struct