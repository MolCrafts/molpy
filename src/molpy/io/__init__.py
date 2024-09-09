import molpy as mp
import numpy as np
from pathlib import Path
from typing import Iterable
from .saver import ChflIO

def load_txt(
    fpath: str | Path,
    delimiter: str = " ",
    skiprows: int = 0,
    usecols: list[int] | None = None,
) -> np.ndarray:
    data = np.loadtxt(fpath, delimiter, skiprows, usecols)
    return data


def load_forcefield(*fpath: tuple[str | Path], format: str = "") -> mp.ForceField:
    if format == "lammps":
        from .forcefield.lammps import LAMMPSForceFieldReader

        return LAMMPSForceFieldReader(fpath, None).read()


def load_log(fpath: str | Path, format: str = ""):
    assert format in ["lammps"]
    if format == "lammps":
        from .log.lammps import LAMMPSLog

        return LAMMPSLog(fpath)

def load_frame(fpath: str | Path, format: str = "") -> mp.Frame:

    if format == "lammps":
        from .data.lammps import LammpsDataReader

        return LammpsDataReader(fpath)
    
def save_frame(frame: mp.Frame, fpath: str | Path, format: str = ""):
    if format == "lammps":
        from .data.lammps import LammpsDataSaver

        return LammpsDataSaver(frame, fpath)