import molpy as mp
import numpy as np
from pathlib import Path

from .loader import TrajLoader

def load_trajectory(fpath: str | Path, format: str = "") -> TrajLoader:
    return TrajLoader(str(fpath), format, 'r')

def load_frame(fpath: str | Path, format: str = "") -> mp.Frame:
    traj_loader = load_trajectory(fpath, format, 'r')
    frame = traj_loader.read()
    return frame

def load_txt(fpath: str | Path, delimiter: str = " ", skiprows: int = 0, usecols: list[int] | None = None) -> np.ndarray:
    data = np.loadtxt(fpath, delimiter, skiprows, usecols)
    return data

def load_forcefield(fpath: str | Path, format: str="", other_files: list=[]):
    files = [fpath] + other_files
    if format == 'lammps':
        from .forcefield.lammps import LAMMPSForceFieldReader
        return LAMMPSForceFieldReader(files, None)

def load_log(fpath: str | Path, format: str=""):
    assert format in ['lammps']
    if format == 'lammps':
        from .log.lammps import LAMMPSLog
        return LAMMPSLog(fpath)