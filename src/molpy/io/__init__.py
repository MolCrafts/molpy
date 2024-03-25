import molpy as mp
import numpy as np
from pathlib import Path

from .loader import ForceFieldLoader, TrajLoader, FrameLoader
from .saver import TrajectorySaver
from .utils import loadtxt

def load_trajectory(fpath: str | Path, format: str = "") -> TrajLoader:
    return TrajLoader(str(fpath), format, 'r')

def load_frame(fpath: str | Path, format: str = "") -> mp.Frame:
    traj_loader = load_trajectory(fpath, format, 'r')
    frame = traj_loader.read()
    return frame

def load_txt(fpath: str | Path, delimiter: str = " ", skiprows: int = 0, usecols: list[int] | None = None) -> np.ndarray:
    data = loadtxt(fpath, delimiter, skiprows, usecols)
    return FrameLoader(data)

def load_forcefield(fpath: str | Path, format: str="", other_files: list=[]):
    files = [fpath] + other_files
    ffl = ForceFieldLoader(files, format)
    return ffl.load()