import molpy as mp
import numpy as np
from pathlib import Path

from .chflloader import TrajLoader, FrameLoader
from .chflsaver import TrajectorySaver
from .utils import loadtxt

def load_trajectory(fpath: str | Path, format: str = "", mode: str = "r") -> TrajLoader:
    return TrajLoader(str(fpath), format, mode)

def load_frame(fpath: str | Path, format: str = "", mode: str = "r") -> mp.Frame:
    traj_loader = load_trajectory(fpath, format, mode)
    frame = traj_loader.read()
    return frame

def load_txt(fpath: str | Path, delimiter: str = " ", skiprows: int = 0, usecols: list[int] | None = None) -> np.ndarray:
    data = loadtxt(fpath, delimiter, skiprows, usecols)
    return FrameLoader(data)