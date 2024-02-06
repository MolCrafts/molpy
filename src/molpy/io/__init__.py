from pathlib import Path
from .chflloader import TrajLoader, FrameLoader
from .utils import loadtxt

def load_trajectory(fpath: str | Path, format: str = "", mode: str = "r") -> TrajLoader:
    return TrajLoader(fpath, format, mode)