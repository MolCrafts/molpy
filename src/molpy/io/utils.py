import numpy as np
from pathlib import Path


def loadtxt(fname, *args, **kwargs) -> np.ndarray:
    return np.loadtxt(fname, *args, **kwargs)