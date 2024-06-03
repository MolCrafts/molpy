import molpy as mp
import numpy as np
from pathlib import Path
from typing import Iterable
from .loader import TrajLoader
from .saver import TrajSaver


def load_trajectory(fpath: str | Path, format: str = "") -> TrajLoader:
    return TrajLoader(str(fpath), format)


def load_frame(fpath: str | Path, format: str = "") -> mp.Frame:
    traj_loader = load_trajectory(fpath, format)
    frame = traj_loader.read()
    return frame


def load_struct(fpath: str | Path, format: str = "") -> mp.Struct:
    frame = load_frame(fpath, format)
    return frame


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


def save_traj(fpath: str | Path, frames: Iterable[mp.Frame], format: str = ""):
    saver = TrajSaver(fpath, format)
    for frame in frames:
        saver.dump(frame)


def save_frame(fpath: str | Path, frame: mp.Frame, format: str = ""):
    saver = TrajSaver(fpath, format)
    saver.dump(frame)
