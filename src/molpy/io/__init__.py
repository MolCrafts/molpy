import molpy as mp
import numpy as np
from pathlib import Path
from typing import Iterable
from .data.chfl import ChflIO

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
    else:
        raise NotImplementedError(f"Forcefield format {format} is not supported.")


def load_log(fpath: str | Path, format: str = ""):
    assert format in ["lammps"]
    if format == "lammps":
        from .log.lammps import LAMMPSLog

        return LAMMPSLog(fpath)
    
def load_traj(fpath: str | Path, format: str = "") -> Iterable[mp.Frame]:

    kernel = ChflIO(fpath)
    return kernel.load_traj(format)


# def save_struct(fpath: str | Path, struct: mp.Struct, format: str = ""):

#     kernel = ChflIO(fpath)
#     kernel.save_struct(struct, format)

# def load_struct(fpath: str | Path, format: str = "") -> mp.Struct:

#     kernel = ChflIO(fpath)
#     return kernel.load_struct(format)

def load_frame(fpath: str | Path, format: str = "") -> mp.Frame:

    kernel = ChflIO(fpath)
    return kernel.load_frame(format)

def save_forcefield(fpath: str | Path, forcefield: mp.ForceField, format: str = ""):

    from .forcefield.lammps import LAMMPSForceFieldWriter
    return LAMMPSForceFieldWriter(fpath, forcefield).write()