from molpy.system import System
from molpy import toASE
from molpy.io.ase import build_ASE_neigh
from typing import Union

FloatOpt = Union[float, None]

def build_neigh(sysObj: System, cutoff : FloatOpt =None, build_style: str = "ASE") -> None:
    if build_style == "ASE":
        return build_ASE_neigh(sysObj, cutoff)
    else:
        raise NotImplementedError(
            f"The build style [{build_style}] is not implemented!"
        )
    atoms = toASE(sysObj)
