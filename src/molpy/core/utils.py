# author: Roy Kid
# contact: lijichen365@126.com
# date: 2023-02-05
# version: 0.0.1
from copy import deepcopy
from .typing import TypeVar
from .entity import Molecule, Atom, Residue

T = TypeVar("T", Molecule, Atom, Residue)

def clone(obj:T)->T:
    """Clone an object.

    Parameters
    ----------
    obj : object
        The object to be cloned.

    Returns
    -------
    object
        The cloned object.
    """

    return deepcopy(obj)