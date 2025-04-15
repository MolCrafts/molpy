import numpy as np
from abc import ABC, abstractmethod
from molpy.core import Struct


class BaseBuilder(ABC):

    @abstractmethod
    def create_sites(self) -> np.ndarray: ...

    @abstractmethod
    def fill(self, struct: Struct) -> Struct: ...

def set_struct(struct: Struct, sites: np.ndarray) -> Struct:
    """
    Set the coordinates of the structure to the given sites.
    """
    ref_point = struct.atoms[0].xyz
    dr = sites - ref_point
    for atom in struct.atoms:
        atom.xyz += dr
    return struct