# author: Roy Kid
# contact: lijichen365@126.com
# date: 2023-01-29
# version: 0.0.1

from typing import List, Optional

from molpy.core.struct import StaticSOA
from molpy.core.topology import Topology
import numpy as np

class Residue:

    def __init__(self, id, name, atoms, **prop):

        self.id = id
        self.name = name
        self.atoms = atoms
        self.props = prop

    @classmethod
    def from_dict(cls, dict):
            
        return cls(**dict)

    @property
    def natoms(self):
        return len(self.atoms)


class Molecule:

    def __init__(self, id:Optional[int]=None, name:Optional[str]=None, ):

        self.id = id
        self.name = name
        self.atoms = StaticSOA()
        self.props = {}
        self.topology = Topology()
    
    @property
    def natoms(self):
        return self.atoms.size

