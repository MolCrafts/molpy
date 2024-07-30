# author: Roy Kid
# contact: lijichen365@126.com
# date: 2024-03-23
# version: 0.0.1
import numpy as np
from .struct import Struct, StructList, ItemDict
from .topology import Topology
from molpy.core.space import Box

class Frame:

    def __init__(self, n_atoms:int=0):

        self.atoms = ItemDict()
        self.bonds = ItemDict()
        self.angles = ItemDict()
        self.dihedrals = ItemDict()
        self.impropers = ItemDict()

        self.topology = Topology(n_atoms)
        self.box = Box()