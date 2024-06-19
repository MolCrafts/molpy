# author: Roy Kid
# contact: lijichen365@126.com
# date: 2024-03-23
# version: 0.0.1
import numpy as np
from .struct import Struct, StructList
from molpy.core.space import Free, OrthogonalBox, RestrictTriclinicBox

class Frame:

    def __init__(self, name:str=""):
        self._name = name
        self._box = Free()
        self._structs = StructList()

    @property
    def name(self):
        return self._name

    @property
    def box(self):
        return self._box

    def set_orthogonal_box(self, lengths: np.ndarray):
        self._box = OrthogonalBox(lengths)

    def set_triclinic_box(self, matrix: np.ndarray):
        self._box = RestrictTriclinicBox(matrix)

    @property
    def n_atoms(self):
        return self._structs.n_atoms
    
    def add_struct(self, struct: Struct):
        self._structs.append(struct)

    def __repr__(self):
        return f"<Frame: {self.name}>"