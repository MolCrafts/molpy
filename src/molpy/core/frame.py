# author: Roy Kid
# contact: lijichen365@126.com
# date: 2024-03-23
# version: 0.0.1
import numpy as np
from .struct import Struct
from molpy.core.space import Free, OrthogonalBox, RestrictTriclinicBox

class Frame(Struct):

    def __init__(self, name:str="", n_atoms:int=0):
        super().__init__(name, n_atoms)
        self._box = Free()

    @property
    def box(self):
        return self._box

    def set_orthogonal_box(self, lengths: np.ndarray):
        self._box = OrthogonalBox(lengths)

    def set_triclinic_box(self, matrix: np.ndarray):
        self._box = RestrictTriclinicBox(matrix)

