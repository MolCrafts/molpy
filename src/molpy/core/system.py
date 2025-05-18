import numpy as np

from .frame import Frame
from .forcefield import ForceField
from .box import Box
from .struct import Struct


class System:

    def __init__(self):
        self._forcefield = None
        self._struct = []

    @property
    def forcefield(self):
        return self._forcefield

    @forcefield.setter
    def forcefield(self, value):
        self._forcefield = value

    def set_forcefield(self, forcefield: ForceField):
        """Set the forcefield for the system."""
        self.forcefield = forcefield

    def get_forcefield(self):
        """Get the forcefield for the system."""
        if self._forcefield is None:
            raise ValueError("Forcefield not set.")
        return self._forcefield

    def def_box(self, matrix, pbc=np.ones(3, dtype=bool), origin=np.zeros(3)):
        self._box = Box(matrix=matrix, pbc=pbc, origin=origin)

    def add_struct(self, struct: Struct):
        self._struct.append(struct)

    def get_frame(self):

        frame = Frame.from_structs(self._struct)
        frame["box"] = self._box
        frame["forcefield"] = self._forcefield
        return frame

