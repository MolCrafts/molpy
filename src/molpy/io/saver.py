# author: Roy Kid
# contact: lijichen365@126.com
# date: 2023-09-29
# version: 0.0.1

from pathlib import Path
from typing import Iterable
import chemfiles as chfl

from ..core.frame import Frame
from molpy import Alias
import numpy as np

class TrajectorySaver:

    def __init__(self, filename: str):
        self._filename = filename
        self._file = chfl.Trajectory(filename, mode='w')

    def dump(self, frame: Frame):
        chfl_frame = chfl.Frame()
        chfl_frame.step = frame.step
        chfl_frame.cell = chfl.UnitCell(frame.box.length)
        xyz = frame.atoms.positions
        types = frame.atoms.types
        for i in range(frame.n_atoms):
            chfl_frame.add_atom(chfl.Atom(str(i), str(types[i])), xyz[i])
        # chfl_frame.velocities = frame.velocity
        for bond in frame.bonds:
            chfl_frame.add_bond(*bond)
        self._file.write(chfl_frame)