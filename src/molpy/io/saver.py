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

class ChflSaver:
    pass

class TrajSaver(ChflSaver):

    def __init__(self, fpath: str | Path, format: str = ""):
        self._fpath = str(fpath)
        self._traj = chfl.Trajectory(self._fpath, mode='w', format=format)

    def dump(self, frame: Frame):
        chfl_frame = FrameSaver(frame).write()
        self._traj.write(chfl_frame)
        

class FrameSaver(ChflSaver):

    def __init__(self, frame: Frame):
        self._frame = frame

    def write(self):
        frame = self._frame
        chfl_frame = chfl.Frame()
        if hasattr(frame, 'step'):
            chfl_frame.step = frame.step
        if hasattr(frame, 'box'):
            chfl_frame.cell = chfl.UnitCell(frame.box.lengths, frame.box.angles)
        xyz = np.atleast_2d(frame.atoms.xyz)
        types = frame.atoms.type
        for i in range(frame.n_atoms):
            chfl_frame.add_atom(chfl.Atom(str(i), str(types[i])), xyz[i])
        
        bonds = frame.topology.bonds
        for bond in bonds:
            chfl_frame.add_bond(*bond)
        return chfl_frame