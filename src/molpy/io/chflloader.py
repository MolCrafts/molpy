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

__all__ = ["DataLoader", "MemoryLoader"]


class ChflLoader:
    pass


class FrameLoader(ChflLoader):

    def __init__(self, chfl_frame):
        
        self._chfl_frame = chfl_frame
        
    def load(self) -> Frame:
        chfl_frame = self._chfl_frame
        frame = Frame()

        # get frame properties
        frame[Alias.timestep] = chfl_frame.step
        box_matrix = chfl_frame.cell.matrix.copy()
        frame.box.set_matrix(box_matrix)
        frame.atoms[Alias.xyz] = chfl_frame.positions.copy()
        if chfl_frame.has_velocities():
            frame.atoms[Alias.velocity] = chfl_frame.velocities.copy()
        frame[Alias.natoms] = len(chfl_frame.atoms)

        # get atom properties
        PROP_ALIAS_MAP = {
            "name": Alias['name'],
            "atomic_number": Alias['Z'],
            "charge": Alias['charge'],
            "mass": Alias['mass'],  
            "type": Alias['atype'],
            "vdw_radius": Alias['vdw_radius'],
        }

        for key, _alias in PROP_ALIAS_MAP.items():
            frame.atoms[_alias.key] = np.array([getattr(atom, key) for atom in chfl_frame.atoms if hasattr(atom, key)], dtype=_alias.type)

        frame[Alias.natoms] = len(chfl_frame.atoms)

        # get connectivity
        bonds = chfl_frame.topology.bonds
        angles = chfl_frame.topology.angles
        dihedrals = chfl_frame.topology.dihedrals
        impropers = chfl_frame.topology.impropers

        for bond in bonds:
            frame._connectivity.add_bond(*bond)

        for angle in angles:
            frame._connectivity.add_angle(*angle)

        for dihedral in dihedrals:
            frame._connectivity.add_dihedral(*dihedral)

        for improper in impropers:
            frame._connectivity.add_improper(*improper)

        residues = chfl_frame.topology.residues
        molid = np.zeros(len(chfl_frame.atoms), dtype=int)
        if residues:
            for residue in residues:
                molid[residue.atoms] = residue.id
            frame.atoms[Alias.molid] = molid

        return frame


class TrajLoader(ChflLoader):
    def __init__(self, fpath: str | Path, format: str = "", mode: str = "r"):
        self._fpath = fpath
        self._format = format
        self._mode = mode
        self._trajectory = chfl.Trajectory(self._fpath, self._mode, self._format)

    @property
    def nsteps(self):
        return self._trajectory.nsteps
    
    @property
    def path(self):
        return self._fpath

    def __iter__(self):

        for chflframe in self._trajectory:
            loader = FrameLoader(chflframe)
            frame = loader.load()
            yield frame

    def close(self):
        self._trajectory.close()

    def read(self, step: int = 0) -> Frame:
        chfl_frame = self._trajectory.read_step(step)
        loader = FrameLoader(chfl_frame)
        return loader.load()


class MemoryLoader(ChflLoader):
    def __init__(self, data="", format="", mode="r"):
        self._data = data
        self._format = format
        self._mode = mode
        self._fileHandler = chfl.MemoryTrajectory(self._data, self._mode, self._format)
