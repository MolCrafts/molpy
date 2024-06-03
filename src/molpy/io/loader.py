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
        frame = Frame("", len(chfl_frame.atoms))

        # get frame properties
        frame[Alias.timestep] = chfl_frame.step
        box_matrix = chfl_frame.cell.matrix.copy()
        if not np.all(box_matrix == 0):  # default Free
            i, j = np.nonzero(box_matrix)
            if np.all(i == j):
                frame.set_orthogonal_box(np.diag(box_matrix))
            else:
                frame.set_triclinic_box(box_matrix)
        frame.atoms[Alias.xyz] = chfl_frame.positions.copy()
        if chfl_frame.has_velocities():
            frame.atoms[Alias.velocity] = chfl_frame.velocities.copy()

        # get atom properties
        PROP_ALIAS_MAP = {
            "name": Alias.get('name'),
            "atomic_number": Alias.get('Z'),
            "charge": Alias.get('charge'),
            "mass": Alias.get('mass'),  
            "type": Alias.get('type'),
            "vdw_radius": Alias.get('vdw_radius'),
        }

        for key, _alias in PROP_ALIAS_MAP.items():
            frame.atoms[_alias.key] = np.array([getattr(atom, key) for atom in chfl_frame.atoms if hasattr(atom, key)], dtype=_alias.type)

        # get connectivity
        bonds = chfl_frame.topology.bonds
        angles = chfl_frame.topology.angles
        dihedrals = chfl_frame.topology.dihedrals
        impropers = chfl_frame.topology.impropers
        
        frame.topology.add_bonds(bonds)

        # for angle in angles:
        #     frame.topology.add_angle(*angle)

        # for dihedral in dihedrals:
        #     frame.topology.add_dihedral(*dihedral)

        # for improper in impropers:
        #     frame.topology.add_improper(*improper)

        residues = chfl_frame.topology.residues
        molid = np.zeros(len(chfl_frame.atoms), dtype=int)
        if residues:
            for residue in residues:
                molid[residue.atoms] = residue.id
            frame.atoms[Alias.molid] = molid

        return frame


class TrajLoader(ChflLoader):
    def __init__(self, fpath: str | Path, format: str = ""):
        self._fpath = str(fpath)
        self._format = format
        self._trajectory = chfl.Trajectory(self._fpath, "r", self._format)

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

