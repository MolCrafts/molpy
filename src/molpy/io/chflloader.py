# author: Roy Kid
# contact: lijichen365@126.com
# date: 2023-09-29
# version: 0.0.1

from pathlib import Path
from typing import Iterable
import chemfiles as chfl

from ..core.frame import Frame
from molpy import alias
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
        frame[alias.timestep] = chfl_frame.step
        box_matrix = chfl_frame.cell.matrix.copy()
        frame.box.set_matrix(box_matrix)
        frame.atoms[alias.xyz] = chfl_frame.positions.copy()
        frame[alias.natoms] = len(chfl_frame.atoms)

        # get atom properties
        INTRINSIC_PROPS = [
            alias.name,
            alias.Z,
            alias.charge,
            alias.mass,
            alias.atype,
        ]

        first_atom = chfl_frame.atoms[0]
        extra_properties = first_atom.list_properties()
        EXTRA_PROPS = [getattr(alias, prop) for prop in extra_properties]
        for prop in INTRINSIC_PROPS + EXTRA_PROPS:
            if hasattr(first_atom, prop):
                frame.atoms[prop] = [getattr(atom, prop) for atom in chfl_frame.atoms]

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

        #   residues = chemfile_frame.topology.residues
        #     if residues:
        #         nresidues = len(residues)
        #         molpy_frame._nresidues = nresidues
        #         names = []
        #         ids = []
        #         index = np.empty((nresidues), dtype=object)
        #         props = []
        #         for i, residue in enumerate(residues):
        #             ids.append(residue.id)
        #             names.append(residue.name)
        #             props.append({k:residue[k] for k in residue.list_properties()})
        #             index[i] = np.array(residue.atoms, copy=True)

        #         molpy_frame.residues['id'] = np.array(ids)
        #         molpy_frame.residues['name'] = np.array(names)
        #         molpy_frame.residues['index'] = index
        #         keys = props[0].keys()
        #         for k in keys:
        #             molpy_frame.residues[k] = np.array([p[k] for p in props])

        return frame


class TrajLoader(ChflLoader):
    def __init__(self, fpath: str | Path, format: str = "", mode: str = "r"):
        self._fpath = fpath
        self._format = format
        self._mode = mode
        self._trajectory = chfl.Trajectory(self._fpath, self._mode, self._format)
        self._join = {}

    def __iter__(self):
        keys = list(self._join.keys())
        values = list(self._join.values())
        for chflframe, v in zip(self._trajectory, *values):
            print(chflframe.positions)
            loader = FrameLoader(chflframe)
            frame = loader.load()
            frame._props.update(dict(zip(keys, np.atleast_1d(v))))
            yield frame

    def join(self, per_frame_data: dict[str, Iterable]):
        self._join.update(per_frame_data)

    def close(self):
        self._trajectory.close()


class MemoryLoader(ChflLoader):
    def __init__(self, data="", format="", mode="r"):
        self._data = data
        self._format = format
        self._mode = mode
        self._fileHandler = chfl.MemoryTrajectory(self._data, self._mode, self._format)
