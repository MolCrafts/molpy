# author: Roy Kid
# contact: lijichen365@126.com
# date: 2023-09-29
# version: 0.0.1

from pathlib import Path
import chemfiles as chfl
import numpy as np

import molpy as mp

class ChflIO:
    
    def __init__(self, fpath:str | Path):
        self._fpath = Path(fpath)

    def _get_saver(self, fpath: str | Path, format: str = ""):
        if not fpath.parent.exists():
            fpath.parent.mkdir(parents=True)
        return chfl.Trajectory(str(fpath), 'w', format)
    
    def _get_loader(self, fpath: str | Path, format: str = ""):
        return chfl.Trajectory(str(fpath), 'r', format)

    def save_struct(self, struct: mp.Struct, format: str = ""):

        chfl_traj_saver = self._get_saver(self._fpath, format=format)
        chfl_frame = chfl.Frame()
        xyz = np.atleast_2d(struct.atoms.xyz)
        types = struct.atoms.type
        for i in range(struct.n_atoms):
            chfl_frame.add_atom(chfl.Atom(str(i), str(types[i])), xyz[i])
        
        bonds = struct.topology.bonds
        for bond in bonds:
            chfl_frame.add_bond(*bond)

        chfl_traj_saver.write(chfl_frame)

    def load_struct(self, format: str = "") -> mp.Struct:
        name = self._fpath.stem
        chfl_traj_loader = self._get_loader(self._fpath, format=format)
        chfl_frame = chfl_traj_loader.read()
        n_atoms = len(chfl_frame.positions)
        struct = mp.Struct(name)
        xyz: np.ndarray = chfl_frame.positions.copy()
        names = []
        types = []
        charges = []
        for atom in chfl_frame.atoms:
            names.append(atom.name)
            types.append(atom.type)
            charges.append(atom.charge)
        
        struct.atoms.xyz = xyz
        struct.atoms.name = names
        struct.atoms.type = types
        struct.atoms.charge = charges

        bonds = chfl_frame.topology.bonds
        struct.topology.add_atoms(names)
        struct.topology.add_bonds(bonds)

        return struct