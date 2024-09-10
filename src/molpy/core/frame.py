from molpy.core.space import Box
from .struct import Struct, ArrayDict
import numpy as np


class Frame(Struct):

    def __init__(self):
        super().__init__()
        self.box = Box()

    @classmethod
    def union(cls, *frames: "Frame") -> "Frame":
        frame = Frame()
        for f in frames:
            frame.box = max(frame.box, f.box, key=lambda x: x.volume)

        structs = {}
        for key in frames[0].props:
            for f in frames:

                if key not in structs:
                    structs[key] = [getattr(f, key)]
                else:
                    structs[key].append(getattr(f, key))

        for key, values in structs.items():
            frame[key] = ArrayDict.union(*values)
        return frame
    
    @classmethod
    def from_struct(cls, struct: Struct) -> "Frame":

        frame = Frame()
        probe_atom = struct.atoms[0]
        atoms = {key: [] for key in probe_atom.keys()}
        for atom in struct.atoms:
            for key, value in atom.items():
                atoms[key].append(value)

        bonds = {key: [] for key in struct.bonds[0].keys()}
        bond_idx = []
        for bond in struct.bonds:
            i, j = bond.itom['id'], bond.jtom['id']
            bond_idx.append([i, j])
            for key, value in bond.items():
                bonds[key].append(value)

        unique_bonds, unique_idx = np.unique(np.sort(np.array(bond_idx), axis=1), axis=0, return_index=True)
        bonds = {key: np.array(value)[unique_idx] for key, value in bonds.items()}
        bonds['i'] = unique_bonds[:, 0]
        bonds['j'] = unique_bonds[:, 1]

        frame["atoms"] = ArrayDict(**atoms)
        frame["bonds"] = ArrayDict(**bonds)
        return frame

    def __setitem__(self, key, value):
        if key == "box":
            self.box = value
        else:
            super().__setitem__(key, value)
