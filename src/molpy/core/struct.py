from collections.abc import MutableMapping
import numpy as np
from copy import deepcopy
from molpy import op


class ArrayDict(MutableMapping[str, np.ndarray]):

    def __init__(self, **kwargs):
        self._data = {k: np.asarray(v) for k, v in kwargs.items()}

    def __delitem__(self, key):
        del self._data[key]

    def __getitem__(self, key: str | int) -> np.ndarray | list:
        if isinstance(key, int):
            return [v[key] for v in self._data.values()]

        elif isinstance(key, str):
            return self._data[key]

    def __iter__(self):
        return iter(self._data)

    def __len__(self):
        # assume all arrays have the same length
        return len(next(iter(self._data.values())))

    def __setitem__(self, key, value):
        self._data[key] = np.asarray(value)

    def __repr__(self):
        info = {k: f"shape: {v.shape}, dtype: {v.dtype}" for k, v in self._data.items()}
        return f"<ArrayDict {info}>"

    def __iter__(self):
        return iter(dict(zip(self._data.keys(), self[i])) for i in range(len(self)))

    @classmethod
    def union(cls, *array_dict: "ArrayDict") -> "ArrayDict":
        ad = ArrayDict()
        for a in array_dict:
            for key, value in a._data.items():
                if key not in ad._data:
                    ad._data[key] = np.atleast_1d(value.copy())
                else:
                    ad._data[key] = np.concatenate(
                        [ad._data[key], np.atleast_1d(value)]
                    )
        return ad


class Struct:

    def __init__(self):

        self.props = []

    def __getitem__(self, key):
        if not hasattr(self, key):
            self[key] = ArrayDict()
        return getattr(self, key)

    def __setitem__(self, key, value):
        setattr(self, key, value)
        self.props.append(key)

    def copy(self):
        return deepcopy(self)

    @classmethod
    def union(self, *structs: "Struct") -> "Struct":
        struct = Struct()
        for s in structs:
            for key, value in s._data.items():
                if key not in struct._data:
                    struct._data[key] = value.copy()
                else:
                    struct._data[key] = np.concatenate([struct._data[key], value])
        return struct


class Atom(dict):

    def __repr__(self):
        return f"<Atom {self['id']}: {self['name']}>"

    def __eq__(self, other):
        return int(self["id"]) == int(other["id"])


class Bond(dict):

    def __init__(self, itom: Atom, jtom: Atom):
        self.itom = itom
        self.jtom = jtom

    def __repr__(self):
        return f"<Bond {self.itom['id']} {self.jtom['id']}>"


class Angle(dict): ...


class Dihedral(dict): ...


class Segment:

    def __init__(self):
        self.atoms = []
        self.bonds = []
        self.angles = []
        self.dihedrals = []

    def add_atom(self, atom: Atom):
        self.atoms.append(atom)

    def add_bond(self, bond: Bond):
        self.bonds.append(bond)

    def add_angle(self, angle: Angle):
        self.angles.append(angle)

    def add_dihedral(self, dihedral: Dihedral):
        self.dihedrals.append(dihedral)

    def del_atom(self, id: int):
        self.atoms = [atom for atom in self.atoms if atom["id"] != id]
        self.bonds = [
            bond
            for bond in self.bonds
            if bond.itom["id"] != id and bond.jtom["id"] != id
        ]

    def get_atom(self, key, value):
        for atom in self.atoms:
            if atom[key] == value:
                return atom
        raise ValueError(f"Atom with {key}={value} not found")

    def add_struct(self, struct: Struct):
        for atom in struct.atoms:
            self.add_atom(atom)
        for bond in struct.bonds:
            self.add_bond(bond)

    def move(self, r):
        for atom in self.atoms:
            atom["xyz"] = op.translate(atom["xyz"], r)
        return self

    def rotate(self, axis, theta):
        for atom in self.atoms:
            atom["xyz"] = op.rotate(atom["xyz"], axis, theta)
        return self

    def copy(self):
        return deepcopy(self)
