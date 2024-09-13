from collections.abc import MutableMapping
import numpy as np
from copy import deepcopy
from molpy import op
from typing import Callable


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
        for value in zip(*self._data.values()):
            yield dict(zip(self._data.keys(), value))

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


class Entity(dict):

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def __eq__(self, other):
        return self.id == other.id

    def clone(self):
        return deepcopy(self)


class Atom(Entity):

    def __init__(self, id: int, *args, **kwargs):
        self.id = id
        super().__init__(*args, **kwargs)

    def __repr__(self):
        return f"<Atom {self.id}>"

    def __hash__(self):
        return id(self)

    def to_dict(self):
        d = dict(self)
        d["id"] = self.id
        return d


class Bond(Entity):

    def __init__(self, itom: Atom, jtom: Atom, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.itom = itom
        self.jtom = jtom

    def __repr__(self):
        return f"<Bond {self.itom} {self.jtom}>"

    def __eq__(self, other):
        if isinstance(other, Bond):
            return {self.itom, self.jtom} == {other.itom, other.jtom}
        return False

    def __hash__(self):
        return hash((self.itom, self.jtom))

    def to_dict(self):
        d = dict(self)
        d["itom"] = self.itom.id
        d["jtom"] = self.jtom.id
        return d


class Angle(Entity): ...


class Dihedral(Entity): ...


class MolpyModel:
    pass


class Entities(list):

    def add(self, entity: Entity):
        self.append(entity)

    def keys(self):
        return self[0].keys()


class Struct(MolpyModel):

    def __init__(self):
        self.atoms = Entities()
        self.bonds = Entities()
        self.angles = Entities()
        self.dihedrals = Entities()

    def __repr__(self):
        return f"<Struct {len(self.atoms)} atoms>"

    def add_atom(self, atom: Atom):
        self.atoms.add(atom)

    def add_bond(self, bond: Bond):
        self.bonds.add(bond)

    def add_angle(self, angle: Angle):
        self.angles.add(angle)

    def add_dihedral(self, dihedral: Dihedral):
        self.dihedrals.add(dihedral)

    def del_atom(self, atom: Atom):

        self.atoms.remove(atom)

        for bond in self.bonds:
            if atom in {bond.itom, bond.jtom}:
                self.bonds.remove(bond)

    def get_atom_by_id(self, id: str):
        for atom in self.atoms:
            if atom.id == id:
                return atom

    def get_atom(self, condition: Callable[[Atom], bool]) -> Atom:
        for atom in self.atoms:
            if condition(atom):
                return atom

    def union(self, other: "Struct") -> "Struct":
        struct = self.copy()
        struct.union_(other)
        return struct

    def union_(self, other: "Struct") -> "Struct":
        self.atoms.update(other.atoms)
        self.bonds.update(other.bonds)
        self.angles.update(other.angles)
        self.dihedrals.update(other.dihedrals)
        return self

    def move(self, r):
        for atom in self.atoms:
            atom["xyz"] = op.translate(atom["xyz"], r)
        return self

    def rotate(self, axis, theta):
        for atom in self.atoms:
            atom["xyz"] = op.rotate(atom["xyz"], axis, theta)
        return self
