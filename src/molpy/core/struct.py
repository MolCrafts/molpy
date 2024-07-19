from typing import Collection

import numpy as np
from .topology import Topology
from copy import deepcopy, copy


class ArrayDict(dict[str, np.ndarray]):

    def concat(self, other):
        for key, value in other.items():
            if key in self:
                self[key] = np.concatenate([self[key], value])
            else:
                self[key] = value
            
    @property
    def size(self):
        len_values = [len(value) for value in self.values()]
        assert all([len_value == len_values[0] for len_value in len_values]), ValueError(f"Values have different lengths")
        if len_values:
            return len_values[0]
        else:
            return 0

class Struct:

    def __init__(self, name: str = ""):

        self.name = name

        self._atoms = ArrayDict()
        self._bonds = ArrayDict()
        self._angles = ArrayDict()
        self._dihedrals = ArrayDict()

        self._topology = Topology()

    def __repr__(self) -> str:
        return f"<Struct {self.name}: {self.n_atoms} atoms>"

    @classmethod
    def join(cls, structs: Collection["Struct"]) -> "Struct":
        # Type consistency check
        assert all(isinstance(struct, cls) for struct in structs), TypeError(
            "All structs must be of the same type"
        )
        # Create a new struct
        struct = cls()
        for s in structs:
            struct.union(s)
        return struct

    def union(self, other: "Struct") -> "Struct":
        """
        union two structs and return self

        Args:
            other (Struct): the other struct

        Returns:
            Struct: this struct
        """
        self._atoms.concat(other.atoms)
        self._bonds.concat(other.bonds)
        self._angles.concat(other.angles)
        self._dihedrals.concat(other.dihedrals)
        self._topology.union(other.topology)
        return self

    @property
    def n_atoms(self):
        return self._atoms.size

    @property
    def n_bonds(self):
        return self._bonds.size

    @property
    def n_angles(self):
        return self._angles.size

    @property
    def n_dihedrals(self):
        return self._dihedrals.size

    @property
    def atoms(self) -> ArrayDict:
        return self._atoms

    @property
    def bonds(self):
        return self._bonds

    @property
    def angles(self):
        return self._angles

    @property
    def dihedrals(self):
        return self._dihedrals
    
    @property
    def topology(self):
        return self._topology

    def clone(self, deep:bool = True):
        if deep:
            copy_fn = deepcopy
        else:
            copy_fn = copy
        struct = copy_fn(self)
        return struct

    def __call__(self) -> "Struct":
        return self.clone()
    