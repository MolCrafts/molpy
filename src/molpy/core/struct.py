from abc import ABC, abstractmethod
from typing import Collection, TypeVar, Any

import numpy as np
import molpy as mp
from .topology import Topology
from .space import Box
from copy import deepcopy, copy


class ItemDict(dict[str, np.ndarray]):

    def __deepcopy__(self, memo):
        new_dict = ItemDict()
        for key, value in self.items():
            new_dict[key] = deepcopy(value)
        return new_dict

    def __getattr__(self, alias:str) -> np.ndarray:
        return self[mp.Alias.get(alias).key]
    
    def __setattr__(self, alias:str, value:np.ndarray) -> None:
        self[mp.Alias.get(alias).key] = value

    # def __getitem__(self, key: str) -> np.ndarray:
    #     return super().__getitem__(mp.Alias.get(key).key)
    
    # def __setitem__(self, key: str, value: np.ndarray) -> None:
    #     super().__setitem__(mp.Alias.get(key).key, value)

    def concat(self, other):
        for key, value in other.items():
            if key in self:
                self[key] = np.concatenate([self[key], value])
            else:
                raise KeyError(f"Key {key} not found in self dict")
            
    @property
    def size(self):
        len_values = [len(value) for value in self.values()]
        assert all([len_value == len_values[0] for len_value in len_values]), ValueError(f"Values have different lengths")
        if len_values:
            return len_values[0]
        else:
            return 0

class BaseStructure(dict):

    def __init__(
        self,
        name: str = "",
    ):
        super().__init__(
            name=name
        )

    @property
    def name(self) -> str:
        return self["name"]

    @property
    @abstractmethod
    def n_atoms(self) -> int:
        """
        return the number of atoms in the struct

        Returns:
            int: the number of atoms
        """
        ...

    @property
    @abstractmethod
    def atoms(self):
        """
        return the atoms in the struct

        Returns:
        """
        ...

    @abstractmethod
    def clone(self) -> "BaseStructure":
        """
        clone the struct

        Returns:
            Structure: a new struct
        """
        ...

    @abstractmethod
    def __call__(self) -> "BaseStructure":
        return self.clone()

    @abstractmethod
    def union(self, other: "BaseStructure") -> "BaseStructure":
        """
        union two structs and return self

        Args:
            other (Structure): the other struct

        Returns:
            Structure: this struct
        """
        ...


class StructList(BaseStructure):

    def __init__(self, name: str = ""):
        super().__init__(name)
        self._structs = []

    @property
    def n_atoms(self):
        return sum(struct.n_atoms for struct in self._structs)


class Struct(BaseStructure):

    def __init__(self, name: str = ""):

        super().__init__(name)
        self._atoms = ItemDict()
        self._bonds = ItemDict()
        self._angles = ItemDict()
        self._dihedrals = ItemDict()

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
    def atoms(self) -> ItemDict:
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
    