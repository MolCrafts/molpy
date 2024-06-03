from abc import ABC, abstractmethod
from typing import Collection, TypeVar, Any

import numpy as np
import molpy as mp
from .topology import Topology
from .space import Box
from copy import deepcopy


class ItemDict(dict[str, np.ndarray]):

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
            
    def __len__(self):
        # return len of first value
        return len(next(iter(self.values())))


class Structure(ABC, dict):

    def __init__(
        self,
        name: str = "",
    ):
        self.name = name

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
    def clone(self) -> "Structure":
        """
        clone the struct

        Returns:
            Structure: a new struct
        """
        ...

    @abstractmethod
    def __call__(self) -> "Structure":
        return self.clone()

    @abstractmethod
    def union(self, other: "Structure") -> "Structure":
        """
        union two structs and return self

        Args:
            other (Structure): the other struct

        Returns:
            Structure: this struct
        """
        ...


class Struct(Structure):

    def __init__(self, name: str = "", n_atoms: int = 0):

        super().__init__(name)

        self._props = {}
        self._atoms = ItemDict()
        self._bonds = ItemDict()
        self._angles = ItemDict()
        self._dihedrals = ItemDict()

        self._topology = Topology(n_atoms, )

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
        return len(self._atoms)

    @property
    def n_bonds(self):
        return len(self._bonds)

    @property
    def n_angles(self):
        return len(self._angles)

    @property
    def n_dihedrals(self):
        return len(self._dihedrals)
    
    @property
    def props(self):
        return super().__getattr__("_props")

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

    def clone(self):
        struct = Struct()
        struct.union(self)
        return struct

    def __call__(self) -> "Struct":
        return self.clone()
    