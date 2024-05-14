# author: Roy Kid
# contact: lijichen365@126.com
# date: 2024-03-23
# version: 0.0.1

from abc import ABC, abstractmethod
from typing import Sequence, Collection, TypeVar

import numpy as np
import molpy as mp
from .topology import Topology
from .box import Box
from .neighborlist import NeighborList
from copy import deepcopy

class Item(dict):

    def __getattribute__(self, alias):
        key = mp.Alias.get(alias)
        if key:
            return self[key.key]
        if alias in self:
            return self[alias]
        return super().__getattribute__(alias)

class ItemList(list):
    
    def __getitem__(self, key):
        if isinstance(key, str):
            return np.array([item[key] for item in self])
        return super().__getitem__(key)

    def __getattribute__(self, alias):

        key = mp.Alias.get(alias)
        if key:
            return self[key.key]
        return super().__getattribute__(alias)
    
class ItemDict(dict):

    def concat(self, other):
        for key, value in other.items():
            if key in self:
                self[key] = np.concatenate([self[key], value])
            else:
                raise KeyError(f"Key {key} not found in self dict")
            
ItemCollection = TypeVar('ItemCollection', ItemList, ItemDict)
    
class Atom(Item):
    
    def __repr__(self):

        return f"<Atom: {super().__repr__()}>"

class Bond(Item):
    ...

class Struct(ABC):

    def __init__(self, ):
    
        self._props = {}

    def __getitem__(self, key):
        return self._props[key]
            
    def __setitem__(self, key, value):
        self._props[key] = value

    @classmethod
    def join(cls, structs: Collection['Struct'])->'Struct':

        if isinstance(structs[0], DynamicStruct):
            return DynamicStruct.join(structs)
        else:
            return StaticStruct.join(structs)

    
    @property
    @abstractmethod
    def n_atoms(self)->int:
        """
        return the number of atoms in the struct

        Returns:
            int: the number of atoms
        """
        ...

    @property
    @abstractmethod
    def atoms(self)->ItemCollection:
        """
        return the atoms in the struct

        Returns:
            ItemCollection: the atoms
        """
        ...

    @abstractmethod
    def clone(self)->'Struct':
        """
        clone the struct

        Returns:
            Struct: a new struct
        """
        ...

    @abstractmethod
    def __call__(self) -> 'Struct':
        return self.clone()
    
    @abstractmethod
    def union(self, other:'Struct')->'Struct':
        """
        union two structs and return self

        Args:
            other (Struct): the other struct

        Returns:
            Struct: this struct
        """
        ...


class DynamicStruct(Struct):

    def __init__(self, n_atoms:int=0, *args, **kwargs):

        super().__init__(*args, **kwargs)
        
        self._atoms = ItemList()
        self._bonds = ItemList()
        self._angles = ItemList()
        self._dihedrals = ItemList()

        self._topology = Topology(n_atoms, )

        self._struct_mask = []
        self._n_struct = 0

    @classmethod
    def join(cls, structs: Collection['DynamicStruct'])->'DynamicStruct':
        # Type consistency check
        assert all(isinstance(struct, cls) for struct in structs), TypeError("All structs must be of the same type")
        # Create a new struct
        struct = cls()
        for s in structs:
            struct.union(s)
        return struct

    @property
    def topology(self):
        return self._topology
    
    @property
    def n_atoms(self):
        return len(self._atoms)
    
    @property
    def atoms(self)->ItemList[Atom]:
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
    
    @topology.setter
    def topology(self, topology):
        self._topology = topology

    def add_atom(self, atom=None, **props):
        if atom:
            self._atoms.append(atom)
        else:
            self._atoms.append(Atom(**props))
        self._struct_mask.append(self._n_struct)

    def add_bond(self, **props):
        self._bonds.append(Bond(**props))

    def add_struct(self, struct: Struct):
        
        self.union(struct)
        self._n_struct += 1
        self._struct_mask.extend([self._n_struct]*struct.n_atoms)

    def union(self, other:'DynamicStruct')->'DynamicStruct':
        """
        union two structs and return self

        Args:
            other (DynamicStruct): the other struct

        Returns:
            DynamicStruct: this struct
        """
        self._atoms.extend(other.atoms)
        self._bonds.extend(other.bonds)
        self._angles.extend(other.angles)
        self._dihedrals.extend(other.dihedrals)
        self._topology.union(other.topology)
        return self

    def clone(self):
        struct = DynamicStruct()
        struct.union(self)
        return struct

class StaticStruct(Struct):
    
    def __init__(self):

        self._props = {}
        self._atoms = ItemDict()
        self._bonds = ItemDict()
        self._angles = ItemDict()
        self._dihedrals = ItemDict()

        self._n_atoms = 0
        self._n_bonds = 0
        self._n_angles = 0
        self._n_dihedrals = 0

        self._topology = Topology()

    @classmethod
    def join(cls, structs: Collection['StaticStruct'])->'StaticStruct':
        # Type consistency check
        assert all(isinstance(struct, cls) for struct in structs), TypeError("All structs must be of the same type")
        # Create a new struct
        struct = cls()
        for s in structs:
            struct.union(s)
        return struct
    
    def union(self, other:'StaticStruct')->'StaticStruct':
        """
        union two structs and return self

        Args:
            other (StaticStruct): the other struct

        Returns:
            StaticStruct: this struct
        """
        self._atoms.concat(other.atoms)
        self._bonds.concat(other.bonds)
        self._angles.concat(other.angles)
        self._dihedrals.concat(other.dihedrals)
        self._topology.union(other.topology)
        return self

    def n_atoms(self):
        return self._n_atoms
    
    def n_bonds(self):
        return self._n_bonds
    
    def n_angles(self):
        return self._n_angles
    
    def n_dihedrals(self):
        return self._n_dihedrals
    
    @property
    def atoms(self)->ItemDict[Atom]:
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

    def clone(self):
        struct = StaticStruct()
        struct.union(self)
        return struct

    def __call__(self) -> 'StaticStruct':
        return self.clone()


class Frame(Struct):

    def __init__(self, *args, **kwargs):

        super().__init__(*args, **kwargs)

        self._box = None
        self._forcefield = None
    
    def add_struct(self, struct):
        self._atoms.extend(struct.atoms)
        self._bonds.extend(struct.bonds)
        self._angles.extend(struct.angles)
        self._dihedrals.extend(struct.dihedrals)
        
    def set_box(self, lx:int, ly:int, lz:int, xy:int=0, xz:int=0, yz:int=0, origin=np.zeros(3), pbc=np.array([True, True, True])):
        self._box = Box(lx, ly, lz, xy, xz, yz, origin, pbc)
        
    def calc_connectivity(self):
        pass

    def clone(self):
        return deepcopy(self)