# author: Roy Kid
# contact: lijichen365@126.com
# date: 2024-03-23
# version: 0.0.1

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
    
class Atom(Item):
    
    def __repr__(self):

        return f"<Atom: {super().__repr__()}>"

class Bond(Item):
    ...

class Struct(dict):

    def __init__(self, n_atoms:int=0, *args, **kwargs):

        super().__init__(*args, **kwargs)

        self._frame_props = {}
        
        self._atoms = ItemList()
        self._bonds = ItemList()
        self._angles = ItemList()
        self._dihedrals = ItemList()

        self._topology = Topology(n_atoms, )

    @property
    def topology(self):
        return self._topology

    @property
    def box(self):
        return self._box
    
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

    def add_bond(self, **props):
        self._bonds.append(Bond(**props))

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

    def calc_neighborlist(self, cutoff:float):
        
        self._nblist = NeighborList(cutoff,)
        mapping, mapping_batch, shifts_idx = self._nblist(self)
        
    def calc_connectivity(self):
        pass

    def clone(self):
        return deepcopy(self)