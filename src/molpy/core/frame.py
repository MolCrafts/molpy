# author: Roy Kid
# contact: lijichen365@126.com
# date: 2024-03-23
# version: 0.0.1

import numpy as np

class Item(dict):
    ...

class ItemList(list):
    
    def __getitem__(self, key):
        if isinstance(key, str):
            return np.array([item[key] for item in self])
        return super().__getitem__(key)
    
class Atom(Item):
    ...

class Bond(Item):
    ...

class Frame(dict):

    def __init__(self):

        self._frame_props = {}
        
        self._atoms = ItemList()
        self._bonds = ItemList()
        self._angles = ItemList()
        self._dihedrals = ItemList()

        self._topology = None

    @property
    def topology(self):
        return self._topology
    
    @property
    def n_atoms(self):
        return len(self._atoms)
    
    @property
    def atoms(self):
        return self._atoms
    
    @topology.setter
    def topology(self, topology):
        self._topology = topology

    def add_atom(self, **props):
        self._atoms.append(Atom(**props))

    def add_bond(self, **props):
        self._bonds.append(Bond(**props))