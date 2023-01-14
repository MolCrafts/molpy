# author: Roy Kid
# contact: lijichen365@126.com
# date: 2023-01-10
# version: 0.0.1

from .struct import DynamicSOA
from .graph import Graph
import numpy as np

class Topology:

    def __init__(self):

        self.reset()


    def reset(self):

        self.bond_idx = []
        self.angle_idx = []
        self.dihedral_idx = []
        self.improper_idx = []

        self.bonds = DynamicSOA()
        self.angles = DynamicSOA()
        self.dihedrals = DynamicSOA()
        self.impropers = DynamicSOA()

        self._graph = GraphProxy()

    @property
    def nbonds(self):
        return len(self.bond_idx)

    @property
    def nangles(self):
        return len(self.angle_idx)

    @property
    def ndihedrals(self):
        return len(self.angle_idx)

    @property
    def nimpropers(self):
        return len(self.bond_idx)

    def add_bond(self, i, j, **properties):

        if i > j:
            i, j = j, i

        self.bond_idx.append((i, j))

        if properties:
            for key, value in properties.items():
                if key not in self.bonds:
                    self.bonds.set_item(key)
                self.bonds.append(key, value)

    def add_angle(self, i, j, k, **properties):

        if i > k:
            i, k = k, i

        self.angle_idx.append((i, j, k))

        if properties:
            for key, value in properties.items():
                if key not in self.angles:
                    self.angles.set_item(key)
                self.angles.append(key, value)

    def add_dihedral(self, i, j, k, l, **properties):

        if j > k:
            i, j, k, l = l, k, j, i
            
        self.dihedral_idx.append((i, j, k, l))

        if properties:
            for key, value in properties.items():
                if key not in self.dihedrals:
                    self.dihedrals.set_item(key)
                
                self.angles.append(key, value)

    def add_improper(self, i, j, k, l, **properties):
            
        self.improper_idx.append((i, *sorted([j, k, l])))

        if properties:
            for key, value in properties.items():
                if key not in self.impropers:
                    self.impropers.set_item(key)
                
                self.impropers.append(key, value)