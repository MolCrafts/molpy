# author: Roy Kid
# contact: lijichen365@126.com
# date: 2023-01-10
# version: 0.0.1

from .struct import StructData, GraphProxy
import numpy as np

class Topology:

    def __init__(self, bond_capacity:int=0, angle_capacity:int=0, dihedral_capacity:int=0, improper_capacity:int=0):

        self.reset(bond_capacity, angle_capacity, dihedral_capacity, improper_capacity)
        self.bond_capacity = bond_capacity
        self.angle_capacity = angle_capacity
        self.dihedral_capacity = dihedral_capacity
        self.improper_capacity = improper_capacity

    def reset(self, nbonds:int, nangles:int, ndihedrals:int, nimpropers:int):

        self.bond_idx = np.zeros((nbonds, 2), dtype=int)
        self.angle_idx = np.zeros((nangles, 3), dtype=int)
        self.dihedral_idx = np.zeros((ndihedrals, 4), dtype=int)
        self.improper_idx = np.zeros((nimpropers, 4), dtype=int)

        self.bonds = StructData()
        self.angles = StructData()
        self.dihedrals = StructData()
        self.impropers = StructData()

        self._nbonds = nbonds
        self._nangles = nangles
        self._ndihedrals = ndihedrals
        self._nimpropers = nimpropers

        self._graph = GraphProxy()

    @property
    def nbonds(self):
        return self._nbonds

    @property
    def nangles(self):
        return self._nangles

    @property
    def ndihedrals(self):
        return self._ndihedrals

    @property
    def nimpropers(self):
        return self._nimpropers

    def add_bond(self, i, j, **properties):

        self.bond_idx[self._nbonds] = (i, j)

        if properties:
            for key, value in properties.items():
                if key not in self.bonds:
                    self.bonds.set_empty_like(key, self.bond_capacity, value)
                self.bonds[key][self._nbonds] = value

        self._nbonds += 1

    def add_angle(self, i, j, k, **properties):

        self.angle_idx[self._nangles] = (i, j, k)

        if properties:
            for key, value in properties.items():
                if key not in self.angles:
                    self.angles.set_empty_like(key, self.angle_capacity, value)
                
                self.angles[key][self._nangles] = value

        self._nangles += 1

    def add_dihedral(self, i, j, k, l, **properties):
            
        self.dihedral_idx[self._ndihedrals] = (i, j, k, l)

        if properties:
            for key, value in properties.items():
                if key not in self.dihedrals:
                    self.dihedrals.set_empty_like(key, self.dihedral_capacity, value)
                
                self.dihedrals[key][self._ndihedrals] = value

        self._ndihedrals += 1

    def add_improper(self, i, j, k, l, **properties):
            
        self.improper_idx[self._nimpropers] = (i, j, k, l)

        if properties:
            for key, value in properties.items():
                if key not in self.impropers:
                    self.impropers.set_empty_like(key, self.improper_capacity, value)
                
                self.impropers[key][self._nimpropers] = value

        self._nimpropers += 1