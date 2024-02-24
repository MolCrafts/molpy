# author: Roy Kid
# contact: lijichen365@126.com
# date: 2023-01-08
# version: 0.0.1

from collections import namedtuple
from typing import Any
import numpy as np
from pathlib import Path
from .box import Box
from molpy import Alias


__all__ = ["Frame", "Connectivity"]

class Frame(dict):

    class Atoms(dict):

        @property
        def positions(self):
            return self[Alias.R]
        
        @positions.setter
        def positions(self, value):
            self[Alias.R] = value

        @property
        def velocities(self):
            return self[Alias.velocity]
        
        @velocities.setter  
        def velocities(self, value):
            self[Alias.velocity] = value

        @property
        def energy(self):
            return self[Alias.energy]
        
        @energy.setter
        def energy(self, value):
            self[Alias.energy] = value
        
        @property
        def forces(self):
            return self[Alias.forces]
        
        @forces.setter
        def forces(self, value):
            self[Alias.forces] = value
        
        @property
        def momenta(self):
            return self[Alias.momenta]
        
        @momenta.setter
        def momenta(self, value):
            self[Alias.momenta] = value
        
        @property
        def charge(self):
            return self[Alias.charge]
        
        @charge.setter
        def charge(self, value):
            self[Alias.charge] = value
        
        @property
        def mass(self):
            return self[Alias.mass]
        
        @mass.setter
        def mass(self, value):
            self[Alias.mass] = value

        @property
        def types(self):
            return self[Alias.atype]
        
        @types.setter
        def types(self, value):
            self[Alias.atype] = value

    def __init__(self, **props):
        super().__init__(**props)
        self._box = Box(0,0,0,0,0,0)
        self._connectivity = Connectivity()
        self._atoms = Frame.Atoms()

    @property
    def box(self):
        return self._box
    
    @box.setter
    def box(self, value):
        if isinstance(value, Box):
            self._box = value
        else:
            self._box = Box(value)

    @property
    def atoms(self):
        return self._atoms

    @property
    def bonds(self):
        return self._connectivity._bonds

    @property
    def n_atoms(self):
        return len(self.atoms.positions)

    @property
    def nbonds(self):
        return self._connectivity.nbonds

    @property
    def nangles(self):
        return self._connectivity.nangles

    @property
    def ndihedrals(self):
        return self._connectivity.ndihedrals

    @property
    def nimpropers(self):
        return self._connectivity.nimpropers
    
    @property
    def step(self):
        return self[Alias.step]
    
    def flatten(self):
        data = self.copy()
        data.update(self._atoms)
        data.update({
            Alias.cell: self._box.matrix,
            Alias.pbc: self._box.pbc
        })
        # TODO: add connectivity
        return data


class Connectivity:
    BondIdx = namedtuple("BondIdx", ["i", "j"])
    AngleIdx = namedtuple("AngleIdx", ["i", "j", "k"])
    DihedralIdx = namedtuple("DihedralIdx", ["i", "j", "k", "l"])
    ImproperIdx = namedtuple("ImproperIdx", ["i", "j", "k", "l"])

    def __init__(self):
        self._bonds = []
        self._angles = []
        self._dihedrals = []
        self._impropers = []

    def add_bond(self, i, j):
        if i > j:
            i, j = j, i
        self._bonds.append(Connectivity.BondIdx(i, j))

    def add_angle(self, i, j, k):
        if i > k:
            i, k = k, i
        self._angles.append(Connectivity.AngleIdx(i, j, k))

    def add_dihedral(self, i, j, k, l):
        if j > k:
            i, j, k, l = l, k, j, i
        self._dihedrals.append(Connectivity.DihedralIdx(i, j, k, l))

    def add_improper(self, i, j, k, l):
        self._impropers.append(Connectivity.ImproperIdx(i, *sorted([j, k, l])))

    @property
    def nbonds(self):
        return len(self._bonds)

    @property
    def nangles(self):
        return len(self._angles)

    @property
    def ndihedrals(self):
        return len(self._dihedrals)

    @property
    def nimpropers(self):
        return len(self._impropers)
