# author: Roy Kid
# contact: lijichen365@126.com
# date: 2023-01-08
# version: 0.0.1

from collections import namedtuple
import numpy as np
from .box import Box
from .keywords import kw


__all__ = ["Frame", "Connectivity"]

class Frame:
    def __init__(self, **props):
        self._box = Box()
        self._props = props
        self._atoms = {}
        self._connectivity = Connectivity()

    def get_box(self):
        return self._box

    @property
    def atoms(self):
        return self._atoms

    @property
    def natoms(self):
        return self._props[kw.natoms]

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
    
    def __setitem__(self, key, value):
        self._props[key] = value

    def __getitem__(self, key):
        return self._props[key]


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
