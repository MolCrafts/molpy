# author: Roy Kid
# contact: lijichen365@126.com
# date: 2022-08-10
# version: 0.0.1

from typing import Optional
from .topology import Topology
from .item import Atom, Bond, Angle, Dihedral
from .box import Box
import numpy as np

class Frame:

    pass

class FlexibleFrame(Frame):

    def __init__(self, box:Optional[Box], topo:Optional[Topology], timestep:Optional[int]=None):
        
        self.timestep = timestep
        self._box = box
        self._atoms = []
        if topo is None:
            self._topo = Topology()
        else:
            self._topo = topo

    def add_atom(self, atom):

        self._atoms.append(atom)

    def add_bond(self, bond):

        self._topo.add_bond(bond)
        return bond

    def add_bond_by_index(self, i:int, j:int, **properties):

        itom = self._atoms[i]
        jtom = self._atoms[j]
        bond = Bond(itom, jtom, **properties)
        self._topo.add_bond(bond)
        return bond

    @property
    def n_atoms(self)->int:
        return len(self._atoms)

    @property
    def box(self):
        return self._box

    @box.setter
    def box(self, b):
        self._box = b

    def __getitem__(self, key):

        if isinstance(key, str):
            return [atom[key] for atom in self._atoms]
        elif isinstance(key, (int, slice)):
            return self._atoms[key]


class ConcreateFrame:

    def __init__(self, box:Optional[Box], topo:Optional[Topology], timestep:Optional[int]=None):

        self._atoms = {}
        self._n_atoms = 0
        self.timestep = timestep
        self._box = box
        if topo is None:
            self._topo = Topology()
        else:
            self._topo = topo

    def add_property(self, key, value):

        if isinstance(value, list):
            self._atoms[key] = np.array(value)
            self._n_atoms = len(value)
        elif isinstance(value, np.ndarray):
            self._atoms[key] = value
            self._n_atoms = value.shape[0]
        

    def add_bond(self, bond):

        self._topo.add_bond(bond)
        return bond

    def add_bond_by_index(self, i:int, j:int, **properties):

        itom = self._atoms[i]
        jtom = self._atoms[j]
        bond = Bond(itom, jtom, **properties)
        self._topo.add_bond(bond)
        return bond

    @property
    def n_atoms(self)->int:
        return len(self._n_atoms)

    @property
    def box(self):
        return self._box

    @box.setter
    def box(self, b):
        self._box = b

    def __getitem__(self, key):

        if isinstance(key, str):
            return self._atoms[key]
        elif isinstance(key, (int, slice)):
            raise NotImplementedError()