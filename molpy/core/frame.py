# author: Roy Kid
# contact: lijichen365@126.com
# date: 2022-08-10
# version: 0.0.1

from typing import Optional
from .topology import Topology
from .item import Atom, Bond, Angle, Dihedral

class Frame:

    def __init__(self, topo:Optional[Topology], timestep:Optional[int]=None):
        
        self.timestep = timestep

        self._atoms = []
        if topo is None:
            self._topo = Topology()
        else:
            self._topo = topo

    def add_atom(self, **properties):

        atom = Atom(**properties)
        self._atoms.append(atom)
        return atom

    def add_bond(self, itom:Atom, jtom:Atom, **properties):

        bond = Bond(itom, jtom, **properties)
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
