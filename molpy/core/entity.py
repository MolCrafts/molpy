# author: Roy Kid
# contact: lijichen365@126.com
# date: 2023-01-29
# version: 0.0.1

from typing import List, Optional

from molpy.core.struct import DynamicSOA, StaticSOA
from molpy.core.topology import Topology
import numpy as np


class Atom(dict):
    def __repr__(self):
        return f"<Atom: {self.name}>"

    @property
    def name(self):
        return self.get("name", self.get("type", "unknown"))

    @property
    def xyz(self):
        return self.get("xyz", self.get("positions"))

    @property
    def type(self):
        return self.get("type")

    @property
    def charge(self):
        return self.get("charge", 0)

    @property
    def mass(self):
        return self.get("mass", 0)


class Bond(dict):
    def __init__(self, i, j, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.i = i
        self.j = j

    def __iter__(self):
        return (self.i, self.j)


class Residue:
    def __init__(self, name: str, id: Optional[int] = None):

        self.id = id
        self.name = name
        self.atoms = []
        self.topology = Topology()

    @property
    def natoms(self):
        return len(self.atoms)

    def add_atom(self, atom: Atom):
        self.atoms.append(atom)

    def add_bonds(self, bonds):
        self.topology.add_bonds(bonds)

    @property
    def bonds(self) -> List[Bond]:
        index = self.topology.bonds["index"]
        bonds = []
        for idx in index:
            atom1 = self.atoms[idx[0]]
            atom2 = self.atoms[idx[1]]
            bond = Bond(atom1, atom2)
            bonds.append(bond)
        return bonds


class Molecule:
    def __init__(self, name: str, id: Optional[int] = None):

        self.id = id
        self.name = name
        self.atoms: List[Atom] = []
        self.residues = []
        self.topology = Topology()

    def add_atom(self, atom:Atom):
        self.atoms.append(atom)

    @property
    def natoms(self):
        return len(self.atoms)

    @property
    def bond_index(self):
        return self.topology.bonds.get("index", np.array([]))

    @property
    def bonds(self)->List[Bond]:
        index = self.topology.bonds["index"]
        bonds = []
        for idx in index:
            # atom1 = self.atoms[idx[0]]
            # atom2 = self.atoms[idx[1]]
            bond = Bond(*idx)
            bonds.append(bond)
        return bonds

    @property
    def xyz(self):
        return np.array([atom["xyz"] for atom in self.atoms])

    def translate(self, dr):
        """
        translate this molcule by displacement vector dr

        Parameters
        ----------
        dr : np.ndarray
            (3, )

        Returns
        -------
        self
        """
        for atom in self.atoms:
            atom["xyz"] += dr
        return self

    def move_to(self, r, base:int=0):
        """
        move the molecule to the target position

        Parameters
        ----------
        r : np.ndarray
            (3, )
        base : int
            index of base atom

        Returns
        -------
        self
        """

        xyz = self.atoms[base]['xyz']
        dr = r - xyz
        self.translate(dr)
        return self

    def __repr__(self) -> str:
        return f"<Molecule: {self.name}>"
