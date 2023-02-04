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
        return self.get('name', self.get('type', 'unknown'))

    @property
    def xyz(self):
        return self.get('xyz', self.get('positions'))

    @property
    def type(self):
        return self.get('type')

    @property
    def charge(self):
        return self.get('charge', 0)

    @property
    def mass(self):
        return self.get('mass', 0)

class Bond(dict):

    def __init__(self, atom1, atom2, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.atom1 = atom1
        self.atom2 = atom2

class Residue:

    def __init__(self, name:str, id:Optional[int]=None):

        self.id = id
        self.name = name
        self.atoms = []
        self.topology = Topology()


    @property
    def natoms(self):
        return len(self.atoms)

    def add_atom(self, atom:Atom):
        self.atoms.append(atom)

    def add_bonds(self, bonds):
        self.topology.add_bonds(bonds)

    @property
    def bonds(self)->List[Bond]:
        index = self.topology.bonds['index']
        bonds = []
        for idx in index:
            atom1 = self.atoms[idx[0]]
            atom2 = self.atoms[idx[1]]
            bond = Bond(atom1, atom2)
            bonds.append(bond)
        return bonds



class Molecule:

    def __init__(self, name:str, id:Optional[int]=None):

        self.id = id
        self.name = name
        self.atoms:List[Atom] = []
        self.residues = []
        self.topology = Topology()

    def add_atom(self, **props):
        self.atoms.append(Atom(**props))
    
    @property
    def natoms(self):
        return len(self.atoms)

    @property
    def bond_index(self):
        return self.topology.bonds['index']

    @property
    def xyz(self):
        return np.array([atom['xyz'] for atom in self.atoms])

    def translate(self, dr):
        for atom in self.atoms:
            atom['xyz'] += dr
        return self

    def __repr__(self) -> str:
        return f"<Molecule: {self.name}>"
