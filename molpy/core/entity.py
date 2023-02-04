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


class Molecule:

    def __init__(self, id:Optional[int]=None, name:Optional[str]=None, ):

        self.id = id
        self.name = name
        self.atoms = StaticSOA()
        self.props = {}
        self.topology = Topology()
    
    @property
    def natoms(self):
        return self.atoms.size
