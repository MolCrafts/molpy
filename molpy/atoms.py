# author: Roy Kid
# contact: lijichen365@126.com
# date: 2022-06-22
# version: 0.0.1

from typing import List
from molpy.graph import Graph

class Bond:

    pass

class Angle:

    pass

class Dihedral:

    pass

class Atoms(Graph):

    def __init__(self):
        super().__init__()

    def get_bonds(self)->List[Bond]:

        pass

    def get_angles(self)->List[Angle]:

        pass

    def get_dihedrals(self)->List[Dihedral]:

        pass

    def add_atoms(self, **attr):

        self.add_nodes(**attr)

    def update(self, atoms, isAtom:bool=True, isBond:bool=True, method='replace'):

        if isAtom:
            self.update_nodes(**atoms.atoms)

        if isBond:
            pass

    @property
    def atoms(self):
        return self.attribs.nodes

    @property
    def n_atoms(self):
        return self.attribs._n_nodes
