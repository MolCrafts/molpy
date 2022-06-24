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

    def __init__(self, n_atoms, n_bonds, ):
        super().__init__(n_atoms, n_bonds)

    def get_bonds(self)->List[Bond]:

        pass

    def get_angles(self)->List[Angle]:

        pass

    def get_dihedrals(self)->List[Dihedral]:

        pass

