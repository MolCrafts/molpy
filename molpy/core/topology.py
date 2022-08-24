# author: Roy Kid
# contact: lijichen365@126.com
# date: 2022-08-10
# version: 0.0.1

from .item import Atom, Bond, Angle, Dihedral
from .graph import Graph

class Topology:

    def __init__(self, ):

        self._bonds = []
        self._angles = []
        self._dihedrals = []

        self._graph = Graph()

    def add_bond(self, bond:Bond):

        self._bonds.append(bond)
        # self._graph.add_bond(bond[0], bond[1])

    def add_angle(self, angle:Angle):

        self._angles.append(angle)

    def add_dihedral(self, dihedral:Dihedral):

        self._dihedrals.append(dihedral)

    def add_bond_by_index(self, i, j, **properties):

        bond = Bond(i, j, type=properties.get('type', None))

        self._bonds.append(bond)

    def add_angle_by_index(self, i, j, k, **properties):

        angle = Angle(i, j, k, type=properties.get('type', None))

        self._angles.append(angle)

    def add_dihedral_by_index(self, i, j, k, l, **properties):
            
        dihedral = Dihedral(i, j, k, l, type=properties.get('type', None))

        self._dihedrals.append(dihedral)

    def add_edge(self, i, j):

        pass