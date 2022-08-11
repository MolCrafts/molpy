# author: Roy Kid
# contact: lijichen365@126.com
# date: 2022-08-10
# version: 0.0.1

from .item import Atom, Bond, Angle, Dihedral

class Topology:

    def __init__(self, ):

        self._bonds = []
        self._angles = []

    def add_bond(self, i, j, **properties):

        bond = Bond(i, j, type=type)

        self._bonds.append(bond)