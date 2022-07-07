# author: Roy Kid
# contact: lijichen365@126.com
# date: 2022-06-22
# version: 0.0.1

from typing import List
import numpy as np
from molpy.topo import Topo

class Bond:

    pass

class Angle:

    pass

class Dihedral:

    pass

class Attrib:

    def __init__(self) -> None:
        
        self.atoms = {}
        self._n_atoms = 0
        self.bonds = {}
        self._n_bonds = 0
        self.angles = {}
        self._n_angles = 0
        self.dihedrals = {}
        self._n_dihedrals = 0

    def add_atoms(self, **attr):

        length = self.check_aligned(**attr)
        
        for k, v in attr.items():
            if k not in self.atoms:
                self.atoms[k] = v
            else:
                self.atoms[k] = np.concatenate((self.atoms[k], v))

        atomids = self.gen_index(self._n_atoms, length)
        self._n_atoms += length
        return atomids

    def add_bonds(self, **attr):

        length = self.check_aligned(**attr)
        for k, v in attr.items():
            if k not in self.bonds:
                self.bonds[k] = v
            else:
                self.bonds[k] = np.concatenate((self.bonds[k], v))

        bondids = self.gen_index(self._n_bonds, length)
        self._n_bonds += length
        return bondids

    def check_aligned(self, **attr):

        lengths = [len(v) for v in attr.values()]
        if len(set(lengths)) != 1:
            raise ValueError("Not aligned")
        else:
            return lengths[0]

    def gen_index(self, start, length):

        return [i for i in range(start, length)]


class Atoms:

    def __init__(self):
        
        self._topo = Topo()
        self._attr = Attrib()

    def add_atoms(self, **attr):
        """
        Add per-atom attributes. If the attribute already exists, it will be appended. If not, it will be created. You can provide multiple attributes at once, either existing or new. In order to keep the attributes aligned, the existing attributes must be aligned. The length of new attribute must be the same as the length of the existing attributes. That means after adding the new attributes, the length of the existing attributes must be the same as the length of the new attributes.

        Args:
            **attr: The attributes to add.

        Examples:
            >>> atoms.add_atoms(atom_id=np.array(1), atom_name=['C'], atom_type=['C'], atom_charge=[0])

        """

        atomids = self._attr.add_atoms(self, **attr)
        self._topo.add_nodes(atomids)

    def add_bonds(self, connect:List[List[int]], **attr):
        """
        Add bonds with attributes. The length of the connect list must be the same as the length of the attributes. The attributes must be aligned.

        Args:
            connectList[List[int]]: The bonds to add.
            **attr: The attributes to add.

        Examples:
            >>> atoms.add_bonds([[0, 1]], bond_type=['C'])

        """
        bondids = self._attr.add_bonds(self, **attr)
        self._topo.add_edges(connect, bondids)

    def add_angles(self, connects:List[List[int]], **attr):
        """
        Add angles with attributes. The length of the connect list must be the same as the length of the attributes. The attributes must be aligned.

        Args:
            connects[List[int]]: The angles to add.
            **attr: The attributes to add.

        Examples:
            >>> atoms.add_angles([[0, 1, 2]], angle_type=['C'])

        """
        angleids = self._attr.add_angles(self, **attr)
        self._topo.add_edges(connects, angleids)
        
    

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
