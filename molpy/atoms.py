# author: Roy Kid
# contact: lijichen365@126.com
# date: 2022-06-22
# version: 0.0.1

from typing import Iterable, List
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

    def add_attr(self, to_, counter, **attr):
        """
        a common method to add attributes to a dictionary. This method will append `attr` to the dictionary `to_`, or create a new key-value pair if item in attr not in `to_`. And this method will return a list of index. NOTE: the methos only check if items in `attr` are aligned, and support item in `attr` all or all not in `to_`. If an item in `attr` is not in `to_`, another item already in `to_`, this method will not raise error and the behavior is undefined.

        example:
            >>> to_ = {'a': [1, 2, 3]},
            >>> attr = {'a': [7, 8, 9]},
            >>> add_attr(to_, 3, **attr)
            >>> [3, 4, 5]

        Args:
            to_ (dict): a dictionary to add attributes to
            counter (int): a counter to start indexing from

        Returns:
            int: indices that were added
        """

        length = self.check_aligned(**attr)
        for k, v in attr.items():
            if k not in to_:
                to_[k] = v
            else:
                to_[k] = np.concatenate((self.atoms[k], v))
        ids = self.gen_index(counter, length)
        return ids

    def add_atoms(self, **attr):

        ids = self.add_attr(self.atoms, self._n_atoms, **attr)
        self._n_atoms += len(ids)
        return ids

    def add_bonds(self, **attr):

        ids = self.add_attr(self.bonds, self._n_bonds, **attr)
        self._n_bonds += len(ids)
        return ids

    def add_angles(self, **attr):

        ids = self.add_attr(self.angles, self._n_angles, **attr)
        self._n_angles += len(ids)
        return ids

    def add_dihedrals(self, **attr):

        ids = self.add_attr(self.dihedrals, self._n_dihedrals, **attr)
        self._n_dihedrals += len(ids)
        return ids
        

    def check_aligned(self, **attr):

        lengths = [len(v) for v in attr.values()]
        if len(set(lengths)) != 1:
            raise ValueError("Not aligned")
        else:
            return lengths[0]

    def gen_index(self, start, length):

        return [i for i in range(start, length)]


class Atoms:

    def __init__(self, positions, topo):
        
        self._topo = Topo()
        self._attr = Attrib()

        if positions:
            self._attr.add_attr()

    def __getitem__(self, key):

        if isinstance(key, str):
            return self._attr.atoms[key]
        elif isinstance(key, (int, slice)):
            return self.get_subatoms(key)

    def add_atoms(self, **attr):
        """
        Add per-atom attributes. If the attribute already exists, it will be appended. If not, it will be created. You can provide multiple attributes at once, either existing or new. In order to keep the attributes aligned, the existing attributes must be aligned. The length of new attribute must be the same as the length of the existing attributes. That means after adding the new attributes, the length of the existing attributes must be the same as the length of the new attributes.

        Args:
            **attr: The attributes to add.

        Examples:
            >>> atoms.add_atoms(atom_id=np.array(1), atom_name=['C'], atom_type=['C'], atom_charge=[0])

        """

        atomids = self._attr.add_atoms(**attr)
        self._topo.add_atoms(atomids)

    def add_bonds(self, connect:List[Iterable[int]], **attr):
        """
        Add bonds with attributes. The length of the connect list must be the same as the length of the attributes. The attributes must be aligned.

        Args:
            connect (List[Iterable[int]]): The bonds to add.
            **attr: The attributes to add.

        Examples:
            >>> atoms.add_bonds([[0, 1]], bond_type=['C'])

        """
        bondids = self._attr.add_bonds(**attr)
        self._topo.add_bonds(connect, bondids)

    def add_angles(self, connects:List[Iterable[int]], **attr):
        """
        Add angles with attributes. The length of the connect list must be the same as the length of the attributes. The attributes must be aligned.

        Args:
            connects[List[int]]: The angles to add.
            **attr: The attributes to add.

        Examples:
            >>> atoms.add_angles([[0, 1, 2]], angle_type=['C'])

        """
        angleids = self._attr.add_angles(**attr)
        self._topo.add_angles(connects, angleids)

    def add_dihedrals(self, connects:List[Iterable[int]], **attr):
        """
        Add dihedrals with attributes. The length of the connect list must be the same as the length of the attributes. The attributes must be aligned.

        Args:
            connects[List[int]]: The dihedrals to add.
            **attr: The attributes to add.

        Examples:
            >>> atoms.add_dihedrals([[0, 1, 2, 3]], dihedral_type=['C'])

        """
        dihedralids = self._attr.add_dihedrals(**attr)
        self._topo.add_dihedrals(connects, dihedralids)

    def get_bonds(self)->List[Bond]:

        pass

    def get_angles(self)->List[Angle]:

        pass

    def get_dihedrals(self)->List[Dihedral]:

        pass

    def get_bonds(self, i, j)->Bond:

        pass

    def update(self, atoms:'Atoms', isAtom:bool=True, isBond:bool=True):

        # check if the number of atoms is the same
        if isAtom:
            if atoms.n_atoms != self.n_atoms:
                raise ValueError("The number of atoms is not the same")
            else:
                self._attr.atoms.update(atoms._attr.atoms)

    @property
    def atoms(self):
        return self._attr.atoms

    @property
    def n_atoms(self):
        return self._attr._n_atoms

    @property
    def bonds(self):
        return self._attr.bonds

    @property
    def n_bonds(self):
        return self._attr._n_bonds

    def append(self, atoms:'Atoms'):

        pass


class Residue(Atoms):

    def __init__(self, name:str):
        super().__init__()
        self.name = name

