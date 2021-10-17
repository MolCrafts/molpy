# author: Roy Kid
# contact: lijichen365@126.com
# date: 2021-10-17
# version: 0.0.1

from molpy.abc import Item
from molpy.atom import Atom
import numpy as np

class Group(Item):
    
    def __init__(self, name) -> None:
        super().__init__(name)
        self.items = self._container
        self._atoms = []
        
    def add(self, item):
        """ Add an Atom or Group to this group as an affiliated item.

        Args:
            item (Item): derived from Item
        """
        self.items.append(item)
        self.status = 'modified'
    
    def getAtoms(self):
        """ get atoms from all the items in this group

        Returns:
            List: List of atoms
        """
        if self.status == 'new':
            return self._atoms
        else:
            for item in self.items:
                if isinstance(item, Atom):
                    self._atoms.append(item)
                elif isinstance(item, Group):
                    self._atoms.extend(item)
            self.status = 'new'
            return self._atoms
    
    def getCovalentMap(self):
        pass
    
    def setTopoByCovalentMap(self, covalentMap: np.ndarray):
        """ set topology info by a numpy-like covalent map.

        Args:
            covalentMap (np.ndarray): 2-d ndarray
        """
        atoms = self.getAtoms()
        for i, nbond in np.ndenumerate(covalentMap):
            if nbond == 1:
                atoms[i[0]].bondto(atoms[i[1]])
        
    @property
    def natoms(self):
        return len(self.getAtoms())
    
    def getAtomByName(self, atomName):
        pass
    
    def getGroupByName(self, groupName):
        pass
    
    def __getitem__(self, idx):
        if isinstance(idx, str):
            for atom in self.getAtoms():
                if atom.name == idx:
                    return atom
                
        elif isinstance(idx, int):
            return self.getAtoms()[idx]
    