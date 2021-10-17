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
        self._atoms
        
    def add(self, item):
        self.items.append(item)
    
    def getAtoms(self):
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
    
    def setTopoByCovalentMap(self, covalentMap):
        atoms = self.getAtoms()
        for i, nbond in np.ndenumerate(covalentMap):
            if nbond == 1:
                atoms[i[0]].bondto(atoms[i[1]])
        
    @property
    def natoms(self):
        return len(self.getAtoms)
    
    @property
    def atoms(self):
        return self.getAtoms()
    
    def getAtomByName(self, atomName):
        pass
    
    def getGroupByName(self, groupName):
        pass
    
    def __getitem__(self, idx):
        if isinstance(idx, str):
            for atom in self.atoms:
                if atom.name == idx:
                    return idx
                
        elif isinstance(idx, int):
            return self.atoms[idx]
    