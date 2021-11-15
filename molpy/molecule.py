# author: Roy Kid
# contact: lijichen365@126.com
# date: 2021-11-14
# version: 0.0.1


from functools import reduce
from molpy.base import Graph
from molpy.group import Group

class Molecule(Graph):
    
    def __init__(self, name) -> None:
        super().__init__(name)
        self._groups = {}
        self._groupList = []
        self._bonds = {}
        self._bondList = []
        
    def addGroup(self, group: Group, copy=False):
        
        if group.name in self._groups:
            raise KeyError(f'group {group.name} has defined, please change its name')
        if copy:
            group = group.copy()
        
        self._groups[group.name] = group
        self._groupList.append(group)
        self._bonds.update(group._bonds)
        self._bondList.extend(group.bonds)
        
    def addBond(self, atom, btom, **attr):
        
        bond = atom.bondto(btom, **attr)
        if atom not in self._bonds:
            self._bonds[atom] = {}

        if btom not in self._bonds:
            self._bonds[btom] = {}
        self._bonds[atom][btom] = bond
        self._bonds[btom][atom] = bond
        self._bondList.append(bond)
        
    def addBondByName(self, atomName, btomName, **attr):
        
        atom = self.getAtomByName(atomName)
        btom = self.getAtomByName(btomName)
        self.addBond(atom, btom, **attr)

    def getAtomByName(self, atomName):
        atomName, groupName = atomName.split('@')
        group = self._groups[groupName]
        atom = group.getAtomByName(atomName)
        return atom
    
    @property
    def natoms(self):
        return reduce(lambda ans, g: ans+g.natoms, self._groupList, 0)
    
    @property
    def nbonds(self):
        return len(self._bondList)