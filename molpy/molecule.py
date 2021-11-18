# author: Roy Kid
# contact: lijichen365@126.com
# date: 2021-11-14
# version: 0.0.1


from functools import reduce
from molpy.group import Group

class Molecule(Group):
    
    def __init__(self, name) -> None:
        super().__init__(name)
        self._groups = {}
        self._groupList = []
        
    def addGroup(self, group: Group, copy=False):
        
        if group.name in self._groups:
            raise KeyError(f'group {group.name} has defined, please change its name')
        if copy:
            group = group.copy()
        
        self._groups[group.name] = group
        self._groupList.append(group)
        self._atomList.extend(group._atomList)
        # self._bonds.update(group._bonds)
        self._bondList.extend(group._bondList)
        # self._angles.update(group._angles)
        self._angleList.extend(group._angleList)
        # self._dihedrals.update(group._dihedrals)
        self._dihedralList.extend(group._dihedralList)

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
    
    @property
    def groups(self):
        return self._groupList
    
    @property
    def angles(self):
        return self._angleList
    
    @property
    def dihedrals(self):
        return self._dihedralList
    
    @property
    def bonds(self):
        return self._bondList
    
    @property
    def atoms(self):
        return self._atomList