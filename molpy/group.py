# author: Roy Kid
# contact: lijichen365@126.com
# date: 2021-10-17
# version: 0.0.1

from molpy.abc import Item
from molpy.atom import Atom
import numpy as np
from itertools import dropwhile, combinations

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
        item.parent = self
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
        """ calculate covalent map from atoms in this group.
        """
        if self.status == 'new' and getattr(self, '_covalentMap', None):
            return self._covalentMap
        
        atoms = self.getAtoms()
        covalentMap = np.zeros((len(atoms), len(atoms)))
        visited = np.zeros_like(covalentMap)
        
        def find(nodes, vis):
            nextLevelNodes = []
            for node in nodes:
                nodeidx = atoms.index(node)
                if vis[nodeidx] == 1:
                    continue
                vis[nodeidx] = 1
                covalentMap[rootidx, nodeidx] = depth
                nextLevelNodes.extend(dropwhile(lambda node: vis[atoms.index(node)] == 1, node.bondedAtoms))
                nextLevelNodes = list(set(nextLevelNodes))
            return nextLevelNodes
        
        for root in atoms:
            rootidx = atoms.index(root)
            vis = visited[rootidx]
            vis[rootidx] = 1
            depth = 1
            nodes = find(root.bondedAtoms, vis)
            depth += 1
            while True:
                if nodes == []:
                    break
                nodes = find(nodes, vis)
                depth += 1
        self._covalentMap = covalentMap
        return covalentMap
    
    @property
    def covalentMap(self):
        return self._covalentMap
    
    @property
    def atoms(self):
        return self.getAtoms()
            
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
        for item in self.items:
            if isinstance(item, Group) and item.name == atomName:
                return item
    
    def getGroupByName(self, groupName):
        for item in self.items:
            if isinstance(item, Group) and item.name == groupName:
                return item
    
    def __getitem__(self, idx):
        if isinstance(idx, str):
            for atom in self.getAtoms():
                if atom.name == idx:
                    return atom
                
        elif isinstance(idx, int):
            return self.getAtoms()[idx]
        
    def delete(self, name):
        
        dropwhile(lambda item: item.name == name, self.items)
        
    def update(self):
        pass
    
    def getBonds(self):
        self.check_properties(covalentMap=np.ndarray)
        covalentMap = self.covalentMap
        self._bonds = []
        for index, nbond in np.ndenumerate(np.triu(covalentMap, 1)):
            if nbond == 1:
                self._bonds.append(
                    (self._atoms[index[0]], self._atoms[index[1]])
                )
        return self._bonds

    def getAngles(self):
        angles = set()
        for atom in self.getAtoms():

            for edge in combinations(atom.bondedAtoms, 2):
                angles.add(
                    (edge[0], atom, edge[1])
                )
                
        self._angles = list(angles)
        return self._angles