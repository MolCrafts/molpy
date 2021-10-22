# author: Roy Kid
# contact: lijichen365@126.com
# date: 2021-10-17
# version: 0.0.1

from typing import Iterable, Union
from molpy.base import Item
from molpy.atom import Atom
import numpy as np
from itertools import dropwhile, combinations

class Group(Item):
    
    def __init__(self, name, **attrs) -> None:
        super().__init__(name)
        self.items = self._container
        self._atoms = [] # node
        self._adj = {}
        for attr in attrs:
            setattr(self, attr, attrs[attr])
            
    def add(self, item):
        if isinstance(item, Atom):
            self.addAtom(item)
        
    def addAtom(self, atom: Atom):
        """ Add an atom to this group

        Args:
            atom ([Atom]): an atom instance
        """
        atom.parent = self
        if atom not in self.items:
            self.atoms.append(atom)
            self.status = 'modified'
            self._adj[atom] = {}
                
    def addAtoms(self, atoms: Iterable[Atom]):
        for atom in atoms:
            self.addAtom(atom)
            
    def removeAtom(self, atom: Atom):
        """Remove atom from this graph but not destory it

        Removes the atom and all adjacent bonds.
        Attempting to remove a non-existent node will raise an exception.

        Args:
            atom (Atom): [description]
        """
        # remove atom from item list
        del self.items[self.items.index(atom)]
        self.status = 'modified'
        
        # remove related bonds
        nbrs = list(self._adj[atom])
        for u in nbrs:
            del self._adj[u][atom]  # remove edges
        del self._adj[atom]  # remove node
    
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
    
    def addBondByIndex(self, atomIdx, atomJdx, **bondType):
        atom1 = self.getAtoms()[atomIdx]
        atom2 = self.getAtoms()[atomJdx]
        self.addBond(atom1, atom2, *bondType)
    
    def getBondByIndex(self, atomIdx, atomJdx):
        atom1 = self.getAtoms()[atomIdx]
        atom2 = self.getAtoms()[atomJdx]
        return atom1.bonds[atom2]
    
    def addBond(self, atom1, atom2, **bondProp):
        
        # add nodes
        self.addAtoms([atom1, atom2])
        bond = atom1.bondto(atom2, *bondProp)
        
        # add edges
        bond = self._adj[atom1].get(atom2, bond)
        bond.update(bondProp)
        self._adj[atom1][atom2] = bond
        self._adj[atom2][atom1] = bond
        
        
    def removeBond(self, atom1, atom2):
        try:
            del self._adj[atom1][atom2]
            if atom1 != atom2:
                del self._adj[atom2][atom1]
        except:
            raise KeyError(f'either {atom1} or {atom2} not in this graph')
        
        atom1.removeBond(atom2)
    
    def __contains__(self, n):
        try:
            return n in self.getAtoms()
        except TypeError:
            return False
        
    def __len__(self):
        return len(self.getAtoms())