# author: Roy Kid
# contact: lijichen365@126.com
# date: 2021-10-17
# version: 0.0.1

from typing import Iterable, Union
from molpy.item import Item
from molpy.atom import Atom
from molpy.bond import Bond
import numpy as np
from itertools import dropwhile, combinations

class Group(Item):
    def __init__(self, name, **attrs) -> None:
        super().__init__(name)

        self._atoms = [] # node
        self._bonds = [] # edge
        self._groups = []
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
        # atom.parent = self
        if atom not in self._atoms:
            self.atoms.append(atom)
            self._adj[atom] = {}
                
    def addAtoms(self, atoms: Iterable[Atom]):
        for atom in atoms:
            self.addAtom(atom)
            
    def removeAtom(self, atom: Atom):
        """Remove atom from this group but not destory it

        Removes the atom and all adjacent bonds.
        Attempting to remove a non-existent node will raise an exception.

        Args:
            atom (Atom): [description]
        """
        # remove atom from atom list
        del self._atoms[self._atoms.index(atom)]
        
        # remove related bonds
        nbrs = list(self._adj[atom])
        for u in nbrs:
            del self._adj[u][atom]  # remove edges
        del self._adj[atom]  # remove node
    
    def getAtoms(self):
        """ get atoms from this group

        Returns:
            List: List of atoms
        """
        return self._atoms
    
    def getCovalentMap(self):
        """ calculate covalent map from atoms in this group.
        """        
        atoms = self.getAtoms()
        covalentMap = np.zeros((len(atoms), len(atoms)), dtype=int)
        visited = np.zeros_like(covalentMap, dtype=int)
        
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
    
    @property
    def nbonds(self):
        return len(self.getBonds())
            
    def setTopoByCovalentMap(self, covalentMap: np.ndarray):
        """ set topology info by a numpy-like covalent map.

        Args:
            covalentMap (np.ndarray): 2-d ndarray
        """
        atoms = self.getAtoms()
        for i, nbond in np.ndenumerate(covalentMap):
            if nbond == 1:
                atom1 = atoms[i[0]]
                atom2 = atoms[i[1]]
                self._adj[atom1][atom2] = atom1.bondto(atom2)
        
    @property
    def natoms(self):
        return len(self.getAtoms())
    
    def getAtomByName(self, atomName):
        for atom in self._atoms:
            if isinstance(atom, Atom) and atom.name == atomName:
                return atom
    
    def getGroupByName(self, groupName):
        for group in self._groups:
            if isinstance(group, Group) and group.name == groupName:
                return group
    
    def __getitem__(self, idx):
        if isinstance(idx, str):
            for atom in self.getAtoms():
                if atom.name == idx:
                    return atom
                
        elif isinstance(idx, int):
            return self.getAtoms()[idx]
        
    def deleteAtom(self, name):
        
        dropwhile(lambda atom: atom.name == name, self._atoms)
        
    def update(self):
        pass
    
    def getBonds(self):

        b = set()
        for u, bonds in self._adj.items():
            for v, bond in bonds.items():
                b.add(bond)
        b = list(b)
        self._bonds = b
        return b

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
    
    def addBondBy(self, atom1, atom2, by):
        atoms = self.getAtoms()
        tmp = {}
        for atom in atoms:
            tmp[atom.get(by)] = atom
        return tmp[atom1].bondto(tmp[atom2])
    
    def addBondsByDict(self, dict, ref):
        atoms = self.getAtoms()
        tmp = {}
        for atom in atoms:
            tmp[atom.get(ref)] = atom
        for c, ps in dict.items():
            for p in ps:
                self.addBond(tmp[c], tmp[p])
    
    def addBond(self, atom1, atom2, **bondProp):
        
        # add nodes
        self.addAtoms([atom1, atom2])
        bond = atom1.bondto(atom2, **bondProp)
        
        # add edges
        bond.update(**bondProp)
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
    
    def getSubGroup(self, name, atoms):
        """Specify some atoms in the group and return the subgroup composed of these atoms

        Args:
            name (str): name of subgroup
            atoms (Iterable[Atom]): list of atoms contained in the subgroup

        Returns:
            Group: new subgroup
        """
        # check
        atoms = list(set(atoms))
        for atom in atoms:
            if atom not in self:
                raise ValueError(f'{atom} not in this group')
        
        # add atoms
        subgroup = Group(name)
        subgroup.addAtoms(atoms)
        
        # add bonds
        for atom in atoms:
            for bondedAtom in atom.bondedAtoms:
                if bondedAtom in subgroup:
                    bond = subgroup._adj.get(atom, {})
                    bond[bondedAtom] = atom.bonds[bondedAtom]
        return subgroup

    def serialize(self):
                   
        props = super().serialize(['_atoms', '_bonds', '_adj'])
        
        # _atoms
        atoms = self.getAtoms()
        tmp = []
        for atom in atoms:
            tmp.append(atom.serialize())
        props['_atoms'] = tmp
        
        # _bonds
        
        return props
    
    def deserialize(self, o):
        if o['_itemType'] != 'Group':
            raise TypeError(f'Group class get incompitable infomation')
        super().deserialize(o, ['_atoms'])
        atoms = o['_atoms']
        tmp = []
        for atom in atoms:
            tmp.append(Atom('').deserialize(atom))
        self.set('_atoms', tmp)
        
        # adj = o['_adj']
        # for c, ps in adj.items():
        #     for p, bond in ps.items():
        #         cuuid = c['_uuid']
        #         puuid = p['_uuid']
        #         for atom in atoms:
        #             if atom.uuid == cuuid:
        #                 catom = atom
        #                 break
        #         for atom in atoms:
        #             if atom.uuid == puuid:
        #                 patom = atom
        #                 break
        #         self.addBond(catom, patom)
        
        return self
