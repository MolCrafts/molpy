# author: Roy Kid
# contact: lijichen365@126.com
# date: 2021-10-17
# version: 0.0.1

from typing import Literal, Iterable
from molpy.angle import Angle
from molpy.base import Graph
from molpy.atom import Atom
from molpy.bond import Bond
import numpy as np
from itertools import dropwhile, combinations

from molpy.dihedral import Dihedral

class Group(Graph):
    """Group describes a bunch of atoms which connect with others and represents a molecule or a functional gropu
    """
    def __init__(self, name, **attr):
        """Initialize a group.

        Args:
            name (str): group's name
        """
        super().__init__(name)
        self._atoms = {}
        self._atomList = [] # [Atom]
        self._atomIndices = {}
        # self._bonds = {} # _bonds: {Atom: {Btom: Bond, Ctom: Bond}}
        self._bondList = []
        # self._angles = {}
        self._angleList = []
        # self._dihedrals = {}
        self._dihedralList = []
        self.update(attr)
            
    def add(self, item, copy=False):
        """(leave for backward compatible)Add an atom or something to this group. Unless the object passed is an atom instance or it not works.

        Args:
            item ([type]): [description]
            copy (bool, optional): [description]. Defaults to False.
        """
        if isinstance(item, Atom):
            self.addAtom(item, copy)
        
    def addAtom(self, atom: Atom, copy=False):
        """Add an atom to this group.

        Args:
            atom (Atom): atom to add to this group
            copy (bool, optional): if call atom.copy(). Defaults to False.
        """
        if atom not in self._atomList:
            if copy:
                atom = atom.copy()
            self._atomIndices[atom] = len(self._atomList)
            self._atomList.append(atom)
            self._atoms[atom.name] = atom
            atom.parent = self
                
    def addAtoms(self, atoms, copy=False):
        """add a sequence of atoms.

        Args:
            atoms (Iterable[Atom]): a set of atoms
            copy (bool, optional): if call atom.copy(). Defaults to False.
        """
        for atom in atoms:
            self.addAtom(atom, copy)
            
    def removeAtom(self, atom: Atom):
        """remove atom from this group, but not deconstruct it.

        Args:
            atom (Atom): atom to be removed

        Raises:
            KeyError: WHEN atom not in this group
        """
        self._atomList.remove(atom)
        
        bonds = self._bonds
        try:
            nbrs = list(bonds[atom])  # list handles self-loops (allows mutation)
        except KeyError as err:  # NetworkXError if n not in self
            raise KeyError(f"{atom} is not in the group.") from err
        for u in nbrs:
            del bonds[u][atom]  # remove all edges atom-u in graph
        del bonds[atom]
        
    def removeAtoms(self, atoms):
        """remove a set of atoms

        Args:
            atoms (Iterable[Atom]): [description]
        """
        for atom in atoms:
            self._atomList.remove(atom)
            self.removeAtom(atom)
        
    @property
    def atoms(self):
        return self._atomList
    
    def getAtoms(self):
        """return all the atoms in this group

        Returns:
            List[Atom]: a list of atom
        """
        return self._atomList

    @property
    def natoms(self):
        return len(self._atomList)
    
    @property
    def nangles(self):
        return len(self._angleList)
    
    @property
    def dihedrals(self):
        return self._dihedralList
    
    @property
    def ndihedrals(self):
        return len(self._dihedralList)
    
    def hasAtom(self, atom: Atom, ref=None):
        """if the atom in this group

        Args:
            atom (Atom): atom to be checked

        Returns:
            bool: result of if atom in the group
        """
        if ref is None:
            return atom in self._atomList
        elif ref == 'name':
            return atom.name in self._atoms
        
    def addBond(self, atom, btom, **attr):
        """define bond between two passed atoms, and set bond properties

        Args:
            atom (Atom): one atom
            btom (Atom): another atom
        """
        if btom not in atom.bondedAtoms or atom not in btom.bondedAtoms:

            bond = atom.bondto(btom, **attr)
            # if atom not in self._bonds:
            #     self._bonds[atom] = {}

            # if btom not in self._bonds:
            #     self._bonds[btom] = {}
            # self._bonds[atom][btom] = bond
            # self._bonds[btom][atom] = bond
            self._bondList.append(bond)
        
    def addBonds(self, atomList, **attr):
        """Batch add bonds. atomList followed format, [(atom, btom)] without bond's properties, or [(atom, btom, {key: value})]. Dict in the atomList is the special properties for the bond, and attr is the general properties for all bond. Special property will cover properties in attr.

        Args:
            atomList ([type]): [description]

        Raises:
            ValueError: [description]
        """
        for e in atomList:
            ne = len(e)
            if ne == 3:
                u, v, dd = e
            elif ne == 2:
                u, v = e
                dd = {}  # doesn't need edge_attr_dict_factory
            else:
                raise ValueError(f"bond tuple {e} must be a 2-tuple or 3-tuple.")

            attr.update(dd)
            self.addBond(u, v, **attr)
            
    def removeBond(self, atom, btom):
        """remove bond between two atoms

        Args:
            atom (Atom): one atom
            btom (Atom): another atom

        Raises:
            KeyError: WHEN bond not exists
        """
        try:
            del self._bonds[atom][btom]
            if atom != btom:  # self-loop needs only one entry removed
                del self._bonds[btom][atom]
        except KeyError as err:
            raise KeyError(f"The bond {atom}-{btom} is not in the graph") from err
        
    def removeBonds(self, atomList):
        """remove a bunch of bond

        Args:
            atomList (Iterable): same format with addBonds
        """
        for atoms in atomList:
            atom, btom = atoms[:2]
            if atom in self._bonds and btom in self._bonds[atom]:
                del self._bonds[atom][btom]
                if atom != btom:
                    del self._bonds[btom][atom]
        
     
    def getSubGroup(self, name, atoms):
        """Specify some atoms in the group and return the subgroup composed of these atoms

        Args:
            name (str): name of subgroup
            atoms (Iterable[Atom]): list of atoms contained in the subgroup

        Returns:
            Group: new subgroup
        """
        # check
        atoms = set(atoms)
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
                    subgroup.addBond(atom, bondedAtom)
        return subgroup

    def hasBond(self, atom, btom):
        """if bond between atom and btom in this graph

        Args:
            atom (Atom): one atom
            btom (Atom): anther atom

        Returns:
            bool: result
        """
        if self._bonds[atom].get(btom, False):
            return True
        else:
            return False
        
    def neighbors(self, atom):
        """return atom's neighbors. equivalent to atom.bondedAtoms

        Args:
            atom (Atom): atom

        Returns:
            List[Atom]: atom's bondedAtom
        """
        return list(self._bonds[atom].keys())
    
    @property
    def bonds(self):
        return self.getBonds()
    
    @property
    def nbonds(self):
        return len(self.getBonds())
    
    def getBond(self, atom, btom):
        """get a certain bond specified with two atoms

        Args:
            atom (Atom): one atom
            btom (Atom): anther atom

        Returns:
            Bond: bond
        """
        return self._bonds[atom].get(btom, False)
    
    def getBonds(self, format='bond'):
        """get all the bonds in this graph

        Returns:
            List[Bond]: a list of bond
        """
        if format == 'bond':
            bonds = set()
            for u, nbs in self._bonds.items():
                for nb in nbs:
                    bonds.add(self._bonds[u][nb])
            return list(bonds)
        elif format == 'index':
            return self.getAdjacencyList()
        
    def getAngles(self, format='angle'):
        if format == 'angle':
            return self._angleList
        elif format == 'index':
            pass
    
    def getAdjacencyList(self):
        bonds = self.bonds
        adjlist = []
        for bond in bonds:
            atom, btom = bond
            adjlist.append([self._atomIndices[atom], self._atomIndices[btom]])
        return adjlist        
    
    def getCovalentMap(self, max_distance=None):
        """return the covalentMap of this graph.

        Args:
            max_distance (int, optional): depth of search. Defaults to None.

        Returns:
            np.ndarray: a matrix of topology distance
        """
        atoms = self.getAtoms()
        covalentMap = np.zeros((len(atoms), len(atoms)), dtype=int)
        visited = np.zeros_like(covalentMap, dtype=int)
        
        def limit():
            if max_distance is None:
                return True
            else:
                return depth < max_distance
        
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
            while limit():
                if nodes == []:
                    break
                nodes = find(nodes, vis)
                depth += 1
        self._covalentMap = covalentMap
        # return atoms, covalentMap
        return covalentMap
    
    @property
    def covalentMap(self):
        return self._covalentMap
            
    def setTopoByCovalentMap(self, covalentMap: np.ndarray):
        """Using a covalentMap to describe topology of the group

        Args:
            covalentMap (np.ndarray): a square & symmety matrix
        """
        atoms = self.getAtoms()
        for i, nbond in np.ndenumerate(covalentMap):
            if nbond == 1:
                atom1 = atoms[i[0]]
                atom2 = atoms[i[1]]
                self.addBond(atom1, atom2)
        
    def getAtomByName(self, atomName):
        """look up an atom by its name

        Args:
            atomName (str): name of the atom

        Returns:
            Atom: return the atom if can be find in graph
        """
        for atom in self._atomList:
            if atom.name == atomName:
                return atom
            
    def getAtomBy(self, by, value):
        """look up an atom by its certain property. NOTE only return the first result

        Args:
            by (str): property
            value (Any): value 

        Returns:
            Atom: if found else NoneType
        """
        for atom in self._atomList:
            if atom.get(by) == value:
                return atom
    
    def __getitem__(self, idx):
        if isinstance(idx, str):
            for atom in self.getAtoms():
                if atom.name == idx:
                    return atom
                
        elif isinstance(idx, int):
            return self.getAtoms()[idx]
        
    def addAngle(self, itom, jtom, ktom, **attr):
        try:
            angle = self._angles[itom][jtom][ktom]
            angle = self._angles[ktom][jtom][itom]
        except KeyError:
            angle = Angle(itom, jtom, ktom, **attr)
            self._angles.setdefault(itom, {}).setdefault(jtom, {}).setdefault(ktom, angle)
            self._angles.setdefault(ktom, {}).setdefault(jtom, {}).setdefault(itom, angle)
            self._angleList.append(angle)        
        return angle
    
    def addAngleByName(self, name1, name2, name3, **attr):
        itom, jtom, ktom = map(self.getAtomByName, [name1, name2, name3])
        return self.addAngle(itom, jtom, ktom, **attr) 

    def searchAngles(self):
        """search all the angles in this group

        Returns:
            List[Angle]: all angles in this group
        """
        # itom-jtom(self)-ktom
        
        for jtom in self.getAtoms():
            if len(jtom.bondedAtoms) < 2:
                continue
            for (itom, ktom) in combinations(jtom.bondedAtoms, 2):
                self.addAngle(itom, jtom, ktom)
                        
        return self._angleList
    
    def searchDihedrals(self):
        
        # itom-jtom(self)-ktom-ltom
        
        # for jtom in self.getAtoms():
        #     if len(jtom.bondedAtoms) < 2:
        #         continue
        #     for (itom, ktom) in combinations(jtom.bondedAtoms, 2):
                
        #         for ltom in filterfalse(lambda atom: atom in (itom, jtom), ktom.bondedAtoms):
        #             try:
        #                 dihe = self._dihedrals[itom][jtom][ktom][ltom]
        #                 dihe = self._dihedrals[ltom][ktom][jtom][itom]
        #             except KeyError:
        #                 dihe = Dihedral(itom, jtom, ktom, ltom)
        #                 self._dihedrals.setdefault(itom, {}).setdefault(jtom, {}).setdefault(ktom, {}).setdefault(ltom, dihe)
        #                 self._dihedrals.setdefault(ltom, {}).setdefault(ktom, {}).setdefault(jtom, {}).setdefault(itom, dihe)
        #                 self._dihedralList.append(dihe)
                        
        #         for ltom in filterfalse(lambda atom: atom in (ktom, jtom), itom.bondedAtoms):
                    
        #             try:
        #                 dihe = self._dihedrals[ltom][itom][jtom][ktom]
        #                 dihe = self._dihedrals[ktom][jtom][itom][ltom]
        #             except KeyError:
        #                 dihe = Dihedral(ltom, itom, jtom, ktom)
        #                 self._dihedrals.setdefault(ltom, {}).setdefault(itom, {}).setdefault(jtom, {}).setdefault(ktom, dihe)
        #                 self._dihedrals.setdefault(ktom, {}).setdefault(jtom, {}).setdefault(itom, {}).setdefault(ltom, dihe)
        #                 self._dihedralList.append(dihe)
        
        # return self._dihedralList
        """search all the dihedrals in this group

        Returns:
            List[Dihedral]: all dihedrals in this group
        """
        for jtom in self.getAtoms():
            
            if len(jtom.bondedAtoms) < 2:
                continue
            
            for ktom in jtom.bondedAtoms:
                
                for itom in jtom.bondedAtoms:
                    
                    if itom == ktom:
                        continue
                    
                    for ltom in ktom.bondedAtoms:
                        
                        if ltom == jtom:
                            continue
                        
                        if itom != ltom:
                            try:
                                dihe = self._dihedrals[itom][jtom][ktom][ltom]
                                dihe = self._dihedrals[ltom][ktom][jtom][itom]
                            except KeyError:
                                dihe = Dihedral(itom, jtom, ktom, ltom)
                                self._dihedrals.setdefault(itom, {}).setdefault(jtom, {}).setdefault(ktom, {}).setdefault(ltom, dihe)
                                self._dihedrals.setdefault(ltom, {}).setdefault(ktom, {}).setdefault(jtom, {}).setdefault(itom, dihe)
                                self._dihedralList.append(dihe)
        return self._dihedralList
                           
    
    def addBondByIndex(self, atomIdx, atomJdx, **bondType):
        """Add a bond refer to the order of atoms. NOTE that the order of atoms depends on the order they are added. 

        Args:
            atomIdx (int): index of list
            atomJdx (int): index of list
        """
        atoms = self.getAtoms()
        atom1 = atoms[atomIdx]
        atom2 = atoms[atomJdx]
        self.addBond(atom1, atom2, **bondType)
        
    def addBondByName(self, name1, name2, **bondType):
        """Add a bond refer to the name of atoms. NOTE that the name of atoms temporarily must be unique, or can not be retrieved correctly.

        Args:
            name1 ([type]): [description]
            name2 ([type]): [description]
        """
        atom = self.getAtomByName(name1)
        btom = self.getAtomByName(name2)
        self.addBond(atom, btom, **bondType)
    
    def getBondByIndex(self, atomIdx, atomJdx):
        """get a bond by its atom'index

        Args:
            atomIdx (int): atom index
            atomJdx (int): btom index

        Returns:
            Bond: result bond
        """
        atoms = self.getAtoms()
        atom1 = atoms[atomIdx]
        atom2 = atoms[atomJdx]
        try:
            assert atom1.bonds[atom2] == atom1.bondto(atom2)
            return atom1.bonds[atom2]
        except KeyError:
            return None
    
    def __contains__(self, n):
        try:
            if isinstance(n, Atom):
                return n in self.getAtoms()
            elif isinstance(n, Bond):
                return n in self.getBonds()
        except TypeError:
            return False
        
    def __len__(self):
        return len(self.getAtoms())

    def getBasisCycles(self, root=None):
         
        """Returns a list of cycles which form a basis for cycles of G.
        A basis for cycles of a network is a minimal collection of
        cycles such that any cycle in the network can be written
        as a sum of cycles in the basis.  Here summation of cycles
        is defined as "exclusive or" of the edges. Cycle bases are
        useful, e.g. when deriving equations for electric circuits
        using Kirchhoff's Laws.
        Parameters
        
        Args: 
            root(Atom): Specify starting node for basis, optional
            
        Returns:
            A list of cycle lists.  Each cycle list is a list of nodes
            which forms a cycle (loop) in G.
            
        Examples:
            >>> G = nx.Graph()
            >>> nx.add_cycle(G, [0, 1, 2, 3])
            >>> nx.add_cycle(G, [0, 3, 4, 5])
            >>> print(nx.cycle_basis(G, 0))
            [[3, 4, 5, 0], [1, 2, 3, 0]]
        Notes:
            -----
            This is adapted from algorithm CACM 491 [1]_.
            References
            ----------
            .. [1] Paton, K. An algorithm for finding a fundamental set of
            cycles of a graph. Comm. ACM 12, 9 (Sept 1969), 514-518.
            See Also
            --------
            simple_cycles
        """        
        
        gnodes = set(self.getAtoms())
        cycles = []
        while gnodes:
            if root is None:
                root = gnodes.pop()
            stack = [root]
            pred = {root: root}
            used = {root: set()}
            while stack:  # walk the spanning tree finding cycles
                z = stack.pop()  # use last-in so cycles easier to find
                zused = used[z]
                for nbr in z.bondedAtoms:
                    if nbr not in used:  # new node
                        pred[nbr] = z
                        stack.append(nbr)
                        used[nbr] = {z}
                    elif nbr == z:  # self loops
                        cycles.append([z])
                    elif nbr not in zused:  # found a cycle
                        pn = used[nbr]
                        cycle = [nbr, z]
                        p = pred[z]
                        while p not in pn:
                            cycle.append(p)
                            p = pred[p]
                        cycle.append(p)
                        cycles.append(cycle)
                        used[nbr].add(z)
            gnodes -= set(pred)
            root = None
        return cycles

    @property
    def degree(self, ):
        tmp = {}
        for atom in self.atoms:
            tmp[atom] = self.neighbors(atom)
        return tmp
    
    def copy(self, name: str=None):
        """Return a new group. Both its atoms and its properties are copied.

        Returns:
            [type]: [description]
        """
        if name is None:
            name = self.name
        g = Group(name)
        g.update(self._attr)
        g.addAtoms(self.atoms, copy=True)
        for bond in self.bonds:
            atom, btom = bond
            g.addBondByName(atom.name, btom.name, **bond._attr)
        return g
        
    def bondto(self, group, atom, btom, mode=Literal['a', 'c']):
        if not isinstance(group, Group):
            raise TypeError('')
        
    def _set_per_atom(self, values: Iterable, method=None):
        atoms = self.atoms
        assert len(atoms) == len(values), ValueError(f'position array must match the number of atoms, but {len(values)} != {len(atoms)}')
        
        if method is not None:
            for atom, value in zip(atoms, values):
                getattr(atom, method)(value)            
        
    def setPositions(self, positions):

        self._set_per_atom(positions, 'setPosition')
            
    def getPositions(self):
        atoms = self.atoms
        R = np.empty((len(atoms), 3))
        for i, atom in enumerate(atoms):
            R[i] = atom.getPosition()
            
        return R
        
    def setAtomTypes(self, atomTypes: Iterable):
        
        self._set_per_atom(atomTypes, 'setAtomType')

            
    def getAtomTypes(self):
        atoms = self.atoms
        at = []
        for atom in atoms:
            at.append(atom.getAtomType())
            
        return at
    
    def getElements(self):
        atoms = self.atoms
        eles = []
        for atom in atoms:
            eles.append(atom.element)
        return eles
    
    def reacto(self, group, method:Literal['addition', 'concentration'], atom=None, btom=None, atomName=None, btomName=None, **attr):
        if atomName is not None:
            atom = self.getAtomByName(atomName)
        if atomName is not None:
            btom = group.getAtomByName(btomName)
        if method == 'addition':
            bond = atom.bondto(btom, **attr)    
        elif method == 'concentration':
            assert atom.nbondedAtoms == 1, ValueError(f'{atom} has {atom.nbondedAtoms} bondedAtoms leading to confusion about how to establish bondage')
            assert btom.nbondedAtoms == 1, ValueError(f'{btom} has {btom.nbondedAtoms} bondedAtoms leading to confusion about how to establish bondage')
            atom1 = atom.bondedAtoms[0]
            btom1 = btom.bondedAtoms[0]
            self.removeAtom(atom)
            group.removeAtom(btom)
            bond = atom1.bondto(btom1, **attr)
        if atom not in self._bonds:
            self._bonds[atom] = {}
        if btom not in group._bonds:
            group._bonds[btom] = {}
        self._bonds[atom][btom] = bond
        group._bonds[btom][atom] = bond
        self._bondList.append(bond)
        group._bondList.append(bond)  
    
    def merge(self, name, group, copy=False):
        """ return a new group that merge multiple groups

        Args:
            group (Group): groups to be merged
            copy (bool, optional): Defaults to False.
        """
        newGroup = Group(name)
        newGroup.addAtoms(group.atoms)
        newGroup.addbonds(*group.bonds)
        
        return newGroup
    
    def __call__(self, **kwargs):
        tmp = self.copy()
        tmp.update(kwargs)
        return tmp
    
    def move(self, *args):
        return self
    
    def rot(self, *args):
        return self