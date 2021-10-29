# author: Roy Kid
# contact: lijichen365@126.com
# date: 2021-10-17
# version: 0.0.1

from pydantic import HashableError
from molpy.base import Graph
from molpy.atom import Atom
from molpy.bond import Bond
import numpy as np
from itertools import dropwhile, combinations

class Group(Graph):
    
    def __init__(self, name, group=None, **attr):
        super().__init__(name)
        self._atoms = [] # [Atom]
        self._bonds = {} # _bonds: {Atom: {Btom: Bond, Ctom: Bond}}
        self.update(attr)
        if isinstance(group, Group):
            pass
            
    def add(self, item, copy=False):
        if isinstance(item, Atom):
            self.addAtom(item, copy)
        
    def addAtom(self, atom: Atom, copy=False):
        if atom not in self._atoms:
            if copy:
                self._atoms.append(atom.copy())
            else:
                self._atoms.append(atom)
                
    def addAtoms(self, atoms, copy=False):
        for atom in atoms:
            self.addAtom(atom, copy)
            
    def removeAtom(self, atom: Atom):
        
        self._atoms.remove(atom)
        
        bonds = self._bonds
        try:
            nbrs = list(bonds[atom])  # list handles self-loops (allows mutation)
        except KeyError as err:  # NetworkXError if n not in self
            raise KeyError(f"{atom} is not in the group.") from err
        for u in nbrs:
            del bonds[u][atom]  # remove all edges atom-u in graph
        del bonds[atom]
        
    def removeAtoms(self, atoms):
        for atom in atoms:
            self._atoms.remove(atom)
            self.removeAtom(atom)
        
    @property
    def atoms(self):
        return self._atoms
    
    def getAtoms(self):
        return self._atoms

    @property
    def natoms(self):
        return len(self._atoms)
    
    def hasAtom(self, atom: Atom):
        return atom in self._atoms
        
    def addBond(self, atom, btom, **attr):
        
        bond = atom.bondto(btom, **attr)
        if atom not in self._bonds:
            self._bonds[atom] = {}

        if btom not in self._bonds:
            self._bonds[btom] = {}
        self._bonds[atom][btom] = bond
        self._bonds[btom][atom] = bond
        
        
    def addBonds(self, atomList, **attr):
        for e in atomList:
            ne = len(e)
            if ne == 3:
                u, v, dd = e
            elif ne == 2:
                u, v = e
                dd = {}  # doesn't need edge_attr_dict_factory
            else:
                raise ValueError(f"bond tuple {e} must be a 2-tuple or 3-tuple.")

            dd.update(attr)
            self.addBond(u, v, **dd)
            
    def addBondsByDict(self, bondDict, ref):
        # tmp = {atom.get(ref, None): atom for atom in self.atoms}
        # for u, nbs in bondDict.items():
        #     for nb in nbs:
        #         self.addBond(tmp[u], tmp[nb])
        pass
            
    def removeBond(self, atom, btom):
        try:
            del self._bonds[atom][btom]
            if atom != btom:  # self-loop needs only one entry removed
                del self._bonds[btom][atom]
        except KeyError as err:
            raise KeyError(f"The bond {atom}-{btom} is not in the graph") from err
        
    def removeBonds(self, atomList):
        for atoms in atomList:
            atom, btom = atoms[:2]
            if atom in self._bonds and btom in self._bonds[atom]:
                del self._bonds[atom][btom]
                if atom != btom:
                    del self._bonds[btom][atom]
        
    # def merge(self, atoms, bonds):
    #     edges = {}
    #     for bond in bonds:
    #         u, v = bond
    #         if u not in edges:
    #             edges[u] = {}
    #         if v not in edges:
    #             edges[v] = {}
    #         edges[u][v] = bond
    #         edges[v][u] = bond
    #     super().update(edges, atoms)
    
    def splite(self):
        #TODO: split graph to subgraph
        pass
    
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
    
    def merge(self):
        #TODO: merge subgraph to graph
        pass    

    def hasBond(self, atom, btom):
        if self._bonds[atom].get(btom, False):
            return True
        else:
            return False
        
    def neighbors(self, atom):
        return list(self._bonds[atom].keys())
    
    @property
    def bonds(self):
        return self.getBonds()
    
    @property
    def nbonds(self):
        return len(self.getBonds())
    
    def getBond(self, atom, btom):
        return self._bonds[atom].get(btom, False)
    
    def getBonds(self):
        bonds = set()
        for u, nbs in self._bonds.items():
            for nb in nbs:
                bonds.add(self._bonds[u][nb])
        return list(bonds)
    
    def getCovalentMap(self, max_distance=None): 
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
        return atoms, covalentMap
    
    @property
    def covalentMap(self):
        return self._covalentMap
            
    def setTopoByCovalentMap(self, covalentMap: np.ndarray):

        atoms = self.getAtoms()
        for i, nbond in np.ndenumerate(covalentMap):
            if nbond == 1:
                atom1 = atoms[i[0]]
                atom2 = atoms[i[1]]
                self.addBond(atom1, atom2)
        
    def getAtomByName(self, atomName):
        for atom in self._atoms:
            if atom.name == atomName:
                return atom
            
    def getAtomBy(self, by, value):
        for atom in self._atoms:
            if atom.get(by) == value:
                return atom
    
    def __getitem__(self, idx):
        if isinstance(idx, str):
            for atom in self.getAtoms():
                if atom.name == idx:
                    return atom
                
        elif isinstance(idx, int):
            return self.getAtoms()[idx]

    def getAngles(self):
        #TODO: 
        angles = set()
        for atom in self.getAtoms():

            for edge in combinations(atom.bondedAtoms, 2):
                angles.add(
                    (edge[0], atom, edge[1])
                )
                
        self._angles = list(angles)
        return self._angles
    
    def addBondByIndex(self, atomIdx, atomJdx, **bondType):
        atoms = self.getAtoms()
        atom1 = atoms[atomIdx]
        atom2 = atoms[atomJdx]
        self.addBond(atom1, atom2, **bondType)
    
    def getBondByIndex(self, atomIdx, atomJdx):
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
    
    def nbunch_iter(self, nbunch=None):
        """Returns an iterator over nodes contained in nbunch that are
        also in the graph.
        The nodes in nbunch are checked for membership in the graph
        and if not are silently ignored.

        Args:
            nbunch (Atom, Iterable[Atom], optional): The view will only report edges incident to these nodes. Defaults to None.

        Raises:
            KeyError: WHEN nbunch is not a node or a sequence of nodes.
            HashableError: WHEN a node in nbunch is not hashable.
            
        Yields:
            iterator: An iterator over nodes in nbunch that are also in the graph.
            If nbunch is None, iterate over all nodes in the graph.
        """
        
        if nbunch is None:  # include all nodes via iterator
            bunch = iter(self._bonds)
        elif nbunch in self:  # if nbunch is a single node
            bunch = iter([nbunch])
        else:  # if nbunch is a sequence of nodes

            def bunch_iter(nlist, adj):
                try:
                    for n in nlist:
                        if n in adj:
                            yield n
                except TypeError as err:
                    exc, message = err, err.args[0]
                    # capture error for non-sequence/iterator nbunch.
                    if "iter" in message:
                        exc = KeyError(
                            "nbunch is not a node or a sequence of nodes."
                        )
                    # capture error for unhashable node.
                    if "hashable" in message:
                        exc = HashableError(
                            f"Node {n} in sequence nbunch is not a valid node."
                        )
                    raise exc

            bunch = bunch_iter(nbunch, self._adj)
        return bunch

    def copy(self):
        g = Group(self.name)
        g.update(self._attr)
        g.addAtoms(self.atoms, copy=True)
        for bond in self.bonds:
            g.addBond(*bond, **bond._attr)
        return g
        