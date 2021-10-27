# author: Roy Kid
# contact: lijichen365@126.com
# date: 2021-10-17
# version: 0.0.1

from molpy.base import Graph
from molpy.atom import Atom
from molpy.bond import Bond
import numpy as np
from itertools import dropwhile, combinations

class Group(Graph):
    
    def __init__(self, name, **attr) -> None:
        super().__init__(name=name, **attr)
        self._atoms = []
        self._bonds = self._adj
            
    def add(self, item):
        if isinstance(item, Atom):
            self.addAtom(item)
        
    def addAtom(self, atom: Atom):
        super().add_node(atom)
        if atom not in self._atoms:
            self._atoms.append(atom)
                
    def addAtoms(self, atoms):
        super().add_nodes_from(atoms)
        self._atoms.extend(atoms)
            
    def removeAtom(self, atom: Atom):
        super().remove_node(atom)
        self._atoms.remove(atom)
        
    def removeAtoms(self, atoms):
        super().remove_nodes_from(atoms)
        for atom in atoms:
            self._atoms.remove(atom)
        
    @property
    def atoms(self):
        return self._atoms
    
    def getAtoms(self):
        return list(self._node.keys())

    @property
    def natoms(self):
        return len(self.atoms)
    
    def hasAtom(self, atom: Atom):
        return super().has_node(atom)
        
    def addBond(self, atom, btom, **attr):
        super().add_edge(atom, btom, **attr)
        bond = atom.bondto(btom)
        self._bonds[atom][btom]['_bond'] = bond
        self._bonds[atom][btom]['_bond'] = bond
        
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
            if u not in self._node:
                self._adj[u] = self.adjlist_inner_dict_factory()
                self._node[u] = self.node_attr_dict_factory()
            if v not in self._node:
                self._adj[v] = self.adjlist_inner_dict_factory()
                self._node[v] = self.node_attr_dict_factory()
            datadict = self._adj[u].get(v, self.edge_attr_dict_factory())
            datadict.update(attr)
            datadict.update(dd)
            bond = u.bondto(v, **datadict)
            datadict['_bond'] = bond
            self._adj[u][v] = datadict
            self._adj[v][u] = datadict
            
    def addBondsByDict(self, bondDict, ref):
        tmp = {atom.get(ref, None): atom for atom in self.atoms}
        for u, nbs in bondDict.items():
            for nb in nbs:
                self.addBond(tmp[u], tmp[nb])
            
    def removeBond(self, atom, btom):
        super().remove_edge(atom, btom)
        
    def removeBonds(self, atomList):
        super().remove_edges_from(atomList)
        
    def merge(self, atoms, bonds):
        edges = {}
        for bond in bonds:
            u, v = bond
            if u not in edges:
                edges[u] = {}
            if v not in edges:
                edges[v] = {}
            edges[u][v] = bond
            edges[v][u] = bond
        super().update(edges, atoms)

    def hasBond(self, atom, btom):
        return super().has_edge(atom, btom)
        
    def neighbors(self, atom):
        return super().neighbors(atom)
    
    @property
    def bonds(self):
        return self.getBonds()
    
    def getBond(self, atom, btom):
        return super().get_edge_data(atom, btom)
    
    def getBonds(self):
        bonds = set()
        print(self._bonds)
        for u, nbs in self._bonds.items():
            for nb in nbs:
                bonds.add(self._bonds[u][nb]['_bond'])
        return list(bonds)
    
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
        return atoms, covalentMap
    
    @property
    def covalentMap(self):
        return self._covalentMap
    
    @property
    def nbonds(self):
        return len(self.getBonds())
            
    def setTopoByCovalentMap(self, covalentMap: np.ndarray):

        atoms = self.getAtoms()
        for i, nbond in np.ndenumerate(covalentMap):
            if nbond == 1:
                atom1 = atoms[i[0]]
                atom2 = atoms[i[1]]
                self.addBond(atom1, atom2)
        
    @property
    def natoms(self):
        return len(self.getAtoms())
    
    def getAtomByName(self, atomName):
        for atom in self.atoms:
            if isinstance(atom, Atom) and atom.name == atomName:
                return atom
    
    def __getitem__(self, idx):
        if isinstance(idx, str):
            for atom in self.getAtoms():
                if atom.name == idx:
                    return atom
                
        elif isinstance(idx, int):
            return self.getAtoms()[idx]

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
