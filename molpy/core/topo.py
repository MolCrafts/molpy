# author: Roy Kid
# contact: lijichen365@126.com
# date: 2022-07-03
# version: 0.0.1

from itertools import combinations
from typing import Dict, Iterable, List, Optional, Tuple
from molpy.utils.typing import ArrayLike
import numpy as np
from itertools import combinations

N = int # represent number of atoms/bonds/angles etc..

class AdjList:

    def __init__(self) -> None:

        self._adj = {}

    def __iter__(self):
        """
        Iterate over the nodes. Use: 'for n in G'.

        Return:
            niter (iterator): An iterator over all nodes in the graph.
        
        Examples:
            >>> [n for n in G]
            [0, 1, 2, 3]
            >>> list(G)
            [0, 1, 2, 3]
        """
        return iter(self._adj)

    def __contains__(self, n):
        """
        Return True if n is a node in the graph. Use: 'n in G'.

        Return:
            bool: True if n is a node in the graph.
        
        Examples:
            >>> G = Graph()
            >>> G.add_nodes_from([0, 1, 2, 3])
            >>> 0 in G
            True
            >>> 4 in G
            False
        """
        return n in self._adj

    def __len__(self):
        """
        Return the number of nodes in the graph. Use: 'len(G)'.

        Return:
            int: The number of nodes in the graph.
        
        Examples:
            >>> G = Graph()
            >>> G.add_nodes_from([0, 1, 2, 3])
            >>> len(G)
            4
        """
        return len(self._adj)

    def __getitem__(self, k):
        """
        Returns a dict of neighbors of node n.  Use: 'G[n]'.

        Args:
            k (int): A node in the graph.
        Return:
            dict: The neighbors of node n.
        Examples:
            >>> G = Graph()
            >>> G.add_nodes_from([0, 1, 2, 3])
            >>> G.add_bond(0, 1)
            >>> G.add_bond(0, 2)
            >>> G[0]
            {1, 2}
        """
        return self._adj[k]


    def add_node(self, node: int):
        """
        Add a single node to the graph.

        Args:
            node (int): The node to add.
        
        Examples:
            >>> G = Graph()
            >>> G.add_node(0)
            >>> G.nodes()
            [0]
        """
        if node not in self._adj:
            self._adj[node] = {}

    def add_nodes(self, nodes:Iterable[int]):
        
        for node in nodes:
            self.add_node(node)

    def del_node(self, n):
        """
        Remove a single node from the graph.

        Args:
            n (int): The node to remove.
        
        Examples:
            >>> G = Graph()
            >>> G.add_nodes_from([0, 1, 2, 3])
            >>> G.add_bond(0, 1)
            >>> G.add_bond(0, 2)
            >>> G.del_node(0)
            >>> G.nodes()
            [1, 2, 3]
        """
        _adj = self._adj
        for k in _adj[n]:
            del _adj[k][n]
        del _adj[n]

    def del_nodes(self, nodes):
        for node in nodes:
            self.del_node(node)

    def nodes(self):
        return self._adj.keys()

    @property
    def n_nodes(self):
        return len(self._adj)

    @property
    def order(self):
        return len(self._adj)

    def had_node(self, n):
        return n in self._adj

    def add_bond(self, u, v):
        """
        Add an edge between u and v.

        Args:
            u (int): The first node.
            v (int): The second node.
        
        Examples:
            >>> G = Graph()
            >>> G.add_nodes_from([0, 1, 2, 3])
            >>> G.add_bond(0, 1)
            >>> G.add_bond(0, 2)
            >>> G.edges()
            [(0, 1), (0, 2)]
        """
        _adj = self._adj
        self.add_nodes([u, v])
        _adj[u][v] = {}
        _adj[v][u] = {}

    def add_bonds(self, edges:Iterable[Iterable[int]]):
        for u, v in edges:
            self.add_bond(u, v)

    def del_edge(self, u, v):
        """
        Remove an edge between u and v.

        Args:
            u (int): The first node.
            v (int): The second node.
        
        Examples:
            >>> G = Graph()
            >>> G.add_nodes_from([0, 1, 2, 3])
            >>> G.add_bond(0, 1)
            >>> G.add_bond(0, 2)
            >>> G.del_edge(0, 1)
            >>> G.edges()
            [(0, 2)]
        """
        _adj = self._adj
        if u not in _adj or v not in _adj:
            raise KeyError('Node not in graph.')
        else:
            del _adj[u][v]
            if u != v:
                del _adj[v][u]
        
    def update(self, G:'AdjList'):
        """
        Update the graph with another graph.

        Args:
            G (Graph): The graph to update with.
        
        Examples:
            >>> G = Graph()
            >>> G.add_nodes_from([0, 1, 2, 3])
            >>> G.add_bond(0, 1)
            >>> G.add_bond(0, 2)
            >>> G.update(G)
            >>> G.edges()
            [(0, 1), (0, 2)]
        """
        self.add_bonds(G.edges)

    @property
    def edges(self):
        """
        Return a list of all edges.

        Return:
            list: A list of all edges.
        
        Examples:
            >>> G = Graph()
            >>> G.add_nodes_from([0, 1, 2, 3])
            >>> G.add_bond(0, 1)
            >>> G.add_bond(0, 2)
            >>> G.edges()
            [(0, 1), (0, 2)]
        """
        _adj = self._adj
        return [(u, v) for u in _adj for v in _adj[u]]

    @property
    def size(self, ):
        """
        Return the number of edges.

        Return:
            int: The number of edges.
        
        Examples:
            >>> G = Graph()
            >>> G.add_nodes_from([0, 1, 2, 3])
            >>> G.add_bond(0, 1)
            >>> G.add_bond(0, 2)
            >>> G.size()
            2
        """
        return len(self.edges())

    def number_of_edges(self, u=None, v=None):
        if u is None:
            return int(self.size())
        if v in self._adj[u]:
            return 1
        return 0

    def has_edge(self, u, v):
        """
        Return True if there is an edge between u and v.

        Args:
            u (int): The first node.
            v (int): The second node.
        
        Examples:
            >>> G = Graph()
            >>> G.add_nodes_from([0, 1, 2, 3])
            >>> G.add_bond(0, 1)
            >>> G.add_bond(0, 2)
            >>> G.has_edge(0, 1)
            True
            >>> G.has_edge(0, 3)
            False
        """
        _adj = self._adj
        return u in _adj and v in _adj[u]

    def get_neighbors_of(self, n):
        """
        Return a list of neighbors of node n.

        Args:
            n (int): The node.
        
        Examples:
            >>> G = Graph()
            >>> G.add_nodes_from([0, 1, 2, 3])
            >>> G.add_bond(0, 1)
            >>> G.add_bond(0, 2)
            >>> G.get_neighbors_of(0)
            [1, 2]
        """
        return self._adj[n].keys()

    @property
    def degree(self)->Dict[int, int]:
        """
        Return the degree of each node.

        Return:
            dict: The degree of each node.
        
        Examples:
            >>> G = Graph()
            >>> G.add_nodes_from([0, 1, 2, 3])
            >>> G.add_bond(0, 1)
            >>> G.add_bond(0, 2)
            >>> G.degree()
            {0: 2, 1: 1, 2: 1, 3: 0}
        """
        _adj = self._adj
        return {n: len(_adj[n]) for n in _adj}

    def items(self):

        return self._adj.items()

    @property
    def adj(self):
        return self._adj

    def subgraph(self, nodes):

        induced_nodes = [node for node in nodes if node in self._adj]
        subgraph = Graph()
        for induced_node in induced_nodes:
            for neighbor in self._adj[induced_node]:
                if neighbor in induced_nodes:
                    subgraph.add_bond(induced_node, neighbor)
        return subgraph

class Graph(AdjList):
    """
    Graph class is the high-api of the AdjList class. The AdjList maybe rewritten by any language in order to get a better performance with the same api.
    """
    def __init__(self):
        super().__init__()
        self.globals = {}
        self.is_directed = False

    @property
    def name(self):
        return self.globals.get('name', '')

    @name.setter
    def name(self, s):
        self.globals['name'] = s

    def calc_bonds(self):
        adj = self._adj
        tmp = []
        for i, js in adj.items():
            for j in js:
                tmp.append([i, j])
        bonds = np.array(tmp)
        bonds = np.where((bonds[:, 0]>bonds[:, 1]).reshape((-1, 1)), bonds[:, ::-1], bonds)
        bonds = np.unique(bonds, axis=0)
        return bonds

    def calc_angles(self):
        adj = self._adj
        tmp = []
        for c, ps in adj.items():
            if len(ps) < 2:
                continue
            for (i, j) in combinations(ps, 2):
                tmp.append([i, c, j])
            
        angles = np.array(tmp)
        angles = np.where((angles[:,0]>angles[:,2]).reshape((-1, 1)), angles[:, ::-1], angles)
        angles = np.unique(angles, axis=0)
        return angles        

    def calc_dihedrals(self):

        topo = self._adj
        rawDihes = []
        for jtom, ps in topo.items():
            if len(ps) < 2:
                continue
            for (itom, ktom) in combinations(ps, 2):
                
                for ltom in topo[itom]:
                    if ltom != jtom:
                        rawDihes.append([ltom, itom, jtom, ktom])
                for ltom in topo[ktom]:
                    if ltom != jtom:
                        rawDihes.append([itom, jtom, ktom, ltom])
        
        # remove duplicates
        dihedrals = np.array(rawDihes)
        dihedrals = np.where((dihedrals[:,1]>dihedrals[:,2]).reshape((-1, 1)), dihedrals[:, ::-1], dihedrals)
        dihedrals = np.unique(dihedrals, axis=0)
        return dihedrals

    def from_networkx(self, G):

        self.add_bonds(G.edges)


class Topo:
    """
    Topo class stores the topology of a molecule, and the state of the topology information(id of bonds, angles, dihedrals etc.).
    """
    def __init__(self):

        self._g = Graph()
        self._bonds:Dict[int, Dict[int, int]] = dict()
        self._angles:Dict[int, Dict[int, Dict[int, int]]] = dict()
        self._dihedrals:Dict[int, Dict[int, Dict[int, Dict[int, int]]]] = dict()

    # ---= data load interface =---
    def add_atoms(self, ids:List[int]):

        self._g.add_nodes(ids)

    def add_bonds(self, connects:ArrayLike[N, 2], indices:Optional[ArrayLike[N]]=None):
        
        if indices is None:
            for i in range(len(connects)):
                self.add_bond(*connects[i], None)
        else:
            for i in range(len(connects)):
                self.add_bond(*connects[i], indices[i])

    def add_angles(self, angles, indices=None):

        for i in range(len(angles)):
            self.add_angle(*angles[i], indices[i])

    def add_dihedrals(self, dihedrals, indices=None):

        for i in range(len(dihedrals)):
            self.add_dihedral(*dihedrals[i], indices[i])

    def add_bond(self, i:int, j:int, index:Optional[int]=None):
        """
        add a bond to the graph. 

        Args:
            i (int): one atom in a bond
            j (int): another atom in a bond
            index (int): bond attribute index match to attrib of atoms
        """
        self._g.add_bond(i, j)
        
        if i not in self._bonds:
            self._bonds[i] = dict()
        if j not in self._bonds:
            self._bonds[j] = dict()

        self._bonds[i][j] = index
        self._bonds[j][i] = index

    def add_angle(self, i, j, k, index):

        self._g.add_bonds([(i, j), (j, k)])

        if i not in self._angles:
            self._angles[i] = dict()
        if j not in self._angles[i]:
            self._angles[i][j] = dict()
        if k not in self._angles:
            self._angles[k] = dict()
        if j not in self._angles[k]:
            self._angles[k][j] = dict()
        self._angles[i][j][k] = index
        self._angles[k][j][i] = index

    def add_dihedral(self, i, j, k, l, index):

        self._g.add_bonds([(i, j), (j, k), (k, l)])

        if i not in self._dihedrals:
            self._dihedrals[i] = dict()
        if j not in self._dihedrals[i]:
            self._dihedrals[i][j] = dict()
        if k not in self._dihedrals[i][j]:
            self._dihedrals[i][j][k] = dict()

        if l not in self._dihedrals:
            self._dihedrals[l] = dict()
        if j not in self._dihedrals[l]:
            self._dihedrals[l][j] = dict()
        if k not in self._dihedrals[l][j]:
            self._dihedrals[l][j][k] = dict()

        self._dihedrals[i][j][k][l] = index
        self._dihedrals[l][j][k][i] = index

    def get_edge(self, i, j):

        return self._bonds[i][j]

    def get_angle(self, i, j, k):

        return self._angles[i][j][k]

    def get_dihedral(self, i, j, k, l):

        return self._dihedrals[i][j][k][l]

    def calc_bonds(self):

        return self._g.calc_bonds()

    def calc_angles(self):

        return self._g.calc_angles()

    def calc_dihedrals(self):

        return self._g.calc_dihedrals()
