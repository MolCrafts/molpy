# author: Roy Kid
# contact: lijichen365@126.com
# date: 2022-07-03
# version: 0.0.1

from itertools import combinations
from typing import Dict, Iterable, List, Optional, Tuple, Any
import numpy as np
from itertools import combinations

class Graph:

    pass

class PyGraph(Graph):

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
            >>> G.add_edge(0, 1)
            >>> G.add_edge(0, 2)
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
            >>> G.add_edge(0, 1)
            >>> G.add_edge(0, 2)
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

    def add_edge(self, u, v):
        """
        Add an edge between u and v.

        Args:
            u (int): The first node.
            v (int): The second node.
        
        Examples:
            >>> G = Graph()
            >>> G.add_nodes_from([0, 1, 2, 3])
            >>> G.add_edge(0, 1)
            >>> G.add_edge(0, 2)
            >>> G.edges()
            [(0, 1), (0, 2)]
        """
        _adj = self._adj
        self.add_nodes([u, v])
        _adj[u][v] = {}
        _adj[v][u] = {}

    def add_edges(self, edges:Iterable[Tuple[int, int]]):
        for u, v in edges:
            self.add_edge(u, v)

    def del_edge(self, u, v):
        """
        Remove an edge between u and v.

        Args:
            u (int): The first node.
            v (int): The second node.
        
        Examples:
            >>> G = Graph()
            >>> G.add_nodes_from([0, 1, 2, 3])
            >>> G.add_edge(0, 1)
            >>> G.add_edge(0, 2)
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
        
    def update(self, G):
        """
        Update the graph with another graph.

        Args:
            G (Graph): The graph to update with.
        
        Examples:
            >>> G = Graph()
            >>> G.add_nodes_from([0, 1, 2, 3])
            >>> G.add_edge(0, 1)
            >>> G.add_edge(0, 2)
            >>> G.update(G)
            >>> G.edges()
            [(0, 1), (0, 2)]
        """
        self.add_edges(G.edges)

    @property
    def edges(self):
        """
        Return a list of all edges.

        Return:
            list: A list of all edges.
        
        Examples:
            >>> G = Graph()
            >>> G.add_nodes_from([0, 1, 2, 3])
            >>> G.add_edge(0, 1)
            >>> G.add_edge(0, 2)
            >>> G.edges()
            [(0, 1), (0, 2)]
        """
        _adj = self._adj
        tmp = []
        for i, js in _adj.items():
            for j in js:
                tmp.append([i, j])
        bonds = np.array(tmp)
        bonds = np.where((bonds[:, 0]>bonds[:, 1]).reshape((-1, 1)), bonds[:, ::-1], bonds)
        bonds = np.unique(bonds, axis=0)
        return bonds

    @property
    def n_edges(self, ):
        """
        Return the number of edges.

        Return:
            int: The number of edges.
        
        Examples:
            >>> G = Graph()
            >>> G.add_nodes_from([0, 1, 2, 3])
            >>> G.add_edge(0, 1)
            >>> G.add_edge(0, 2)
            >>> G.size()
            2
        """
        return len(self.edges)

    def has_edge(self, u, v):
        """
        Return True if there is an edge between u and v.

        Args:
            u (int): The first node.
            v (int): The second node.
        
        Examples:
            >>> G = Graph()
            >>> G.add_nodes_from([0, 1, 2, 3])
            >>> G.add_edge(0, 1)
            >>> G.add_edge(0, 2)
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
            >>> G.add_edge(0, 1)
            >>> G.add_edge(0, 2)
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
            >>> G.add_edge(0, 1)
            >>> G.add_edge(0, 2)
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
                    subgraph.add_edge(induced_node, neighbor)
        return subgraph

class Topology:

    def __init__(self, ):

        # self._graph = PyGraph()
        self._bonds:Dict[int, Dict[int, int]] = dict()
        self._angles:Dict[int, Dict[int, Dict[int, int]]] = dict()
        self._dihedrals:Dict[int, Dict[int, Dict[int, Dict[int, int]]]] = dict()

    @property
    def n_atoms(self):
        return len(self._bonds)

    @property
    def n_bonds(self):
        n_bonds = 0
        for k, v in self._bonds.items():
                n_bonds += len(v)
        return int(n_bonds / 2)

    @property
    def bonds(self):
        return self.calc_bonds()

    def add_atom(self, atom_id):
        # self._graph.add_node(atom_id)
        pass

    def del_atom(self, atom_id):

        bond_ids = []
        bonds = self._bonds
        if atom_id in bonds:
            nbrs = bonds.pop(atom_id)
            for nbr in nbrs:
                bond_id = bonds[nbr].pop(atom_id)
                bond_ids.append(bond_id)

        return bond_ids                    

    def add_bond(self, i, j, bond_id):

        if i not in self._bonds:
            self._bonds[i] = {}
        if j not in self._bonds:
            self._bonds[j] = {}

        self._bonds[i][j] = bond_id
        self._bonds[j][i] = bond_id
        
        # update graph
        # self._graph.add_edge(i, j)

    def add_bonds(self, bonds:Iterable[Tuple[int, int]], bond_ids:Iterable[int]):

        n_bonds = len(bonds)
        for i in range(n_bonds):
            self.add_bond(bonds[i][0], bonds[i][1], bond_ids[i])

            # update graph
            # self._graph.add_edges(bonds)

    def del_bond(self, i, j):
        """
        delete a bond in topology

        Parameters
        ----------
        i : int
            _description_
        j : int
            _description_

        Returns
        -------
        bond index
            bond index point to bond object in frame's bond list

        Raises
        ------
        KeyError
            _description_
        """

        _bond = self._bonds
        bond_id = _bond[i][j]
        if i not in _bond or j not in _bond:
            raise KeyError('Node not in graph.')
        else:
            del _bond[i][j]
            del _bond[j][i]

        return bond_id

    def get_bond(self, i, j):
        
        if i not in self._bonds or j not in self._bonds:
            raise KeyError('bond not exist.')
        
        return self._bonds[i][j]

    def calc_bonds(self):
        if len(self._bonds) == 0:
            return np.array([])
        adj = self._bonds
        tmp = []
        for i, js in adj.items():
            for j in js:
                tmp.append([i, j])
        bonds = np.array(tmp)
        bonds = np.where((bonds[:, 0]>bonds[:, 1]).reshape((-1, 1)), bonds[:, ::-1], bonds)
        bonds = np.unique(bonds, axis=0)
        return bonds

    # def calc_angles(self):
    #     adj = self._graph._adj
    #     tmp = []
    #     for c, ps in adj.items():
    #         if len(ps) < 2:
    #             continue
    #         for (i, j) in combinations(ps, 2):
    #             tmp.append([i, c, j])
            
    #     angles = np.array(tmp)
    #     angles = np.where((angles[:,0]>angles[:,2]).reshape((-1, 1)), angles[:, ::-1], angles)
    #     angles = np.unique(angles, axis=0)
    #     return angles        

    # def calc_dihedrals(self):

    #     topo = self._graph._adj
    #     rawDihes = []
    #     for jtom, ps in topo.items():
    #         if len(ps) < 2:
    #             continue
    #         for (itom, ktom) in combinations(ps, 2):
                
    #             for ltom in topo[itom]:
    #                 if ltom != jtom:
    #                     rawDihes.append([ltom, itom, jtom, ktom])
    #             for ltom in topo[ktom]:
    #                 if ltom != jtom:
    #                     rawDihes.append([itom, jtom, ktom, ltom])
        
    #     # remove duplicates
    #     dihedrals = np.array(rawDihes)
    #     dihedrals = np.where((dihedrals[:,1]>dihedrals[:,2]).reshape((-1, 1)), dihedrals[:, ::-1], dihedrals)
    #     dihedrals = np.unique(dihedrals, axis=0)
    #     return dihedrals

# class Topo:
#     """
#     Topo class stores the topology of a molecule, and the state of the topology information(id of bonds, angles, dihedrals etc.).
#     """
#     def __init__(self):

#         self._g = Graph()
#         self._bonds:Dict[int, Dict[int, int]] = dict()
#         self._angles:Dict[int, Dict[int, Dict[int, int]]] = dict()
#         self._dihedrals:Dict[int, Dict[int, Dict[int, Dict[int, int]]]] = dict()

#     # ---= data load interface =---
#     def add_atoms(self, ids:List[int]):

#         self._g.add_nodes(ids)

#     def add_bonds(self, connects:ArrayLike[N, 2], indices:Optional[ArrayLike[N]]=None):
        
#         if indices is None:
#             for i in range(len(connects)):
#                 self.add_bond(*connects[i], None)
#         else:
#             for i in range(len(connects)):
#                 self.add_bond(*connects[i], indices[i])

#     def add_angles(self, angles, indices=None):

#         for i in range(len(angles)):
#             self.add_angle(*angles[i], indices[i])

#     def add_dihedrals(self, dihedrals, indices=None):

#         for i in range(len(dihedrals)):
#             self.add_dihedral(*dihedrals[i], indices[i])

#     def add_bond(self, i:int, j:int, index:Optional[int]=None):
#         """
#         add a bond to the graph. 

#         Args:
#             i (int): one atom in a bond
#             j (int): another atom in a bond
#             index (int): bond attribute index match to attrib of atoms
#         """
#         self._g.add_bond(i, j)
        
#         if i not in self._bonds:
#             self._bonds[i] = dict()
#         if j not in self._bonds:
#             self._bonds[j] = dict()

#         self._bonds[i][j] = index
#         self._bonds[j][i] = index

#     def add_angle(self, i, j, k, index):

#         self._g.add_bonds([(i, j), (j, k)])

#         if i not in self._angles:
#             self._angles[i] = dict()
#         if j not in self._angles[i]:
#             self._angles[i][j] = dict()
#         if k not in self._angles:
#             self._angles[k] = dict()
#         if j not in self._angles[k]:
#             self._angles[k][j] = dict()
#         self._angles[i][j][k] = index
#         self._angles[k][j][i] = index

#     def add_dihedral(self, i, j, k, l, index):

#         self._g.add_bonds([(i, j), (j, k), (k, l)])

#         if i not in self._dihedrals:
#             self._dihedrals[i] = dict()
#         if j not in self._dihedrals[i]:
#             self._dihedrals[i][j] = dict()
#         if k not in self._dihedrals[i][j]:
#             self._dihedrals[i][j][k] = dict()

#         if l not in self._dihedrals:
#             self._dihedrals[l] = dict()
#         if j not in self._dihedrals[l]:
#             self._dihedrals[l][j] = dict()
#         if k not in self._dihedrals[l][j]:
#             self._dihedrals[l][j][k] = dict()

#         self._dihedrals[i][j][k][l] = index
#         self._dihedrals[l][j][k][i] = index

#     def get_edge(self, i, j):

#         return self._bonds[i][j]

#     def get_angle(self, i, j, k):

#         return self._angles[i][j][k]

#     def get_dihedral(self, i, j, k, l):

#         return self._dihedrals[i][j][k][l]

#     def calc_bonds(self):

#         return self._g.calc_bonds()

#     def calc_angles(self):

#         return self._g.calc_angles()

#     def calc_dihedrals(self):

#         return self._g.calc_dihedrals()
