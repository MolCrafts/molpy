# author: Roy Kid
# contact: lijichen365@126.com
# date: 2023-01-10
# version: 0.0.1

import numpy as np
from igraph import Graph
from molpy import Alias

class Item(dict):
    pass

class Atom(Item):

    def __init__(self, name:str, **props):
        super().__init__(**props)
        self._name = name

    @property
    def name(self):
        return self._name

class Bond(Item):
    pass

class Angle(Item):
    pass


class Topology:

    def __init__(self, n_atoms=0, n_bonds=None, graph_attrs=None, vertex_attrs=None, edge_attrs=None):

        self._graph = Graph(
            n=n_atoms,
            edges=n_bonds,
            directed=False,
            graph_attrs=graph_attrs,
            vertex_attrs=vertex_attrs,
            edge_attrs=edge_attrs,
        )

    @property
    def graph(self):
        return self._graph

    def add_atom(self, name:str, **props):

        self._graph.add_vertex(name=name, **props)
        atom = Atom(name=name, **props)
        return atom
    
    def add_atoms(self, n_atoms:int, **props):
        self._graph.add_vertices(n_atoms, **props)
    
    def delete_atom(self, index:int|list[int]):
        self._graph.delete_vertrices(index)

    def add_bond(self, idx_i:int, idx_j:int, **props):
        self._graph.add_edge(idx_i, idx_j, **props)
        bond = Bond(**props)
        return bond
    
    def delete_bond(self, index:None|int|list[int]|list[tuple[int]]):
        self._graph.delete_edges(index)

    def add_bonds(self, bond_idx:list[tuple[int, int]], **props):
        self._graph.add_edges(bond_idx, **props)

    def add_angle(self, idx_i:int, idx_j:int, idx_k:int, **props):
        
        self.add_bond(idx_i, idx_j)
        self.add_bond(idx_j, idx_k)
        angle = Angle(**props)
        return angle
        
    def add_angles(self, angle_idx:list[tuple[int, int, int]]):
        self.add_bonds(angle_idx[:, :2])
        self.add_bonds(angle_idx[:, 1:])
