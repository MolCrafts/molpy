# author: Roy Kid
# contact: lijichen365@126.com
# date: 2023-01-10
# version: 0.0.1
import numpy as np
from igraph import Graph


class Topology:

    def __init__(
        self, n_atoms=0, n_bonds=None, graph_attrs={}, vertex_attrs={}, edge_attrs={}
    ):

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

    @property
    def n_atoms(self):
        return self._graph.vcount()

    @property
    def n_bonds(self):
        return self._graph.ecount()

    @property
    def n_angles(self):
        return int(
            self._graph.count_subisomorphisms_vf2(Graph(3, [[0, 1], [1, 2]])) / 2
        )

    @property
    def n_dihedrals(self):
        return int(
            self._graph.count_subisomorphisms_vf2(Graph(4, [[0, 1], [1, 2], [2, 3]]))
            / 2
        )

    @property
    def atoms(self):
        _atom_id = []
        for edge in self.bonds:
            _atom_id.extend(edge[0])
        return np.array(_atom_id)

    @property
    def bonds(self):
        return np.array(self._graph.get_edgelist())

    @property
    def angles(self):
        duplicated_angles = np.array(
            self._graph.get_subisomorphisms_vf2(Graph(3, [[0, 1], [1, 2]]))
        )
        mask = duplicated_angles[:, 0] < duplicated_angles[:, 2]
        return duplicated_angles[mask]

    @property
    def dihedrals(self):
        duplicated_dihedrals = np.array(
            self._graph.get_subisomorphisms_vf2(Graph(4, [[0, 1], [1, 2], [2, 3]]))
        )
        mask = duplicated_dihedrals[:, 1] < duplicated_dihedrals[:, 2]
        return duplicated_dihedrals[mask]

    @property
    def improper(self):
        duplicated_impropers = np.array(
            self._graph.get_subisomorphisms_vf2(Graph(4, [[0, 1], [0, 2], [0, 3]]))
        )
        impropers = np.sort(duplicated_impropers[:, 1:])
        return duplicated_impropers[np.unique(impropers, return_index=True, axis=0)[0]]

    def add_atom(self, name: str, **props):

        self._graph.add_vertex(name=name, **props)

    def add_atoms(self, n_atoms: int, **props):
        self._graph.add_vertices(n_atoms, **props)

    def delete_atom(self, index: int | list[int]):
        self._graph.delete_vertrices(index)

    def add_bond(self, idx_i: int, idx_j: int, **props):
        self._graph.add_edge(idx_i, idx_j, **props)

    def delete_bond(self, index: None | int | list[int] | list[tuple[int]]):
        self._graph.delete_edges(index)

    def add_bonds(self, bond_idx: list[tuple[int, int]], **props):
        self._graph.add_edges(bond_idx, **props)

    def get_bonds(self):
        return np.array(self._graph.get_edgelist())

    def add_angle(self, idx_i: int, idx_j: int, idx_k: int, **props):

        self.add_bond(idx_i, idx_j)
        self.add_bond(idx_j, idx_k)

    def add_angles(self, angle_idx: list[tuple[int, int, int]]):
        angle_idx = np.array(angle_idx)
        self.add_bonds(angle_idx[:, :2])
        self.add_bonds(angle_idx[:, 1:])

    def simplify(self):
        self._graph.simplify()
