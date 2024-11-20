# author: Roy Kid
# contact: lijichen365@126.com
# date: 2023-01-10
# version: 0.0.1
import numpy as np
from igraph import Graph


class Topology(Graph):

    @property
    def n_atoms(self):
        return self.vcount()

    @property
    def n_bonds(self):
        return self.ecount()

    @property
    def n_angles(self):
        return int(
            self.count_subisomorphisms_vf2(Graph(3, [[0, 1], [1, 2]])) / 2
        )

    @property
    def n_dihedrals(self):
        return int(
            self.count_subisomorphisms_vf2(Graph(4, [[0, 1], [1, 2], [2, 3]]))
            / 2
        )

    @property
    def atoms(self):
        return np.array([v.index for v in self.vs])

    @property
    def bonds(self):
        return np.array(self.get_edgelist())

    @property
    def angles(self):
        duplicated_angles = np.array(
            self.get_subisomorphisms_vf2(Graph(3, [[0, 1], [1, 2]]))
        )
        mask = duplicated_angles[:, 0] < duplicated_angles[:, 2]
        return duplicated_angles[mask]

    @property
    def dihedrals(self):
        duplicated_dihedrals = np.array(
            self.get_subisomorphisms_vf2(Graph(4, [[0, 1], [1, 2], [2, 3]]))
        )
        mask = duplicated_dihedrals[:, 1] < duplicated_dihedrals[:, 2]
        return duplicated_dihedrals[mask]

    @property
    def improper(self):
        duplicated_impropers = np.array(
            self.get_subisomorphisms_vf2(Graph(4, [[0, 1], [0, 2], [0, 3]]))
        )
        impropers = np.sort(duplicated_impropers[:, 1:])
        return duplicated_impropers[np.unique(impropers, return_index=True, axis=0)[0]]

    def add_atom(self, name: str, **props):

        self.add_vertex(name, props)

    def add_atoms(self, n_atoms: int, **props):
        self.add_vertices(n_atoms, props)

    def delete_atom(self, index: int | list[int]):
        self.delete_vertrices(index)

    def add_bond(self, idx_i: int, idx_j: int, **props):
        if not self.are_adjacent(idx_i, idx_j):
            self.add_edge(idx_i, idx_j, **props)

    def delete_bond(self, index: None | int | list[int] | list[tuple[int]]):
        self.delete_edges(index)

    def add_bonds(self, bond_idx: list[tuple[int, int]], **props):
        self.add_edges(bond_idx, **props)

    def add_angle(self, idx_i: int, idx_j: int, idx_k: int, **props):

        if not self.are_adjacent(idx_i, idx_j):
            self.add_bond(idx_i, idx_j)
        if not self.are_adjacent(idx_j, idx_k):
            self.add_bond(idx_j, idx_k)

    def add_angles(self, angle_idx: list[tuple[int, int, int]]):
        angle_idx = np.array(angle_idx)
        self.add_bonds(angle_idx[:, :2])
        self.add_bonds(angle_idx[:, 1:])

    def union(self, other: "Topology"):
        self.union(other.graph)
        return self
