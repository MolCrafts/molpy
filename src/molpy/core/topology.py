"""Molecular topology graph using igraph.

Provides graph-based representation of molecular connectivity with
automated detection of angles, dihedrals, and impropers.
"""
# author: Roy Kid
# contact: lijichen365@126.com
# date: 2023-01-10
# version: 0.0.1
from typing import Any

import numpy as np
from igraph import Graph
from numpy.typing import ArrayLike


class Topology(Graph):
    """Topology graph with bidirectional entity-to-index mapping.

    Attributes:
        entity_to_idx: Dictionary mapping Entity objects to their vertex indices
        idx_to_entity: List mapping vertex indices to Entity objects
    """

    def __init__(
        self,
        *args,
        entity_to_idx: dict[Any, int] | None = None,
        idx_to_entity: list[Any] | None = None,
        **kwargs,
    ):
        """Initialize Topology graph.

        Args:
            *args: Arguments passed to igraph.Graph.__init__
            entity_to_idx: Optional dictionary mapping entities to indices
            idx_to_entity: Optional list mapping indices to entities
            **kwargs: Keyword arguments passed to igraph.Graph.__init__
        """
        super().__init__(*args, **kwargs)
        # Initialize bidirectional mapping members
        self.entity_to_idx: dict[Any, int] = (
            entity_to_idx if entity_to_idx is not None else {}
        )
        self.idx_to_entity: list[Any] = (
            idx_to_entity if idx_to_entity is not None else []
        )

    @property
    def n_atoms(self):
        """Number of atoms (vertices)."""
        return self.vcount()

    @property
    def n_bonds(self):
        """Number of bonds (edges)."""
        return self.ecount()

    @property
    def n_angles(self):
        """Number of unique angles (i-j-k triplets)."""
        return int(self.count_subisomorphisms_vf2(Graph(3, [[0, 1], [1, 2]])) / 2)

    @property
    def n_dihedrals(self):
        """Number of unique proper dihedrals (i-j-k-l quartets)."""
        return int(
            self.count_subisomorphisms_vf2(Graph(4, [[0, 1], [1, 2], [2, 3]])) / 2
        )

    @property
    def atoms(self):
        """Array of atom indices."""
        return np.array([v.index for v in self.vs])

    @property
    def bonds(self):
        """Array of bond pairs (N×2)."""
        return np.array(self.get_edgelist())

    @property
    def angles(self):
        """Array of unique angle triplets (N×3), deduplicated."""
        angle_matches = self.get_subisomorphisms_vf2(Graph(3, [[0, 1], [1, 2]]))
        if not angle_matches:
            return np.array([]).reshape(0, 3)
        duplicated_angles = np.array(angle_matches)
        mask = duplicated_angles[:, 0] < duplicated_angles[:, 2]
        return duplicated_angles[mask]

    @property
    def dihedrals(self):
        """Array of unique proper dihedral quartets (N×4), deduplicated."""
        dihedral_matches = self.get_subisomorphisms_vf2(
            Graph(4, [[0, 1], [1, 2], [2, 3]])
        )
        if not dihedral_matches:
            return np.array([]).reshape(0, 4)
        duplicated_dihedrals = np.array(dihedral_matches)
        mask = duplicated_dihedrals[:, 1] < duplicated_dihedrals[:, 2]
        return duplicated_dihedrals[mask]

    @property
    def improper(self):
        """Array of unique improper dihedral quartets (N×4)."""
        improper_matches = self.get_subisomorphisms_vf2(
            Graph(4, [[0, 1], [0, 2], [0, 3]])
        )
        if not improper_matches:
            return np.array([]).reshape(0, 4)
        duplicated_impropers = np.array(improper_matches)
        impropers = np.sort(duplicated_impropers[:, 1:])
        return duplicated_impropers[np.unique(impropers, return_index=True, axis=0)[0]]

    def add_atom(self, name: str, **props):
        """Add a single atom vertex."""
        self.add_vertex(name, **props)

    def add_atoms(self, n_atoms: int, **props):
        """Add multiple atom vertices."""
        self.add_vertices(n_atoms, props)

    def delete_atom(self, index: int | list[int]):
        """Delete atom(s) by index."""
        self.delete_vertrices(index)

    def add_bond(self, idx_i: int, idx_j: int, **props):
        """Add bond between atoms i and j if not already connected."""
        if not self.are_adjacent(idx_i, idx_j):
            self.add_edge(idx_i, idx_j, **props)

    def delete_bond(self, index: None | int | list[int] | ArrayLike):
        """Delete bond(s) by index."""
        self.delete_edges(index)

    def add_bonds(self, bond_idx: ArrayLike, **props):
        """Add multiple bonds from array of pairs."""
        self.add_edges(bond_idx, **props)

    def add_angle(self, idx_i: int, idx_j: int, idx_k: int, **props):
        """Add angle by ensuring bonds i-j and j-k exist."""
        if not self.are_adjacent(idx_i, idx_j):
            self.add_bond(idx_i, idx_j)
        if not self.are_adjacent(idx_j, idx_k):
            self.add_bond(idx_j, idx_k)

    def add_angles(self, angle_idx: ArrayLike):
        """Add multiple angles from array of triplets."""
        angle_idx = np.array(angle_idx)
        self.add_bonds(angle_idx[:, :2])
        self.add_bonds(angle_idx[:, 1:])

    def union(self, other: "Topology"):
        """Merge another topology into this one."""
        self.union(other)
        return self
