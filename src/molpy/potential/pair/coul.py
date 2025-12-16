"""
Coulomb pair potential with cutoff.
"""

import numpy as np
from numpy.typing import NDArray

from .base import PairPotential


class CoulCut(PairPotential):
    """
    Coulomb pair potential with cutoff.

    The potential is defined as:
    V(r) = q_i * q_j / r

    The force is:
    F(r) = q_i * q_j / r^3 * dr
    """

    name = "coul/cut"
    type = "pair"

    def calc_energy(
        self,
        r: NDArray[np.floating],
        pair_idx: NDArray[np.integer],
        charges: NDArray[np.floating],
    ) -> float:
        """
        Calculate Coulomb energy.

        Args:
            r: Atom coordinates (shape: (n_atoms, 3))
            pair_idx: Pair indices (shape: (n_pairs, 2))
            charges: Atom charges (shape: (n_atoms,))

        Returns:
            Total Coulomb energy
        """
        # Calculate distances
        dr = r[pair_idx[:, 1]] - r[pair_idx[:, 0]]
        dr_norm = np.linalg.norm(dr, axis=1)

        # Calculate energy
        energy = charges[pair_idx[:, 0]] * charges[pair_idx[:, 1]] / dr_norm

        return float(np.sum(energy))

    def calc_forces(
        self,
        r: NDArray[np.floating],
        pair_idx: NDArray[np.integer],
        charges: NDArray[np.floating],
    ) -> NDArray[np.floating]:
        """
        Calculate Coulomb forces.

        Args:
            r: Atom coordinates (shape: (n_atoms, 3))
            pair_idx: Pair indices (shape: (n_pairs, 2))
            charges: Atom charges (shape: (n_atoms,))

        Returns:
            Array of forces on each atom (shape: (n_atoms, 3))
        """
        n_atoms = len(r)

        # Calculate distances
        dr = r[pair_idx[:, 1]] - r[pair_idx[:, 0]]
        dr_norm = np.linalg.norm(dr, axis=1, keepdims=True)

        # Calculate forces
        forces = charges[pair_idx[:, 0]] * charges[pair_idx[:, 1]] / dr_norm**3 * dr

        # Accumulate forces on atoms
        per_atom_forces = np.zeros((n_atoms, 3), dtype=np.float64)
        np.add.at(per_atom_forces, pair_idx[:, 0], -forces)
        np.add.at(per_atom_forces, pair_idx[:, 1], forces)

        return per_atom_forces


class CoulLong(PairPotential):
    """
    Coulomb long-range pair potential with TypeIndexedArray support.

    This version uses the same interface as LJ126 for composition.
    """

    name = "coul/long"
    type = "pair"

    def __init__(
        self,
        charges: dict[str, float] | float | NDArray[np.float64],
    ) -> None:
        """
        Initialize CoulLong potential.

        Args:
            charges: Atomic charges, can be dict (type->value), scalar, or array
        """
        from molpy.potential.utils import TypeIndexedArray

        if isinstance(charges, dict):
            self.charges = TypeIndexedArray(charges)
        else:
            self.charges = np.array(charges, dtype=np.float64).reshape(-1)

    def calc_energy(
        self,
        dr: NDArray[np.floating],
        dr_norm: NDArray[np.floating],
        pair_types_i: NDArray,
        pair_types_j: NDArray,
    ) -> float:
        """
        Calculate Coulomb energy.

        Args:
            dr: Pair displacement vectors (shape: (n_pairs, 3))
            dr_norm: Pair distances (shape: (n_pairs,))
            pair_types_i: Atom types for first atom in each pair
            pair_types_j: Atom types for second atom in each pair

        Returns:
            Total Coulomb energy
        """
        if len(pair_types_i) == 0:
            return 0.0

        # Get charges by type
        q_i = self.charges[pair_types_i]
        q_j = self.charges[pair_types_j]

        # Ensure dr_norm is 1D
        if dr_norm.ndim > 1:
            dr_norm = dr_norm.squeeze()

        # Calculate energy
        energy = q_i * q_j / dr_norm

        return float(np.sum(energy))

    def calc_forces(
        self,
        dr: NDArray[np.floating],
        dr_norm: NDArray[np.floating],
        pair_types_i: NDArray,
        pair_types_j: NDArray,
        pair_idx: NDArray[np.integer],
        n_atoms: int,
    ) -> NDArray[np.floating]:
        """
        Calculate Coulomb forces.

        Args:
            dr: Pair displacement vectors (shape: (n_pairs, 3))
            dr_norm: Pair distances (shape: (n_pairs,))
            pair_types_i: Atom types for first atom in each pair
            pair_types_j: Atom types for second atom in each pair
            pair_idx: Pair indices (shape: (n_pairs, 2))
            n_atoms: Number of atoms

        Returns:
            Array of forces on each atom (shape: (n_atoms, 3))
        """
        if len(pair_types_i) == 0:
            return np.zeros((n_atoms, 3), dtype=np.float64)

        # Get charges by type
        q_i = self.charges[pair_types_i]
        q_j = self.charges[pair_types_j]

        # Ensure dr_norm has correct shape
        if dr_norm.ndim == 1:
            dr_norm = dr_norm[:, None]

        # Calculate force magnitude
        force_mag = (q_i * q_j / (dr_norm.squeeze() ** 3))[:, None]

        # Calculate force vectors
        forces = force_mag * dr

        # Accumulate forces on atoms
        per_atom_forces = np.zeros((n_atoms, 3), dtype=np.float64)
        np.add.at(per_atom_forces, pair_idx[:, 0], -forces)
        np.add.at(per_atom_forces, pair_idx[:, 1], forces)

        return per_atom_forces
