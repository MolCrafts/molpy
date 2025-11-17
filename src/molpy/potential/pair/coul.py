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
