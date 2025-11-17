"""
Lennard-Jones 12-6 pair potential.
"""

import numpy as np
from numpy.typing import NDArray

from .base import PairPotential


class LJ126(PairPotential):
    """
    Lennard-Jones 12-6 pair potential with cutoff.

    The potential is defined as:
    V(r) = 4 * ε * ((σ/r)^12 - (σ/r)^6)

    The force is:
    F(r) = 24 * ε * (2 * (σ/r)^12 - (σ/r)^6) * dr / r^2

    Attributes:
        epsilon: Depth of potential well for each atom type [energy]
        sigma: Finite distance at which potential is zero [length]
    """

    name = "lj126/cut"
    type = "pair"

    def __init__(
        self,
        epsilon: float | NDArray[np.float64],
        sigma: float | NDArray[np.float64],
    ) -> None:
        """
        Initialize LJ126 potential.

        Args:
            epsilon: Depth of potential well, can be scalar or array for multiple types
            sigma: Finite distance at which potential is zero, can be scalar or array for multiple types
        """
        self.epsilon = np.array(epsilon, dtype=np.float64).reshape(-1, 1)
        self.sigma = np.array(sigma, dtype=np.float64).reshape(-1, 1)

        if self.epsilon.shape != self.sigma.shape:
            raise ValueError("epsilon and sigma must have the same shape")

    def calc_energy(
        self,
        dr: NDArray[np.floating],
        dr_norm: NDArray[np.floating],
        pair_types: NDArray[np.integer],
    ) -> float:
        """
        Calculate pair energy.

        Args:
            dr: Pair displacement vectors (shape: (n_pairs, 3))
            dr_norm: Pair distances (shape: (n_pairs, 1) or (n_pairs,))
            pair_types: Pair types (shape: (n_pairs,))

        Returns:
            Total pair energy
        """
        if len(pair_types) == 0:
            return 0.0

        if np.any(pair_types >= len(self.epsilon)) or np.any(pair_types < 0):
            raise ValueError(
                f"pair_types contains invalid indices. Must be in range [0, {len(self.epsilon)})"
            )

        # Ensure dr_norm has correct shape
        if dr_norm.ndim == 1:
            dr_norm = dr_norm[:, None]

        eps = self.epsilon[pair_types]
        sig = self.sigma[pair_types]

        # Calculate energy
        energy = 4 * eps * ((sig / dr_norm) ** 12 - (sig / dr_norm) ** 6)

        return float(np.sum(energy))

    def calc_forces(
        self,
        dr: NDArray[np.floating],
        dr_norm: NDArray[np.floating],
        pair_types: NDArray[np.integer],
        pair_idx: NDArray[np.integer],
        n_atoms: int,
    ) -> NDArray[np.floating]:
        """
        Calculate pair forces.

        Args:
            dr: Pair displacement vectors (shape: (n_pairs, 3))
            dr_norm: Pair distances (shape: (n_pairs, 1) or (n_pairs,))
            pair_types: Pair types (shape: (n_pairs,))
            pair_idx: Pair indices (shape: (n_pairs, 2))
            n_atoms: Number of atoms

        Returns:
            Array of forces on each atom (shape: (n_atoms, 3))
        """
        if len(pair_types) == 0:
            return np.zeros((n_atoms, 3), dtype=np.float64)

        if np.any(pair_types >= len(self.epsilon)) or np.any(pair_types < 0):
            raise ValueError(
                f"pair_types contains invalid indices. Must be in range [0, {len(self.epsilon)})"
            )

        # Ensure dr_norm has correct shape
        if dr_norm.ndim == 1:
            dr_norm = dr_norm[:, None]

        eps = self.epsilon[pair_types]
        sig = self.sigma[pair_types]

        # Calculate force magnitude
        force_mag = (
            24 * eps * (2 * (sig / dr_norm) ** 12 - (sig / dr_norm) ** 6) / (dr_norm**2)
        )

        # Calculate force vectors
        forces = force_mag * dr

        # Accumulate forces on atoms
        per_atom_forces = np.zeros((n_atoms, 3), dtype=np.float64)
        np.add.at(per_atom_forces, pair_idx[:, 0], -forces.squeeze())
        np.add.at(per_atom_forces, pair_idx[:, 1], forces.squeeze())

        return per_atom_forces
