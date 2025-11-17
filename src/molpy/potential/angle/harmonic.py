"""
Harmonic angle potential.
"""

import numpy as np
from numpy.typing import NDArray

from .base import AnglePotential


class Harmonic(AnglePotential):
    name = "harmonic"
    type = "angle"

    def __init__(
        self, k: NDArray[np.floating] | float, theta0: NDArray[np.floating] | float
    ):
        """
        Initialize harmonic angle potential.

        Args:
            k: Force constant (array for multiple types, or scalar)
            theta0: Equilibrium angle in degrees (array for multiple types, or scalar)
        """
        self.k = np.array(k, dtype=np.float64)
        self.theta0 = np.array(theta0, dtype=np.float64)

    def calc_energy(
        self,
        r: NDArray[np.floating],
        angle_idx: NDArray[np.integer],
        angle_types: NDArray[np.integer],
    ) -> float:
        """
        Calculate angle energy.

        Args:
            r: Atom coordinates (shape: (n_atoms, 3))
            angle_idx: Angle indices (shape: (n_angles, 3))
            angle_types: Angle types (shape: (n_angles,))

        Returns:
            Total angle energy
        """
        # Extract atom positions
        r1 = r[angle_idx[:, 0]]
        r2 = r[angle_idx[:, 1]]
        r3 = r[angle_idx[:, 2]]

        # Calculate vectors
        a = r2 - r1
        b = r2 - r3

        norm_a = np.linalg.norm(a, axis=-1, keepdims=True)
        norm_b = np.linalg.norm(b, axis=-1, keepdims=True)

        ua = a / norm_a
        ub = b / norm_b

        # Calculate angle (in radians)
        cos_theta = np.sum(ua * ub, axis=1)
        cos_theta = np.clip(cos_theta, -1.0, 1.0)
        theta_rad = np.arccos(cos_theta)

        # Convert theta0 to radians (assuming it's in degrees)
        theta0_rad = np.deg2rad(self.theta0[angle_types])

        # Calculate energy
        energy = 0.5 * self.k[angle_types] * (theta_rad - theta0_rad) ** 2

        return float(np.sum(energy))

    def calc_forces(
        self,
        r: NDArray[np.floating],
        angle_idx: NDArray[np.integer],
        angle_types: NDArray[np.integer],
    ) -> NDArray[np.floating]:
        """
        Calculate angle forces.

        Args:
            r: Atom coordinates (shape: (n_atoms, 3))
            angle_idx: Angle indices (shape: (n_angles, 3))
            angle_types: Angle types (shape: (n_angles,))

        Returns:
            Array of forces on each atom (shape: (n_atoms, 3))
        """
        n_atoms = len(r)

        # Extract atom positions
        r1 = r[angle_idx[:, 0]]
        r2 = r[angle_idx[:, 1]]
        r3 = r[angle_idx[:, 2]]

        # Calculate vectors
        a = r2 - r1
        b = r2 - r3
        c = np.cross(a, b)

        norm_a = np.linalg.norm(a, axis=-1, keepdims=True)
        norm_b = np.linalg.norm(b, axis=-1, keepdims=True)

        ua = a / norm_a
        ub = b / norm_b

        # Calculate angle (in radians)
        cos_theta = np.sum(ua * ub, axis=1, keepdims=True)
        cos_theta = np.clip(cos_theta, -1.0, 1.0)
        theta = np.arccos(cos_theta)

        # Convert theta0 to radians (assuming it's in degrees)
        theta0_rad = np.deg2rad(self.theta0[angle_types])

        # Calculate force magnitude
        dtheta = -self.k[angle_types] * (theta.squeeze() - theta0_rad)

        # Calculate force directions
        norm_c = np.linalg.norm(c, axis=-1, keepdims=True)
        # Avoid division by zero
        norm_c = np.where(norm_c < 1e-10, 1.0, norm_c)

        f = dtheta[:, None] / norm_c
        F1 = f * (np.cross(c, a) / norm_a**2)
        F3 = f * (np.cross(b, c) / norm_b**2)
        F2 = -(F1 + F3)

        # Accumulate forces on atoms
        per_atom_forces = np.zeros((n_atoms, 3), dtype=np.float64)
        np.add.at(per_atom_forces, angle_idx[:, 0], F1)
        np.add.at(per_atom_forces, angle_idx[:, 1], F2)
        np.add.at(per_atom_forces, angle_idx[:, 2], F3)

        return per_atom_forces
