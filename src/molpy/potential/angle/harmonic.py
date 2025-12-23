"""
Harmonic angle potential and force field styles.
"""

import numpy as np
from numpy.typing import NDArray

from molpy.core.forcefield import AngleStyle, AngleType, AtomType
from molpy.potential.utils import TypeIndexedArray

from .base import AnglePotential


class AngleHarmonic(AnglePotential):
    name = "harmonic"
    type = "angle"

    def __init__(
        self,
        k: NDArray[np.floating] | float | dict[str, float],
        theta0: NDArray[np.floating] | float | dict[str, float],
    ):
        """
        Initialize harmonic angle potential.

        Args:
            k: Force constant in kcal/mol/rad² (array, scalar, or dict mapping type names to values)
            theta0: Equilibrium angle in degrees (array, scalar, or dict mapping type names to values)
        """
        # TypeIndexedArray automatically handles both integer and string indexing
        self.k = TypeIndexedArray(k).reshape(-1, 1)
        self.theta0 = TypeIndexedArray(theta0).reshape(-1, 1)

    def calc_energy(
        self,
        r: NDArray[np.floating],
        angle_idx: NDArray[np.integer],
        angle_types: NDArray[np.integer] | NDArray[np.str_],
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

        # Calculate angle in radians
        cos_theta = np.sum(ua * ub, axis=1)
        cos_theta = np.clip(cos_theta, -1.0, 1.0)
        theta_rad = np.arccos(cos_theta)

        # theta0 is stored in degrees, convert to radians for calculation
        # TypeIndexedArray automatically handles both integer and string indexing
        theta0_rad = np.deg2rad(self.theta0[angle_types])

        # Calculate energy using LAMMPS formula: E = k * (theta_rad - theta0_rad)^2
        # k is in kcal/mol/rad²
        energy = self.k[angle_types] * (theta_rad - theta0_rad) ** 2

        return float(np.sum(energy))

    def calc_forces(
        self,
        r: NDArray[np.floating],
        angle_idx: NDArray[np.integer],
        angle_types: NDArray[np.integer] | NDArray[np.str_],
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

        # Calculate angle in radians
        cos_theta = np.sum(ua * ub, axis=1, keepdims=True)
        cos_theta = np.clip(cos_theta, -1.0, 1.0)
        theta_rad = np.arccos(cos_theta)

        # theta0 is stored in degrees, convert to radians for calculation
        # TypeIndexedArray automatically handles both integer and string indexing
        theta0_rad = np.deg2rad(self.theta0[angle_types])

        # Calculate force magnitude using LAMMPS formula
        # dE/dtheta_rad = 2 * k * (theta_rad - theta0_rad)
        # k is in kcal/mol/rad²
        # Use np.atleast_1d to ensure arrays stay 1D even with single angle
        k_vals = np.atleast_1d(self.k[angle_types].squeeze())
        theta0_rad_vals = np.atleast_1d(theta0_rad.squeeze())
        theta_rad_vals = np.atleast_1d(theta_rad.squeeze())
        dtheta = -2.0 * k_vals * (theta_rad_vals - theta0_rad_vals)

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


# ===================================================================
#               Force Field Style and Type Classes
# ===================================================================


class AngleHarmonicType(AngleType):
    """Harmonic angle type with k and theta0 parameters."""

    def __init__(
        self,
        name: str,
        itom: AtomType,
        jtom: AtomType,
        ktom: AtomType,
        k: float,
        theta0: float,
    ):
        """
        Args:
            name: Type name
            itom: First atom type
            jtom: Central atom type
            ktom: Third atom type
            k: Force constant
            theta0: Equilibrium angle in degrees
        """
        super().__init__(name, itom, jtom, ktom, k=k, theta0=theta0)


class AngleHarmonicStyle(AngleStyle):
    """Harmonic angle style with fixed name='harmonic'."""

    def __init__(self):
        super().__init__("harmonic")

    def def_type(
        self,
        itom: AtomType,
        jtom: AtomType,
        ktom: AtomType,
        k: float,
        theta0: float,
        name: str = "",
    ) -> AngleHarmonicType:
        """Define harmonic angle type.

        Args:
            itom: First atom type
            jtom: Central atom type
            ktom: Third atom type
            k: Force constant
            theta0: Equilibrium angle in degrees
            name: Optional name (defaults to itom-jtom-ktom)

        Returns:
            AngleHarmonicType instance
        """
        if not name:
            name = f"{itom.name}-{jtom.name}-{ktom.name}"
        at = AngleHarmonicType(name, itom, jtom, ktom, k, theta0)
        self.types.add(at)
        return at
