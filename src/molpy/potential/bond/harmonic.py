"""
Harmonic bond potential and force field styles.
"""

import numpy as np
from numpy.typing import NDArray

from molpy.core.forcefield import AtomType, BondStyle, BondType
from molpy.potential.utils import TypeIndexedArray

from .base import BondPotential


class BondHarmonic(BondPotential):
    name = "harmonic"
    type = "bond"

    def __init__(
        self, 
        k: NDArray[np.floating] | float | dict[str, float], 
        r0: NDArray[np.floating] | float | dict[str, float]
    ):
        """
        Initialize harmonic bond potential.

        Args:
            k: Force constant (array for multiple types, scalar, or dict mapping type names to values)
            r0: Equilibrium bond length (array for multiple types, scalar, or dict mapping type names to values)
        """
        # TypeIndexedArray automatically handles both integer and string indexing
        self.k = TypeIndexedArray(k).reshape(-1, 1)
        self.r0 = TypeIndexedArray(r0).reshape(-1, 1)

    def calc_energy(
        self,
        r: NDArray[np.floating],
        bond_idx: NDArray[np.integer],
        bond_types: NDArray[np.integer] | NDArray[np.str_],
    ) -> float:
        """
        Calculate bond energy.

        Args:
            r: Atom coordinates (shape: (n_atoms, 3))
            bond_idx: Bond indices (shape: (n_bonds, 2))
            bond_types: Bond types (shape: (n_bonds,)) - can be integer indices or string type names

        Returns:
            Total bond energy
        """
        # Calculate bond vectors
        dr = r[bond_idx[:, 1]] - r[bond_idx[:, 0]]
        dr_norm = np.linalg.norm(dr, axis=1, keepdims=True)

        # TypeIndexedArray automatically handles both integer and string indexing
        energy = 0.5 * self.k[bond_types] * (dr_norm - self.r0[bond_types]) ** 2

        return float(np.sum(energy))

    def calc_forces(
        self,
        r: NDArray[np.floating],
        bond_idx: NDArray[np.integer],
        bond_types: NDArray[np.integer] | NDArray[np.str_],
    ) -> NDArray[np.floating]:
        """
        Calculate bond forces.

        Args:
            r: Atom coordinates (shape: (n_atoms, 3))
            bond_idx: Bond indices (shape: (n_bonds, 2))
            bond_types: Bond types (shape: (n_bonds,)) - can be integer indices or string type names

        Returns:
            Array of forces on each atom (shape: (n_atoms, 3))
        """
        n_atoms = len(r)

        # Calculate bond vectors
        dr = r[bond_idx[:, 1]] - r[bond_idx[:, 0]]
        dr_norm = np.linalg.norm(dr, axis=1, keepdims=True)

        # TypeIndexedArray handles both integer and string indexing automatically
        forces = -self.k[bond_types] * (dr_norm - self.r0[bond_types]) * dr / dr_norm

        # Accumulate forces on atoms
        per_atom_forces = np.zeros((n_atoms, 3), dtype=np.float64)
        np.add.at(per_atom_forces, bond_idx[:, 0], -forces.squeeze())
        np.add.at(per_atom_forces, bond_idx[:, 1], forces.squeeze())

        return per_atom_forces


# ===================================================================
#               Force Field Style and Type Classes
# ===================================================================


class BondHarmonicType(BondType):
    """Harmonic bond type with k and r0 parameters."""

    def __init__(
        self,
        name: str,
        itom: AtomType,
        jtom: AtomType,
        k: float,
        r0: float,
    ):
        """
        Args:
            name: Type name
            itom: First atom type
            jtom: Second atom type
            k: Force constant
            r0: Equilibrium bond length
        """
        super().__init__(name, itom, jtom, k=k, r0=r0)


class BondHarmonicStyle(BondStyle):
    """Harmonic bond style with fixed name='harmonic'."""

    def __init__(self):
        super().__init__("harmonic")

    def def_type(
        self,
        itom: AtomType,
        jtom: AtomType,
        k: float,
        r0: float,
        name: str = "",
    ) -> BondHarmonicType:
        """Define harmonic bond type.

        Args:
            itom: First atom type
            jtom: Second atom type
            k: Force constant
            r0: Equilibrium bond length
            name: Optional name (defaults to itom-jtom)

        Returns:
            BondHarmonicType instance
        """
        if not name:
            name = f"{itom.name}-{jtom.name}"
        bt = BondHarmonicType(name, itom, jtom, k, r0)
        self.types.add(bt)
        return bt
