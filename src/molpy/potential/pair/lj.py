"""
Lennard-Jones 12-6 pair potential and force field styles.
"""

import numpy as np
from numpy.typing import NDArray

from molpy.core.forcefield import AtomType, PairStyle, PairType

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


# ===================================================================
#               Force Field Style and Type Classes
# ===================================================================


class PairLJ126Type(PairType):
    """Lennard-Jones 12-6 pair type with epsilon and sigma parameters."""

    def __init__(
        self,
        name: str,
        itom: AtomType,
        jtom: AtomType | None = None,
        epsilon: float = 0.0,
        sigma: float = 0.0,
        charge: float = 0.0,
    ):
        """
        Args:
            name: Type name
            itom: First atom type
            jtom: Second atom type (None for self-interaction)
            epsilon: LJ epsilon parameter
            sigma: LJ sigma parameter
            charge: Atomic charge (optional)
        """
        if jtom is None:
            jtom = itom
        super().__init__(name, itom, jtom, epsilon=epsilon, sigma=sigma, charge=charge)


class PairLJ126Style(PairStyle):
    """Lennard-Jones 12-6 pair style with fixed name='lj126'."""

    def __init__(self, cutoff: float = 10.0):
        """
        Args:
            cutoff: Cutoff distance in Angstroms (default: 10.0)
        """
        super().__init__("lj126", cutoff)

    def def_type(
        self,
        itom: AtomType,
        jtom: AtomType | None = None,
        epsilon: float = 0.0,
        sigma: float = 0.0,
        charge: float = 0.0,
        name: str = "",
    ) -> PairLJ126Type:
        """Define LJ 12-6 pair type.

        Args:
            itom: First atom type
            jtom: Second atom type (None for self-interaction)
            epsilon: LJ epsilon parameter
            sigma: LJ sigma parameter
            charge: Atomic charge (optional)
            name: Optional name

        Returns:
            PairLJ126Type instance
        """
        if jtom is None:
            jtom = itom
        if not name:
            name = itom.name if itom == jtom else f"{itom.name}-{jtom.name}"
        pt = PairLJ126Type(name, itom, jtom, epsilon, sigma, charge)
        self.types.add(pt)
        return pt


class PairCoulLongStyle(PairStyle):
    """Coulomb long-range pair style with fixed name='coul/long'."""

    def __init__(self, cutoff: float = 10.0):
        """
        Args:
            cutoff: Cutoff distance in Angstroms (default: 10.0)
        """
        super().__init__("coul/long", cutoff)


class PairLJ126CoulLongStyle(PairStyle):
    """Combined LJ 12-6 and Coulomb long pair style.

    This is a composite style that combines PairLJ126Style and PairCoulLongStyle.
    The name is 'lj/cut/coul/long' for LAMMPS compatibility.
    """

    def __init__(
        self,
        lj_cutoff: float = 10.0,
        coul_cutoff: float = 10.0,
        coulomb14scale: float = 0.5,
        lj14scale: float = 0.5,
    ):
        """
        Args:
            lj_cutoff: LJ cutoff distance in Angstroms (default: 10.0)
            coul_cutoff: Coulomb cutoff distance in Angstroms (default: 10.0)
            coulomb14scale: 1-4 Coulomb scaling factor (default: 0.5)
            lj14scale: 1-4 LJ scaling factor (default: 0.5)
        """
        super().__init__("lj/cut/coul/long", lj_cutoff, coul_cutoff)
        self.params.kwargs["coulomb14scale"] = coulomb14scale
        self.params.kwargs["lj14scale"] = lj14scale

    def def_type(
        self,
        itom: AtomType,
        jtom: AtomType | None = None,
        epsilon: float = 0.0,
        sigma: float = 0.0,
        charge: float = 0.0,
        name: str = "",
    ) -> PairLJ126Type:
        """Define LJ 12-6 pair type (same as PairLJ126Style).

        Args:
            itom: First atom type
            jtom: Second atom type (None for self-interaction)
            epsilon: LJ epsilon parameter
            sigma: LJ sigma parameter
            charge: Atomic charge (optional)
            name: Optional name

        Returns:
            PairLJ126Type instance
        """
        if jtom is None:
            jtom = itom
        if not name:
            name = itom.name if itom == jtom else f"{itom.name}-{jtom.name}"
        pt = PairLJ126Type(name, itom, jtom, epsilon, sigma, charge)
        self.types.add(pt)
        return pt


class PairLJ126CoulCutStyle(PairStyle):
    """Combined LJ 12-6 and Coulomb cut pair style.

    This is a composite style that combines PairLJ126Style and Coulomb cut.
    The name is 'lj/cut/coul/cut' for LAMMPS compatibility.
    """

    def __init__(
        self,
        lj_cutoff: float = 10.0,
        coul_cutoff: float = 10.0,
        coulomb14scale: float = 0.5,
        lj14scale: float = 0.5,
    ):
        """
        Args:
            lj_cutoff: LJ cutoff distance in Angstroms (default: 10.0)
            coul_cutoff: Coulomb cutoff distance in Angstroms (default: 10.0)
            coulomb14scale: 1-4 Coulomb scaling factor (default: 0.5)
            lj14scale: 1-4 LJ scaling factor (default: 0.5)
        """
        super().__init__("lj/cut/coul/cut", lj_cutoff, coul_cutoff)
        self.params.kwargs["coulomb14scale"] = coulomb14scale
        self.params.kwargs["lj14scale"] = lj14scale

    def def_type(
        self,
        itom: AtomType,
        jtom: AtomType | None = None,
        epsilon: float = 0.0,
        sigma: float = 0.0,
        charge: float = 0.0,
        name: str = "",
    ) -> PairLJ126Type:
        """Define LJ 12-6 pair type (same as PairLJ126Style).

        Args:
            itom: First atom type
            jtom: Second atom type (None for self-interaction)
            epsilon: LJ epsilon parameter
            sigma: LJ sigma parameter
            charge: Atomic charge (optional)
            name: Optional name

        Returns:
            PairLJ126Type instance
        """
        if jtom is None:
            jtom = itom
        if not name:
            name = itom.name if itom == jtom else f"{itom.name}-{jtom.name}"
        pt = PairLJ126Type(name, itom, jtom, epsilon, sigma, charge)
        self.types.add(pt)
        return pt
