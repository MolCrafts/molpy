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


class LJ126CoulLong(PairPotential):
    """
    Combined Lennard-Jones 12-6 and Coulomb long-range pair potential.
    
    This is a composite potential that combines LJ and Coulomb interactions.
    Uses PairTypeIndexedArray internally for automatic combining rules.
    
    V(r) = V_LJ(r) + V_Coul(r)
    """
    
    name = "lj/cut/coul/long"
    type = "pair"
    
    def __init__(
        self,
        epsilon: NDArray[np.float64],
        sigma: NDArray[np.float64],
        charges: NDArray[np.float64],
        type_names: list[str],
    ) -> None:
        """
        Initialize LJ126CoulLong potential.
        
        Args:
            epsilon: Per-atom-type epsilon values (numpy array)
            sigma: Per-atom-type sigma values (numpy array)
            charges: Per-atom-type charges (numpy array)
            type_names: List of atom type names corresponding to array indices
        """
        from molpy.potential.utils import TypeIndexedArray
        from molpy.potential.pair_params import PairTypeIndexedArray
        
        # Create dictionaries from arrays and type names
        epsilon_dict = {name: float(eps) for name, eps in zip(type_names, epsilon)}
        sigma_dict = {name: float(sig) for name, sig in zip(type_names, sigma)}
        charge_dict = {name: float(q) for name, q in zip(type_names, charges)}
        
        # Create TypeIndexedArrays with combining rules
        self.epsilon = PairTypeIndexedArray(epsilon_dict, combining_rule='geometric')
        self.sigma = PairTypeIndexedArray(sigma_dict, combining_rule='arithmetic')
        self.charges = TypeIndexedArray(charge_dict)
    
    def calc_energy(
        self,
        dr: NDArray[np.floating],
        dr_norm: NDArray[np.floating],
        pair_types_i: NDArray,
        pair_types_j: NDArray,
    ) -> float:
        """
        Calculate combined LJ + Coulomb energy.
        
        Uses PairTypeIndexedArray to automatically apply combining rules.
        """
        if len(pair_types_i) == 0:
            return 0.0
        
        # Ensure dr_norm has correct shape
        if dr_norm.ndim == 1:
            dr_norm = dr_norm[:, None]
        
        # Use PairTypeIndexedArray pair indexing (automatic combining rules)
        pair_types = np.column_stack([pair_types_i, pair_types_j])
        eps = self.epsilon[pair_types]
        sig = self.sigma[pair_types]
        
        # Ensure correct shape for broadcasting
        if isinstance(eps, np.ndarray) and eps.ndim == 1:
            eps = eps[:, None]
        if isinstance(sig, np.ndarray) and sig.ndim == 1:
            sig = sig[:, None]
        
        # Calculate LJ energy
        lj_energy = 4 * eps * ((sig / dr_norm) ** 12 - (sig / dr_norm) ** 6)
        
        # Calculate Coulomb energy
        q_i = self.charges[pair_types_i]
        q_j = self.charges[pair_types_j]
        coul_energy = (q_i * q_j / dr_norm.squeeze())
        
        if isinstance(coul_energy, np.ndarray) and coul_energy.ndim == 1:
            coul_energy = coul_energy[:, None]
        
        total_energy = lj_energy + coul_energy
        return float(np.sum(total_energy))
    
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
        Calculate combined LJ + Coulomb forces.
        
        Uses PairTypeIndexedArray to automatically apply combining rules.
        """
        if len(pair_types_i) == 0:
            return np.zeros((n_atoms, 3), dtype=np.float64)
        
        # Ensure dr_norm has correct shape
        if dr_norm.ndim == 1:
            dr_norm = dr_norm[:, None]
        
        # Use PairTypeIndexedArray pair indexing
        pair_types = np.column_stack([pair_types_i, pair_types_j])
        eps = self.epsilon[pair_types]
        sig = self.sigma[pair_types]
        
        # Ensure correct shape for broadcasting
        if isinstance(eps, np.ndarray) and eps.ndim == 1:
            eps = eps[:, None]
        if isinstance(sig, np.ndarray) and sig.ndim == 1:
            sig = sig[:, None]
        
        # Calculate LJ force magnitude
        lj_force_mag = (
            24 * eps * (2 * (sig / dr_norm) ** 12 - (sig / dr_norm) ** 6) / (dr_norm**2)
        )
        
        # Calculate Coulomb force magnitude
        q_i = self.charges[pair_types_i]
        q_j = self.charges[pair_types_j]
        coul_force_mag = (q_i * q_j / (dr_norm.squeeze() ** 3))
        
        if isinstance(coul_force_mag, np.ndarray) and coul_force_mag.ndim == 1:
            coul_force_mag = coul_force_mag[:, None]
        
        # Total force magnitude
        total_force_mag = lj_force_mag + coul_force_mag
        
        # Calculate force vectors
        forces = total_force_mag * dr
        
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
    
    def to_potential(self):
        """Convert this style to a Potential object."""
        from molpy.core.forcefield import PairType
        
        pair_types = list(self.types.bucket(PairType))
        if not pair_types:
            raise ValueError("No pair types defined in style")
        
        # Extract parameters as lists
        type_names = []
        epsilon_list = []
        sigma_list = []
        charge_list = []
        
        for pt in pair_types:
            epsilon = pt.params.kwargs.get("epsilon")
            sigma = pt.params.kwargs.get("sigma")
            charge = pt.params.kwargs.get("charge", 0.0)
            
            if epsilon is None or sigma is None:
                raise ValueError(
                    f"PairType '{pt.name}' is missing required parameters: "
                    f"epsilon={epsilon}, sigma={sigma}"
                )
            
            type_names.append(pt.itom.name)
            epsilon_list.append(epsilon)
            sigma_list.append(sigma)
            charge_list.append(charge)
        
        # Convert to numpy arrays
        epsilon_array = np.array(epsilon_list, dtype=np.float64)
        sigma_array = np.array(sigma_list, dtype=np.float64)
        charges_array = np.array(charge_list, dtype=np.float64)
        
        # Create Potential instance with numpy arrays
        return LJ126CoulLong(
            epsilon=epsilon_array,
            sigma=sigma_array,
            charges=charges_array,
            type_names=type_names,
        )


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
