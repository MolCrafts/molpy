"""Utility functions and classes for potentials."""

from typing import TYPE_CHECKING

import numpy as np
from numpy.typing import NDArray

from molpy.core.frame import Frame

if TYPE_CHECKING:
    from molpy.potential.base import Potential


class TypeIndexedArray:
    """Array-like container that supports both integer and string type name indexing.
    
    This class allows potentials to accept either integer indices or string type labels
    for indexing parameters. It maintains an internal mapping between type names and indices.
    
    Examples:
        >>> # Create from dictionary (type name -> value)
        >>> k = TypeIndexedArray({"CT-CT": 100.0, "CT-OH": 80.0})
        >>> k[0]  # Access by integer index
        100.0
        >>> k["CT-CT"]  # Access by type name
        100.0
        >>> k[np.array([0, 1])]  # Array indexing with integers
        array([100.,  80.])
        >>> k[np.array(["CT-CT", "CT-OH"])]  # Array indexing with strings
        array([100.,  80.])
        
        >>> # Create from array (for backward compatibility)
        >>> k = TypeIndexedArray(np.array([100.0, 80.0]))
        >>> k[0]
        100.0
    """
    
    def __init__(self, data: dict[str, float] | NDArray[np.floating] | float):
        """
        Initialize TypeIndexedArray.
        
        Args:
            data: Either a dictionary mapping type names to values, or an array/float
                 for backward compatibility
        """
        if isinstance(data, dict):
            # Dictionary mode: maintain type name -> index mapping
            self._type_names = list(data.keys())
            self._type_to_idx = {name: idx for idx, name in enumerate(self._type_names)}
            self._values = np.array([data[name] for name in self._type_names], dtype=np.float64)
            self._use_labels = True
        else:
            # Array mode: traditional integer indexing only
            self._values = np.array(data, dtype=np.float64)
            if self._values.ndim == 0:
                self._values = self._values.reshape(1)
            self._type_names = None
            self._type_to_idx = None
            self._use_labels = False
    
    def __getitem__(self, key):
        """Index into the array using either integer indices or type names.
        
        Args:
            key: Integer index, string type name, or array of either
            
        Returns:
            Single value or numpy array of values
        """
        if isinstance(key, (int, np.integer)):
            return self._values[key]
        elif isinstance(key, str):
            if not self._use_labels:
                raise ValueError("Cannot use string indexing: array was created without type labels")
            return self._values[self._type_to_idx[key]]
        elif isinstance(key, np.ndarray):
            # Handle array indexing
            if key.dtype.kind in ('i', 'u'):  # Integer array
                return self._values[key]
            elif key.dtype.kind in ('U', 'S', 'O'):  # String array
                if not self._use_labels:
                    raise ValueError("Cannot use string indexing: array was created without type labels")
                # Convert string type names to indices
                indices = np.array([self._type_to_idx[str(t)] for t in key], dtype=np.int64)
                return self._values[indices]
            else:
                raise ValueError(f"Unsupported array dtype: {key.dtype}")
        else:
            raise TypeError(f"Unsupported index type: {type(key)}")
    
    def __len__(self):
        """Return the number of types."""
        return len(self._values)
    
    @property
    def values(self) -> NDArray[np.floating]:
        """Get the underlying values array."""
        return self._values
    
    @property
    def type_names(self) -> list[str] | None:
        """Get the list of type names (if available)."""
        return self._type_names
    
    def reshape(self, *args, **kwargs):
        """Reshape the values array (for backward compatibility)."""
        self._values = self._values.reshape(*args, **kwargs)
        return self
    
    def __array__(self, dtype=None):
        """Allow conversion to numpy array."""
        return np.asarray(self._values, dtype=dtype)
    
    def __array_ufunc__(self, ufunc, method, *inputs, **kwargs):
        """Support numpy ufuncs (like multiplication, subtraction, etc.)."""
        # Convert inputs to numpy arrays
        converted_inputs = []
        for inp in inputs:
            if isinstance(inp, TypeIndexedArray):
                converted_inputs.append(inp._values)
            else:
                converted_inputs.append(inp)
        
        # Call the ufunc
        result = getattr(ufunc, method)(*converted_inputs, **kwargs)
        
        # Return result directly (don't wrap it back in TypeIndexedArray)
        return result


def extract_bond_data(frame: Frame) -> tuple[NDArray, NDArray, NDArray]:
    """Extract bond data from Frame.
    
    Returns:
        (r, bond_idx, bond_types) where:
        - r: atom coordinates (n_atoms, 3)
        - bond_idx: bond indices (n_bonds, 2)
        - bond_types: bond types (n_bonds,) - can be integers or strings
    """
    # Get coordinates from x, y, z fields (never use xyz)
    x = frame["atoms"]["x"]
    y = frame["atoms"]["y"]
    z = frame["atoms"]["z"]
    r = np.column_stack([x, y, z])
    bonds = frame["bonds"]
    bond_idx = bonds[["atom_i", "atom_j"]]
    bond_types = bonds["type"]
    return r, bond_idx, bond_types


def extract_angle_data(frame: Frame) -> tuple[NDArray, NDArray, NDArray]:
    """Extract angle data from Frame.
    
    Returns:
        (r, angle_idx, angle_types) where:
        - r: atom coordinates (n_atoms, 3)
        - angle_idx: angle indices (n_angles, 3)
        - angle_types: angle types (n_angles,) - can be integers or strings
    """
    # Get coordinates from x, y, z fields (never use xyz)
    x = frame["atoms"]["x"]
    y = frame["atoms"]["y"]
    z = frame["atoms"]["z"]
    r = np.column_stack([x, y, z])
    angles = frame["angles"]
    angle_idx = angles[["atom_i", "atom_j", "atom_k"]]
    angle_types = angles["type"]
    return r, angle_idx, angle_types


def extract_pair_data(frame: Frame):
    """Extract pair interaction data from Frame.
    
    Generates all pairwise interactions between atoms and calculates
    displacement vectors and distances.
    
    Returns:
        (dr, dr_norm, pair_types_i, pair_types_j, pair_idx, n_atoms) where:
        - dr: displacement vectors (n_pairs, 3)
        - dr_norm: pair distances (n_pairs,)
        - pair_types_i: atom types for first atom in each pair (n_pairs,)
        - pair_types_j: atom types for second atom in each pair (n_pairs,)
        - pair_idx: pair indices (n_pairs, 2)
        - n_atoms: number of atoms
    """
    # Get coordinates from x, y, z fields
    x = frame["atoms"]["x"]
    y = frame["atoms"]["y"]
    z = frame["atoms"]["z"]
    r = np.column_stack([x, y, z])
    n_atoms = len(r)
    
    # Get atom types
    atom_types = frame["atoms"]["type"]
    
    # Generate all pairs (i, j) where i < j
    # This avoids double counting and self-interactions
    pair_idx = []
    pair_types_i_list = []
    pair_types_j_list = []
    
    for i in range(n_atoms):
        for j in range(i + 1, n_atoms):
            pair_idx.append([i, j])
            # Store both atom types for proper parameter lookup
            pair_types_i_list.append(atom_types[i])
            pair_types_j_list.append(atom_types[j])
    
    pair_idx = np.array(pair_idx, dtype=np.int64)
    # Keep pair_types as-is (can be strings or integers)
    pair_types_i = np.array(pair_types_i_list)
    pair_types_j = np.array(pair_types_j_list)
    
    # Calculate displacement vectors
    if len(pair_idx) > 0:
        dr = r[pair_idx[:, 1]] - r[pair_idx[:, 0]]
        dr_norm = np.linalg.norm(dr, axis=1)
    else:
        dr = np.zeros((0, 3), dtype=np.float64)
        dr_norm = np.zeros(0, dtype=np.float64)
    
    return dr, dr_norm, pair_types_i, pair_types_j, pair_idx, n_atoms


def extract_coul_data(frame: Frame):
    """Extract Coulomb interaction data from Frame."""
    # Implementation depends on Coulomb potential type
    raise NotImplementedError("Coulomb data extraction not yet implemented")


def calc_energy_from_frame(potential: "Potential", frame: Frame) -> float:
    """
    Calculate energy from Frame for a potential.

    This is a convenience function that extracts the necessary data from Frame
    and calls the potential's calc_energy method.

    Args:
        potential: Potential instance
        frame: Frame containing the necessary data

    Returns:
        Potential energy

    Raises:
        TypeError: If potential type is not recognized
        ValueError: If required data is missing from frame
    """
    # Check potential type by looking at its type attribute
    potential_type = getattr(potential, "type", None)

    if potential_type == "bond":
        r, bond_idx, bond_types = extract_bond_data(frame)
        return potential.calc_energy(r, bond_idx, bond_types)
    elif potential_type == "angle":
        r, angle_idx, angle_types = extract_angle_data(frame)
        return potential.calc_energy(r, angle_idx, angle_types)
    elif potential_type == "pair":
        # Check if it's LJ126CoulLong (combined) or separate LJ/Coul
        potential_name = getattr(potential, "name", "")
        if "lj" in potential_name.lower() and "coul" in potential_name.lower():
            # Combined LJ + Coulomb potential (like lj/cut/coul/long)
            dr, dr_norm, pair_types_i, pair_types_j, _, _ = extract_pair_data(frame)
            return potential.calc_energy(dr, dr_norm, pair_types_i, pair_types_j)
        elif "lj" in potential_name.lower():
            # Pure LJ potential
            dr, dr_norm, pair_types_i, pair_types_j, _, _ = extract_pair_data(frame)
            return potential.calc_energy(dr, dr_norm, pair_types_i, pair_types_j)
        elif "coul" in potential_name.lower():
            r, pair_idx, charges = extract_coul_data(frame)
            return potential.calc_energy(r, pair_idx, charges)
        else:
            raise TypeError(f"Unknown pair potential type: {potential_name}")
    else:
        raise TypeError(f"Unknown potential type: {potential_type}")


def calc_forces_from_frame(potential: "Potential", frame: Frame) -> NDArray:
    """
    Calculate forces from Frame for a potential.

    This is a convenience function that extracts the necessary data from Frame
    and calls the potential's calc_forces method.

    Args:
        potential: Potential instance
        frame: Frame containing the necessary data

    Returns:
        Array of forces on each atom (shape: (n_atoms, 3))

    Raises:
        TypeError: If potential type is not recognized
        ValueError: If required data is missing from frame
    """
    # Check potential type by looking at its type attribute
    potential_type = getattr(potential, "type", None)

    if potential_type == "bond":
        r, bond_idx, bond_types = extract_bond_data(frame)
        return potential.calc_forces(r, bond_idx, bond_types)
    elif potential_type == "angle":
        r, angle_idx, angle_types = extract_angle_data(frame)
        return potential.calc_forces(r, angle_idx, angle_types)
    elif potential_type == "pair":
        # Check if it's LJ126CoulLong (combined) or separate LJ/Coul
        potential_name = getattr(potential, "name", "")
        if "lj" in potential_name.lower() and "coul" in potential_name.lower():
            # Combined LJ + Coulomb potential (like lj/cut/coul/long)
            dr, dr_norm, pair_types_i, pair_types_j, pair_idx, n_atoms = extract_pair_data(frame)
            return potential.calc_forces(dr, dr_norm, pair_types_i, pair_types_j, pair_idx, n_atoms)
        elif "lj" in potential_name.lower():
            # Pure LJ potential
            dr, dr_norm, pair_types_i, pair_types_j, pair_idx, n_atoms = extract_pair_data(frame)
            return potential.calc_forces(dr, dr_norm, pair_types_i, pair_types_j, pair_idx, n_atoms)
        elif "coul" in potential_name.lower():
            r, pair_idx, charges = extract_coul_data(frame)
            return potential.calc_forces(r, pair_idx, charges)
        else:
            raise TypeError(f"Unknown pair potential type: {potential_name}")
    else:
        raise TypeError(f"Unknown potential type: {potential_type}")


def calc_energy_from_frame_multi(potentials, frame: Frame) -> float:
    """Calculate total energy from multiple potentials."""
    total_energy = 0.0
    for potential in potentials:
        total_energy += calc_energy_from_frame(potential, frame)
    return total_energy
