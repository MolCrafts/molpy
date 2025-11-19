"""
Utility functions for working with potential functions and Frame objects.

This module provides helper functions to extract data from Frame objects
and call potential functions with the extracted data.
"""

import numpy as np
from numpy.typing import NDArray

from molpy import Frame
from molpy.potential.base import Potential, Potentials


def extract_bond_data(frame: Frame) -> tuple[NDArray, NDArray, NDArray]:
    """
    Extract bond data from Frame.

    Args:
        frame: Frame containing atom coordinates and bond information

    Returns:
        Tuple of (r, bond_idx, bond_types)
        - r: Atom coordinates (shape: (n_atoms, 3))
        - bond_idx: Bond indices (shape: (n_bonds, 2))
        - bond_types: Bond types (shape: (n_bonds,))
    """
    r = frame["atoms"]["xyz"]
    bonds = frame["bonds"]
    bond_idx = bonds[["atom_i", "atom_j"]]
    bond_types = bonds["type"]
    return r, bond_idx, bond_types


def extract_angle_data(frame: Frame) -> tuple[NDArray, NDArray, NDArray]:
    """
    Extract angle data from Frame.

    Args:
        frame: Frame containing atom coordinates and angle information

    Returns:
        Tuple of (r, angle_idx, angle_types)
        - r: Atom coordinates (shape: (n_atoms, 3))
        - angle_idx: Angle indices (shape: (n_angles, 3))
        - angle_types: Angle types (shape: (n_angles,))
    """
    r = frame["atoms"]["xyz"]
    angles = frame["angles"]
    angle_idx = angles[["atom_i", "atom_j", "atom_k"]]
    angle_types = angles["type"]
    return r, angle_idx, angle_types


def extract_pair_data(frame: Frame) -> tuple[NDArray, NDArray, NDArray, NDArray, int]:
    """
    Extract pair data from Frame.

    Args:
        frame: Frame containing atom coordinates and pair information

    Returns:
        Tuple of (dr, dr_norm, pair_types, pair_idx, n_atoms)
        - dr: Pair displacement vectors (shape: (n_pairs, 3))
        - dr_norm: Pair distances (shape: (n_pairs, 1))
        - pair_types: Pair types (shape: (n_pairs,))
        - pair_idx: Pair indices (shape: (n_pairs, 2))
        - n_atoms: Number of atoms
    """
    r = frame["atoms"]["xyz"]
    n_atoms = len(r)

    if "pairs" in frame:
        pairs = frame["pairs"]
        dr = pairs["dr"]
        dr_norm = np.linalg.norm(dr, axis=1, keepdims=True)
        pair_types = pairs["type"]
        pair_idx = pairs[["i", "j"]]
    else:
        # Simple all-pairs calculation (for testing/small systems)
        # In production, you'd use a neighbor list
        dr_list = []
        pair_types_list = []
        pair_idx_list = []

        atom_types = frame["atoms"].get("type", np.zeros(n_atoms, dtype=np.int32))

        for i in range(n_atoms):
            for j in range(i + 1, n_atoms):
                dr_vec = r[j] - r[i]
                dr_list.append(dr_vec)
                pair_types_list.append(max(atom_types[i], atom_types[j]))
                pair_idx_list.append([i, j])

        if dr_list:
            dr = np.array(dr_list)
            dr_norm = np.linalg.norm(dr, axis=1, keepdims=True)
            pair_types = np.array(pair_types_list, dtype=np.int32)
            pair_idx = np.array(pair_idx_list)
        else:
            dr = np.empty((0, 3))
            dr_norm = np.empty((0, 1))
            pair_types = np.empty(0, dtype=np.int32)
            pair_idx = np.empty((0, 2), dtype=np.int32)

    return dr, dr_norm, pair_types, pair_idx, n_atoms


def extract_coul_data(frame: Frame) -> tuple[NDArray, NDArray, NDArray]:
    """
    Extract Coulomb data from Frame.

    Args:
        frame: Frame containing atom coordinates, charges, and pair information

    Returns:
        Tuple of (r, pair_idx, charges)
        - r: Atom coordinates (shape: (n_atoms, 3))
        - pair_idx: Pair indices (shape: (n_pairs, 2))
        - charges: Atom charges (shape: (n_atoms,))
    """
    r = frame["atoms"]["xyz"]

    if "charge" not in frame["atoms"]:
        raise ValueError("Frame must contain 'charge' column in atoms")
    charges = frame["atoms"]["charge"]

    if "pairs" in frame:
        pairs = frame["pairs"]
        pair_idx = pairs[["i", "j"]]
    else:
        # Simple all-pairs case (for testing)
        n_atoms = len(r)
        pair_idx_list = []
        for i in range(n_atoms):
            for j in range(i + 1, n_atoms):
                pair_idx_list.append([i, j])
        pair_idx = np.array(pair_idx_list)

    return r, pair_idx, charges


def calc_energy_from_frame(potential: Potential, frame: Frame) -> float:
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
        # Check if it's LJ126 or CoulCut by looking at the name
        potential_name = getattr(potential, "name", "")
        if "lj" in potential_name.lower() or "lj126" in potential_name.lower():
            dr, dr_norm, pair_types, _, _ = extract_pair_data(frame)
            return potential.calc_energy(dr, dr_norm, pair_types)
        elif "coul" in potential_name.lower():
            r, pair_idx, charges = extract_coul_data(frame)
            return potential.calc_energy(r, pair_idx, charges)
        else:
            raise TypeError(f"Unknown pair potential type: {potential_name}")
    else:
        raise TypeError(f"Unknown potential type: {potential_type}")


def calc_forces_from_frame(potential: Potential, frame: Frame) -> NDArray:
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
        # Check if it's LJ126 or CoulCut by looking at the name
        potential_name = getattr(potential, "name", "")
        if "lj" in potential_name.lower() or "lj126" in potential_name.lower():
            dr, dr_norm, pair_types, pair_idx, n_atoms = extract_pair_data(frame)
            return potential.calc_forces(dr, dr_norm, pair_types, pair_idx, n_atoms)
        elif "coul" in potential_name.lower():
            r, pair_idx, charges = extract_coul_data(frame)
            return potential.calc_forces(r, pair_idx, charges)
        else:
            raise TypeError(f"Unknown pair potential type: {potential_name}")
    else:
        raise TypeError(f"Unknown potential type: {potential_type}")


def calc_energy_from_frame_multi(potentials: Potentials, frame: Frame) -> float:
    """
    Calculate total energy from Frame for multiple potentials.

    Args:
        potentials: Collection of potentials
        frame: Frame containing the necessary data

    Returns:
        Total potential energy
    """
    return sum(calc_energy_from_frame(pot, frame) for pot in potentials)


def calc_forces_from_frame_multi(potentials: Potentials, frame: Frame) -> NDArray:
    """
    Calculate total forces from Frame for multiple potentials.

    Args:
        potentials: Collection of potentials
        frame: Frame containing the necessary data

    Returns:
        Array of total forces on each atom (shape: (n_atoms, 3))
    """
    r = frame["atoms"]["xyz"]
    n_atoms = len(r)
    total_forces = np.zeros((n_atoms, 3), dtype=np.float64)

    for pot in potentials:
        forces = calc_forces_from_frame(pot, frame)
        total_forces += forces

    return total_forces
