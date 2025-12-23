"""Potential wrappers for optimizer compatibility.

Since potentials in molpy work with raw arrays (positions, indices, types),
not Frame objects, we need simple wrappers to extract data from Frames.

Each wrapper implements calc_energy(frame) and calc_forces(frame) by
extracting the appropriate data and calling the underlying potential.
"""

import numpy as np


class BondPotentialWrapper:
    """Wrapper for bond potentials to work with Frame interface.

    Extracts (r, bond_idx, bond_types) from Frame and calls underlying potential.
    """

    def __init__(self, potential):
        self.potential = potential
        self.type = "bond"
        self.name = getattr(potential, "name", "bond")

    def calc_energy(self, frame):
        """Extract bond data from Frame and compute energy."""
        # Get coordinates from x, y, z fields (never use xyz)
        x = frame["atoms"]["x"]
        y = frame["atoms"]["y"]
        z = frame["atoms"]["z"]
        r = np.column_stack([x, y, z])
        bonds = frame["bonds"]
        bond_idx = bonds[["atom_i", "atom_j"]]
        bond_types = bonds["type"]
        return self.potential.calc_energy(r, bond_idx, bond_types)

    def calc_forces(self, frame):
        """Extract bond data from Frame and compute forces."""
        # Get coordinates from x, y, z fields (never use xyz)
        x = frame["atoms"]["x"]
        y = frame["atoms"]["y"]
        z = frame["atoms"]["z"]
        r = np.column_stack([x, y, z])
        bonds = frame["bonds"]
        bond_idx = bonds[["atom_i", "atom_j"]]
        bond_types = bonds["type"]
        return self.potential.calc_forces(r, bond_idx, bond_types)


class AnglePotentialWrapper:
    """Wrapper for angle potentials to work with Frame interface.

    Extracts (r, angle_idx, angle_types) from Frame and calls underlying potential.
    """

    def __init__(self, potential):
        self.potential = potential
        self.type = "angle"
        self.name = getattr(potential, "name", "angle")

    def calc_energy(self, frame):
        """Extract angle data from Frame and compute energy."""
        # Get coordinates from x, y, z fields (never use xyz)
        x = frame["atoms"]["x"]
        y = frame["atoms"]["y"]
        z = frame["atoms"]["z"]
        r = np.column_stack([x, y, z])
        angles = frame["angles"]
        angle_idx = angles[["atom_i", "atom_j", "atom_k"]]
        angle_types = angles["type"]
        return self.potential.calc_energy(r, angle_idx, angle_types)

    def calc_forces(self, frame):
        """Extract angle data from Frame and compute forces."""
        # Get coordinates from x, y, z fields (never use xyz)
        x = frame["atoms"]["x"]
        y = frame["atoms"]["y"]
        z = frame["atoms"]["z"]
        r = np.column_stack([x, y, z])
        angles = frame["angles"]
        angle_idx = angles[["atom_i", "atom_j", "atom_k"]]
        angle_types = angles["type"]
        return self.potential.calc_forces(r, angle_idx, angle_types)
