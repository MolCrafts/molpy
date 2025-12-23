from typing import TYPE_CHECKING

import numpy as np

from molpy.core.frame import Frame

if TYPE_CHECKING:
    from .constraint import Constraint

# Note: All constraint classes are now in molpy.pack.constraint
# No need for molpack dependency


class Target:
    def __init__(
        self,
        frame: Frame,
        number: int,
        constraint: "Constraint",
        is_fixed: bool = False,
        name: str = "",
    ):
        self.frame = frame
        self.number = number
        self.constraint = constraint
        self.is_fixed = is_fixed
        self.name = name

    def _get_n_atoms(self) -> int:
        """Helper method to get number of atoms from frame."""
        atoms = self.frame["atoms"]
        if "id" in atoms:
            return len(atoms["id"])
        elif "x" in atoms:
            return len(atoms["x"])
        else:
            return 0

    def __repr__(self):
        n_atoms = self._get_n_atoms()
        return f"<Target {self.name}: {n_atoms} atoms in {self.constraint}>"

    @property
    def n_points(self):
        return self._get_n_atoms() * self.number

    def _extract_coordinates(self) -> np.ndarray:
        """Extract coordinates from frame atoms block."""
        atoms = self.frame["atoms"]
        n_atoms = self._get_n_atoms()

        if n_atoms == 0:
            # Return empty array with correct shape for empty frames
            return np.empty((0, 3))

        if "xyz" in atoms:
            coords = atoms["xyz"].values
        elif all(coord in atoms for coord in ["x", "y", "z"]):
            # Extract coordinates using simple array stacking
            x = atoms["x"].values
            y = atoms["y"].values
            z = atoms["z"].values
            coords = np.column_stack([x, y, z])
        else:
            raise ValueError(
                "Frame must contain either 'xyz' or 'x', 'y', 'z' coordinates"
            )

        return coords

    @property
    def points(self) -> np.ndarray:
        """Get all points (coordinates replicated for each copy)."""
        coords = self._extract_coordinates()
        # Replicate coordinates for each copy of the molecule
        return np.tile(coords, (self.number, 1))
