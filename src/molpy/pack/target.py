import molpy as mp
import numpy as np
from typing import TYPE_CHECKING, Union

# Note: All constraint classes are now in molpy.pack.constraint
# No need for molpack dependency


class Target:

    def __init__(
        self,
        frame: mp.Frame,
        number: int,
        constraint,  # Remove specific type annotation since mpk might not be available
        is_fixed: bool = False,
        optimizer=None,
        name: str = "",
    ):
        self.frame = frame
        self.number = number
        self.constraint = constraint
        self.is_fixed = is_fixed
        self.optimizer = optimizer
        self.name = name

    def __repr__(self):
        atoms = self.frame["atoms"]
        n_atoms = len(atoms['id']) if 'id' in atoms else len(atoms['x']) if 'x' in atoms else 0
        return f"<Target {self.name}: {n_atoms} atoms in {self.constraint}>"

    @property
    def n_points(self):
        atoms = self.frame["atoms"]
        n_atoms = len(atoms['id']) if 'id' in atoms else len(atoms['x']) if 'x' in atoms else 0
        return n_atoms * self.number

    @property
    def points(self):
        atoms = self.frame["atoms"]
        # Handle different coordinate formats and empty frames
        if 'id' in atoms:
            n_atoms = len(atoms['id'])
        elif 'x' in atoms:
            n_atoms = len(atoms['x'])
        else:
            n_atoms = 0
            
        if n_atoms == 0:
            # Return empty array with correct shape for empty frames
            return np.empty((0, 3))
        
        if "xyz" in atoms:
            coords = atoms["xyz"].values
        elif all(coord in atoms for coord in ["x", "y", "z"]):
            # Extract coordinates using simple array stacking
            x = atoms['x'].values
            y = atoms['y'].values
            z = atoms['z'].values
            coords = np.column_stack([x, y, z])
        else:
            raise ValueError("Frame must contain either 'xyz' or 'x', 'y', 'z' coordinates")
        
        # Replicate coordinates for each copy of the molecule
        return np.tile(coords, (self.number, 1))
