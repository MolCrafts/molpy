import numpy as np
from molpy.core import Struct, Atom
from .base import set_struct

class Lattice:
    """
    A class representing a crystal lattice structure.
    
    This class encapsulates both the positions of lattice sites and the cell vectors
    that define the periodicity of the crystal structure. It provides methods for
    manipulating and extending the lattice.
    
    Attributes:
        sites (np.ndarray): Array of coordinates for lattice points
        cell (np.ndarray): 3x3 matrix where rows are lattice vectors
    """
    
    def __init__(self, sites: np.ndarray, cell: np.ndarray):
        """
        Initialize a Lattice object.
        
        Args:
            sites (np.ndarray): Array of coordinates for lattice points
            cell (np.ndarray): 3x3 matrix where rows are lattice vectors
        """
        self.sites = sites
        self.cell = np.asarray(cell, float)

    def repeat(self, nx: int = 1, ny: int = 1, nz: int = 1):
        """
        Create a new lattice by repeating the current one in each direction.
        
        Args:
            nx (int): Number of repetitions in x direction
            ny (int): Number of repetitions in y direction
            nz (int): Number of repetitions in z direction
            
        Returns:
            Lattice: New lattice object with repeated structure
        """
        shape = (nx, ny, nz)
        basis = self.sites
        cell = self.cell
        reps = []
        for i in range(nx):
            for j in range(ny):
                for k in range(nz):
                    offset = i*cell[0] + j*cell[1] + k*cell[2]
                    reps.append(basis + offset)
        sites = np.vstack(reps)
        cell = cell * np.array(shape)
        return Lattice(sites, cell)

    def fill(self, struct: Struct) -> Struct:
        """
        Create a structure by placing copies of a template structure at each lattice site.
        
        Args:
            struct (Struct): Template structure to place at each lattice site
            
        Returns:
            Struct: New structure containing copies at all lattice sites
        """
        result = Struct()
        for pos in self.sites:
            s = struct.copy()
            set_struct(s, pos)
            result = Struct.merge([result, s])
        return result


