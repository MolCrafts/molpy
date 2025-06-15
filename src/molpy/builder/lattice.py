from .base import LatticeBuilder
import numpy as np

class FCCBuilder(LatticeBuilder):
    """
    Builder for Face-Centered Cubic (FCC) crystal lattice structures.
    
    The FCC lattice has atoms at each corner of the cubic cell and at the center 
    of each face. This results in 4 atoms per unit cell.
    """
    
    def __init__(self, a: float, shape: tuple[int,int,int]):
        """
        Initialize FCC lattice builder.
        
        Args:
            a (float): Lattice constant (edge length of the cubic cell)
            shape (tuple[int,int,int]): Number of unit cells in x, y, z directions
        """
        self.a = a
        self.shape = shape
        
    def create_sites(self) -> np.ndarray:
        """
        Generate coordinates for FCC lattice sites.
        
        Returns:
            np.ndarray: Array of 3D coordinates for all lattice points
        """
        nx, ny, nz = self.shape
        base = [(0,0,0), (0.5,0.5,0), (0.5,0,0.5), (0,0.5,0.5)]
        pts = []
        for i in range(nx):
            for j in range(ny):
                for k in range(nz):
                    for x, y, z in base:
                        pts.append(((i+x)*self.a, (j+y)*self.a, (k+z)*self.a))
        return np.array(pts)


class BCCBuilder(LatticeBuilder):
    """
    Builder for Body-Centered Cubic (BCC) crystal lattice structures.
    
    The BCC lattice has atoms at each corner of the cubic cell and one atom 
    at the center of the cell. This results in 2 atoms per unit cell.
    """
    
    def __init__(self, a: float, shape: tuple[int,int,int]):
        """
        Initialize BCC lattice builder.
        
        Args:
            a (float): Lattice constant (edge length of the cubic cell)
            shape (tuple[int,int,int]): Number of unit cells in x, y, z directions
        """
        self.a = a
        self.shape = shape
    
    def create_sites(self) -> np.ndarray:
        """
        Generate coordinates for BCC lattice sites.
        
        Returns:
            np.ndarray: Array of 3D coordinates for all lattice points
        """
        nx, ny, nz = self.shape
        base = [(0,0,0), (0.5,0.5,0.5)]
        pts = []
        for i in range(nx):
            for j in range(ny):
                for k in range(nz):
                    for (x,y,z) in base:
                        pts.append(((i+x)*self.a, (j+y)*self.a, (k+z)*self.a))
        return np.array(pts)

class HCPBuilder(LatticeBuilder):
    """
    Builder for Hexagonal Close-Packed (HCP) crystal lattice structures.
    
    The HCP lattice consists of alternating layers of hexagonally arranged atoms.
    It has a hexagonal unit cell with two characteristic lengths: the basal plane
    lattice constant 'a' and the perpendicular height 'c'.
    """
    
    def __init__(self, a: float, c: float, shape: tuple[int,int,int]):
        """
        Initialize HCP lattice builder.
        
        Args:
            a (float): Basal plane lattice constant
            c (float): Height of the unit cell (perpendicular to basal plane)
            shape (tuple[int,int,int]): Number of unit cells in x, y, z directions
        """
        self.a, self.c, self.shape = a, c, shape 

    def create_sites(self) -> np.ndarray:
        """
        Generate coordinates for HCP lattice sites.
        
        The HCP structure is created by stacking hexagonal layers in an ABAB pattern,
        where each layer is shifted relative to the layers above and below it.
        
        Returns:
            np.ndarray: Array of 3D coordinates for all lattice points
        """
        nx, ny, nz = self.shape
        # Definition of HCP unit vertices
        base = [
            (0, 0, 0),
            (2/3, 1/3, 0.5),
            (1/3, 2/3, 0.5),
            (0, 0, 1),
            (2/3, 1/3, 1.5),
            (1/3, 2/3, 1.5)
        ]
        pts = []
        for i in range(nx):
            for j in range(ny):
                for k in range(nz):
                    for (u,v,w) in base:
                        x = (i + u) * self.a
                        y = (j + v) * self.a * np.sqrt(3)/2
                        z = (k + w) * self.c
                        pts.append((x,y,z))
        return np.array(pts)