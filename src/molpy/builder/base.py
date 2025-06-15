import numpy as np
from abc import ABC, abstractmethod
from molpy.core import Struct
from .crystal import Lattice

class BaseBuilder(ABC):
    """
    Abstract base class for structure builders.
    
    This class defines the basic interface that all builders must implement
    for creating molecular structures.
    """
    
    @abstractmethod
    def create_sites(self) -> np.ndarray:
        """
        Create an array of site coordinates.
        
        Returns:
            np.ndarray: Array of 3D coordinates
        """
        ...

    @abstractmethod
    def fill(self, struct: Struct) -> Struct:
        """
        Fill the sites with a given structure.
        
        Args:
            struct (Struct): Structure to place at sites
            
        Returns:
            Struct: Resulting structure
        """
        ...

def set_struct(struct: Struct, sites: np.ndarray) -> Struct:
    """
    Set the coordinates of the structure to the given sites.
    
    This function translates a structure to align with given site coordinates
    while preserving its internal geometry.
    
    Args:
        struct (Struct): Structure to move
        sites (np.ndarray): Target coordinates
        
    Returns:
        Struct: Translated structure
    """
    ref_point = struct.atoms[0].xyz
    dr = sites - ref_point
    for atom in struct.atoms:
        atom.xyz += dr
    return struct

class LatticeBuilder(ABC):
    """
    Abstract base class for lattice builders.
    
    This class defines the interface for creating crystal lattices with
    specific geometries and periodicities.
    """
    
    @abstractmethod
    def create_sites(self, **params) -> Lattice:
        """
        Create a lattice with sites and cell vectors.
        
        Args:
            **params: Additional parameters specific to the lattice type
            
        Returns:
            Lattice: Object containing both sites and cell vectors
        """
        ...

class StructBuilder(ABC):
    """
    Abstract base class for structure builders.
    
    This class defines the interface for populating lattice sites with
    atoms or molecules to create complete structures.
    """
    
    @abstractmethod
    def populate(self, lattice: Lattice, **params) -> Struct:
        """
        Populate lattice sites with atoms/molecules.
        
        Args:
            lattice (Lattice): Lattice object containing sites and cell vectors
            **params: Additional parameters for structure building
            
        Returns:
            Struct: Populated structure
        """
        ...

class BuildManager:
    """
    Coordinates the building process between lattice and structure builders.
    
    This class manages the workflow of creating a complete structure by
    coordinating between a LatticeBuilder that creates the sites and a
    StructBuilder that populates those sites.
    """
    
    def __init__(
        self,
        lattice_builder: LatticeBuilder,
        struct_builder: StructBuilder,
        base_struct: Struct = None
    ):
        """
        Initialize BuildManager.
        
        Args:
            lattice_builder (LatticeBuilder): Builder for creating lattice sites
            struct_builder (StructBuilder): Builder for populating sites
            base_struct (Struct, optional): Base structure to use. Defaults to None.
        """
        self.lattice_builder = lattice_builder
        self.struct_builder = struct_builder
        self.base_struct = base_struct or Struct()

    def build(
        self,
        lat_kwargs: dict | None = None,
        struct_kwargs: dict | None = None
    ) -> Struct:
        """
        Build the complete structure.
        
        Args:
            lat_kwargs (dict, optional): Parameters for lattice building. Defaults to None.
            struct_kwargs (dict, optional): Parameters for structure building. Defaults to None.
            
        Returns:
            Struct: Complete built structure
        """
        lat_kwargs = lat_kwargs or {}
        struct_kwargs = struct_kwargs or {}
        sites = self.lattice_builder.create_sites(**lat_kwargs)
        struct_copy = self.base_struct.copy()
        result = self.struct_builder.populate(sites, **struct_kwargs)
        return result

class CrystalBuilder(LatticeBuilder, StructBuilder):
    """
    Combined builder for crystal structures.
    
    This class implements both LatticeBuilder and StructBuilder interfaces
    to provide a complete solution for building crystal structures with
    specified lattice parameters and atomic arrangements.
    """
    
    def __init__(
        self,
        a: float,
        b: float | None = None,
        c: float | None = None,
        *,
        alpha: float | None = None,
        covera: float | None = None,
        u: float | None = None,
        orthorhombic: bool = False,
        cubic: bool = False,
        basis: np.ndarray | None = None,
        cell: np.ndarray | None = None,
        repeat: tuple[int,int,int] = (1,1,1),
    ):
        """
        Initialize CrystalBuilder.
        
        Args:
            a (float): First lattice constant
            b (float, optional): Second lattice constant. Defaults to a.
            c (float, optional): Third lattice constant. Defaults to a.
            alpha (float, optional): Angle between b and c vectors (degrees)
            covera (float, optional): c/a ratio for hexagonal cells
            u (float, optional): Internal parameter for certain structures
            orthorhombic (bool): Whether to use orthorhombic cell
            cubic (bool): Whether to use cubic cell
            basis (np.ndarray, optional): Basis vectors for the crystal
            cell (np.ndarray, optional): Cell vectors
            repeat (tuple[int,int,int]): Number of repetitions in each direction
        """
        self.a = a
        self.b = b or a
        self.c = c or a
        self.alpha = alpha
        self.covera = covera
        self.u = u
        self.orthorhombic = orthorhombic
        self.cubic = cubic
        self.basis = np.asarray(basis, float)
        self.cell = np.asarray(cell, float)
        self.repeat_dims = repeat

    def create_sites(self) -> np.ndarray:
        """
        Create crystal lattice sites.
        
        Returns:
            np.ndarray: Array of site coordinates
        """
        sites = self.basis * np.array([self.a, self.b, self.c])
        lattice = Lattice(sites, self.cell)
        lattice = lattice.repeat(*self.repeat_dims)
        return lattice.sites

    def populate(self, sites: np.ndarray, struct: Struct) -> Struct:
        """
        Populate crystal sites with atoms.
        
        Args:
            sites (np.ndarray): Array of site coordinates
            struct (Struct): Structure to place at each site
            
        Returns:
            Struct: Complete crystal structure
        """
        result = Struct()
        for pos in sites:
            s = struct.copy()
            set_struct(s, pos)
            result = Struct.merge([result, s])
        return result

class PolymerBuilder(StructBuilder):
    def __init__(self, monomer: Struct, repeat: int, spacing: float):
        self.monomer, self.repeat, self.spacing = monomer, repeat, spacing

    def populate(self, sites=None, **params) -> Struct:
        # Ignorujemy sites – budujemy wzdłuż jednej osi
        result = Struct()
        offset = 0.0
        for i in range(self.repeat):
            mcopy = self.monomer.copy()
            mcopy.translate((offset, 0, 0))
            result = Struct.merge([result, mcopy])
            offset += self.spacing
        return result

class ClusterBuilder(StructBuilder):
    def __init__(self, monomer: Struct, radius: float, count: int):
        self.monomer, self.radius, self.count = monomer, radius, count

    def populate(self, sites=None, **params) -> Struct:
        import random
        result = Struct()
        for _ in range(self.count):
            phi = random.random()*2*np.pi
            cost = random.uniform(-1,1)
            u = random.random()
            r = self.radius * u**(1/3)
            x = r * np.sqrt(1-cost**2) * np.cos(phi)
            y = r * np.sqrt(1-cost**2) * np.sin(phi)
            z = r * cost
            mcopy = self.monomer.copy()
            mcopy.translate((x,y,z))
            result = Struct.merge([result, mcopy])
        return result



    
 

