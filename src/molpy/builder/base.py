import numpy as np
from abc import ABC, abstractmethod
from molpy.core import Struct


class BaseBuilder(ABC):

    @abstractmethod
    def create_sites(self) -> np.ndarray: ...

    @abstractmethod
    def fill(self, struct: Struct) -> Struct: ...

def set_struct(struct: Struct, sites: np.ndarray) -> Struct:
    """
    Set the coordinates of the structure to the given sites.
    """
    ref_point = struct.atoms[0].xyz
    dr = sites - ref_point
    for atom in struct.atoms:
        atom.xyz += dr
    return struct

class LatticeBuilder(ABC):
        @abstractmethod
        def create_sites(self, **params):
            ...

class StructBuilder(ABC):
        @abstractmethod
        def populate(self, sites, **params):
            ...


class FCCBuilder(LatticeBuilder):
    def __init__(self, a: float, shape: tuple[int,int,int]):
          self.a = a
          self.shape = shape
    def create_sites(self) -> np.ndarray:
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
    def __init__(self, a: float, shape: tuple[int,int,int]):
        self.a = a
        self.shape = shape
    
    def create_sites(self) -> np.ndarray:
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
    def __init__(self, a: float, c: float, shape: tuple[int,int,int]):
        self.a, self.c, self.shape = a, c, shape 

    def create_sites(self) -> np.ndarray:
        nx, ny, nz = self.shape
        
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

class BuildManager:
    def __init__(
        self,
        lattice_builder: LatticeBuilder,
        struct_builder: StructBuilder,
        base_struct: Struct = None
    ):
       
        self.lattice_builder = lattice_builder
        self.struct_builder  = struct_builder
        self.base_struct     = base_struct or Struct()

    def build(
        self,
        lat_kwargs: dict | None    = None,
        struct_kwargs: dict | None = None
    ) -> Struct:
       
        lat_kwargs    = lat_kwargs    or {}
        struct_kwargs = struct_kwargs or {}

      
        sites = self.lattice_builder.create_sites(**lat_kwargs)

     
        struct_copy = self.base_struct.copy()
        result = self.struct_builder.populate(sites, **struct_kwargs)

     
        return result