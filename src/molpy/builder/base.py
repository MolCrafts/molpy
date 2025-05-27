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

class BuildManager:
    def __init__(self, lattice_builder: LatticeBuilder, struct_builder: StructBuilder):
        self.lattice = lattice_builder
        self.struct  = struct_builder

    def build(self, **params):
        keys = list(params.keys())
        n = len(keys) //2
        lat_kwargs = {k: params[k] for k in keys[:n]}
        struct_kwargs = {k: params[k] for k in keys[n:]}
        sites = self.lattice.create_sites(**lat_kwargs)
        struct = self.struct.populate(sites, **struct_kwargs)
        return struct