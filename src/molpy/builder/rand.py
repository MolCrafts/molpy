import numpy as np
import random
from molpy.core import Struct, Atom
from molpy.core.region import Region
from .base import LatticeBuilder, StructBuilder, set_struct

class UniformRandomBuilder(LatticeBuilder, StructBuilder):
    def __init__(self, n: int, region: Region):
        self.n = n
        self.region = region

    def create_sites(self) -> np.ndarray:
        xlo, xhi = self.region.bounds[0]
        ylo, yhi = self.region.bounds[1]
        zlo, zhi = self.region.bounds[2]
        sites = np.random.uniform(
            low=[xlo, ylo, zlo],
            high=[xhi, yhi, zhi],
            size=(self.n, 3),
        )
        mask = self.region.isin(sites)
        return sites[mask]

    def populate(self, sites: np.ndarray, monomer: Struct = None) -> Struct:
        result = Struct()
        base = monomer or Struct()
        for site in sites:
            s = base.copy()
            set_struct(s, site)
            result = Struct.merge([result, s])
        return result

class RandomLatticeBuilder(LatticeBuilder):
    def __init__(self, n_steps: int = 100, step_size: float = 1.0, seed: int = None):
        self.n_steps = n_steps
        self.step_size = step_size
        self.seed = seed

    def create_sites(self) -> np.ndarray:
        if self.seed is not None:
            random.seed(self.seed)
        coords = [(0.0, 0.0, 0.0)]
        for _ in range(self.n_steps):
            x, y, z = coords[-1]
            dx = random.uniform(-self.step_size, self.step_size)
            dy = random.uniform(-self.step_size, self.step_size)
            dz = random.uniform(-self.step_size, self.step_size)
            coords.append((x + dx, y + dy, z + dz))
        return np.array(coords)

class RandomStructBuilder(StructBuilder):
    def populate(self, sites: np.ndarray, monomer: str = 'M', **params) -> Struct:
        result = Struct()
        for pos in sites:
            s = Struct(name=monomer)
            s.add_atom(Atom(monomer, *pos))
            result = Struct.merge([result, s])
        return result
