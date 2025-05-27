import numpy as np
import random
from molpy.core import Struct
from molpy.core.region import Region
from .base import LatticeBuilder, StructBuilder
from .base import BaseBuilder, set_struct


class UniformRandomBuilder(LatticeBuilder, StructBuilder):
    """
    A builder that generates random coordinates uniformly in a given range.
    """

    def __init__(self, n: int, region: Region):
        self.n = n
        self.region = region

    def create_sites(self) -> Lattice:
        """
        Generate random coordinates uniformly in the given region.
        """
        # Get the shape of the region
        bounds = self.region.bounds
        xlo, xhi = bounds[0]
        ylo, yhi = bounds[1]
        zlo, zhi = bounds[2]

        sites = np.random.uniform(
            low=[xlo, ylo, zlo],
            high=[xhi, yhi, zhi],
            size=(self.n, 3),
        )
        site_mask = self.region.isin(sites)
        return Lattice(sites[site_mask], bounds)
    
    def build_structs(self, struct: Struct):
        """
        Fill the structure with random coordinates.
        """
        structs = []
        sites = self.create_sites(n)
        for site in sites:
            new_struct = struct.clone()
            set_struct(new_struct, site)
            structs.append(new_struct)
        return structs


class RandomLatticeBuilder(LatticeBuilder):
     def create_sites(self,
                     n_steps: int = 100,
                     step_size: float = 1.0,
                     seed: int = None):
        if seed is not None:
            random.seed(seed)
        coords = [(0.0, 0.0, 0.0)]
        for _ in range(n_steps):
            x, y, z = coords[-1]
            dx = random.uniform(-step_size, step_size)
            dy = random.uniform(-step_size, step_size)
            dz = random.uniform(-step_size, step_size)
            coords.append((x + dx, y + dy, z + dz))
        return coords
     

class RandomStructBuilder(StructBuilder):
    def populate(self, sites, monomer: str = 'M', **params):
        mols = []
        for pos in sites:
            mols.append({
                'monomer': monomer,
                'position': pos
            })
        return mols