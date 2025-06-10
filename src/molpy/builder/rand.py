import numpy as np

from molpy.core import Struct
from molpy.core.region import Region

from .base import Lattice, LatticeBuilder, StructBuilder


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
        sites = self.create_sites()

        return sites.fill(struct)