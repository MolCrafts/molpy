import numpy as np

from molpy.core import Struct
from molpy.core.region import Region

from .base import BaseBuilder, set_struct


class UniformRandomBuilder(BaseBuilder):
    """
    A builder that generates random coordinates uniformly in a given range.
    """

    def __init__(self, region: Region):
        self.region = region

    def create_sites(self, n: int) -> np.ndarray:
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
            size=(n, 3),
        )
        site_mask = self.region.isin(sites)
        return sites[site_mask]
    
    def fill(self, n:int, struct: Struct):
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