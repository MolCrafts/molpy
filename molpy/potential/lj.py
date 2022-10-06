from typing import NamedTuple
import jax.numpy as np
import jax
from .utils import make_pbc
import molpy as mp

class LJCut:

    class Params(NamedTuple):

        epsilon: np.array
        sigma: np.array

    def __init__(self, r_cutoff, is_pbc=True) -> None:
        
        self.r_cutoff = r_cutoff
        self.is_pbc = is_pbc

    def _compute(self, xyz: np.array, pairs: np.array, box: np.array, params:Params) -> np.array:
            
        epsilon = params['epsilon']
        sigma = params['sigma']

        r_ij = xyz[pairs[:, 0]] - xyz[pairs[:, 1]]

        if self.is_pbc:
            r_ij = make_pbc(r_ij, box)

        r2 = np.sum(r_ij ** 2, axis=1)
        r6 = r2 ** 3
        r12 = r6 ** 2

        sigma2 =  sigma ** 2
        sigma6 = sigma2 ** 3
        sigma12 = sigma6 ** 2

        V = 4 * epsilon * (sigma12 / r12 - sigma6 / r6)

        mask = r2 < self.r_cutoff ** 2

        return np.sum(V * mask)

    def energy(self, xyz: np.array, pairs: np.array, box: mp.Box, params:Params) -> np.array:
        # box_vec = box.to_matrix().astype(float)
        return self._compute(xyz, pairs, box, params)

    def force(self, xyz: np.array, pairs: np.array, box: mp.Box, params:Params) -> np.array:
        # box_vec = box.to_matrix().astype(float)
        return jax.grad(self._compute)(xyz, pairs, box, params)
