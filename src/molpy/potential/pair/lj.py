# author: Roy Kid
# contact: lijichen365@126.com
# date: 2024-02-18
# version: 0.0.1

import numpy as np
import molpy as mp
from molpy.potential.base import Potential

def segment_sum(data, segment_ids, dim_size):
    data = np.asarray(data)
    s = np.zeros((dim_size,) + data.shape[1:], dtype=data.dtype)
    np.add.at(s, segment_ids, data)
    return s

class LJ126(Potential):

    def __init__(self, epsilon, sigma, cutoff):

        self.epsilon = epsilon
        self.sigma = sigma
        self.cutoff = cutoff

    def forward(self, input):

        rij = input[mp.Alias.Rij]
        idx_i = input[mp.Alias.idx_i]
        idx_j = input[mp.Alias.idx_j]

        # energy = self.energy(rij, idx_i, idx_j)
        # forces = self.forces(rij, idx_i, idx_j)
        # input[mp.Alias.energy] = energy
        # input.atoms[mp.Alias.forces] = per_atom_forces
        return input

    def energy(self, rij):
        """
        compute energy of pair potential

        Args:
            rij (np.ndarray (n_pairs, dim)): displacement vectors

        Returns:
            nd.ndarray (n_pairs, 1): pair energy
        """
        dij = np.linalg.norm(rij, axis=-1, keepdims=True)
        power_6 = np.power(self.sigma / dij, 6)
        power_12 = np.square(power_6)

        e = 4 * self.epsilon * (power_12 - power_6)
        return e
    
    def forces(self, rij):
        """
        compute forces of pair potential

        Args:
            rij (np.ndarray (n_paris, dim)): displacement vectors

        Returns:
            np.ndarray (n_pairs, dim): pair forces
        """
        dij = np.linalg.norm(rij, axis=-1, keepdims=True)

        power_6 = np.power(self.sigma / dij, 6)
        power_12 = np.square(power_6)

        f = 24 * self.epsilon * (2 * power_12 - power_6) / dij**2 * rij
        return f