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

    @staticmethod
    def F(dij, eps, sig):
        power6 = np.power(sig / dij, 6)
        power12 = np.square(power6)
        return 4 * eps * (power12 - power6)

    def __init__(self, epsilon, sigma, cutoff):
        super().__init__('LJ126', 'pair')
        self.epsilon = epsilon
        self.sigma = sigma
        self.cutoff = cutoff

    # def forward(self, input):

    #     rij = input[mp.Alias.Rij]
    #     idx_i = input[mp.Alias.idx_i]
    #     idx_j = input[mp.Alias.idx_j]

    #     energy = self.energy(rij)
    #     pairs_forces = self.forces(rij)
    #     input[mp.Alias.energy] += np.sum(energy)
    #     np.add.at(input.atoms[mp.Alias.forces], idx_i, pairs_forces)
    #     np.add.at(input.atoms[mp.Alias.forces], idx_j, -pairs_forces)
    #     return input

    def energy(self, R, atomtype, idx_i, idx_j):
        """
        compute energy of pair potential

        Args:
            rij (np.ndarray (n_pairs, dim)): displacement vectors

        Returns:
            nd.ndarray (n_pairs, 1): pair energy
        """
        rij = R[idx_j] - R[idx_i]
        pair_eps = self.epsilon[atomtype[idx_i], atomtype[idx_j]]
        pair_sigma = self.sigma[atomtype[idx_i], atomtype[idx_j]]

        dij = np.linalg.norm(rij, axis=-1)  # TODO : PBC
        power_6 = np.power(pair_sigma / dij, 6)
        power_12 = np.square(power_6)
        eps = self.epsilon
        sig = self.sigma
        return LJ126.F(dij, eps, sig)
    
    def forces(self, R, atomtype, idx_i, idx_j):
        """
        compute forces of pair potential

        Args:
            rij (np.ndarray (n_paris, dim)): displacement vectors

        Returns:
            np.ndarray (n_pairs, dim): pair forces
        """
        rij = R[idx_j] - R[idx_i]
        pair_eps = self.epsilon[atomtype[idx_i], atomtype[idx_j]]
        pair_sigma = self.sigma[atomtype[idx_i], atomtype[idx_j]]
        dij = np.linalg.norm(rij, axis=-1)
        power_6 = np.power(pair_sigma / dij, 6)
        power_12 = np.square(power_6)

        f = (24 * pair_eps * (2 * power_12 - power_6) / dij**2)[:, None] * rij
        return f