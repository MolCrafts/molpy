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

def e_lj126(rij:np.ndarray, eps:np.ndarray, sig:np.ndarray):

    power_6 = np.power(sig / rij, 6)
    power_12 = np.square(power_6)

    return 4 * eps * (power_12 - power_6)

def f_lj126(rij:np.ndarray, eps:np.ndarray, sig:np.ndarray):

    power_6 = np.power(sig / rij, 6)
    power_12 = np.square(power_6)

    return (24 * eps * (2 * power_12 - power_6) / np.square(rij))[:, None] * rij

class LJ126(Potential):

    E = e_lj126
    F = f_lj126

    def __init__(self, epsilon, sigma, cutoff):
        super().__init__('LJ126', 'pair')
        self.epsilon = epsilon
        self.sigma = sigma
        self.cutoff = cutoff

    def energy(self, R, atomtype, idx_i, idx_j, offset):
        """
        compute energy of pair potential

        Args:
            rij (np.ndarray (n_pairs, dim)): displacement vectors

        Returns:
            nd.ndarray (n_pairs, 1): pair energy
        """
        r_ij = R[idx_j] - R[idx_i] + offset
        eps = self.epsilon[atomtype[idx_i], atomtype[idx_j]]
        sigma = self.sigma[atomtype[idx_i], atomtype[idx_j]]

        d_ij = np.linalg.norm(r_ij, axis=-1, keepdims=True)
        cutoff_mask = d_ij < self.cutoff
        energy = LJ126.F(d_ij, eps, sigma)
        return energy * cutoff_mask
    
    def forces(self, R, atomtype, idx_i, idx_j, offset):
        """
        compute forces of pair potential

        Args:
            rij (np.ndarray (n_paris, dim)): displacement vectors

        Returns:
            np.ndarray (n_pairs, dim): pair forces
        """
        r_ij = R[idx_j] - R[idx_i] + offset
        eps = self.epsilon[atomtype[idx_i], atomtype[idx_j]]
        sigma = self.sigma[atomtype[idx_i], atomtype[idx_j]]

        d_ij = np.linalg.norm(r_ij, axis=-1, keepdims=True)
        cutoff_mask = d_ij < self.cutoff

        return LJ126.F(d_ij, eps, sigma) * cutoff_mask