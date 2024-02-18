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

        R = input.atoms[mp.Alias.xyz]
        idx_i = input[mp.Alias.idx_i]
        idx_j = input[mp.Alias.idx_j]

        energy = self.energy(R, idx_i, idx_j)
        forces = self.forces(R, idx_i, idx_j)
        input.atoms[mp.Alias.energy] += energy
        input.atoms[mp.Alias.forces] += forces
        return input


    def energy(self, R, idx_i, idx_j):

        Ri = R[idx_i]
        Rj = R[idx_j]

        dij = np.linalg.norm(Rj - Ri, axis=-1, keepdims=True)
        power_6 = np.power(self.sigma / dij, 6)
        power_12 = np.square(power_6)

        energy = 4 * self.epsilon * (power_12 - power_6)
        energy = segment_sum(energy, idx_i, R.shape[0])
        return energy
    
    def forces(self, R, idx_i, idx_j):

        Ri = R[idx_i]
        Rj = R[idx_j]

        rij = Rj - Ri
        dij = np.linalg.norm(rij, axis=-1, keepdims=True)

        power_6 = np.power(self.sigma / dij, 6)
        power_12 = np.square(power_6)

        f = 24 * self.epsilon * (2 * power_12 - power_6) / dij**2 * rij
        f = segment_sum(f, idx_i, R.shape[0])

        return f