# author: Roy Kid
# contact: lijichen365@126.com
# date: 2024-02-18
# version: 0.0.1

import numpy as np
import molpy as mp
from molpy.potential.base import Potential
from molpy.core.space import Free

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

    def forward(self, frame, output):

        box = getattr(frame, "box", Free())

        xyz = frame.atoms['xyz']
        pairs = frame.topology.pairs
        idx_i = pairs[:, 0]
        idx_j = pairs[:, 1]
        type_i = frame.atoms['atomtype'][idx_i]
        type_j = frame.atoms['atomtype'][idx_j]

        dr = box.diff(xyz[idx_j], xyz[idx_i])
        cutoff_mask = dr < self.cutoff
        r = np.linalg.norm(dr, axis=-1)[cutoff_mask]

        eps = self.epsilon[type_i, type_j]
        sig = self.sigma[type_i, type_j]

        energy = LJ126.E(r, eps, sig)
        output['lj126_energy'] = segment_sum(energy, idx_i, frame.n_atoms)
        output['per_atom_lj126_energy'] = energy / 2
        output['lj126_force'] = np.zeros((frame.n_atoms, 3))
        output['lj126_force'][idx_i] = LJ126.F(r, eps, sig)

        return frame, output
