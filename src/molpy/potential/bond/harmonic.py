from .base import BondPotential
import numpy as np


class Harmonic(BondPotential):

    name = "harmonic"

    def __init__(self, k: np.ndarray, r0: np.ndarray):
        self.k = np.array(k).reshape(-1, 1)
        self.r0 = np.array(r0).reshape(-1, 1)

    @BondPotential.or_frame
    def calc_energy(
        self, r: np.ndarray, bond_idx: np.ndarray, bond_types: np.ndarray
    ) -> np.ndarray:
        dr = r[bond_idx[:, 1]] - r[bond_idx[:, 0]]
        dr_norm = np.linalg.norm(dr, axis=1, keepdims=True)
        return 0.5 * self.k[bond_types] * (dr_norm - self.r0[bond_types]) ** 2

    @BondPotential.or_frame
    def calc_force(self, r: np.ndarray, bond_idx: np.ndarray, bond_types: np.ndarray):
        dr = r[bond_idx[:, 1]] - r[bond_idx[:, 0]]
        dr_norm = np.linalg.norm(dr, axis=1, keepdims=True)
        forces = -self.k[bond_types] * (dr_norm - self.r0[bond_types]) * dr / dr_norm
        per_atom_forces = np.zeros((len(r), 3))
        np.add.at(per_atom_forces, bond_idx[:, 0], -forces)
        np.add.at(per_atom_forces, bond_idx[:, 1], forces)
        return per_atom_forces
