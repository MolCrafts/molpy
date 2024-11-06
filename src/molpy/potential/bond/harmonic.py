from .base import BondPotential
import numpy as np


class Harmonic(BondPotential):

    name = "harmonic"

    def __init__(self, k: np.ndarray, r0: np.ndarray):
        super().__init__()
        self.k = k
        self.r0 = r0

    def calc_energy(
        self, r: np.ndarray, bond_idx: np.ndarray, bond_types: np.ndarray
    ) -> np.ndarray:
        dr = r[bond_idx[:, 1]] - r[bond_idx[:, 0]]
        dr_norm = np.linalg.norm(dr, axis=1)
        return 0.5 * self.k[bond_types] * (dr_norm - self.r0[bond_types]) ** 2

    def calc_force(self, r: np.ndarray, bond_idx: np.ndarray, bond_types: np.ndarray):
        dr = r[bond_idx[:, 1]] - r[bond_idx[:, 0]]
        dr_norm = np.linalg.norm(dr, axis=1)
        return -self.k * (dr_norm - self.r0)[bond_types] * dr / dr_norm[bond_types]
