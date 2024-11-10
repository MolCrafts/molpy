from .base import PairPotential
import numpy as np


class LJ126(PairPotential):

    name = "lj126/cut"

    def __init__(self, epsilon: np.ndarray, sigma: np.ndarray):
        self.epsilon = np.array(epsilon).reshape(-1, 1)
        self.sigma = np.array(sigma).reshape(-1, 1)

    @PairPotential.or_frame
    def calc_energy(
        self, r: np.ndarray, pair_idx: np.ndarray, pair_types: np.ndarray
    ) -> np.ndarray:
        dr = r[pair_idx[:, 1]] - r[pair_idx[:, 0]]
        dr_norm = np.linalg.norm(dr, axis=1, keepdims=True)
        return (
            4
            * self.epsilon[pair_types]
            * (
                (self.sigma[pair_types] / dr_norm) ** 12
                - (self.sigma[pair_types] / dr_norm) ** 6
            )
        )

    @PairPotential.or_frame
    def calc_force(self, r: np.ndarray, pair_idx: np.ndarray, pair_types: np.ndarray):
        dr = r[pair_idx[:, 1]] - r[pair_idx[:, 0]]
        dr_norm = np.linalg.norm(dr, axis=1, keepdims=True)
        forces = (
            24
            * self.epsilon[pair_types]
            * (
                2 * (self.sigma[pair_types] / dr_norm) ** 12
                - (self.sigma[pair_types] / dr_norm) ** 6
            )
            * dr
            / dr_norm**2
        )
        per_atom_forces = np.zeros((len(r), 3))
        np.add.at(per_atom_forces, pair_idx[:, 0], -forces)
        np.add.at(per_atom_forces, pair_idx[:, 1], forces)
        return per_atom_forces
