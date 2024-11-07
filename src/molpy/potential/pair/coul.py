from .base import PairPotential
import numpy as np


class CoulCut(PairPotential):

    name = "coul/cut"

    @PairPotential.or_frame("charge")
    def calc_energy(
        self,
        r: np.ndarray,
        pair_idx: np.ndarray,
        pair_types: np.ndarray,
        charge: np.ndarray,
    ) -> np.ndarray:
        dr = r[pair_idx[:, 1]] - r[pair_idx[:, 0]]
        dr_norm = np.linalg.norm(dr, axis=1)
        return np.sum(charge[pair_idx[:, 0]] * charge[pair_idx[:, 1]] / dr_norm)

    @PairPotential.or_frame("charge")
    def calc_force(
        self,
        r: np.ndarray,
        pair_idx: np.ndarray,
        pair_types: np.ndarray,
        charges: np.ndarray,
    ):
        dr = r[pair_idx[:, 1]] - r[pair_idx[:, 0]]
        dr_norm = np.linalg.norm(dr, axis=1)
        return np.sum(
            charges[pair_idx[:, 0]] * charges[pair_idx[:, 1]] / dr_norm**3 * dr, axis=1
        )
