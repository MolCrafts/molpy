from .base import AnglePotential
import numpy as np


class Harmonic(AnglePotential):

    name = "harmonic"

    def __init__(self, k: np.ndarray, theta0: np.ndarray):
        self.k = np.array(k)
        self.theta0 = np.array(theta0)

    @AnglePotential.or_frame
    def calc_energy(
        self, r: np.ndarray, angle_idx: np.ndarray, angle_types: np.ndarray
    ) -> np.ndarray:
        
        theta = np.arccos(np.sum(r[angle_idx[:, 0]] * r[angle_idx[:, 1]], axis=1))

        return 0.5 * self.k[angle_types] * (theta - self.theta0[angle_types]) ** 2

    @AnglePotential.or_frame
    def calc_force(self, r: np.ndarray, angle_idx: np.ndarray, angle_types: np.ndarray):

        r1 = r[angle_idx[:, 0]]
        r2 = r[angle_idx[:, 1]]
        r3 = r[angle_idx[:, 2]]
        a = r2 - r1
        b = r2 - r3
        c = np.cross(a, b)

        norm_a = np.linalg.norm(a, axis=-1)
        norm_b = np.linalg.norm(b, axis=-1)

        ua = a / norm_a[:, None]
        ub = b / norm_b[:, None]

        theta = np.rad2deg(np.arccos(ua @ ub.T))
        dtheta = - self.k[angle_types] * (theta - self.theta0[angle_types])
    
        f = dtheta / np.linalg.norm(c)
        F1 = f * (np.cross(c, a) / norm_a[:, None]**2)
        F3 = f * (np.cross(b, c) / norm_b[:, None]**2)
        F2 = - (F1 + F3)

        per_atom_forces = np.zeros((len(r), 3))
        np.add.at(per_atom_forces, angle_idx[:, 0], F1)
        np.add.at(per_atom_forces, angle_idx[:, 1], F2)
        np.add.at(per_atom_forces, angle_idx[:, 2], F3)
        return per_atom_forces
