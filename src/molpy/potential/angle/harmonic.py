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
        a = r[angle_idx[:, 0]] - r[angle_idx[:, 1]]
        b = r[angle_idx[:, 2]] - r[angle_idx[:, 1]]

        norm_a = np.linalg.norm(a, axis=-1)
        norm_b = np.linalg.norm(b, axis=-1)

        theta = np.arccos(a * b / norm_a / norm_b)
        dtheta = theta - self.theta0[angle_types]
    
        c = np.cross(a, b)
        f = -dtheta / np.linalg.norm(c)
        F1 = f * (np.cross(b, c) / norm_a**2)
        F3 = f * (np.cross(c, a) / norm_b**2)
        F2 = - (F1 + F3)

        return np.vstack((F1, F2, F3))
