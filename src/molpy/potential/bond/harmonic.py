import numpy as np

from molpy.potential.base import Potential
from molpy.core.box import Box


def F(r, k, r0):
    return -k * (r - r0)


def E(r, k, r0):
    return 0.5 * k * (r - r0) ** 2


class Harmonic(Potential):

    F = F
    E = E

    def __init__(self):
        super().__init__("harmonic", "bond")

    def _prepare(self, xyz, idx_i, idx_j, type_i, type_j, k, r0):

        dr = self.box.diff(xyz[idx_j], xyz[idx_i])
        r = np.linalg.norm(dr, axis=-1)
        k_ = k[type_i, type_j]
        r0_ = r0[type_i, type_j]
        return r, k_, r0_

    def energy(self, xyz, idx_i, idx_j, type_i, type_j, k, r0):
        return Harmonic.E(*self._prepare(xyz, idx_i, idx_j, type_i, type_j, k, r0))

    def force(self, xyz, idx_i, idx_j, type_i, type_j, k, r0):
        return Harmonic.F(*self._prepare(xyz, idx_i, idx_j, type_i, type_j, k, r0))
