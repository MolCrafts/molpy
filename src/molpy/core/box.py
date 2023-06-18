# author: Roy Kid
# contact: lijichen365@126.com
# date: 2022-06-12
# version: 0.0.1

try:
    import jax.numpy as np
except ImportError:
    import numpy as np
from numpy.typing import ArrayLike

class Box:
    """
    A class to represent a box.

    ref: https://scicomp.stackexchange.com/questions/20165/periodic-boundary-conditions-for-triclinic-box
    """
    def __init__(self, xhi=0, yhi=0, zhi=0, xlo=0, ylo=0, zlo=0, xy=0, xz=0, yz=0):
        self.reset(xhi, yhi, zhi, xlo, ylo, zlo, xy, xz, yz)

    def reset(self, xhi:float, yhi:float, zhi:float, xlo=0., ylo=0., zlo=0., xy=0., xz=0., yz=0.):
        self.xhi = xhi
        self.yhi = yhi
        self.zhi = zhi
        self.xlo = xlo
        self.ylo = ylo
        self.zlo = zlo
        self.xy = xy
        self.xz = xz
        self.yz = yz
        lattice_a = np.array([xhi-xlo, 0, 0])
        lattice_b = np.array([xy, yhi-ylo, 0])
        lattice_c = np.array([xz, yz, zhi-zlo])
        self._matrix = np.array([lattice_a, lattice_b, lattice_c]).T
        self.L = self._matrix.diagonal()

    @classmethod
    def from_matrix(cls, matrix):
        xhi = matrix[0, 0]
        yhi = matrix[1, 1]
        zhi = matrix[2, 2]
        xlo = 0.
        ylo = 0.
        zlo = 0.
        xy = matrix[0, 1]
        xz = matrix[0, 2]
        yz = matrix[1, 2]
        return cls(xhi, yhi, zhi, xlo, ylo, zlo, xy, xz, yz)

    @property
    def inv_box(self):
        try:
            return np.linalg.inv(self._matrix)
        except np.linalg.LinAlgError:
            raise ValueError(f"Box matrix {self._matrix} is singular")

    def wrap(self, r):
        """
        shift position vector(s) back to periodic boundary condition box

        Parameters
        ----------
        r : ArrayLike
            (3, ) or (N, 3)

        Returns
        -------
        np.ndarray
            (3, ) or (N, 3)
        """
        r = np.array(r)
        if r.ndim == 2 and r.shape[-1] != 3:
            raise ValueError("r must be (N, 3)")
        elif r.ndim == 1 and r.shape[0] != 3:
            raise ValueError("r must be (3, )")
        elif r.ndim > 2:
            raise ValueError("r must be (N, 3) or (3, )")

        reciprocal_r = np.dot(self.inv_box, r.T)
        shifted_reci_r = reciprocal_r - np.floor(reciprocal_r)
        real_r = np.dot(self._matrix, shifted_reci_r)

        return real_r.T

    def displacement(self, r1, r2):

        dr = r2 - r1
        dr = np.mod(dr + self.L/2, self.L) - self.L/2

        return dr
