# author: Roy Kid
# contact: lijichen365@126.com
# date: 2023-09-29
# version: 0.0.1

import numpy as np
from numpy.typing import ArrayLike
from typing import Optional

__all__ = ["Box"]


class Box:
    """
    A class to represent a box.

    ref: https://scicomp.stackexchange.com/questions/20165/periodic-boundary-conditions-for-triclinic-box
         https://docs.lammps.org/Howto_triclinic.html
    """

    def __init__(self, pbc=np.array([1, 1, 1]), matrix: Optional[ArrayLike | 'Box'] = None, origin: Optional[ArrayLike] = None):
        if isinstance(matrix, Box):
            self._matrix = matrix.get_matrix()
        elif matrix is None:
            self._matrix = np.eye(3)
        else:
            trail = np.array(matrix)
            assert trail.shape == (3, 3), "matrix must be (3, 3)"
            self._matrix = trail
        self._origin = np.array(origin)
        self._pbc = np.array(pbc)

    def set_lengths_angles(
        self,
        lengths: ArrayLike,
        angles: ArrayLike = (90, 90, 90),
    ):
        """init or reset the parallelepiped box with lengths and angles"""
        a, b, c = np.array(lengths)
        alpha, beta, gamma = np.radians(angles)
        lx = a
        xy = b * np.cos(gamma)
        xz = c * np.cos(beta)
        ly = np.sqrt(b**2 - xy**2)
        yz = (b * c * np.cos(alpha) - xy * xz) / ly
        lz = np.sqrt(c**2 - xz**2 - yz**2)

        self._matrix = np.array(
            [
                [lx, xy, xz],
                [0, ly, yz],
                [0, 0, lz],
            ]
        )

    def set_matrix(self, matrix: ArrayLike):
        """init or reset the parallelepiped box with matrix"""
        self._matrix = np.array(matrix)
        assert self._matrix.shape == (3, 3), "matrix must be (3, 3)"

    def set_origin(self, origin: ArrayLike):
        """init or reset the parallelepiped box with origin"""
        self._origin = np.array(origin)
        assert self._origin.shape == (3,), "origin must be (3, )"

    def get_inverse(self) -> np.ndarray:
        """inverse of box matrix"""
        try:
            return np.linalg.inv(self._matrix)
        except np.linalg.LinAlgError:
            raise ValueError(f"Box matrix {self._matrix} is singular")

    def get_matrix(self) -> np.ndarray:
        """box matrix"""
        return self._matrix
    
    @property
    def matrix(self) -> np.ndarray:
        """box matrix"""
        return self.get_matrix()
    
    @property
    def pbc(self) -> np.ndarray:
        """periodic boundary condition"""
        return self._pbc
    
    @pbc.setter
    def pbc(self, value):
        self._pbc = np.array(value)

    def get_tilts(self) -> np.ndarray:
        """box tilt"""
        xy = self._matrix[0, 1]
        xz = self._matrix[0, 2]
        yz = self._matrix[1, 2]
        return np.array([xy, xz, yz])

    def get_angles(self) -> np.ndarray:
        """box angles"""
        xy, xz, yz = self.get_tilts()
        ly = self._matrix[1, 1]
        lz = self._matrix[2, 2]
        alpha = self._matrix[0, 0]
        beta = np.sqrt(ly**2 + xy**2)
        gamma = np.sqrt(lz**2 + xz**2 + yz**2)
        return np.array([alpha, beta, gamma])

    def wrap(self, r: ArrayLike):
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

        reciprocal_r = np.dot(self.get_inverse(), r.T)
        shifted_reci_r = reciprocal_r - np.floor(reciprocal_r)
        real_r = np.dot(self._matrix, shifted_reci_r)

        return real_r.T

    def get_volume(self) -> float:
        """box volume"""
        return np.linalg.det(self._matrix)

    def diff(self, r1: ArrayLike, r2: ArrayLike) -> np.ndarray:
        """difference between two positions"""
        return self.wrap(np.array(r1) - np.array(r2))

    def dist(self, r1: ArrayLike, r2: ArrayLike) -> np.ndarray:
        """distance between two positions"""
        return np.linalg.norm(self.diff(r1, r2), axis=-1)

    def make_fractional(self, r: ArrayLike) -> np.ndarray:
        """convert position to fractional coordinates"""
        return np.dot(r, self.get_inverse())
    
    def make_absolute(self, r: ArrayLike) -> np.ndarray:
        """convert position to absolute coordinates"""
        return np.dot(r, self._matrix)
    