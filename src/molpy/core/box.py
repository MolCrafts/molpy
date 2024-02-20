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

    def __init__(self, lx:int, ly:int, lz:int, xy:int=0, xz:int=0, yz:int=0, origin=np.zeros(3), pbc=np.array([True, True, True])):
        
        self.set_lengths_tilts(lx, ly, lz, xy, xz, yz)
        self._origin = np.array(origin)
        self._pbc = np.array(pbc)

    @classmethod
    def cube(cls, l):
        """init box with cube"""
        return cls.from_lengths(l, l, l)

    @property
    def length(self) -> np.ndarray:
        """box length"""
        return np.diag(self._matrix)
    
    @property
    def tilt(self) -> np.ndarray:
        """box tilt"""
        return np.array([self.xy, self.xz, self.yz])
    
    @property
    def lx(self) -> float:
        """box length in x direction"""
        return self._matrix[0, 0]
    
    @property
    def ly(self) -> float:
        """box length in y direction"""
        return self._matrix[1, 1]
    
    @property
    def lz(self) -> float:
        """box length in z direction"""
        return self._matrix[2, 2]
    
    @property
    def xy(self) -> float:
        """box tilt in xy direction"""
        return self._matrix[0, 1]
    
    @property
    def xz(self) -> float:
        """box tilt in xz direction"""
        return self._matrix[0, 2]
    
    @property
    def yz(self) -> float:
        """box tilt in yz direction"""
        return self._matrix[1, 2]
    
    @property
    def matrix(self) -> np.ndarray:
        """box matrix"""
        return self._matrix

    @classmethod
    def from_lengths(cls, lx, ly, lz):
        """init box with lengths"""
        box = cls(lx, ly, lz)
        return box
    
    @classmethod
    def from_lengths_and_angles(cls, lx, ly, lz, alpha, beta, gamma):
        """init box with lengths and angles"""
        box = cls()
        box.set_lengths_angles(lx, ly, lz, alpha, beta, gamma)
        return box
    
    @classmethod
    def from_box(cls, box: "Box"):
        """init box with another box"""
        new_box = cls()
        new_box.set_matrix(box.get_matrix())
        new_box.set_origin(box._origin)
        return new_box
    
    @classmethod
    def from_matrix(cls, matrix: ArrayLike):
        """init box with matrix"""
        lx, ly, lz = np.diag(matrix)
        box = cls(lx, ly, lz, matrix[0, 1], matrix[0, 2], matrix[1, 2])
        return box

    def get_image(self, r):
        """get image of position vector"""
        r = np.atleast_2d(r)
        reciprocal_r = np.einsum('ij,nj->ni', self.get_inverse(), r)
        return np.floor(reciprocal_r)

    def set_lengths_tilts(
        self,
        lx, ly, lz,
        xy=0, xz=0, yz=0
    ):
        """init or reset the parallelepiped box with lengths and tilts"""
        self._matrix = np.array(
            [
                [lx, xy, xz],
                [0, ly, yz],
                [0, 0, lz],
            ]
        )
        # assert all(np.array([xy, xz, yz]) < np.array([lx, lx, ly])), "tilts must be less than lengths"

    def get_matrix(self) -> np.ndarray:
        """box matrix"""
        return self._matrix
    
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

    def wrap(self, r: ArrayLike):
        """
        shift position vector(s) back to periodic boundary condition box
        """
        r = np.atleast_2d(r)

        reciprocal_r = np.einsum('ij,...j->...i', self.get_inverse(), r)
        shifted_reci_r = reciprocal_r - np.floor(reciprocal_r)
        real_r = np.einsum('ij,...j->...i', self._matrix, shifted_reci_r)
        # real_r = real_r[np.logical_not(self.pbc)] = r[np.logical_not(self.pbc)]
        return real_r
    
    def unwrap(self, r, images):
        r = np.atleast_2d(r)
        images = np.atleast_2d(images)

        return r + np.einsum('ij,kj->ik', images, self._matrix)
    
    @property
    def v1(self) -> np.ndarray:
        """box vector 1"""
        return self._matrix[:, 0]
    
    @property
    def v2(self) -> np.ndarray:
        """box vector 2"""
        return self._matrix[:, 1]
    
    @property
    def v3(self) -> np.ndarray:
        """box vector 3"""
        return self._matrix[:, 2]

    def get_volume(self) -> float:
        """box volume"""
        return np.abs(np.dot(np.cross(self.v1, self.v2), self.v3))
    
    def _diff(self, dr: ArrayLike) -> np.ndarray:
        """difference between two positions"""
        return self.wrap(np.fmod(dr + self.length / 2, self.length)) - self.length / 2

    def diff(self, r1: ArrayLike, r2: ArrayLike) -> np.ndarray:
        """difference between two positions"""
        return self._diff(r1 - r2)
    
    def all_diff(self, r1: ArrayLike, r2: ArrayLike) -> np.ndarray:
        """difference between two positions"""
        pairs = (r1[:, None, :] - r2).reshape((-1, 3))
        return self._diff(pairs)
        
    def self_diff(self, r: ArrayLike) -> np.ndarray:
        """difference between two positions"""
        pairs = r[:, None, :] - r
        paris = pairs[np.triu_indices(len(r), k=1)]
        return self._diff(pairs)
    
    def make_fractional(self, r: ArrayLike) -> np.ndarray:
        """convert position to fractional coordinates"""
        return np.dot(r, self.get_inverse())
    
    def make_absolute(self, r: ArrayLike) -> np.ndarray:
        """convert position to absolute coordinates"""
        return np.dot(r, self._matrix)
    