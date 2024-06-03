# author: Roy Kid
# contact: lijichen365@126.com
# date: 2023-09-29
# version: 0.0.1

import numpy as np
from abc import ABC, abstractmethod
from molpy.core.region import Region

class Boundary(Region):

    def __init__(self, condition: np.ndarray):
        self.condition = condition

    @abstractmethod
    def wrap(self, r: np.ndarray) -> np.ndarray:
        """
        shift position vector(s) back to periodic boundary condition box

        Args:
            r (ArrayLike): position vector(s), shape (n, 3)

        Returns:
            wrapped_vector: wrapped position vector(s), shape (n, 3)
        """
        ...

    @abstractmethod
    def diff_dr(self, dr: np.ndarray) -> np.ndarray:
        """calculate displacement vector in the box, where `dr` displacement vector between two points"""
        ...

    @abstractmethod
    def diff_pair(self, r1: np.ndarray, r2: np.ndarray) -> np.ndarray:
        """calculate pair-wise displacement vector in the box, where displacement vector dr = r1 - r2"""
        ...

    @abstractmethod
    def diff_all(self, r1: np.ndarray, r2: np.ndarray) -> np.ndarray:
        """calculate all pairs of displacement vector between two set of positions, where dr = r1 - r2. Say r1 and r2 should have shape (n, 3) and (m, 3), and return shape is (n, m, 3)"""
        pairs = r1[:, None, :] - r2
        return self.diff_dr(pairs)

    @abstractmethod
    def diff_self(self, r: np.ndarray) -> np.ndarray:
        """calculate pair_wise interaction of a set of positions. Say r should have shape (n, 3), and return shape is (n, n, 3)"""
        return self.diff_all(r, r)


class Box(Boundary):

    # def __new__(cls, *arg, **kwargs):
    #     return RestrictTriclinicBox()

    def __init__(self, matrix: np.ndarray, pbc: np.ndarray):
        self._matrix = Box.validate_matrix(matrix)
        self._pbc = pbc

    @property
    def pbc(self) -> np.ndarray:
        return self._pbc

    @staticmethod
    def validate_matrix(matrix: np.ndarray) -> np.ndarray:
        assert isinstance(matrix, np.ndarray), "matrix must be np.ndarray"
        assert matrix.shape == (3, 3), "matrix must be (3, 3)"
        assert np.linalg.det(matrix) != 0, "matrix must be non-singular"
        return matrix

    @staticmethod
    def general2restrict(matrix: np.ndarray) -> np.ndarray:
        """
        Convert general triclinc box matrix to restricted triclinic box matrix

        Ref:
            https://docs.lammps.org/Howto_triclinic.html#transformation-from-general-to-restricted-triclinic-boxes

        Args:
            matrix (np.ndarray): (3, 3) general triclinc box matrix

        Returns:
            np.ndarray: (3, 3) restricted triclinc box matrix
        """
        A = matrix[:, 0]
        B = matrix[:, 1]
        C = matrix[:, 2]
        ax = np.linalg.norm(A)
        uA = A / ax
        bx = np.dot(B, uA)
        by = np.linalg.norm(np.cross(uA, B))
        cx = np.dot(C, uA)
        AxB = np.cross(A, B)
        uAxB = AxB / np.linalg.norm(AxB)
        cy = np.dot(C, np.cross(uAxB, uA))
        cz = np.dot(C, uAxB)
        # validation code
        # import numpy.testing as npt
        # gamma = np.arccos(np.dot(A, C) / np.linalg.norm(A) / np.linalg.norm(C))
        # beta = np.arccos(np.dot(A, B) / np.linalg.norm(A) / np.linalg.norm(B))
        # npt.assert_allclose(
        #     bx,
        #     np.linalg.norm(B) * np.cos(gamma),
        #     err_msg=f"{bx} != {np.linalg.norm(B) * np.cos(gamma)}",
        # )
        # npt.assert_allclose(
        #     by,
        #     np.linalg.norm(B) * np.sin(gamma),
        #     err_msg=f"{by} != {np.linalg.norm(B) * np.sin(gamma)}",
        # )
        # npt.assert_allclose(
        #     cx,
        #     np.linalg.norm(C) * np.cos(beta),
        #     err_msg=f"{cx} != {np.linalg.norm(C) * np.cos(beta)}",
        # )
        # npt.assert_allclose(
        #     cy,
        #     (np.dot(B, C) - bx * cx) / by,
        #     err_msg=f"{cy} != {(np.dot(B, C) - bx * cx) / by}",
        # )
        # npt.assert_allclose(
        #     cz,
        #     np.sqrt(np.linalg.norm(C) ** 2 - cx**2 - cy**2),
        #     err_msg=f"{cz} != {np.sqrt(np.linalg.norm(C) ** 2 - cx ** 2 - cy ** 2)}",
        # )
        # TODO: extract origin and direction
        return np.array([[ax, bx, cx], [0, by, cy], [0, 0, cz]])


class RestrictTriclinicBox(Box):

    def __init__(
        self,
        utri_matrix: np.ndarray,
        origin: np.ndarray,
        direction: np.ndarray,
        pbc: np.ndarray,
    ):
        super().__init__(utri_matrix, pbc)
        self.origin = origin
        self.direction = direction

    @property
    def bounds(self) -> np.ndarray:
        """
        Get the bounds of the box. Bounds is a ndim vectors, which can construct a rectangular cuboid wraps the box(not the lengths of edges).

        Returns:
            np.ndarray: bounds of the box
        """
        return np.diag(self._matrix)
    
    def isin(self, xyz:np.ndarray) -> np.ndarray:

        reciprocal_r = np.einsum("ij,...j->...i", self.get_inverse(), xyz)
        result = np.logical_and(np.all(reciprocal_r >= 0, axis=-1), np.all(reciprocal_r <= 1, axis=-1))
        result = np.logical_or(result, self._pbc)
        return result

    def get_inverse(self) -> np.ndarray:
        """inverse of box matrix"""
        return np.linalg.inv(self._matrix)

    def wrap(self, r: np.ndarray) -> np.ndarray:

        reciprocal_r = np.einsum("ij,...j->...i", self.get_inverse(), r)
        shifted_reci_r = reciprocal_r - np.floor(reciprocal_r)
        real_r = np.einsum("ij,...j->...i", self._matrix, shifted_reci_r)
        not_pbc = np.logical_not(self._pbc)
        real_r[..., not_pbc] = r[..., not_pbc]
        return real_r

    def diff_dr(self, dr: np.ndarray) -> np.ndarray:

        remainder = np.remainder(dr + self.bounds / 2, self.bounds)
        return self.wrap(remainder) - self.bounds / 2

    def diff_pair(self, r1: np.ndarray, r2: np.ndarray) -> np.ndarray:
        return self.diff_dr(r1 - r2)

    def diff_all(self, r1: np.ndarray, r2: np.ndarray) -> np.ndarray:
        pairs = r1[:, None, :] - r2
        return self.diff_dr(pairs)

    def diff_self(self, r: np.ndarray) -> np.ndarray:
        return self.diff_all(r, r)

    def make_fractional(self, r: np.ndarray) -> np.ndarray:
        return np.dot(r, self.get_inverse())

    def make_absolute(self, r: np.ndarray) -> np.ndarray:
        return np.dot(r, self._matrix)


class GeneralTriclinicBox(Box):

    def __init__(self, matrix: np.ndarray, pbc: np.ndarray):
        super().__init__(matrix, pbc)


class OrthogonalBox(RestrictTriclinicBox):

    def __init__(
        self,
        lengths: np.ndarray,
        origin: np.ndarray = np.zeros(3),
        direction: np.ndarray = np.eye(3),
        pbc: np.ndarray = np.array([True, True, True])
    ):
        super().__init__(np.diag(lengths), origin, direction, pbc)


class Free(Boundary):

    def __init__(self):
        super().__init__(np.array([False, False, False]))

    def isin(self, xyz:np.ndarray) -> np.ndarray:
        return np.ones(xyz.shape[0], dtype=bool)

    def wrap(self, r: np.ndarray) -> np.ndarray:
        return r

    def diff_dr(self, dr: np.ndarray) -> np.ndarray:
        return dr

    def diff_pair(self, r1: np.ndarray, r2: np.ndarray) -> np.ndarray:
        return r1 - r2

    def diff_all(self, r1: np.ndarray, r2: np.ndarray) -> np.ndarray:
        return r1[:, None, :] - r2

    def diff_self(self, r: np.ndarray) -> np.ndarray:
        return r[:, None, :] - r
    
    @property
    def lengths(self):
        return np.zeros(3)
    
    @property
    def angles(self):
        return np.array([90, 90, 90])


# def get_matrix_from_length_angle(a, b, c, alpha, beta, gamma):
#     """
#     get restricted triclinic box matrix from lengths and angles

#     Args:
#         a (float): lattice constant a
#         b (float): lattice constant b
#         c (float): lattice constant c
#         alpha (float): angle between b and c, degree
#         beta (float): angle between a and c, degree
#         gamma (float): angle between a and b, degree

#     Returns:
#         np.array: restricted triclinic box matrix
#     """
#     lx = a
#     ly = b * np.sin(gamma)


#     xy = b * np.cos(gamma)
#     xz = c * np.cos(beta)
#     yz = (b * c * np.cos(alpha) - xy * xz) / ly

#     lz = np.sqrt(c ** 2 - xz ** 2 - yz ** 2)

#     return np.array([[lx, xy, xz], [0, ly, yz], [0, 0, lz]])


# class Box:
#     """
#     A class to represent a box.

#     ref: https://scicomp.stackexchange.com/questions/20165/periodic-boundary-conditions-for-triclinic-box
#          https://docs.lammps.org/Howto_triclinic.html
#     """

#     def __init__(
#         self,
#         matrix: ArrayLike | None = None,
#         origin=np.zeros(3),
#         direction=np.eye(3),
#         pbc: bool | np.ndarray = np.array([True, True, True]),
#     ):
#         """
#         init box with lengths and tilts.

#         Examples:
#         ```python
#         Box()  # free space box
#         Box([10, 10, 10])  # cube box
#         Box([10, 10, 10], [1, 2, 3])  # parallelepiped box


#         Args:
#             lengths (ArrayLike, optional): _description_. Defaults to [0, 0, 0].
#             tilts (ArrayLike, optional): _description_. Defaults to [0, 0, 0].
#             origin (_type_, optional): _description_. Defaults to np.zeros(3).
#             pbc (_type_, optional): _description_. Defaults to np.array([True, True, True]).
#         """
#         if matrix is None or np.all(matrix == 0):
#             self._matrix = None
#         else:
#             matrix = np.asarray(matrix)
#             if matrix.shape == (3,):
#                 matrix = np.diag(matrix)
#             self._matrix = general2restrict(validate_matrix(matrix))

#         self._origin = np.asarray(origin)
#         assert self._origin.shape == (
#             3,
#         ), f"origin.shape must be (3, ) rather than {self._origin.shape}"

#         self._direction = np.asarray(direction)

#         if isinstance(pbc, (bool, np.bool_)):
#             self._pbc = np.array([pbc, pbc, pbc])
#         else:
#             self._pbc = np.asarray(pbc, dtype=bool)

#     @classmethod
#     def cube(cls, l):
#         """init box with cube"""
#         return cls.from_lengths(l, l, l)

#     @classmethod
#     def free(cls):
#         return cls(np.diag([1, 1, 1]), origin=np.zeros(3,), pbc=False)

#     @property
#     def bounds(self) -> np.ndarray:
#         """
#         Get the bounds of the box. Definition of size of orthorhombic box is the lengths of the box; for triclinic box, it is the bounds of the box, not the lengths of edges.

#         Returns:
#             np.ndarray: bounds of the box
#         """
#         return np.diag(self._matrix)

#     @property
#     def size(self) -> np.ndarray:
#         """
#         Get size of box, same as bounds.

#         Returns:
#             np.ndarray: size of box
#         """
#         return self.bounds

#     @property
#     def lengths(self) -> np.ndarray:
#         """
#         Get the length of box edges.

#         Returns:
#             np.ndarray: length of the box edges
#         """
#         return np.linalg.norm(self._matrix, axis=0)

#     @property
#     def angles(self) -> np.ndarray:
#         a = self.lx
#         b = np.sqrt(self.ly**2 + self.xy**2)
#         c = np.sqrt(self.lz**2 + self.xz**2 + self.yz**2)
#         alpha = np.arccos((self.xy*self.xz+self.ly*self.yz)/(b*c))
#         beta = np.arccos(self.xz/c)
#         gamma = np.arccos(self.xy/b)
#         return np.rad2deg(np.array([alpha, beta, gamma]))

#     @property
#     def inv(self) -> np.ndarray:
#         """inverse of box matrix"""
#         return np.linalg.inv(self._matrix)

#     @property
#     def tilts(self) -> np.ndarray:
#         """box tilt"""
#         return np.array([self.xy, self.xz, self.yz])

#     @property
#     def lx(self) -> float:
#         """box lengths in x direction"""
#         return self._matrix[0, 0]

#     @property
#     def ly(self) -> float:
#         """box lengths in y direction"""
#         return self._matrix[1, 1]

#     @property
#     def lz(self) -> float:
#         """box lengths in z direction"""
#         return self._matrix[2, 2]

#     @property
#     def xy(self) -> float:
#         """box tilt in xy direction"""
#         return self._matrix[0, 1]

#     @property
#     def xz(self) -> float:
#         """box tilt in xz direction"""
#         return self._matrix[0, 2]

#     @property
#     def yz(self) -> float:
#         """box tilt in yz direction"""
#         return self._matrix[1, 2]

#     @property
#     def matrix(self) -> np.ndarray:
#         """get restricted triclinic box matrix"""
#         return self._matrix

#     @classmethod
#     def from_lengths(cls, lx:float, ly:float, lz:float):
#         """init periodic orthogonal box with lengths"""
#         box = cls([lx, ly, lz])
#         return box

#     @classmethod
#     def from_lengths_and_angles(cls, a:float, b:float, c:float, alpha:float, beta:float, gamma:float):
#         """init periodic box with lengths and angles"""
#         return cls(get_matrix_from_length_angle(a, b, c, alpha, beta, gamma))

#     @classmethod
#     def from_lengths_tilts(cls, lx:float, ly:float, lz:float, xy:float, xz:float, yz:float):
#         """init or reset the parallelepiped box with lengths and tilts"""
#         return cls(np.array(
#             [
#                 [lx, xy, xz],
#                 [0, ly, yz],
#                 [0, 0, lz],
#             ]
#         ))

#     def get_image(self, r):
#         """get image of position vector"""
#         r = np.atleast_2d(r)
#         reciprocal_r = np.einsum("ij,nj->ni", self.get_inverse(), r)
#         return np.floor(reciprocal_r)

#     def set_origin(self, origin: ArrayLike):
#         """init or reset the parallelepiped box with origin"""
#         self._origin = np.array(origin)
#         assert self._origin.shape == (3,), "origin must be (3, )"

#     def get_inverse(self) -> np.ndarray:
#         """inverse of box matrix"""
#         return np.linalg.inv(self._matrix)

#     @property
#     def pbc(self) -> np.ndarray:
#         """periodic boundary condition"""
#         return self._pbc

#     @pbc.setter
#     def pbc(self, value):
#         self._pbc = np.array(value)

#     def wrap(self, r: ArrayLike) -> np.ndarray:
#         """
#         shift position vector(s) back to periodic boundary condition box

#         Args:
#             r (ArrayLike): position vector(s), shape (n, 3)

#         Returns:
#             wrapped_vector: wrapped position vector(s), shape (n, 3)
#         """
#         r = np.atleast_2d(r)

#         reciprocal_r = np.einsum("ij,...j->...i", self.get_inverse(), r)
#         shifted_reci_r = reciprocal_r - np.floor(reciprocal_r)
#         real_r = np.einsum("ij,...j->...i", self._matrix, shifted_reci_r)
#         not_pbc = np.logical_not(self._pbc)
#         real_r[..., not_pbc] = r[..., not_pbc]
#         return real_r

#     def unwrap(self, r, images):
#         r = np.atleast_2d(r)
#         images = np.atleast_2d(images)

#         return r + np.einsum("ij,kj->ik", images, self._matrix)

#     @property
#     def a(self) -> np.ndarray:
#         """box vector 1"""
#         return self._matrix[:, 0]

#     @property
#     def b(self) -> np.ndarray:
#         """box vector 2"""
#         return self._matrix[:, 1]

#     @property
#     def c(self) -> np.ndarray:
#         """box vector 3"""
#         return self._matrix[:, 2]

#     def get_volume(self) -> float:
#         """box volume"""
#         return np.abs(np.dot(np.cross(self.a, self.b), self.c))

#     def diff_dr(self, dr: ArrayLike) -> np.ndarray:
#         """calculate distance in the box, where `dr` displacement vector between two points"""
#         if all(self._pbc == False):
#             return dr
#         if self._matrix is None:  # free space
#             return dr
#         # apply pbc as mask
#         remainder = np.remainder(dr + self.bounds / 2, self.bounds)
#         return self.wrap(remainder) - self.bounds / 2

#     def diff(self, r1: ArrayLike, r2: ArrayLike) -> np.ndarray:
#         """calculate distance in the box, where displacement vector dr = r1 - r2"""
#         return self.diff_dr(r1 - r2)

#     def diff_all(self, r1: ArrayLike, r2: ArrayLike) -> np.ndarray:
#         """calculate pairs distances between two set of positions, where dr = r1 - r2. Say r1 and r2 should have shape (n, 3) and (m, 3), and return shape is (n, m, 3)"""
#         pairs = r1[:, None, :] - r2
#         return self.diff_dr(pairs)

#     def diff_self(self, r: ArrayLike) -> np.ndarray:
#         """calculate pair_wise interaction of a set of positions. Say r should have shape (n, 3), and return shape is (n, n, 3)"""
#         return self.diff_all(r, r)

#     def make_fractional(self, r: ArrayLike) -> np.ndarray:
#         """convert position to fractional coordinates"""
#         return np.dot(r, self.get_inverse())

#     def make_absolute(self, r: ArrayLike) -> np.ndarray:
#         """convert position to absolute coordinates"""
#         return np.dot(r, self._matrix)
