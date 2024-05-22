# author: Roy Kid
# contact: lijichen365@126.com
# date: 2023-09-29
# version: 0.0.1

import numpy as np
from numpy.typing import ArrayLike


class Box:
    """
    A class to represent a box.

    ref: https://scicomp.stackexchange.com/questions/20165/periodic-boundary-conditions-for-triclinic-box
         https://docs.lammps.org/Howto_triclinic.html
    """

    def __init__(
        self,
        lengths: ArrayLike = [0, 0, 0],
        tilts: ArrayLike = [0, 0, 0],
        origin=np.zeros(3),
        pbc: bool | np.ndarray = np.array([True, True, True]),
    ):
        """
        init box with lengths and tilts.

        Examples:
        ```python
        Box()  # free space box
        Box([10, 10, 10])  # cube box
        Box([10, 10, 10], [1, 2, 3])  # parallelepiped box


        Args:
            lengths (ArrayLike, optional): _description_. Defaults to [0, 0, 0].
            tilts (ArrayLike, optional): _description_. Defaults to [0, 0, 0].
            origin (_type_, optional): _description_. Defaults to np.zeros(3).
            pbc (_type_, optional): _description_. Defaults to np.array([True, True, True]).
        """
        lengths = np.asarray(lengths)
        tilts = np.asarray(tilts)
        if lengths.shape == (3,) and tilts.shape == (3,):
            self.set_lengths_tilts(lengths, tilts)
        else:
            raise ValueError("lengths and tilts must be (3, )")
        self._origin = np.array(origin)
        if isinstance(pbc, (bool, np.bool_)):
            self._pbc = np.array([pbc, pbc, pbc])
        else:
            self._pbc = np.array(pbc, dtype=bool)

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
        box = cls([lx, ly, lz])
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
    def from_matrix(cls, matrix: ArrayLike, pbc: ArrayLike = [True, True, True]):
        """init box with matrix"""
        box = cls()
        A = matrix[:, 0]
        B = matrix[:, 1]
        C = matrix[:, 2]
        gamma = np.arccos(np.dot(A, C) / np.linalg.norm(A) / np.linalg.norm(C))
        beta = np.arccos(np.dot(A, B) / np.linalg.norm(A) / np.linalg.norm(B))
        ax = np.linalg.norm(A)
        uA = A / ax
        bx = np.dot(B, uA)
        import numpy.testing as npt

        npt.assert_allclose(
            bx,
            np.linalg.norm(B) * np.cos(gamma),
            err_msg=f"{bx} != {np.linalg.norm(B) * np.cos(gamma)}",
        )
        by = np.linalg.norm(np.cross(uA, B))
        npt.assert_allclose(
            by,
            np.linalg.norm(B) * np.sin(gamma),
            err_msg=f"{by} != {np.linalg.norm(B) * np.sin(gamma)}",
        )
        cx = np.dot(C, uA)
        npt.assert_allclose(
            cx,
            np.linalg.norm(C) * np.cos(beta),
            err_msg=f"{cx} != {np.linalg.norm(C) * np.cos(beta)}",
        )
        AxB = np.cross(A, B)
        uAxB = AxB / np.linalg.norm(AxB)
        cy = np.dot(C, np.cross(uAxB, uA))
        npt.assert_allclose(
            cy,
            (np.dot(B, C) - bx * cx) / by,
            err_msg=f"{cy} != {(np.dot(B, C) - bx * cx) / by}",
        )
        cz = np.dot(C, uAxB)
        npt.assert_allclose(
            cz,
            np.sqrt(np.linalg.norm(C) ** 2 - cx**2 - cy**2),
            err_msg=f"{cz} != {np.sqrt(np.linalg.norm(C) ** 2 - cx ** 2 - cy ** 2)}",
        )
        box._matrix = np.array([[ax, bx, cx], [0, by, cy], [0, 0, cz]])
        return box

    def get_image(self, r):
        """get image of position vector"""
        r = np.atleast_2d(r)
        reciprocal_r = np.einsum("ij,nj->ni", self.get_inverse(), r)
        return np.floor(reciprocal_r)

    def set_lengths_tilts(self, lengths, tilts):
        """init or reset the parallelepiped box with lengths and tilts"""
        lx, ly, lz = lengths
        xy, xz, yz = tilts
        self._matrix = np.array(
            [
                [lx, xy, xz],
                [0, ly, yz],
                [0, 0, lz],
            ]
        )
        # assert all(np.array([xy, xz, yz]) < np.array([lx, lx, ly])), "tilts must be less than lengths"

    def set_lengths_angles(self, lx, ly, lz, alpha, beta, gamma):

        # Handle orthorhombic cells separately to avoid rounding errors
        eps = 2 * np.spacing(90.0, dtype=np.float64)  # around 1.4e-14
        # alpha
        if abs(abs(alpha) - 90) < eps:
            cos_alpha = 0.0
        else:
            cos_alpha = np.cos(alpha * np.pi / 180.0)
        # beta
        if abs(abs(beta) - 90) < eps:
            cos_beta = 0.0
        else:
            cos_beta = np.cos(beta * np.pi / 180.0)
        # gamma
        if abs(gamma - 90) < eps:
            cos_gamma = 0.0
            sin_gamma = 1.0
        elif abs(gamma + 90) < eps:
            cos_gamma = 0.0
            sin_gamma = -1.0
        else:
            cos_gamma = np.cos(gamma * np.pi / 180.0)
            sin_gamma = np.sin(gamma * np.pi / 180.0)

        # Build the cell vectors
        va = lx * np.array([1, 0, 0])
        vb = ly * np.array([cos_gamma, sin_gamma, 0])
        cx = cos_beta
        cy = (cos_alpha - cos_beta * cos_gamma) / sin_gamma
        cz_sqr = 1.0 - cx * cx - cy * cy
        assert cz_sqr >= 0
        cz = np.sqrt(cz_sqr)
        vc = lz * np.array([cx, cy, cz])

        # Convert to the Cartesian x,y,z-system
        abc = np.vstack((va, vb, vc))
        ad = np.array([0, 0, 1.0])
        ab_normal = np.array([0, 0, 1.0])
        Z = ab_normal
        X = ad - np.dot(ad, Z) * Z
        X /= np.linalg.norm(X)
        Y = np.cross(Z, X)
        T = np.vstack((X, Y, Z))
        cell = np.dot(abc, T)

        return cell

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

        Args:
            r (ArrayLike): position vector(s), shape (n, 3)

        Returns:
            wrapped_vector: wrapped position vector(s), shape (n, 3)
        """
        r = np.atleast_2d(r)

        reciprocal_r = np.einsum("ij,...j->...i", self.get_inverse(), r)
        shifted_reci_r = reciprocal_r - np.floor(reciprocal_r)
        real_r = np.einsum("ij,...j->...i", self._matrix, shifted_reci_r)
        not_pbc = np.logical_not(self._pbc)
        real_r[..., not_pbc] = r[..., not_pbc]
        return real_r

    def unwrap(self, r, images):
        r = np.atleast_2d(r)
        images = np.atleast_2d(images)

        return r + np.einsum("ij,kj->ik", images, self._matrix)

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

    def diff_dr(self, dr: ArrayLike) -> np.ndarray:
        """calculate distance in the box, where `dr` displacement vector between two points"""
        if all(self._pbc == False):
            return dr

        # apply pbc as mask
        remainder = np.remainder(dr + self.length / 2, self.length)
        return self.wrap(remainder) - self.length / 2

    def diff(self, r1: ArrayLike, r2: ArrayLike) -> np.ndarray:
        """calculate distance in the box, where displacement vector dr = r1 - r2"""
        return self.diff_dr(r1 - r2)

    def diff_all(self, r1: ArrayLike, r2: ArrayLike) -> np.ndarray:
        """calculate pairs distances between two set of positions, where dr = r1 - r2. Say r1 and r2 should have shape (n, 3) and (m, 3), and return shape is (n, m, 3)"""
        pairs = r1[:, None, :] - r2
        return self.diff_dr(pairs)

    def diff_self(self, r: ArrayLike) -> np.ndarray:
        """calculate pair_wise interaction of a set of positions. Say r should have shape (n, 3), and return shape is (n, n, 3)"""
        return self.diff_all(r, r)

    def make_fractional(self, r: ArrayLike) -> np.ndarray:
        """convert position to fractional coordinates"""
        return np.dot(r, self.get_inverse())

    def make_absolute(self, r: ArrayLike) -> np.ndarray:
        """convert position to absolute coordinates"""
        return np.dot(r, self._matrix)
