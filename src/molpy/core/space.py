# author: Roy Kid
# contact: lijichen365@126.com
# date: 2023-09-29
# version: 0.0.1

import numpy as np
from abc import ABC, abstractmethod
from .region import Region, Boundary
from enum import Enum


class Box(Region, Boundary):

    class Style(Enum):
        FREE = 0
        ORTHOGONAL = 1
        TRICLINIC = 2

    def __init__(
        self,
        matrix: np.ndarray | None = None,
        pbc: np.ndarray = np.zeros(3, dtype=bool),
    ):
        if matrix is None or np.all(matrix == 0):
            self._matrix = np.zeros((3, 3))
        else:
            _matrix = np.asarray(matrix)
            if _matrix.shape == (3, ):
                _matrix = np.diag(_matrix)
            self._matrix = Box.check_matrix(_matrix)
        self._pbc = pbc
        self._style = self.calc_style_from_matrix(self._matrix)

    def __repr__(self):
        match self.style:
            case Box.Style.FREE:
                return f"<Box: Free>"
            case Box.Style.ORTHOGONAL:
                return f"<Box: Orthogonal: {self.lengths}>"
            case Box.Style.TRICLINIC:
                return f"<Box: Triclinic: {self._matrix}>"
            
    @property
    def xlo(self) -> float:
        return 0
    
    @property
    def xhi(self) -> float:
        return self._matrix[0, 0]
    
    @property
    def ylo(self) -> float:
        return 0
    
    @property
    def yhi(self) -> float:
        return self._matrix[1, 1]
    
    @property
    def zlo(self) -> float:
        return 0
    
    @property
    def zhi(self) -> float:
        return self._matrix[2, 2]

    @property
    def style(self) -> Style:
        return self.calc_style_from_matrix(self._matrix)

    @property
    def pbc(self) -> np.ndarray:
        return self._pbc
    
    @property
    def matrix(self) -> np.ndarray:
        return self._matrix

    @staticmethod
    def check_matrix(matrix: np.ndarray) -> np.ndarray:
        assert isinstance(matrix, np.ndarray), "matrix must be np.ndarray"
        assert matrix.shape == (3, 3), "matrix must be (3, 3)"
        assert np.linalg.det(matrix) != 0, "matrix must be non-singular"
        return matrix

    @property
    def lengths(self) -> np.ndarray:
        match self.style:
            case Box.Style.FREE:
                return np.zeros(3)
            case Box.Style.ORTHOGONAL | Box.Style.TRICLINIC:
                return self.calc_lengths_from_matrix(self._matrix)


    @property
    def angles(self) -> np.ndarray:
        return self.calc_angles_from_matrix(self._matrix)

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

    @staticmethod
    def calc_matrix_from_lengths_angles(
        lengths: np.ndarray, angles: np.ndarray
    ) -> np.ndarray:
        """
        Get restricted triclinic box matrix from lengths and angles

        Args:
            lengths (np.ndarray): lengths of box edges
            angles (np.ndarray): angles between box edges in degree

        Returns:
            np.ndarray: restricted triclinic box matrix
        """
        a, b, c = lengths
        alpha, beta, gamma = np.deg2rad(angles)
        lx = a
        ly = b * np.sin(gamma)
        xy = b * np.cos(gamma)
        xz = c * np.cos(beta)
        yz = (b * c * np.cos(alpha) - xy * xz) / ly
        lz = np.sqrt(c**2 - xz**2 - yz**2)
        return np.array([[lx, xy, xz], [0, ly, yz], [0, 0, lz]])

    @staticmethod
    def calc_matrix_from_size_tilts(sizes, tilts) -> np.ndarray:
        """
        Get restricted triclinic box matrix from sizes and tilts

        Args:
            sizes (np.ndarray): sizes of box edges
            tilts (np.ndarray): tilts between box edges

        Returns:
            np.ndarray: restricted triclinic box matrix
        """
        lx, ly, lz = sizes
        xy, xz, yz = tilts
        return np.array([[lx, xy, xz], [0, ly, yz], [0, 0, lz]])

    @staticmethod
    def calc_lengths_from_matrix(matrix: np.ndarray) -> np.ndarray:
        return np.linalg.norm(matrix, axis=1)

    @staticmethod
    def calc_angles_from_matrix(matrix: np.ndarray) -> np.ndarray:
        a = np.linalg.norm(matrix[:, 0])
        b = np.linalg.norm(matrix[:, 1])
        c = np.linalg.norm(matrix[:, 2])
        alpha = np.arccos((matrix[:, 1] @ matrix[:, 2]) / b / c)
        beta = np.arccos((matrix[:, 0] @ matrix[:, 2]) / a / c)
        gamma = np.arccos((matrix[:, 0] @ matrix[:, 1]) / a / b)
        return np.rad2deg(np.array([alpha, beta, gamma]))

    @staticmethod
    def calc_style_from_matrix(matrix: np.ndarray) -> Style:

        if np.allclose(matrix, np.eye(3)):
            return Box.Style.FREE
        elif np.allclose(matrix, np.diag(np.diagonal(matrix))):
            return Box.Style.ORTHOGONAL
        elif np.tril(matrix, 1).sum() == 0:
            return Box.Style.TRICLINIC

    def set_lengths(self, lengths: np.ndarray):
        self._matrix = self.calc_matrix_from_lengths_angles(lengths, self.angles)

    def set_angles(self, angles: np.ndarray):
        self._matrix = self.calc_matrix_from_lengths_angles(self.lengths, angles)

    def set_matrix(self, matrix: np.ndarray):
        self._matrix = matrix

    def set_lengths_angles(self, lengths: np.ndarray, angles: np.ndarray):
        self._matrix = self.calc_matrix_from_lengths_angles(lengths, angles)

    def set_lengths_tilts(self, lengths: np.ndarray, tilts: np.ndarray):
        self._matrix = self.calc_matrix_from_size_tilts(lengths, tilts)

    @property
    def volume(self) -> float:
        match self.style:
            case Box.Style.FREE:
                return 0
            case Box.Style.ORTHOGONAL:
                return np.prod(self.lengths)
            case Box.Style.TRICLINIC:
                return np.abs(np.linalg.det(self._matrix))

    def get_distance_between_faces(self) -> np.ndarray:
        match self.style:
            case Box.Style.FREE:
                return np.zeros(3)
            case Box.Style.ORTHOGONAL:
                return self.lengths
            case Box.Style.TRICLINIC:
                a = self._matrix[:, 0]
                b = self._matrix[:, 1]
                c = self._matrix[:, 2]

                na = np.cross(b, c)
                nb = np.cross(c, a)
                nc = np.cross(a, b)
                na /= np.linalg.norm(na)
                nb /= np.linalg.norm(nb)
                nc /= np.linalg.norm(nc)

                return np.array([np.dot(na, a), np.dot(nb, b), np.dot(nc, c)])

    def wrap(self, xyz: np.ndarray) -> np.ndarray:

        match self.style:
            case Box.Style.FREE:
                return self.wrap_free(xyz)
            case Box.Style.ORTHOGONAL:
                return self.wrap_orthogonal(xyz)
            case Box.Style.TRICLINIC:
                return self.wrap_triclinic(xyz)

    def wrap_free(self, xyz: np.ndarray) -> np.ndarray:
        return xyz

    def wrap_orthogonal(self, xyz: np.ndarray) -> np.ndarray:
        lengths = self.lengths
        return xyz - np.floor(xyz / lengths) * lengths

    def wrap_triclinic(self, xyz: np.ndarray) -> np.ndarray:
        fractional = np.dot(self.get_inv(), xyz.T)
        return np.dot(self._matrix, fractional - np.floor(fractional)).T

    def get_inv(self) -> np.ndarray:
        return np.linalg.inv(self._matrix)

    def diff_dr(self, dr: np.ndarray) -> np.ndarray:

        match self.style:
            case Box.Style.FREE:
                return dr
            case Box.Style.ORTHOGONAL | Box.Style.TRICLINIC:
                fractional = self.make_fractional(dr)
                fractional -= np.round(fractional)
                return np.dot(self._matrix, fractional.T).T

    def diff(self, r1: np.ndarray, r2: np.ndarray) -> np.ndarray:
        return self.diff_dr(r1 - r2)

    def make_fractional(self, r: np.ndarray) -> np.ndarray:
        return np.dot(r, self.get_inv())

    def make_absolute(self, r: np.ndarray) -> np.ndarray:
        return np.dot(r, self._matrix)

    def isin(self, xyz: np.ndarray) -> bool:
        return np.all(np.abs(self.wrap(xyz) - xyz) < 1e-5)
