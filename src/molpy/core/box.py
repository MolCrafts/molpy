# author: Roy Kid
# contact: lijichen365@126.com
# date: 2023-09-29
# version: 0.0.1

from enum import Enum

import numpy as np
from numpy.typing import ArrayLike

from .region import PeriodicBoundary


class Box(PeriodicBoundary):
    """Simulation box representing a periodic domain in 3D space.

    The box is defined by a 3x3 upper-triangular matrix whose columns are
    the lattice vectors, an origin point, and per-axis periodic boundary
    flags.  Three styles are supported:

    - **FREE** -- no boundaries (zero matrix).
    - **ORTHOGONAL** -- axis-aligned cuboid (diagonal matrix).
    - **TRICLINIC** -- general parallelepiped (upper-triangular matrix
      with at least one nonzero off-diagonal element).

    All length quantities are in Angstroms.  Angles are in degrees
    unless stated otherwise.

    Args:
        matrix: A 3x3 upper-triangular box matrix with lattice vectors
            as columns, shape ``(3, 3)``.  ``None`` or an all-zero matrix
            produces a FREE box.  A 1-D array of shape ``(3,)`` is
            promoted to a diagonal matrix.
        pbc: Boolean periodic-boundary flags per axis, shape ``(3,)``.
        origin: Cartesian origin of the box in Angstroms, shape ``(3,)``.
    """

    class Style(Enum):
        """Enumeration of simulation-box geometries.

        Attributes:
            FREE: No bounding box (vacuum / non-periodic).
            ORTHOGONAL: Axis-aligned cuboid with three independent edge
                lengths.
            TRICLINIC: General parallelepiped described by three edge
                lengths and three tilt factors (xy, xz, yz).
        """

        FREE = 0
        ORTHOGONAL = 1
        TRICLINIC = 2

    def __init__(
        self,
        matrix: ArrayLike | None = None,
        pbc: ArrayLike = np.ones(3, dtype=bool),
        origin: ArrayLike = np.zeros(3),
    ):
        """
        Initialize a Box object.

        Parameters:
            matrix (np.ndarray | None, optional): A 3x3 matrix representing the box dimensions.
                If None or all elements are zero, a zero matrix is used. If a 1D array of shape (3,)
                is provided, it is converted to a diagonal matrix. Defaults to None.
            pbc (np.ndarray, optional): A 1D boolean array of shape (3,) indicating periodic boundary
                conditions along each axis. Defaults to an array of ones (True for all axes).
            origin (np.ndarray, optional): A 1D array of shape (3,) representing the origin of the box.
                Defaults to an array of zeros.
        """
        super().__init__()
        if matrix is None or np.all(matrix == 0):
            _matrix = np.zeros((3, 3))
        else:
            _matrix = np.asarray(matrix)
            if _matrix.shape == (3,):
                _matrix = np.diag(_matrix)
            _matrix = Box.check_matrix(_matrix)
        self._matrix: np.ndarray = np.array(_matrix, dtype=float)
        self._pbc: np.ndarray = np.array(pbc, dtype=bool)
        self._origin: np.ndarray = np.array(origin, dtype=float)

    def __repr__(self):
        match self.style:
            case Box.Style.FREE:
                return "<Free Box>"
            case Box.Style.ORTHOGONAL:
                return f"<Orthogonal Box: {self.lengths}>"
            case Box.Style.TRICLINIC:
                return f"<Triclinic Box: {self.lengths}, {self.tilts}>"
        return "<Box>"

    def __mul__(self, other: float):
        _matrix = self._matrix * other
        return Box(_matrix, self._pbc, self._origin)

    def __rmul__(self, other: float):
        _matrix = self._matrix * other
        return Box(_matrix, self._pbc, self._origin)

    def __eq__(self, value: object) -> bool:
        if not isinstance(value, Box):
            return False
        return (
            np.allclose(self._matrix, value.matrix)
            and np.allclose(self._origin, value.origin)
            and np.allclose(self._pbc, value.pbc)
        )

    def plot(self):
        """
        Plot the box representation. This method is a placeholder and should be implemented
        to visualize the box in 3D space.
        """
        ...

    @classmethod
    def cubic(
        cls,
        length: float,
        pbc: ArrayLike = np.ones(3, dtype=bool),
        origin: ArrayLike = np.zeros(3),
        central: bool = False,
    ) -> "Box":
        """Create a cubic box with equal edge lengths.

        Args:
            length: Edge length of the cube in Angstroms.
            pbc: Periodic boundary flags per axis, shape ``(3,)``.
            origin: Cartesian origin of the box in Angstroms, shape ``(3,)``.
            central: If True, shift the origin so the box is centred at
                the coordinate origin.

        Returns:
            A new cubic ``Box`` instance.
        """
        if central:
            origin = np.full(3, -length / 2)
        return cls(np.diag(np.full(3, length)), pbc, origin)

    @classmethod
    def orth(
        cls,
        lengths: ArrayLike,
        pbc: ArrayLike = np.ones(3, dtype=bool),
        origin: ArrayLike = np.zeros(3),
        central: bool = False,
    ) -> "Box":
        """Create an orthogonal (axis-aligned cuboid) box.

        Args:
            lengths: Edge lengths ``[lx, ly, lz]`` in Angstroms,
                shape ``(3,)``.
            pbc: Periodic boundary flags per axis, shape ``(3,)``.
            origin: Cartesian origin of the box in Angstroms, shape ``(3,)``.
            central: If True, shift the origin so the box is centred at
                the coordinate origin.

        Returns:
            A new orthogonal ``Box`` instance.
        """
        if central:
            origin = np.zeros(3) - np.asarray(lengths) / 2
        return cls(np.diag(lengths), pbc, origin)

    @classmethod
    def tric(
        cls,
        lengths: ArrayLike,
        tilts: ArrayLike,
        pbc: ArrayLike = np.ones(3, dtype=bool),
        origin: ArrayLike = np.zeros(3),
        central: bool = False,
    ) -> "Box":
        """Create a triclinic box from edge lengths and tilt factors.

        Args:
            lengths: Diagonal edge lengths ``[lx, ly, lz]`` in Angstroms,
                shape ``(3,)``.
            tilts: Off-diagonal tilt factors ``[xy, xz, yz]`` in Angstroms,
                shape ``(3,)``.
            pbc: Periodic boundary flags per axis, shape ``(3,)``.
            origin: Cartesian origin of the box in Angstroms, shape ``(3,)``.
            central: If True, shift the origin so the box is centred at
                the coordinate origin.

        Returns:
            A new triclinic ``Box`` instance.
        """
        if central:
            origin = np.asarray(lengths) / 2
        return cls(cls.calc_matrix_from_size_tilts(lengths, tilts), pbc, origin)

    @classmethod
    def from_box(cls, box: "Box") -> "Box":
        """
        Create a new box from an existing box.

        Args:
            box (Box): The existing box.

        Returns:
            Box: A new box with the same properties as the existing box.
        """
        return cls(box.matrix.copy(), box.pbc.copy(), box.origin.copy())

    @property
    def xlo(self) -> float:
        """
        Calculate the lower bound of the box along the x-axis.

        Returns:
            float: The x-coordinate of the lower bound, calculated as the
            negative of the x-component of the origin.
        """
        return self._origin[0]

    @property
    def xhi(self) -> float:
        """
        Calculate the upper boundary of the box along the x-axis.

        Returns:
            float: The x-coordinate of the upper boundary of the box,
            calculated as the difference between the first element of
            the matrix's first row and the x-coordinate of the origin.
        """
        return self._matrix[0, 0] - self._origin[0]

    @property
    def ylo(self) -> float:
        """
        Get the lower boundary of the box along the y-axis.

        Returns:
            float: The negative value of the y-coordinate of the origin.
        """
        return self._origin[1]

    @property
    def yhi(self) -> float:
        """
        Calculate the upper boundary of the box along the y-axis.

        Returns:
            float: The upper boundary value of the box in the y-dimension,
            calculated as the difference between the y-component of the
            matrix and the y-component of the origin.
        """
        return self._matrix[1, 1] - self._origin[1]

    @property
    def zlo(self) -> float:
        """
        Calculate the lower boundary of the box along the z-axis.

        Returns:
            float: The z-coordinate of the lower boundary, calculated as the
            negative value of the third component of the origin vector.
        """
        return self._origin[2]

    @property
    def zhi(self) -> float:
        """
        Calculate the z-component of the box's upper boundary.

        Returns:
            float: The z-coordinate of the upper boundary, calculated as the
            difference between the z-component of the matrix and the z-component
            of the origin.
        """
        return self._matrix[2, 2] - self._origin[2]

    @property
    def bounds(self) -> np.ndarray:
        """
        Get the bounds of the box.

        Returns:
            np.ndarray: A 2D array with shape (3, 2) representing the bounds of the box.
        """
        return np.array(
            [[self.xlo, self.xhi], [self.ylo, self.yhi], [self.zlo, self.zhi]]
        ).T

    @property
    def style(self) -> Style:
        """
        Determine the style of the box based on its matrix.

        Returns:
            Style: The style of the box (FREE, ORTHOGONAL, or TRICLINIC).
        """
        return self.calc_style_from_matrix(self._matrix)

    @property
    def pbc(self) -> np.ndarray:
        """
        Get the periodic boundary conditions of the box.

        Returns:
            np.ndarray: A boolean array indicating periodicity along each axis.
        """
        return self._pbc

    @property
    def matrix(self) -> np.ndarray:
        """
        Get the matrix representation of the box.

        Returns:
            np.ndarray: A 3x3 matrix representing the box dimensions.
        """
        return self._matrix

    def __array__(self, dtype=None, copy=None):
        """Allow Box to be converted to numpy array (returns 3x3 matrix).

        This enables np.array(box) to automatically convert Box to its matrix
        representation, which is useful for serialization and numerical operations.

        Args:
            dtype: Optional numpy dtype for the array.
            copy: Optional copy flag (for numpy 2.0+ compatibility).

        Returns:
            np.ndarray: The 3x3 box matrix.
        """
        matrix = self._matrix
        if dtype is not None:
            matrix = matrix.astype(dtype)
        if copy is True:
            matrix = matrix.copy()
        return matrix

    @property
    def volume(self) -> float:
        """
        Calculate the volume of the box.

        Returns:
            float: The volume of the box.
        """
        return np.abs(np.linalg.det(self._matrix))

    @property
    def origin(self) -> np.ndarray:
        """Cartesian origin of the box in Angstroms.

        Returns:
            np.ndarray: Origin coordinates, shape ``(3,)``.
        """
        return self._origin

    @origin.setter
    def origin(self, value: np.ndarray):
        value = np.asarray(value)
        assert value.shape == (3,), "origin must be (3,)"
        self._origin = value

    @property
    def lx(self) -> float:
        """
        Get the length of the box along the x-axis.

        Returns:
            float: The length of the box in the x-direction, derived from the
            first element of the matrix representing the box dimensions.
        """
        return self._matrix[0, 0]

    @lx.setter
    def lx(self, value: float):
        self._matrix[0, 0] = value

    @property
    def ly(self) -> float:
        """
        Get the length of the simulation box along the y-axis.

        Returns:
            float: The length of the box in the y-direction.
        """
        return self._matrix[1, 1]

    @ly.setter
    def ly(self, value: float):
        self._matrix[1, 1] = value

    @property
    def lz(self) -> float:
        """
        Get the length of the simulation box along the z-axis.

        Returns:
            float: The length of the box in the z-direction.
        """
        return self._matrix[2, 2]

    @lz.setter
    def lz(self, value: float):
        self._matrix[2, 2] = value

    @property
    def l(self) -> np.ndarray:
        """
        Get the lengths of the box along each axis.

        Returns:
            np.ndarray: A 1D array containing the lengths of the box along
            the x, y, and z axes.
        """
        return self._matrix.diagonal()

    @property
    def l_inv(self) -> np.ndarray:
        """Reciprocal of the box edge lengths in inverse Angstroms.

        Returns:
            np.ndarray: ``1 / [lx, ly, lz]``, shape ``(3,)``.
        """
        return 1 / self.l

    @property
    def xy(self) -> float:
        """
        Retrieve the xy component of the matrix.

        Returns:
            float: The value at the (0, 1) position in the matrix.
        """
        return self._matrix[0, 1]

    @xy.setter
    def xy(self, value: float):
        self._matrix[0, 1] = value

    @property
    def xz(self) -> float:
        """Tilt factor between the x and z axes in Angstroms.

        Returns:
            float: The ``(0, 2)`` element of the box matrix.
        """
        return self._matrix[0, 2]

    @xz.setter
    def xz(self, value: float):
        self._matrix[0, 2] = value

    @property
    def yz(self) -> float:
        """Tilt factor between the y and z axes in Angstroms.

        Returns:
            float: The ``(1, 2)`` element of the box matrix.
        """
        return self._matrix[1, 2]

    @yz.setter
    def yz(self, value: float):
        self._matrix[1, 2] = value

    @property
    def a(self) -> np.ndarray:
        """First lattice vector of the box in Angstroms.

        Returns:
            np.ndarray: Column 0 of the box matrix, shape ``(3,)``.
        """
        return self._matrix[:, 0]

    @property
    def b(self) -> np.ndarray:
        """Second lattice vector of the box in Angstroms.

        Returns:
            np.ndarray: Column 1 of the box matrix, shape ``(3,)``.
        """
        return self._matrix[:, 1]

    @property
    def c(self) -> np.ndarray:
        """Third lattice vector of the box in Angstroms.

        Returns:
            np.ndarray: Column 2 of the box matrix, shape ``(3,)``.
        """
        return self._matrix[:, 2]

    @property
    def periodic(self) -> bool:
        """Whether the box is periodic in all three directions.

        Returns:
            bool: True if periodic boundary conditions are active along
            every axis.
        """
        return bool(self._pbc.all())

    @periodic.setter
    def periodic(self, value: bool | list[bool]):
        if isinstance(value, list):
            assert len(value) == 3, "value must be list of length 3"
            self._pbc = np.array(value, dtype=bool)
        else:
            self._pbc = np.full(3, value, dtype=bool)

    @property
    def periodic_x(self) -> bool:
        """Whether the box is periodic along the x-axis.

        Returns:
            bool: True if periodic in x.
        """
        return self._pbc[0]

    @periodic_x.setter
    def periodic_x(self, value: bool):
        self._pbc[0] = value

    @property
    def periodic_y(self) -> bool:
        """Whether the box is periodic along the y-axis.

        Returns:
            bool: True if periodic in y.
        """
        return self._pbc[1]

    @periodic_y.setter
    def periodic_y(self, value: bool):
        self._pbc[1] = value

    @property
    def periodic_z(self) -> bool:
        """Whether the box is periodic along the z-axis.

        Returns:
            bool: True if periodic in z.
        """
        return self._pbc[2]

    @periodic_z.setter
    def periodic_z(self, value: bool):
        self._pbc[2] = value

    @property
    def tilts(self) -> np.ndarray:
        """Off-diagonal tilt factors of the box matrix in Angstroms.

        Returns:
            np.ndarray: ``[xy, xz, yz]``, shape ``(3,)``.
        """
        return np.array([self.xy, self.xz, self.yz])

    @property
    def is_periodic(self) -> bool:
        """Check if the box has periodic boundary conditions in all directions."""
        return self.periodic

    @staticmethod
    def check_matrix(matrix: np.ndarray) -> np.ndarray:
        """
        Validate the box matrix.

        Args:
            matrix (np.ndarray): A 3x3 matrix to validate.

        Returns:
            np.ndarray: The validated matrix.

        Raises:
            AssertionError: If the matrix is not valid.
        """
        assert isinstance(matrix, np.ndarray), "matrix must be np.ndarray"
        assert matrix.shape == (3, 3), "matrix must be (3, 3)"
        assert not np.isclose(np.linalg.det(matrix), 0), "matrix must be non-singular"
        return matrix

    @property
    def lengths(self) -> np.ndarray:
        """Lattice vector magnitudes ``[a, b, c]`` in Angstroms.

        For a FREE box all lengths are zero.  For orthogonal and triclinic
        boxes the lengths are computed from the box matrix.

        Returns:
            np.ndarray: Edge lengths, shape ``(3,)``.
        """
        match self.style:
            case Box.Style.FREE:
                return np.zeros(3)
            case Box.Style.ORTHOGONAL | Box.Style.TRICLINIC:
                return self.calc_lengths_angles_from_matrix(self._matrix)[0]
        raise ValueError("Invalid box style")

    @lengths.setter
    def lengths(self, value: np.ndarray):
        value = np.asarray(value)
        assert value.shape == (3,), ValueError("lengths must be (3,)")
        matrix = self.calc_matrix_from_lengths_angles(value, self.angles)
        self._matrix = matrix

    @property
    def angles(self) -> np.ndarray:
        """Lattice angles ``[alpha, beta, gamma]`` in degrees.

        Alpha is the angle between lattice vectors **b** and **c**, beta
        between **a** and **c**, and gamma between **a** and **b**.

        Returns:
            np.ndarray: Angles in degrees, shape ``(3,)``.
        """
        return self.calc_lengths_angles_from_matrix(self._matrix)[1]

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
        abc: ArrayLike, angles: ArrayLike
    ) -> np.ndarray:
        """
        Compute restricted triclinic box matrix from lengths and angles.

        Args:
            abc (np.ndarray): [a, b, c] lattice vector lengths
            angles (np.ndarray): [alpha, beta, gamma] in degrees (angles between (b,c), (a,c), (a,b))

        Returns:
            np.ndarray: 3x3 box matrix with lattice vectors as columns: [a | b | c]
        """
        a, b, c = abc
        angles = alpha, beta, gamma = np.deg2rad(angles)
        cos_a, cos_b, cos_c = np.cos(angles)

        # Optional: volume sanity check
        cos_check = cos_a**2 + cos_b**2 + cos_c**2 - 2 * cos_a * cos_b * cos_c
        if cos_check >= 1.0:
            raise ValueError(
                f"Invalid box: angles produce non-physical volume. abc={abc}, angles={angles}"
            )

        if not (0 < alpha < np.pi):
            raise ValueError("alpha must be in (0, 180)")
        if not (0 < beta < np.pi):
            raise ValueError("beta must be in (0, 180)")
        if not (0 < gamma < np.pi):
            raise ValueError("gamma must be in (0, 180)")

        # Construct box
        lx = a
        xy = b * cos_c
        xz = c * cos_b
        ly = np.sqrt(b**2 - xy**2)
        yz = (b * c * cos_a - xy * xz) / ly
        tmp = c**2 - xz**2 - yz**2
        lz = np.sqrt(tmp)

        return np.array([[lx, xy, xz], [0.0, ly, yz], [0.0, 0.0, lz]])

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
    def calc_lengths_angles_from_matrix(
        matrix: np.ndarray,
    ) -> tuple[np.ndarray, np.ndarray]:
        """
        Calculate the lengths of the box edges and angles from its matrix.

        Args:
            matrix (np.ndarray): A 3x3 matrix representing the box.

        Returns:
            tuple[np.ndarray, np.ndarray]: lengths and angles
        """

        lx = matrix[0, 0]
        ly = matrix[1, 1]
        lz = matrix[2, 2]
        xy = matrix[0, 1]
        xz = matrix[0, 2]
        yz = matrix[1, 2]

        a = lx
        b = (ly**2 + xy**2) ** 0.5
        c = (lz**2 + xz**2 + yz**2) ** 0.5
        cos_a = (xy * xz + ly * yz) / (b * c)
        cos_b = xz / c
        cos_c = xy / b
        return np.array([a, b, c]), np.rad2deg(np.arccos([cos_a, cos_b, cos_c]))

    @staticmethod
    def calc_style_from_matrix(matrix: np.ndarray) -> Style:
        """
        Determine the style of the box based on its matrix.

        Args:
            matrix (np.ndarray): A 3x3 matrix representing the box.

        Returns:
            Style: The style of the box (FREE, ORTHOGONAL, or TRICLINIC).

        Raises:
            ValueError: If the matrix does not correspond to a valid style.
        """
        if np.allclose(matrix, np.zeros(3)):
            return Box.Style.FREE
        elif np.allclose(matrix, np.diag(np.diagonal(matrix))):
            return Box.Style.ORTHOGONAL
        elif (matrix[np.tril_indices(3, -1)] == 0).all() and (
            matrix[np.triu_indices(3, 1)] != 0
        ).any():
            return Box.Style.TRICLINIC
        else:
            raise ValueError("Invalid box matrix")

    @classmethod
    def from_lengths_angles(cls, lengths: ArrayLike, angles: ArrayLike) -> "Box":
        """
        Get box matrix from lengths and angles

        Args:
            lengths (np.ndarray): lengths of box edges
            angles (np.ndarray): angles between box edges in degree

        Returns:
            Box: Box instance constructed from lengths and angles.
        """
        return cls(cls.calc_matrix_from_lengths_angles(lengths, angles))

    def to_lengths_angles(self) -> tuple[np.ndarray, np.ndarray]:
        """
        Get lengths and angles from box matrix

        Returns:
            tuple[np.ndarray, np.ndarray]: lengths and angles
        """
        return self.calc_lengths_angles_from_matrix(self._matrix)

    def set_lengths(self, lengths: np.ndarray):
        """Set the lattice vector magnitudes, preserving current angles.

        Args:
            lengths: New edge lengths ``[a, b, c]`` in Angstroms,
                shape ``(3,)``.
        """
        self._matrix = self.calc_matrix_from_lengths_angles(lengths, self.angles)

    def set_angles(self, angles: np.ndarray):
        """Set the lattice angles, preserving current edge lengths.

        Args:
            angles: New angles ``[alpha, beta, gamma]`` in degrees,
                shape ``(3,)``.
        """
        self._matrix = self.calc_matrix_from_lengths_angles(self.lengths, angles)

    def set_matrix(self, matrix: np.ndarray):
        """Replace the box matrix directly.

        Args:
            matrix: New 3x3 upper-triangular box matrix, shape ``(3, 3)``.
        """
        self._matrix = matrix

    def set_lengths_angles(self, lengths: np.ndarray, angles: np.ndarray):
        """Set both edge lengths and lattice angles simultaneously.

        Args:
            lengths: Edge lengths ``[a, b, c]`` in Angstroms, shape ``(3,)``.
            angles: Angles ``[alpha, beta, gamma]`` in degrees, shape ``(3,)``.
        """
        self._matrix = self.calc_matrix_from_lengths_angles(lengths, angles)

    def set_lengths_tilts(self, lengths: np.ndarray, tilts: np.ndarray):
        """Set edge lengths and tilt factors simultaneously.

        Args:
            lengths: Diagonal edge lengths ``[lx, ly, lz]`` in Angstroms,
                shape ``(3,)``.
            tilts: Off-diagonal tilt factors ``[xy, xz, yz]`` in Angstroms,
                shape ``(3,)``.
        """
        self._matrix = self.calc_matrix_from_size_tilts(lengths, tilts)

    def get_distance_between_faces(self) -> np.ndarray:
        """Perpendicular distances between opposite faces in Angstroms.

        For an orthogonal box these equal the edge lengths.  For a
        triclinic box the distances are computed by projecting each
        lattice vector onto the normal of the plane spanned by the other
        two vectors.

        Returns:
            np.ndarray: Distances ``[d_x, d_y, d_z]`` in Angstroms,
                shape ``(3,)``.  All zeros for a FREE box.
        """
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
        """Wrap Cartesian coordinates into the primary image of the box.

        Delegates to style-specific implementations (free, orthogonal,
        or triclinic).

        Args:
            xyz: Cartesian positions in Angstroms, shape ``(N, 3)`` or
                ``(3,)``.

        Returns:
            np.ndarray: Wrapped coordinates with the same shape as the
                input, in Angstroms.
        """
        xyz = np.asarray(xyz)
        match self.style:
            case Box.Style.FREE:
                wrapped = self.wrap_free(xyz)
            case Box.Style.ORTHOGONAL:
                wrapped = self.wrap_orthogonal(xyz)
            case Box.Style.TRICLINIC:
                wrapped = self.wrap_triclinic(xyz)
            case _:
                raise ValueError(f"Unknown box style: {self.style}")
        return wrapped

    def wrap_free(self, xyz: np.ndarray) -> np.ndarray:
        """
        Wrap coordinates for a free box style.

        Args:
            xyz (np.ndarray): The coordinates to wrap.

        Returns:
            np.ndarray: The wrapped coordinates.
        """
        return xyz

    def wrap_orthogonal(self, xyz: np.ndarray) -> np.ndarray:
        """
        Wrap coordinates for an orthogonal box style.

        Args:
            xyz (np.ndarray): The coordinates to wrap.

        Returns:
            np.ndarray: The wrapped coordinates.
        """
        lengths = self.lengths  # Should be shape (3,)
        return xyz - np.floor(xyz / lengths) * lengths

    def wrap_triclinic(self, xyz: np.ndarray) -> np.ndarray:
        """
        Wrap coordinates for a triclinic box style.

        Args:
            xyz (np.ndarray): The coordinates to wrap.

        Returns:
            np.ndarray: The wrapped coordinates.
        """
        xyz = np.atleast_2d(xyz)  # Ensure xyz is a 2D array
        frac = self.make_fractional(xyz)
        frac_wrapped = frac - np.floor(frac)
        return self.make_absolute(frac_wrapped)

    def unwrap(self, xyz: np.ndarray, image: np.ndarray) -> np.ndarray:
        """
        Unwrap the coordinates of a particle based on its image.

        Args:
            xyz (np.ndarray): The coordinates of the particle.
            image (np.ndarray): The image of the particle.

        Returns:
            np.ndarray: The unwrapped coordinates.
        """
        return xyz + image @ self._matrix.T

    def get_images(self, xyz: np.ndarray) -> np.ndarray:
        """
        Get the image flags of particles, accounting for box origin and triclinic shape.
        """
        fractional = self.make_fractional(xyz)
        # Add small epsilon to avoid numerical floor error at box edges
        return np.floor(fractional + 1e-8).astype(int)

    def get_inv(self) -> np.ndarray:
        """
        Get the inverse of the box matrix.

        Returns:
            np.ndarray: The inverse of the box matrix.
        """
        return np.linalg.inv(self._matrix)

    def diff_dr(self, dr: np.ndarray) -> np.ndarray:
        """
        Calculate the difference vector considering periodic boundary conditions.

        Args:
            dr (np.ndarray): The difference vector.

        Returns:
            np.ndarray: The adjusted difference vector.
        """
        match self.style:
            case Box.Style.FREE:
                return dr
            case Box.Style.ORTHOGONAL | Box.Style.TRICLINIC:
                fractional = self.make_fractional(dr)
                fractional -= np.round(fractional)
                return np.dot(self._matrix, fractional.T).T

    def diff(self, r1: np.ndarray, r2: np.ndarray) -> np.ndarray:
        """
        Calculate the difference between two points considering periodic boundary conditions.

        Args:
            r1 (np.ndarray): The first point.
            r2 (np.ndarray): The second point.

        Returns:
            np.ndarray: The difference vector.
        """
        return self.diff_dr(r1 - r2)

    def diff_all(self, r1: np.ndarray, r2: np.ndarray) -> np.ndarray:
        """
        Calculate the difference between all pairs of points in two sets.

        Args:
            r1 (np.ndarray): The first set of points.
            r2 (np.ndarray): The second set of points.

        Returns:
            np.ndarray: The difference vectors for all pairs.
        """
        all_dr = r1[:, np.newaxis, :] - r2[np.newaxis, :, :]
        original_shape = all_dr.shape
        all_dr = all_dr.reshape(-1, 3)
        all_dr = self.diff_dr(all_dr)
        all_dr = all_dr.reshape(original_shape)
        return all_dr

    def dist(self, r1: np.ndarray, r2: np.ndarray) -> np.ndarray:
        """
        Calculate the distance between two points.

        Args:
            r1 (np.ndarray): The first point.
            r2 (np.ndarray): The second point.

        Returns:
            np.ndarray: The distance between the points.
        """
        dr = self.diff(r1, r2)
        return np.linalg.norm(dr, axis=1)

    def dist_all(self, r1: np.ndarray, r2: np.ndarray) -> np.ndarray:
        """
        Calculate the distances between all pairs of points in two sets.

        Args:
            r1 (np.ndarray): The first set of points.
            r2 (np.ndarray): The second set of points.

        Returns:
            np.ndarray: The distances for all pairs.
        """
        dr = self.diff_all(r1, r2)
        return np.linalg.norm(dr, axis=-1)

    def make_fractional(self, xyz: np.ndarray) -> np.ndarray:
        """
        Convert absolute coordinates to fractional coordinates.

        Args:
            xyz (np.ndarray): The absolute coordinates.

        Returns:
            np.ndarray: The fractional coordinates.
        """
        return (xyz - self._origin) @ self.get_inv().T

    def make_absolute(self, xyz: np.ndarray) -> np.ndarray:
        """
        Convert fractional coordinates to absolute coordinates.

        Args:
            xyz (np.ndarray): The fractional coordinates.

        Returns:
            np.ndarray: The absolute coordinates.
        """
        return xyz @ self._matrix.T + self._origin

    def isin(self, xyz: np.ndarray):
        """
        Check if point(s) xyz are inside the box.
        Args:
            xyz (np.ndarray): shape (..., 3)
        Returns:
            np.ndarray: boolean array, True if inside
        """
        xyz = np.asarray(xyz)
        fractional = self.make_fractional(xyz)
        return np.all((fractional >= 0) & (fractional < 1), axis=-1)

    def merge(self, other: "Box") -> "Box":
        """
        Merge two boxes to find their common space.

        Args:
            other (Box): The other box to merge with.

        Returns:
            Box: A new box representing the common space.
        """
        return Box(matrix=other.matrix)

    def transform(self, transformation_matrix: np.ndarray) -> "Box":
        """Transform the box using a transformation matrix.

        Args:
            transformation_matrix: 3x3 transformation matrix.

        Returns:
            New Box with transformed dimensions.
        """
        # Transform the box matrix
        new_matrix = self._matrix @ transformation_matrix

        # Keep the same periodic boundary conditions and origin
        return Box(matrix=new_matrix, pbc=self._pbc.copy(), origin=self._origin.copy())

    def to_dict(self) -> dict:
        """
        Convert the box to a dictionary representation.

        Returns:
            dict: A dictionary containing the box properties.
        """
        return {
            "matrix": self._matrix.tolist(),
            "pbc": self._pbc.tolist(),
            "origin": self._origin.tolist(),
        }
