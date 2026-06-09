# author: Roy Kid
# contact: lijichen365@126.com
# date: 2023-09-29
# version: 0.1.0  — molrs.Box inheritance refactor

from enum import Enum

import molrs
import numpy as np
from numpy.typing import ArrayLike


class Box(molrs.Box):
    """Simulation box — molpy front for the molrs spatial primitive.

    Inherits ``molrs.Box`` directly, so a ``molpy.Box`` instance is
    accepted by every molrs API (``NeighborQuery``, ``RDF``, ``wrap``,
    ``isin``, …) without conversion. molpy adds:

    - the ``Style`` enum (FREE / ORTHOGONAL / TRICLINIC),
    - molpy-style accessors (``lx``, ``ly``, ``lz``, ``xy``, ``xz``,
      ``yz``, ``a``, ``b``, ``c``, ``lengths``, ``angles``, ``tilts``,
      ``bounds``, ``xlo``/``xhi``/…),
    - convenience factories (``cubic``, ``orth``, ``tric``,
      ``from_lengths_angles``, ``from_bounds``, ``from_box``),
    - PBC-aware geometry helpers (``wrap``, ``unwrap``, ``diff``,
      ``dist``, ``make_fractional``, ``make_absolute``,
      ``get_distance_between_faces``, ``get_images``).

    **Immutable.** State lives in the molrs base; per-axis setters and
    ``set_*`` methods were removed in this refactor (use one of the
    classmethod factories to construct a new ``Box`` instead). This
    matches molpy's own ``coding-style.md`` "avoid mutation" rule.

    Args:
        matrix: A ``(3, 3)`` upper-triangular box matrix (lattice vectors
            as columns). ``None`` or an all-zero matrix produces a FREE
            (non-periodic) box. A ``(3,)`` array is promoted to a
            diagonal matrix.
        pbc: Boolean periodic-boundary flags per axis, shape ``(3,)``.
            Defaults to ``[True, True, True]``.
        origin: Cartesian origin in Angstroms, shape ``(3,)``. Defaults
            to ``[0, 0, 0]``.
    """

    class Style(str, Enum):
        """Enumeration of simulation-box geometries.

        Values are the canonical molrs style strings so a ``molpy.Box.Style``
        member compares equal to the string returned by ``molrs.Box.style``
        (e.g. ``Box.Style.ORTHOGONAL == "orthogonal"``), letting ``frame.box``
        (a molrs box) interoperate with molpy style checks.
        """

        FREE = "free"
        ORTHOGONAL = "orthogonal"
        TRICLINIC = "triclinic"

    # FREE boxes need a non-singular placeholder for the molrs base
    # (which rejects det(h)==0). The Python-side ``_is_free`` flag
    # restores zero-volume / zero-matrix semantics for FREE.
    _PLACEHOLDER_H: np.ndarray = np.eye(3)

    # ────────────────────────────────────────────────────────────────────
    # construction
    # ────────────────────────────────────────────────────────────────────

    def __new__(
        cls,
        matrix: ArrayLike | None = None,
        pbc: ArrayLike | None = None,
        origin: ArrayLike | None = None,
    ):
        is_free, h = cls._normalize_matrix(matrix)
        # A FREE box is non-periodic on every axis. Defaulting its pbc to
        # all-False is what lets the molrs Store (which only keeps matrix /
        # origin / pbc, not molpy's Python ``_is_free`` flag) reconstruct
        # ``box.is_free`` / ``box.style == "free"`` after a round-trip.
        if pbc is None and is_free:
            pbc_arr = np.zeros(3, dtype=bool)
        else:
            pbc_arr = cls._normalize_pbc(pbc)
        origin_arr = cls._normalize_origin(origin)
        instance = super().__new__(cls, h, origin=origin_arr, pbc=pbc_arr)
        instance._is_free = is_free
        return instance

    def __init__(
        self,
        matrix: ArrayLike | None = None,
        pbc: ArrayLike | None = None,
        origin: ArrayLike | None = None,
    ):
        # State already set by molrs.Box during __new__; nothing to do.
        # This shim only exists so ``Box(...)`` keyword forwarding to
        # ``__init__`` does not error.
        pass

    @classmethod
    def _normalize_matrix(cls, matrix: ArrayLike | None) -> tuple[bool, np.ndarray]:
        if matrix is None:
            return True, cls._PLACEHOLDER_H.copy()
        m = np.asarray(matrix, dtype=float)
        if m.shape == (3,):
            m = np.diag(m)
        if m.shape != (3, 3):
            raise ValueError(f"matrix must be (3, 3) or (3,), got {m.shape}")
        if np.allclose(m, 0.0):
            return True, cls._PLACEHOLDER_H.copy()
        if np.isclose(np.linalg.det(m), 0.0):
            raise ValueError("matrix must be non-singular")
        return False, m

    @staticmethod
    def _normalize_pbc(pbc: ArrayLike | None) -> np.ndarray:
        if pbc is None:
            return np.ones(3, dtype=bool)
        return np.asarray(pbc, dtype=bool).reshape(3)

    @staticmethod
    def _normalize_origin(origin: ArrayLike | None) -> np.ndarray:
        if origin is None:
            return np.zeros(3, dtype=float)
        return np.asarray(origin, dtype=float).reshape(3)

    # ────────────────────────────────────────────────────────────────────
    # core read-through accessors (override / shadow molrs)
    # ────────────────────────────────────────────────────────────────────

    @property
    def matrix(self) -> np.ndarray:
        """Box matrix with lattice vectors as columns, shape ``(3, 3)``."""
        if self._is_free:
            return np.zeros((3, 3))
        return np.asarray(self.h)

    @property
    def is_free(self) -> bool:
        """``True`` if this box is FREE (no periodicity, zero volume)."""
        return self._is_free

    @property
    def style(self) -> "Box.Style":
        """FREE / ORTHOGONAL / TRICLINIC depending on the matrix shape."""
        if self._is_free:
            return Box.Style.FREE
        return self.calc_style_from_matrix(self.matrix)

    @property
    def volume(self) -> float:
        """Box volume in Angstroms³ (zero for FREE)."""
        if self._is_free:
            return 0.0
        return float(np.abs(np.linalg.det(self.matrix)))

    # ── backward-compat private aliases ────────────────────────────────
    # Internal methods (wrap / diff / make_fractional / transform / …)
    # still reference ``self._matrix`` / ``self._origin`` / ``self._pbc``
    # from the original implementation. Expose them as read-only aliases
    # over the new property-backed accessors.

    @property
    def _matrix(self) -> np.ndarray:
        return self.matrix

    @property
    def _origin(self) -> np.ndarray:
        return np.asarray(self.origin)

    @property
    def _pbc(self) -> np.ndarray:
        return np.asarray(self.pbc)

    # ────────────────────────────────────────────────────────────────────
    # dunder
    # ────────────────────────────────────────────────────────────────────

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
        return Box(self.matrix * other, self._pbc.copy(), self._origin.copy())

    def __rmul__(self, other: float):
        return self.__mul__(other)

    def __eq__(self, value: object) -> bool:
        if not isinstance(value, Box):
            return False
        return (
            np.allclose(self.matrix, value.matrix)
            and np.allclose(self.origin, value.origin)
            and np.allclose(self.pbc, value.pbc)
        )

    def __hash__(self) -> int:
        return id(self)

    def __array__(self, dtype=None, copy=None):
        """``np.array(box)`` → 3x3 box matrix."""
        m = self.matrix
        if dtype is not None:
            m = m.astype(dtype)
        if copy is True:
            m = m.copy()
        return m

    def plot(self):
        """Placeholder for 3D box visualization."""
        ...

    # ────────────────────────────────────────────────────────────────────
    # factories
    # ────────────────────────────────────────────────────────────────────

    @classmethod
    def cubic(
        cls,
        length: float,
        pbc: ArrayLike | None = None,
        origin: ArrayLike | None = None,
        central: bool = False,
    ) -> "Box":
        """Cubic box with three equal edge lengths."""
        if central:
            origin = np.full(3, -length / 2)
        return cls(np.diag(np.full(3, length)), pbc, origin)

    @classmethod
    def orth(
        cls,
        lengths: ArrayLike,
        pbc: ArrayLike | None = None,
        origin: ArrayLike | None = None,
        central: bool = False,
    ) -> "Box":
        """Orthogonal (axis-aligned cuboid) box."""
        if central:
            origin = -np.asarray(lengths) / 2
        return cls(np.diag(lengths), pbc, origin)

    @classmethod
    def tric(
        cls,
        lengths: ArrayLike,
        tilts: ArrayLike,
        pbc: ArrayLike | None = None,
        origin: ArrayLike | None = None,
        central: bool = False,
    ) -> "Box":
        """Triclinic box from edge lengths and tilt factors."""
        if central:
            origin = np.asarray(lengths) / 2
        return cls(cls.calc_matrix_from_size_tilts(lengths, tilts), pbc, origin)

    @classmethod
    def from_box(cls, box: "Box") -> "Box":
        """Copy / upgrade constructor.

        Accepts a molpy ``Box`` or a bare ``molrs.Box`` (e.g. ``frame.box``),
        reading only the public ``matrix`` / ``pbc`` / ``origin`` accessors so it
        works across both. A free source box reconstructs a free molpy box.
        """
        if getattr(box, "is_free", False):
            return cls()
        return cls(
            np.asarray(box.matrix).copy(),
            np.asarray(box.pbc).copy(),
            np.asarray(box.origin).copy(),
        )

    @classmethod
    def from_bounds(
        cls,
        points: ArrayLike,
        padding: float | ArrayLike = 0.0,
        pbc: ArrayLike | None = None,
    ) -> "Box":
        """Tight orthogonal box around a point cloud (non-periodic by default)."""
        coords = np.asarray(points, dtype=float)
        if coords.ndim != 2 or coords.shape[1] != 3:
            raise ValueError(f"points must have shape (N, 3), got {coords.shape}")
        if coords.shape[0] == 0:
            raise ValueError("points must contain at least one coordinate")

        pad = np.broadcast_to(np.asarray(padding, dtype=float), (3,))
        mins = coords.min(axis=0) - pad
        maxs = coords.max(axis=0) + pad
        if pbc is None:
            pbc = np.zeros(3, dtype=bool)
        return cls(np.diag(maxs - mins), pbc=pbc, origin=mins)

    @classmethod
    def from_lengths_angles(cls, lengths: ArrayLike, angles: ArrayLike) -> "Box":
        """Triclinic box from edge lengths and lattice angles (degrees)."""
        return cls(cls.calc_matrix_from_lengths_angles(lengths, angles))

    def to_lengths_angles(self) -> tuple[np.ndarray, np.ndarray]:
        """Return ``(lengths, angles)``; angles in degrees."""
        return self.calc_lengths_angles_from_matrix(self.matrix)

    # ────────────────────────────────────────────────────────────────────
    # bounds (xlo / xhi / … / bounds)
    # ────────────────────────────────────────────────────────────────────

    @property
    def xlo(self) -> float:
        return float(self._origin[0])

    @property
    def xhi(self) -> float:
        return float(self._matrix[0, 0] - self._origin[0])

    @property
    def ylo(self) -> float:
        return float(self._origin[1])

    @property
    def yhi(self) -> float:
        return float(self._matrix[1, 1] - self._origin[1])

    @property
    def zlo(self) -> float:
        return float(self._origin[2])

    @property
    def zhi(self) -> float:
        return float(self._matrix[2, 2] - self._origin[2])

    @property
    def bounds(self) -> np.ndarray:
        return np.array(
            [[self.xlo, self.xhi], [self.ylo, self.yhi], [self.zlo, self.zhi]]
        ).T

    # ────────────────────────────────────────────────────────────────────
    # diagonal / off-diagonal accessors
    # ────────────────────────────────────────────────────────────────────

    @property
    def lx(self) -> float:
        return float(self._matrix[0, 0])

    @property
    def ly(self) -> float:
        return float(self._matrix[1, 1])

    @property
    def lz(self) -> float:
        return float(self._matrix[2, 2])

    @property
    def l(self) -> np.ndarray:
        return self._matrix.diagonal().copy()

    @property
    def l_inv(self) -> np.ndarray:
        l = self.l
        with np.errstate(divide="ignore"):
            return np.where(l != 0.0, 1.0 / np.where(l == 0.0, 1.0, l), 0.0)

    @property
    def xy(self) -> float:
        return float(self._matrix[0, 1])

    @property
    def xz(self) -> float:
        return float(self._matrix[0, 2])

    @property
    def yz(self) -> float:
        return float(self._matrix[1, 2])

    @property
    def a(self) -> np.ndarray:
        return self._matrix[:, 0].copy()

    @property
    def b(self) -> np.ndarray:
        return self._matrix[:, 1].copy()

    @property
    def c(self) -> np.ndarray:
        return self._matrix[:, 2].copy()

    @property
    def tilts(self) -> np.ndarray:
        return np.array([self.xy, self.xz, self.yz])

    # ────────────────────────────────────────────────────────────────────
    # PBC flags
    # ────────────────────────────────────────────────────────────────────

    @property
    def periodic(self) -> bool:
        return bool(self._pbc.all())

    @property
    def periodic_x(self) -> bool:
        return bool(self._pbc[0])

    @property
    def periodic_y(self) -> bool:
        return bool(self._pbc[1])

    @property
    def periodic_z(self) -> bool:
        return bool(self._pbc[2])

    @property
    def is_periodic(self) -> bool:
        return self.periodic

    # ────────────────────────────────────────────────────────────────────
    # lengths / angles
    # ────────────────────────────────────────────────────────────────────

    @property
    def lengths(self) -> np.ndarray:
        """Lattice vector magnitudes ``[a, b, c]`` in Angstroms.

        Returns zeros for FREE.
        """
        if self._is_free:
            return np.zeros(3)
        return self.calc_lengths_angles_from_matrix(self.matrix)[0]

    @property
    def angles(self) -> np.ndarray:
        """Lattice angles ``[alpha, beta, gamma]`` in degrees."""
        return self.calc_lengths_angles_from_matrix(self.matrix)[1]

    # ────────────────────────────────────────────────────────────────────
    # static helpers
    # ────────────────────────────────────────────────────────────────────

    @staticmethod
    def check_matrix(matrix: np.ndarray) -> np.ndarray:
        assert isinstance(matrix, np.ndarray), "matrix must be np.ndarray"
        assert matrix.shape == (3, 3), "matrix must be (3, 3)"
        assert not np.isclose(np.linalg.det(matrix), 0), "matrix must be non-singular"
        return matrix

    @staticmethod
    def general2restrict(matrix: np.ndarray) -> np.ndarray:
        """General → restricted-triclinic conversion (LAMMPS convention)."""
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
        return np.array([[ax, bx, cx], [0, by, cy], [0, 0, cz]])

    @staticmethod
    def calc_matrix_from_lengths_angles(
        abc: ArrayLike, angles: ArrayLike
    ) -> np.ndarray:
        a, b, c = abc
        angles = alpha, beta, gamma = np.deg2rad(angles)
        cos_a, cos_b, cos_c = np.cos(angles)

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
        lx, ly, lz = sizes
        xy, xz, yz = tilts
        return np.array([[lx, xy, xz], [0, ly, yz], [0, 0, lz]])

    @staticmethod
    def calc_lengths_angles_from_matrix(
        matrix: np.ndarray,
    ) -> tuple[np.ndarray, np.ndarray]:
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
    def calc_style_from_matrix(matrix: np.ndarray) -> "Box.Style":
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

    # ────────────────────────────────────────────────────────────────────
    # geometry — face distances, wrap, unwrap, diff, dist, fractional
    # ────────────────────────────────────────────────────────────────────

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
        return xyz

    def wrap_orthogonal(self, xyz: np.ndarray) -> np.ndarray:
        lengths = self.lengths
        return xyz - np.floor(xyz / lengths) * lengths

    def wrap_triclinic(self, xyz: np.ndarray) -> np.ndarray:
        xyz = np.atleast_2d(xyz)
        frac = self.make_fractional(xyz)
        frac_wrapped = frac - np.floor(frac)
        return self.make_absolute(frac_wrapped)

    def unwrap(self, xyz: np.ndarray, image: np.ndarray) -> np.ndarray:
        return xyz + image @ self._matrix.T

    def get_images(self, xyz: np.ndarray) -> np.ndarray:
        fractional = self.make_fractional(xyz)
        return np.floor(fractional + 1e-8).astype(int)

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

    def diff_all(self, r1: np.ndarray, r2: np.ndarray) -> np.ndarray:
        all_dr = r1[:, np.newaxis, :] - r2[np.newaxis, :, :]
        original_shape = all_dr.shape
        all_dr = all_dr.reshape(-1, 3)
        all_dr = self.diff_dr(all_dr)
        all_dr = all_dr.reshape(original_shape)
        return all_dr

    def dist(self, r1: np.ndarray, r2: np.ndarray) -> np.ndarray:
        dr = self.diff(r1, r2)
        return np.linalg.norm(dr, axis=1)

    def dist_all(self, r1: np.ndarray, r2: np.ndarray) -> np.ndarray:
        dr = self.diff_all(r1, r2)
        return np.linalg.norm(dr, axis=-1)

    def make_fractional(self, xyz: np.ndarray) -> np.ndarray:
        return (xyz - self._origin) @ self.get_inv().T

    def make_absolute(self, xyz: np.ndarray) -> np.ndarray:
        return xyz @ self._matrix.T + self._origin

    def isin(self, xyz: np.ndarray):
        xyz = np.asarray(xyz)
        fractional = self.make_fractional(xyz)
        return np.all((fractional >= 0) & (fractional < 1), axis=-1)

    def merge(self, other: "Box") -> "Box":
        return Box(matrix=other.matrix)

    def transform(self, transformation_matrix: np.ndarray) -> "Box":
        new_matrix = self._matrix @ transformation_matrix
        return Box(matrix=new_matrix, pbc=self._pbc.copy(), origin=self._origin.copy())

    def to_dict(self) -> dict:
        return {
            "matrix": self._matrix.tolist(),
            "pbc": self._pbc.tolist(),
            "origin": self._origin.tolist(),
        }
