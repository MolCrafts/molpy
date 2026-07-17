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
        (e.g. ``Box.Style.ORTHOGONAL == "orthogonal"``), letting ``frame.simbox``
        (a molrs box) interoperate with molpy style checks.
        """

        FREE = "free"
        ORTHOGONAL = "orthogonal"
        TRICLINIC = "triclinic"

    # FREE boxes carry an identity placeholder cell so the molrs base (which
    # rejects det(h)==0) can hold them and its geometry ops degrade to no-ops.
    # Their "no-cell" nature is recorded in the molrs base as
    # ``cell_defined=False`` (the single source of truth) — there is no
    # Python-side ``_is_free`` shadow.
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
        # A FREE box defaults to non-periodic on every axis (pbc all-False), in
        # addition to being marked cell_defined=False below.
        if pbc is None and is_free:
            pbc_arr = np.zeros(3, dtype=bool)
        else:
            pbc_arr = cls._normalize_pbc(pbc)
        origin_arr = cls._normalize_origin(origin)
        # ``cell_defined=False`` records the FREE (no-cell) nature in the molrs
        # base; ``is_free`` / ``style`` / ``volume`` read it back from there.
        return super().__new__(
            cls, h, origin=origin_arr, pbc=pbc_arr, cell_defined=not is_free
        )

    def __init__(
        self,
        matrix: ArrayLike | None = None,
        pbc: ArrayLike | None = None,
        origin: ArrayLike | None = None,
    ):
        # PyO3 constructs the Rust base in __new__. The base exposes
        # object.__init__, so this Python initializer deliberately consumes the
        # constructor arguments after __new__ has completed the initialization.
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
        # The molrs base constructor is the canonical matrix validator and
        # rejects singular cells.
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
        if self.is_free:
            return np.zeros((3, 3))
        return np.asarray(self.h)

    @property
    def is_free(self) -> bool:
        """``True`` if this box is FREE (no defined cell, zero volume).

        Derived from the molrs base's ``cell_defined`` flag — the single source
        of truth — not a Python-side shadow.
        """
        return not self.cell_defined

    @property
    def style(self) -> "Box.Style":
        """FREE / ORTHOGONAL / TRICLINIC depending on the matrix shape."""
        if self.is_free:
            return Box.Style.FREE
        return self.calc_style_from_matrix(self.matrix)

    @property
    def volume(self) -> float:
        """Box volume in Angstroms³ (zero for FREE)."""
        if self.is_free:
            return 0.0
        # Delegate to the inherited molrs Rust kernel (the FREE placeholder
        # matrix has no meaningful volume, hence the guard above).
        return float(molrs.Box.volume(self))

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
        return Box(
            self.matrix * other,
            np.asarray(self.pbc).copy(),
            np.asarray(self.origin).copy(),
        )

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

        Accepts a molpy ``Box`` or a bare ``molrs.Box`` (e.g. ``frame.simbox``),
        reading only the public ``matrix`` / ``pbc`` / ``origin`` accessors so it
        works across both. A free source box reconstructs a free molpy box.
        """
        if not getattr(box, "cell_defined", True):
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

        pad = np.ascontiguousarray(
            np.broadcast_to(np.asarray(padding, dtype=float), (3,))
        )
        if pbc is None:
            pbc = np.zeros(3, dtype=bool)
        native = molrs.Box.from_bounds(coords, pad, np.asarray(pbc, dtype=bool))
        return cls.from_box(native)

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
        return float(self.bounds[0, 0])

    @property
    def xhi(self) -> float:
        return float(self.bounds[1, 0])

    @property
    def ylo(self) -> float:
        return float(self.bounds[0, 1])

    @property
    def yhi(self) -> float:
        return float(self.bounds[1, 1])

    @property
    def zlo(self) -> float:
        return float(self.bounds[0, 2])

    @property
    def zhi(self) -> float:
        return float(self.bounds[1, 2])

    @property
    def bounds(self) -> np.ndarray:
        return np.asarray(super().bounds).T

    # ────────────────────────────────────────────────────────────────────
    # diagonal / off-diagonal accessors
    # ────────────────────────────────────────────────────────────────────

    @property
    def lx(self) -> float:
        return float(self.matrix[0, 0])

    @property
    def ly(self) -> float:
        return float(self.matrix[1, 1])

    @property
    def lz(self) -> float:
        return float(self.matrix[2, 2])

    @property
    def l(self) -> np.ndarray:
        return self.matrix.diagonal().copy()

    @property
    def l_inv(self) -> np.ndarray:
        l = self.l
        with np.errstate(divide="ignore"):
            return np.where(l != 0.0, 1.0 / np.where(l == 0.0, 1.0, l), 0.0)

    @property
    def xy(self) -> float:
        return float(self.matrix[0, 1])

    @property
    def xz(self) -> float:
        return float(self.matrix[0, 2])

    @property
    def yz(self) -> float:
        return float(self.matrix[1, 2])

    @property
    def a(self) -> np.ndarray:
        return self.matrix[:, 0].copy()

    @property
    def b(self) -> np.ndarray:
        return self.matrix[:, 1].copy()

    @property
    def c(self) -> np.ndarray:
        return self.matrix[:, 2].copy()

    @property
    def tilts(self) -> np.ndarray:
        if self.is_free:
            return np.array([self.xy, self.xz, self.yz])
        # Inherited molrs Rust kernel (returns [xy, xz, yz]).
        return np.asarray(super().tilts)

    # ────────────────────────────────────────────────────────────────────
    # PBC flags
    # ────────────────────────────────────────────────────────────────────

    @property
    def periodic(self) -> bool:
        return bool(np.asarray(self.pbc).all())

    @property
    def periodic_x(self) -> bool:
        return bool(self.pbc[0])

    @property
    def periodic_y(self) -> bool:
        return bool(self.pbc[1])

    @property
    def periodic_z(self) -> bool:
        return bool(self.pbc[2])

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
        if self.is_free:
            return np.zeros(3)
        # Inherited molrs Rust kernel (FREE placeholder has no lattice).
        return np.asarray(super().lengths)

    @property
    def angles(self) -> np.ndarray:
        """Lattice angles ``[alpha, beta, gamma]`` in degrees."""
        if self.is_free:
            return np.zeros(3)
        return np.asarray(super().angles)

    # ────────────────────────────────────────────────────────────────────
    # static helpers
    # ────────────────────────────────────────────────────────────────────

    @staticmethod
    def check_matrix(matrix: np.ndarray) -> np.ndarray:
        assert isinstance(matrix, np.ndarray), "matrix must be np.ndarray"
        assert matrix.shape == (3, 3), "matrix must be (3, 3)"
        try:
            molrs.Box(matrix)
        except ValueError as exc:
            raise AssertionError("matrix must be non-singular") from exc
        return matrix

    @staticmethod
    def general2restrict(matrix: np.ndarray) -> np.ndarray:
        """General → restricted-triclinic conversion (LAMMPS convention)."""
        return np.asarray(molrs.Box.restricted_matrix(np.asarray(matrix, dtype=float)))

    @staticmethod
    def calc_matrix_from_lengths_angles(
        abc: ArrayLike, angles: ArrayLike
    ) -> np.ndarray:
        return np.asarray(molrs.Box.matrix_from_lengths_angles(abc, angles))

    @staticmethod
    def calc_matrix_from_size_tilts(sizes, tilts) -> np.ndarray:
        return np.asarray(molrs.Box.matrix_from_lengths_tilts(sizes, tilts))

    @staticmethod
    def calc_lengths_angles_from_matrix(
        matrix: np.ndarray,
    ) -> tuple[np.ndarray, np.ndarray]:
        native = molrs.Box(np.asarray(matrix, dtype=float))
        return np.asarray(native.lengths), np.asarray(native.angles)

    @staticmethod
    def calc_style_from_matrix(matrix: np.ndarray) -> "Box.Style":
        if np.allclose(matrix, np.zeros(3)):
            return Box.Style.FREE
        return Box.Style(molrs.Box(np.asarray(matrix, dtype=float)).style)

    # ────────────────────────────────────────────────────────────────────
    # geometry — face distances, wrap, unwrap, diff, dist, fractional
    # ────────────────────────────────────────────────────────────────────

    def get_distance_between_faces(self) -> np.ndarray:
        if self.is_free:
            return np.zeros(3)
        return np.asarray(self.nearest_plane_distance)

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
        # Inherited molrs Rust kernel (PBC-aware, per-axis).
        return molrs.Box.wrap(self, np.asarray(xyz))

    def wrap_triclinic(self, xyz: np.ndarray) -> np.ndarray:
        # Inherited molrs Rust kernel (fractional wrap via to_frac/to_cart).
        return molrs.Box.wrap(self, np.asarray(xyz))

    def unwrap(self, xyz: np.ndarray, image: np.ndarray) -> np.ndarray:
        return molrs.Box.unwrap(
            self, np.asarray(xyz), np.asarray(image, dtype=np.int64)
        )

    def get_images(self, xyz: np.ndarray) -> np.ndarray:
        return molrs.Box.images(self, np.asarray(xyz))

    def get_inv(self) -> np.ndarray:
        return np.asarray(self.inverse)

    def diff_dr(self, dr: np.ndarray) -> np.ndarray:
        dr = np.asarray(dr, dtype=float)
        if self.style is Box.Style.FREE:
            return dr
        # Minimum-image displacement via the inherited molrs Rust kernel:
        # ``delta(a, b)`` is ``minimum_image(b - a)``, so ``delta(0, dr)`` is
        # ``minimum_image(dr)``.
        if dr.ndim == 1:
            return molrs.Box.delta(
                self, np.zeros((1, 3)), dr.reshape(1, 3), minimum_image=True
            )[0]
        return molrs.Box.delta(self, np.zeros_like(dr), dr, minimum_image=True)

    def diff(self, r1: np.ndarray, r2: np.ndarray) -> np.ndarray:
        return self.diff_dr(r1 - r2)

    def diff_all(self, r1: np.ndarray, r2: np.ndarray) -> np.ndarray:
        # Native pairwise_delta returns r2-r1; reverse the arguments to retain
        # molpy's historical r1-r2 convention.
        return molrs.Box.pairwise_delta(self, np.asarray(r2), np.asarray(r1)).transpose(
            1, 0, 2
        )

    def dist(self, r1: np.ndarray, r2: np.ndarray) -> np.ndarray:
        return molrs.Box.distances(self, np.asarray(r1), np.asarray(r2))

    def dist_all(self, r1: np.ndarray, r2: np.ndarray) -> np.ndarray:
        return molrs.Box.pairwise_distances(self, np.asarray(r1), np.asarray(r2))

    def make_fractional(self, xyz: np.ndarray) -> np.ndarray:
        # Cartesian -> fractional via the inherited molrs Rust kernel (to_frac);
        # the free-box placeholder-identity matrix gives the same result as the
        # former NumPy ``(xyz - origin) @ inv(matrix).T``.
        arr = np.asarray(xyz, dtype=float)
        if arr.ndim == 1:
            return self.to_frac(arr.reshape(1, 3))[0]
        return self.to_frac(arr)

    def make_absolute(self, xyz: np.ndarray) -> np.ndarray:
        # Fractional -> Cartesian via the inherited molrs Rust kernel (to_cart).
        arr = np.asarray(xyz, dtype=float)
        if arr.ndim == 1:
            return self.to_cart(arr.reshape(1, 3))[0]
        return self.to_cart(arr)

    def isin(self, xyz: np.ndarray):
        # Inside-primary-cell test via the inherited molrs Rust kernel.
        arr = np.asarray(xyz, dtype=float)
        if arr.ndim == 1:
            return bool(molrs.Box.isin(self, arr.reshape(1, 3))[0])
        return molrs.Box.isin(self, arr)

    def merge(self, other: "Box") -> "Box":
        return Box(matrix=other.matrix)

    def transform(self, transformation_matrix: np.ndarray) -> "Box":
        return Box.from_box(
            molrs.Box.transformed(self, np.asarray(transformation_matrix, dtype=float))
        )

    def to_dict(self) -> dict:
        return {
            "matrix": self.matrix.tolist(),
            "pbc": list(self.pbc),
            "origin": list(self.origin),
        }
