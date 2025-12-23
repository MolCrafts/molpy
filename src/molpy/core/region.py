from abc import abstractmethod

import numpy as np
from numpy.typing import ArrayLike

from .frame import Block
from .selector import MaskPredicate

__all__ = [
    "AndRegion",
    "BoxRegion",
    "Cube",
    "NotRegion",
    "OrRegion",
    "Region",
    "SphereRegion",
]


class Region(MaskPredicate):
    """Geometric region that is also a MaskPredicate.

    Responsibilities:
    - purely geometric point-in-region logic
    - combination via & | ~ returning Region objects
    - expose axis-aligned bounding box via .bounds (shape (2,3))
    - implement mask(block) by extracting coordinates from block
    """

    def __init__(self, coord_field: str = "xyz"):
        self.coord_field = coord_field

    @abstractmethod
    def isin(self, xyz: np.ndarray) -> np.ndarray:
        """Return boolean mask for points inside the region.
        Parameters
        ----------
        xyz : (N,3) array
        Returns
        -------
        mask : (N,) bool array
        """
        ...

    @property
    @abstractmethod
    def bounds(self) -> np.ndarray:
        """Return axis-aligned bounding box as array[[xmin,ymin,zmin],[xmax,ymax,zmax]]."""
        ...

    def mask(self, block: Block) -> np.ndarray:  # type: ignore[override]
        """Extract coordinates from block and apply geometric predicate."""
        coords = block[self.coord_field]
        return self.isin(coords)

    # boolean combinators - return Region instances for geometric semantics
    def __and__(self, other: "Region") -> "Region":  # type: ignore[override]
        return AndRegion(self, other)

    def __or__(self, other: "Region") -> "Region":  # type: ignore[override]
        return OrRegion(self, other)

    def __invert__(self) -> "Region":  # type: ignore[override]
        return NotRegion(self)

    __rand__ = __and__
    __ror__ = __or__


class PeriodicBoundary(Region): ...


class BoxRegion(Region):
    def __init__(
        self,
        lengths: ArrayLike,
        origin: ArrayLike | None = None,
        coord_field: str = "xyz",
    ):
        super().__init__(coord_field)
        self.lengths = np.asarray(lengths, dtype=float)
        if origin is None:
            origin = np.zeros(3)
        self.origin = np.asarray(origin, dtype=float)

        # Validate dimensions
        if self.lengths.shape != (3,):
            raise ValueError(f"lengths must be 3D, got shape {self.lengths.shape}")
        if self.origin.shape != (3,):
            raise ValueError(f"origin must be 3D, got shape {self.origin.shape}")

    def mask(self, block: Block) -> np.ndarray:  # type: ignore[override]
        """Extract coordinates from block and apply geometric predicate."""
        coords = block[self.coord_field]
        return self.isin(coords)

    def isin(self, xyz: np.ndarray) -> np.ndarray:
        xyz = np.asarray(xyz, dtype=float)
        lower = self.origin
        upper = self.origin + self.lengths
        return np.all((xyz >= lower) & (xyz <= upper), axis=1)

    @property
    def bounds(self) -> np.ndarray:
        return np.vstack([self.origin, self.origin + self.lengths])

    def __eq__(self, other) -> bool:
        if not isinstance(other, BoxRegion):
            return False
        return (
            bool(np.allclose(self.lengths, other.lengths))
            and bool(np.allclose(self.origin, other.origin))
            and self.coord_field == other.coord_field
        )

    def __repr__(self) -> str:
        return f"BoxRegion(lengths={self.lengths}, origin={self.origin}, coord_field='{self.coord_field}')"


class Cube(BoxRegion):
    def __init__(
        self, edge: float, origin: ArrayLike | None = None, coord_field: str = "xyz"
    ):
        if origin is None:
            origin = np.zeros(3)
        lengths = np.array([edge, edge, edge])
        super().__init__(lengths, origin, coord_field)
        self.edge = float(edge)

        # Validate edge length
        if self.edge <= 0:
            raise ValueError(f"edge must be positive, got {self.edge}")

    def __eq__(self, other) -> bool:
        if not isinstance(other, Cube):
            return False
        return (
            bool(np.isclose(self.edge, other.edge))
            and bool(np.allclose(self.origin, other.origin))
            and self.coord_field == other.coord_field
        )

    def __repr__(self) -> str:
        return f"Cube(edge={self.edge}, origin={self.origin}, coord_field='{self.coord_field}')"


class SphereRegion(Region):
    def __init__(
        self, radius: float, center: ArrayLike | None = None, coord_field: str = "xyz"
    ):
        super().__init__(coord_field)
        self.radius = float(radius)
        if center is None:
            center = np.zeros(3)
        self.center = np.asarray(center, dtype=float)

        # Validate dimensions and values
        if self.radius <= 0:
            raise ValueError(f"radius must be positive, got {self.radius}")
        if self.center.shape != (3,):
            raise ValueError(f"center must be 3D, got shape {self.center.shape}")

    def mask(self, block: Block) -> np.ndarray:  # type: ignore[override]
        """Extract coordinates from block and apply geometric predicate."""
        coords = block[self.coord_field]
        return self.isin(coords)

    @property
    def bounds(self) -> np.ndarray:
        c = self.center
        r = self.radius
        return np.vstack([c - r, c + r])

    def isin(self, xyz: np.ndarray) -> np.ndarray:
        xyz = np.asarray(xyz, dtype=float)
        diff = xyz - self.center
        return np.einsum("ij,ij->i", diff, diff) <= self.radius * self.radius

    def __eq__(self, other) -> bool:
        if not isinstance(other, SphereRegion):
            return False
        return (
            bool(np.isclose(self.radius, other.radius))
            and bool(np.allclose(self.center, other.center))
            and self.coord_field == other.coord_field
        )

    def __repr__(self) -> str:
        return f"SphereRegion(radius={self.radius}, center={self.center}, coord_field='{self.coord_field}')"


class AndRegion(Region):
    def __init__(self, a: Region, b: Region, coord_field: str = "xyz"):
        super().__init__(coord_field)
        self.a = a
        self.b = b

    def isin(self, xyz: np.ndarray) -> np.ndarray:  # type: ignore[override]
        return self.a.isin(xyz) & self.b.isin(xyz)

    @property
    def bounds(self) -> np.ndarray:
        # intersection box = overlap of bounds
        a_b = self.a.bounds
        b_b = self.b.bounds
        lower = np.maximum(a_b[0], b_b[0])
        upper = np.minimum(a_b[1], b_b[1])
        return np.vstack([lower, upper])

    def __eq__(self, other) -> bool:
        if not isinstance(other, AndRegion):
            return False
        return (
            self.a == other.a
            and self.b == other.b
            and self.coord_field == other.coord_field
        )

    def __repr__(self) -> str:
        return f"AndRegion(a={self.a}, b={self.b}, coord_field='{self.coord_field}')"


class OrRegion(Region):
    def __init__(self, a: Region, b: Region, coord_field: str = "xyz"):
        super().__init__(coord_field)
        self.a = a
        self.b = b

    def isin(self, xyz: np.ndarray) -> np.ndarray:  # type: ignore[override]
        return self.a.isin(xyz) | self.b.isin(xyz)

    @property
    def bounds(self) -> np.ndarray:
        a_b = self.a.bounds
        b_b = self.b.bounds
        lower = np.minimum(a_b[0], b_b[0])
        upper = np.maximum(a_b[1], b_b[1])
        return np.vstack([lower, upper])

    def __eq__(self, other) -> bool:
        if not isinstance(other, OrRegion):
            return False
        return (
            self.a == other.a
            and self.b == other.b
            and self.coord_field == other.coord_field
        )

    def __repr__(self) -> str:
        return f"OrRegion(a={self.a}, b={self.b}, coord_field='{self.coord_field}')"


class NotRegion(Region):
    def __init__(self, a: Region, coord_field: str = "xyz"):
        super().__init__(coord_field)
        self.a = a

    def isin(self, xyz: np.ndarray) -> np.ndarray:  # type: ignore[override]
        return ~self.a.isin(xyz)

    @property
    def bounds(self) -> np.ndarray:
        # Complement is unbounded; approximate by original bounds for practicality.
        return self.a.bounds

    def __eq__(self, other) -> bool:
        if not isinstance(other, NotRegion):
            return False
        return self.a == other.a and self.coord_field == other.coord_field

    def __repr__(self) -> str:
        return f"NotRegion(a={self.a}, coord_field='{self.coord_field}')"
