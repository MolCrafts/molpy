"""Selection sugar over molrs' native geometric regions."""

from __future__ import annotations

from abc import abstractmethod

import molrs
import numpy as np
from numpy.typing import ArrayLike

from molrs import Block
from .selector import MaskPredicate


class Region(MaskPredicate):
    """Mixin that adds ``coord_field`` and ``mask(Block)`` to native regions."""

    coord_field: str

    def mask(self, block: Block) -> np.ndarray:  # type: ignore[override]
        return self.isin(block[self.coord_field])

    @abstractmethod
    def isin(self, xyz: np.ndarray) -> np.ndarray: ...


class _NativeRegionSugar:
    coord_field: str

    def isin(self, xyz: np.ndarray):
        array = np.asarray(xyz, dtype=float)
        if array.ndim == 1:
            return bool(self.contains(array.reshape(1, 3))[0])
        return self.contains(array)

    @property
    def bounds(self) -> np.ndarray:
        if isinstance(self, molrs.Cuboid):
            native = molrs.Cuboid.bounds(self)
        elif isinstance(self, molrs.Sphere):
            native = molrs.Sphere.bounds(self)
        else:
            native = molrs.Region.bounds(self)
        return np.asarray(native).T

    def mask(self, block: Block) -> np.ndarray:
        return self.isin(block[self.coord_field])

    def __and__(self, other):
        return AndRegion(self, other, self.coord_field)

    def __or__(self, other):
        return OrRegion(self, other, self.coord_field)

    def __invert__(self):
        return NotRegion(self, self.coord_field)


class BoxRegion(molrs.Cuboid, _NativeRegionSugar, Region):
    isin = _NativeRegionSugar.isin
    bounds = _NativeRegionSugar.bounds
    mask = _NativeRegionSugar.mask
    __and__ = _NativeRegionSugar.__and__
    __or__ = _NativeRegionSugar.__or__
    __invert__ = _NativeRegionSugar.__invert__

    def __new__(
        cls,
        lengths: ArrayLike,
        origin: ArrayLike | None = None,
        coord_field: str = "xyz",
    ):
        del coord_field
        lengths_array = np.asarray(lengths, dtype=float)
        origin_array = (
            np.zeros(3) if origin is None else np.asarray(origin, dtype=float)
        )
        return super().__new__(cls, origin_array, lengths_array)

    def __init__(
        self,
        lengths: ArrayLike,
        origin: ArrayLike | None = None,
        coord_field: str = "xyz",
    ) -> None:
        self.lengths = np.asarray(lengths, dtype=float)
        self.origin = np.zeros(3) if origin is None else np.asarray(origin, dtype=float)
        self.coord_field = coord_field

    def __eq__(self, other: object) -> bool:
        return isinstance(other, BoxRegion) and bool(
            np.allclose(self.lengths, other.lengths)
            and np.allclose(self.origin, other.origin)
            and self.coord_field == other.coord_field
        )

    def __repr__(self) -> str:
        return f"BoxRegion(lengths={self.lengths}, origin={self.origin}, coord_field='{self.coord_field}')"


class Cube(BoxRegion):
    def __new__(
        cls, edge: float, origin: ArrayLike | None = None, coord_field: str = "xyz"
    ):
        return super().__new__(cls, np.full(3, edge), origin, coord_field)

    def __init__(
        self, edge: float, origin: ArrayLike | None = None, coord_field: str = "xyz"
    ) -> None:
        if edge <= 0:
            raise ValueError(f"edge must be positive, got {edge}")
        super().__init__(np.full(3, edge), origin, coord_field)
        self.edge = float(edge)

    def __repr__(self) -> str:
        return f"Cube(edge={self.edge}, origin={self.origin}, coord_field='{self.coord_field}')"


class SphereRegion(molrs.Sphere, _NativeRegionSugar, Region):
    isin = _NativeRegionSugar.isin
    bounds = _NativeRegionSugar.bounds
    mask = _NativeRegionSugar.mask
    __and__ = _NativeRegionSugar.__and__
    __or__ = _NativeRegionSugar.__or__
    __invert__ = _NativeRegionSugar.__invert__

    def __new__(
        cls,
        radius: float,
        center: ArrayLike | None = None,
        coord_field: str = "xyz",
    ):
        del coord_field
        center_array = (
            np.zeros(3) if center is None else np.asarray(center, dtype=float)
        )
        return super().__new__(cls, center_array, radius)

    def __init__(
        self,
        radius: float,
        center: ArrayLike | None = None,
        coord_field: str = "xyz",
    ) -> None:
        if radius <= 0:
            raise ValueError(f"radius must be positive, got {radius}")
        self.radius = float(radius)
        self.center = np.zeros(3) if center is None else np.asarray(center, dtype=float)
        self.coord_field = coord_field

    def __eq__(self, other: object) -> bool:
        return isinstance(other, SphereRegion) and bool(
            np.isclose(self.radius, other.radius)
            and np.allclose(self.center, other.center)
            and self.coord_field == other.coord_field
        )

    def __repr__(self) -> str:
        return f"SphereRegion(radius={self.radius}, center={self.center}, coord_field='{self.coord_field}')"


class _ComposedRegion(molrs.Region, _NativeRegionSugar, Region):
    isin = _NativeRegionSugar.isin
    bounds = _NativeRegionSugar.bounds
    mask = _NativeRegionSugar.mask
    __and__ = _NativeRegionSugar.__and__
    __or__ = _NativeRegionSugar.__or__
    __invert__ = _NativeRegionSugar.__invert__


class AndRegion(_ComposedRegion):
    def __new__(cls, a, b, coord_field: str = "xyz"):
        del coord_field
        return super().__new__(
            cls,
            molrs.Cuboid.__and__(a, b)
            if isinstance(a, molrs.Cuboid)
            else molrs.Sphere.__and__(a, b)
            if isinstance(a, molrs.Sphere)
            else molrs.Region.__and__(a, b),
        )

    def __init__(self, a, b, coord_field: str = "xyz") -> None:
        self.a, self.b, self.coord_field = a, b, coord_field


class OrRegion(_ComposedRegion):
    def __new__(cls, a, b, coord_field: str = "xyz"):
        del coord_field
        return super().__new__(
            cls,
            molrs.Cuboid.__or__(a, b)
            if isinstance(a, molrs.Cuboid)
            else molrs.Sphere.__or__(a, b)
            if isinstance(a, molrs.Sphere)
            else molrs.Region.__or__(a, b),
        )

    def __init__(self, a, b, coord_field: str = "xyz") -> None:
        self.a, self.b, self.coord_field = a, b, coord_field


class NotRegion(_ComposedRegion):
    def __new__(cls, a, coord_field: str = "xyz"):
        del coord_field
        return super().__new__(
            cls,
            molrs.Cuboid.__invert__(a)
            if isinstance(a, molrs.Cuboid)
            else molrs.Sphere.__invert__(a)
            if isinstance(a, molrs.Sphere)
            else molrs.Region.__invert__(a),
        )

    def __init__(self, a, coord_field: str = "xyz") -> None:
        self.a, self.coord_field = a, coord_field


__all__ = [
    "AndRegion",
    "BoxRegion",
    "Cube",
    "NotRegion",
    "OrRegion",
    "Region",
    "SphereRegion",
]
