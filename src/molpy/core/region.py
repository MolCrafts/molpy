"""Selection sugar over molrs' native geometric regions."""

from __future__ import annotations

from abc import abstractmethod

import molrs
import numpy as np
from numpy.typing import ArrayLike

from molrs import Block
from .selector import MaskPredicate


class Region(MaskPredicate):
    """Mixin that adds ``mask(Block)`` to native regions.

    A block stores coordinates in ``x``/``y``/``z``, so that is where a region
    reads them. There is no configurable coordinate field: the alternative it
    used to allow — one packed Nx3 column paralleling the canonical three —
    is not a second convention molpy supports, and defaulting to it meant
    masking an ordinary block raised ``KeyError``.
    """

    def mask(self, block: Block) -> np.ndarray:  # type: ignore[override]
        return self.isin(block["x", "y", "z"])

    @abstractmethod
    def isin(self, xyz: np.ndarray) -> np.ndarray: ...


class _NativeRegionSugar:
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
        return self.isin(block["x", "y", "z"])

    def __and__(self, other):
        return AndRegion(self, other)

    def __or__(self, other):
        return OrRegion(self, other)

    def __invert__(self):
        return NotRegion(self)


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
    ):
        lengths_array = np.asarray(lengths, dtype=float)
        origin_array = (
            np.zeros(3) if origin is None else np.asarray(origin, dtype=float)
        )
        return super().__new__(cls, origin_array, lengths_array)

    def __init__(
        self,
        lengths: ArrayLike,
        origin: ArrayLike | None = None,
    ) -> None:
        self.lengths = np.asarray(lengths, dtype=float)
        self.origin = np.zeros(3) if origin is None else np.asarray(origin, dtype=float)

    def __eq__(self, other: object) -> bool:
        return isinstance(other, BoxRegion) and bool(
            np.allclose(self.lengths, other.lengths)
            and np.allclose(self.origin, other.origin)
        )

    def __repr__(self) -> str:
        return f"BoxRegion(lengths={self.lengths}, origin={self.origin})"


class Cube(BoxRegion):
    def __new__(cls, edge: float, origin: ArrayLike | None = None):
        return super().__new__(cls, np.full(3, edge), origin)

    def __init__(self, edge: float, origin: ArrayLike | None = None) -> None:
        if edge <= 0:
            raise ValueError(f"edge must be positive, got {edge}")
        super().__init__(np.full(3, edge), origin)
        self.edge = float(edge)

    def __repr__(self) -> str:
        return f"Cube(edge={self.edge}, origin={self.origin})"


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
    ):
        center_array = (
            np.zeros(3) if center is None else np.asarray(center, dtype=float)
        )
        return super().__new__(cls, center_array, radius)

    def __init__(
        self,
        radius: float,
        center: ArrayLike | None = None,
    ) -> None:
        if radius <= 0:
            raise ValueError(f"radius must be positive, got {radius}")
        self.radius = float(radius)
        self.center = np.zeros(3) if center is None else np.asarray(center, dtype=float)

    def __eq__(self, other: object) -> bool:
        return isinstance(other, SphereRegion) and bool(
            np.isclose(self.radius, other.radius)
            and np.allclose(self.center, other.center)
        )

    def __repr__(self) -> str:
        return f"SphereRegion(radius={self.radius}, center={self.center})"


class _ComposedRegion(molrs.Region, _NativeRegionSugar, Region):
    isin = _NativeRegionSugar.isin
    bounds = _NativeRegionSugar.bounds
    mask = _NativeRegionSugar.mask
    __and__ = _NativeRegionSugar.__and__
    __or__ = _NativeRegionSugar.__or__
    __invert__ = _NativeRegionSugar.__invert__


class AndRegion(_ComposedRegion):
    def __new__(cls, a, b):
        return super().__new__(
            cls,
            molrs.Cuboid.__and__(a, b)
            if isinstance(a, molrs.Cuboid)
            else molrs.Sphere.__and__(a, b)
            if isinstance(a, molrs.Sphere)
            else molrs.Region.__and__(a, b),
        )

    def __init__(self, a, b) -> None:
        self.a, self.b = a, b


class OrRegion(_ComposedRegion):
    def __new__(cls, a, b):
        return super().__new__(
            cls,
            molrs.Cuboid.__or__(a, b)
            if isinstance(a, molrs.Cuboid)
            else molrs.Sphere.__or__(a, b)
            if isinstance(a, molrs.Sphere)
            else molrs.Region.__or__(a, b),
        )

    def __init__(self, a, b) -> None:
        self.a, self.b = a, b


class NotRegion(_ComposedRegion):
    def __new__(cls, a):
        return super().__new__(
            cls,
            molrs.Cuboid.__invert__(a)
            if isinstance(a, molrs.Cuboid)
            else molrs.Sphere.__invert__(a)
            if isinstance(a, molrs.Sphere)
            else molrs.Region.__invert__(a),
        )

    def __init__(self, a) -> None:
        self.a = a


__all__ = [
    "AndRegion",
    "BoxRegion",
    "Cube",
    "NotRegion",
    "OrRegion",
    "Region",
    "SphereRegion",
]
