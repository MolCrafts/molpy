"""Pure numeric primitives — ``molrs::op``.

Weighted rigid superposition (Horn quaternion with a scale-free eigen-gap
test), weighted centroids, and NeRF placement of a point from internal
coordinates. Points cross as ``(k, 3)`` float64 arrays and
rotations as ``(3, 3)`` row-major matrices. All computation is in Rust; this
module is a thin re-export of ``_core.op``.
"""

from molpy import _core

DEFAULT_GAP_TOL = _core.op.DEFAULT_GAP_TOL

Superposition = _core.op.Superposition

superpose = _core.op.superpose

centroid = _core.op.centroid

place_from_internal_coords = _core.op.place_from_internal_coords

__all__ = [
    "DEFAULT_GAP_TOL",
    "Superposition",
    "centroid",
    "place_from_internal_coords",
    "superpose",
]
