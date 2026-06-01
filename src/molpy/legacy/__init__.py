"""Legacy NumPy-backed analysis operations.

These are the original pure-NumPy mean-squared-displacement and
cross-displacement-correlation operators. The maintained trunk lives in
:mod:`molpy.compute` (e.g. :class:`molpy.compute.MSD`,
:class:`molpy.compute.MCDCompute`); the operators here are retained for
direct NDArray workflows that do not need trajectory coupling.
"""

from __future__ import annotations

from .cross_correlation import DisplacementCorrelation, displacement_correlation
from .msd import MSD, msd

__all__ = [
    "MSD",
    "msd",
    "DisplacementCorrelation",
    "displacement_correlation",
]
