"""Compute abstraction for molecular computations.

This module provides a unified interface for defining and executing
computational operations on molecular structures. The core abstractions are:

- Result: Base class for computation outputs
- Compute: Base class for computation operations
- TimeSeriesResult, MCDResult, PMSDResult: Results for trajectory analysis
- MCDCompute: Mean Displacement Correlation for diffusion analysis
- PMSDCompute: Polarization Mean Square Displacement for ionic systems

Example usage:
    >>> from molpy.compute import MCDCompute
    >>> from molpy.io import read_h5_trajectory
    >>> traj = read_h5_trajectory("trajectory.h5")
    >>> mcd = MCDCompute(tags=["1"], max_dt=30.0, dt=0.01)
    >>> result = mcd(traj)
    >>> print(result.correlations["1"])  # MSD values at each time lag
"""

from .base import Compute
from .mcd import MCDCompute
from .pmsd import PMSDCompute
from .result import MCDResult, PMSDResult, Result, TimeSeriesResult
from .time_series import TimeAverage, TimeCache, compute_acf, compute_msd

# Optional RDKit compute nodes
try:  # pragma: no cover
    from .rdkit import Generate3D, OptimizeGeometry

    _HAS_RDKIT = True
except ModuleNotFoundError:  # rdkit missing
    _HAS_RDKIT = False
    Generate3D = None  # type: ignore[assignment]
    OptimizeGeometry = None  # type: ignore[assignment]

__all__ = [
    "Compute",
    "Result",
    "TimeSeriesResult",
    "MCDResult",
    "PMSDResult",
    "MCDCompute",
    "PMSDCompute",
    "TimeCache",
    "TimeAverage",
    "compute_msd",
    "compute_acf",
]

if _HAS_RDKIT:
    __all__ += ["Generate3D", "OptimizeGeometry"]
