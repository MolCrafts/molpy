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
    >>> print(result.diffusion_coefficients)
"""

from .base import Compute
from .mcd import MCDCompute
from .pmsd import PMSDCompute
from .result import MCDResult, PMSDResult, Result, TimeSeriesResult
from .time_series import TimeAverage, TimeCache, compute_acf, compute_msd

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

