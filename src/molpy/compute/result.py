"""Result classes for compute operations.

This module defines result types returned by compute operations.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np
from numpy.typing import NDArray


@dataclass
class Result:
    """Base class for computation results.

    Subclasses should define specific fields for their result data.
    """

    meta: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        """Convert result to dictionary representation."""
        return {k: v for k, v in self.__dict__.items()}


@dataclass
class TimeSeriesResult(Result):
    """Base class for time-series analysis results.

    Attributes:
        time: Time points for the analysis (in ps or frames)
    """

    time: NDArray[np.float64] = field(default_factory=lambda: np.array([]))


@dataclass
class MCDResult(TimeSeriesResult):
    """Results from Mean Displacement Correlation calculation.

    Attributes:
        time: Time lag values (in ps)
        correlations: Dictionary mapping tag names to correlation function arrays (MSD values).
            Each array has shape (n_time_lags,)
    """

    correlations: dict[str, NDArray[np.float64]] = field(default_factory=dict)


@dataclass
class PMSDResult(TimeSeriesResult):
    """Results from Polarization Mean Square Displacement calculation.

    Attributes:
        time: Time lag values (in ps)
        pmsd: Polarization MSD values at each time lag, shape (n_time_lags,)
    """

    pmsd: NDArray[np.float64] = field(default_factory=lambda: np.array([]))


@dataclass
class ACFResult(TimeSeriesResult):
    """Autocorrelation function result.

    Attributes:
        time: Time lag values (in ps)
        acf: Autocorrelation values at each time lag, shape (n_lags,)
        n_lags: Number of time lags
    """

    acf: NDArray[np.float64] = field(default_factory=lambda: np.array([]))
    n_lags: int = 0


@dataclass
class SpectralResult(TimeSeriesResult):
    """Frequency-domain spectrum result.

    Attributes:
        time: Frequency values (in THz)
        frequency: Angular frequency grid, shape (n_freq,)
        spectrum: Spectral density, shape (n_freq,)
    """

    frequency: NDArray[np.float64] = field(default_factory=lambda: np.array([]))
    spectrum: NDArray[np.float64] = field(default_factory=lambda: np.array([]))


@dataclass
class DielectricResult(TimeSeriesResult):
    """Single-route dielectric susceptibility result.

    Attributes:
        time: Frequency grid (in THz)
        frequency: Angular frequencies, shape (n_freq,)
        epsilon_real: Real part epsilon'(omega), shape (n_freq,)
        epsilon_imag: Imaginary part epsilon''(omega), shape (n_freq,)
        epsilon_static: Static dielectric constant epsilon(0)
        epsilon_inf: High-frequency dielectric constant
        route: Computation route ("einstein-helfand" or "green-kubo")
        component: System component ("full", "water", "ion")
        conductivity: Optional conductivity spectrum sigma(omega), shape (n_freq,)
    """

    frequency: NDArray[np.float64] = field(default_factory=lambda: np.array([]))
    epsilon_real: NDArray[np.float64] = field(default_factory=lambda: np.array([]))
    epsilon_imag: NDArray[np.float64] = field(default_factory=lambda: np.array([]))
    epsilon_static: float = 0.0
    epsilon_inf: float = 1.0
    route: str = ""
    component: str = ""
    conductivity: NDArray[np.float64] | None = None


@dataclass
class DielectricSusceptibilityResult(Result):
    """Aggregate dielectric susceptibility result.

    Attributes:
        results: Mapping from route-component key to DielectricResult
        metadata: Trajectory parameters and computation info
    """

    results: dict[str, DielectricResult] = field(default_factory=dict)
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        """Serialize with nested DielectricResult recursion."""
        d = super().to_dict()
        d["results"] = {k: v.to_dict() for k, v in self.results.items()}
        return d
