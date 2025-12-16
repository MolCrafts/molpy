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
        correlations: Dictionary mapping tag names to correlation function arrays.
            Each array has shape (n_time_lags,)
        diffusion_coefficients: Dictionary mapping tag names to fitted diffusion
            coefficients (in Å²/ps or appropriate units)
    """

    correlations: dict[str, NDArray[np.float64]] = field(default_factory=dict)
    diffusion_coefficients: dict[str, float] = field(default_factory=dict)


@dataclass
class PMSDResult(TimeSeriesResult):
    """Results from Polarization Mean Square Displacement calculation.

    Attributes:
        time: Time lag values (in ps)
        pmsd: Polarization MSD values at each time lag, shape (n_time_lags,)
    """

    pmsd: NDArray[np.float64] = field(default_factory=lambda: np.array([]))
