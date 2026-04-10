"""Mean Squared Displacement computation.

Operates on plain NDArrays — no trajectory coupling.
"""

from __future__ import annotations

from dataclasses import dataclass

from numpy.typing import NDArray

from .base import Compute
from .time_series import compute_msd as _compute_msd


@dataclass(frozen=True)
class MSD(Compute):
    """Compute mean squared displacement at each time lag.

    MSD(dt) = <(r_i(t+dt) - r_i(t))^2>_{i, t}

    Args:
        max_lag: Maximum time lag in frames.

    Examples:
        Self-diffusion::

            cation_coords = unwrapped[:, cation_mask, :]  # (n_frames, n_cations, 3)
            msd = MSD(max_lag=3000)
            msd_values = msd(cation_coords)               # -> NDArray (max_lag,)

        Polarization MSD (no dedicated class needed)::

            polarization = (
                coords[:, cat_mask, :].sum(axis=1)
                - coords[:, an_mask, :].sum(axis=1)
            )  # (n_frames, 3)
            pmsd_values = msd(polarization[:, None, :])    # -> NDArray (max_lag,)
    """

    max_lag: int

    def run(self, positions: NDArray) -> NDArray:
        """Compute MSD from positions.

        Args:
            positions: Coordinate array with shape ``(n_frames, n_particles, n_dim)``.
                For polarization MSD, reshape a ``(n_frames, 3)`` vector to
                ``(n_frames, 1, 3)``.

        Returns:
            MSD values at each time lag, shape ``(max_lag,)``.
        """
        if positions.ndim != 3:
            raise ValueError(
                f"Expected 3D array (n_frames, n_particles, n_dim), "
                f"got shape {positions.shape}"
            )
        return _compute_msd(positions, cache_size=self.max_lag)


def msd(positions: NDArray, *, max_lag: int) -> NDArray:
    """Compute MSD.  Shorthand for ``MSD(max_lag=max_lag)(positions)``."""
    return MSD(max_lag=max_lag).run(positions)
