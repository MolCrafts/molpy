"""Cross-displacement correlation computation.

Operates on plain NDArrays — no trajectory coupling.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray

from .base import Compute
from .time_series import TimeAverage, TimeCache


@dataclass(frozen=True)
class DisplacementCorrelation(Compute):
    """Compute cross-displacement correlation between two groups.

    For two groups A and B the correlation at time lag dt is:

        C(dt) = <sum_i dr_i^A(dt) . sum_j dr_j^B(dt)> / N_A

    where dr_i(dt) = r_i(t+dt) - r_i(t).

    When ``exclude_self=True`` and both inputs are the same species,
    the self-terms are subtracted so only *distinct* correlations remain:

        C_distinct(dt) = <dr_i . (sum_j dr_j - dr_i)>_{i, t}

    Args:
        max_lag: Maximum time lag in frames.
        exclude_self: If True, subtract self-correlation (for same-species
            distinct diffusion).

    Examples:
        Cross-species (cation-anion)::

            xdc = DisplacementCorrelation(max_lag=3000)
            corr = xdc(cation_coords, anion_coords)  # -> NDArray (max_lag,)

        Same-species distinct (exclude self-correlation)::

            xdc = DisplacementCorrelation(max_lag=3000, exclude_self=True)
            corr = xdc(cation_coords, cation_coords)  # -> NDArray (max_lag,)
    """

    max_lag: int
    exclude_self: bool = False

    def run(self, positions_a: NDArray, positions_b: NDArray) -> NDArray:
        """Compute displacement correlation.

        Args:
            positions_a: Coordinates of group A, shape ``(n_frames, n_a, n_dim)``.
            positions_b: Coordinates of group B, shape ``(n_frames, n_b, n_dim)``.

        Returns:
            Correlation values at each time lag, shape ``(max_lag,)``.
        """
        if positions_a.ndim != 3 or positions_b.ndim != 3:
            raise ValueError(
                "Expected 3D arrays (n_frames, n_particles, n_dim), "
                f"got shapes {positions_a.shape} and {positions_b.shape}"
            )
        if positions_a.shape[0] != positions_b.shape[0]:
            raise ValueError(
                f"Frame count mismatch: {positions_a.shape[0]} vs {positions_b.shape[0]}"
            )
        if positions_a.shape[2] != positions_b.shape[2]:
            raise ValueError(
                f"Dimension mismatch: {positions_a.shape[2]} vs {positions_b.shape[2]}"
            )

        n_frames, n_a, n_dim = positions_a.shape
        n_b = positions_b.shape[1]

        # Build displacement caches for both groups
        dr_a = self._compute_displacements(positions_a)  # (max_lag, n_a, n_dim)
        dr_b = self._compute_displacements(positions_b)  # (max_lag, n_b, n_dim)

        if self.exclude_self:
            if n_a != n_b:
                raise ValueError(
                    "exclude_self=True requires equal particle counts, "
                    f"got {n_a} and {n_b}"
                )
            # Same species: <dr_i . (sum_j dr_j - dr_i)>
            dr_b_sum = np.sum(dr_b, axis=1, keepdims=True)  # (max_lag, 1, n_dim)
            dr_b_others = dr_b_sum - dr_a  # (max_lag, n_a, n_dim)
            corr = np.mean(np.sum(dr_a * dr_b_others, axis=2), axis=1)  # (max_lag,)
        else:
            # Cross-species: <sum_i dr_i^A . sum_j dr_j^B>
            dr_a_sum = np.sum(dr_a, axis=1)  # (max_lag, n_dim)
            dr_b_sum = np.sum(dr_b, axis=1)  # (max_lag, n_dim)
            corr = np.sum(dr_a_sum * dr_b_sum, axis=1)  # (max_lag,)

        return corr

    def _compute_displacements(self, positions: NDArray) -> NDArray:
        """Compute time-averaged displacements for a group of particles.

        Args:
            positions: Coordinates, shape ``(n_frames, n_particles, n_dim)``.

        Returns:
            Average displacement at each lag, shape ``(max_lag, n_particles, n_dim)``.
        """
        n_frames, n_particles, n_dim = positions.shape

        cache = TimeCache(self.max_lag, shape=(n_particles, n_dim))
        avg = TimeAverage(shape=(self.max_lag, n_particles, n_dim))

        for frame_idx in range(n_frames):
            current = positions[frame_idx]
            cache.update(current)
            cached = cache.get()  # (max_lag, n_particles, n_dim)
            dr = cached - current[None, :, :]
            avg.update(dr)

        return avg.get()


def displacement_correlation(
    positions_a: NDArray,
    positions_b: NDArray,
    *,
    max_lag: int,
    exclude_self: bool = False,
) -> NDArray:
    """Compute displacement correlation.

    Shorthand for
    ``DisplacementCorrelation(max_lag=max_lag, exclude_self=exclude_self)(positions_a, positions_b)``.
    """
    return DisplacementCorrelation(max_lag=max_lag, exclude_self=exclude_self).run(
        positions_a, positions_b
    )
