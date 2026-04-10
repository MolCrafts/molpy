"""Tests for DisplacementCorrelation tool."""

import numpy as np
import pytest

from molpy.tool.cross_correlation import DisplacementCorrelation


class TestDisplacementCorrelation:
    """Tests for the DisplacementCorrelation dataclass."""

    def test_init_defaults(self):
        xdc = DisplacementCorrelation(max_lag=100)
        assert xdc.max_lag == 100
        assert xdc.exclude_self is False

    def test_cross_species(self):
        """Cross-correlation between two independent random walks."""
        np.random.seed(42)
        n_frames, n_a, n_b = 500, 5, 5
        pos_a = np.cumsum(np.random.randn(n_frames, n_a, 3) * 0.1, axis=0)
        pos_b = np.cumsum(np.random.randn(n_frames, n_b, 3) * 0.1, axis=0)

        xdc = DisplacementCorrelation(max_lag=50)
        result = xdc(pos_a, pos_b)

        assert result.shape == (50,)
        # Independent walkers: cross-correlation should be near zero on average
        # (sum of products of independent random increments)
        assert np.abs(result[10:].mean()) < np.abs(result).max()

    def test_same_species_distinct(self):
        """Same-species with exclude_self removes self-contribution."""
        np.random.seed(0)
        n_frames, n_particles = 500, 10
        positions = np.cumsum(np.random.randn(n_frames, n_particles, 3) * 0.1, axis=0)

        xdc = DisplacementCorrelation(max_lag=50, exclude_self=True)
        result = xdc(positions, positions)

        assert result.shape == (50,)

    def test_exclude_self_requires_equal_counts(self):
        """exclude_self=True needs n_a == n_b."""
        pos_a = np.zeros((10, 3, 3))
        pos_b = np.zeros((10, 5, 3))

        xdc = DisplacementCorrelation(max_lag=5, exclude_self=True)
        with pytest.raises(ValueError, match="equal particle counts"):
            xdc(pos_a, pos_b)

    def test_rejects_wrong_ndim(self):
        xdc = DisplacementCorrelation(max_lag=5)
        with pytest.raises(ValueError, match="Expected 3D"):
            xdc(np.zeros((10, 3)), np.zeros((10, 3)))

    def test_rejects_frame_count_mismatch(self):
        xdc = DisplacementCorrelation(max_lag=5)
        with pytest.raises(ValueError, match="Frame count mismatch"):
            xdc(np.zeros((10, 3, 3)), np.zeros((20, 3, 3)))

    def test_rejects_dim_mismatch(self):
        xdc = DisplacementCorrelation(max_lag=5)
        with pytest.raises(ValueError, match="Dimension mismatch"):
            xdc(np.zeros((10, 3, 3)), np.zeros((10, 3, 2)))

    def test_frozen_dataclass(self):
        xdc = DisplacementCorrelation(max_lag=100)
        with pytest.raises(AttributeError):
            xdc.max_lag = 200  # type: ignore[misc]
