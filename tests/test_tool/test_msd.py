"""Tests for MSD tool."""

import numpy as np
import pytest

from molpy.tool.msd import MSD


class TestMSD:
    """Tests for the MSD dataclass."""

    def test_init(self):
        msd = MSD(max_lag=100)
        assert msd.max_lag == 100

    def test_random_walk_linear_growth(self):
        """MSD of a 1D random walk should grow approximately linearly."""
        np.random.seed(42)
        n_frames, n_particles = 1000, 100
        steps = np.random.randn(n_frames, n_particles, 1)
        positions = np.cumsum(steps, axis=0)

        msd = MSD(max_lag=100)
        result = msd(positions)

        assert result.shape == (100,)
        # Monotonic growth
        assert result[0] < result[50] < result[99]
        # Roughly linear slope
        slope = (result[50] - result[10]) / (50 - 10)
        assert 0.5 < slope < 1.5

    def test_stationary_particles(self):
        """MSD should be zero for stationary particles."""
        positions = np.zeros((100, 10, 3))

        msd = MSD(max_lag=50)
        result = msd(positions)

        assert np.allclose(result, 0.0, atol=1e-10)

    def test_3d_random_walk(self):
        """MSD of a 3D random walk grows linearly, slope ~ 3."""
        np.random.seed(0)
        n_frames, n_particles = 2000, 50
        steps = np.random.randn(n_frames, n_particles, 3)
        positions = np.cumsum(steps, axis=0)

        msd = MSD(max_lag=200)
        result = msd(positions)

        # In 3D, MSD(t) = 6D*t with D = 0.5 => slope ~ 3
        slope = (result[100] - result[20]) / (100 - 20)
        assert 2.0 < slope < 4.0

    def test_polarization_msd_pattern(self):
        """Demonstrate PMSD usage: MSD on summed coordinates."""
        np.random.seed(42)
        n_frames = 500
        n_cations = 5
        n_anions = 5

        cat_pos = np.cumsum(np.random.randn(n_frames, n_cations, 3) * 0.1, axis=0)
        an_pos = np.cumsum(np.random.randn(n_frames, n_anions, 3) * 0.1, axis=0)

        # Polarization = sum(cation_coords) - sum(anion_coords)
        polarization = cat_pos.sum(axis=1) - an_pos.sum(axis=1)  # (n_frames, 3)

        msd = MSD(max_lag=50)
        result = msd(polarization[:, None, :])  # (n_frames, 1, 3)

        assert result.shape == (50,)
        # Should grow (it's a random walk in summed coordinates)
        assert result[-1] > result[0]

    def test_rejects_wrong_ndim(self):
        """Must pass a 3D array."""
        msd = MSD(max_lag=10)
        with pytest.raises(ValueError, match="Expected 3D array"):
            msd(np.zeros((10, 3)))

    def test_frozen_dataclass(self):
        """MSD instances are immutable."""
        msd = MSD(max_lag=100)
        with pytest.raises(AttributeError):
            msd.max_lag = 200  # type: ignore[misc]
