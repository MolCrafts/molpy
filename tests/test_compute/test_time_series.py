"""Tests for time-series analysis operations."""

import numpy as np
import pytest

from molpy.compute.time_series import (
    TimeAverage,
    TimeCache,
    compute_acf,
    compute_msd,
)


class TestTimeCache:
    """Tests for TimeCache class."""

    def test_initialization(self):
        """Test TimeCache initialization."""
        cache = TimeCache(cache_size=10, shape=(5, 3))
        assert cache.cache_size == 10
        assert cache.shape == (5, 3)
        assert cache.cache.shape == (10, 5, 3)
        assert np.all(np.isnan(cache.cache))

    def test_update(self):
        """Test updating cache with new data."""
        cache = TimeCache(cache_size=3, shape=(2,))

        # Add first frame
        cache.update(np.array([1.0, 2.0]))
        assert np.allclose(cache.cache[0], [1.0, 2.0])
        assert np.all(np.isnan(cache.cache[1:]))

        # Add second frame
        cache.update(np.array([3.0, 4.0]))
        assert np.allclose(cache.cache[0], [3.0, 4.0])
        assert np.allclose(cache.cache[1], [1.0, 2.0])
        assert np.all(np.isnan(cache.cache[2:]))

        # Add third frame
        cache.update(np.array([5.0, 6.0]))
        assert np.allclose(cache.cache[0], [5.0, 6.0])
        assert np.allclose(cache.cache[1], [3.0, 4.0])
        assert np.allclose(cache.cache[2], [1.0, 2.0])

        # Add fourth frame (should drop oldest)
        cache.update(np.array([7.0, 8.0]))
        assert np.allclose(cache.cache[0], [7.0, 8.0])
        assert np.allclose(cache.cache[1], [5.0, 6.0])
        assert np.allclose(cache.cache[2], [3.0, 4.0])

    def test_wrong_shape(self):
        """Test error on wrong data shape."""
        cache = TimeCache(cache_size=5, shape=(3,))
        with pytest.raises(ValueError, match="shape"):
            cache.update(np.array([1.0, 2.0]))  # Wrong shape

    def test_reset(self):
        """Test resetting cache."""
        cache = TimeCache(cache_size=3, shape=(2,))
        cache.update(np.array([1.0, 2.0]))
        cache.update(np.array([3.0, 4.0]))

        cache.reset()
        assert np.all(np.isnan(cache.cache))
        assert cache._count == 0


class TestTimeAverage:
    """Tests for TimeAverage class."""

    def test_basic_averaging(self):
        """Test basic time averaging."""
        avg = TimeAverage(shape=(2,), dropnan="none")

        avg.update(np.array([1.0, 2.0]))
        avg.update(np.array([3.0, 4.0]))
        avg.update(np.array([5.0, 6.0]))

        result = avg.get()
        assert np.allclose(result, [3.0, 4.0])  # (1+3+5)/3, (2+4+6)/3

    def test_dropnan_partial(self):
        """Test partial NaN dropping."""
        avg = TimeAverage(shape=(3,), dropnan="partial")

        avg.update(np.array([1.0, 2.0, np.nan]))
        avg.update(np.array([3.0, np.nan, 3.0]))
        avg.update(np.array([5.0, 6.0, 5.0]))

        result = avg.get()
        assert np.allclose(result, [3.0, 4.0, 4.0])  # (1+3+5)/3, (2+6)/2, (3+5)/2

    def test_dropnan_all(self):
        """Test dropping entire frame with any NaN."""
        avg = TimeAverage(shape=(2,), dropnan="all")

        avg.update(np.array([1.0, 2.0]))
        avg.update(np.array([3.0, np.nan]))  # Should be skipped
        avg.update(np.array([5.0, 6.0]))

        result = avg.get()
        assert np.allclose(result, [3.0, 4.0])  # (1+5)/2, (2+6)/2

    def test_reset(self):
        """Test resetting accumulator."""
        avg = TimeAverage(shape=(2,))
        avg.update(np.array([1.0, 2.0]))
        avg.update(np.array([3.0, 4.0]))

        avg.reset()
        result = avg.get()
        assert np.all(np.isnan(result))


class TestComputeMSD:
    """Tests for compute_msd function."""

    def test_simple_diffusion(self):
        """Test MSD for simple 1D diffusion."""
        # Create a simple 1D random walk
        np.random.seed(42)
        n_frames = 1000
        n_particles = 100

        # Random walk: position = cumsum of random steps
        steps = np.random.randn(n_frames, n_particles, 1)
        positions = np.cumsum(steps, axis=0)

        # Compute MSD
        cache_size = 100
        msd = compute_msd(positions, cache_size=cache_size)

        # For 1D random walk with step variance σ²=1:
        # MSD(t) ≈ 2*D*t where D = σ²/(2*dt) = 0.5 (with dt=1)
        # So MSD(t) ≈ t
        time_lags = np.arange(cache_size)

        # Check that MSD grows approximately linearly
        # (with some tolerance due to finite sampling)
        assert msd.shape == (cache_size,)
        assert msd[0] < msd[50] < msd[99]  # Monotonic growth

        # Check approximate linear growth in middle range
        slope = (msd[50] - msd[10]) / (50 - 10)
        assert 0.5 < slope < 1.5  # Rough check

    def test_zero_diffusion(self):
        """Test MSD for stationary particles."""
        n_frames = 100
        n_particles = 10

        # All particles stay at same position
        positions = np.zeros((n_frames, n_particles, 3))

        msd = compute_msd(positions, cache_size=50)

        # MSD should be zero for stationary particles
        assert np.allclose(msd, 0.0, atol=1e-10)


class TestComputeACF:
    """Tests for compute_acf function."""

    def test_constant_velocity(self):
        """Test ACF for constant velocity."""
        n_frames = 100
        n_particles = 10

        # Constant velocity in x direction
        velocities = np.zeros((n_frames, n_particles, 3))
        velocities[:, :, 0] = 1.0  # vx = 1

        acf = compute_acf(velocities, cache_size=50)

        # ACF should be constant = 1.0 for constant velocity
        assert np.allclose(acf, 1.0, atol=1e-10)

    def test_random_velocities(self):
        """Test ACF for uncorrelated random velocities."""
        np.random.seed(42)
        n_frames = 1000
        n_particles = 100

        # Random velocities (uncorrelated in time)
        velocities = np.random.randn(n_frames, n_particles, 3)

        acf = compute_acf(velocities, cache_size=50)

        # ACF[0] should be variance
        # ACF[t>0] should decay to zero for uncorrelated data
        assert acf[0] > 0  # Non-zero at t=0
        assert np.abs(acf[10:].mean()) < 0.1  # Small for t>0 (with tolerance)
