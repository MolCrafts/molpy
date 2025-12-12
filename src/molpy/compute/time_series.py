"""Time-series analysis operations for trajectory data.

This module provides utilities for computing time-correlation functions,
mean squared displacements, and other time-series statistics commonly used
in molecular dynamics trajectory analysis.

Adapted from the tame library (https://github.com/Roy-Kid/tame).
"""

from __future__ import annotations

from typing import Literal

import numpy as np
from numpy.typing import NDArray


class TimeCache:
    """Cache previous N frames of trajectory data for correlation calculations.
    
    This class maintains a rolling buffer of the most recent N frames of data,
    which is essential for computing time-correlation functions like MSD and ACF.
    
    Args:
        cache_size: Number of frames to cache (maximum time lag)
        shape: Shape of data arrays to cache (e.g., (n_atoms, 3) for coordinates)
        dtype: Data type for cached arrays
        default_val: Default value to fill cache initially (default: NaN)
    
    Examples:
        >>> cache = TimeCache(cache_size=100, shape=(10, 3))
        >>> coords = np.random.randn(10, 3)
        >>> cache.update(coords)
        >>> cached_data = cache.get()  # Shape: (100, 10, 3)
    """
    
    def __init__(
        self,
        cache_size: int,
        shape: tuple[int, ...],
        dtype: np.dtype = np.float64,
        default_val: float = np.nan,
    ):
        self.cache_size = cache_size
        self.shape = shape
        self.dtype = dtype
        # Initialize cache with default values
        self.cache = np.full((cache_size, *shape), default_val, dtype=dtype)
        self._count = 0
    
    def update(self, new_data: NDArray) -> None:
        """Add new frame to cache, shifting older frames.
        
        Args:
            new_data: New data array to add (shape must match self.shape)
        """
        if new_data.shape != self.shape:
            raise ValueError(
                f"Data shape {new_data.shape} doesn't match cache shape {self.shape}"
            )
        
        # Shift cache and add new data at front
        new_val = np.expand_dims(new_data, 0)
        self.cache = np.concatenate([new_val, self.cache], axis=0)[:-1]
        self._count += 1
    
    def get(self) -> NDArray:
        """Get cached data array.
        
        Returns:
            Cached data with shape (cache_size, *data_shape)
        """
        return self.cache
    
    def reset(self) -> None:
        """Reset cache to initial state."""
        self.cache.fill(np.nan)
        self._count = 0


class TimeAverage:
    """Compute running time average with NaN handling.
    
    This class accumulates data over time and computes the average,
    with options for handling NaN values.
    
    Args:
        shape: Shape of data arrays to average
        dtype: Data type for accumulated arrays
        dropnan: How to handle NaN values:
            - 'none': Include NaN values in average (result may be NaN)
            - 'partial': Ignore individual NaN entries
            - 'all': Skip entire frame if any NaN is present
    
    Examples:
        >>> avg = TimeAverage(shape=(10,), dropnan='partial')
        >>> avg.update(np.array([1.0, 2.0, np.nan, 4.0]))
        >>> avg.update(np.array([2.0, 3.0, 3.0, 5.0]))
        >>> result = avg.get()  # [1.5, 2.5, 3.0, 4.5]
    """
    
    def __init__(
        self,
        shape: tuple[int, ...],
        dtype: np.dtype = np.float64,
        dropnan: Literal["none", "partial", "all"] = "partial",
    ):
        self.shape = shape
        self.dtype = dtype
        self.dropnan = dropnan
        self.cumsum = np.zeros(shape, dtype=dtype)
        self.count = np.zeros(shape, dtype=dtype) if dropnan == "partial" else 0
        self._n_frames = 0
    
    def update(self, new_data: NDArray) -> None:
        """Add new data to running average.
        
        Args:
            new_data: New data array to include in average
        """
        if new_data.shape != self.shape:
            raise ValueError(
                f"Data shape {new_data.shape} doesn't match expected shape {self.shape}"
            )
        
        nan_mask = np.isnan(new_data)
        
        if self.dropnan == "all":
            # Skip entire frame if any NaN present
            if nan_mask.any():
                return
            self.cumsum += new_data
            self.count += 1
        elif self.dropnan == "partial":
            # Ignore individual NaN entries
            self.cumsum += np.nan_to_num(new_data, nan=0.0)
            self.count += (~nan_mask).astype(self.dtype)
        else:  # dropnan == "none"
            # Include NaN values (may propagate NaN)
            self.cumsum += new_data
            self.count += 1
        
        self._n_frames += 1
    
    def get(self) -> NDArray:
        """Get current time-averaged value.
        
        Returns:
            Time-averaged data array
        """
        if isinstance(self.count, np.ndarray):
            # Avoid division by zero
            with np.errstate(divide='ignore', invalid='ignore'):
                result = self.cumsum / self.count
                result[self.count == 0] = np.nan
            return result
        else:
            if self.count == 0:
                return np.full(self.shape, np.nan, dtype=self.dtype)
            return self.cumsum / self.count
    
    def reset(self) -> None:
        """Reset accumulator to initial state."""
        self.cumsum.fill(0)
        if isinstance(self.count, np.ndarray):
            self.count.fill(0)
        else:
            self.count = 0
        self._n_frames = 0


def compute_msd(
    data: NDArray,
    cache_size: int,
    dropnan: Literal["none", "partial", "all"] = "partial",
) -> NDArray:
    """Compute mean squared displacement over trajectory.
    
    Calculates: <(r_i(t+dt) - r_i(t))^2>_{i,t}
    
    The particle dimension is averaged, and the time dimension is accumulated
    using a rolling cache to compute correlations at different time lags.
    
    Args:
        data: Trajectory data with shape (n_frames, n_particles, n_dim)
        cache_size: Maximum time lag (dt) to compute, in frames
        dropnan: How to handle NaN values in averaging
    
    Returns:
        MSD array with shape (cache_size,) containing MSD at each time lag
    
    Examples:
        >>> # Simple 1D random walk
        >>> n_frames, n_particles = 1000, 100
        >>> positions = np.cumsum(np.random.randn(n_frames, n_particles, 1), axis=0)
        >>> msd = compute_msd(positions, cache_size=100)
        >>> # MSD should grow linearly with time for random walk
    """
    n_frames, n_particles, n_dim = data.shape
    
    # Initialize cache and accumulator
    cache = TimeCache(cache_size, shape=(n_particles, n_dim))
    avg = TimeAverage(shape=(cache_size,), dropnan=dropnan)
    
    # Iterate through trajectory
    for frame_idx in range(n_frames):
        current_data = data[frame_idx]
        cache.update(current_data)
        
        # Compute displacement from current frame to all cached frames
        cached_data = cache.get()  # (cache_size, n_particles, n_dim)
        displacement = cached_data - current_data[np.newaxis, :, :]
        
        # Compute squared displacement and average over particles
        sq_displacement = np.sum(displacement**2, axis=2)  # (cache_size, n_particles)
        msd_frame = np.mean(sq_displacement, axis=1)  # (cache_size,)
        
        # Accumulate time average
        avg.update(msd_frame)
    
    return avg.get()


def compute_acf(
    data: NDArray,
    cache_size: int,
    dropnan: Literal["none", "partial", "all"] = "partial",
) -> NDArray:
    """Compute autocorrelation function over trajectory.
    
    Calculates: <v_i(0) · v_i(dt)>_{i,t}
    
    The particle dimension is averaged, and the time dimension is accumulated
    using a rolling cache to compute correlations at different time lags.
    
    Args:
        data: Trajectory data with shape (n_frames, n_particles, n_dim)
        cache_size: Maximum time lag (dt) to compute, in frames
        dropnan: How to handle NaN values in averaging
    
    Returns:
        ACF array with shape (cache_size,) containing ACF at each time lag
    
    Examples:
        >>> # Velocity autocorrelation
        >>> n_frames, n_particles = 1000, 100
        >>> velocities = np.random.randn(n_frames, n_particles, 3)
        >>> acf = compute_acf(velocities, cache_size=100)
    """
    n_frames, n_particles, n_dim = data.shape
    
    # Initialize cache and accumulator
    cache = TimeCache(cache_size, shape=(n_particles, n_dim))
    avg = TimeAverage(shape=(cache_size,), dropnan=dropnan)
    
    # Iterate through trajectory
    for frame_idx in range(n_frames):
        current_data = data[frame_idx]
        cache.update(current_data)
        
        # Compute dot product with all cached frames
        cached_data = cache.get()  # (cache_size, n_particles, n_dim)
        dot_product = cached_data * current_data[np.newaxis, :, :]
        
        # Sum over dimensions and average over particles
        acf_frame = np.mean(np.sum(dot_product, axis=2), axis=1)  # (cache_size,)
        
        # Accumulate time average
        avg.update(acf_frame)
    
    return avg.get()
