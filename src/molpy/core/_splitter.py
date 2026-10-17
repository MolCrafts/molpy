"""Trajectory splitting: molpy's strategies over the native trajectory.

The container is :class:`molrs.core.Trajectory` itself (``mp.Trajectory``):
an eager frame sequence with optional ``step`` / ``time`` labels, negative
indices, slicing (a sub-trajectory with its labels sliced alike) and ``map``.
molpy adds only how to cut one into segments: a :class:`SplitStrategy` picks
the split indices and :class:`TrajectorySplitter` slices the trajectory there.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import Callable

import numpy as np
from molrs.core import Trajectory

#: The frame-meta key a frame's simulation time is stored under.
_TIME = "time"


class SplitStrategy(ABC):
    """Abstract base class for trajectory splitting strategies.

    Subclasses implement different strategies for dividing a trajectory into
    segments based on various criteria (frame count, time intervals, etc.).
    """

    @abstractmethod
    def get_split_indices(self, trajectory: Trajectory) -> list[int]:
        """Get split-point indices for dividing the trajectory.

        Args:
            trajectory: The trajectory to split.

        Returns:
            Indices where the trajectory should be split, rising strictly from
            0 to the total number of frames (``[0]`` for an empty trajectory).
        """
        raise NotImplementedError


class FrameIntervalStrategy(SplitStrategy):
    """Split a trajectory at regular frame intervals.

    Splits the trajectory every ``interval`` frames, creating segments of equal
    size (except possibly the last).

    Args:
        interval: Number of frames per segment. Must be positive.
    """

    def __init__(self, interval: int) -> None:
        if interval <= 0:
            raise ValueError(f"Interval must be positive, got {interval}")
        self.interval = interval

    def get_split_indices(self, trajectory: Trajectory) -> list[int]:
        length = len(trajectory)
        return [*range(0, length, self.interval), length]


class TimeIntervalStrategy(SplitStrategy):
    """Split a trajectory into fixed-width simulation-time bins.

    Bin ``k`` holds the frames with ``t0 + k * interval <= t < t0 + (k + 1) *
    interval``, where ``t0`` is the first frame's time; a segment starts at
    each frame that enters a new bin, and empty bins produce no segment.

    The times are the trajectory's native ``time`` array
    (``Trajectory(frames, time=...)``) or, when it has none, each frame's
    ``meta["time"]``.

    Args:
        interval: Time interval for splitting (same units as the times). Must
            be positive and finite.

    Raises:
        ValueError: From :meth:`get_split_indices`, if a frame has no time or
            the times decrease or contain non-finite values.
    """

    def __init__(self, interval: float) -> None:
        if not np.isfinite(interval) or interval <= 0:
            raise ValueError(f"Interval must be positive, got {interval}")
        self.interval = interval

    def get_split_indices(self, trajectory: Trajectory) -> list[int]:
        times = _frame_times(trajectory)
        n = len(times)
        if n == 0:
            return [0]
        if not np.all(np.isfinite(times)):
            raise ValueError("TimeIntervalStrategy: frame times must be finite")
        if np.any(times[1:] < times[:-1]):
            raise ValueError("TimeIntervalStrategy: frame times must not decrease")
        with np.errstate(over="ignore", invalid="ignore"):
            bins = np.floor((times - times[0]) / self.interval)
        if not np.all(np.isfinite(bins)):
            raise ValueError("TimeIntervalStrategy: time bins exceed finite range")
        starts = np.flatnonzero(bins[1:] != bins[:-1]) + 1
        return [0, *starts.tolist(), n]


def _frame_times(trajectory: Trajectory) -> np.ndarray:
    """Per-frame times: ``trajectory.time``, else each frame's ``meta["time"]``."""
    if trajectory.time is not None:
        return np.asarray(trajectory.time, dtype=np.float64)
    times = []
    for i in range(len(trajectory)):
        frame = trajectory[i]
        if _TIME not in frame.meta:
            raise ValueError(
                f"TimeIntervalStrategy: the trajectory has no time array and "
                f"frame {i} has no meta[{_TIME!r}]; pass Trajectory(frames, "
                "time=...) or set each frame's time"
            )
        times.append(float(frame.meta[_TIME]))
    return np.asarray(times, dtype=np.float64)


class CustomStrategy(SplitStrategy):
    """Split a trajectory using a user-provided function.

    Args:
        split_func: A callable taking a :class:`Trajectory` and returning a
            list of split indices, rising strictly from 0 to the total frame
            count. :meth:`TrajectorySplitter.split` rejects any other list.
    """

    def __init__(self, split_func: Callable[[Trajectory], list[int]]) -> None:
        self.split_func = split_func

    def get_split_indices(self, trajectory: Trajectory) -> list[int]:
        return list(self.split_func(trajectory))


class TrajectorySplitter:
    """Split a trajectory into sub-trajectories using a strategy.

    The resulting segments are :class:`Trajectory` slices of the original,
    each with its ``step`` / ``time`` labels.

    Args:
        trajectory: The trajectory to split.
    """

    def __init__(self, trajectory: Trajectory) -> None:
        self.trajectory = trajectory

    def split(self, strategy: SplitStrategy) -> list[Trajectory]:
        """Split the trajectory using ``strategy``.

        Args:
            strategy: The splitting strategy to apply.

        Returns:
            A list of :class:`Trajectory` segments, each a contiguous,
            non-empty slice of the original frames. An empty trajectory has
            no segments.

        Raises:
            ValueError: If the strategy's indices are not integers rising
                strictly from 0 to the frame count.
        """
        n = len(self.trajectory)
        indices = _checked(strategy.get_split_indices(self.trajectory), n)
        return [self.trajectory[a:b] for a, b in zip(indices, indices[1:])]


def _checked(indices: list[int], n: int) -> list[int]:
    """``indices`` if they rise strictly from 0 to ``n``; else ``ValueError``."""
    if not all(isinstance(i, (int, np.integer)) for i in indices):
        raise ValueError(f"split indices must be integers, got {indices!r}")
    indices = [int(i) for i in indices]
    if not indices or indices[0] != 0 or indices[-1] != n:
        raise ValueError(
            f"split indices must start at 0 and end at the frame count {n}, "
            f"got {indices!r}"
        )
    if any(b <= a for a, b in zip(indices, indices[1:])):
        raise ValueError(f"split indices must rise strictly, got {indices!r}")
    return indices
