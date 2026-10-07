"""Trajectory splitting: molpy's strategies over the native trajectory.

The container is :class:`molrs.store.Trajectory` itself (``mp.Trajectory``):
an eager frame sequence with optional ``step`` / ``time`` labels, negative
indices, slicing (a sub-trajectory with its labels sliced alike) and ``map``.
molpy adds only how to cut one into segments: a :class:`SplitStrategy` picks
the split indices and :class:`TrajectorySplitter` slices the trajectory there.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import Callable

from molrs.store import Trajectory


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
            Indices where the trajectory should be split. The first index is 0
            and the last is the total number of frames.
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
        indices = list(range(0, length, self.interval))
        if indices[-1] != length:
            indices.append(length)
        return indices


class TimeIntervalStrategy(SplitStrategy):
    """Split a trajectory by simulation-time intervals.

    Splits based on the trajectory's native per-frame ``time`` array (the native
    container's time representation, set via ``Trajectory(frames, time=...)``).
    A trajectory without a ``time`` array is left unsplit (single segment).

    Args:
        interval: Time interval for splitting (same units as the trajectory's
            ``time`` array). Must be positive.
    """

    def __init__(self, interval: float) -> None:
        if interval <= 0:
            raise ValueError(f"Interval must be positive, got {interval}")
        self.interval = interval

    def get_split_indices(self, trajectory: Trajectory) -> list[int]:
        n = len(trajectory)
        times = trajectory.time
        if times is None:
            return [0, n]

        indices = [0]
        start_time = None
        for i, frame_time in enumerate(times):
            if start_time is None:
                start_time = frame_time
            if frame_time >= start_time + len(indices) * self.interval:
                indices.append(i)

        if indices[-1] != n:
            indices.append(n)

        return indices


class CustomStrategy(SplitStrategy):
    """Split a trajectory using a user-provided function.

    Args:
        split_func: A callable taking a :class:`Trajectory` and returning a
            list of split indices. The first index should be 0 and the last the
            total frame count.
    """

    def __init__(self, split_func: Callable[[Trajectory], list[int]]) -> None:
        self.split_func = split_func

    def get_split_indices(self, trajectory: Trajectory) -> list[int]:
        return self.split_func(trajectory)


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
            A list of :class:`Trajectory` segments, each a contiguous slice of
            the original frames.
        """
        indices = strategy.get_split_indices(self.trajectory)

        segments = []
        for i in range(len(indices) - 1):
            start, end = indices[i], indices[i + 1]
            segments.append(self.trajectory[start:end])

        return segments
