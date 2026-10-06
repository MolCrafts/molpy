"""Trajectory container (native-backed) + split extensions.

The trajectory *container* sinks to the native core: :class:`molpy.Trajectory` subclasses
:class:`molrs.store.Trajectory` — an eager, materialized sequence of frames with
optional ``step`` / ``time`` arrays. molpy adds Python-side conveniences: an
associated topology, slice indexing that returns a sub-trajectory, and frame
mapping. Lazy, seekable reading from disk lives in the native core as ``TrajectoryReader``
(``molpy.io.read_lammps_trajectory`` / ``read_xyz_trajectory``), not here.

Trajectory *splitting* stays in molpy as an extension layer that operates on the
native-backed container (:class:`SplitStrategy` and :class:`TrajectorySplitter`).
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import Callable, Iterable
from typing import Any, overload

from molrs.store import Frame
from molrs.store import Trajectory as _NativeTrajectory


class Trajectory(_NativeTrajectory):
    """An eager sequence of molecular frames with an optional topology.

    Subclasses :class:`molrs.store.Trajectory`: frame storage, ``len()``, integer
    indexing, and the ``frames`` / ``step`` / ``time`` accessors all live in
    the Rust container. molpy adds an associated ``topology``, slice indexing
    (returns a sub-:class:`Trajectory`), and :meth:`map`.

    Frames must be :class:`~molpy.Frame` objects — the Rust container copies
    each into its column store on construction. For lazy, seekable reading from
    disk use ``molpy.io.read_lammps_trajectory`` / ``read_xyz_trajectory``, which
    return a lazy ``TrajectoryReader`` instead of materializing every frame.

    Args:
        frames: Sequence of :class:`~molpy.Frame` objects.
        topology: Optional connectivity/topology object carried alongside the
            frames (stored and passed through unchanged). Defaults to None.
        step: Optional per-frame integer step indices (forwarded to the native core).
        time: Optional per-frame simulation times (forwarded to the native core).

    Examples:
        >>> traj = Trajectory([frame0, frame1, frame2])
        >>> len(traj)
        3
        >>> traj[0]            # integer index -> Frame
        Frame(...)
        >>> traj[0:2]          # slice -> sub-Trajectory
        Trajectory(n_frames=2, topology=None)
    """

    def __new__(
        cls,
        frames: Iterable[Frame],
        topology: Any | None = None,
        step: Any | None = None,
        time: Any | None = None,
    ) -> "Trajectory":
        # molrs.store.Trajectory is constructed in __new__ (PyO3 #[new]); it needs a
        # materialized sequence and does not accept ``topology``.
        return super().__new__(cls, list(frames), step, time)

    def __init__(
        self,
        frames: Iterable[Frame],
        topology: Any | None = None,
        step: Any | None = None,
        time: Any | None = None,
    ) -> None:
        self._topology = topology

    @property
    def topology(self) -> Any | None:
        """The topology object associated with this trajectory (or None)."""
        return self._topology

    @overload
    def __getitem__(self, key: int) -> Frame: ...

    @overload
    def __getitem__(self, key: slice) -> "Trajectory": ...

    def __getitem__(self, key: int | slice) -> "Frame | Trajectory":
        """Get a frame (int key) or a sub-trajectory (slice key).

        Args:
            key: Integer index or slice.

        Returns:
            The :class:`Frame` for an integer key, or a new :class:`Trajectory`
            (sharing this trajectory's topology, with its ``step`` / ``time``
            labels sliced alike) for a slice.
        """
        if isinstance(key, slice):
            return type(self)(
                self.frames[key],
                self._topology,
                step=None if self.step is None else self.step[key],
                time=None if self.time is None else self.time[key],
            )
        if key < 0:
            key += len(self)
        return super().__getitem__(key)

    def map(self, func: Callable[[Frame], Frame]) -> "Trajectory":
        """Apply ``func`` to every frame, returning a new trajectory.

        Args:
            func: A callable mapping a :class:`Frame` to a :class:`Frame`.

        Returns:
            A new :class:`Trajectory` of the mapped frames, sharing this
            trajectory's topology and ``step`` / ``time`` labels. The original
            is not modified.
        """
        return type(self)(
            [func(frame) for frame in self],
            self._topology,
            step=self.step,
            time=self.time,
        )

    def __repr__(self) -> str:
        topo = "present" if self._topology is not None else "None"
        return f"Trajectory(n_frames={len(self)}, topology={topo})"


# ====================== Trajectory Splitters ====================


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

    The resulting segments are :class:`Trajectory` objects that share the same
    topology as the original.

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
            A list of :class:`Trajectory` segments, each a contiguous subset of
            the original frames sharing its topology.
        """
        indices = strategy.get_split_indices(self.trajectory)

        segments = []
        for i in range(len(indices) - 1):
            start, end = indices[i], indices[i + 1]
            segments.append(self.trajectory[start:end])

        return segments
