from abc import ABC, abstractmethod
from collections.abc import Callable, Generator, Iterable, Iterator, Sequence
from typing import overload

from .frame import Frame


class Trajectory:
    """A sequence of molecular frames with optional topology.

    Supports iteration, indexing, slicing, and mapping operations.
    Can be used as an iterator with manual next() calls.

    Args:
        frames: An iterable or sequence of Frame objects. If a Sequence is provided,
            the trajectory supports length and indexing. If an Iterable (e.g., generator)
            is provided, only iteration is supported.
        topology: Optional topology information for the trajectory.
    """

    def __init__(self, frames: Iterable[Frame], topology=None):
        """Initialize trajectory with frames."""
        self._frames = frames
        self._topology = topology
        self._iterator: Iterator[Frame] | None = None

    def __iter__(self) -> Iterator[Frame]:
        """Return an iterator over frames.

        Each call creates a new iterator, allowing multiple independent iterations.
        """
        return iter(self._frames)

    def __next__(self) -> Frame:
        """Get the next frame in the trajectory.

        Supports manual iteration using next(trajectory).
        Creates a new iterator on first call or reuses existing one.

        Raises:
            StopIteration: When all frames have been consumed.
        """
        if self._iterator is None:
            self._iterator = iter(self._frames)
        return next(self._iterator)

    def __len__(self) -> int:
        """Return the number of frames in the trajectory."""
        if isinstance(self._frames, Sequence):
            return len(self._frames)
        else:
            raise TypeError(
                "Length not available for generator-based trajectories. "
                "Use count_frames() to exhaust and count if needed."
            )

    def has_length(self) -> bool:
        """Check if this trajectory has a known length without computing it."""
        return isinstance(self._frames, Sequence)

    @overload
    def __getitem__(self, key: int) -> Frame: ...

    @overload
    def __getitem__(self, key: slice) -> "Trajectory": ...

    def __getitem__(self, key: int | slice) -> "Frame | Trajectory":
        """Get a frame or slice of frames."""
        if isinstance(key, int):
            if isinstance(self._frames, Sequence):
                return self._frames[key]
            else:
                raise TypeError(
                    "Indexing not supported for generator-based trajectories. "
                    "Convert to list first or use iteration."
                )
        elif isinstance(key, slice):
            if isinstance(self._frames, Sequence):
                sliced_frames = self._frames[key]
                return Trajectory(sliced_frames, self._topology)
            else:
                # For generators, we need to materialize the slice
                frames_list = list(self._frames)
                sliced_frames = frames_list[key]
                return Trajectory(sliced_frames, self._topology)
        else:
            raise TypeError(f"Invalid key type: {type(key)}")

    def map(self, func: Callable[[Frame], Frame]) -> "Trajectory":
        """Apply a function to each frame, returning a new trajectory.

        Args:
            func: A function that takes a Frame and returns a Frame.
                The function will be applied lazily as frames are accessed.

        Returns:
            A new Trajectory with mapped frames.

        Example:
            >>> def center_frame(frame):
            ...     # Center coordinates at origin
            ...     atoms = frame["atoms"]
            ...     xyz = atoms[["x", "y", "z"]]
            ...     center = xyz.mean(axis=0)
            ...     atoms["x"] = atoms["x"] - center[0]
            ...     atoms["y"] = atoms["y"] - center[1]
            ...     atoms["z"] = atoms["z"] - center[2]
            ...     return frame
            >>> centered_traj = traj.map(center_frame)
        """

        def mapped_generator() -> Generator[Frame, None, None]:
            for frame in self._frames:
                yield func(frame)

        return Trajectory(mapped_generator(), self._topology)


# ====================== Trajectory Splitters ====================


class SplitStrategy(ABC):
    """Abstract splitting strategy."""

    @abstractmethod
    def get_split_indices(self, trajectory: Trajectory) -> list[int]:
        """Get split point indices."""
        raise NotImplementedError


class FrameIntervalStrategy(SplitStrategy):
    """Split every N frames."""

    def __init__(self, interval: int):
        self.interval = interval

    def get_split_indices(self, trajectory: Trajectory) -> list[int]:
        if not trajectory.has_length():
            raise TypeError(
                "Frame interval splitting requires trajectory with known length"
            )

        length = len(trajectory)
        indices = list(range(0, length, self.interval))
        if indices[-1] != length:
            indices.append(length)
        return indices


class TimeIntervalStrategy(SplitStrategy):
    """Split by time intervals."""

    def __init__(self, interval: float):
        self.interval = interval

    def get_split_indices(self, trajectory: Trajectory) -> list[int]:
        indices = [0]
        start_time = None
        frame_count = 0

        for i, frame in enumerate(trajectory):
            frame_count = i + 1  # Keep track of total frames seen
            # Check if frame has time information in metadata
            frame_time = frame.metadata.get("time", None)
            if frame_time is not None:
                if start_time is None:
                    start_time = frame_time

                # Check if we've reached the next interval
                if frame_time >= start_time + len(indices) * self.interval:
                    indices.append(i)

        # Add final index using the count from iteration
        final_index = frame_count
        if indices[-1] != final_index:
            indices.append(final_index)

        return indices


class CustomStrategy(SplitStrategy):
    """Split using custom function."""

    def __init__(self, split_func: Callable[[Trajectory], list[int]]):
        self.split_func = split_func

    def get_split_indices(self, trajectory: Trajectory) -> list[int]:
        return self.split_func(trajectory)


class TrajectorySplitter:
    """Splits trajectories into lazy segments."""

    def __init__(self, trajectory: Trajectory):
        self.trajectory = trajectory

    def split(self, strategy: SplitStrategy) -> list[Trajectory]:
        """Split trajectory using strategy, returning lazy segments."""
        indices = strategy.get_split_indices(self.trajectory)

        segments = []
        for i in range(len(indices) - 1):
            start, end = indices[i], indices[i + 1]
            # Use trajectory slicing instead of accessing private members
            segment = self.trajectory[start:end]
            segments.append(segment)

        return segments

    def split_frames(self, interval: int) -> list[Trajectory]:
        """Convenience method for frame-based splitting."""
        return self.split(FrameIntervalStrategy(interval))

    def split_time(self, interval: float) -> list[Trajectory]:
        """Convenience method for time-based splitting."""
        return self.split(TimeIntervalStrategy(interval))
