from abc import ABC, abstractmethod
from collections.abc import Callable, Generator, Iterable, Iterator, Sequence
from typing import TYPE_CHECKING, overload

from .frame import Frame

if TYPE_CHECKING:
    from .topology import Topology


class Trajectory:
    """A sequence of molecular frames with optional topology.

    Trajectory is a container for a sequence of Frame objects, supporting both
    eager (Sequence) and lazy (Iterable/Generator) frame storage. It provides
    iteration, indexing, slicing, and mapping operations.

    Attributes:
        _frames (Iterable[Frame]): The underlying frame sequence. Can be a
            Sequence (list, tuple) for random access, or an Iterable (generator)
            for lazy loading.
        _topology (Topology | None): Optional topology information shared across
            all frames in the trajectory.

    Examples:
        Create from a list of frames:

        >>> frames = [Frame(), Frame(), Frame()]
        >>> traj = Trajectory(frames)
        >>> len(traj)
        3
        >>> traj[0]  # Access first frame
        Frame(...)

        Create from a generator (lazy loading):

        >>> def frame_generator():
        ...     for i in range(10):
        ...         yield Frame()
        >>> traj = Trajectory(frame_generator())
        >>> # Can iterate but not index
        >>> for frame in traj:
        ...     pass

        Slice a trajectory:

        >>> traj_slice = traj[0:5]  # First 5 frames
        >>> len(traj_slice)
        5

        Map a function over frames:

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

    def __init__(
        self, frames: Iterable[Frame], topology: "Topology | None" = None
    ) -> None:
        """Initialize trajectory with frames.

        Args:
            frames (Iterable[Frame]): An iterable or sequence of Frame objects.
                If a Sequence is provided, the trajectory supports length and
                indexing. If an Iterable (e.g., generator) is provided, only
                iteration is supported.
            topology (Topology | None, optional): Optional topology information
                for the trajectory. Defaults to None.
        """
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
        The iterator is automatically reset when exhausted.

        Raises:
            StopIteration: When all frames have been consumed.

        Examples:
            >>> traj = Trajectory([Frame(), Frame(), Frame()])
            >>> next(traj)  # Get first frame
            Frame(...)
            >>> next(traj)  # Get second frame
            Frame(...)
        """
        if self._iterator is None:
            self._iterator = iter(self._frames)
        try:
            return next(self._iterator)
        except StopIteration:
            # Reset iterator when exhausted to allow re-iteration
            self._iterator = None
            raise

    def __len__(self) -> int:
        """Return the number of frames in the trajectory.

        Returns:
            int: Number of frames in the trajectory.

        Raises:
            TypeError: If the trajectory is based on a generator/iterator that
                doesn't support length calculation.

        Examples:
            >>> traj = Trajectory([Frame(), Frame(), Frame()])
            >>> len(traj)
            3
        """
        if isinstance(self._frames, Sequence):
            return len(self._frames)
        else:
            raise TypeError(
                "Length not available for generator-based trajectories. "
                "Use count_frames() to exhaust and count if needed."
            )

    def has_length(self) -> bool:
        """Check if this trajectory has a known length without computing it.

        Returns:
            bool: True if the trajectory supports len(), False otherwise.

        Examples:
            >>> traj1 = Trajectory([Frame(), Frame()])
            >>> traj1.has_length()
            True
            >>> traj2 = Trajectory((f for f in [Frame(), Frame()]))
            >>> traj2.has_length()
            False
        """
        return isinstance(self._frames, Sequence)

    @overload
    def __getitem__(self, key: int) -> Frame: ...

    @overload
    def __getitem__(self, key: slice) -> "Trajectory": ...

    def __getitem__(self, key: int | slice) -> "Frame | Trajectory":
        """Get a frame or slice of frames.

        Args:
            key (int | slice): Index or slice to access frames.

        Returns:
            Frame | Trajectory: Single Frame if key is int, or new Trajectory
                if key is slice.

        Raises:
            TypeError: If indexing is attempted on a generator-based trajectory.
            IndexError: If the index is out of range.

        Examples:
            >>> traj = Trajectory([Frame(), Frame(), Frame()])
            >>> frame = traj[0]  # Get first frame
            >>> subtraj = traj[0:2]  # Get first two frames as new trajectory
            >>> len(subtraj)
            2
        """
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
                # Warning: This consumes the entire generator!
                frames_list = list(self._frames)
                sliced_frames = frames_list[key]
                return Trajectory(sliced_frames, self._topology)
        else:
            raise TypeError(f"Invalid key type: {type(key)}")

    def map(self, func: Callable[[Frame], Frame]) -> "Trajectory":
        """Apply a function to each frame, returning a new trajectory.

        The mapping is applied lazily - the function is only called when
        frames are accessed, making it memory-efficient for large trajectories.

        Args:
            func (Callable[[Frame], Frame]): A function that takes a Frame and
                returns a Frame. The function will be applied lazily as frames
                are accessed.

        Returns:
            Trajectory: A new Trajectory with mapped frames. The original
                trajectory is not modified.

        Examples:
            >>> traj = Trajectory([Frame(), Frame()])
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
            >>> len(centered_traj)
            2
        """

        def mapped_generator() -> Generator[Frame, None, None]:
            for frame in self._frames:
                yield func(frame)

        return Trajectory(mapped_generator(), self._topology)

    def __repr__(self) -> str:
        """Return string representation of the trajectory.

        Returns:
            str: String representation showing trajectory type and frame count
                (if available).
        """
        if self.has_length():
            return f"Trajectory(n_frames={len(self)}, topology={'present' if self._topology is not None else 'None'})"
        else:
            return f"Trajectory(n_frames=unknown, topology={'present' if self._topology is not None else 'None'})"


# ====================== Trajectory Splitters ====================


class SplitStrategy(ABC):
    """Abstract base class for trajectory splitting strategies.

    Subclasses implement different strategies for dividing a trajectory
    into segments based on various criteria (frame count, time intervals, etc.).
    """

    @abstractmethod
    def get_split_indices(self, trajectory: Trajectory) -> list[int]:
        """Get split point indices for dividing the trajectory.

        Args:
            trajectory (Trajectory): The trajectory to split.

        Returns:
            list[int]: List of indices where the trajectory should be split.
                The first index should be 0, and the last should be the total
                number of frames.

        Raises:
            NotImplementedError: Must be implemented by subclasses.
        """
        raise NotImplementedError


class FrameIntervalStrategy(SplitStrategy):
    """Split trajectory at regular frame intervals.

    Splits the trajectory every N frames, creating segments of equal
    size (except possibly the last segment).

    Args:
        interval (int): Number of frames per segment. Must be positive.

    Examples:
        >>> traj = Trajectory([Frame() for _ in range(10)])
        >>> strategy = FrameIntervalStrategy(interval=3)
        >>> splitter = TrajectorySplitter(traj)
        >>> segments = splitter.split(strategy)
        >>> len(segments)  # Creates segments of 3, 3, 3, and 1 frames
        4
    """

    def __init__(self, interval: int) -> None:
        if interval <= 0:
            raise ValueError(f"Interval must be positive, got {interval}")
        self.interval = interval

    def get_split_indices(self, trajectory: Trajectory) -> list[int]:
        """Get split indices at regular frame intervals.

        Args:
            trajectory (Trajectory): The trajectory to split.

        Returns:
            list[int]: List of split point indices.

        Raises:
            TypeError: If the trajectory doesn't have a known length.
        """
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
    """Split trajectory by time intervals.

    Splits the trajectory based on time information stored in frame metadata.
    Frames must have a "time" key in their metadata dictionary.

    Args:
        interval (float): Time interval for splitting (in the same units as
            frame time metadata). Must be positive.

    Examples:
        >>> frames = [Frame() for _ in range(10)]
        >>> for i, frame in enumerate(frames):
        ...     frame.metadata["time"] = i * 0.1  # 0.0, 0.1, 0.2, ...
        >>> traj = Trajectory(frames)
        >>> strategy = TimeIntervalStrategy(interval=0.5)
        >>> splitter = TrajectorySplitter(traj)
        >>> segments = splitter.split(strategy)
    """

    def __init__(self, interval: float) -> None:
        if interval <= 0:
            raise ValueError(f"Interval must be positive, got {interval}")
        self.interval = interval

    def get_split_indices(self, trajectory: Trajectory) -> list[int]:
        """Get split indices based on time intervals.

        Args:
            trajectory (Trajectory): The trajectory to split. Frames should
                have "time" in metadata.

        Returns:
            list[int]: List of split point indices based on time intervals.

        Note:
            Frames without time information are skipped. The first frame with
            time information is used as the reference start time.
        """
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
    """Split trajectory using a custom function.

    Allows users to define their own splitting logic by providing
    a function that returns split point indices.

    Args:
        split_func (Callable[[Trajectory], list[int]]): A function that takes
            a Trajectory and returns a list of split indices. The first index
            should be 0, and the last should be the total frame count.

    Examples:
        >>> def custom_split(traj):
        ...     # Split at frames 0, 5, 10, 15
        ...     return [0, 5, 10, 15]
        >>> strategy = CustomStrategy(custom_split)
        >>> splitter = TrajectorySplitter(traj)
        >>> segments = splitter.split(strategy)
    """

    def __init__(self, split_func: Callable[[Trajectory], list[int]]) -> None:
        self.split_func = split_func

    def get_split_indices(self, trajectory: Trajectory) -> list[int]:
        """Get split indices using the custom function.

        Args:
            trajectory (Trajectory): The trajectory to split.

        Returns:
            list[int]: List of split point indices from the custom function.
        """
        return self.split_func(trajectory)


class TrajectorySplitter:
    """Splits trajectories into lazy segments.

    Provides a convenient interface for splitting trajectories using
    various strategies. The resulting segments are lazy Trajectory objects
    that share the same topology as the original.

    Args:
        trajectory (Trajectory): The trajectory to split.

    Examples:
        >>> traj = Trajectory([Frame() for _ in range(100)])
        >>> splitter = TrajectorySplitter(traj)
        >>> # Split every 10 frames
        >>> segments = splitter.split_frames(interval=10)
        >>> len(segments)
        10
        >>> len(segments[0])
        10
    """

    def __init__(self, trajectory: Trajectory) -> None:
        """Initialize the splitter with a trajectory.

        Args:
            trajectory (Trajectory): The trajectory to split.
        """
        self.trajectory = trajectory

    def split(self, strategy: SplitStrategy) -> list[Trajectory]:
        """Split trajectory using a strategy, returning lazy segments.

        Args:
            strategy (SplitStrategy): The splitting strategy to use.

        Returns:
            list[Trajectory]: List of trajectory segments. Each segment is a
                new Trajectory containing a subset of frames from the original.

        Examples:
            >>> traj = Trajectory([Frame() for _ in range(20)])
            >>> strategy = FrameIntervalStrategy(interval=5)
            >>> splitter = TrajectorySplitter(traj)
            >>> segments = splitter.split(strategy)
            >>> len(segments)
            4
        """
        indices = strategy.get_split_indices(self.trajectory)

        segments = []
        for i in range(len(indices) - 1):
            start, end = indices[i], indices[i + 1]
            # Use trajectory slicing instead of accessing private members
            segment = self.trajectory[start:end]
            segments.append(segment)

        return segments

    def split_frames(self, interval: int) -> list[Trajectory]:
        """Convenience method for frame-based splitting.

        Args:
            interval (int): Number of frames per segment.

        Returns:
            list[Trajectory]: List of trajectory segments.

        Examples:
            >>> traj = Trajectory([Frame() for _ in range(20)])
            >>> splitter = TrajectorySplitter(traj)
            >>> segments = splitter.split_frames(interval=5)
            >>> len(segments)
            4
        """
        return self.split(FrameIntervalStrategy(interval))

    def split_time(self, interval: float) -> list[Trajectory]:
        """Convenience method for time-based splitting.

        Args:
            interval (float): Time interval for splitting (in frame time units).

        Returns:
            list[Trajectory]: List of trajectory segments.

        Examples:
            >>> frames = [Frame() for _ in range(10)]
            >>> for i, frame in enumerate(frames):
            ...     frame.metadata["time"] = i * 0.1
            >>> traj = Trajectory(frames)
            >>> splitter = TrajectorySplitter(traj)
            >>> segments = splitter.split_time(interval=0.5)
        """
        return self.split(TimeIntervalStrategy(interval))
