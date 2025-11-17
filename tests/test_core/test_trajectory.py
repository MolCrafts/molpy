"""Comprehensive tests for the trajectory module."""

from unittest.mock import Mock

import pytest

from molpy import Frame
from molpy.core.trajectory import (
    CustomStrategy,
    FrameIntervalStrategy,
    SplitStrategy,
    TimeIntervalStrategy,
    Trajectory,
    TrajectorySplitter,
)


@pytest.fixture
def mock_frame():
    """Create a mock frame for testing."""
    frame = Mock(spec=Frame)
    frame.metadata = {}
    return frame


@pytest.fixture
def mock_frames():
    """Create a list of mock frames for testing."""
    frames = []
    for i in range(10):
        frame = Mock(spec=Frame)
        frame.metadata = {"time": i * 0.5}  # Time: 0.0, 0.5, 1.0, ..., 4.5
        frames.append(frame)
    return frames


@pytest.fixture
def frame_generator(mock_frames):
    """Create a generator of frames for testing."""

    def gen():
        yield from mock_frames

    return gen()


class TestTrajectory:
    """Test the Trajectory class."""

    def test_init_with_list(self, mock_frames):
        traj = Trajectory(mock_frames)
        assert traj._frames is mock_frames
        assert traj._topology is None

    def test_init_with_topology(self, mock_frames):
        topology = Mock()
        traj = Trajectory(mock_frames, topology)
        assert traj._topology is topology

    def test_init_with_generator(self, frame_generator):
        traj = Trajectory(frame_generator)
        assert traj._topology is None

    def test_iteration_with_list(self, mock_frames):
        traj = Trajectory(mock_frames)
        result = list(traj)
        assert len(result) == len(mock_frames)

    def test_iteration_with_generator(self, frame_generator):
        traj = Trajectory(frame_generator)
        result = list(traj)
        assert len(result) == 10

    def test_len_with_list(self, mock_frames):
        traj = Trajectory(mock_frames)
        assert len(traj) == len(mock_frames)

    def test_len_with_generator_raises(self, frame_generator):
        traj = Trajectory(frame_generator)

        with pytest.raises(TypeError, match="Length not available for generator-based"):
            len(traj)

    def test_has_length_with_list(self, mock_frames):
        traj = Trajectory(mock_frames)
        assert traj.has_length() is True

    def test_has_length_with_generator(self, frame_generator):
        traj = Trajectory(frame_generator)
        assert traj.has_length() is False

    def test_getitem_int_returns_frame(self, mock_frames):
        traj = Trajectory(mock_frames)
        frame = traj[3]
        assert frame is mock_frames[3]

    def test_getitem_int_with_generator_raises(self, frame_generator):
        traj = Trajectory(frame_generator)

        with pytest.raises(
            TypeError, match="Indexing not supported for generator-based"
        ):
            traj[3]

    def test_getitem_slice_returns_trajectory(self, mock_frames):
        traj = Trajectory(mock_frames)
        sub_traj = traj[2:5]

        assert isinstance(sub_traj, Trajectory)
        assert sub_traj._topology is traj._topology
        assert len(sub_traj) == 3

    def test_getitem_slice_with_generator(self, frame_generator):
        """Slicing a generator-based trajectory should materialize it."""
        traj = Trajectory(frame_generator)
        sub_traj = traj[2:5]

        assert isinstance(sub_traj, Trajectory)
        assert len(sub_traj) == 3

    def test_getitem_invalid_type(self, mock_frames):
        traj = Trajectory(mock_frames)

        with pytest.raises(TypeError, match="Invalid key type"):
            traj[1.5]  # type: ignore[arg-type]

    def test_map_function_with_list(self, mock_frames):
        traj = Trajectory(mock_frames)

        def transform_frame(frame):
            new_frame = Mock(spec=Frame)
            new_frame.metadata = {"transformed": True}
            return new_frame

        mapped_traj = traj.map(transform_frame)

        assert isinstance(mapped_traj, Trajectory)

        # Test that mapping actually works
        result = list(mapped_traj)
        assert len(result) == len(mock_frames)
        assert all(f.metadata.get("transformed") for f in result)

    def test_map_function_with_generator(self, frame_generator):
        traj = Trajectory(frame_generator)

        def transform_frame(frame):
            new_frame = Mock(spec=Frame)
            new_frame.metadata = {"transformed": True}
            return new_frame

        mapped_traj = traj.map(transform_frame)

        assert isinstance(mapped_traj, Trajectory)

        # Test that mapping actually works
        result = list(mapped_traj)
        assert len(result) == 10
        assert all(f.metadata.get("transformed") for f in result)

    def test_next_manual_iteration(self, mock_frames):
        """Test manual iteration using next()."""
        traj = Trajectory(mock_frames)

        # Get first frame
        frame0 = next(traj)
        assert frame0 is mock_frames[0]

        # Get second frame
        frame1 = next(traj)
        assert frame1 is mock_frames[1]

        # Get third frame
        frame2 = next(traj)
        assert frame2 is mock_frames[2]

    def test_next_stop_iteration(self, mock_frames):
        """Test that next() raises StopIteration when exhausted."""
        traj = Trajectory(mock_frames)

        # Consume all frames
        for _ in range(len(mock_frames)):
            next(traj)

        # Next call should raise StopIteration
        with pytest.raises(StopIteration):
            next(traj)

    def test_next_with_generator(self, frame_generator):
        """Test next() with generator-based trajectory."""
        traj = Trajectory(frame_generator)

        # Get first few frames manually
        frame0 = next(traj)
        assert frame0.metadata["time"] == 0.0

        frame1 = next(traj)
        assert frame1.metadata["time"] == 0.5

    def test_next_and_iter_independent(self, mock_frames):
        """Test that next() and iter() use independent iterators."""
        traj = Trajectory(mock_frames)

        # Use next() to get first frame
        frame0_next = next(traj)
        assert frame0_next is mock_frames[0]

        # Use iter() to iterate from beginning
        frames_iter = list(traj)
        assert len(frames_iter) == len(mock_frames)
        assert frames_iter[0] is mock_frames[0]

    def test_map_preserves_next_functionality(self, mock_frames):
        """Test that mapped trajectory also supports next()."""
        traj = Trajectory(mock_frames)

        def add_marker(frame):
            new_frame = Mock(spec=Frame)
            new_frame.metadata = frame.metadata.copy()
            new_frame.metadata["marked"] = True
            return new_frame

        mapped_traj = traj.map(add_marker)

        # Use next() on mapped trajectory
        mapped_frame0 = next(mapped_traj)
        assert mapped_frame0.metadata.get("marked") is True

        mapped_frame1 = next(mapped_traj)
        assert mapped_frame1.metadata.get("marked") is True


class TestSplitStrategy:
    """Test the abstract SplitStrategy class."""

    def test_abstract_method(self):
        # Cannot instantiate abstract class directly
        with pytest.raises(TypeError, match="Can't instantiate abstract class"):
            SplitStrategy()  # type: ignore[abstract]


class TestFrameIntervalStrategy:
    """Test the FrameIntervalStrategy class."""

    def test_init(self):
        strategy = FrameIntervalStrategy(5)
        assert strategy.interval == 5

    def test_get_split_indices_sized_trajectory(self, mock_frames):
        traj = Trajectory(mock_frames)
        strategy = FrameIntervalStrategy(3)

        indices = strategy.get_split_indices(traj)
        expected = [0, 3, 6, 9, 10]  # For 10 frames with interval 3
        assert indices == expected

    def test_get_split_indices_exact_multiple(self, mock_frames):
        # Create a trajectory with exactly 9 frames
        traj = Trajectory(mock_frames[:9])
        strategy = FrameIntervalStrategy(3)

        indices = strategy.get_split_indices(traj)
        expected = [0, 3, 6, 9]
        assert indices == expected

    def test_get_split_indices_generator_raises(self, frame_generator):
        traj = Trajectory(frame_generator)
        strategy = FrameIntervalStrategy(3)

        with pytest.raises(
            TypeError,
            match="Frame interval splitting requires trajectory with known length",
        ):
            strategy.get_split_indices(traj)


class TestTimeIntervalStrategy:
    """Test the TimeIntervalStrategy class."""

    def test_init(self):
        strategy = TimeIntervalStrategy(1.0)
        assert strategy.interval == 1.0

    def test_get_split_indices_with_time_metadata(self, mock_frames):
        # Mock frames have time metadata: 0.0, 0.5, 1.0, 1.5, 2.0, ...
        traj = Trajectory(mock_frames)
        strategy = TimeIntervalStrategy(1.0)  # Split every 1.0 time unit

        indices = strategy.get_split_indices(traj)
        # Should split at times: 0.0, 1.0, 2.0, 3.0, 4.0
        # Corresponding to frame indices: 0, 2, 4, 6, 8, (10)
        expected = [0, 2, 4, 6, 8, 10]
        assert indices == expected

    def test_get_split_indices_no_time_metadata(self, mock_frames):
        # Remove time metadata
        for frame in mock_frames:
            frame.metadata = {}

        traj = Trajectory(mock_frames)
        strategy = TimeIntervalStrategy(1.0)

        indices = strategy.get_split_indices(traj)
        # Should only have start and end indices
        expected = [0, 10]
        assert indices == expected

    def test_get_split_indices_generator_exhausts(self, frame_generator):
        traj = Trajectory(frame_generator)
        strategy = TimeIntervalStrategy(1.0)

        indices = strategy.get_split_indices(traj)
        # Should work but exhaust the generator
        assert len(indices) >= 2
        assert indices[0] == 0
        assert indices[-1] == 10


class TestCustomStrategy:
    """Test the CustomStrategy class."""

    def test_init_and_call(self, mock_frames):
        def custom_split_func(traj):
            return [0, 5, 10]

        strategy = CustomStrategy(custom_split_func)
        traj = Trajectory(mock_frames)

        indices = strategy.get_split_indices(traj)
        assert indices == [0, 5, 10]


class TestTrajectorySplitter:
    """Test the TrajectorySplitter class."""

    def test_init(self, mock_frames):
        traj = Trajectory(mock_frames)
        splitter = TrajectorySplitter(traj)
        assert splitter.trajectory is traj

    def test_split_with_frame_interval(self, mock_frames):
        traj = Trajectory(mock_frames)
        splitter = TrajectorySplitter(traj)
        strategy = FrameIntervalStrategy(3)

        segments = splitter.split(strategy)

        assert len(segments) == 4  # [0:3], [3:6], [6:9], [9:10]
        assert all(isinstance(seg, Trajectory) for seg in segments)

    def test_split_frames_convenience(self, mock_frames):
        traj = Trajectory(mock_frames)
        splitter = TrajectorySplitter(traj)

        segments = splitter.split_frames(4)

        assert len(segments) == 3  # [0:4], [4:8], [8:10]
        assert all(isinstance(seg, Trajectory) for seg in segments)

    def test_split_time_convenience(self, mock_frames):
        traj = Trajectory(mock_frames)
        splitter = TrajectorySplitter(traj)

        segments = splitter.split_time(1.0)

        assert len(segments) >= 1
        assert all(isinstance(seg, Trajectory) for seg in segments)

    def test_split_preserves_topology(self, mock_frames):
        topology = Mock()
        traj = Trajectory(mock_frames, topology)
        splitter = TrajectorySplitter(traj)

        segments = splitter.split_frames(3)

        assert all(seg._topology is topology for seg in segments)


class TestErrorHandling:
    """Test error handling throughout the module."""

    def test_trajectory_slicing_edge_cases(self, mock_frames):
        traj = Trajectory(mock_frames)

        # Empty slice
        empty_traj = traj[5:5]
        assert isinstance(empty_traj, Trajectory)
        assert len(empty_traj) == 0

        # Slice beyond bounds
        beyond_traj = traj[8:20]
        assert isinstance(beyond_traj, Trajectory)
        assert len(beyond_traj) == 2  # Only frames 8 and 9


class TestIntegration:
    """Integration tests combining multiple components."""

    def test_full_workflow_with_generator(self, frame_generator):
        # Create trajectory from generator
        traj = Trajectory(frame_generator)

        # Apply transformation
        def add_id(frame):
            new_frame = Mock(spec=Frame)
            new_frame.metadata = frame.metadata.copy()
            new_frame.metadata["id"] = id(frame)
            return new_frame

        transformed_traj = traj.map(add_id)

        # Collect results
        result = list(transformed_traj)
        assert len(result) == 10
        assert all("id" in f.metadata for f in result)

    def test_full_workflow_with_splitting(self, mock_frames):
        # Create trajectory
        topology = Mock()
        traj = Trajectory(mock_frames, topology)

        # Split trajectory
        splitter = TrajectorySplitter(traj)
        segments = splitter.split_frames(3)

        # Verify segments
        assert len(segments) == 4

        # Test accessing frames in segments
        first_segment = segments[0]
        frames_in_first = list(first_segment)
        assert len(frames_in_first) == 3

        # Verify topology is preserved
        assert all(seg._topology is topology for seg in segments)
