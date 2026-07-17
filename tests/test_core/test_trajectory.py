"""Tests for the trajectory module (molrs-backed eager container + splitters)."""

import numpy as np
import pytest

from molrs import Frame, MetaValue
from molpy.core.trajectory import (
    CustomStrategy,
    FrameIntervalStrategy,
    SplitStrategy,
    TimeIntervalStrategy,
    Trajectory,
    TrajectorySplitter,
)


def _make_frame(time: float | None = None) -> Frame:
    """A minimal real Frame with one atom and optional ``time`` metadata."""
    frame = Frame()
    frame["atoms"] = {
        "x": np.array([0.0]),
        "y": np.array([0.0]),
        "z": np.array([0.0]),
    }
    if time is not None:
        frame.meta = {"time": MetaValue("f64", time)}
    return frame


@pytest.fixture
def frames():
    """Ten real frames with time metadata 0.0, 0.5, 1.0, ..., 4.5."""
    return [_make_frame(time=i * 0.5) for i in range(10)]


class TestTrajectory:
    """Test the molrs-backed Trajectory container."""

    def test_is_molrs_trajectory_subclass(self):
        import molrs

        assert issubclass(Trajectory, molrs.Trajectory)

    def test_init_with_list(self, frames):
        traj = Trajectory(frames)
        assert len(traj) == len(frames)
        assert traj.topology is None

    def test_init_with_topology(self, frames):
        topology = object()
        traj = Trajectory(frames, topology)
        assert traj._topology is topology
        assert traj.topology is topology

    def test_iteration(self, frames):
        traj = Trajectory(frames)
        assert len(list(traj)) == len(frames)

    def test_len(self, frames):
        traj = Trajectory(frames)
        assert len(traj) == len(frames)

    def test_getitem_int_returns_frame(self, frames):
        traj = Trajectory(frames)
        frame = traj[3]
        assert isinstance(frame, Frame)

    def test_getitem_slice_returns_trajectory(self, frames):
        traj = Trajectory(frames)
        sub_traj = traj[2:5]

        assert isinstance(sub_traj, Trajectory)
        assert sub_traj._topology is traj._topology
        assert len(sub_traj) == 3

    def test_getitem_slice_preserves_topology(self, frames):
        topology = object()
        traj = Trajectory(frames, topology)
        assert traj[2:5]._topology is topology

    def test_getitem_invalid_type_raises(self, frames):
        traj = Trajectory(frames)
        with pytest.raises(TypeError):
            traj[1.5]  # type: ignore[arg-type]

    def test_map_function(self, frames):
        traj = Trajectory(frames)

        def shift(frame):
            atoms = frame["atoms"]
            atoms["x"] = atoms["x"] + 1.0
            return frame

        mapped_traj = traj.map(shift)

        assert isinstance(mapped_traj, Trajectory)
        result = list(mapped_traj)
        assert len(result) == len(frames)
        # Column data round-trips through the molrs store; x shifted 0 -> 1.
        assert all(f["atoms"]["x"][0] == 1.0 for f in result)

    def test_map_preserves_topology(self, frames):
        topology = object()
        traj = Trajectory(frames, topology)
        assert traj.map(lambda f: f)._topology is topology

    def test_repr(self, frames):
        traj = Trajectory(frames)
        assert "n_frames=10" in repr(traj)


class TestSplitStrategy:
    """Test the abstract SplitStrategy class."""

    def test_abstract_method(self):
        with pytest.raises(TypeError, match="Can't instantiate abstract class"):
            SplitStrategy()  # type: ignore[abstract]


class TestFrameIntervalStrategy:
    """Test the FrameIntervalStrategy class."""

    def test_init(self):
        assert FrameIntervalStrategy(5).interval == 5

    def test_init_rejects_nonpositive(self):
        with pytest.raises(ValueError, match="must be positive"):
            FrameIntervalStrategy(0)

    def test_get_split_indices(self, frames):
        traj = Trajectory(frames)
        indices = FrameIntervalStrategy(3).get_split_indices(traj)
        assert indices == [0, 3, 6, 9, 10]

    def test_get_split_indices_exact_multiple(self, frames):
        traj = Trajectory(frames[:9])
        indices = FrameIntervalStrategy(3).get_split_indices(traj)
        assert indices == [0, 3, 6, 9]


class TestTimeIntervalStrategy:
    """Test the TimeIntervalStrategy class."""

    def test_init(self):
        assert TimeIntervalStrategy(1.0).interval == 1.0

    def test_get_split_indices_with_time_array(self, frames):
        # Native time array 0.0, 0.5, 1.0, ...
        times = np.array([i * 0.5 for i in range(10)])
        traj = Trajectory(frames, time=times)
        indices = TimeIntervalStrategy(1.0).get_split_indices(traj)
        assert indices == [0, 2, 4, 6, 8, 10]

    def test_get_split_indices_no_time(self):
        traj = Trajectory([_make_frame() for _ in range(10)])
        indices = TimeIntervalStrategy(1.0).get_split_indices(traj)
        assert indices == [0, 10]


class TestCustomStrategy:
    """Test the CustomStrategy class."""

    def test_init_and_call(self, frames):
        strategy = CustomStrategy(lambda traj: [0, 5, 10])
        traj = Trajectory(frames)
        assert strategy.get_split_indices(traj) == [0, 5, 10]


class TestTrajectorySplitter:
    """Test the TrajectorySplitter class."""

    def test_init(self, frames):
        traj = Trajectory(frames)
        splitter = TrajectorySplitter(traj)
        assert splitter.trajectory is traj

    def test_split_with_frame_interval(self, frames):
        traj = Trajectory(frames)
        segments = TrajectorySplitter(traj).split(FrameIntervalStrategy(3))
        assert len(segments) == 4  # [0:3], [3:6], [6:9], [9:10]
        assert all(isinstance(seg, Trajectory) for seg in segments)

    def test_split_frames_convenience(self, frames):
        traj = Trajectory(frames)
        segments = TrajectorySplitter(traj).split_frames(4)
        assert len(segments) == 3  # [0:4], [4:8], [8:10]

    def test_split_time_convenience(self, frames):
        times = np.array([i * 0.5 for i in range(10)])
        traj = Trajectory(frames, time=times)
        segments = TrajectorySplitter(traj).split_time(1.0)
        assert len(segments) >= 1
        assert all(isinstance(seg, Trajectory) for seg in segments)

    def test_split_preserves_topology(self, frames):
        topology = object()
        traj = Trajectory(frames, topology)
        segments = TrajectorySplitter(traj).split_frames(3)
        assert all(seg._topology is topology for seg in segments)


class TestErrorHandling:
    """Edge cases."""

    def test_trajectory_slicing_edge_cases(self, frames):
        traj = Trajectory(frames)

        empty_traj = traj[5:5]
        assert isinstance(empty_traj, Trajectory)
        assert len(empty_traj) == 0

        beyond_traj = traj[8:20]
        assert isinstance(beyond_traj, Trajectory)
        assert len(beyond_traj) == 2  # Only frames 8 and 9


class TestIntegration:
    """Integration tests combining multiple components."""

    def test_full_workflow_with_splitting(self, frames):
        topology = object()
        traj = Trajectory(frames, topology)

        segments = TrajectorySplitter(traj).split_frames(3)
        assert len(segments) == 4

        first_segment = segments[0]
        assert len(list(first_segment)) == 3
        assert all(seg._topology is topology for seg in segments)
