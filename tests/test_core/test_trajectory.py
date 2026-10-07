"""molpy's trajectory splitters over the native ``mp.Trajectory``.

The container itself (indexing, slicing, ``map``) is ``molrs.core.Trajectory``,
tested in molrs; ``mp.Trajectory is molrs.core.Trajectory`` is ``test_init``'s.
"""

import numpy as np
import pytest

from molpy.core import (
    CustomStrategy,
    Frame,
    FrameIntervalStrategy,
    MetaValue,
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

    def test_time_split_segments_keep_their_times(self, frames):
        times = np.array([i * 0.5 for i in range(10)])
        traj = Trajectory(frames, time=times)
        segments = TrajectorySplitter(traj).split(TimeIntervalStrategy(1.0))
        assert [len(seg) for seg in segments] == [2, 2, 2, 2, 2]
        np.testing.assert_allclose(segments[1].time, [1.0, 1.5])
