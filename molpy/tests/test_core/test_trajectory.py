"""molpy's trajectory splitters over the native ``mp.Trajectory``.

The container itself (indexing, slicing, ``map``) is ``molrs.core.Trajectory``,
tested in molrs; ``mp.Trajectory is molrs.core.Trajectory`` is ``test_init``'s.
"""

import numpy as np
import pytest
import molpy as mp


def _make_frame(time: float | None = None) -> mp.Frame:
    """A minimal real Frame with one atom and optional ``time`` metadata."""
    frame = mp.Frame()
    frame["atoms"] = {
        "x": np.array([0.0]),
        "y": np.array([0.0]),
        "z": np.array([0.0]),
    }
    if time is not None:
        frame.meta = {"time": mp.core.MetaValue("f64", time)}
    return frame


@pytest.fixture
def frames():
    """Ten real frames with time metadata 0.0, 0.5, 1.0, ..., 4.5."""
    return [_make_frame(time=i * 0.5) for i in range(10)]


class TestSplitStrategy:
    """Test the abstract SplitStrategy class."""

    def test_abstract_method(self):
        with pytest.raises(TypeError, match="Can't instantiate abstract class"):
            mp.core.SplitStrategy()  # type: ignore[abstract]


class TestFrameIntervalStrategy:
    """Test the FrameIntervalStrategy class."""

    def test_init(self):
        assert mp.core.FrameIntervalStrategy(5).interval == 5

    def test_init_rejects_nonpositive(self):
        with pytest.raises(ValueError, match="must be positive"):
            mp.core.FrameIntervalStrategy(0)

    def test_get_split_indices(self, frames):
        traj = mp.Trajectory(frames)
        indices = mp.core.FrameIntervalStrategy(3).get_split_indices(traj)
        assert indices == [0, 3, 6, 9, 10]

    def test_get_split_indices_exact_multiple(self, frames):
        traj = mp.Trajectory(frames[:9])
        indices = mp.core.FrameIntervalStrategy(3).get_split_indices(traj)
        assert indices == [0, 3, 6, 9]


class TestTimeIntervalStrategy:
    """Test the TimeIntervalStrategy class."""

    def test_init(self):
        assert mp.core.TimeIntervalStrategy(1.0).interval == 1.0

    def test_get_split_indices_with_time_array(self, frames):
        # Native time array 0.0, 0.5, 1.0, ...
        times = np.array([i * 0.5 for i in range(10)])
        traj = mp.Trajectory(frames, time=times)
        indices = mp.core.TimeIntervalStrategy(1.0).get_split_indices(traj)
        assert indices == [0, 2, 4, 6, 8, 10]

    def test_bins_are_fixed_width_from_the_first_time(self):
        # 3.0, 3.2 and 3.4 share bin [3, 4); 0.5 shares bin [0, 1) with 0.0.
        times = np.array([0.0, 0.5, 3.0, 3.2, 3.4])
        traj = mp.Trajectory([_make_frame() for _ in times], time=times)
        assert mp.core.TimeIntervalStrategy(1.0).get_split_indices(traj) == [0, 2, 5]

    def test_bins_start_at_the_first_time(self):
        times = np.array([10.0, 10.9, 11.0, 12.5])
        traj = mp.Trajectory([_make_frame() for _ in times], time=times)
        assert mp.core.TimeIntervalStrategy(1.0).get_split_indices(traj) == [0, 2, 3, 4]

    def test_falls_back_to_each_frames_meta_time(self, frames):
        traj = mp.Trajectory(frames)  # no time array; frames carry meta["time"]
        assert traj.time is None
        indices = mp.core.TimeIntervalStrategy(1.0).get_split_indices(traj)
        assert indices == [0, 2, 4, 6, 8, 10]

    def test_no_time_anywhere_raises(self):
        traj = mp.Trajectory([_make_frame() for _ in range(10)])
        with pytest.raises(ValueError, match="no time array"):
            mp.core.TimeIntervalStrategy(1.0).get_split_indices(traj)

    def test_decreasing_times_raise(self):
        times = np.array([0.0, 2.0, 1.0])
        traj = mp.Trajectory([_make_frame() for _ in times], time=times)
        with pytest.raises(ValueError, match="must not decrease"):
            mp.core.TimeIntervalStrategy(1.0).get_split_indices(traj)

    @pytest.mark.parametrize(
        "times, message",
        [
            ([-1e308, 1e308], "time bins exceed finite range"),
            ([1e308, -1e308], "must not decrease"),
        ],
    )
    def test_extreme_finite_times_raise_value_error(self, times, message):
        traj = mp.Trajectory([mp.Frame(), mp.Frame()], time=np.asarray(times))
        with pytest.raises(ValueError, match=message):
            mp.core.TimeIntervalStrategy(1.0).get_split_indices(traj)

    def test_time_bins_above_int64_range_remain_distinct(self):
        traj = mp.Trajectory(
            [mp.Frame(), mp.Frame(), mp.Frame()], time=np.array([0.0, 1.0, 2.0])
        )
        assert mp.core.TimeIntervalStrategy(1e-20).get_split_indices(traj) == [
            0,
            1,
            2,
            3,
        ]


class TestCustomStrategy:
    """Test the CustomStrategy class."""

    def test_init_and_call(self, frames):
        strategy = mp.core.CustomStrategy(lambda traj: [0, 5, 10])
        traj = mp.Trajectory(frames)
        assert strategy.get_split_indices(traj) == [0, 5, 10]


class TestTrajectorySplitter:
    """Test the TrajectorySplitter class."""

    def test_init(self, frames):
        traj = mp.Trajectory(frames)
        splitter = mp.core.TrajectorySplitter(traj)
        assert splitter.trajectory is traj

    def test_split_with_frame_interval(self, frames):
        traj = mp.Trajectory(frames)
        segments = mp.core.TrajectorySplitter(traj).split(
            mp.core.FrameIntervalStrategy(3)
        )
        assert len(segments) == 4  # [0:3], [3:6], [6:9], [9:10]
        assert all(isinstance(seg, mp.Trajectory) for seg in segments)

    def test_time_split_segments_keep_their_times(self, frames):
        times = np.array([i * 0.5 for i in range(10)])
        traj = mp.Trajectory(frames, time=times)
        segments = mp.core.TrajectorySplitter(traj).split(
            mp.core.TimeIntervalStrategy(1.0)
        )
        assert [len(seg) for seg in segments] == [2, 2, 2, 2, 2]
        np.testing.assert_allclose(segments[1].time, [1.0, 1.5])

    def test_time_split_without_a_time_array_uses_meta(self, frames):
        segments = mp.core.TrajectorySplitter(mp.Trajectory(frames)).split(
            mp.core.TimeIntervalStrategy(2.0)
        )
        assert [len(seg) for seg in segments] == [4, 4, 2]

    @pytest.mark.parametrize(
        "strategy",
        [mp.core.FrameIntervalStrategy(3), mp.core.TimeIntervalStrategy(1.0)],
        ids=["frames", "time"],
    )
    def test_an_empty_trajectory_has_no_segments(self, strategy):
        assert mp.core.TrajectorySplitter(mp.Trajectory([])).split(strategy) == []

    @pytest.mark.parametrize(
        "indices",
        [
            [],
            [1, 10],  # does not start at 0
            [0, 5],  # does not end at the frame count
            [0, 5, 5, 10],  # an empty segment
            [0, 6, 4, 10],  # not rising
            [0, 5.0, 10],  # not integers
            [0, 11],  # past the end
        ],
    )
    def test_custom_indices_are_validated(self, frames, indices):
        splitter = mp.core.TrajectorySplitter(mp.Trajectory(frames))
        with pytest.raises(ValueError, match="split indices"):
            splitter.split(mp.core.CustomStrategy(lambda traj: indices))

    def test_custom_indices_accept_numpy_integers(self, frames):
        segments = mp.core.TrajectorySplitter(mp.Trajectory(frames)).split(
            mp.core.CustomStrategy(lambda traj: np.array([0, 4, 10]))
        )
        assert [len(seg) for seg in segments] == [4, 6]


def test_time_bins_reject_nonfinite_values():
    import numpy as np
    import pytest

    for interval in [np.nan, np.inf, -np.inf]:
        with pytest.raises(ValueError):
            mp.core.TimeIntervalStrategy(interval)
    trajectory = mp.Trajectory([mp.Frame(), mp.Frame()], time=np.array([0.0, np.nan]))
    with pytest.raises(ValueError, match="finite"):
        mp.core.TimeIntervalStrategy(1.0).get_split_indices(trajectory)
