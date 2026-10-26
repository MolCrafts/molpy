"""``mp.core.Trajectory``: the in-memory frame sequence."""

import molpy as mp
import numpy as np
import pytest
from molpy.core import Frame, Trajectory


def _frames(n: int) -> list[Frame]:
    frames = []
    for i in range(n):
        frame = Frame()
        frame["atoms"] = {"x": np.array([float(i)])}
        frames.append(frame)
    return frames


def _x(frame: Frame) -> float:
    return float(frame["atoms"]["x"][0])


def _labelled(n: int = 5) -> Trajectory:
    return Trajectory(
        _frames(n),
        step=np.arange(n, dtype=np.int64) * 10,
        time=np.arange(n, dtype=np.float64) * 0.5,
    )


def test_integer_index_counts_from_either_end():
    traj = _labelled()
    assert _x(traj[0]) == 0.0
    assert _x(traj[-1]) == 4.0
    assert _x(traj[-5]) == 0.0
    with pytest.raises(IndexError):
        traj[5]
    with pytest.raises(IndexError):
        traj[-6]


def test_an_indexed_frame_is_the_trajectorys_own():
    """``traj[i]`` is live, as ``frame["atoms"]`` is: an edit through it is
    an edit to the trajectory, and a slice shares the same frames."""
    traj = _labelled()
    traj[1]["atoms"]["x"] = np.array([42.0])
    assert _x(traj[1]) == 42.0
    assert _x(traj[1:3][0]) == 42.0
    traj[1:3][1]["atoms"]["x"] = np.array([7.0])
    assert _x(traj[2]) == 7.0


def test_a_slice_is_a_sub_trajectory_with_its_labels():
    sub = _labelled()[1:5:2]
    assert isinstance(sub, Trajectory)
    assert [_x(f) for f in sub] == [1.0, 3.0]
    np.testing.assert_array_equal(sub.step, [10, 30])
    np.testing.assert_array_equal(sub.time, [0.5, 1.5])

    reverse = _labelled()[::-1]
    assert [_x(f) for f in reverse] == [4.0, 3.0, 2.0, 1.0, 0.0]
    assert len(_labelled()[3:1]) == 0


def test_an_unlabelled_slice_stays_unlabelled():
    sub = Trajectory(_frames(3))[1:]
    assert len(sub) == 2
    assert sub.step is None and sub.time is None


def test_map_builds_a_new_trajectory_and_keeps_the_labels():
    traj = _labelled(3)

    def shift(frame: Frame) -> Frame:
        frame["atoms"]["x"] = frame["atoms"]["x"] + 100.0
        return frame

    mapped = traj.map(shift)
    assert [_x(f) for f in mapped] == [100.0, 101.0, 102.0]
    assert [_x(f) for f in traj] == [0.0, 1.0, 2.0]
    np.testing.assert_array_equal(mapped.step, traj.step)
    np.testing.assert_array_equal(mapped.time, traj.time)


def test_map_refuses_a_non_frame():
    with pytest.raises(TypeError, match="Frame"):
        _labelled(2).map(lambda frame: 1)


def test_repr_counts_the_frames():
    assert repr(_labelled(3)) == "Trajectory(n_frames=3)"


def test_labels_must_match_the_frames():
    with pytest.raises(ValueError):
        Trajectory(_frames(2), step=np.array([0], dtype=np.int64))


def test_is_the_core_trajectory_class():
    assert mp.core.Trajectory is Trajectory


def test_observable_record_mirrors_new_name_kind_values():
    record = mp.core.ObservableRecord("rg", "scalar", np.array([1.0, 2.0]))
    assert (record.name, record.kind) == ("rg", "scalar")
    np.testing.assert_array_equal(record.values, [1.0, 2.0])
    assert record.description == ""
    assert record.unit is None
    assert record.axes == []
    assert record.time_dependent is False


def test_observable_record_carries_its_metadata_and_another_kind_verbatim():
    record = mp.core.ObservableRecord(
        "dipole",
        "vector",
        np.zeros((2, 3)),
        description="total dipole",
        unit="e*angstrom",
        axes=["frame", "xyz"],
        time_dependent=True,
        sampling="every step",
        domain="time",
        target="atoms",
    )
    assert record.kind == "vector"
    assert (record.unit, record.axes, record.time_dependent) == (
        "e*angstrom",
        ["frame", "xyz"],
        True,
    )
    assert (record.sampling, record.domain, record.target) == (
        "every step",
        "time",
        "atoms",
    )
    spectrum = mp.core.ObservableRecord("s", "spectrum", np.array([0.5]))
    assert spectrum.kind == "spectrum"


def test_observable_record_pickles():
    import pickle

    record = mp.core.ObservableRecord(
        "names", "scalar", ["a", "b"], unit="1", axes=["i"]
    )
    back = pickle.loads(pickle.dumps(record))
    assert (back.name, back.kind, back.values, back.unit, back.axes) == (
        "names",
        "scalar",
        ["a", "b"],
        "1",
        ["i"],
    )


def test_observable_record_refuses_an_unsupported_value():
    with pytest.raises(TypeError):
        mp.core.ObservableRecord("bad", "scalar", object())


def test_the_constructor_takes_its_own_copies():
    source = _frames(1)[0]
    trajectory = Trajectory([source])
    source["atoms"]["x"][0] = 41
    assert _x(trajectory[0]) == 0


@pytest.mark.parametrize(
    "values",
    [
        np.array([2**40], dtype=np.int64),
        np.array([7], dtype=np.uint8),
        np.array([1 + 2j], dtype=np.complex128),
        2**53 + 1,
        True,
    ],
)
def test_observable_preserves_values_and_dtype(values):
    result = mp.core.ObservableRecord("n", "scalar", values).values
    expected = np.asarray(values)
    np.testing.assert_array_equal(result, expected)
    assert result.dtype == expected.dtype


@pytest.mark.parametrize("shape", [(2, 2), (0, 2)])
def test_string_observable_shape_survives_values_and_pickle(shape):
    import pickle

    values = np.full(shape, "a")
    record = mp.core.ObservableRecord("labels", "vector", values)
    np.testing.assert_array_equal(np.asarray(record.values), values)
    restored = pickle.loads(pickle.dumps(record))
    np.testing.assert_array_equal(np.asarray(restored.values), values)
