"""Strict schema-v2 tests for HDF5 trajectory I/O."""

from __future__ import annotations

from pathlib import Path

import h5py
import numpy as np
import pytest
from molrs import Block, Frame, MetaValue

from molpy.core.box import Box
from molpy.io import read_h5_trajectory, write_h5_trajectory
from molpy.io.data.h5 import FRAME_SCHEMA_VERSION
from molpy.io.trajectory.h5 import (
    TRAJECTORY_SCHEMA_VERSION,
    HDF5TrajectoryReader,
    HDF5TrajectoryWriter,
)


def _frame(step: int) -> Frame:
    frame = Frame(
        {
            "atoms": Block(
                {
                    "id": np.array([1, 2], dtype=np.int64),
                    "position": np.array(
                        [[step, 0.0, 0.0], [step + 1.0, 0.0, 0.0]],
                        dtype=np.float64,
                    ),
                    "element": np.array(["H", "He"]),
                }
            )
        }
    )
    frame.simbox = Box.cubic(10.0 + step, origin=[-1.0, -1.0, -1.0])
    frame.meta = {
        "step": MetaValue("i64", step),
        "time": MetaValue("f64", step * 0.5),
        "label": MetaValue("string", f"frame-{step}"),
    }
    return frame


def _assert_frame(actual: Frame, expected: Frame) -> None:
    assert set(actual.keys()) == set(expected.keys())
    for block_name in expected.keys():
        assert set(actual[block_name].keys()) == set(expected[block_name].keys())
        for field in expected[block_name].keys():
            assert actual[block_name][field].dtype == expected[block_name][field].dtype
            np.testing.assert_array_equal(
                actual[block_name][field], expected[block_name][field]
            )
    np.testing.assert_array_equal(actual.simbox.matrix, expected.simbox.matrix)
    np.testing.assert_array_equal(actual.simbox.origin, expected.simbox.origin)
    np.testing.assert_array_equal(actual.simbox.pbc, expected.simbox.pbc)
    assert set(actual.meta) == set(expected.meta)
    for key in expected.meta:
        assert actual.meta[key].dtype == expected.meta[key].dtype
        assert actual.meta[key].value == expected.meta[key].value


@pytest.mark.parametrize("compression", ["gzip", "lzf", None])
def test_trajectory_roundtrip_random_access_and_iteration(
    tmp_path: Path, compression: str | None
):
    expected = [_frame(index) for index in range(4)]
    path = tmp_path / f"trajectory-{compression}.h5"
    write_h5_trajectory(path, expected, compression=compression)

    reader = read_h5_trajectory(path)
    assert reader.n_frames == 4
    assert len(reader) == 4
    _assert_frame(reader.read_frame(0), expected[0])
    _assert_frame(reader.read_frame(-1), expected[-1])
    _assert_frame(reader[2], expected[2])
    assert [frame.meta["step"].value for frame in reader] == [0, 1, 2, 3]

    with h5py.File(path, "r") as h5_file:
        assert h5_file.attrs["trajectory_schema_version"] == TRAJECTORY_SCHEMA_VERSION
        assert h5_file.attrs["n_frames"] == 4
        assert set(h5_file) == {"frames"}
        for index in range(4):
            frame_group = h5_file[f"frames/{index}"]
            assert frame_group.attrs["frame_schema_version"] == FRAME_SCHEMA_VERSION
            assert set(frame_group) == {"blocks", "simbox", "meta"}
            assert frame_group["blocks/atoms/position"].compression == compression


def test_empty_trajectory_is_valid_schema_v2(tmp_path: Path):
    path = tmp_path / "empty.h5"
    write_h5_trajectory(path, [])

    reader = HDF5TrajectoryReader(path)
    assert reader.n_frames == 0
    assert list(reader) == []
    with h5py.File(path, "r") as h5_file:
        assert h5_file.attrs["trajectory_schema_version"] == 2
        assert h5_file.attrs["n_frames"] == 0
        assert set(h5_file["frames"]) == set()


def test_incremental_append_preserves_contiguous_indices(tmp_path: Path):
    path = tmp_path / "append.h5"
    with HDF5TrajectoryWriter(path) as writer:
        writer.write_frame(_frame(0))
        writer.write_frame(_frame(1))
    with HDF5TrajectoryWriter(path) as writer:
        writer.write_frame(_frame(2))

    reader = HDF5TrajectoryReader(path)
    assert reader.n_frames == 3
    assert [reader[index].meta["step"].value for index in range(3)] == [0, 1, 2]


def test_reader_and_writer_context_lifecycle(tmp_path: Path):
    path = tmp_path / "context.h5"
    writer = HDF5TrajectoryWriter(path)
    with writer:
        writer.write_frame(_frame(0))
    assert writer._file is None

    reader = HDF5TrajectoryReader(path)
    with reader:
        assert reader.read_frame(0).meta["label"].value == "frame-0"
    assert reader._file is None


@pytest.mark.parametrize("index", [1, -2, 100])
def test_invalid_frame_index_is_rejected(tmp_path: Path, index: int):
    path = tmp_path / "index.h5"
    write_h5_trajectory(path, [_frame(0)])
    with pytest.raises(IndexError, match="out of range"):
        HDF5TrajectoryReader(path).read_frame(index)


@pytest.mark.parametrize("version", [None, 1, 3, "2"])
def test_old_missing_future_or_wrong_type_trajectory_schema_is_rejected(
    tmp_path: Path, version: int | str | None
):
    path = tmp_path / "bad-version.h5"
    write_h5_trajectory(path, [_frame(0)])
    with h5py.File(path, "a") as h5_file:
        if version is None:
            del h5_file.attrs["trajectory_schema_version"]
        else:
            h5_file.attrs["trajectory_schema_version"] = version

    with pytest.raises(ValueError, match="schema|requires exactly"):
        HDF5TrajectoryReader(path)
    with pytest.raises(ValueError, match="schema|requires exactly"):
        HDF5TrajectoryWriter(path)


@pytest.mark.parametrize(
    ("corruption", "match"),
    [
        ("missing_frames", "exactly the frames group"),
        ("unknown_group", "exactly the frames group"),
        ("unknown_attr", "requires exactly"),
        ("count_mismatch", "contiguous and match"),
        ("noncontiguous", "contiguous and match"),
        ("nframes_type", "non-negative integer"),
    ],
)
def test_malformed_trajectory_surface_is_rejected(
    tmp_path: Path, corruption: str, match: str
):
    path = tmp_path / "bad-surface.h5"
    write_h5_trajectory(path, [_frame(0), _frame(1)])
    with h5py.File(path, "a") as h5_file:
        if corruption == "missing_frames":
            del h5_file["frames"]
        elif corruption == "unknown_group":
            h5_file.create_group("metadata")
        elif corruption == "unknown_attr":
            h5_file.attrs["metadata"] = "old"
        elif corruption == "count_mismatch":
            h5_file.attrs["n_frames"] = 3
        elif corruption == "noncontiguous":
            h5_file["frames"].move("1", "2")
        else:
            h5_file.attrs["n_frames"] = "2"

    with pytest.raises(ValueError, match=match):
        HDF5TrajectoryReader(path)


def test_corrupt_embedded_frame_schema_is_rejected_on_read(tmp_path: Path):
    path = tmp_path / "bad-frame.h5"
    write_h5_trajectory(path, [_frame(0)])
    with h5py.File(path, "a") as h5_file:
        h5_file["frames/0"].attrs["frame_schema_version"] = 1

    reader = HDF5TrajectoryReader(path)
    with pytest.raises(ValueError, match="Frame schema"):
        reader.read_frame(0)


def test_old_embedded_box_metadata_layout_is_rejected(tmp_path: Path):
    path = tmp_path / "old-frame-layout.h5"
    write_h5_trajectory(path, [_frame(0)])
    with h5py.File(path, "a") as h5_file:
        frame = h5_file["frames/0"]
        frame.move("simbox", "box")
        frame.move("meta", "metadata")

    with pytest.raises(ValueError, match="missing groups|unknown groups"):
        HDF5TrajectoryReader(path).read_frame(0)


def test_failed_frame_write_does_not_leave_partial_group(tmp_path: Path):
    path = tmp_path / "atomic.h5"
    writer = HDF5TrajectoryWriter(path)
    with pytest.raises(ValueError, match="Cannot write empty frame"):
        writer.write_frame(Frame())
    writer.close()

    reader = HDF5TrajectoryReader(path)
    assert reader.n_frames == 0
    with h5py.File(path, "r") as h5_file:
        assert set(h5_file["frames"]) == set()
