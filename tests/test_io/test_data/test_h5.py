"""Strict schema-v2 tests for single-Frame HDF5 I/O."""

from __future__ import annotations

from pathlib import Path

import h5py
import numpy as np
import pytest
from molrs import Block, Frame, MetaValue

from molpy.core.box import Box
from molpy.io import read_h5, write_h5
from molpy.io.data.h5 import FRAME_SCHEMA_VERSION, HDF5Reader, HDF5Writer


def _all_meta_values() -> dict[str, MetaValue]:
    values = {
        "bool": True,
        "i32": -32,
        "i64": -(2**40),
        "u32": 32,
        "u64": 2**40,
        "f32": 1.25,
        "f64": np.pi,
        "string": "typed/元数据",
        "bool3": [True, False, True],
        "i32x3": [-1, 0, 1],
        "i64x3": [-(2**40), 0, 2**40],
        "u32x3": [1, 2, 3],
        "u64x3": [2**40, 2**41, 2**42],
        "f32x3": [1.25, 2.5, 5.0],
        "f64x3": [np.pi, np.e, np.sqrt(2.0)],
        "f32x6": list(np.arange(6, dtype=np.float32) / 3),
        "f64x6": list(np.arange(6, dtype=np.float64) / 7),
        "f32x9": list(np.arange(9, dtype=np.float32) / 11),
        "f64x9": list(np.arange(9, dtype=np.float64) / 13),
    }
    return {dtype: MetaValue(dtype, value) for dtype, value in values.items()}


def _frame(step: int = 0) -> Frame:
    frame = Frame(
        {
            "atoms": Block(
                {
                    "id": np.array([1, 2, 3], dtype=np.int64),
                    "position": np.arange(9, dtype=np.float64).reshape(3, 3),
                    "charge": np.array([-0.2, 0.1, 0.1], dtype=np.float32),
                    "element": np.array(["O", "H", "H"]),
                }
            ),
            "bonds": Block(
                {
                    "atomi": np.array([0, 0], dtype=np.int32),
                    "atomj": np.array([1, 2], dtype=np.int32),
                    "type": np.array(["O-H", "O-H"]),
                }
            ),
        }
    )
    frame.simbox = Box.tric(
        [10.0, 11.0, 12.0],
        [0.5, 1.0, -0.25],
        pbc=[True, False, True],
        origin=[-1.0, 2.0, 3.0],
    )
    meta = _all_meta_values()
    meta["step"] = MetaValue("i64", step)
    frame.meta = meta
    return frame


def _assert_meta_equal(actual: dict[str, MetaValue], expected: dict[str, MetaValue]):
    assert set(actual) == set(expected)
    for key, expected_value in expected.items():
        actual_value = actual[key]
        assert actual_value.dtype == expected_value.dtype
        if isinstance(expected_value.value, list):
            np.testing.assert_array_equal(actual_value.value, expected_value.value)
        else:
            assert actual_value.value == expected_value.value


def _assert_frame_equal(actual: Frame, expected: Frame) -> None:
    assert set(actual.keys()) == set(expected.keys())
    for block_name in expected.keys():
        actual_block = actual[block_name]
        expected_block = expected[block_name]
        assert set(actual_block.keys()) == set(expected_block.keys())
        for field in expected_block.keys():
            assert actual_block[field].dtype == expected_block[field].dtype
            np.testing.assert_array_equal(actual_block[field], expected_block[field])

    assert actual.simbox is not None
    assert expected.simbox is not None
    np.testing.assert_array_equal(actual.simbox.matrix, expected.simbox.matrix)
    np.testing.assert_array_equal(actual.simbox.origin, expected.simbox.origin)
    np.testing.assert_array_equal(actual.simbox.pbc, expected.simbox.pbc)
    _assert_meta_equal(actual.meta, expected.meta)


@pytest.mark.parametrize("compression", ["gzip", "lzf", None])
def test_roundtrip_exact_frame_surface(tmp_path: Path, compression: str | None):
    expected = _frame()
    path = tmp_path / f"frame-{compression}.h5"

    write_h5(path, expected, compression=compression)
    actual = read_h5(path)

    _assert_frame_equal(actual, expected)
    with h5py.File(path, "r") as h5_file:
        assert h5_file.attrs["frame_schema_version"] == FRAME_SCHEMA_VERSION
        assert set(h5_file) == {"blocks", "simbox", "meta"}
        assert h5_file["meta"].attrs["schema_version"] == FRAME_SCHEMA_VERSION
        assert len(h5_file["meta"]) == 20
        assert h5_file["blocks/atoms/position"].compression == compression


def test_all_nineteen_meta_dtypes_roundtrip_exactly(tmp_path: Path):
    frame = _frame()
    path = tmp_path / "typed-meta.h5"
    HDF5Writer(path).write(frame)

    actual = HDF5Reader(path).read()

    expected = _all_meta_values()
    _assert_meta_equal(
        {key: actual.meta[key] for key in expected},
        expected,
    )


def test_frame_without_simbox_and_empty_meta_roundtrips(tmp_path: Path):
    frame = Frame({"atoms": Block({"x": np.array([1.0])})})
    path = tmp_path / "minimal.h5"
    write_h5(path, frame)

    actual = read_h5(path)
    assert actual.simbox is None
    assert actual.meta == {}
    with h5py.File(path, "r") as h5_file:
        assert set(h5_file) == {"blocks", "meta"}


def test_reader_can_populate_explicit_frame(tmp_path: Path):
    path = tmp_path / "populate.h5"
    write_h5(path, _frame())
    target = Frame()

    result = read_h5(path, target)

    assert result is target
    assert result["atoms"].nrows == 3


def test_context_managers_use_schema_v2(tmp_path: Path):
    path = tmp_path / "context.h5"
    with HDF5Writer(path) as writer:
        writer.write(_frame())
    with HDF5Reader(path) as reader:
        actual = reader.read()
    assert actual.meta["string"].value == "typed/元数据"


def test_empty_frame_is_rejected(tmp_path: Path):
    with pytest.raises(ValueError, match="Cannot write empty frame"):
        write_h5(tmp_path / "empty.h5", Frame())


@pytest.mark.parametrize("version", [None, 1, 3, "2"])
def test_old_missing_future_or_wrong_type_frame_schema_is_rejected(
    tmp_path: Path, version: int | str | None
):
    path = tmp_path / "bad-version.h5"
    write_h5(path, _frame())
    with h5py.File(path, "a") as h5_file:
        if version is None:
            del h5_file.attrs["frame_schema_version"]
        else:
            h5_file.attrs["frame_schema_version"] = version

    with pytest.raises(ValueError, match="schema|requires exactly"):
        read_h5(path)


def test_old_box_and_metadata_layout_is_rejected(tmp_path: Path):
    path = tmp_path / "old-layout.h5"
    with h5py.File(path, "w") as h5_file:
        h5_file.attrs["frame_schema_version"] = FRAME_SCHEMA_VERSION
        h5_file.create_group("blocks")
        h5_file.create_group("box")
        h5_file.create_group("metadata")

    with pytest.raises(ValueError, match="missing groups|unknown groups"):
        read_h5(path)


@pytest.mark.parametrize("corruption", ["unknown_group", "unknown_attr"])
def test_unknown_frame_surface_is_rejected(tmp_path: Path, corruption: str):
    path = tmp_path / "unknown.h5"
    write_h5(path, _frame())
    with h5py.File(path, "a") as h5_file:
        if corruption == "unknown_group":
            h5_file.create_group("extra")
        else:
            h5_file.attrs["extra"] = 1

    with pytest.raises(ValueError, match="unknown groups|requires exactly"):
        read_h5(path)


@pytest.mark.parametrize(
    ("corruption", "match"),
    [
        ("meta_version", "meta schema"),
        ("unknown_dtype", "unknown dtype"),
        ("missing_dtype", "exactly key and dtype"),
        ("bad_name", "contiguous numeric"),
        ("duplicate_key", "Duplicate"),
        ("wrong_shape", "requires shape"),
    ],
)
def test_malformed_typed_meta_is_rejected(tmp_path: Path, corruption: str, match: str):
    path = tmp_path / "bad-meta.h5"
    write_h5(path, _frame())
    with h5py.File(path, "a") as h5_file:
        meta = h5_file["meta"]
        if corruption == "meta_version":
            meta.attrs["schema_version"] = 1
        elif corruption == "unknown_dtype":
            meta["00000000"].attrs["dtype"] = "json"
        elif corruption == "missing_dtype":
            del meta["00000000"].attrs["dtype"]
        elif corruption == "bad_name":
            meta.move("00000000", "entry")
        elif corruption == "duplicate_key":
            meta["00000001"].attrs["key"] = meta["00000000"].attrs["key"]
        else:
            dataset = meta["00000000"]
            key = dataset.attrs["key"]
            del meta["00000000"]
            replacement = meta.create_dataset(
                "00000000", data=np.array([True, False], dtype=np.bool_)
            )
            replacement.attrs["key"] = key
            replacement.attrs["dtype"] = "bool"

    with pytest.raises(ValueError, match=match):
        read_h5(path)


def test_block_without_exact_dtype_marker_is_rejected(tmp_path: Path):
    path = tmp_path / "bad-block.h5"
    write_h5(path, _frame())
    with h5py.File(path, "a") as h5_file:
        del h5_file["blocks/atoms/id"].attrs["dtype"]

    with pytest.raises(ValueError, match="exactly one 'dtype'"):
        read_h5(path)


def test_malformed_simbox_is_rejected(tmp_path: Path):
    path = tmp_path / "bad-simbox.h5"
    write_h5(path, _frame())
    with h5py.File(path, "a") as h5_file:
        del h5_file["simbox/origin"]

    with pytest.raises(ValueError, match="simbox must contain exactly"):
        read_h5(path)
