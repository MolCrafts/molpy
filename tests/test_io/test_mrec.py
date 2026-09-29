"""Unit tests for ``*.mrec`` scientific-record I/O on :mod:`molpy.io`."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

import molpy as mp
from molpy import Block, Frame, Trajectory


_N_ATOMS = 3
_ATOM_X = (0.0, 1.0, 0.5)
_ATOM_Y = (0.25, 0.0, 2.0)
_ATOM_Z = (0.0, 4.0, 0.125)


def _coords_frame() -> Frame:
    atoms = Block()
    atoms["x"] = np.array(_ATOM_X, dtype=np.float64)
    atoms["y"] = np.array(_ATOM_Y, dtype=np.float64)
    atoms["z"] = np.array(_ATOM_Z, dtype=np.float64)
    frame = Frame()
    frame["atoms"] = atoms
    return frame


def _assert_coords(frame: Frame) -> None:
    atoms = frame["atoms"]
    assert atoms.nrows == _N_ATOMS
    np.testing.assert_array_equal(
        np.asarray(atoms["x"]), np.array(_ATOM_X, dtype=np.float64)
    )
    np.testing.assert_array_equal(
        np.asarray(atoms["y"]), np.array(_ATOM_Y, dtype=np.float64)
    )
    np.testing.assert_array_equal(
        np.asarray(atoms["z"]), np.array(_ATOM_Z, dtype=np.float64)
    )


class TestTrajectoryReader:
    def test_read_frame(self, tmp_path: Path) -> None:
        path = tmp_path / "traj.mrec"
        mp.io.write_mrec_trajectory(path, Trajectory([_coords_frame()]))
        reader = mp.io.mrec.TrajectoryReader(path)
        _assert_coords(reader.read_frame(0))


class TestWriteMrec:
    def test_round_trips_coordinates(self, tmp_path: Path) -> None:
        path = tmp_path / "snapshot.mrec"
        mp.io.write_mrec(path, _coords_frame())
        _assert_coords(mp.io.read_mrec(path))
        assert mp.io.mrec_sections(path) == frozenset({"meta", "frame"})
        meta = mp.io.read_mrec_meta(path)
        mp.io.mrec.schema.validate_meta(meta)
        assert meta["molrec_version"] == mp.io.mrec.schema.MOLREC_VERSION
        assert "format_name" not in meta


class TestWriteMrecSystem:
    def test_round_trips_coordinates(self, tmp_path: Path) -> None:
        path = tmp_path / "system.mrec"
        mp.io.write_mrec_system(path, _coords_frame())
        _assert_coords(mp.io.read_mrec_system(path))
        assert "frame" not in mp.io.mrec_sections(path)


class TestWriteMrecTrajectory:
    def test_round_trips_coordinates(self, tmp_path: Path) -> None:
        path = tmp_path / "traj.mrec"
        mp.io.write_mrec_trajectory(path, Trajectory([_coords_frame()]))
        loaded = mp.io.read_mrec_trajectory(path)
        assert len(loaded) == 1
        _assert_coords(loaded[0])


class TestSchema:
    def test_sole_version_key_is_molrec_version(self) -> None:
        assert mp.io.mrec.schema.MOLREC_VERSION == 1
        with pytest.raises(Exception, match="molrec_version"):
            mp.io.mrec.schema.validate_meta(
                {"record_schema_version": 1, "format_name": "mrec"}
            )
