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


def _water_forcefield() -> mp.ForceField:
    ff = mp.ForceField(name="water", units="real")
    atoms = ff.def_style("atom", "full")
    ow = atoms.def_type("OW", mass=15.999, charge=-0.834)
    hw = atoms.def_type("HW", mass=1.008, charge=0.417)
    ff.def_style("bond", "harmonic").def_type("OW-HW", ow, hw, k=450.0, r0=0.9572)
    pair = ff.def_style("pair", "lj/cut", {"cutoff": 10.0})
    pair.def_type("OW", ow, epsilon=0.1521, sigma=3.1507)
    pair.def_type("HW", hw, epsilon=0.0, sigma=0.0)
    return ff


def _ff_rows(ff: mp.ForceField) -> dict:
    return {
        (style.category, style.name): {t.name: dict(t.params) for t in style.types}
        for style in ff.styles
    }


class TestForceFieldSection:
    def test_rides_along_a_snapshot(self, tmp_path: Path) -> None:
        path = tmp_path / "snapshot.mrec"
        ff = _water_forcefield()
        mp.io.write_mrec(path, _coords_frame(), forcefield=ff)
        assert "forcefield" in mp.io.mrec_sections(path)
        section = mp.io.read_mrec_forcefield(path)
        assert isinstance(section, mp.io.mrec.ForceFieldSection)
        loaded = mp.ForceField.from_section(section)
        assert loaded.units == "real"
        assert _ff_rows(loaded) == _ff_rows(ff)
        _assert_coords(mp.io.read_mrec(path))

    def test_standalone_package(self, tmp_path: Path) -> None:
        path = tmp_path / "ff.mrec"
        ff = _water_forcefield()
        mp.io.write_mrec_forcefield(path, ff)
        loaded = mp.ForceField.from_section(mp.io.read_mrec_forcefield(path))
        assert _ff_rows(loaded) == _ff_rows(ff)

    def test_absent_section_reads_none(self, tmp_path: Path) -> None:
        path = tmp_path / "snapshot.mrec"
        mp.io.write_mrec(path, _coords_frame())
        assert mp.io.read_mrec_forcefield(path) is None


class TestSchema:
    def test_molrec_version_is_checked_only_when_present(self) -> None:
        assert mp.io.mrec.schema.MOLREC_VERSION == 1
        # Absent: no version check (molrec contract; 0.14 refused this).
        mp.io.mrec.schema.validate_meta(
            {"record_schema_version": 1, "format_name": "mrec"}
        )
        mp.io.mrec.schema.validate_meta(
            {"molrec_version": mp.io.mrec.schema.MOLREC_VERSION}
        )

    @pytest.mark.parametrize(
        "bad", [None, 0, "1", 1.5, True, mp.io.mrec.schema.MOLREC_VERSION + 1]
    )
    def test_molrec_version_present_must_be_supported_integer(self, bad) -> None:
        with pytest.raises(Exception, match="molrec_version"):
            mp.io.mrec.schema.validate_meta({"molrec_version": bad})
