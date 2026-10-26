"""Unit tests for ``*.mrec`` scientific-record I/O on :mod:`molpy.io`."""

from __future__ import annotations

from pathlib import Path

import numpy as np

import molpy as mp

_N_ATOMS = 3
_ATOM_X = (0.0, 1.0, 0.5)
_ATOM_Y = (0.25, 0.0, 2.0)
_ATOM_Z = (0.0, 4.0, 0.125)


def _coords_frame() -> mp.Frame:
    atoms = mp.Block()
    atoms["x"] = np.array(_ATOM_X, dtype=np.float64)
    atoms["y"] = np.array(_ATOM_Y, dtype=np.float64)
    atoms["z"] = np.array(_ATOM_Z, dtype=np.float64)
    frame = mp.Frame()
    frame["atoms"] = atoms
    return frame


def _assert_coords(frame: mp.Frame) -> None:
    atoms = frame["atoms"]
    assert atoms.n_rows == _N_ATOMS
    np.testing.assert_array_equal(
        np.asarray(atoms["x"]), np.array(_ATOM_X, dtype=np.float64)
    )
    np.testing.assert_array_equal(
        np.asarray(atoms["y"]), np.array(_ATOM_Y, dtype=np.float64)
    )
    np.testing.assert_array_equal(
        np.asarray(atoms["z"]), np.array(_ATOM_Z, dtype=np.float64)
    )


class TestFrameSequence:
    def test_read_frame(self, tmp_path: Path) -> None:
        path = tmp_path / "traj.mrec"
        mp.io.write_mrec(path, trajectory=mp.Trajectory([_coords_frame()]))
        reader = mp.io.mrec.MrecReader(path)
        _assert_coords(reader[0])


class TestWrite:
    def test_round_trips_coordinates(self, tmp_path: Path) -> None:
        path = tmp_path / "snapshot.mrec"
        mp.io.write_mrec(path, frame=_coords_frame())
        _assert_coords(mp.io.mrec.MrecFile(path).frame())
        assert set(mp.io.mrec.MrecFile(path).sections()) == frozenset({"meta", "frame"})
        meta = mp.io.mrec.MrecFile(path).meta()
        assert "format_name" not in meta


class TestWriteSystem:
    def test_round_trips_coordinates(self, tmp_path: Path) -> None:
        path = tmp_path / "system.mrec"
        mp.io.write_mrec(path, system=_coords_frame())
        _assert_coords(mp.io.mrec.MrecFile(path).system())
        assert "frame" not in set(mp.io.mrec.MrecFile(path).sections())


class TestWriteTrajectory:
    def test_round_trips_coordinates(self, tmp_path: Path) -> None:
        path = tmp_path / "traj.mrec"
        mp.io.write_mrec(path, trajectory=mp.Trajectory([_coords_frame()]))
        loaded = mp.io.mrec.MrecReader(path).to_trajectory()
        assert len(loaded) == 1
        _assert_coords(loaded[0])


def _water_forcefield() -> mp.ff.forcefield.ForceField:
    ff = mp.ff.forcefield.ForceField(name="water", units="real")
    atoms = ff.def_style("atom", "full")
    ow = atoms.def_type("OW", mass=15.999, charge=-0.834)
    hw = atoms.def_type("HW", mass=1.008, charge=0.417)
    ff.def_style("bond", "harmonic").def_type("OW-HW", ow, hw, k=450.0, r0=0.9572)
    pair = ff.def_style("pair", "lj/cut", {"cutoff": 10.0})
    pair.def_type("OW", ow, epsilon=0.1521, sigma=3.1507)
    pair.def_type("HW", hw, epsilon=0.0, sigma=0.0)
    return ff


def _ff_rows(ff: mp.ff.forcefield.ForceField) -> dict:
    return {
        (style.category, style.name): {
            t.name: dict(t.params) for t in style.get_types()
        }
        for style in ff.get_styles(mp.ff.forcefield.Style)
    }


class TestForceFieldSection:
    def test_rides_along_a_snapshot(self, tmp_path: Path) -> None:
        path = tmp_path / "snapshot.mrec"
        ff = _water_forcefield()
        mp.io.write_mrec(path, frame=_coords_frame(), forcefield=ff)
        assert "forcefield" in set(mp.io.mrec.MrecFile(path).sections())
        section = mp.io.mrec.MrecFile(path).forcefield()
        assert isinstance(section, mp.io.mrec.ForceFieldSection)
        loaded = section.to_forcefield()
        assert loaded.units == "real"
        assert _ff_rows(loaded) == _ff_rows(ff)
        _assert_coords(mp.io.mrec.MrecFile(path).frame())

    def test_standalone_package(self, tmp_path: Path) -> None:
        path = tmp_path / "ff.mrec"
        ff = _water_forcefield()
        mp.io.write_mrec(path, forcefield=ff)
        loaded = mp.io.mrec.MrecFile(path).forcefield().to_forcefield()
        assert _ff_rows(loaded) == _ff_rows(ff)

    def test_absent_section_reads_none(self, tmp_path: Path) -> None:
        path = tmp_path / "snapshot.mrec"
        mp.io.write_mrec(path, frame=_coords_frame())
        assert mp.io.mrec.MrecFile(path).forcefield() is None
