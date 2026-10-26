"""Shared pytest fixtures for the molrs Python bindings.

Binding tests are **self-contained**: they never require third-party scientific
software (RDKit, AmberTools, freud, …) and never require the optional
chemfiles-derived ``tests-data/`` corpus. Format smoke tests build tiny
frames in-process and round-trip them through molrs writers/readers.
"""

from __future__ import annotations

from pathlib import Path

import molpy as mp
import numpy as np
import pytest


def cube_box(edge: float = 10.0) -> mp.core.Box:
    """A periodic cubic box of edge ``edge`` at the origin.

    Plain function (not a fixture) so helpers and parametrizations can call it.
    """
    return mp.core.Box(mp.core.Cell.lengths([edge, edge, edge]))


@pytest.fixture
def cubic_box():
    """A 10x10x10 cubic box."""
    return cube_box(10.0)


@pytest.fixture
def ortho_box():
    """A 5x10x15 orthorhombic box."""
    return mp.core.Box(
        mp.core.Cell.lengths(np.array([5.0, 10.0, 15.0], dtype=np.float64))
    )


@pytest.fixture
def sample_points():
    """5 points inside a 10x10x10 box."""
    return np.array(
        [
            [1.0, 2.0, 3.0],
            [5.0, 5.0, 5.0],
            [9.0, 8.0, 7.0],
            [0.1, 0.1, 0.1],
            [9.9, 9.9, 9.9],
        ],
        dtype=np.float64,
    )


def _water_frame(*, for_lammps: bool = False) -> mp.core.Frame:
    """Minimal 3-atom frame with box.

    Keep columns minimal so writers (esp. extended XYZ) do not emit fields the
    matching reader cannot parse. LAMMPS data needs ``type_id`` and ``mass``
    (+ optional charge).
    """
    f = mp.core.Frame()
    b = mp.core.Block()
    b.insert("element", ["O", "H", "H"])
    b.insert("x", np.array([0.0, 0.96, -0.24], dtype=np.float64))
    b.insert("y", np.array([0.0, 0.0, 0.93], dtype=np.float64))
    b.insert("z", np.array([0.0, 0.0, 0.0], dtype=np.float64))
    b.insert("name", ["O", "H1", "H2"])
    b.insert("res_name", ["SOL", "SOL", "SOL"])
    b.insert("res_id", np.array([1, 1, 1], dtype=np.uint32))
    b.insert("id", np.array([1, 2, 3], dtype=np.uint32))
    if for_lammps:
        b.insert("type_id", np.array([1, 2, 2], dtype=np.uint32))
        b.insert("charge", np.array([-0.834, 0.417, 0.417], dtype=np.float64))
        b.insert("mol_id", np.array([1, 1, 1], dtype=np.uint32))
        # LAMMPS data writes a Masses section per atom type; there is no
        # default mass.
        b.insert("mass", np.array([15.999, 1.008, 1.008], dtype=np.float64))
    f["atoms"] = b
    f.box = cube_box(10.0)
    return f


def _write_frames(writer, frames) -> None:
    """Write ``frames`` through a multi-frame ``<Fmt>Writer`` and close it."""
    for frame in frames:
        writer.write(frame)
    writer.close()


@pytest.fixture
def water_frame() -> mp.core.Frame:
    """Fresh water-like frame (callers may mutate)."""
    return _water_frame()


@pytest.fixture
def water_xyz(tmp_path: Path) -> Path:
    path = tmp_path / "water.xyz"
    mp.io.write_xyz(path, _water_frame())
    return path


@pytest.fixture
def water_pdb(tmp_path: Path) -> Path:
    path = tmp_path / "water.pdb"
    mp.io.write_pdb(path, _water_frame())
    return path


@pytest.fixture
def water_gro(tmp_path: Path) -> Path:
    path = tmp_path / "water.gro"
    mp.io.write_gro(path, _water_frame())
    return path


@pytest.fixture
def water_dcd(tmp_path: Path) -> Path:
    path = tmp_path / "water.dcd"
    frame = _water_frame()
    _write_frames(mp.io.dcd.DcdWriter(path), [frame, frame])
    return path


@pytest.fixture
def water_trr(tmp_path: Path) -> Path:
    path = tmp_path / "water.trr"
    frame = _water_frame()
    _write_frames(mp.io.trr.TrrWriter(path), [frame, frame])
    return path


@pytest.fixture
def water_xtc(tmp_path: Path) -> Path:
    path = tmp_path / "water.xtc"
    frame = _water_frame()
    _write_frames(mp.io.xtc.XtcWriter(path), [frame, frame])
    return path


@pytest.fixture
def water_lammpstrj(tmp_path: Path) -> Path:
    path = tmp_path / "water.lammpstrj"
    frame = _water_frame(for_lammps=True)
    _write_frames(mp.io.lammps.LammpsDumpWriter(path), [frame, frame])
    return path


@pytest.fixture
def water_lammps_data(tmp_path: Path) -> Path:
    path = tmp_path / "water.data"
    mp.io.write_lammps_data(path, _water_frame(for_lammps=True))
    return path


def make_frame(pts, box_len=10.0):
    """Wrap an (N, 3) float64 array into a Frame with x/y/z columns and a cubic box.

    Shared by test_compute.py and test_neighborlist.py — one Frame-construction
    contract, one home. Plain function (not a fixture) so helpers can call it.
    """
    frame = mp.core.Frame()
    block = mp.core.Block()
    block.insert("x", np.ascontiguousarray(pts[:, 0], dtype=np.float64))
    block.insert("y", np.ascontiguousarray(pts[:, 1], dtype=np.float64))
    block.insert("z", np.ascontiguousarray(pts[:, 2], dtype=np.float64))
    frame["atoms"] = block
    frame.box = cube_box(box_len)
    return frame


def octahedron_frame(cx=5.0, cy=5.0, cz=5.0, box_len=20.0):
    """Centre particle + ±x/y/z neighbors at unit distance; returns (frame, pts)."""
    pts = np.array(
        [
            [cx, cy, cz],
            [cx + 1, cy, cz],
            [cx - 1, cy, cz],
            [cx, cy + 1, cz],
            [cx, cy - 1, cz],
            [cx, cy, cz + 1],
            [cx, cy, cz - 1],
        ],
        dtype=np.float64,
    )
    return make_frame(pts, box_len=box_len), pts


def capture(writer, *args, **kwargs):
    """Capture a production writer's bytearray sink for text assertions."""
    buffer = bytearray()
    writer(buffer, *args, **kwargs)
    return buffer.decode("utf-8")
