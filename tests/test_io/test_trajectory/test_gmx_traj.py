"""GROMACS TRR/XTC trajectory read+write via the molpy facade.

The TRR/XTC readers/writers are thin delegations to the native molrs backend.
These tests build frames, round-trip them through ``molpy.io.write_trr`` /
``write_xtc`` and the lazy ``read_*_trajectory`` readers, and check that the
molpy surface (lazy reader object, n_frames, indexing) behaves correctly.
"""

from __future__ import annotations

import numpy as np

import molrs

import molpy
from molpy.io import (
    read_trr_trajectory,
    read_xtc_trajectory,
    write_trr,
    write_xtc,
)


def _frame(n: int, shift: float = 0.0) -> molrs.Frame:
    frame = molrs.Frame()
    frame["atoms"] = {
        "id": np.arange(1, n + 1, dtype=np.int32),
        "x": np.linspace(0.0, 1.0, n) + shift,
        "y": np.linspace(1.0, 2.0, n) + shift,
        "z": np.linspace(2.0, 3.0, n) + shift,
    }
    return frame


def test_trr_write_then_read(tmp_path):
    frames = [_frame(12, 0.0), _frame(12, 0.1)]
    path = tmp_path / "traj.trr"
    write_trr(path, frames)

    reader = read_trr_trajectory(path)
    assert reader.n_frames == len(reader) == 2
    # TRR single precision: coordinates preserved to f32 tolerance.
    got = reader.read_frame(0)["atoms"].view("x")
    assert np.allclose(got, frames[0]["atoms"].view("x"), atol=1e-5)
    # Negative indexing reaches the last frame.
    assert reader.read_frame(-1)["atoms"].nrows == 12


def test_xtc_write_then_read(tmp_path):
    frames = [_frame(12, 0.0), _frame(12, 0.05)]
    path = tmp_path / "traj.xtc"
    write_xtc(path, frames)

    reader = read_xtc_trajectory(path)
    assert reader.n_frames == 2
    # XTC is lossy at 1/precision (default 1000 → 1e-3 nm).
    got = reader.read_frame(1)["atoms"].view("x")
    assert np.allclose(got, frames[1]["atoms"].view("x"), atol=2e-3)


def test_trr_reader_slicing_and_iteration(tmp_path):
    frames = [_frame(6), _frame(6), _frame(6)]
    path = tmp_path / "traj.trr"
    write_trr(path, frames)

    reader = read_trr_trajectory(path)
    assert len(reader[:]) == 3
    assert sum(1 for _ in reader) == 3
    assert len(reader.read_all()) == 3
