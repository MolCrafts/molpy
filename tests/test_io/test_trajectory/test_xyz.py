"""``mp.io.write_xyz_trajectory``: one count line per frame, readable back."""

from __future__ import annotations

import numpy as np

import molpy as mp
from molpy.io import read_xyz_trajectory, write_xyz_trajectory


def _frame(n: int) -> mp.Frame:
    frame = mp.Frame()
    frame["atoms"] = {
        "element": np.array(["C"] * n),
        "x": np.arange(n, dtype=float),
        "y": np.zeros(n),
        "z": np.zeros(n),
    }
    return frame


def test_atom_count_line_is_row_count(tmp_path):
    path = tmp_path / "traj.xyz"
    write_xyz_trajectory(path, [_frame(2), _frame(3)])

    lines = path.read_text().splitlines()
    # Frame 0: count line, comment, 2 atoms.
    assert lines[0] == "2"
    # Frame 1 starts after 2 atom rows: index 0 + 1 (comment) + 2 (atoms) = 4.
    assert lines[4] == "3"


def test_roundtrips_through_reader(tmp_path):
    path = tmp_path / "traj.xyz"
    write_xyz_trajectory(path, [_frame(2), _frame(3)])

    reader = read_xyz_trajectory(path)
    frames = list(reader)
    assert [f["atoms"].n_rows for f in frames] == [2, 3]
