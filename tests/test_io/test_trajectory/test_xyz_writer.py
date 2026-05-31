"""XYZ trajectory writer: the atom-count line must be the number of atoms.

Regression guard for a bug where ``write_frame`` wrote ``len(atoms)`` — the
number of *columns* in the atoms Block (a MutableMapping) — instead of the
number of *rows* (atoms). For a frame with element/x/y/z that printed ``4``
regardless of the real atom count, producing malformed (unreadable) XYZ.
"""

from __future__ import annotations

import numpy as np

import molpy
from molpy.io import XYZTrajectoryReader, write_xyz_trajectory


def _frame(n: int) -> molpy.Frame:
    frame = molpy.Frame()
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

    reader = XYZTrajectoryReader(path)
    frames = reader.read_all()
    assert [f["atoms"].nrows for f in frames] == [2, 3]
