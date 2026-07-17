"""Reader-hierarchy contract tests.

End state after the LAMMPS/XYZ trajectory readers were sunk to molrs:

- ``HDF5TrajectoryReader`` subclasses ``BaseTrajectoryReader`` (pure, non-mmap)
  and inherits the iteration/random-access API; it must NOT define its own
  ``__iter__`` / ``__getitem__`` / ``__len__``, only ``read_frame`` /
  ``n_frames``.
- ``BaseReader``, ``BaseTrajectoryReader``, and ``TrajectoryWriter`` are
  re-exported from both ``molpy.io`` and ``molpy.io.trajectory``.
"""

import numpy as np
import pytest

import molrs
from molrs import MetaValue

import molpy as mp
from molpy.io.trajectory.base import BaseTrajectoryReader

# Atom counts written into each fixture frame (frame 0 ... frame n-1).
H5_ATOM_COUNTS = [2, 3]


def _make_frame(n_atoms: int, timestep: int) -> molrs.Frame:
    """Build a tiny frame with ``n_atoms`` atoms and a 10 A cubic box."""
    frame = molrs.Frame()
    frame["atoms"] = {
        "id": list(range(1, n_atoms + 1)),
        "type": [1] * n_atoms,
        "x": np.arange(n_atoms, dtype=float),
        "y": np.zeros(n_atoms, dtype=float),
        "z": np.zeros(n_atoms, dtype=float),
    }
    frame.meta = {"timestep": MetaValue("i64", timestep)}
    frame.simbox = mp.Box(np.eye(3) * 10.0)
    return frame


def _build_h5(tmp_path) -> object:
    """Write an HDF5 trajectory with ``H5_ATOM_COUNTS`` frames, return its path."""
    from molpy.io.trajectory.h5 import HDF5TrajectoryWriter

    path = tmp_path / "traj.h5"
    writer = HDF5TrajectoryWriter(str(path))
    for i, n in enumerate(H5_ATOM_COUNTS):
        writer.write_frame(_make_frame(n, timestep=i * 100))
    writer.close()
    return path


# ---------------------------------------------------------------------------
# ac-001: HDF5TrajectoryReader IS-A BaseTrajectoryReader
# ---------------------------------------------------------------------------
def test_h5_reader_is_base_trajectory_reader(tmp_path):
    # ac-001
    pytest.importorskip("h5py")
    from molpy.io.trajectory.h5 import HDF5TrajectoryReader

    path = _build_h5(tmp_path)
    reader = HDF5TrajectoryReader(str(path))

    assert isinstance(reader, BaseTrajectoryReader) is True


# ---------------------------------------------------------------------------
# ac-002: HDF5TrajectoryReader inherits iteration API, defines only read_frame/n_frames
# ---------------------------------------------------------------------------
def test_h5_reader_does_not_define_inherited_iteration_api():
    # ac-002
    pytest.importorskip("h5py")
    from molpy.io.trajectory.h5 import HDF5TrajectoryReader

    own = HDF5TrajectoryReader.__dict__
    assert "__iter__" not in own
    assert "__getitem__" not in own
    assert "__len__" not in own


def test_h5_reader_defines_read_frame_and_n_frames_itself():
    # ac-002
    pytest.importorskip("h5py")
    from molpy.io.trajectory.h5 import HDF5TrajectoryReader

    own = HDF5TrajectoryReader.__dict__
    assert "read_frame" in own
    assert "n_frames" in own


def test_h5_reader_iteration_yields_n_frames_with_expected_atom_counts(tmp_path):
    # ac-002
    pytest.importorskip("h5py")
    from molpy.io.trajectory.h5 import HDF5TrajectoryReader

    path = _build_h5(tmp_path)
    reader = HDF5TrajectoryReader(str(path))

    frames = list(reader)
    assert len(frames) == reader.n_frames
    assert len(frames) == len(H5_ATOM_COUNTS)
    assert frames[0]["atoms"].nrows == H5_ATOM_COUNTS[0]
    assert frames[-1]["atoms"].nrows == H5_ATOM_COUNTS[-1]


# ---------------------------------------------------------------------------
# ac-004: parity - len() and [0]/[-1] expose written data, frames contiguous
# ---------------------------------------------------------------------------
def test_h5_reader_parity_len_and_endpoints(tmp_path):
    # ac-004
    pytest.importorskip("h5py")
    from molpy.io.trajectory.h5 import HDF5TrajectoryReader

    path = _build_h5(tmp_path)
    reader = HDF5TrajectoryReader(str(path))

    assert len(reader) == len(H5_ATOM_COUNTS)
    assert reader[0]["atoms"].nrows == H5_ATOM_COUNTS[0]
    assert reader[-1]["atoms"].nrows == H5_ATOM_COUNTS[-1]


# ---------------------------------------------------------------------------
# ac-005: import surface from molpy.io and molpy.io.trajectory
# ---------------------------------------------------------------------------
def test_import_surface_from_molpy_io():
    # ac-005
    from molpy.io import (  # noqa: F401
        BaseReader,
        BaseTrajectoryReader,
        TrajectoryWriter,
    )


def test_molpy_io_all_contains_hierarchy_names():
    # ac-005
    import molpy.io as io

    for name in (
        "BaseReader",
        "BaseTrajectoryReader",
        "TrajectoryWriter",
    ):
        assert name in io.__all__


def test_import_surface_from_molpy_io_trajectory():
    # ac-005
    from molpy.io.trajectory import (  # noqa: F401
        BaseReader,
        BaseTrajectoryReader,
        TrajectoryWriter,
    )
