"""Reader-hierarchy contract tests (spec frame-reader-hierarchy-02-reparent).

End-state spec after the 02 reparent:

- ``HDF5TrajectoryReader`` subclasses ``BaseTrajectoryReader`` (pure, non-mmap)
  and inherits the iteration/random-access API; it must NOT define its own
  ``__iter__`` / ``__getitem__`` / ``__len__``, only ``read_frame`` /
  ``n_frames``.
- ``XYZTrajectoryReader`` and ``LammpsTrajectoryReader`` subclass
  ``MmapTrajectoryReader``.
- ``BaseReader``, ``BaseTrajectoryReader``, ``MmapTrajectoryReader``, and
  ``FrameLocation`` are re-exported from both ``molpy.io`` and
  ``molpy.io.trajectory``.

These tests encode that end state and are expected to FAIL until 02 lands.
"""

import numpy as np
import pytest

import molpy as mp
from molpy.io.trajectory.base import BaseTrajectoryReader, MmapTrajectoryReader
from molpy.io.trajectory.lammps import LammpsTrajectoryReader, LammpsTrajectoryWriter
from molpy.io.trajectory.xyz import XYZTrajectoryReader

# Atom counts written into each fixture frame (frame 0 ... frame n-1).
H5_ATOM_COUNTS = [2, 3]
XYZ_ELEMENTS_PER_FRAME = [["C", "O"], ["C", "O", "H"]]
LAMMPS_ATOM_COUNTS = [2, 3]


def _make_frame(n_atoms: int, timestep: int) -> mp.Frame:
    """Build a tiny frame with ``n_atoms`` atoms and a 10 A cubic box."""
    frame = mp.Frame()
    frame["atoms"] = {
        "id": list(range(1, n_atoms + 1)),
        "type": [1] * n_atoms,
        "x": np.arange(n_atoms, dtype=float),
        "y": np.zeros(n_atoms, dtype=float),
        "z": np.zeros(n_atoms, dtype=float),
    }
    frame.metadata["timestep"] = timestep
    frame.box = mp.Box(np.eye(3) * 10.0)
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


def _build_xyz(tmp_path) -> object:
    """Write a canonical multi-frame XYZ file from ``XYZ_ELEMENTS_PER_FRAME``.

    Written as plain XYZ text rather than via ``XYZTrajectoryWriter`` so the
    fixture is format-correct and deterministic regardless of writer-side
    header bugs; the reader contract under test is on the read path.
    """
    path = tmp_path / "traj.xyz"
    lines: list[str] = []
    for i, elements in enumerate(XYZ_ELEMENTS_PER_FRAME):
        lines.append(str(len(elements)))
        lines.append(f"Step={i * 100}")
        for j, element in enumerate(elements):
            lines.append(f"{element} {float(j)} 0.0 0.0")
    path.write_text("\n".join(lines) + "\n")
    return path


def _build_lammps(tmp_path) -> object:
    """Write a LAMMPS trajectory with ``LAMMPS_ATOM_COUNTS`` frames, return path."""
    path = tmp_path / "traj.dump"
    writer = LammpsTrajectoryWriter(str(path))
    for i, n in enumerate(LAMMPS_ATOM_COUNTS):
        writer.write_frame(_make_frame(n, timestep=i * 100))
    writer.close()
    return path


# ---------------------------------------------------------------------------
# ac-001: HDF5TrajectoryReader IS-A BaseTrajectoryReader, NOT a MmapTrajectoryReader
# ---------------------------------------------------------------------------
def test_h5_reader_is_base_trajectory_reader_not_mmap(tmp_path):
    # ac-001
    pytest.importorskip("h5py")
    from molpy.io.trajectory.h5 import HDF5TrajectoryReader

    path = _build_h5(tmp_path)
    reader = HDF5TrajectoryReader(str(path))

    assert isinstance(reader, BaseTrajectoryReader) is True
    assert isinstance(reader, MmapTrajectoryReader) is False


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
# ac-003: XYZ and LAMMPS readers ARE MmapTrajectoryReader
# ---------------------------------------------------------------------------
def test_xyz_reader_is_mmap_trajectory_reader(tmp_path):
    # ac-003
    path = _build_xyz(tmp_path)
    assert isinstance(XYZTrajectoryReader(str(path)), MmapTrajectoryReader) is True


def test_lammps_reader_is_mmap_trajectory_reader(tmp_path):
    # ac-003
    path = _build_lammps(tmp_path)
    assert isinstance(LammpsTrajectoryReader(str(path)), MmapTrajectoryReader) is True


# ---------------------------------------------------------------------------
# ac-004: parity - len() and [0]/[-1] expose written data, frames contiguous
# ---------------------------------------------------------------------------
def test_xyz_reader_parity_len_and_endpoints(tmp_path):
    # ac-004
    path = _build_xyz(tmp_path)
    reader = XYZTrajectoryReader(str(path))

    assert len(reader) == len(XYZ_ELEMENTS_PER_FRAME)
    assert list(reader[0]["atoms"]["element"]) == XYZ_ELEMENTS_PER_FRAME[0]
    assert list(reader[-1]["atoms"]["element"]) == XYZ_ELEMENTS_PER_FRAME[-1]


def test_lammps_reader_parity_len_and_endpoints(tmp_path):
    # ac-004
    path = _build_lammps(tmp_path)
    reader = LammpsTrajectoryReader(str(path))

    assert len(reader) == len(LAMMPS_ATOM_COUNTS)
    assert reader[0]["atoms"].nrows == LAMMPS_ATOM_COUNTS[0]
    assert reader[-1]["atoms"].nrows == LAMMPS_ATOM_COUNTS[-1]


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
        FrameLocation,
        MmapTrajectoryReader,
    )


def test_molpy_io_all_contains_hierarchy_names():
    # ac-005
    import molpy.io as io

    for name in (
        "BaseReader",
        "BaseTrajectoryReader",
        "MmapTrajectoryReader",
        "FrameLocation",
    ):
        assert name in io.__all__


def test_import_surface_from_molpy_io_trajectory():
    # ac-005
    from molpy.io.trajectory import (  # noqa: F401
        BaseReader,
        BaseTrajectoryReader,
        FrameLocation,
        MmapTrajectoryReader,
    )
