"""Reader-hierarchy contract tests.

After trajectory format readers were sunk to molrs, molpy keeps only
writer helpers plus the shared base classes. This file locks the public
re-export surface so callers can still import bases from either package
path.
"""

import molpy as mp
from molpy.io.trajectory.base import BaseTrajectoryReader, TrajectoryWriter
from molpy.io.base import BaseReader


def test_base_classes_reexported_from_io_and_trajectory():
    assert mp.io.BaseReader is BaseReader
    assert mp.io.BaseTrajectoryReader is BaseTrajectoryReader
    assert mp.io.TrajectoryWriter is TrajectoryWriter
    assert mp.io.trajectory.BaseReader is BaseReader
    assert mp.io.trajectory.BaseTrajectoryReader is BaseTrajectoryReader
    assert mp.io.trajectory.TrajectoryWriter is TrajectoryWriter


def test_hdf5_trajectory_surface_is_gone():
    assert not hasattr(mp.io, "HDF5TrajectoryReader")
    assert not hasattr(mp.io, "HDF5TrajectoryWriter")
    assert not hasattr(mp.io, "read_h5")
    assert not hasattr(mp.io, "write_h5")
    assert not hasattr(mp.io, "read_h5_trajectory")
    assert not hasattr(mp.io, "write_h5_trajectory")
