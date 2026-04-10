"""Tests for sequential iteration prefetching."""

import numpy as np

import molpy as mp
from molpy.io.trajectory.lammps import LammpsTrajectoryReader, LammpsTrajectoryWriter
from molpy.io.trajectory.xyz import XYZTrajectoryReader


def _write_lammps_traj(path, n_frames=10):
    writer = LammpsTrajectoryWriter(str(path))
    for i in range(n_frames):
        frame = mp.Frame()
        frame["atoms"] = {
            "id": [0, 1, 2],
            "type": [1, 1, 2],
            "x": [float(i), float(i) + 1, float(i) + 0.5],
            "y": [0.0, 0.0, 1.0],
            "z": [0.0, 0.0, 0.0],
        }
        frame.metadata["timestep"] = i * 100
        frame.box = mp.Box(np.eye(3) * 10.0)
        writer.write_frame(frame)
    writer.close()


def _write_xyz_traj(path, n_frames=10):
    with open(path, "w") as f:
        for i in range(n_frames):
            f.write("3\n")
            f.write(f"Step={i}\n")
            f.write(f"C {float(i)} 0.0 0.0\n")
            f.write(f"O {float(i) + 1} 0.0 0.0\n")
            f.write(f"H {float(i) + 0.5} 1.0 0.0\n")


class TestLammpsPrefetch:
    def test_iteration_returns_all_frames(self, tmp_path):
        """Test that iterator with prefetch returns all frames correctly."""
        traj = tmp_path / "test.dump"
        _write_lammps_traj(traj, n_frames=10)

        reader = LammpsTrajectoryReader(str(traj))
        frames = list(reader)
        assert len(frames) == 10
        for i, frame in enumerate(frames):
            assert frame.metadata["timestep"] == i * 100
        reader.close()

    def test_iteration_matches_direct_read(self, tmp_path):
        """Test that iterated frames match directly-read frames."""
        traj = tmp_path / "test.dump"
        _write_lammps_traj(traj, n_frames=5)

        reader = LammpsTrajectoryReader(str(traj))
        iterated = list(reader)
        direct = [reader.read_frame(i) for i in range(5)]

        for it_frame, dir_frame in zip(iterated, direct):
            assert it_frame.metadata["timestep"] == dir_frame.metadata["timestep"]
            np.testing.assert_array_equal(
                it_frame["atoms"]["x"], dir_frame["atoms"]["x"]
            )
        reader.close()

    def test_partial_iteration(self, tmp_path):
        """Test breaking out of iteration early."""
        traj = tmp_path / "test.dump"
        _write_lammps_traj(traj, n_frames=20)

        reader = LammpsTrajectoryReader(str(traj))
        frames = []
        for frame in reader:
            frames.append(frame)
            if len(frames) >= 5:
                break
        assert len(frames) == 5
        assert frames[0].metadata["timestep"] == 0
        assert frames[4].metadata["timestep"] == 400
        reader.close()


class TestXYZPrefetch:
    def test_iteration_returns_all_frames(self, tmp_path):
        """Test XYZ reader iteration with prefetch."""
        traj = tmp_path / "test.xyz"
        _write_xyz_traj(traj, n_frames=8)

        reader = XYZTrajectoryReader(str(traj))
        frames = list(reader)
        assert len(frames) == 8
        for i, frame in enumerate(frames):
            assert "atoms" in frame
            assert frame["atoms"].nrows == 3
        reader.close()

    def test_single_frame_iteration(self, tmp_path):
        """Test iteration with just one frame."""
        traj = tmp_path / "test.xyz"
        _write_xyz_traj(traj, n_frames=1)

        reader = XYZTrajectoryReader(str(traj))
        frames = list(reader)
        assert len(frames) == 1
        reader.close()
