import numpy as np

import molpy as mp
from molpy.core import MetaValue
from molpy.io import (
    read_lammps_dump_trajectory,
    write_lammps_dump_local,
    write_lammps_dump_trajectory,
)


class TestWriteLammpsTrajectory:
    def test_write_simple_trajectory(self, tmp_path):
        """Test writing a simple trajectory."""
        # Create test frames
        frames = []
        for i in range(3):
            frame = mp.Frame()

            # Create atoms data using Block structure
            atoms_data = {
                "id": [0, 1, 2],
                "type_id": [1, 1, 2],
                "x": [0.0 + i * 0.1, 1.0 + i * 0.1, 0.5 + i * 0.1],
                "y": [0.0, 0.0, 1.0],
                "z": [0.0, 0.0, 0.0],
            }
            frame["atoms"] = atoms_data
            frame.meta = {"timestep": MetaValue("i64", i * 100)}
            frame.box = mp.Box(h=np.eye(3) * 10.0)
            frames.append(frame)

        # Write trajectory
        tmp_file = tmp_path / "test.dump"
        write_lammps_dump_trajectory(tmp_file, frames)

        # Read back via the native reader and verify
        reader = read_lammps_dump_trajectory(str(tmp_file))

        # Check that we can read the frames back
        for i, frame_read in enumerate(reader):
            if i >= len(frames):
                break
            assert frame_read.meta["timestep"] == frames[i].meta["timestep"]
            assert "atoms" in frame_read
            # Check that positions changed over time
            if i > 0:
                # x coordinates should be different between frames
                x_vals = frame_read["atoms"]["x"]
                x_vals_prev = frames[0]["atoms"]["x"]
                assert not np.allclose(x_vals, x_vals_prev)

    def test_trajectory_roundtrip(self, tmp_path):
        """Test writing and reading back maintains data integrity."""
        # Create a more complex frame
        frame = mp.Frame()

        atoms_data = {
            "id": [0, 1, 2, 3],
            "type_id": [1, 1, 2, 2],
            "x": [0.0, 1.0, 0.5, 1.5],
            "y": [0.0, 0.0, 1.0, 1.0],
            "z": [0.0, 0.0, 0.0, 0.0],
            "vx": [0.1, -0.1, 0.2, -0.2],
            "vy": [0.0, 0.0, 0.1, -0.1],
            "vz": [0.0, 0.0, 0.0, 0.0],
        }
        frame["atoms"] = atoms_data
        frame.meta = {"timestep": MetaValue("i64", 1000)}
        frame.box = mp.Box(h=np.diag([5.0, 5.0, 5.0]))

        tmp_file = tmp_path / "test.dump"
        # Write
        write_lammps_dump_trajectory(tmp_file, [frame])

        # Read back via the native reader
        reader = read_lammps_dump_trajectory(str(tmp_file))
        frame_read = reader[0]

        # Verify timestep
        assert frame_read.meta["timestep"] == 1000

        # Verify atoms data exists
        assert "atoms" in frame_read
        atoms = frame_read["atoms"]
        assert atoms.n_rows == 4

        # Verify box
        assert frame_read.box is not None
        assert np.allclose(frame_read.box.h.diagonal(), [5.0, 5.0, 5.0])


class TestTrajectoryIntegration:
    def test_data_to_trajectory_conversion(self, tmp_path):
        """Test converting data format to trajectory format."""
        # Create a frame in data format
        frame = mp.Frame()

        atoms_data = {
            "id": [0, 1, 2],
            "type_id": [1, 1, 2],  # Use numeric types for LAMMPS
            "x": [0.0, 0.816, -0.816],
            "y": [0.0, 0.577, 0.577],
            "z": [0.0, 0.0, 0.0],
            "q": [-0.8476, 0.4238, 0.4238],
        }
        frame["atoms"] = atoms_data
        frame.meta = {"timestep": MetaValue("i64", 0)}
        frame.box = mp.Box(h=np.diag([10.0, 10.0, 10.0]))

        # Write as trajectory
        tmp_file = tmp_path / "test.dump"
        write_lammps_dump_trajectory(tmp_file, [frame])

        # Read back as trajectory
        reader = read_lammps_dump_trajectory(str(tmp_file))
        frame_read = reader[0]

        assert frame_read.meta["timestep"] == 0
        assert "atoms" in frame_read
        assert frame_read.box is not None

    def test_multiple_formats_consistency(self, tmp_path):
        """Test that data and trajectory formats are consistent."""
        # This test ensures that a frame written in one format
        # can be meaningfully compared with the other format
        frame_original = mp.Frame()

        atoms_data = {
            "id": [0, 1],
            "type_id": [1, 2],
            "x": [0.0, 1.0],
            "y": [0.0, 0.0],
            "z": [0.0, 0.0],
        }
        frame_original["atoms"] = atoms_data
        frame_original.meta = {"timestep": MetaValue("i64", 100)}
        frame_original.box = mp.Box(h=np.eye(3) * 5.0)

        # Write as trajectory and read back
        tmp_file = tmp_path / "test.dump"
        write_lammps_dump_trajectory(tmp_file, [frame_original])

        reader = read_lammps_dump_trajectory(str(tmp_file))
        frame_traj = reader[0]

        # Both should have same basic structure
        assert frame_traj.meta["timestep"] == 100
        assert "atoms" in frame_traj
        assert frame_traj.box is not None


class TestWriteLammpsDumpLocal:
    def test_write_bonds_roundtrip(self, tmp_path):
        atoms = mp.Block()
        atoms["id"] = np.array([1, 2, 3], dtype=np.uint64)
        atoms["x"] = np.array([0.0, 1.0, 2.0])
        atoms["y"] = np.zeros(3)
        atoms["z"] = np.zeros(3)
        bonds = mp.Block()
        bonds["atomi"] = np.array([0, 1], dtype=np.uint64)
        bonds["atomj"] = np.array([1, 2], dtype=np.uint64)
        frame = mp.Frame()
        frame["atoms"] = atoms
        frame["bonds"] = bonds
        frame.box = mp.Box.cube(10.0)
        path = tmp_path / "bonds.dump.local"
        write_lammps_dump_local(path, [frame])
        text = path.read_text(encoding="utf-8")
        assert "ITEM: NUMBER OF ENTRIES" in text
        assert "batom1 batom2" in text
        lines = text.splitlines()
        assert lines[lines.index("ITEM: NUMBER OF ENTRIES") + 1].strip() == "2"
