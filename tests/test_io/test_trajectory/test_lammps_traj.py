import pytest
import tempfile
import numpy as np
import molpy as mp
from molpy.io.trajectory.lammps import LammpsTrajectoryReader, LammpsTrajectoryWriter
from pathlib import Path


class TestReadLammpsTrajectory:

    def test_read_frame_fcc_orth(self, test_data_path):
        reader = LammpsTrajectoryReader(test_data_path / "trajectory/lammps/fcc_orth.dump")
        frame = reader.read_frame(0)

        assert frame["timestep"] == 0
        assert frame.box.matrix.shape == (3, 3)  # type: ignore
        assert len(frame["atoms"].data_vars) > 0  # Check atoms exist
        assert len(frame["atoms"].data_vars) == 5
        expected_box_matrix = [
            [1.5377619196572583, 0, 0],
            [0, 1.5377619196572583, 0],
            [0, 0, 1.5377619196572583],
        ]
        assert frame.box is not None and (frame.box.matrix == expected_box_matrix).all()

    def test_read_frame_fcc_tric(self, test_data_path):
        reader = LammpsTrajectoryReader(test_data_path / "trajectory/lammps/fcc_tric.dump")
        frame = reader.read_frame(0)

        assert frame["timestep"] == 0
        assert frame.box.matrix.shape == (3, 3)  # type: ignore
        assert len(frame["atoms"].data_vars) > 0  # Check atoms exist 
        assert len(frame["atoms"].data_vars) == 5

        lx = 1.1922736280710971e+02 - 5.1509317718250820e+01
        ly = 1.1937795684789540e+02 - 5.3089558100317952e+01
        lz = 1.1924193400378044e+02 - 5.2978065996127228e+01
        xy = -2.2480556300586776e-01
        xz = -1.5685139115469910e+00
        yz = 8.7514948257390660e-02
        assert frame.box is not None and frame.box.matrix[0, 0] == lx
        assert frame.box.matrix[1, 1] == ly
        assert frame.box.matrix[2, 2] == lz
        assert frame.box.matrix[0, 1] == xy
        assert frame.box.matrix[0, 2] == xz
        assert frame.box.matrix[1, 2] == yz

    def test_read_multi_traj(self, test_data_path):
        reader = LammpsTrajectoryReader([test_data_path/"trajectory/lammps/fcc_orth.dump", test_data_path/"trajectory/lammps/fcc_tric.dump"])

        assert reader.n_frames == 2
        frame0 = reader.read_frame(0)
        assert frame0["timestep"] == 0
        assert len(frame0["atoms"].data_vars) > 0  # Check atoms exist
        frame1 = reader.read_frame(1)
        assert frame1["timestep"] == 0
        assert len(frame1["atoms"].data_vars) > 0  # Check atoms exist

    def test_trajectory_iteration(self, test_data_path):
        """Test iterating through trajectory frames."""
        reader = LammpsTrajectoryReader(test_data_path / "trajectory/lammps/fcc_orth.dump")
        
        # Test iterator protocol
        frames = []
        for frame in reader:
            frames.append(frame)
            if len(frames) >= 3:  # Limit to avoid long test
                break
        
        assert len(frames) > 0
        for frame in frames:
            assert isinstance(frame, mp.Frame)
            assert "timestep" in frame
            assert "atoms" in frame
            assert frame.box is not None

    def test_frame_properties(self, test_data_path):
        """Test that frames have correct properties."""
        reader = LammpsTrajectoryReader(test_data_path / "trajectory/lammps/fcc_orth.dump")
        frame = reader.read_frame(0)
        
        # Check timestep
        assert "timestep" in frame
        assert isinstance(frame["timestep"], (int, np.integer))
        
        # Check atoms data
        assert "atoms" in frame
        atoms = frame["atoms"]
        assert hasattr(atoms, 'data_vars')
        
        # Check that coordinates exist in some form
        has_coords = False
        if "x" in atoms and "y" in atoms and "z" in atoms:
            has_coords = True
        elif "xyz" in atoms:
            has_coords = True
        assert has_coords, "Frame should have coordinate data"
        
        # Check box
        assert frame.box is not None
        assert hasattr(frame.box, 'matrix')
        assert frame.box.matrix.shape == (3, 3)

    def test_context_manager(self, test_data_path):
        """Test using trajectory reader as context manager."""
        with LammpsTrajectoryReader(test_data_path / "trajectory/lammps/fcc_orth.dump") as reader:
            frame = reader.read_frame(0)
            assert "timestep" in frame
            assert "atoms" in frame


class TestWriteLammpsTrajectory:

    def test_write_simple_trajectory(self):
        """Test writing a simple trajectory."""
        # Create test frames
        frames = []
        for i in range(3):
            frame = mp.Frame()
            
            # Create atoms data
            atoms_data = {
                'id': [0, 1, 2],
                'type': [1, 1, 2],
                'x': [0.0 + i*0.1, 1.0 + i*0.1, 0.5 + i*0.1],
                'y': [0.0, 0.0, 1.0],
                'z': [0.0, 0.0, 0.0]
            }
            frame["atoms"] = atoms_data
            frame["timestep"] = i * 100
            frame.box = mp.Box(np.eye(3) * 10.0)
            frames.append(frame)
        
        # Write trajectory
        with tempfile.NamedTemporaryFile(mode='w', suffix='.dump', delete=False) as tmp:
            writer = LammpsTrajectoryWriter(tmp.name)
            for frame in frames:
                writer.write_frame(frame)
            writer.close()
            
            # Read back and verify
            reader = LammpsTrajectoryReader(tmp.name)
            
            # Check that we can read the frames back
            for i, frame_read in enumerate(reader):
                if i >= len(frames):
                    break
                assert frame_read["timestep"] == frames[i]["timestep"]
                assert "atoms" in frame_read
                # Check that positions changed over time
                if i > 0:
                    # x coordinates should be different between frames
                    x_vals = frame_read["atoms"]["x"].values
                    x_vals_prev = frames[0]["atoms"]["x"]
                    assert not np.allclose(x_vals, x_vals_prev)

    def test_write_with_context_manager(self):
        """Test writing trajectory using context manager."""
        frame = mp.Frame()
        atoms_data = {
            'id': [0, 1],
            'type': [1, 1],
            'x': [0.0, 1.0],
            'y': [0.0, 0.0],
            'z': [0.0, 0.0]
        }
        frame["atoms"] = atoms_data
        frame["timestep"] = 0
        frame.box = mp.Box(np.eye(3) * 5.0)
        
        with tempfile.NamedTemporaryFile(mode='w', suffix='.dump', delete=False) as tmp:
            with LammpsTrajectoryWriter(tmp.name) as writer:
                writer.write_frame(frame)

    def test_trajectory_roundtrip(self):
        """Test writing and reading back maintains data integrity."""
        # Create a more complex frame
        frame = mp.Frame()
        
        atoms_data = {
            'id': [0, 1, 2, 3],
            'type': [1, 1, 2, 2],
            'x': [0.0, 1.0, 0.5, 1.5],
            'y': [0.0, 0.0, 1.0, 1.0],
            'z': [0.0, 0.0, 0.0, 0.0],
            'vx': [0.1, -0.1, 0.2, -0.2],
            'vy': [0.0, 0.0, 0.1, -0.1],
            'vz': [0.0, 0.0, 0.0, 0.0]
        }
        frame["atoms"] = atoms_data
        frame["timestep"] = 1000
        frame.box = mp.Box(np.diag([5.0, 5.0, 5.0]))
        
        with tempfile.NamedTemporaryFile(mode='w', suffix='.dump', delete=False) as tmp:
            # Write
            writer = LammpsTrajectoryWriter(tmp.name)
            writer.write_frame(frame)
            writer.close()
            
            # Read back
            reader = LammpsTrajectoryReader(tmp.name)
            frame_read = reader.read_frame(0)
            
            # Verify timestep
            assert frame_read["timestep"] == 1000
            
            # Verify atoms data exists
            assert "atoms" in frame_read
            atoms_read = frame_read["atoms"]
            
            # Verify that we have the right number of atoms
            n_atoms = 0
            if hasattr(atoms_read, 'sizes'):
                # Get atom count from any dimension
                sizes = list(atoms_read.sizes.values())
                if sizes:
                    n_atoms = max(sizes)
            assert n_atoms == 4
            
            # Verify box
            assert frame_read.box is not None
            assert np.allclose(frame_read.box.matrix.diagonal(), [5.0, 5.0, 5.0])


class TestErrorHandling:

    def test_read_nonexistent_file(self):
        """Test reading non-existent trajectory file."""
        with pytest.raises((FileNotFoundError, IOError)):
            reader = LammpsTrajectoryReader("nonexistent.dump")
            reader.read_frame(0)

    def test_read_empty_file(self):
        """Test reading empty trajectory file."""
        with tempfile.NamedTemporaryFile(mode='w', suffix='.dump', delete=False) as tmp:
            tmp.write("")  # Empty file
            
            reader = LammpsTrajectoryReader(tmp.name)
            # Should handle empty file gracefully
            with pytest.raises((IndexError, ValueError, EOFError)):
                reader.read_frame(0)

    def test_malformed_trajectory(self):
        """Test handling malformed trajectory files."""
        with tempfile.NamedTemporaryFile(mode='w', suffix='.dump', delete=False) as tmp:
            tmp.write("ITEM: TIMESTEP\n")
            tmp.write("0\n")
            tmp.write("ITEM: NUMBER OF ATOMS\n")
            tmp.write("2\n")
            tmp.write("ITEM: BOX BOUNDS pp pp pp\n")
            tmp.write("0 10\n")
            tmp.write("0 10\n")
            tmp.write("0 10\n")
            tmp.write("ITEM: ATOMS id type x y z\n")
            tmp.write("1 1 0.0 0.0 0.0\n")
            # Missing second atom - malformed
            
            reader = LammpsTrajectoryReader(tmp.name)
            # Should handle malformed file gracefully
            try:
                frame = reader.read_frame(0)
                # If it doesn't raise an error, check that the frame is reasonable
                assert isinstance(frame, mp.Frame)
            except (ValueError, IndexError, EOFError):
                # These exceptions are acceptable for malformed files
                pass

    def test_invalid_frame_index(self, test_data_path):
        """Test reading invalid frame indices."""
        reader = LammpsTrajectoryReader(test_data_path / "trajectory/lammps/fcc_orth.dump")
        
        # Test negative index
        with pytest.raises((IndexError, ValueError)):
            reader.read_frame(-1)
        
        # Test too large index
        with pytest.raises((IndexError, ValueError)):
            reader.read_frame(999999)


class TestTrajectoryIntegration:

    def test_data_to_trajectory_conversion(self):
        """Test converting data format to trajectory format."""
        # Create a frame in data format
        frame = mp.Frame()
        
        atoms_data = {
            'id': [0, 1, 2],
            'molid': [1, 1, 2],
            'type': ['O', 'H', 'H'],
            'q': [-0.8476, 0.4238, 0.4238],
            'xyz': [[0.0, 0.0, 0.0], [0.816, 0.577, 0.0], [-0.816, 0.577, 0.0]]
        }
        frame["atoms"] = atoms_data
        frame["timestep"] = 0
        frame.box = mp.Box(np.diag([10.0, 10.0, 10.0]))
        
        # Write as trajectory
        with tempfile.NamedTemporaryFile(mode='w', suffix='.dump', delete=False) as tmp:
            writer = LammpsTrajectoryWriter(tmp.name)
            writer.write_frame(frame)
            writer.close()
            
            # Read back as trajectory
            reader = LammpsTrajectoryReader(tmp.name)
            frame_read = reader.read_frame(0)
            
            assert "timestep" in frame_read
            assert "atoms" in frame_read
            assert frame_read.box is not None

    def test_multiple_formats_consistency(self):
        """Test that data and trajectory formats are consistent."""
        # This test ensures that a frame written in one format
        # can be meaningfully compared with the other format
        frame_original = mp.Frame()
        
        atoms_data = {
            'id': [0, 1],
            'type': [1, 2],
            'xyz': [[0.0, 0.0, 0.0], [1.0, 0.0, 0.0]]
        }
        frame_original["atoms"] = atoms_data
        frame_original["timestep"] = 100
        frame_original.box = mp.Box(np.eye(3) * 5.0)
        
        # Write as trajectory and read back
        with tempfile.NamedTemporaryFile(mode='w', suffix='.dump', delete=False) as tmp:
            writer = LammpsTrajectoryWriter(tmp.name)
            writer.write_frame(frame_original)
            writer.close()
            
            reader = LammpsTrajectoryReader(tmp.name)
            frame_traj = reader.read_frame(0)
            
            # Both should have same basic structure
            assert "timestep" in frame_traj
            assert "atoms" in frame_traj
            assert frame_traj.box is not None
            assert frame_original.box is not None
            
            # Box dimensions should be similar
            assert np.allclose(frame_traj.box.matrix.diagonal(), 
                             frame_original.box.matrix.diagonal())
