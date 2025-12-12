"""Tests for HDF5 Trajectory I/O.

This module tests reading and writing Trajectory objects to/from HDF5 format.
Tests use LAMMPS trajectory files as input sources to ensure compatibility with
real-world molecular trajectory data.
"""

import numpy as np
import pytest

import molpy as mp


def _check_h5py():
    """Check if h5py is available and can be imported."""
    try:
        import h5py

        # Try to actually use it to catch version incompatibility
        _ = h5py.File.__name__
        return True
    except (ImportError, ValueError, AttributeError):
        return False


HAS_H5PY = _check_h5py()

if HAS_H5PY:
    from molpy.io import read_h5_trajectory, read_lammps_trajectory, write_h5_trajectory
    from molpy.io.trajectory.h5 import HDF5TrajectoryReader, HDF5TrajectoryWriter

pytestmark = pytest.mark.skipif(
    not HAS_H5PY, reason="h5py is not installed or incompatible"
)


class TestHDF5TrajectoryWriter:
    """Test HDF5TrajectoryWriter with various trajectory structures."""

    def test_write_simple_trajectory(self, TEST_DATA_DIR, tmp_path):
        """Test writing a simple trajectory."""
        # Read original LAMMPS trajectory
        reader = read_lammps_trajectory(TEST_DATA_DIR / "lammps/unwrapped.lammpstrj")

        # Collect first few frames
        frames = []
        for i, frame in enumerate(reader):
            frames.append(frame)
            if i >= 2:  # Limit to 3 frames for speed
                break

        # Write to HDF5
        h5_file = tmp_path / "test.h5"
        write_h5_trajectory(h5_file, frames)

        # Verify file exists
        assert h5_file.exists()

    def test_write_trajectory_with_properties(self, TEST_DATA_DIR, tmp_path):
        """Test writing trajectory with various atom properties."""
        # Read trajectory with properties
        reader = read_lammps_trajectory(TEST_DATA_DIR / "lammps/properties.lammpstrj")

        # Collect first few frames
        frames = []
        for i, frame in enumerate(reader):
            frames.append(frame)
            if i >= 2:
                break

        # Write to HDF5
        h5_file = tmp_path / "test.h5"
        write_h5_trajectory(h5_file, frames)

        assert h5_file.exists()

    def test_write_with_compression(self, TEST_DATA_DIR, tmp_path):
        """Test writing with different compression options."""
        reader = read_lammps_trajectory(TEST_DATA_DIR / "lammps/unwrapped.lammpstrj")

        frames = []
        for i, frame in enumerate(reader):
            frames.append(frame)
            if i >= 2:
                break

        # Test with gzip compression
        h5_file = tmp_path / "test_gzip.h5"
        write_h5_trajectory(h5_file, frames, compression="gzip", compression_opts=6)

        assert h5_file.exists()

        # Test without compression
        h5_file_no_comp = tmp_path / "test_no_comp.h5"
        write_h5_trajectory(h5_file_no_comp, frames, compression=None)

        assert h5_file_no_comp.exists()

    def test_write_empty_trajectory(self, tmp_path):
        """Test writing empty trajectory."""
        h5_file = tmp_path / "test.h5"
        write_h5_trajectory(h5_file, [])

        # Should create file but with no frames
        assert h5_file.exists()
        reader = read_h5_trajectory(h5_file)
        assert reader.n_frames == 0

    def test_write_incremental(self, TEST_DATA_DIR, tmp_path):
        """Test writing frames incrementally using context manager."""
        reader = read_lammps_trajectory(TEST_DATA_DIR / "lammps/unwrapped.lammpstrj")

        h5_file = tmp_path / "test.h5"
        with HDF5TrajectoryWriter(h5_file) as writer:
            for i, frame in enumerate(reader):
                writer.write_frame(frame)
                if i >= 2:
                    break

        assert h5_file.exists()

        # Verify frame count
        read_reader = read_h5_trajectory(h5_file)
        assert read_reader.n_frames == 3


class TestHDF5TrajectoryReader:
    """Test HDF5TrajectoryReader with various trajectory structures."""

    def test_read_simple_trajectory(self, TEST_DATA_DIR, tmp_path):
        """Test reading a simple trajectory."""
        # Read original
        original_reader = read_lammps_trajectory(
            TEST_DATA_DIR / "lammps/unwrapped.lammpstrj"
        )

        # Collect frames
        original_frames = []
        for i, frame in enumerate(original_reader):
            original_frames.append(frame)
            if i >= 2:
                break

        # Write to HDF5
        h5_file = tmp_path / "test.h5"
        write_h5_trajectory(h5_file, original_frames)

        # Read back
        read_reader = read_h5_trajectory(h5_file)

        assert read_reader.n_frames == len(original_frames)

        # Compare frames
        for i in range(len(original_frames)):
            orig_frame = original_frames[i]
            read_frame = read_reader.read_frame(i)

            # Compare atoms
            orig_atoms = orig_frame["atoms"]
            read_atoms = read_frame["atoms"]

            assert orig_atoms.nrows == read_atoms.nrows
            assert set(orig_atoms.keys()) == set(read_atoms.keys())

            # Compare numeric fields
            for key in orig_atoms.keys():
                if orig_atoms[key].dtype.kind in "biufc":
                    np.testing.assert_array_almost_equal(
                        orig_atoms[key], read_atoms[key], decimal=6
                    )
                elif orig_atoms[key].dtype.kind == "U":
                    np.testing.assert_array_equal(orig_atoms[key], read_atoms[key])

    def test_read_trajectory_with_properties(self, TEST_DATA_DIR, tmp_path):
        """Test reading trajectory with various atom properties."""
        # Read original
        original_reader = read_lammps_trajectory(
            TEST_DATA_DIR / "lammps/properties.lammpstrj"
        )

        # Collect frames
        original_frames = []
        for i, frame in enumerate(original_reader):
            original_frames.append(frame)
            if i >= 2:
                break

        # Write to HDF5
        h5_file = tmp_path / "test.h5"
        write_h5_trajectory(h5_file, original_frames)

        # Read back
        read_reader = read_h5_trajectory(h5_file)

        assert read_reader.n_frames == len(original_frames)

        # Compare first frame in detail
        orig_frame = original_frames[0]
        read_frame = read_reader.read_frame(0)

        orig_atoms = orig_frame["atoms"]
        read_atoms = read_frame["atoms"]

        # Check all properties are preserved
        assert set(orig_atoms.keys()) == set(read_atoms.keys())

        for key in orig_atoms.keys():
            if orig_atoms[key].dtype.kind in "biufc":
                np.testing.assert_array_almost_equal(
                    orig_atoms[key], read_atoms[key], decimal=6
                )

    def test_read_metadata(self, TEST_DATA_DIR, tmp_path):
        """Test reading trajectory frame metadata."""
        # Read original
        original_reader = read_lammps_trajectory(
            TEST_DATA_DIR / "lammps/unwrapped.lammpstrj"
        )

        # Collect frames
        original_frames = []
        for i, frame in enumerate(original_reader):
            original_frames.append(frame)
            if i >= 2:
                break

        # Write to HDF5
        h5_file = tmp_path / "test.h5"
        write_h5_trajectory(h5_file, original_frames)

        # Read back
        read_reader = read_h5_trajectory(h5_file)

        # Compare metadata for each frame
        for i in range(len(original_frames)):
            orig_frame = original_frames[i]
            read_frame = read_reader.read_frame(i)

            # Check timestep
            if "timestep" in orig_frame.metadata:
                assert "timestep" in read_frame.metadata
                assert (
                    orig_frame.metadata["timestep"] == read_frame.metadata["timestep"]
                )

            # Check box
            if "box" in orig_frame.metadata and orig_frame.metadata["box"] is not None:
                assert "box" in read_frame.metadata
                orig_box = orig_frame.metadata["box"]
                read_box = read_frame.metadata["box"]
                assert read_box is not None, "Box should not be None"
                assert isinstance(
                    read_box, mp.Box
                ), f"Expected Box, got {type(read_box)}"
                if orig_box is not None:
                    np.testing.assert_array_almost_equal(
                        orig_box.matrix, read_box.matrix, decimal=6
                    )
                    np.testing.assert_array_equal(orig_box.pbc, read_box.pbc)
                    np.testing.assert_array_almost_equal(
                        orig_box.origin, read_box.origin, decimal=6
                    )

    def test_trajectory_box_metadata(self, TEST_DATA_DIR, tmp_path):
        """Test trajectory with Box metadata in each frame."""
        # Read original
        original_reader = read_lammps_trajectory(
            TEST_DATA_DIR / "lammps/unwrapped.lammpstrj"
        )

        # Collect frames
        original_frames = []
        for i, frame in enumerate(original_reader):
            original_frames.append(frame)
            if i >= 2:
                break

        # Verify original frames have box
        boxes_found = 0
        for frame in original_frames:
            if "box" in frame.metadata and frame.metadata["box"] is not None:
                boxes_found += 1
        assert boxes_found > 0, "No boxes found in original frames"

        # Write to HDF5
        h5_file = tmp_path / "test.h5"
        write_h5_trajectory(h5_file, original_frames)

        # Read back
        read_reader = read_h5_trajectory(h5_file)

        # Check each frame's box
        for i in range(len(original_frames)):
            orig_frame = original_frames[i]
            read_frame = read_reader.read_frame(i)

            if "box" in orig_frame.metadata and orig_frame.metadata["box"] is not None:
                assert "box" in read_frame.metadata, f"Box missing in frame {i}"
                read_box = read_frame.metadata["box"]
                assert read_box is not None, f"Box is None in frame {i}"
                assert isinstance(
                    read_box, mp.Box
                ), f"Box type wrong in frame {i}: {type(read_box)}"

                orig_box = orig_frame.metadata["box"]
                np.testing.assert_array_almost_equal(
                    orig_box.matrix, read_box.matrix, decimal=6
                )
                np.testing.assert_array_equal(orig_box.pbc, read_box.pbc)
                np.testing.assert_array_almost_equal(
                    orig_box.origin, read_box.origin, decimal=6
                )

    def test_read_by_index(self, TEST_DATA_DIR, tmp_path):
        """Test reading specific frames by index."""
        # Read original
        original_reader = read_lammps_trajectory(
            TEST_DATA_DIR / "lammps/unwrapped.lammpstrj"
        )

        # Collect frames
        original_frames = []
        for i, frame in enumerate(original_reader):
            original_frames.append(frame)
            if i >= 4:
                break

        # Write to HDF5
        h5_file = tmp_path / "test.h5"
        write_h5_trajectory(h5_file, original_frames)

        # Read back
        read_reader = read_h5_trajectory(h5_file)

        # Test reading specific indices
        frame_0 = read_reader.read_frame(0)
        frame_2 = read_reader.read_frame(2)
        frame_last = read_reader.read_frame(-1)

        assert frame_0["atoms"].nrows == original_frames[0]["atoms"].nrows
        assert frame_2["atoms"].nrows == original_frames[2]["atoms"].nrows
        assert frame_last["atoms"].nrows == original_frames[-1]["atoms"].nrows

    def test_read_by_slice(self, TEST_DATA_DIR, tmp_path):
        """Test reading frames by slice."""
        # Read original
        original_reader = read_lammps_trajectory(
            TEST_DATA_DIR / "lammps/unwrapped.lammpstrj"
        )

        # Collect frames
        original_frames = []
        for i, frame in enumerate(original_reader):
            original_frames.append(frame)
            if i >= 4:
                break

        # Write to HDF5
        h5_file = tmp_path / "test.h5"
        write_h5_trajectory(h5_file, original_frames)

        # Read back
        read_reader = read_h5_trajectory(h5_file)

        # Test slicing
        frames_slice = read_reader[1:4]
        assert len(frames_slice) == 3
        assert frames_slice[0]["atoms"].nrows == original_frames[1]["atoms"].nrows

    def test_iterate_trajectory(self, TEST_DATA_DIR, tmp_path):
        """Test iterating through trajectory."""
        # Read original
        original_reader = read_lammps_trajectory(
            TEST_DATA_DIR / "lammps/unwrapped.lammpstrj"
        )

        # Collect frames
        original_frames = []
        for i, frame in enumerate(original_reader):
            original_frames.append(frame)
            if i >= 2:
                break

        # Write to HDF5
        h5_file = tmp_path / "test.h5"
        write_h5_trajectory(h5_file, original_frames)

        # Read back and iterate
        read_reader = read_h5_trajectory(h5_file)

        read_frames = []
        for frame in read_reader:
            read_frames.append(frame)

        assert len(read_frames) == len(original_frames)

        # Compare frames
        for orig_frame, read_frame in zip(original_frames, read_frames):
            orig_atoms = orig_frame["atoms"]
            read_atoms = read_frame["atoms"]

            assert orig_atoms.nrows == read_atoms.nrows
            for key in orig_atoms.keys():
                if orig_atoms[key].dtype.kind in "biufc":
                    np.testing.assert_array_almost_equal(
                        orig_atoms[key], read_atoms[key], decimal=6
                    )


class TestHDF5TrajectoryRoundTrip:
    """Test round-trip conversion: read -> write -> read."""

    def test_roundtrip_simple_trajectory(self, TEST_DATA_DIR, tmp_path):
        """Test round-trip for simple trajectory."""
        # Read original
        original_reader = read_lammps_trajectory(
            TEST_DATA_DIR / "lammps/unwrapped.lammpstrj"
        )

        # Collect frames
        original_frames = []
        for i, frame in enumerate(original_reader):
            original_frames.append(frame)
            if i >= 2:
                break

        # Write to HDF5
        h5_file = tmp_path / "test.h5"
        write_h5_trajectory(h5_file, original_frames)

        # Read back
        read_reader = read_h5_trajectory(h5_file)

        assert read_reader.n_frames == len(original_frames)

        # Compare all frames
        for i in range(len(original_frames)):
            orig_frame = original_frames[i]
            read_frame = read_reader.read_frame(i)

            # Compare atoms
            orig_atoms = orig_frame["atoms"]
            read_atoms = read_frame["atoms"]

            assert orig_atoms.nrows == read_atoms.nrows
            assert set(orig_atoms.keys()) == set(read_atoms.keys())

            # Compare all fields
            for key in orig_atoms.keys():
                if orig_atoms[key].dtype.kind in "biufc":
                    np.testing.assert_array_almost_equal(
                        orig_atoms[key], read_atoms[key], decimal=6
                    )
                elif orig_atoms[key].dtype.kind == "U":
                    np.testing.assert_array_equal(orig_atoms[key], read_atoms[key])

    def test_roundtrip_with_compression(self, TEST_DATA_DIR, tmp_path):
        """Test round-trip with compression enabled."""
        # Read original
        original_reader = read_lammps_trajectory(
            TEST_DATA_DIR / "lammps/unwrapped.lammpstrj"
        )

        # Collect frames
        original_frames = []
        for i, frame in enumerate(original_reader):
            original_frames.append(frame)
            if i >= 2:
                break

        # Write with compression
        h5_file = tmp_path / "test.h5"
        write_h5_trajectory(
            h5_file, original_frames, compression="gzip", compression_opts=6
        )

        # Read back
        read_reader = read_h5_trajectory(h5_file)

        assert read_reader.n_frames == len(original_frames)

        # Compare first frame
        orig_frame = original_frames[0]
        read_frame = read_reader.read_frame(0)

        orig_atoms = orig_frame["atoms"]
        read_atoms = read_frame["atoms"]

        for key in orig_atoms.keys():
            if orig_atoms[key].dtype.kind in "biufc":
                np.testing.assert_array_almost_equal(
                    orig_atoms[key], read_atoms[key], decimal=6
                )

    def test_roundtrip_multiple_frames(self, TEST_DATA_DIR, tmp_path):
        """Test round-trip with multiple frames."""
        # Read original
        original_reader = read_lammps_trajectory(
            TEST_DATA_DIR / "lammps/unwrapped.lammpstrj"
        )

        # Collect more frames
        original_frames = []
        for i, frame in enumerate(original_reader):
            original_frames.append(frame)
            if i >= 4:
                break

        # Write to HDF5
        h5_file = tmp_path / "test.h5"
        write_h5_trajectory(h5_file, original_frames)

        # Read back
        read_reader = read_h5_trajectory(h5_file)

        assert read_reader.n_frames == len(original_frames)

        # Compare all frames
        for i in range(len(original_frames)):
            orig_frame = original_frames[i]
            read_frame = read_reader.read_frame(i)

            orig_atoms = orig_frame["atoms"]
            read_atoms = read_frame["atoms"]

            assert orig_atoms.nrows == read_atoms.nrows

            # Compare coordinates
            if "x" in orig_atoms and "x" in read_atoms:
                np.testing.assert_array_almost_equal(
                    orig_atoms["x"], read_atoms["x"], decimal=6
                )


class TestHDF5TrajectoryErrorHandling:
    """Test error handling and edge cases."""

    def test_read_nonexistent_file(self, tmp_path):
        """Test reading non-existent file."""
        h5_file = tmp_path / "nonexistent.h5"
        with pytest.raises((FileNotFoundError, OSError)):
            read_h5_trajectory(h5_file)

    def test_read_invalid_frame_index(self, TEST_DATA_DIR, tmp_path):
        """Test reading invalid frame index."""
        # Read original
        original_reader = read_lammps_trajectory(
            TEST_DATA_DIR / "lammps/unwrapped.lammpstrj"
        )

        # Collect frames
        original_frames = []
        for i, frame in enumerate(original_reader):
            original_frames.append(frame)
            if i >= 2:
                break

        # Write to HDF5
        h5_file = tmp_path / "test.h5"
        write_h5_trajectory(h5_file, original_frames)

        # Read back
        read_reader = read_h5_trajectory(h5_file)

        # Test invalid indices
        with pytest.raises(IndexError):
            read_reader.read_frame(999)

        with pytest.raises(IndexError):
            read_reader.read_frame(-999)

    def test_read_empty_trajectory(self, tmp_path):
        """Test reading empty trajectory."""
        h5_file = tmp_path / "test.h5"
        write_h5_trajectory(h5_file, [])

        read_reader = read_h5_trajectory(h5_file)
        assert read_reader.n_frames == 0

        # Iterating should yield nothing
        frames = list(read_reader)
        assert len(frames) == 0


class TestHDF5TrajectoryContextManager:
    """Test context manager usage."""

    def test_writer_context_manager(self, TEST_DATA_DIR, tmp_path):
        """Test using HDF5TrajectoryWriter as context manager."""
        original_reader = read_lammps_trajectory(
            TEST_DATA_DIR / "lammps/unwrapped.lammpstrj"
        )

        h5_file = tmp_path / "test.h5"
        with HDF5TrajectoryWriter(h5_file) as writer:
            for i, frame in enumerate(original_reader):
                writer.write_frame(frame)
                if i >= 2:
                    break

        assert h5_file.exists()

        # Read back to verify
        read_reader = read_h5_trajectory(h5_file)
        assert read_reader.n_frames == 3

    def test_reader_context_manager(self, TEST_DATA_DIR, tmp_path):
        """Test using HDF5TrajectoryReader as context manager."""
        original_reader = read_lammps_trajectory(
            TEST_DATA_DIR / "lammps/unwrapped.lammpstrj"
        )

        # Collect frames
        original_frames = []
        for i, frame in enumerate(original_reader):
            original_frames.append(frame)
            if i >= 2:
                break

        # Write to HDF5
        h5_file = tmp_path / "test.h5"
        write_h5_trajectory(h5_file, original_frames)

        # Read with context manager
        with HDF5TrajectoryReader(h5_file) as reader:
            read_frame = reader.read_frame(0)

        assert read_frame["atoms"].nrows == original_frames[0]["atoms"].nrows


class TestHDF5TrajectoryAppend:
    """Test appending frames to existing trajectory."""

    def test_append_frames(self, TEST_DATA_DIR, tmp_path):
        """Test appending frames to existing HDF5 trajectory."""
        original_reader = read_lammps_trajectory(
            TEST_DATA_DIR / "lammps/unwrapped.lammpstrj"
        )

        # Write first batch
        frames_1 = []
        for i, frame in enumerate(original_reader):
            frames_1.append(frame)
            if i >= 1:
                break

        h5_file = tmp_path / "test.h5"
        write_h5_trajectory(h5_file, frames_1)

        # Append more frames
        frames_2 = []
        for i, frame in enumerate(original_reader):
            frames_2.append(frame)
            if i >= 1:
                break

        with HDF5TrajectoryWriter(h5_file) as writer:
            for frame in frames_2:
                writer.write_frame(frame)

        # Read back and verify
        read_reader = read_h5_trajectory(h5_file)
        assert read_reader.n_frames == len(frames_1) + len(frames_2)


class TestHDF5TrajectoryCompression:
    """Comprehensive tests for HDF5 trajectory compression options."""

    @pytest.mark.parametrize("compression", ["gzip", "lzf", None])
    def test_compression_options_trajectory(self, TEST_DATA_DIR, tmp_path, compression):
        """Test all compression options for Trajectory I/O."""
        reader = read_lammps_trajectory(TEST_DATA_DIR / "lammps/unwrapped.lammpstrj")

        frames = []
        for i, frame in enumerate(reader):
            frames.append(frame)
            if i >= 2:
                break

        # Write with specified compression
        h5_file = tmp_path / f"test_{compression or 'none'}.h5"
        if compression == "gzip":
            write_h5_trajectory(h5_file, frames, compression="gzip", compression_opts=4)
        elif compression == "lzf":
            write_h5_trajectory(h5_file, frames, compression="lzf")
        else:
            write_h5_trajectory(h5_file, frames, compression=None)

        assert h5_file.exists()

        # Read back and verify
        read_reader = read_h5_trajectory(h5_file)
        assert read_reader.n_frames == len(frames)

        # Compare all frames
        for i in range(len(frames)):
            orig_frame = frames[i]
            read_frame = read_reader.read_frame(i)

            orig_atoms = orig_frame["atoms"]
            read_atoms = read_frame["atoms"]

            assert orig_atoms.nrows == read_atoms.nrows
            for key in orig_atoms.keys():
                if orig_atoms[key].dtype.kind in "biufc":
                    np.testing.assert_array_almost_equal(
                        orig_atoms[key], read_atoms[key], decimal=6
                    )

    @pytest.mark.parametrize("compression_opts", [1, 4, 9])
    def test_gzip_compression_levels_trajectory(
        self, TEST_DATA_DIR, tmp_path, compression_opts
    ):
        """Test different gzip compression levels for trajectories."""
        reader = read_lammps_trajectory(TEST_DATA_DIR / "lammps/unwrapped.lammpstrj")

        frames = []
        for i, frame in enumerate(reader):
            frames.append(frame)
            if i >= 2:
                break

        h5_file = tmp_path / f"test_gzip_{compression_opts}.h5"
        write_h5_trajectory(
            h5_file, frames, compression="gzip", compression_opts=compression_opts
        )

        assert h5_file.exists()

        # Read back and verify data integrity
        read_reader = read_h5_trajectory(h5_file)
        assert read_reader.n_frames == len(frames)

        for i in range(len(frames)):
            orig_frame = frames[i]
            read_frame = read_reader.read_frame(i)

            orig_atoms = orig_frame["atoms"]
            read_atoms = read_frame["atoms"]

            assert orig_atoms.nrows == read_atoms.nrows
            for key in orig_atoms.keys():
                if orig_atoms[key].dtype.kind in "biufc":
                    np.testing.assert_array_almost_equal(
                        orig_atoms[key], read_atoms[key], decimal=6
                    )

    def test_compression_with_trajectory_metadata(self, TEST_DATA_DIR, tmp_path):
        """Test compression with trajectories containing metadata including Box."""
        reader = read_lammps_trajectory(TEST_DATA_DIR / "lammps/unwrapped.lammpstrj")

        frames = []
        for i, frame in enumerate(reader):
            frames.append(frame)
            if i >= 2:
                break

        # Verify frames have box
        boxes_found = sum(
            1 for f in frames if "box" in f.metadata and f.metadata["box"] is not None
        )
        assert boxes_found > 0, "No boxes found in frames"

        # Test gzip with metadata
        h5_file_gzip = tmp_path / "test_gzip_metadata.h5"
        write_h5_trajectory(
            h5_file_gzip, frames, compression="gzip", compression_opts=4
        )

        read_reader_gzip = read_h5_trajectory(h5_file_gzip)
        for i in range(len(frames)):
            orig_frame = frames[i]
            read_frame = read_reader_gzip.read_frame(i)

            if "box" in orig_frame.metadata and orig_frame.metadata["box"] is not None:
                assert "box" in read_frame.metadata
                orig_box = orig_frame.metadata["box"]
                read_box = read_frame.metadata["box"]
                assert isinstance(read_box, mp.Box)
                np.testing.assert_array_almost_equal(
                    orig_box.matrix, read_box.matrix, decimal=6
                )

        # Test lzf with metadata
        h5_file_lzf = tmp_path / "test_lzf_metadata.h5"
        write_h5_trajectory(h5_file_lzf, frames, compression="lzf")

        read_reader_lzf = read_h5_trajectory(h5_file_lzf)
        for i in range(len(frames)):
            orig_frame = frames[i]
            read_frame = read_reader_lzf.read_frame(i)

            if "box" in orig_frame.metadata and orig_frame.metadata["box"] is not None:
                assert "box" in read_frame.metadata
                orig_box = orig_frame.metadata["box"]
                read_box = read_frame.metadata["box"]
                assert isinstance(read_box, mp.Box)
                np.testing.assert_array_almost_equal(
                    orig_box.matrix, read_box.matrix, decimal=6
                )

    def test_compression_file_sizes_trajectory(self, TEST_DATA_DIR, tmp_path):
        """Test that compression options work for trajectories."""
        reader = read_lammps_trajectory(TEST_DATA_DIR / "lammps/unwrapped.lammpstrj")

        frames = []
        for i, frame in enumerate(reader):
            frames.append(frame)
            if i >= 4:  # Use more frames for better compression test
                break

        # Write without compression
        h5_file_no_comp = tmp_path / "test_no_comp.h5"
        write_h5_trajectory(h5_file_no_comp, frames, compression=None)
        size_no_comp = h5_file_no_comp.stat().st_size
        assert size_no_comp > 0

        # Write with gzip
        h5_file_gzip = tmp_path / "test_gzip.h5"
        write_h5_trajectory(
            h5_file_gzip, frames, compression="gzip", compression_opts=9
        )
        size_gzip = h5_file_gzip.stat().st_size
        assert size_gzip > 0

        # Write with lzf
        h5_file_lzf = tmp_path / "test_lzf.h5"
        write_h5_trajectory(h5_file_lzf, frames, compression="lzf")
        size_lzf = h5_file_lzf.stat().st_size
        assert size_lzf > 0

        # Verify all files are readable
        read_reader_no_comp = read_h5_trajectory(h5_file_no_comp)
        read_reader_gzip = read_h5_trajectory(h5_file_gzip)
        read_reader_lzf = read_h5_trajectory(h5_file_lzf)

        assert read_reader_no_comp.n_frames == len(frames)
        assert read_reader_gzip.n_frames == len(frames)
        assert read_reader_lzf.n_frames == len(frames)

    def test_compression_with_writer_class_trajectory(self, TEST_DATA_DIR, tmp_path):
        """Test compression using HDF5TrajectoryWriter class directly."""
        reader = read_lammps_trajectory(TEST_DATA_DIR / "lammps/unwrapped.lammpstrj")

        frames = []
        for i, frame in enumerate(reader):
            frames.append(frame)
            if i >= 2:
                break

        # Test gzip with HDF5TrajectoryWriter
        h5_file_gzip = tmp_path / "test_writer_gzip.h5"
        with HDF5TrajectoryWriter(
            h5_file_gzip, compression="gzip", compression_opts=4
        ) as writer_gzip:
            for frame in frames:
                writer_gzip.write_frame(frame)

        read_reader_gzip = read_h5_trajectory(h5_file_gzip)
        assert read_reader_gzip.n_frames == len(frames)

        # Test lzf with HDF5TrajectoryWriter
        h5_file_lzf = tmp_path / "test_writer_lzf.h5"
        with HDF5TrajectoryWriter(h5_file_lzf, compression="lzf") as writer_lzf:
            for frame in frames:
                writer_lzf.write_frame(frame)

        read_reader_lzf = read_h5_trajectory(h5_file_lzf)
        assert read_reader_lzf.n_frames == len(frames)

    def test_compression_with_context_manager_trajectory(self, TEST_DATA_DIR, tmp_path):
        """Test compression using context manager for trajectories."""
        reader = read_lammps_trajectory(TEST_DATA_DIR / "lammps/unwrapped.lammpstrj")

        frames = []
        for i, frame in enumerate(reader):
            frames.append(frame)
            if i >= 2:
                break

        # Test gzip with context manager
        h5_file_gzip = tmp_path / "test_ctx_gzip.h5"
        with HDF5TrajectoryWriter(
            h5_file_gzip, compression="gzip", compression_opts=4
        ) as writer:
            for frame in frames:
                writer.write_frame(frame)

        read_reader_gzip = read_h5_trajectory(h5_file_gzip)
        assert read_reader_gzip.n_frames == len(frames)

        # Test lzf with context manager
        h5_file_lzf = tmp_path / "test_ctx_lzf.h5"
        with HDF5TrajectoryWriter(h5_file_lzf, compression="lzf") as writer:
            for frame in frames:
                writer.write_frame(frame)

        read_reader_lzf = read_h5_trajectory(h5_file_lzf)
        assert read_reader_lzf.n_frames == len(frames)

    def test_compression_with_append_trajectory(self, TEST_DATA_DIR, tmp_path):
        """Test compression when appending frames to trajectory."""
        reader = read_lammps_trajectory(TEST_DATA_DIR / "lammps/unwrapped.lammpstrj")

        frames_1 = []
        for i, frame in enumerate(reader):
            frames_1.append(frame)
            if i >= 1:
                break

        frames_2 = []
        for i, frame in enumerate(reader):
            frames_2.append(frame)
            if i >= 1:
                break

        # Write first batch with gzip
        h5_file = tmp_path / "test_append_gzip.h5"
        write_h5_trajectory(h5_file, frames_1, compression="gzip", compression_opts=4)

        # Append second batch (should use same compression)
        with HDF5TrajectoryWriter(
            h5_file, compression="gzip", compression_opts=4
        ) as writer:
            for frame in frames_2:
                writer.write_frame(frame)

        read_reader = read_h5_trajectory(h5_file)
        assert read_reader.n_frames == len(frames_1) + len(frames_2)

        # Verify all frames are readable
        for i in range(len(frames_1)):
            orig_frame = frames_1[i]
            read_frame = read_reader.read_frame(i)
            assert orig_frame["atoms"].nrows == read_frame["atoms"].nrows

        for i in range(len(frames_2)):
            orig_frame = frames_2[i]
            read_frame = read_reader.read_frame(len(frames_1) + i)
            assert orig_frame["atoms"].nrows == read_frame["atoms"].nrows

    def test_compression_mixed_data_types_trajectory(self, TEST_DATA_DIR, tmp_path):
        """Test compression with trajectories containing mixed data types."""
        reader = read_lammps_trajectory(TEST_DATA_DIR / "lammps/unwrapped.lammpstrj")

        frames = []
        for i, frame in enumerate(reader):
            frames.append(frame)
            if i >= 2:
                break

        # Test gzip with mixed types
        h5_file_gzip = tmp_path / "test_gzip_mixed.h5"
        write_h5_trajectory(
            h5_file_gzip, frames, compression="gzip", compression_opts=4
        )

        read_reader_gzip = read_h5_trajectory(h5_file_gzip)
        for i in range(len(frames)):
            orig_frame = frames[i]
            read_frame = read_reader_gzip.read_frame(i)

            # Check all blocks
            for block_name in orig_frame._blocks.keys():
                orig_block = orig_frame[block_name]
                read_block = read_frame[block_name]

                for var_name in orig_block.keys():
                    orig_data = orig_block[var_name]
                    read_data = read_block[var_name]

                    if orig_data.dtype.kind in "biufc":
                        np.testing.assert_array_almost_equal(
                            orig_data, read_data, decimal=6
                        )
                    elif orig_data.dtype.kind == "U":
                        np.testing.assert_array_equal(orig_data, read_data)

        # Test lzf with mixed types
        h5_file_lzf = tmp_path / "test_lzf_mixed.h5"
        write_h5_trajectory(h5_file_lzf, frames, compression="lzf")

        read_reader_lzf = read_h5_trajectory(h5_file_lzf)
        for i in range(len(frames)):
            orig_frame = frames[i]
            read_frame = read_reader_lzf.read_frame(i)

            # Check all blocks
            for block_name in orig_frame._blocks.keys():
                orig_block = orig_frame[block_name]
                read_block = read_frame[block_name]

                for var_name in orig_block.keys():
                    orig_data = orig_block[var_name]
                    read_data = read_block[var_name]

                    if orig_data.dtype.kind in "biufc":
                        np.testing.assert_array_almost_equal(
                            orig_data, read_data, decimal=6
                        )
                    elif orig_data.dtype.kind == "U":
                        np.testing.assert_array_equal(orig_data, read_data)

