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
            if orig_frame.box is not None:
                assert read_frame.box is not None
                orig_box = orig_frame.box
                read_box = read_frame.box
                assert read_box is not None, "Box should not be None"
                assert isinstance(read_box, mp.Box), (
                    f"Expected Box, got {type(read_box)}"
                )
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
            if frame.box is not None:
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

            if orig_frame.box is not None:
                assert read_frame.box is not None, f"Box missing in frame {i}"
                read_box = read_frame.box
                assert read_box is not None, f"Box is None in frame {i}"
                assert isinstance(read_box, mp.Box), (
                    f"Box type wrong in frame {i}: {type(read_box)}"
                )

                orig_box = orig_frame.box
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


class TestHDF5Downsample:
    """Test downsampling functionality for trajectories."""

    def test_downsample_with_stride_2(self, TEST_DATA_DIR, tmp_path):
        """Test downsampling with stride 2 (every other frame)."""
        reader = read_lammps_trajectory(TEST_DATA_DIR / "lammps/unwrapped.lammpstrj")

        # Collect frames
        all_frames = []
        for i, frame in enumerate(reader):
            all_frames.append(frame)
            if i >= 9:  # Get 10 frames
                break

        # Write with stride 2 (every other frame)
        h5_file = tmp_path / "downsampled.h5"
        downsampled_frames = all_frames[::2]
        write_h5_trajectory(h5_file, downsampled_frames)

        # Read back
        read_reader = read_h5_trajectory(h5_file)
        assert read_reader.n_frames == len(downsampled_frames)

        # Verify we got the right frames
        for i, expected_frame in enumerate(downsampled_frames):
            read_frame = read_reader.read_frame(i)
            expected_atoms = expected_frame["atoms"]
            read_atoms = read_frame["atoms"]

            assert expected_atoms.nrows == read_atoms.nrows
            # Compare positions (xu, yu, zu in unwrapped LAMMPS)
            if "xu" in expected_atoms and "xu" in read_atoms:
                np.testing.assert_array_almost_equal(
                    expected_atoms["xu"], read_atoms["xu"], decimal=6
                )
            elif "x" in expected_atoms and "x" in read_atoms:
                np.testing.assert_array_almost_equal(
                    expected_atoms["x"], read_atoms["x"], decimal=6
                )

    def test_downsample_with_stride_3(self, TEST_DATA_DIR, tmp_path):
        """Test downsampling with stride 3 (every third frame)."""
        reader = read_lammps_trajectory(TEST_DATA_DIR / "lammps/unwrapped.lammpstrj")

        # Collect frames
        all_frames = []
        for i, frame in enumerate(reader):
            all_frames.append(frame)
            if i >= 11:  # Get 12 frames
                break

        # Write with stride 3
        h5_file = tmp_path / "downsampled_s3.h5"
        downsampled_frames = all_frames[::3]
        write_h5_trajectory(h5_file, downsampled_frames)

        # Read back
        read_reader = read_h5_trajectory(h5_file)
        assert read_reader.n_frames == len(downsampled_frames)
        assert read_reader.n_frames == 4  # 0, 3, 6, 9

    def test_downsample_with_slice_range(self, TEST_DATA_DIR, tmp_path):
        """Test downsampling with specific slice range [start:stop:step]."""
        reader = read_lammps_trajectory(TEST_DATA_DIR / "lammps/unwrapped.lammpstrj")

        # Collect frames
        all_frames = []
        for i, frame in enumerate(reader):
            all_frames.append(frame)
            if i >= 9:
                break

        # Write frames 2:8:2 (indices 2, 4, 6)
        h5_file = tmp_path / "downsampled_range.h5"
        downsampled_frames = all_frames[2:8:2]
        write_h5_trajectory(h5_file, downsampled_frames)

        # Read back
        read_reader = read_h5_trajectory(h5_file)
        assert read_reader.n_frames == 3

        # Verify we got the right frames
        for i, orig_idx in enumerate([2, 4, 6]):
            expected_frame = all_frames[orig_idx]
            read_frame = read_reader.read_frame(i)
            expected_atoms = expected_frame["atoms"]
            read_atoms = read_frame["atoms"]
            # Compare positions
            if "xu" in expected_atoms and "xu" in read_atoms:
                np.testing.assert_array_almost_equal(
                    expected_atoms["xu"], read_atoms["xu"], decimal=6
                )
            elif "x" in expected_atoms and "x" in read_atoms:
                np.testing.assert_array_almost_equal(
                    expected_atoms["x"], read_atoms["x"], decimal=6
                )

    def test_metadata_preservation_after_downsample(self, TEST_DATA_DIR, tmp_path):
        """Test that downsampling preserves frame metadata (timestep, box, etc.)."""
        reader = read_lammps_trajectory(TEST_DATA_DIR / "lammps/unwrapped.lammpstrj")

        # Collect frames
        all_frames = []
        for i, frame in enumerate(reader):
            all_frames.append(frame)
            if i >= 5:
                break

        # Downsample
        downsampled_frames = all_frames[::2]
        h5_file = tmp_path / "downsampled_meta.h5"
        write_h5_trajectory(h5_file, downsampled_frames)

        # Read back and verify metadata
        read_reader = read_h5_trajectory(h5_file)
        for i, orig_frame in enumerate(downsampled_frames):
            read_frame = read_reader.read_frame(i)

            # Check timestep
            if "timestep" in orig_frame.metadata:
                assert "timestep" in read_frame.metadata
                assert (
                    orig_frame.metadata["timestep"] == read_frame.metadata["timestep"]
                )

            # Check box
            if orig_frame.box is not None:
                assert read_frame.box is not None
                orig_box = orig_frame.box
                read_box = read_frame.box
                assert isinstance(read_box, mp.Box)
                np.testing.assert_array_almost_equal(
                    orig_box.matrix, read_box.matrix, decimal=6
                )


class TestHDF5Roundtrip:
    """Test complete roundtrip for various formats: Format -> H5 -> Frame (lossless)."""

    def test_roundtrip_from_lammps_basic(self, TEST_DATA_DIR, tmp_path):
        """Test basic LAMMPS trajectory -> H5 -> Trajectory roundtrip."""
        # Read original LAMMPS trajectory
        lammps_file = TEST_DATA_DIR / "lammps/unwrapped.lammpstrj"
        original_reader = read_lammps_trajectory(lammps_file)

        # Collect frames
        original_frames = []
        for i, frame in enumerate(original_reader):
            original_frames.append(frame)
            if i >= 4:
                break

        # Write to H5
        h5_file = tmp_path / "roundtrip.h5"
        write_h5_trajectory(h5_file, original_frames)

        # Read back from H5
        h5_reader = read_h5_trajectory(h5_file)
        assert h5_reader.n_frames == len(original_frames)

        # Verify all frames are identical
        for i in range(len(original_frames)):
            orig_frame = original_frames[i]
            read_frame = h5_reader.read_frame(i)

            # Check all blocks
            assert set(orig_frame._blocks.keys()) == set(read_frame._blocks.keys())

            for block_name in orig_frame._blocks.keys():
                orig_block = orig_frame[block_name]
                read_block = read_frame[block_name]

                # Check all fields in block
                assert set(orig_block.keys()) == set(read_block.keys())

                for field_name in orig_block.keys():
                    orig_data = orig_block[field_name]
                    read_data = read_block[field_name]

                    # Check dtype
                    assert orig_data.dtype == read_data.dtype

                    # Check values
                    if orig_data.dtype.kind in "biufc":
                        np.testing.assert_array_almost_equal(
                            orig_data,
                            read_data,
                            decimal=6,
                            err_msg=f"Frame {i}, block {block_name}, field {field_name}",
                        )
                    elif orig_data.dtype.kind == "U":
                        np.testing.assert_array_equal(
                            orig_data,
                            read_data,
                            err_msg=f"Frame {i}, block {block_name}, field {field_name}",
                        )

            # Check metadata
            assert set(orig_frame.metadata.keys()) == set(read_frame.metadata.keys())

            for meta_key in orig_frame.metadata.keys():
                if meta_key == "box":
                    if orig_frame.box is not None:
                        orig_box = orig_frame.box
                        read_box = read_frame.box
                        assert isinstance(read_box, mp.Box)
                        np.testing.assert_array_almost_equal(
                            orig_box.matrix, read_box.matrix, decimal=6
                        )
                        np.testing.assert_array_equal(orig_box.pbc, read_box.pbc)
                        np.testing.assert_array_almost_equal(
                            orig_box.origin, read_box.origin, decimal=6
                        )
                else:
                    assert (
                        orig_frame.metadata[meta_key] == read_frame.metadata[meta_key]
                    )

    def test_roundtrip_from_lammps_with_properties(self, TEST_DATA_DIR, tmp_path):
        """Test LAMMPS trajectory -> H5 roundtrip with additional properties."""
        lammps_file = TEST_DATA_DIR / "lammps/properties.lammpstrj"
        original_reader = read_lammps_trajectory(lammps_file)

        # Collect frames
        original_frames = []
        for i, frame in enumerate(original_reader):
            original_frames.append(frame)
            if i >= 2:
                break

        # Write to H5
        h5_file = tmp_path / "roundtrip_props.h5"
        write_h5_trajectory(h5_file, original_frames)

        # Read back
        h5_reader = read_h5_trajectory(h5_file)

        # Verify all properties are preserved
        for i in range(len(original_frames)):
            orig_frame = original_frames[i]
            read_frame = h5_reader.read_frame(i)

            orig_atoms = orig_frame["atoms"]
            read_atoms = read_frame["atoms"]

            # All fields should be present
            assert set(orig_atoms.keys()) == set(read_atoms.keys())

            # Check each field
            for field_name in orig_atoms.keys():
                orig_data = orig_atoms[field_name]
                read_data = read_atoms[field_name]

                if orig_data.dtype.kind in "biufc":
                    np.testing.assert_array_almost_equal(
                        orig_data,
                        read_data,
                        decimal=6,
                        err_msg=f"Frame {i}, field {field_name}",
                    )

    def test_roundtrip_from_lammps_all_metadata(self, TEST_DATA_DIR, tmp_path):
        """Test that all metadata is preserved in LAMMPS trajectory -> H5 roundtrip."""
        lammps_file = TEST_DATA_DIR / "lammps/unwrapped.lammpstrj"
        original_reader = read_lammps_trajectory(lammps_file)

        # Collect frames
        original_frames = []
        for i, frame in enumerate(original_reader):
            original_frames.append(frame)
            if i >= 2:
                break

        # Write to H5
        h5_file = tmp_path / "roundtrip_allmeta.h5"
        write_h5_trajectory(h5_file, original_frames)

        # Read back
        h5_reader = read_h5_trajectory(h5_file)

        # Check every metadata field in every frame
        for i in range(len(original_frames)):
            orig_frame = original_frames[i]
            read_frame = h5_reader.read_frame(i)

            # Metadata keys should match
            assert set(orig_frame.metadata.keys()) == set(read_frame.metadata.keys())

            # Check each metadata entry
            for key, orig_value in orig_frame.metadata.items():
                read_value = read_frame.metadata[key]

                if isinstance(orig_value, mp.Box):
                    assert isinstance(read_value, mp.Box)
                    np.testing.assert_array_almost_equal(
                        orig_value.matrix, read_value.matrix, decimal=6
                    )
                    np.testing.assert_array_equal(orig_value.pbc, read_value.pbc)
                    np.testing.assert_array_almost_equal(
                        orig_value.origin, read_value.origin, decimal=6
                    )
                elif isinstance(orig_value, np.ndarray):
                    np.testing.assert_array_almost_equal(
                        orig_value, read_value, decimal=6
                    )
                else:
                    assert orig_value == read_value

    def test_roundtrip_from_xyz_basic(self, TEST_DATA_DIR, tmp_path):
        """Test basic XYZ trajectory -> H5 -> Frame roundtrip."""
        from molpy.io import read_xyz_trajectory

        # Read original XYZ trajectory (compressed multi-frame file)
        xyz_file = TEST_DATA_DIR / "xyz/water.9.xyz.gz"
        original_reader = read_xyz_trajectory(xyz_file)

        # Collect frames
        original_frames = []
        for i, frame in enumerate(original_reader):
            original_frames.append(frame)
            if i >= 4:
                break

        # Write to H5
        h5_file = tmp_path / "roundtrip_xyz.h5"
        write_h5_trajectory(h5_file, original_frames)

        # Read back from H5
        h5_reader = read_h5_trajectory(h5_file)
        assert h5_reader.n_frames == len(original_frames)

        # Verify all frames are identical
        for i in range(len(original_frames)):
            orig_frame = original_frames[i]
            read_frame = h5_reader.read_frame(i)

            # Check all blocks
            assert set(orig_frame._blocks.keys()) == set(read_frame._blocks.keys())

            for block_name in orig_frame._blocks.keys():
                orig_block = orig_frame[block_name]
                read_block = read_frame[block_name]

                # Check all fields
                assert set(orig_block.keys()) == set(read_block.keys())

                for field_name in orig_block.keys():
                    orig_data = orig_block[field_name]
                    read_data = read_block[field_name]

                    # Check dtype
                    assert orig_data.dtype == read_data.dtype

                    # Check values
                    if orig_data.dtype.kind in "biufc":
                        np.testing.assert_array_almost_equal(
                            orig_data,
                            read_data,
                            decimal=6,
                            err_msg=f"Frame {i}, block {block_name}, field {field_name}",
                        )
                    elif orig_data.dtype.kind == "U":
                        np.testing.assert_array_equal(
                            orig_data,
                            read_data,
                            err_msg=f"Frame {i}, block {block_name}, field {field_name}",
                        )

    def test_roundtrip_from_xyz_extended(self, TEST_DATA_DIR, tmp_path):
        """Test XYZ trajectory -> H5 roundtrip with extended XYZ format."""
        from molpy.io import read_xyz_trajectory

        # Read extended XYZ with properties
        xyz_file = TEST_DATA_DIR / "xyz/extended.xyz"
        original_reader = read_xyz_trajectory(xyz_file)

        # Collect frames
        original_frames = []
        for i, frame in enumerate(original_reader):
            original_frames.append(frame)
            if i >= 2:
                break

        # Write to H5
        h5_file = tmp_path / "roundtrip_xyz_ext.h5"
        write_h5_trajectory(h5_file, original_frames)

        # Read back
        h5_reader = read_h5_trajectory(h5_file)

        # Verify all data
        for i in range(len(original_frames)):
            orig_frame = original_frames[i]
            read_frame = h5_reader.read_frame(i)

            # Check atoms block
            orig_atoms = orig_frame["atoms"]
            read_atoms = read_frame["atoms"]

            # All fields should be preserved
            assert set(orig_atoms.keys()) == set(read_atoms.keys())

            for field_name in orig_atoms.keys():
                orig_data = orig_atoms[field_name]
                read_data = read_atoms[field_name]

                if orig_data.dtype.kind in "biufc":
                    np.testing.assert_array_almost_equal(
                        orig_data,
                        read_data,
                        decimal=6,
                        err_msg=f"Frame {i}, field {field_name}",
                    )
                elif orig_data.dtype.kind == "U":
                    np.testing.assert_array_equal(
                        orig_data, read_data, err_msg=f"Frame {i}, field {field_name}"
                    )

    def test_roundtrip_from_xyz_single_frame(self, TEST_DATA_DIR, tmp_path):
        """Test single XYZ frame -> H5 -> Frame roundtrip."""
        from molpy.io import read_xyz

        # Read single frame
        xyz_file = TEST_DATA_DIR / "xyz/water.xyz"
        original_frame = read_xyz(xyz_file)

        # Write to H5 (single frame trajectory)
        h5_file = tmp_path / "single_frame.h5"
        write_h5_trajectory(h5_file, [original_frame])

        # Read back
        h5_reader = read_h5_trajectory(h5_file)
        assert h5_reader.n_frames == 1

        read_frame = h5_reader.read_frame(0)

        # Verify frame is identical
        orig_atoms = original_frame["atoms"]
        read_atoms = read_frame["atoms"]

        assert set(orig_atoms.keys()) == set(read_atoms.keys())

        for field_name in orig_atoms.keys():
            orig_data = orig_atoms[field_name]
            read_data = read_atoms[field_name]

            if orig_data.dtype.kind in "biufc":
                np.testing.assert_array_almost_equal(orig_data, read_data, decimal=6)
            elif orig_data.dtype.kind == "U":
                np.testing.assert_array_equal(orig_data, read_data)

    def test_roundtrip_from_xyz_metadata_preservation(self, TEST_DATA_DIR, tmp_path):
        """Test that XYZ trajectory comment line metadata is preserved in roundtrip."""
        from molpy.io import read_xyz_trajectory

        xyz_file = TEST_DATA_DIR / "xyz/water.9.xyz.gz"
        original_reader = read_xyz_trajectory(xyz_file)

        # Collect frames
        original_frames = []
        for i, frame in enumerate(original_reader):
            original_frames.append(frame)
            if i >= 2:
                break

        # Write to H5
        h5_file = tmp_path / "roundtrip_xyz_meta.h5"
        write_h5_trajectory(h5_file, original_frames)

        # Read back
        h5_reader = read_h5_trajectory(h5_file)

        # Check metadata
        for i in range(len(original_frames)):
            orig_frame = original_frames[i]
            read_frame = h5_reader.read_frame(i)

            # Check that metadata keys match
            assert set(orig_frame.metadata.keys()) == set(read_frame.metadata.keys())

            # Check each metadata value
            for key in orig_frame.metadata.keys():
                orig_value = orig_frame.metadata[key]
                read_value = read_frame.metadata[key]

                if isinstance(orig_value, np.ndarray):
                    np.testing.assert_array_almost_equal(
                        orig_value, read_value, decimal=6
                    )
                else:
                    assert orig_value == read_value


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
        boxes_found = sum(1 for f in frames if f.box is not None)
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

            if orig_frame.box is not None:
                assert read_frame.box is not None
                orig_box = orig_frame.box
                read_box = read_frame.box
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

            if orig_frame.box is not None:
                assert read_frame.box is not None
                orig_box = orig_frame.box
                read_box = read_frame.box
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
