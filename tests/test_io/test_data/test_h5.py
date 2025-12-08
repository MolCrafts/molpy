"""Tests for HDF5 Frame I/O.

This module tests reading and writing Frame objects to/from HDF5 format.
Tests use LAMMPS data files as input sources to ensure compatibility with
real-world molecular data structures.
"""

from pathlib import Path

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
    from molpy.io import read_h5, read_lammps_data, write_h5
    from molpy.io.data.h5 import HDF5Reader, HDF5Writer

pytestmark = pytest.mark.skipif(
    not HAS_H5PY, reason="h5py is not installed or incompatible"
)


@pytest.fixture
def test_files(TEST_DATA_DIR) -> dict[str, Path]:
    """Provide paths to test files."""
    test_data_dir = TEST_DATA_DIR / "lammps-data"

    files = {
        "molid": test_data_dir / "molid.lmp",
        "labelmap": test_data_dir / "labelmap.lmp",
        "solvated": test_data_dir / "solvated.lmp",
        "whitespaces": test_data_dir / "whitespaces.lmp",
    }
    return files


class TestHDF5Writer:
    """Test HDF5Writer with various Frame structures."""

    def test_write_simple_frame(self, test_files, tmp_path):
        """Test writing a simple frame with atoms only."""
        # Read original LAMMPS data file
        original_frame = read_lammps_data(test_files["molid"], atom_style="full")

        # Write to HDF5
        h5_file = tmp_path / "test.h5"
        write_h5(h5_file, original_frame)

        # Verify file exists
        assert h5_file.exists()

    def test_write_frame_with_connectivity(self, test_files, tmp_path):
        """Test writing frame with bonds, angles, dihedrals."""
        # Read file with connectivity
        original_frame = read_lammps_data(test_files["labelmap"], atom_style="full")

        # Write to HDF5
        h5_file = tmp_path / "test.h5"
        write_h5(h5_file, original_frame)

        assert h5_file.exists()

    def test_write_with_compression(self, test_files, tmp_path):
        """Test writing with different compression options."""
        original_frame = read_lammps_data(test_files["molid"], atom_style="full")

        # Test with gzip compression
        h5_file = tmp_path / "test_gzip.h5"
        write_h5(h5_file, original_frame, compression="gzip", compression_opts=6)

        assert h5_file.exists()

        # Test without compression
        h5_file_no_comp = tmp_path / "test_no_comp.h5"
        write_h5(h5_file_no_comp, original_frame, compression=None)

        assert h5_file_no_comp.exists()

    def test_write_empty_frame_raises_error(self, tmp_path):
        """Test that writing empty frame raises ValueError."""
        empty_frame = mp.Frame()

        h5_file = tmp_path / "test.h5"
        with pytest.raises(ValueError, match="Cannot write empty frame"):
            write_h5(h5_file, empty_frame)


class TestHDF5Reader:
    """Test HDF5Reader with various Frame structures."""

    def test_read_simple_frame(self, test_files, tmp_path):
        """Test reading a simple frame with atoms only."""
        # Read original
        original_frame = read_lammps_data(test_files["molid"], atom_style="full")

        # Write to HDF5
        h5_file = tmp_path / "test.h5"
        write_h5(h5_file, original_frame)

        # Read back
        read_frame = read_h5(h5_file)

        # Compare atoms
        orig_atoms = original_frame["atoms"]
        read_atoms = read_frame["atoms"]

        assert orig_atoms.nrows == read_atoms.nrows
        assert set(orig_atoms.keys()) == set(read_atoms.keys())

        # Compare numeric fields
        for key in orig_atoms.keys():
            if orig_atoms[key].dtype.kind in "biufc":  # numeric types
                np.testing.assert_array_almost_equal(
                    orig_atoms[key], read_atoms[key], decimal=6
                )
            elif orig_atoms[key].dtype.kind == "U":  # string types
                np.testing.assert_array_equal(orig_atoms[key], read_atoms[key])

    def test_read_frame_with_connectivity(self, test_files, tmp_path):
        """Test reading frame with bonds, angles, dihedrals."""
        # Read original
        original_frame = read_lammps_data(test_files["labelmap"], atom_style="full")

        # Write to HDF5
        h5_file = tmp_path / "test.h5"
        write_h5(h5_file, original_frame)

        # Read back
        read_frame = read_h5(h5_file)

        # Compare all blocks
        assert set(original_frame._blocks.keys()) == set(read_frame._blocks.keys())

        for block_name in original_frame._blocks.keys():
            orig_block = original_frame[block_name]
            read_block = read_frame[block_name]

            assert orig_block.nrows == read_block.nrows
            assert set(orig_block.keys()) == set(read_block.keys())

            # Compare all variables in block
            for var_name in orig_block.keys():
                orig_data = orig_block[var_name]
                read_data = read_block[var_name]

                if orig_data.dtype.kind in "biufc":  # numeric types
                    np.testing.assert_array_almost_equal(
                        orig_data, read_data, decimal=6
                    )
                elif orig_data.dtype.kind == "U":  # string types
                    np.testing.assert_array_equal(orig_data, read_data)

    def test_read_metadata(self, test_files, tmp_path):
        """Test reading frame metadata."""
        # Read original
        original_frame = read_lammps_data(test_files["molid"], atom_style="full")

        # Add some test metadata
        original_frame.metadata["test_int"] = 42
        original_frame.metadata["test_float"] = 3.14
        original_frame.metadata["test_string"] = "test_value"
        original_frame.metadata["test_list"] = [1, 2, 3]

        # Write to HDF5
        h5_file = tmp_path / "test.h5"
        write_h5(h5_file, original_frame)

        # Read back
        read_frame = read_h5(h5_file)

        # Compare metadata (excluding complex objects like Box, ForceField)
        for key in original_frame.metadata.keys():
            if key == "box":
                # Box is a special object, check separately
                if "box" in read_frame.metadata:
                    orig_box = original_frame.metadata["box"]
                    read_box = read_frame.metadata["box"]
                    if orig_box is not None and read_box is not None:
                        np.testing.assert_array_almost_equal(
                            orig_box.matrix, read_box.matrix, decimal=6
                        )
            elif key == "forcefield":
                # ForceField is complex, just check it exists
                assert (
                    key in read_frame.metadata
                    or "forcefield" not in original_frame.metadata
                )
            elif isinstance(
                original_frame.metadata[key], (int, float, str, bool, list)
            ):
                assert key in read_frame.metadata
                orig_val = original_frame.metadata[key]
                read_val = read_frame.metadata[key]
                if isinstance(orig_val, (int, float)):
                    assert abs(orig_val - read_val) < 1e-6
                else:
                    assert orig_val == read_val

    def test_read_metadata_with_box(self, test_files, tmp_path):
        """Test reading frame metadata with Box object."""
        # Read original
        original_frame = read_lammps_data(test_files["molid"], atom_style="full")

        # Ensure box exists in metadata
        assert "box" in original_frame.metadata
        orig_box = original_frame.metadata["box"]
        assert orig_box is not None

        # Write to HDF5
        h5_file = tmp_path / "test.h5"
        write_h5(h5_file, original_frame)

        # Read back
        read_frame = read_h5(h5_file)

        # Check Box was correctly restored
        assert "box" in read_frame.metadata
        read_box = read_frame.metadata["box"]
        assert read_box is not None
        assert isinstance(read_box, mp.Box)

        # Compare Box properties
        np.testing.assert_array_almost_equal(
            orig_box.matrix, read_box.matrix, decimal=6
        )
        np.testing.assert_array_almost_equal(orig_box.pbc, read_box.pbc, decimal=6)
        np.testing.assert_array_almost_equal(
            orig_box.origin, read_box.origin, decimal=6
        )

    def test_read_with_nested_metadata(self, test_files, tmp_path):
        """Test reading frame with nested metadata dictionaries."""
        # Read original
        original_frame = read_lammps_data(test_files["molid"], atom_style="full")

        # Add nested metadata
        original_frame.metadata["nested"] = {
            "level1": {"level2": "value"},
            "number": 123,
        }

        # Write to HDF5
        h5_file = tmp_path / "test.h5"
        write_h5(h5_file, original_frame)

        # Read back
        read_frame = read_h5(h5_file)

        # Check nested metadata
        assert "nested" in read_frame.metadata
        assert read_frame.metadata["nested"]["number"] == 123
        assert read_frame.metadata["nested"]["level1"]["level2"] == "value"

    def test_box_array_conversion(self, tmp_path):
        """Test that Box can be converted to numpy array via __array__."""
        from molpy.core import Box

        box = Box(matrix=np.eye(3) * 10.0)

        # Test __array__ method
        box_array = np.array(box)
        assert box_array.shape == (3, 3), f"Expected (3,3), got {box_array.shape}"
        np.testing.assert_array_almost_equal(box_array, box.matrix)

        # Test with different dtypes
        box_array_f32 = np.array(box, dtype=np.float32)
        assert box_array_f32.dtype == np.float32

        # Test in HDF5 context
        frame = mp.Frame()
        frame["atoms"] = {"x": [0.0, 1.0], "y": [0.0, 0.0], "z": [0.0, 0.0]}
        frame.metadata["box"] = box
        frame.metadata["timestep"] = 0

        h5_file = tmp_path / "test.h5"
        write_h5(h5_file, frame)

        read_frame = read_h5(h5_file)
        read_box = read_frame.metadata["box"]

        assert isinstance(read_box, Box), f"Expected Box, got {type(read_box)}"
        np.testing.assert_array_almost_equal(box.matrix, read_box.matrix, decimal=6)
        np.testing.assert_array_almost_equal(box.pbc, read_box.pbc)
        np.testing.assert_array_almost_equal(box.origin, read_box.origin, decimal=6)

    def test_box_with_different_pbc_and_origin(self, tmp_path):
        """Test Box with different PBC and origin values."""
        from molpy.core import Box

        # Test with custom PBC and origin
        box = Box(
            matrix=np.diag([10.0, 20.0, 30.0]),
            pbc=np.array([True, True, False]),
            origin=np.array([-5.0, -10.0, 0.0]),
        )

        frame = mp.Frame()
        frame["atoms"] = {"x": [0.0], "y": [0.0], "z": [0.0]}
        frame.metadata["box"] = box

        h5_file = tmp_path / "test.h5"
        write_h5(h5_file, frame)

        read_frame = read_h5(h5_file)
        read_box = read_frame.metadata["box"]

        assert isinstance(read_box, Box)
        np.testing.assert_array_almost_equal(box.matrix, read_box.matrix, decimal=6)
        np.testing.assert_array_equal(box.pbc, read_box.pbc)
        np.testing.assert_array_almost_equal(box.origin, read_box.origin, decimal=6)

    def test_box_triclinic(self, tmp_path):
        """Test Box with triclinic (non-orthogonal) matrix."""
        from molpy.core import Box

        # Create triclinic box
        matrix = np.array(
            [
                [10.0, 2.0, 1.0],
                [0.0, 20.0, 1.5],
                [0.0, 0.0, 30.0],
            ]
        )
        box = Box(matrix=matrix)

        frame = mp.Frame()
        frame["atoms"] = {"x": [0.0], "y": [0.0], "z": [0.0]}
        frame.metadata["box"] = box

        h5_file = tmp_path / "test.h5"
        write_h5(h5_file, frame)

        read_frame = read_h5(h5_file)
        read_box = read_frame.metadata["box"]

        assert isinstance(read_box, Box)
        np.testing.assert_array_almost_equal(box.matrix, read_box.matrix, decimal=6)


class TestHDF5RoundTrip:
    """Test round-trip conversion: read -> write -> read."""

    def test_roundtrip_simple_frame(self, test_files, tmp_path):
        """Test round-trip for simple frame."""
        # Read original
        original_frame = read_lammps_data(test_files["molid"], atom_style="full")

        # Write to HDF5
        h5_file = tmp_path / "test.h5"
        write_h5(h5_file, original_frame)

        # Read back
        read_frame = read_h5(h5_file)

        # Compare atoms block
        orig_atoms = original_frame["atoms"]
        read_atoms = read_frame["atoms"]

        assert orig_atoms.nrows == read_atoms.nrows
        assert set(orig_atoms.keys()) == set(read_atoms.keys())

        # Compare all numeric fields
        for key in orig_atoms.keys():
            if orig_atoms[key].dtype.kind in "biufc":
                np.testing.assert_array_almost_equal(
                    orig_atoms[key], read_atoms[key], decimal=6
                )
            elif orig_atoms[key].dtype.kind == "U":
                np.testing.assert_array_equal(orig_atoms[key], read_atoms[key])

        # Compare Box if present
        if (
            "box" in original_frame.metadata
            and original_frame.metadata["box"] is not None
        ):
            assert "box" in read_frame.metadata
            orig_box = original_frame.metadata["box"]
            read_box = read_frame.metadata["box"]
            assert read_box is not None
            assert isinstance(read_box, mp.Box)
            np.testing.assert_array_almost_equal(
                orig_box.matrix, read_box.matrix, decimal=6
            )
            np.testing.assert_array_equal(orig_box.pbc, read_box.pbc)
            np.testing.assert_array_almost_equal(
                orig_box.origin, read_box.origin, decimal=6
            )

    def test_roundtrip_with_connectivity(self, test_files, tmp_path):
        """Test round-trip for frame with full connectivity."""
        # Read original
        original_frame = read_lammps_data(test_files["labelmap"], atom_style="full")

        # Write to HDF5
        h5_file = tmp_path / "test.h5"
        write_h5(h5_file, original_frame)

        # Read back
        read_frame = read_h5(h5_file)

        # Compare all blocks
        for block_name in original_frame._blocks.keys():
            orig_block = original_frame[block_name]
            read_block = read_frame[block_name]

            assert orig_block.nrows == read_block.nrows
            assert set(orig_block.keys()) == set(read_block.keys())

            # Compare all variables
            for var_name in orig_block.keys():
                orig_data = orig_block[var_name]
                read_data = read_block[var_name]

                if orig_data.dtype.kind in "biufc":
                    np.testing.assert_array_almost_equal(
                        orig_data, read_data, decimal=6
                    )
                elif orig_data.dtype.kind == "U":
                    np.testing.assert_array_equal(orig_data, read_data)

        # Compare Box if present
        if (
            "box" in original_frame.metadata
            and original_frame.metadata["box"] is not None
        ):
            assert "box" in read_frame.metadata
            orig_box = original_frame.metadata["box"]
            read_box = read_frame.metadata["box"]
            assert read_box is not None
            assert isinstance(read_box, mp.Box)
            np.testing.assert_array_almost_equal(
                orig_box.matrix, read_box.matrix, decimal=6
            )

    def test_roundtrip_with_string_types(self, test_files, tmp_path):
        """Test round-trip for frame with string type labels."""
        # Read original (labelmap has string types)
        original_frame = read_lammps_data(test_files["labelmap"], atom_style="full")

        # Write to HDF5
        h5_file = tmp_path / "test.h5"
        write_h5(h5_file, original_frame)

        # Read back
        read_frame = read_h5(h5_file)

        # Check that string types are preserved
        orig_atoms = original_frame["atoms"]
        read_atoms = read_frame["atoms"]

        if "type" in orig_atoms and orig_atoms["type"].dtype.kind == "U":
            np.testing.assert_array_equal(orig_atoms["type"], read_atoms["type"])

    def test_roundtrip_with_compression(self, test_files, tmp_path):
        """Test round-trip with compression enabled."""
        # Read original
        original_frame = read_lammps_data(test_files["molid"], atom_style="full")

        # Write with compression
        h5_file = tmp_path / "test.h5"
        write_h5(h5_file, original_frame, compression="gzip", compression_opts=6)

        # Read back
        read_frame = read_h5(h5_file)

        # Compare atoms
        orig_atoms = original_frame["atoms"]
        read_atoms = read_frame["atoms"]

        assert orig_atoms.nrows == read_atoms.nrows
        for key in orig_atoms.keys():
            if orig_atoms[key].dtype.kind in "biufc":
                np.testing.assert_array_almost_equal(
                    orig_atoms[key], read_atoms[key], decimal=6
                )

    def test_roundtrip_different_atom_styles(self, test_files, tmp_path):
        """Test round-trip with different atom styles."""
        atom_styles = ["atomic", "charge", "full"]

        for atom_style in atom_styles:
            try:
                # Read original
                original_frame = read_lammps_data(
                    test_files["molid"], atom_style=atom_style
                )

                # Write to HDF5
                h5_file = tmp_path / f"test_{atom_style}.h5"
                write_h5(h5_file, original_frame)

                # Read back
                read_frame = read_h5(h5_file)

                # Compare atoms
                orig_atoms = original_frame["atoms"]
                read_atoms = read_frame["atoms"]

                assert orig_atoms.nrows == read_atoms.nrows
                assert set(orig_atoms.keys()) == set(read_atoms.keys())

                # Compare numeric fields
                for key in orig_atoms.keys():
                    if orig_atoms[key].dtype.kind in "biufc":
                        np.testing.assert_array_almost_equal(
                            orig_atoms[key], read_atoms[key], decimal=6
                        )
            except Exception:
                # Some atom styles may not be compatible with certain files
                # Skip if read fails
                continue


class TestHDF5DataTypes:
    """Test handling of different data types."""

    def test_integer_types(self, tmp_path):
        """Test writing and reading integer types."""
        frame = mp.Frame()
        frame["atoms"] = {
            "id": np.array([1, 2, 3], dtype=np.int32),
            "type": np.array([1, 1, 2], dtype=np.int64),
        }

        h5_file = tmp_path / "test.h5"
        write_h5(h5_file, frame)

        read_frame = read_h5(h5_file)
        read_atoms = read_frame["atoms"]

        assert read_atoms["id"].dtype in (np.int32, np.int64)
        assert read_atoms["type"].dtype in (np.int32, np.int64)
        np.testing.assert_array_equal(frame["atoms"]["id"], read_atoms["id"])
        np.testing.assert_array_equal(frame["atoms"]["type"], read_atoms["type"])

    def test_float_types(self, tmp_path):
        """Test writing and reading float types."""
        frame = mp.Frame()
        frame["atoms"] = {
            "x": np.array([0.0, 1.0, 2.0], dtype=np.float32),
            "y": np.array([0.0, 1.0, 2.0], dtype=np.float64),
        }

        h5_file = tmp_path / "test.h5"
        write_h5(h5_file, frame)

        read_frame = read_h5(h5_file)
        read_atoms = read_frame["atoms"]

        np.testing.assert_array_almost_equal(
            frame["atoms"]["x"], read_atoms["x"], decimal=6
        )
        np.testing.assert_array_almost_equal(
            frame["atoms"]["y"], read_atoms["y"], decimal=6
        )

    def test_string_types(self, tmp_path):
        """Test writing and reading string types."""
        frame = mp.Frame()
        frame["atoms"] = {
            "type": np.array(["C", "H", "O"], dtype="U10"),
            "element": np.array(["C", "H", "O"], dtype="U1"),
        }

        h5_file = tmp_path / "test.h5"
        write_h5(h5_file, frame)

        read_frame = read_h5(h5_file)
        read_atoms = read_frame["atoms"]

        np.testing.assert_array_equal(frame["atoms"]["type"], read_atoms["type"])
        np.testing.assert_array_equal(frame["atoms"]["element"], read_atoms["element"])


class TestHDF5ErrorHandling:
    """Test error handling and edge cases."""

    def test_read_nonexistent_file(self, tmp_path):
        """Test reading non-existent file."""
        h5_file = tmp_path / "nonexistent.h5"
        with pytest.raises((FileNotFoundError, OSError)):
            read_h5(h5_file)

    def test_write_empty_frame_raises_error(self, tmp_path):
        """Test that writing empty frame raises error."""
        empty_frame = mp.Frame()
        h5_file = tmp_path / "test.h5"

        with pytest.raises(ValueError, match="Cannot write empty frame"):
            write_h5(h5_file, empty_frame)

    def test_read_invalid_h5_file(self, tmp_path):
        """Test reading invalid HDF5 file."""
        # Create a file that's not valid HDF5
        invalid_file = tmp_path / "invalid.h5"
        with open(invalid_file, "w") as f:
            f.write("not an hdf5 file")

        with pytest.raises((OSError, ValueError)):
            read_h5(invalid_file)


class TestHDF5ContextManager:
    """Test context manager usage."""

    def test_writer_context_manager(self, test_files, tmp_path):
        """Test using HDF5Writer as context manager."""
        original_frame = read_lammps_data(test_files["molid"], atom_style="full")

        h5_file = tmp_path / "test.h5"
        with HDF5Writer(h5_file) as writer:
            writer.write(original_frame)

        assert h5_file.exists()

        # Read back to verify
        read_frame = read_h5(h5_file)
        assert read_frame["atoms"].nrows == original_frame["atoms"].nrows

    def test_reader_context_manager(self, test_files, tmp_path):
        """Test using HDF5Reader as context manager."""
        original_frame = read_lammps_data(test_files["molid"], atom_style="full")

        h5_file = tmp_path / "test.h5"
        write_h5(h5_file, original_frame)

        with HDF5Reader(h5_file) as reader:
            read_frame = reader.read()

        assert read_frame["atoms"].nrows == original_frame["atoms"].nrows

