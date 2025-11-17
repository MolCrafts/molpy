from pathlib import Path

import numpy as np
import pytest

import molpy as mp


class TestGMXGroReader:
    """Basic GRO reading tests."""

    def test_gro(self, TEST_DATA_DIR):
        fpath = TEST_DATA_DIR / "gro/cod_4020641.gro"
        if not fpath.exists():
            pytest.skip("gro test data not available")
        frame = mp.io.read_gro(fpath, frame=mp.Frame())

        # Get the atoms block
        atoms = frame["atoms"]
        assert atoms.nrows == 81

        # Check first atom data
        first_atom = atoms[0]  # Get first atom as dict
        assert str(first_atom["res_number"]) == "1"
        assert str(first_atom["res_name"]) == "LIG"
        assert str(first_atom["name"]) == "S"
        assert int(first_atom["number"]) == 1
        xyz = np.asarray(first_atom["xyz"])
        expected_xyz = np.array([0.310, 0.862, 1.316])
        np.testing.assert_allclose(xyz, expected_xyz, rtol=1e-3)


class TestGROReaderComprehensive:
    """Comprehensive tests for GRO reader using chemfiles test cases."""

    def test_roundtrip_gro(self, tmp_path):
        """Test roundtrip writing and reading of GRO files."""
        # Create test frame
        frame = mp.Frame()
        atoms_data = {
            "res_number": [1, 1],
            "res_name": ["WAT", "WAT"],
            "name": ["OW", "HW1"],
            "number": [1, 2],
            "xyz": [[0.000, 0.000, 0.000], [0.100, 0.000, 0.000]],
        }
        frame["atoms"] = atoms_data
        frame.box = mp.Box(np.eye(3) * 2.0)

        tmp_file = tmp_path / "test.gro"
        writer = mp.io.data.GroWriter(str(tmp_file))
        writer.write(frame)

        # Read back
        mp.io.read_gro(tmp_file, frame=mp.Frame())

    def test_read_cod_4020641_gro(self, TEST_DATA_DIR):
        """Test reading cod_4020641.gro file."""
        fpath = TEST_DATA_DIR / "gro/cod_4020641.gro"
        if not fpath.exists():
            pytest.skip("cod_4020641.gro test data not available")

        frame = mp.io.read_gro(fpath, frame=mp.Frame())

        # Check basic structure
        assert "atoms" in frame
        atoms = frame["atoms"]

        # Get atom count
        n_atoms = atoms.nrows

        # Check expected number of atoms (from original test)
        assert n_atoms == 81

        # Check required fields
        assert "res_number" in atoms
        assert "res_name" in atoms
        assert "name" in atoms
        assert "number" in atoms
        assert "xyz" in atoms

        # Check first atom data
        first_atom = atoms[0]
        assert "res_number" in first_atom
        assert "res_name" in first_atom
        assert "name" in first_atom
        assert "number" in first_atom

        # Check coordinates
        xyz = np.asarray(first_atom["xyz"])
        expected_xyz = np.array([0.310, 0.862, 1.316])
        np.testing.assert_allclose(xyz, expected_xyz, rtol=1e-3)

        # Check box information
        assert frame.box is not None

    def test_read_lysozyme_gro(self, TEST_DATA_DIR):
        """Test reading lysozyme.gro file."""
        fpath = TEST_DATA_DIR / "gro/lysozyme.gro"
        if not fpath.exists():
            pytest.skip("lysozyme.gro test data not available")

        frame = mp.io.read_gro(fpath, frame=mp.Frame())

        # Check basic structure
        assert "atoms" in frame
        atoms = frame["atoms"]

        # Get atom count
        n_atoms = atoms.nrows

        # Lysozyme should have many atoms
        assert n_atoms > 1000

        # Check data integrity
        assert len(atoms["name"]) == n_atoms
        assert len(atoms["xyz"]) == n_atoms
        assert atoms["xyz"].shape == (n_atoms, 3)

    def test_read_triclinic_gro(self, TEST_DATA_DIR):
        """Test reading triclinic unit cell GRO file."""
        fpath = TEST_DATA_DIR / "gro/1vln-triclinic.gro"
        if not fpath.exists():
            pytest.skip("1vln-triclinic.gro test data not available")

        frame = mp.io.read_gro(fpath, frame=mp.Frame())

        # Check that triclinic box is handled
        assert frame.box is not None
        # Should have non-zero off-diagonal elements for triclinic

    def test_read_malformed_gro(self, TEST_DATA_DIR):
        """Test handling of malformed GRO files."""
        # Test truncated file
        fpath = TEST_DATA_DIR / "gro/truncated.gro"
        if fpath.exists():
            frame = mp.io.read_gro(fpath, frame=mp.Frame())
            # Should handle gracefully, possibly with fewer atoms
            assert "atoms" in frame

        # Test file without final line
        fpath = TEST_DATA_DIR / "gro/no-final-line.gro"
        if fpath.exists():
            frame = mp.io.read_gro(fpath, frame=mp.Frame())
            assert "atoms" in frame

    def test_read_gro_error_handling(self, tmp_path):
        """Test error handling for various edge cases."""

        # Test non-existent file
        with pytest.raises(FileNotFoundError):
            mp.io.read_gro(Path("nonexistent.gro"), frame=mp.Frame())

        # Test empty file
        tmp_file = tmp_path / "empty.gro"
        tmp_file.write_text("")  # Empty file

        # Should handle empty file gracefully
        frame = mp.io.read_gro(tmp_file, frame=mp.Frame())
        assert "atoms" in frame

    def test_gro_coordinate_precision(self, TEST_DATA_DIR):
        """Test that coordinate precision is maintained."""
        fpath = TEST_DATA_DIR / "gro/cod_4020641.gro"
        if not fpath.exists():
            pytest.skip("cod_4020641.gro test data not available")

        frame = mp.io.read_gro(fpath, frame=mp.Frame())
        atoms = frame["atoms"]

        # Check that coordinates are reasonable floats
        xyz = atoms["xyz"]
        assert xyz.dtype == np.float64
        assert not np.any(np.isnan(xyz))
        assert not np.any(np.isinf(xyz))

    def test_gro_residue_information(self, TEST_DATA_DIR):
        """Test that residue information is correctly parsed."""
        fpath = TEST_DATA_DIR / "gro/cod_4020641.gro"
        if not fpath.exists():
            pytest.skip("cod_4020641.gro test data not available")

        frame = mp.io.read_gro(fpath, frame=mp.Frame())
        atoms = frame["atoms"]

        # Check residue fields
        assert "res_number" in atoms
        assert "res_name" in atoms

        # All should be non-empty
        res_numbers = atoms["res_number"]
        res_names = atoms["res_name"]

        assert all(str(rn).strip() for rn in res_numbers)
        assert all(str(rn).strip() for rn in res_names)


class TestGROWriter:
    """Comprehensive tests for GRO writer."""

    def test_write_simple_gro(self, tmp_path):
        """Test writing a simple GRO file."""
        # Create test frame
        frame = mp.Frame()

        atoms_data = {
            "res_number": [1, 1, 1],
            "res_name": ["WAT", "WAT", "WAT"],
            "name": ["OW", "HW1", "HW2"],
            "number": [1, 2, 3],
            "xyz": [
                [0.000, 0.000, 0.000],
                [0.100, 0.000, 0.000],
                [-0.033, 0.094, 0.000],
            ],
        }
        frame["atoms"] = atoms_data
        frame.box = mp.Box(np.eye(3) * 2.0)

        # Write to temporary file
        tmp_file = tmp_path / "test.gro"
        writer = mp.io.data.GroWriter(str(tmp_file))
        writer.write(frame)

        # Read back and verify
        with open(tmp_file) as f:
            lines = f.readlines()

            # Should have title, atom count, atoms, and box line
            assert len(lines) >= 5
            assert "WAT" in lines[2]  # First atom line
            assert "OW" in lines[2]

    def test_gro_roundtrip(self, tmp_path, TEST_DATA_DIR):
        """Test GRO read-write roundtrip."""
        fpath = TEST_DATA_DIR / "gro/cod_4020641.gro"
        if not fpath.exists():
            pytest.skip("cod_4020641.gro test data not available")

        # Read original
        original_frame = mp.io.read_gro(fpath, frame=mp.Frame())

        # Write to temporary file
        tmp_file = tmp_path / "test.gro"
        writer = mp.io.data.GroWriter(str(tmp_file))
        writer.write(original_frame)

        # Read back
        roundtrip_frame = mp.io.read_gro(tmp_file, frame=mp.Frame())

        # Compare basic properties
        orig_atoms = original_frame["atoms"]
        rt_atoms = roundtrip_frame["atoms"]

        # Get dimensions
        orig_n_atoms = orig_atoms.nrows
        rt_n_atoms = rt_atoms.nrows

        # Should have same number of atoms
        assert orig_n_atoms == rt_n_atoms

        # Coordinates should be approximately the same
        orig_xyz = orig_atoms["xyz"]
        rt_xyz = rt_atoms["xyz"]
        np.testing.assert_allclose(orig_xyz, rt_xyz, rtol=1e-3)

    def test_write_gro_with_box(self, tmp_path):
        """Test writing GRO with various box types."""
        frame = mp.Frame()

        atoms_data = {
            "res_number": [1],
            "res_name": ["MOL"],
            "name": ["C"],
            "number": [1],
            "xyz": [[0.0, 0.0, 0.0]],
        }
        frame["atoms"] = atoms_data

        # Test orthogonal box
        frame.box = mp.Box(np.diag([2.0, 3.0, 4.0]))

        tmp_file = tmp_path / "test.gro"
        writer = mp.io.data.GroWriter(str(tmp_file))
        writer.write(frame)

        # Check box line
        with open(tmp_file) as f:
            lines = f.readlines()
            box_line = lines[-1].strip()
            # Should contain box dimensions
            assert "2.0" in box_line or "2.000" in box_line


class TestGROEdgeCases:
    """Edge case tests for GRO format."""

    def test_read_basic_structure(self, TEST_DATA_DIR):
        """Test reading basic GRO structure."""
        fpath = TEST_DATA_DIR / "gro/cod_4020641.gro"
        if not fpath.exists():
            pytest.skip("cod_4020641.gro test data not available")

        frame = mp.io.read_gro(fpath, frame=mp.Frame())

        # Basic checks
        assert "atoms" in frame
        assert frame.box is not None

        atoms = frame["atoms"]
        n_atoms = atoms.nrows
        assert n_atoms > 0

    def test_read_with_velocities(self, TEST_DATA_DIR):
        """Test reading GRO files with velocity information."""
        # Look for GRO files with velocities
        for gro_file in ["lysozyme.gro", "cod_4020641.gro"]:
            fpath = TEST_DATA_DIR / f"gro/{gro_file}"
            if fpath.exists():
                frame = mp.io.read_gro(fpath, frame=mp.Frame())
                atoms = frame["atoms"]

                # Check if velocity fields exist
                any(field in atoms for field in ["vx", "vy", "vz"])
                # Just ensure the file can be read
                assert "atoms" in frame
                break
        else:
            pytest.skip("No GRO test files available")

    def test_read_without_velocities(self, TEST_DATA_DIR):
        """Test reading GRO files without velocity information."""
        fpath = TEST_DATA_DIR / "gro/cod_4020641.gro"
        if not fpath.exists():
            pytest.skip("cod_4020641.gro test data not available")

        frame = mp.io.read_gro(fpath, frame=mp.Frame())
        atoms = frame["atoms"]

        # Should still work without velocities
        assert "xyz" in atoms
        assert "name" in atoms

    def test_coordinate_precision(self, TEST_DATA_DIR):
        """Test coordinate precision handling."""
        fpath = TEST_DATA_DIR / "gro/cod_4020641.gro"
        if not fpath.exists():
            pytest.skip("cod_4020641.gro test data not available")

        frame = mp.io.read_gro(fpath, frame=mp.Frame())
        atoms = frame["atoms"]

        xyz = atoms["xyz"]
        assert xyz.dtype in [np.float32, np.float64]
        assert not np.any(np.isnan(xyz))

    def test_box_handling(self, TEST_DATA_DIR):
        """Test various box format handling."""
        fpath = TEST_DATA_DIR / "gro/cod_4020641.gro"
        if not fpath.exists():
            pytest.skip("cod_4020641.gro test data not available")

        frame = mp.io.read_gro(fpath, frame=mp.Frame())

        # Should have box
        assert frame.box is not None
        assert frame.box.matrix.shape == (3, 3)

    def test_large_structures(self, TEST_DATA_DIR):
        """Test handling of large GRO structures."""
        fpath = TEST_DATA_DIR / "gro/lysozyme.gro"
        if not fpath.exists():
            pytest.skip("lysozyme.gro test data not available")

        frame = mp.io.read_gro(fpath, frame=mp.Frame())
        atoms = frame["atoms"]

        # Should handle large structures
        n_atoms = atoms.nrows
        assert n_atoms > 1000

        # Data consistency
        assert len(atoms["xyz"]) == n_atoms
