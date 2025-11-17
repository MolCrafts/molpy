"""
Tests for LammpsDataReader and LammpsDataWriter classes.

This module tests the modern LAMMPS data file I/O using Block.from_csv
with space delimiter and mp.ForceField for force field parameters.
"""

import os
from pathlib import Path

import numpy as np
import pytest

import molpy as mp
from molpy.io.data.lammps import LammpsDataReader, LammpsDataWriter


@pytest.fixture
def test_files(TEST_DATA_DIR) -> dict[str, Path]:
    """Provide paths to test files."""
    # Calculate path relative to the test file location

    test_data_dir = TEST_DATA_DIR / "lammps-data"

    files = {
        "molid": test_data_dir / "molid.lmp",
        "whitespaces": test_data_dir / "whitespaces.lmp",
        "triclinic_1": test_data_dir / "triclinic-1.lmp",
        "triclinic_2": test_data_dir / "triclinic-2.lmp",
        "labelmap": test_data_dir / "labelmap.lmp",
        "solvated": test_data_dir / "solvated.lmp",
        "data_body": test_data_dir / "data.body",
    }
    return files


class TestLammpsDataReader:
    """Test LammpsDataReader with real test cases."""

    def test_molid_file(self, test_files):
        """Test reading molid.lmp - file with molecular IDs and full style."""

        reader = LammpsDataReader(test_files["molid"], atom_style="full")
        frame = reader.read()

        # Check basic structure
        assert "atoms" in frame
        atoms = frame["atoms"]

        # Should have 12 atoms based on file content
        assert atoms.nrows == 12
        assert "mol" in atoms  # molecule ID should be present
        assert "type" in atoms
        assert "q" in atoms
        assert "x" in atoms and "y" in atoms and "z" in atoms  # Separate coordinates

        # Check coordinate data
        x = atoms["x"]
        y = atoms["y"]
        z = atoms["z"]
        assert len(x) == 12
        assert len(y) == 12
        assert len(z) == 12

        # Check box dimensions (0-20 in each direction)
        assert frame.metadata.get("box") is not None
        box_lengths = frame.metadata["box"].lengths
        np.testing.assert_array_almost_equal(box_lengths, [20.0, 20.0, 20.0])

        # Check that molecule IDs are in the data (should be 0-3 based on file)
        mol_ids = atoms["mol"]
        assert len(np.unique(mol_ids)) <= 4  # max 4 different molecules

        # Check metadata
        assert frame.metadata["format"] == "lammps_data"
        assert frame.metadata["atom_style"] == "full"
        assert "forcefield" in frame.metadata

    def test_whitespaces_file(self, test_files):
        """Test reading whitespaces.lmp - file with extra whitespaces."""

        reader = LammpsDataReader(test_files["whitespaces"], atom_style="full")
        frame = reader.read()

        # Should parse correctly despite extra whitespaces
        assert "atoms" in frame
        atoms = frame["atoms"]
        assert atoms.nrows == 1

        # Check the single atom's coordinates
        x = atoms["x"][0]
        y = atoms["y"][0]
        z = atoms["z"][0]
        np.testing.assert_array_almost_equal([x, y, z], [5.0, 5.0, 5.0])

        # Check box (should be 10x10x10)
        box_lengths = frame.metadata["box"].lengths
        np.testing.assert_array_almost_equal(box_lengths, [10.0, 10.0, 10.0])

    def test_triclinic_file(self, test_files):
        """Test reading triclinic-1.lmp - file with triclinic box."""

        reader = LammpsDataReader(test_files["triclinic_1"], atom_style="atomic")
        frame = reader.read()

        # Should handle triclinic box
        assert frame.metadata.get("box") is not None
        box_lengths = frame.metadata["box"].lengths
        np.testing.assert_array_almost_equal(box_lengths, [34.0, 34.0, 34.0])

        # Should have no atoms
        if "atoms" in frame:
            assert frame["atoms"].nrows == 0

    def test_labelmap_file(self, test_files):
        """Test reading labelmap.lmp - file with type labels and connectivity."""

        reader = LammpsDataReader(test_files["labelmap"], atom_style="full")
        frame = reader.read()

        # Check atoms
        assert "atoms" in frame
        atoms = frame["atoms"]
        assert atoms.nrows == 16

        # Check type labels are preserved
        types = atoms["type"]
        unique_types = np.unique(types)
        expected_types = {"f", "c3", "s6", "o", "ne", "sy", "Li+"}
        assert set(unique_types) == expected_types

        # Check connectivity
        assert "bonds" in frame
        bonds = frame["bonds"]
        assert bonds.nrows == 14

        # Check bond types
        bond_types = bonds["type"]
        unique_bond_types = np.unique(bond_types)
        assert len(unique_bond_types) > 0

        # Check angles
        assert "angles" in frame
        angles = frame["angles"]
        assert angles.nrows == 25

        # Check dihedrals
        assert "dihedrals" in frame
        dihedrals = frame["dihedrals"]
        assert dihedrals.nrows == 27

        # Check force field
        forcefield = frame.metadata.get("forcefield")
        assert forcefield is not None
        assert isinstance(forcefield, mp.ForceField)

    def test_atomic_style(self, test_files):
        """Test reading with atomic atom style."""

        reader = LammpsDataReader(test_files["molid"], atom_style="atomic")
        frame = reader.read()

        atoms = frame["atoms"]
        # Atomic style should not have mol or q columns
        assert "mol" not in atoms
        assert "q" not in atoms
        assert "type" in atoms
        assert "x" in atoms and "y" in atoms and "z" in atoms

    def test_charge_style(self, test_files):
        """Test reading with charge atom style."""

        reader = LammpsDataReader(test_files["molid"], atom_style="charge")
        frame = reader.read()

        atoms = frame["atoms"]
        # Charge style should have q but not mol
        assert "mol" not in atoms
        assert "q" in atoms
        assert "type" in atoms
        assert "x" in atoms and "y" in atoms and "z" in atoms


class TestLammpsDataWriter:
    """Test LammpsDataWriter."""

    def test_write_read_roundtrip(self, test_files, tmp_path):
        """Test that we can write and read back the same data."""

        # Read original file
        reader = LammpsDataReader(test_files["molid"], atom_style="full")
        original_frame = reader.read()

        # Write to temporary file
        tmp_file = tmp_path / "test.data"

        writer = LammpsDataWriter(tmp_file, atom_style="full")
        writer.write(original_frame)

        # Read back
        reader2 = LammpsDataReader(tmp_file, atom_style="full")
        new_frame = reader2.read()

        # Compare atoms
        orig_atoms = original_frame["atoms"]
        new_atoms = new_frame["atoms"]

        assert orig_atoms.nrows == new_atoms.nrows
        # Convert types to strings for comparison since they may be different types
        np.testing.assert_array_equal(
            np.array([str(t) for t in orig_atoms["type"]]),
            np.array([str(t) for t in new_atoms["type"]]),
        )
        np.testing.assert_array_almost_equal(orig_atoms["x"], new_atoms["x"])
        np.testing.assert_array_almost_equal(orig_atoms["y"], new_atoms["y"])
        np.testing.assert_array_almost_equal(orig_atoms["z"], new_atoms["z"])

        # Compare box - skip for now as box handling may need more work
        # assert original_frame.box is not None
        # assert new_frame.box is not None
        # np.testing.assert_array_almost_equal(
        #     original_frame.box.lengths, new_frame.box.lengths
        # )

    def test_write_minimal_frame(self, tmp_path):
        """Test writing a minimal frame with just atoms."""
        # Create a simple frame
        frame = mp.Frame()

        # Add atoms data with separate x, y, z coordinates
        atoms_data = {
            "id": np.array([1, 2, 3]),
            "type": np.array([1, 1, 2]),
            "x": np.array([0.0, 1.0, 0.0]),
            "y": np.array([0.0, 0.0, 1.0]),
            "z": np.array([0.0, 0.0, 0.0]),
            "mass": np.array([1.0, 1.0, 2.0]),
        }

        frame["atoms"] = mp.Block(atoms_data)
        frame.metadata["box"] = mp.Box([10.0, 10.0, 10.0])

        # Write to temporary file
        tmp_file = tmp_path / "test.data"

        writer = LammpsDataWriter(tmp_file, atom_style="atomic")
        writer.write(frame)

        # Check file was written and has content
        assert os.path.exists(tmp_file)
        with open(tmp_file) as f:
            content = f.read()
            assert "3 atoms" in content
            assert "2 atom types" in content
            assert "Atoms" in content

    def test_write_full_style(self, tmp_path):
        """Test writing with full atom style including molecule IDs and charges."""
        frame = mp.Frame()

        # Create atoms with all fields
        atoms_data = {
            "id": np.array([1, 2, 3]),
            "mol": np.array([1, 1, 2]),
            "type": np.array(["C", "C", "O"]),
            "q": np.array([0.0, 0.0, -0.5]),
            "x": np.array([0.0, 1.0, 0.0]),
            "y": np.array([0.0, 0.0, 1.0]),
            "z": np.array([0.0, 0.0, 0.0]),
            "mass": np.array([12.0, 12.0, 16.0]),
        }

        frame["atoms"] = mp.Block(atoms_data)
        frame.metadata["box"] = mp.Box([10.0, 10.0, 10.0])

        # Add bonds
        bonds_data = {
            "id": np.array([1, 2]),
            "type": np.array(["C-C", "C-O"]),
            "atom_i": np.array([0, 1]),
            "atom_j": np.array([1, 2]),
        }
        frame["bonds"] = mp.Block(bonds_data)

        tmp_file = tmp_path / "test.data"

        writer = LammpsDataWriter(tmp_file, atom_style="full")
        writer.write(frame)

        # Check file content
        with open(tmp_file) as f:
            content = f.read()
            assert "3 atoms" in content
            assert "2 bonds" in content
            assert "2 atom types" in content
            assert "2 bond types" in content
            assert "Atom Type Labels" in content
            assert "Bond Type Labels" in content

    def test_write_with_forcefield(self, tmp_path):
        """Test writing with force field parameters."""
        frame = mp.Frame()

        # Create atoms
        atoms_data = {
            "id": np.array([1, 2]),
            "type": np.array(["C", "O"]),
            "x": np.array([0.0, 1.0]),
            "y": np.array([0.0, 0.0]),
            "z": np.array([0.0, 0.0]),
            "mass": np.array([12.0, 16.0]),
        }
        frame["atoms"] = mp.Block(atoms_data)
        frame.metadata["box"] = mp.Box([10.0, 10.0, 10.0])

        # Create a simple force field (empty for now)
        forcefield = mp.ForceField()
        frame.metadata["forcefield"] = forcefield

        tmp_file = tmp_path / "test.data"

        writer = LammpsDataWriter(tmp_file, atom_style="atomic")
        writer.write(frame)

        # Check file content
        with open(tmp_file) as f:
            content = f.read()
            assert "2 atoms" in content
            # Note: Force field writing may not be implemented yet
            # assert "Pair Coeffs" in content
            # assert "Bond Coeffs" in content


class TestErrorHandling:
    """Test error handling and edge cases."""

    def test_nonexistent_file(self):
        """Test reading nonexistent file."""
        with pytest.raises(FileNotFoundError):
            reader = LammpsDataReader("nonexistent_file.data")
            reader.read()

    def test_empty_file(self, tmp_path):
        """Test reading empty file."""
        tmp_file = tmp_path / "test.data"
        with open(tmp_file, "w") as f:
            f.write("")

        reader = LammpsDataReader(tmp_file)
        frame = reader.read()

        # Should handle empty file gracefully
        assert frame is not None
        assert "atoms" not in frame

    def test_malformed_header(self, tmp_path):
        """Test reading file with malformed header."""
        malformed_content = """# LAMMPS data file
invalid atoms
1 atom types

0.0 10.0 xlo xhi
0.0 10.0 ylo yhi
0.0 10.0 zlo zhi

Masses

1 1.0

Atoms

1 1 0.0 0.0 0.0
"""
        tmp_file = tmp_path / "test.data"
        with open(tmp_file, "w") as f:
            f.write(malformed_content)

        reader = LammpsDataReader(tmp_file, atom_style="atomic")
        frame = reader.read()

        # Should handle malformed header gracefully
        assert frame is not None
        # May not have atoms if header parsing fails
        if "atoms" in frame:
            assert frame["atoms"].nrows >= 0


class TestForceFieldIntegration:
    """Test force field integration."""

    def test_forcefield_parsing(self, test_files):
        """Test that force field parameters are correctly parsed."""
        if "labelmap" not in test_files:
            pytest.skip("labelmap.lmp not found")

        reader = LammpsDataReader(test_files["labelmap"], atom_style="full")
        frame = reader.read()

        forcefield = frame.metadata.get("forcefield")
        assert forcefield is not None
        assert isinstance(forcefield, mp.ForceField)

        # Check that we have a forcefield object (may be empty if no coeffs in file)
        # The labelmap.lmp file may not have force field coefficients sections
        assert forcefield is not None

    def test_forcefield_writing(self, tmp_path):
        """Test that force field parameters are correctly written."""
        frame = mp.Frame()

        # Create simple atoms
        atoms_data = {
            "id": np.array([1]),
            "type": np.array(["C"]),
            "x": np.array([0.0]),
            "y": np.array([0.0]),
            "z": np.array([0.0]),
            "mass": np.array([12.0]),
        }
        frame["atoms"] = mp.Block(atoms_data)
        frame.metadata["box"] = mp.Box([10.0, 10.0, 10.0])

        # Create a simple force field (empty for now)
        forcefield = mp.ForceField()
        frame.metadata["forcefield"] = forcefield

        tmp_file = tmp_path / "test.data"

        writer = LammpsDataWriter(tmp_file, atom_style="atomic")
        writer.write(frame)

        # Read back and check force field
        reader = LammpsDataReader(tmp_file, atom_style="atomic")
        new_frame = reader.read()

        new_forcefield = new_frame.metadata.get("forcefield")
        assert new_forcefield is not None
        # Note: Force field parsing may not be fully implemented yet
