"""
Tests for LAMMPS molecule file I/O functionality.

This module tests both native and JSON format molecule files
for reading and writing capabilities.
"""

import json
from pathlib import Path

import numpy as np
import pytest

import molpy as mp
from molpy.io.data.lammps_molecule import LammpsMoleculeReader, LammpsMoleculeWriter


@pytest.fixture
def test_files():
    """Get paths to test files."""
    test_dir = Path(__file__).parent / "test_files"
    return {
        "water_native": test_dir / "water_tip3p.mol",
        "water_json": test_dir / "water_tip3p.json",
        "ethane_native": test_dir / "ethane.mol",
    }


class TestLammpsMoleculeReader:
    """Test LAMMPS molecule file reading functionality."""

    def test_read_native_water(self, test_files):
        """Test reading native format water molecule."""
        reader = LammpsMoleculeReader(test_files["water_native"])
        frame = reader.read()

        # Check metadata
        assert frame.metadata["format"] == "lammps_molecule"
        assert frame.metadata["source_format"] == "native"
        assert frame.metadata["n_atoms"] == 3
        assert frame.metadata["n_bonds"] == 2
        assert frame.metadata["n_angles"] == 1

        # Check atoms
        assert "atoms" in frame
        atoms = frame["atoms"]
        assert atoms.nrows == 3

        # Check atom properties
        assert "id" in atoms
        assert "type" in atoms
        assert "x" in atoms and "y" in atoms and "z" in atoms
        assert "q" in atoms

        # Check specific values
        np.testing.assert_array_equal(atoms["id"], [1, 2, 3])
        np.testing.assert_array_equal(atoms["type"], ["1", "2", "2"])
        np.testing.assert_array_almost_equal(atoms["q"], [-0.834, 0.417, 0.417])

        # Check coordinates
        expected_coords = np.array(
            [
                [0.00000, -0.06556, 0.00000],
                [0.75695, 0.52032, 0.00000],
                [-0.75695, 0.52032, 0.00000],
            ]
        )
        coords = np.column_stack([atoms["x"], atoms["y"], atoms["z"]])
        np.testing.assert_array_almost_equal(coords, expected_coords)

        # Check bonds
        assert "bonds" in frame
        bonds = frame["bonds"]
        assert bonds.nrows == 2
        np.testing.assert_array_equal(bonds["type"], ["1", "1"])
        np.testing.assert_array_equal(bonds["atom1"], [1, 1])
        np.testing.assert_array_equal(bonds["atom2"], [2, 3])

        # Check angles
        assert "angles" in frame
        angles = frame["angles"]
        assert angles.nrows == 1
        np.testing.assert_array_equal(angles["type"], ["1"])
        np.testing.assert_array_equal(angles["atom1"], [2])
        np.testing.assert_array_equal(angles["atom2"], [1])
        np.testing.assert_array_equal(angles["atom3"], [3])

    def test_read_json_water(self, test_files):
        """Test reading JSON format water molecule."""
        reader = LammpsMoleculeReader(test_files["water_json"])
        frame = reader.read()

        # Check metadata
        assert frame.metadata["format"] == "lammps_molecule"
        assert frame.metadata["source_format"] == "json"
        assert frame.metadata["title"] == "Water molecule. TIP3P geometry"
        assert frame.metadata["units"] == "real"
        assert frame.metadata["revision"] == 1

        # Check atoms
        assert "atoms" in frame
        atoms = frame["atoms"]
        assert atoms.nrows == 3

        # Check atom properties
        assert "id" in atoms
        assert "type" in atoms
        assert "x" in atoms and "y" in atoms and "z" in atoms
        assert "q" in atoms

        # Check specific values
        np.testing.assert_array_equal(atoms["id"], [1, 2, 3])
        np.testing.assert_array_equal(atoms["type"], ["OW", "HO1", "HO1"])
        np.testing.assert_array_almost_equal(atoms["q"], [-0.834, 0.417, 0.417])

        # Check coordinates
        expected_coords = np.array(
            [
                [0.00000, -0.06556, 0.00000],
                [0.75695, 0.52032, 0.00000],
                [-0.75695, 0.52032, 0.00000],
            ]
        )
        coords = np.column_stack([atoms["x"], atoms["y"], atoms["z"]])
        np.testing.assert_array_almost_equal(coords, expected_coords)

        # Check bonds
        assert "bonds" in frame
        bonds = frame["bonds"]
        assert bonds.nrows == 2
        np.testing.assert_array_equal(bonds["type"], ["OW-HO1", "OW-HO1"])
        np.testing.assert_array_equal(bonds["atom1"], [1, 1])
        np.testing.assert_array_equal(bonds["atom2"], [2, 3])

        # Check angles
        assert "angles" in frame
        angles = frame["angles"]
        assert angles.nrows == 1
        np.testing.assert_array_equal(angles["type"], ["HO1-OW-HO1"])
        np.testing.assert_array_equal(angles["atom1"], [2])
        np.testing.assert_array_equal(angles["atom2"], [1])
        np.testing.assert_array_equal(angles["atom3"], [3])

    def test_read_ethane_native(self, test_files):
        """Test reading more complex native format ethane molecule."""
        reader = LammpsMoleculeReader(test_files["ethane_native"])
        frame = reader.read()

        # Check metadata
        assert frame.metadata["format"] == "lammps_molecule"
        assert frame.metadata["source_format"] == "native"
        assert frame.metadata["n_atoms"] == 8
        assert frame.metadata["n_bonds"] == 7
        assert frame.metadata["n_angles"] == 12
        assert frame.metadata["n_dihedrals"] == 9

        # Check atoms
        assert "atoms" in frame
        atoms = frame["atoms"]
        assert atoms.nrows == 8

        # Check all atom properties are present
        assert "id" in atoms
        assert "type" in atoms
        assert "x" in atoms and "y" in atoms and "z" in atoms
        assert "q" in atoms
        assert "mass" in atoms
        assert "mol" in atoms

        # Check masses
        expected_masses = np.array(
            [12.011, 12.011, 1.008, 1.008, 1.008, 1.008, 1.008, 1.008]
        )
        np.testing.assert_array_almost_equal(atoms["mass"], expected_masses)

        # Check molecule IDs (all should be 1)
        np.testing.assert_array_equal(atoms["mol"], np.ones(8, dtype=int))

        # Check connectivity
        assert "bonds" in frame
        assert "angles" in frame
        assert "dihedrals" in frame

        bonds = frame["bonds"]
        angles = frame["angles"]
        dihedrals = frame["dihedrals"]

        assert bonds.nrows == 7
        assert angles.nrows == 12
        assert dihedrals.nrows == 9

    def test_empty_file_error(self, tmp_path):
        """Test error handling for empty files."""
        tmp_file = tmp_path / "test.mol"
        with open(tmp_file, "w") as f:
            f.write("")
        reader = LammpsMoleculeReader(tmp_file)
        with pytest.raises(ValueError, match="Empty molecule file"):
            reader.read()

    def test_nonexistent_file_error(self):
        """Test error handling for nonexistent files."""
        reader = LammpsMoleculeReader("nonexistent_file.mol")
        with pytest.raises(FileNotFoundError):
            reader.read()

    def test_invalid_json_format(self, tmp_path):
        """Test error handling for invalid JSON format."""
        tmp_file = tmp_path / "test.json"
        with open(tmp_file, "w") as f:
            json.dump({"format": "invalid"}, f)
        reader = LammpsMoleculeReader(tmp_file)
        with pytest.raises(ValueError, match="JSON file must have format='molecule'"):
            reader.read()

    def test_missing_types_section_json(self, tmp_path):
        """Test error handling for missing types section in JSON."""
        tmp_file = tmp_path / "test.json"
        with open(tmp_file, "w") as f:
            json.dump({"application": "LAMMPS", "format": "molecule", "revision": 1}, f)
        reader = LammpsMoleculeReader(tmp_file)
        with pytest.raises(
            ValueError, match="JSON molecule file must contain 'types' section"
        ):
            reader.read()

    def test_missing_types_section_native(self, tmp_path):
        """Test error handling for missing Types section in native format."""
        tmp_file = tmp_path / "test.mol"
        with open(tmp_file, "w") as f:
            f.write("# Test molecule\n")
            f.write("1 atoms\n")
            f.write("\n")
            f.write("Coords\n")
            f.write("\n")
            f.write("1 0.0 0.0 0.0\n")
        reader = LammpsMoleculeReader(tmp_file)
        with pytest.raises(
            ValueError, match="Native molecule file must contain Types section"
        ):
            reader.read()


class TestLammpsMoleculeWriter:
    """Test LAMMPS molecule file writing functionality."""

    def test_write_native_format(self, test_files, tmp_path):
        """Test writing in native format."""
        # Read a molecule first
        reader = LammpsMoleculeReader(test_files["water_native"])
        frame = reader.read()

        # Write to temporary file
        tmp_file = tmp_path / "test.mol"
        writer = LammpsMoleculeWriter(tmp_file, format_type="native")
        writer.write(frame)

        # Read back and compare
        reader2 = LammpsMoleculeReader(tmp_file)
        frame2 = reader2.read()

        # Compare key properties
        assert frame2["atoms"].nrows == frame["atoms"].nrows
        assert frame2["bonds"].nrows == frame["bonds"].nrows
        assert frame2["angles"].nrows == frame["angles"].nrows

        # Check atom types and charges
        np.testing.assert_array_equal(frame2["atoms"]["type"], frame["atoms"]["type"])
        np.testing.assert_array_almost_equal(frame2["atoms"]["q"], frame["atoms"]["q"])

    def test_write_json_format(self, test_files, tmp_path):
        """Test writing in JSON format."""
        # Read a molecule first
        reader = LammpsMoleculeReader(test_files["water_json"])
        frame = reader.read()

        # Write to temporary file
        tmp_file = tmp_path / "test.json"
        writer = LammpsMoleculeWriter(tmp_file, format_type="json")
        writer.write(frame)

        # Read back and compare
        reader2 = LammpsMoleculeReader(tmp_file)
        frame2 = reader2.read()

        # Compare key properties
        assert frame2["atoms"].nrows == frame["atoms"].nrows
        assert frame2["bonds"].nrows == frame["bonds"].nrows
        assert frame2["angles"].nrows == frame["angles"].nrows

        # Check atom types and charges
        np.testing.assert_array_equal(frame2["atoms"]["type"], frame["atoms"]["type"])
        np.testing.assert_array_almost_equal(frame2["atoms"]["q"], frame["atoms"]["q"])

    def test_roundtrip_native_to_json(self, test_files, tmp_path):
        """Test roundtrip conversion from native to JSON format."""
        # Read native format
        reader = LammpsMoleculeReader(test_files["ethane_native"])
        frame = reader.read()

        # Write as JSON
        temp_json_path = tmp_path / "test.json"

        # Write as native again
        temp_native_path = tmp_path / "test.mol"

        # Native -> JSON
        writer_json = LammpsMoleculeWriter(temp_json_path, format_type="json")
        writer_json.write(frame)

        # JSON -> Native
        reader_json = LammpsMoleculeReader(temp_json_path)
        frame_from_json = reader_json.read()

        writer_native = LammpsMoleculeWriter(temp_native_path, format_type="native")
        writer_native.write(frame_from_json)

        # Read final native and compare
        reader_final = LammpsMoleculeReader(temp_native_path)
        frame_final = reader_final.read()

        # Compare key properties
        assert frame_final["atoms"].nrows == frame["atoms"].nrows
        assert frame_final["bonds"].nrows == frame["bonds"].nrows
        assert frame_final["angles"].nrows == frame["angles"].nrows
        assert frame_final["dihedrals"].nrows == frame["dihedrals"].nrows

    def test_write_without_atoms_error(self, tmp_path):
        """Test error when trying to write frame without atoms."""
        frame = mp.Frame()

        tmp_file = tmp_path / "test.mol"
        writer = LammpsMoleculeWriter(tmp_file, format_type="native")
        with pytest.raises(ValueError, match="Frame must contain atoms data"):
            writer.write(frame)

    def test_invalid_format_type_error(self):
        """Test error for invalid format type."""
        with pytest.raises(ValueError, match="format_type must be 'native' or 'json'"):
            LammpsMoleculeWriter("test.mol", format_type="invalid")

    def test_auto_json_extension(self):
        """Test automatic .json extension for JSON format."""
        writer = LammpsMoleculeWriter("test.mol", format_type="json")
        assert writer._path.suffix == ".json"

    def test_metadata_preservation(self, test_files, tmp_path):
        """Test that metadata is preserved during write/read cycle."""
        # Read original
        reader = LammpsMoleculeReader(test_files["water_json"])
        frame = reader.read()

        # Add custom metadata
        frame.metadata["custom_field"] = 42
        frame.metadata["total_mass"] = 18.015
        frame.metadata["center_of_mass"] = np.array([0.0, 0.0, 0.0])
        frame.metadata["inertia"] = np.array([1.0, 2.0, 3.0, 0.0, 0.0, 0.0])

        # Write JSON format
        tmp_file = tmp_path / "test.json"
        writer = LammpsMoleculeWriter(tmp_file, format_type="json")
        writer.write(frame)

        # Read back
        reader2 = LammpsMoleculeReader(tmp_file)
        frame2 = reader2.read()

        # Check if metadata is preserved (at least the ones we wrote)
        # Note: custom_field won't be preserved in standard LAMMPS format
        assert "total_mass" in frame2.metadata or "masstotal" in frame2.metadata


class TestIntegrationWithMolpyIO:
    """Test integration with molpy.io module functions."""

    def test_read_lammps_molecule_function(self, test_files):
        """Test the high-level read_lammps_molecule function."""
        frame = mp.io.read_lammps_molecule(test_files["water_native"])

        assert "atoms" in frame
        assert frame["atoms"].nrows == 3
        assert frame.metadata["format"] == "lammps_molecule"

    def test_write_lammps_molecule_function(self, test_files, tmp_path):
        """Test the high-level write_lammps_molecule function."""
        # Read a molecule
        frame = mp.io.read_lammps_molecule(test_files["water_native"])

        # Write using high-level function
        tmp_file = tmp_path / "test.mol"
        mp.io.write_lammps_molecule(tmp_file, frame, format_type="native")

        # Read back and verify
        frame2 = mp.io.read_lammps_molecule(tmp_file)
        assert frame2["atoms"].nrows == frame["atoms"].nrows

    def test_write_json_format_function(self, test_files, tmp_path):
        """Test the high-level write function with JSON format."""
        # Read a molecule
        frame = mp.io.read_lammps_molecule(test_files["water_native"])

        # Write using high-level function
        tmp_file = tmp_path / "test.json"

        mp.io.write_lammps_molecule(tmp_file, frame, format_type="json")

        # Read back and verify
        frame2 = mp.io.read_lammps_molecule(tmp_file)
        assert frame2["atoms"].nrows == frame["atoms"].nrows
        assert frame2.metadata["source_format"] == "json"
