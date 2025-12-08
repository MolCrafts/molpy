"""Unit tests for PDB writer focusing on required fields and None handling."""

import numpy as np
import pytest
from pathlib import Path

from molpy.core.frame import Frame, Block
from molpy.io.data.pdb import PDBWriter, PDBReader


class TestPDBWriterRequiredFields:
    """Test that PDB writer correctly handles required fields and None values."""

    def test_missing_required_field_x(self, tmp_path):
        """Test that missing 'x' field raises ValueError."""
        frame = Frame()
        atoms = Block(
            {
                "y": np.array([4.0, 5.0]),
                "z": np.array([7.0, 8.0]),
            }
        )
        frame["atoms"] = atoms

        writer = PDBWriter(tmp_path / "test.pdb")
        with pytest.raises(ValueError, match="Required field 'x' is missing"):
            writer.write(frame)

    def test_missing_required_field_y(self, tmp_path):
        """Test that missing 'y' field raises ValueError."""
        frame = Frame()
        atoms = Block(
            {
                "x": np.array([1.0, 2.0]),
                "z": np.array([7.0, 8.0]),
            }
        )
        frame["atoms"] = atoms

        writer = PDBWriter(tmp_path / "test.pdb")
        with pytest.raises(ValueError, match="Required field 'y' is missing"):
            writer.write(frame)

    def test_missing_required_field_z(self, tmp_path):
        """Test that missing 'z' field raises ValueError."""
        frame = Frame()
        atoms = Block(
            {
                "x": np.array([1.0, 2.0]),
                "y": np.array([4.0, 5.0]),
            }
        )
        frame["atoms"] = atoms

        writer = PDBWriter(tmp_path / "test.pdb")
        with pytest.raises(ValueError, match="Required field 'z' is missing"):
            writer.write(frame)

    def test_none_value_in_required_field(self, tmp_path):
        """Test that None value in required field raises ValueError."""
        frame = Frame()
        atoms = Block(
            {
                "x": np.array([None, 2.0], dtype=object),
                "y": np.array([4.0, 5.0]),
                "z": np.array([7.0, 8.0]),
            }
        )
        frame["atoms"] = atoms

        writer = PDBWriter(tmp_path / "test.pdb")
        with pytest.raises(ValueError, match="Required field 'x' contains None"):
            writer.write(frame)

    def test_valid_minimal_frame(self, tmp_path):
        """Test that minimal valid frame (only x, y, z) writes correctly."""
        frame = Frame()
        atoms = Block(
            {
                "x": np.array([1.0, 2.0, 3.0]),
                "y": np.array([4.0, 5.0, 6.0]),
                "z": np.array([7.0, 8.0, 9.0]),
            }
        )
        frame["atoms"] = atoms
        frame.metadata["elements"] = "C C H"

        writer = PDBWriter(tmp_path / "test.pdb")
        writer.write(frame)

        # Verify file was created and has correct structure
        assert (tmp_path / "test.pdb").exists()

        # Verify PDB file content directly (without reader)
        with open(tmp_path / "test.pdb") as f:
            lines = f.readlines()
            atom_lines = [l for l in lines if l.startswith("ATOM")]
            assert len(atom_lines) == 3

            # Check coordinates are correct
            for i, line in enumerate(atom_lines):
                x = float(line[30:38])
                y = float(line[38:46])
                z = float(line[46:54])
                assert abs(x - (1.0 + i)) < 0.001
                assert abs(y - (4.0 + i)) < 0.001
                assert abs(z - (7.0 + i)) < 0.001

    def test_elements_from_metadata(self, tmp_path):
        """Test that elements are correctly extracted from metadata."""
        frame = Frame()
        atoms = Block(
            {
                "x": np.array([1.0, 2.0, 3.0, 4.0]),
                "y": np.array([1.0, 2.0, 3.0, 4.0]),
                "z": np.array([1.0, 2.0, 3.0, 4.0]),
                "id": np.array([1, 2, 3, 4]),
            }
        )
        frame["atoms"] = atoms
        frame.metadata["elements"] = "C O N H"

        writer = PDBWriter(tmp_path / "test.pdb")
        writer.write(frame)

        # Check PDB file content
        with open(tmp_path / "test.pdb") as f:
            lines = f.readlines()
            atom_lines = [l for l in lines if l.startswith("ATOM")]
            assert len(atom_lines) == 4

            # Check element symbols (columns 77-78)
            elements = [line[76:78].strip() for line in atom_lines]
            assert elements == ["C", "O", "N", "H"]

    def test_elements_from_atom_data(self, tmp_path):
        """Test that elements are extracted from atom data if metadata not available."""
        frame = Frame()
        atoms = Block(
            {
                "x": np.array([1.0, 2.0]),
                "y": np.array([1.0, 2.0]),
                "z": np.array([1.0, 2.0]),
                "element": np.array(["C", "H"], dtype=object),
            }
        )
        frame["atoms"] = atoms

        writer = PDBWriter(tmp_path / "test.pdb")
        writer.write(frame)

        # Check elements in output
        with open(tmp_path / "test.pdb") as f:
            lines = f.readlines()
            atom_lines = [l for l in lines if l.startswith("ATOM")]
            elements = [line[76:78].strip() for line in atom_lines]
            assert elements == ["C", "H"]

    def test_optional_fields_none_ignored(self, tmp_path):
        """Test that None values in optional fields are ignored."""
        frame = Frame()
        atoms = Block(
            {
                "x": np.array([1.0, 2.0]),
                "y": np.array([1.0, 2.0]),
                "z": np.array([1.0, 2.0]),
                "name": np.array([None, "C"], dtype=object),  # None should be ignored
                "occupancy": np.array(
                    [None, 1.0], dtype=object
                ),  # None should use default
            }
        )
        frame["atoms"] = atoms
        frame.metadata["elements"] = "X C"

        writer = PDBWriter(tmp_path / "test.pdb")
        writer.write(frame)  # Should not raise error

        assert (tmp_path / "test.pdb").exists()

    def test_atom_ids_from_field(self, tmp_path):
        """Test that atom IDs are correctly used from id field."""
        frame = Frame()
        atoms = Block(
            {
                "x": np.array([1.0, 2.0]),
                "y": np.array([1.0, 2.0]),
                "z": np.array([1.0, 2.0]),
                "id": np.array([100, 200]),
            }
        )
        frame["atoms"] = atoms
        frame.metadata["elements"] = "C H"

        writer = PDBWriter(tmp_path / "test.pdb")
        writer.write(frame)

        # Check atom serial numbers (columns 7-11)
        with open(tmp_path / "test.pdb") as f:
            lines = f.readlines()
            atom_lines = [l for l in lines if l.startswith("ATOM")]
            serials = [int(line[6:11].strip()) for line in atom_lines]
            assert serials == [100, 200]

    def test_atom_ids_default_to_index(self, tmp_path):
        """Test that atom IDs default to index+1 if id field missing."""
        frame = Frame()
        atoms = Block(
            {
                "x": np.array([1.0, 2.0, 3.0]),
                "y": np.array([1.0, 2.0, 3.0]),
                "z": np.array([1.0, 2.0, 3.0]),
            }
        )
        frame["atoms"] = atoms
        frame.metadata["elements"] = "C C H"

        writer = PDBWriter(tmp_path / "test.pdb")
        writer.write(frame)

        # Check atom serial numbers default to 1, 2, 3
        with open(tmp_path / "test.pdb") as f:
            lines = f.readlines()
            atom_lines = [l for l in lines if l.startswith("ATOM")]
            serials = [int(line[6:11].strip()) for line in atom_lines]
            assert serials == [1, 2, 3]


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
