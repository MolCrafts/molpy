"""Unit tests for PDB writer focusing on required fields and None handling."""

import importlib
from pathlib import Path

import numpy as np
import pytest

from molpy.core.frame import Block, Frame


@pytest.fixture(
    params=["molpy.io.data.pdb"],
    ids=["molpy"],
)
def pdb_backend(request):
    return importlib.import_module(request.param)


class TestPDBWriterRequiredFields:
    """Test that PDB writer correctly handles required fields and None values."""

    @pytest.mark.parametrize("missing", ["x", "y", "z"])
    def test_missing_required_field(self, tmp_path, pdb_backend, missing):
        """A missing required coordinate field raises ValueError."""
        columns = {
            "x": np.array([1.0, 2.0]),
            "y": np.array([4.0, 5.0]),
            "z": np.array([7.0, 8.0]),
        }
        del columns[missing]
        frame = Frame()
        frame["atoms"] = Block(columns)

        writer = pdb_backend.PDBWriter(tmp_path / "test.pdb")
        with pytest.raises(ValueError, match=f"Required field '{missing}' is missing"):
            writer.write(frame)

    def test_none_value_in_required_field(self, tmp_path, pdb_backend):
        """A None-bearing column is rejected at Block construction.

        Under the numpy-only Store contract a column cannot hold ``None``, so
        the failure is fail-fast when the Block is built — earlier than (and
        superseding) the writer's own required-field check.
        """
        import molrs

        with pytest.raises(molrs.BlockDtypeError):
            Block(
                {
                    "x": np.array([None, 2.0]),
                    "y": np.array([4.0, 5.0]),
                    "z": np.array([7.0, 8.0]),
                }
            )

    def test_valid_minimal_frame(self, tmp_path, pdb_backend):
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

        writer = pdb_backend.PDBWriter(tmp_path / "test.pdb")
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

    def test_elements_from_metadata(self, tmp_path, pdb_backend):
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

        writer = pdb_backend.PDBWriter(tmp_path / "test.pdb")
        writer.write(frame)

        # Check PDB file content
        with open(tmp_path / "test.pdb") as f:
            lines = f.readlines()
            atom_lines = [l for l in lines if l.startswith("ATOM")]
            assert len(atom_lines) == 4

            # Check element symbols (columns 77-78)
            elements = [line[76:78].strip() for line in atom_lines]
            assert elements == ["C", "O", "N", "H"]

    def test_elements_from_atom_data(self, tmp_path, pdb_backend):
        """Test that elements are extracted from atom data if metadata not available."""
        frame = Frame()
        atoms = Block(
            {
                "x": np.array([1.0, 2.0]),
                "y": np.array([1.0, 2.0]),
                "z": np.array([1.0, 2.0]),
                "element": np.array(["C", "H"]),
            }
        )
        frame["atoms"] = atoms

        writer = pdb_backend.PDBWriter(tmp_path / "test.pdb")
        writer.write(frame)

        # Check elements in output
        with open(tmp_path / "test.pdb") as f:
            lines = f.readlines()
            atom_lines = [l for l in lines if l.startswith("ATOM")]
            elements = [line[76:78].strip() for line in atom_lines]
            assert elements == ["C", "H"]

    def test_optional_field_none_rejected_at_construction(self, tmp_path, pdb_backend):
        """None-bearing optional columns are rejected at Block construction.

        The numpy-only Store has no place for ``None`` — a sparse optional field
        must be expressed as a typed column (e.g. empty string / default value)
        rather than a None-bearing object array.
        """
        import molrs

        with pytest.raises(molrs.BlockDtypeError):
            Block(
                {
                    "x": np.array([1.0, 2.0]),
                    "y": np.array([1.0, 2.0]),
                    "z": np.array([1.0, 2.0]),
                    "occupancy": np.array([None, 1.0], dtype=object),
                }
            )

        # The valid form: a typed column with a real default writes fine.
        frame = Frame()
        frame["atoms"] = Block(
            {
                "x": np.array([1.0, 2.0]),
                "y": np.array([1.0, 2.0]),
                "z": np.array([1.0, 2.0]),
                "name": np.array(["X", "C"]),
                "occupancy": np.array([0.0, 1.0]),
            }
        )
        frame.metadata["elements"] = "X C"
        pdb_backend.PDBWriter(tmp_path / "test.pdb").write(frame)
        assert (tmp_path / "test.pdb").exists()

    def test_atom_ids_from_field(self, tmp_path, pdb_backend):
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

        writer = pdb_backend.PDBWriter(tmp_path / "test.pdb")
        writer.write(frame)

        # Check atom serial numbers (columns 7-11)
        with open(tmp_path / "test.pdb") as f:
            lines = f.readlines()
            atom_lines = [l for l in lines if l.startswith("ATOM")]
            serials = [int(line[6:11].strip()) for line in atom_lines]
            assert serials == [100, 200]

    def test_atom_ids_default_to_index(self, tmp_path, pdb_backend):
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

        writer = pdb_backend.PDBWriter(tmp_path / "test.pdb")
        writer.write(frame)

        # Check atom serial numbers default to 1, 2, 3
        with open(tmp_path / "test.pdb") as f:
            lines = f.readlines()
            atom_lines = [l for l in lines if l.startswith("ATOM")]
            serials = [int(line[6:11].strip()) for line in atom_lines]
            assert serials == [1, 2, 3]


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
