"""``mp.io.write_pdb``: required coordinates and the ``element`` column."""

import numpy as np
import pytest

from molpy import Block, Frame, MetaValue
import molpy as mp


class TestWritePdb:
    """Test that PDB writer correctly handles required fields and None values."""

    @pytest.mark.parametrize("missing", ["x", "y", "z"])
    def test_missing_required_field(self, tmp_path, missing):
        """A missing required coordinate column is refused by name."""
        columns = {
            "x": np.array([1.0, 2.0]),
            "y": np.array([4.0, 5.0]),
            "z": np.array([7.0, 8.0]),
        }
        del columns[missing]
        frame = Frame()
        frame["atoms"] = Block(columns)

        with pytest.raises(OSError, match=f"Missing '{missing}' column"):
            mp.io.write_pdb(tmp_path / "test.pdb", frame)

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
        frame.meta = {"elements": MetaValue("string", "C C H")}

        mp.io.write_pdb(tmp_path / "test.pdb", frame)

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

    def test_a_missing_element_is_written_as_x(self, tmp_path):
        """Without an ``element`` column the PDB element field is ``X``, never
        a guess from frame meta."""
        frame = Frame()
        frame["atoms"] = Block(
            {
                "x": np.array([1.0, 2.0]),
                "y": np.array([1.0, 2.0]),
                "z": np.array([1.0, 2.0]),
                "id": np.array([1, 2]),
            }
        )
        frame.meta = {"elements": MetaValue("string", "C O")}
        mp.io.write_pdb(tmp_path / "test.pdb", frame)
        atom_lines = [
            line
            for line in (tmp_path / "test.pdb").read_text().splitlines()
            if line.startswith("ATOM")
        ]
        assert [line[76:78].strip() for line in atom_lines] == ["X", "X"]

    def test_elements_from_atom_data(self, tmp_path):
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

        mp.io.write_pdb(tmp_path / "test.pdb", frame)

        # Check elements in output
        with open(tmp_path / "test.pdb") as f:
            lines = f.readlines()
            atom_lines = [l for l in lines if l.startswith("ATOM")]
            elements = [line[76:78].strip() for line in atom_lines]
            assert elements == ["C", "H"]

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
        frame.meta = {"elements": MetaValue("string", "C H")}

        mp.io.write_pdb(tmp_path / "test.pdb", frame)

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
        frame.meta = {"elements": MetaValue("string", "C C H")}

        mp.io.write_pdb(tmp_path / "test.pdb", frame)

        # Check atom serial numbers default to 1, 2, 3
        with open(tmp_path / "test.pdb") as f:
            lines = f.readlines()
            atom_lines = [l for l in lines if l.startswith("ATOM")]
            serials = [int(line[6:11].strip()) for line in atom_lines]
            assert serials == [1, 2, 3]
