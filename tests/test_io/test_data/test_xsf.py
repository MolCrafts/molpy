"""Core XSF format tests - simplified version with only essential functionality."""

import numpy as np
import pytest

import molpy as mp


class TestXSFCore:
    """Core XSF functionality tests."""

    def test_read_crystal_structure(self, tmp_path):
        """Test reading a crystal structure with periodic box."""
        tmp_file = tmp_path / "test.xsf"
        tmp_file.write_text(
            "CRYSTAL\n"
            "PRIMVEC\n"
            "3.0 0.0 0.0\n"
            "0.0 3.0 0.0\n"
            "0.0 0.0 3.0\n"
            "PRIMCOORD\n"
            "2 1\n"
            "1  0.0  0.0  0.0\n"
            "8  1.5  1.5  1.5\n"
        )

        frame = mp.io.read_xsf(tmp_file)

        # Check atoms
        assert len(frame["atoms"]["atomic_number"]) == 2
        assert frame["atoms"]["atomic_number"][0] == 1  # Hydrogen
        assert frame["atoms"]["atomic_number"][1] == 8  # Oxygen

        # Check box
        assert frame.metadata["box"].style == mp.Box.Style.ORTHOGONAL
        np.testing.assert_array_almost_equal(
            frame.metadata["box"].matrix, np.diag([3.0, 3.0, 3.0])
        )

    def test_read_molecule_structure(self, tmp_path):
        """Test reading a molecule structure (non-periodic)."""
        tmp_file = tmp_path / "test.xsf"
        tmp_file.write_text(
            "MOLECULE\nPRIMCOORD\n2 1\n1  0.0  0.0  0.0\n1  1.0  0.0  0.0\n"
        )

        frame = mp.io.read_xsf(tmp_file)

        # Check atoms
        assert len(frame["atoms"]["atomic_number"]) == 2
        assert all(an == 1 for an in frame["atoms"]["atomic_number"])

        # Should have a free box for molecule (non-periodic)
        assert frame.metadata["box"].style == mp.Box.Style.FREE

    def test_write_crystal_structure(self, tmp_path):
        """Test writing a crystal structure."""
        # Create test system
        frame = mp.Frame()
        frame["atoms"] = mp.Block(
            {
                "atomic_number": np.array([1, 8]),
                "xyz": np.array([[0.0, 0.0, 0.0], [1.5, 1.5, 1.5]]),
                "element": np.array(["H", "O"]),
                "x": np.array([0.0, 1.5]),
                "y": np.array([0.0, 1.5]),
                "z": np.array([0.0, 1.5]),
            }
        )

        box = mp.Box(matrix=np.diag([3.0, 3.0, 3.0]))
        frame.metadata["box"] = box

        # Write to file
        tmp_file = tmp_path / "test.xsf"
        mp.io.write_xsf(str(tmp_file), frame)

        # Read back and verify
        frame2 = mp.io.read_xsf(tmp_file)

        # Check atoms
        assert len(frame2["atoms"]["atomic_number"]) == 2
        np.testing.assert_array_equal(frame2["atoms"]["atomic_number"], [1, 8])

        # Check box
        assert frame2.metadata["box"].style == mp.Box.Style.ORTHOGONAL
        np.testing.assert_array_almost_equal(
            frame2.metadata["box"].matrix, np.diag([3.0, 3.0, 3.0])
        )

    def test_write_molecule_structure(self, tmp_path):
        """Test writing a molecule structure."""
        # Create test system
        frame = mp.Frame()
        frame["atoms"] = mp.Block(
            {
                "atomic_number": np.array([1, 1]),
                "xyz": np.array([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0]]),
                "element": np.array(["H", "H"]),
                "x": np.array([0.0, 1.0]),
                "y": np.array([0.0, 0.0]),
                "z": np.array([0.0, 0.0]),
            }
        )

        # Create with free box
        box = mp.Box()  # Free box
        frame.metadata["box"] = box

        # Write to file
        tmp_file = tmp_path / "test.xsf"
        mp.io.write_xsf(str(tmp_file), frame)

        # Read back and verify
        frame2 = mp.io.read_xsf(tmp_file)

        # Check atoms
        assert len(frame2["atoms"]["atomic_number"]) == 2
        assert all(an == 1 for an in frame2["atoms"]["atomic_number"])

        # Should have free box
        assert frame2.metadata["box"].style == mp.Box.Style.FREE

    def test_roundtrip_consistency(self, tmp_path):
        """Test that write->read maintains data consistency."""
        # Original system
        frame = mp.Frame()
        frame["atoms"] = mp.Block(
            {
                "atomic_number": np.array([6, 1, 1, 1, 1]),
                "xyz": np.array(
                    [
                        [0.0, 0.0, 0.0],
                        [1.0, 0.0, 0.0],
                        [-1.0, 0.0, 0.0],
                        [0.0, 1.0, 0.0],
                        [0.0, -1.0, 0.0],
                    ]
                ),
                "element": np.array(["C", "H", "H", "H", "H"]),
                "x": np.array([0.0, 1.0, -1.0, 0.0, 0.0]),
                "y": np.array([0.0, 0.0, 0.0, 1.0, -1.0]),
                "z": np.array([0.0, 0.0, 0.0, 0.0, 0.0]),
            }
        )

        box = mp.Box(matrix=[[5.0, 0.0, 0.0], [0.0, 5.0, 0.0], [0.0, 0.0, 5.0]])
        frame.metadata["box"] = box

        # Write and read back
        tmp_file = tmp_path / "test.xsf"
        mp.io.write_xsf(str(tmp_file), frame)
        frame2 = mp.io.read_xsf(tmp_file)

        # Check consistency
        np.testing.assert_array_equal(
            frame["atoms"]["atomic_number"], frame2["atoms"]["atomic_number"]
        )
        np.testing.assert_array_almost_equal(
            frame["atoms"]["xyz"], frame2["atoms"]["xyz"]
        )
        np.testing.assert_array_almost_equal(
            frame.metadata["box"].matrix, frame2.metadata["box"].matrix
        )

    def test_error_handling(self, tmp_path):
        """Test basic error handling."""
        # Test reading non-existent file
        with pytest.raises(FileNotFoundError):
            mp.io.read_xsf("nonexistent.xsf")

        # Test reading empty file
        tmp_file = tmp_path / "empty.xsf"
        tmp_file.write_text("")  # Empty file

        with pytest.raises(ValueError, match="Empty XSF file"):
            mp.io.read_xsf(tmp_file)

        # Test malformed PRIMCOORD section
        tmp_file2 = tmp_path / "malformed.xsf"
        tmp_file2.write_text(
            "MOLECULE\nPRIMCOORD\ninvalid_number 1\n"  # Invalid atom count
        )

        with pytest.raises(ValueError):
            mp.io.read_xsf(tmp_file2)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
