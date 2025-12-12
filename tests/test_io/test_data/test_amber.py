import numpy as np
import molpy as mp


class TestAmberInpcrdReader:
    """Test AMBER coordinate file reader."""

    def test_amber_inpcrd_litfsi(self, TEST_DATA_DIR):
        """Test reading LiTFSI.inpcrd file."""

        frame = mp.Frame()
        result = mp.io.read_amber_inpcrd(
            inpcrd=TEST_DATA_DIR / "inpcrd/LiTFSI.inpcrd", frame=frame
        )

        assert isinstance(result, mp.Frame)
        assert "atoms" in result

        atoms = result["atoms"]
        n_atoms = atoms.nrows

        assert n_atoms == 16  # Based on the file content

        # Check coordinates - stored as separate x, y, z fields
        assert "x" in atoms
        assert "y" in atoms
        assert "z" in atoms
        
        # Combine into xyz for comparison
        xyz = np.column_stack([atoms["x"], atoms["y"], atoms["z"]])
        assert xyz.shape == (n_atoms, 3)
