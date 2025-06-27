import molpy as mp
import pytest
import tempfile
from pathlib import Path
import numpy as np

class TestAmberInpcrdReader:
    """Test AMBER coordinate file reader."""

    def test_amber_inpcrd_litfsi(self, TEST_DATA_DIR):
        """Test reading LiTFSI.inpcrd file."""

        frame = mp.Frame()
        result = mp.io.read_amber(
            prmtop=TEST_DATA_DIR / "forcefield/amber/LiTFSI.prmtop",
            inpcrd=TEST_DATA_DIR / "data/inpcrd/LiTFSI.inpcrd",
            frame=frame
        )
        
        assert isinstance(result, mp.Frame)
        assert "atoms" in result
        
        atoms = result["atoms"]
        n_atoms = atoms.nrows

        assert n_atoms == 16  # Based on the file content
        
        xyz = atoms["xyz"]
        assert xyz.shape == (n_atoms, 3)
        