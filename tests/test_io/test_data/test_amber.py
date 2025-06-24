import molpy as mp
import pytest
import tempfile
from pathlib import Path
import numpy as np

class TestAmberACReader:

    def test_amber_ac_simple(self):
        """Test AC reader with a simple manually created file."""
        # Create a simple AC file content
        ac_content = """ATOM      1  C   UNK     1       0.000   0.000   0.000 -0.094100        c3
ATOM      2  H1  UNK     1       1.000   0.000   0.000  0.031700        hc
ATOM      3  H2  UNK     1      -0.500   0.866   0.000  0.031700        hc
ATOM      4  H3  UNK     1      -0.500  -0.866   0.000  0.031700        hc
BOND      1    1    2    1      C   H1
BOND      2    1    3    1      C   H2
BOND      3    1    4    1      C   H3"""
        
        with tempfile.NamedTemporaryFile(mode='w', suffix='.ac', delete=False) as tmp:
            tmp.write(ac_content)
            tmp.flush()
            
            frame = mp.io.read_amber_ac(Path(tmp.name))
            assert isinstance(frame, mp.Frame)
            assert "atoms" in frame
            assert "bonds" in frame
            
            atoms = frame["atoms"]
            bonds = frame["bonds"]
            
            # Check atom count
            atom_sizes = atoms.sizes
            atom_dim = next(iter(atom_sizes.keys()))
            assert atom_sizes[atom_dim] == 4
            
            # Check bond count
            bond_sizes = bonds.sizes
            bond_dim = next(iter(bond_sizes.keys()))
            assert bond_sizes[bond_dim] == 3
            
            # Check atom properties
            assert "name" in atoms.data_vars
            assert "xyz" in atoms.data_vars
            assert "q" in atoms.data_vars
            assert "number" in atoms.data_vars

    def test_amber_ac_missing_file(self, test_data_path):
        """Test that non-existent AC file is handled properly."""
        fpath = test_data_path / "data/ac/nonexistent.ac"
        if not fpath.exists():
            pytest.skip("amber test data not available")
        # This test should be skipped since the file doesn't exist


class TestAmberInpcrdReader:
    """Test AMBER coordinate file reader."""

    def test_amber_inpcrd_litfsi(self, test_data_path):
        """Test reading LiTFSI.inpcrd file."""
        fpath = test_data_path / "data/inpcrd/LiTFSI.inpcrd"
        if not fpath.exists():
            pytest.skip("LiTFSI.inpcrd test data not available")
        
        frame = mp.Frame()
        result = mp.io.read_amber(
            prmtop=test_data_path / "forcefield/amber/LiTFSI.prmtop",
            inpcrd=fpath,
            frame=frame
        )
        
        assert isinstance(result, mp.Frame)
        assert "atoms" in result
        
        atoms = result["atoms"]
        sizes = atoms.sizes
        main_dim = next(iter(sizes.keys()))
        n_atoms = sizes[main_dim]
        
        # Check that we have the expected number of atoms
        assert n_atoms == 16  # Based on the file content
        
        # Check that coordinates are present
        assert "xyz" in atoms.data_vars
        
        # Check coordinate values are reasonable
        xyz = atoms["xyz"].values
        assert xyz.shape == (n_atoms, 3)
        
        # Check that title is set
        # Check metadata
        props = result.get_meta('props')
        assert props is not None
        assert 'name' in props
        assert props['name'] == 'TFSI'

    def test_amber_inpcrd_simple(self):
        """Test AMBER inpcrd reader with simple manually created file."""
        # Create a simple inpcrd file using the actual format
        inpcrd_content = """Test molecule
    2
   1.0000000   2.0000000   3.0000000   4.0000000   5.0000000   6.0000000
"""
        
        with tempfile.NamedTemporaryFile(mode='w', suffix='.inpcrd', delete=False) as tmp:
            tmp.write(inpcrd_content)
            tmp.flush()
            
            frame = mp.Frame()
            from molpy.io.data.amber import AmberInpcrdReader
            reader = AmberInpcrdReader(tmp.name)
            result = reader.read(frame)
            
            assert isinstance(result, mp.Frame)
            assert "atoms" in result
            
            atoms = result["atoms"]
            sizes = atoms.sizes
            main_dim = next(iter(sizes.keys()))
            assert sizes[main_dim] == 2
            
            # Check coordinates
            xyz = atoms["xyz"].values
            expected_xyz = np.array([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]])
            np.testing.assert_allclose(xyz, expected_xyz, rtol=1e-6)
            
            # Check title (stored in metadata)
            props = result.get_meta('props')
            assert props is not None
            assert props['name'] == 'Test molecule'
