"""
Modern tests for LAMMPS data I/O using chemfiles-testcases data.

Tests the LammpsDataReader and LammpsDataWriter classes with real
test cases from the chemfiles project. All operations use xarray.Dataset.
Also tests Frame initialization with xarray to_dict format.
"""

import pytest
import tempfile
import numpy as np
from pathlib import Path
import os

# Import molpy components
import molpy as mp
from molpy.io.data.lammps import LammpsDataReader, LammpsDataWriter

@pytest.fixture
def test_files(TEST_DATA_DIR):
    """Provide paths to test files."""
    files = {
        'data_body': TEST_DATA_DIR / "data.body",
        'labelmap': TEST_DATA_DIR / "labelmap.lmp",
        'molid': TEST_DATA_DIR / "molid.lmp", 
        'solvated': TEST_DATA_DIR / "solvated.lmp",
        'triclinic_1': TEST_DATA_DIR / "triclinic-1.lmp",
        'triclinic_2': TEST_DATA_DIR / "triclinic-2.lmp",
        'whitespaces': TEST_DATA_DIR / "whitespaces.lmp"
    }
    
    # Check which files actually exist
    existing_files = {k: v for k, v in files.items() if v.exists()}
    return existing_files


class TestLammpsDataReader:
    """Test LAMMPS data file reader with real chemfiles test cases."""

    def test_molid_file(self, test_files):
        """Test reading molid.lmp - file with molecular IDs and full style."""
        if 'molid' not in test_files:
            pytest.skip("molid.lmp test file not found")
            
        reader = LammpsDataReader(test_files['molid'], atom_style="full")
        frame = reader.read()
        
        # Check basic structure
        assert 'atoms' in frame
        atoms = frame['atoms']
        
        # Should have 12 atoms based on file content
        assert len(atoms['id']) == 12
        assert 'mol' in atoms.data_vars  # molecule ID should be present
        assert 'type' in atoms.data_vars
        assert 'q' in atoms.data_vars
        assert 'xyz' in atoms.data_vars
        
        # Check coordinate data shape
        xyz = atoms['xyz'].values
        assert xyz.shape == (12, 3)  # 12 atoms, 3 coordinates
        
        # Check box dimensions (0-20 in each direction)
        assert frame.box is not None
        box_lengths = frame.box.lengths
        np.testing.assert_array_almost_equal(box_lengths, [20.0, 20.0, 20.0])
        
        # Check that molecule IDs are in the data (should be 0-3 based on file)
        mol_ids = atoms['mol'].values
        assert len(np.unique(mol_ids)) <= 4  # max 4 different molecules

    def test_whitespaces_file(self, test_files):
        """Test reading whitespaces.lmp - file with extra whitespaces."""
        if 'whitespaces' not in test_files:
            pytest.skip("whitespaces.lmp test file not found")
            
        reader = LammpsDataReader(test_files['whitespaces'], atom_style="full")
        frame = reader.read()
        
        # Should parse correctly despite extra whitespaces
        assert 'atoms' in frame
        atoms = frame['atoms']
        assert len(atoms['id']) == 1
        
        # Check the single atom's coordinates
        xyz = atoms['xyz'].values
        np.testing.assert_array_almost_equal(xyz[0], [5.0, 5.0, 5.0])
        
        # Check box (should be 10x10x10)
        box_lengths = frame.box.lengths
        np.testing.assert_array_almost_equal(box_lengths, [10.0, 10.0, 10.0])

    def test_triclinic_boxes(self, test_files):
        """Test reading triclinic box files."""
        # Test triclinic-1.lmp (zero tilt factors)
        if 'triclinic_1' in test_files:
            reader = LammpsDataReader(test_files['triclinic_1'])
            frame = reader.read()
            
            assert frame.box is not None
            # Should have 34x34x34 box with no tilt
            box_lengths = frame.box.lengths
            np.testing.assert_array_almost_equal(box_lengths, [34.0, 34.0, 34.0])
            
        # Test triclinic-2.lmp (non-zero tilt factors)
        if 'triclinic_2' in test_files:
            reader = LammpsDataReader(test_files['triclinic_2'])
            frame = reader.read()
            
            assert frame.box is not None
            # Should still have 34x34x34 basic dimensions
            box_lengths = frame.box.lengths  
            np.testing.assert_array_almost_equal(box_lengths, [34.0, 34.0, 34.0])

    def test_labelmap_file(self, test_files):
        """Test reading labelmap.lmp - file with atom/bond type labels."""
        if 'labelmap' not in test_files:
            pytest.skip("labelmap.lmp test file not found")
        
        # With unified string-based type handling, this should work now
        reader = LammpsDataReader(test_files['labelmap'], atom_style="full")
        frame = reader.read()
        
        # Check basic structure
        assert 'atoms' in frame
        atoms = frame['atoms']
        
        # Should have 16 atoms based on file content
        assert len(atoms['id']) == 16
        assert 'type' in atoms.data_vars
        assert 'xyz' in atoms.data_vars
        
        # Check that atom types are strings (labels like 'f', 'c3', etc.)
        atom_types = atoms['type'].values
        assert atom_types.dtype.kind == 'U'  # Unicode string
        
        # Check for expected labels
        unique_types = set(atom_types.flat)
        expected_labels = {'f', 'c3', 's6', 'o', 'ne', 'sy', 'Li+'}
        assert unique_types.issubset(expected_labels)
        
        # Check bonds if present
        if 'bonds' in frame:
            bonds = frame['bonds']
            assert bonds['type'].values.dtype.kind == 'U'  # String bond types too
            
        print(f"✓ labelmap.lmp: {len(atoms['id'])} atoms with string labels: {sorted(unique_types)}")

    def test_solvated_file(self, test_files):
        """Test reading solvated.lmp - large file with all topology types."""
        if 'solvated' not in test_files:
            pytest.skip("solvated.lmp test file not found")
            
        reader = LammpsDataReader(test_files['solvated'], atom_style="full")
        frame = reader.read()
        
        # Check atoms (should be 7772 atoms)
        assert 'atoms' in frame
        atoms = frame['atoms']
        assert len(atoms['id']) == 7772
        
        # Check all topology sections exist
        assert 'bonds' in frame
        assert 'angles' in frame
        assert 'dihedrals' in frame
        assert 'impropers' in frame
        
        # Check counts match file header
        bonds = frame['bonds']
        assert len(bonds['id']) == 6248
        
        angles = frame['angles']
        assert len(angles['id']) == 8100
        
        dihedrals = frame['dihedrals']
        assert len(dihedrals['id']) == 10720
        
        impropers = frame['impropers']
        assert len(impropers['id']) == 1376
        
        # Check atom types (should be 11 types)
        type_values = atoms['type'].values
        unique_types = np.unique(type_values)
        assert len(unique_types) == 11


class TestLammpsDataWriter:
    """Test LAMMPS data file writer."""

    def test_write_read_roundtrip(self, test_files):
        """Test that we can write and read back the same data."""
        if 'molid' not in test_files:
            pytest.skip("molid.lmp test file not found")
        
        # Read original file
        reader = LammpsDataReader(test_files['molid'], atom_style="full")
        original_frame = reader.read()
        
        # Write to temporary file
        with tempfile.NamedTemporaryFile(mode='w', suffix='.data', delete=False) as tmp:
            tmp_path = tmp.name
        
        try:
            writer = LammpsDataWriter(tmp_path, atom_style="full")
            writer.write(original_frame)
            
            # Read back
            reader2 = LammpsDataReader(tmp_path, atom_style="full")
            new_frame = reader2.read()
            
            # Compare atoms
            orig_atoms = original_frame['atoms']
            new_atoms = new_frame['atoms']
            
            assert len(orig_atoms['id']) == len(new_atoms['id'])
            np.testing.assert_array_equal(orig_atoms['type'].values, new_atoms['type'].values)
            np.testing.assert_array_almost_equal(orig_atoms['xyz'].values, new_atoms['xyz'].values)
            
            # Compare box
            assert original_frame.box is not None
            assert new_frame.box is not None
            np.testing.assert_array_almost_equal(
                original_frame.box.lengths, new_frame.box.lengths
            )
            
        finally:
            os.unlink(tmp_path)

    def test_write_minimal_frame(self):
        """Test writing a minimal frame with just atoms."""
        # Create a simple frame
        frame = mp.Frame()
        
        # Add atoms data
        atoms_data = {
            'id': np.array([1, 2, 3]),
            'type': np.array([1, 1, 2]),
            'xyz': np.array([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [0.0, 1.0, 0.0]]),
            'mass': np.array([1.0, 1.0, 2.0])
        }
        
        data_vars = {
            'id': (['atoms_id'], atoms_data['id']),
            'type': (['atoms_id'], atoms_data['type']),
            'xyz': (['atoms_id', 'spatial'], atoms_data['xyz']),
            'mass': (['atoms_id'], atoms_data['mass'])
        }
        
        coords = {
            'atoms_id': np.arange(3),
            'spatial': ['x', 'y', 'z']
        }
        
        frame['atoms'] = xr.Dataset(data_vars, coords=coords)
        frame.box = mp.Box([10.0, 10.0, 10.0])
        
        # Write to temporary file
        with tempfile.NamedTemporaryFile(mode='w', suffix='.data', delete=False) as tmp:
            tmp_path = tmp.name
        
        try:
            writer = LammpsDataWriter(tmp_path, atom_style="atomic")
            writer.write(frame)
            
            # Check file was written and has content
            assert os.path.exists(tmp_path)
            with open(tmp_path, 'r') as f:
                content = f.read()
                assert "3 atoms" in content
                assert "2 atom types" in content
                assert "Atoms" in content
                
        finally:
            os.unlink(tmp_path)


class TestFrameXarrayFormat:
    """Test Frame initialization with xarray to_dict format."""
    
    def test_frame_from_xarray_dict(self):
        """Test creating Frame from xarray Dataset.to_dict() format."""
        # Create a sample Dataset
        data_vars = {
            'id': (['atoms_id'], np.array([1, 2, 3])),
            'type': (['atoms_id'], np.array([1, 1, 2])),
            'xyz': (['atoms_id', 'spatial'], np.array([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [0.0, 1.0, 0.0]]))
        }
        coords = {
            'atoms_id': np.arange(3),
            'spatial': ['x', 'y', 'z']
        }
        
        original_dataset = xr.Dataset(data_vars, coords=coords)
        
        # Convert to dict format
        dict_format = original_dataset.to_dict()
        
        # Create Frame from dict format
        frame = mp.Frame({'atoms': dict_format})
        
        # Verify the data was reconstructed correctly
        assert 'atoms' in frame
        atoms = frame['atoms']
        
        assert len(atoms['id']) == 3
        np.testing.assert_array_equal(atoms['id'].values, [1, 2, 3])
        np.testing.assert_array_equal(atoms['type'].values, [1, 1, 2])
        np.testing.assert_array_almost_equal(
            atoms['xyz'].values, 
            [[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [0.0, 1.0, 0.0]]
        )
    
    def test_frame_roundtrip_with_dict_format(self, test_files):
        """Test Frame -> xarray dict -> Frame roundtrip."""
        if 'molid' not in test_files:
            pytest.skip("molid.lmp test file not found")
        
        # Read a frame
        reader = LammpsDataReader(test_files['molid'], atom_style="full")
        original_frame = reader.read()
        
        # Convert atoms to dict format
        atoms_dict = original_frame['atoms'].to_dict()
        
        # Create new frame from dict format
        new_frame = mp.Frame({'atoms': atoms_dict})
        new_frame.box = original_frame.box  # Copy box separately
        
        # Compare the frames
        orig_atoms = original_frame['atoms']
        new_atoms = new_frame['atoms']
        
        assert len(orig_atoms['id']) == len(new_atoms['id'])
        np.testing.assert_array_equal(orig_atoms['id'].values, new_atoms['id'].values)
        np.testing.assert_array_equal(orig_atoms['type'].values, new_atoms['type'].values)
        np.testing.assert_array_almost_equal(orig_atoms['xyz'].values, new_atoms['xyz'].values)
        
        if 'mol' in orig_atoms.data_vars:
            np.testing.assert_array_equal(orig_atoms['mol'].values, new_atoms['mol'].values)
        
        if 'q' in orig_atoms.data_vars:
            np.testing.assert_array_almost_equal(orig_atoms['q'].values, new_atoms['q'].values)


class TestMoleculeReader:
    """Test LAMMPS molecule template reader (if files exist)."""
    
    def test_molecule_reader_basic(self):
        """Test basic molecule reader functionality."""
        # Create a simple molecule template file for testing
        molecule_content = """# Test molecule
3 atoms
2 bonds

Coords
1 0.0 0.0 0.0
2 1.0 0.0 0.0  
3 0.0 1.0 0.0

Types
1 1
2 1
3 2

Bonds
1 1 1 2
2 1 2 3
"""
        
        with tempfile.NamedTemporaryFile(mode='w', suffix='.mol', delete=False) as tmp:
            tmp.write(molecule_content)
            tmp_path = tmp.name
        
        try:
            from molpy.io.data.lammps import LammpsMoleculeReader
            reader = LammpsMoleculeReader(tmp_path)
            frame = reader.read()
            
            # Check atoms
            assert 'atoms' in frame
            atoms = frame['atoms']
            assert len(atoms['id']) == 3
            assert 'type' in atoms.data_vars
            assert 'xyz' in atoms.data_vars
            
            # Check bonds
            assert 'bonds' in frame
            bonds = frame['bonds']
            assert len(bonds['id']) == 2
            
        finally:
            os.unlink(tmp_path)


if __name__ == "__main__":
    pytest.main([__file__])
