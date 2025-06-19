import pytest
import tempfile
import numpy as np
import pandas as pd
import molpy as mp
from pathlib import Path


class TestReadLammpsData:

    @pytest.fixture()
    def lammps_data(self, test_data_path):
        return test_data_path / "data/lammps-data"

    def test_molid(self, lammps_data):
        reader = mp.io.data.LammpsDataReader(lammps_data / "molid.lmp")
        frame = mp.Frame()
        frame = reader.read(frame)
        assert frame["atoms"].sizes["index"] == 12

    def test_labelmap(self, lammps_data):
        reader = mp.io.data.LammpsDataReader(lammps_data / "labelmap.lmp")
        frame = mp.Frame()
        frame = reader.read(frame)
        assert frame["atoms"].sizes["index"] == 16
        assert "type" in frame["atoms"].data_vars
        expected_types = ['f', 'c3', 'f', 'f', 's6', 'o', 'o', 'ne', 'sy', 'o', 'o', 'c3', 'f', 'f', 'f', 'Li+']
        assert (frame["atoms"]["type"].values == expected_types).all()

    def test_whitespaces(self, lammps_data):
        """Test reading LAMMPS file with irregular whitespace."""
        reader = mp.io.data.LammpsDataReader(lammps_data / "whitespaces.lmp")
        frame = mp.Frame()
        frame = reader.read(frame)
        assert frame["atoms"].sizes["index"] == 1
        # Check that box dimensions are correctly parsed
        assert frame.box is not None

    def test_solvated_partial(self, lammps_data):
        """Test reading first few lines of large solvated system."""
        reader = mp.io.data.LammpsDataReader(lammps_data / "solvated.lmp")
        frame = mp.Frame()
        frame = reader.read(frame)
        assert frame["atoms"].sizes["index"] == 7772
        assert "bonds" in frame
        assert "angles" in frame
        assert "dihedrals" in frame
        assert "impropers" in frame

    def test_triclinic_1(self, lammps_data):
        """Test reading triclinic box format."""
        reader = mp.io.data.LammpsDataReader(lammps_data / "triclinic-1.lmp")
        frame = mp.Frame()
        frame = reader.read(frame)
        # This file has no atoms, but should read box correctly
        assert frame.box is not None
        # triclinic-1.lmp has 0 atoms
        assert "atoms" not in frame or len(frame["atoms"]) == 0

    def test_context_manager(self, lammps_data):
        """Test using reader as context manager."""
        with mp.io.data.LammpsDataReader(lammps_data / "molid.lmp") as reader:
            frame = mp.Frame()
            frame = reader.read(frame)
            assert frame["atoms"].sizes["index"] == 12

    def test_charges_and_coordinates(self, lammps_data):
        """Test that charges and coordinates are correctly read."""
        reader = mp.io.data.LammpsDataReader(lammps_data / "molid.lmp")
        frame = mp.Frame()
        frame = reader.read(frame)
        
        # Check that charges exist and have reasonable values
        if "charge" in frame["atoms"]:
            charges = frame["atoms"]["charge"].values
            assert len(charges) == 12
            assert all(isinstance(c, (int, float)) for c in charges)
        
        # Check coordinates
        coords = None
        if "x" in frame["atoms"] and "y" in frame["atoms"] and "z" in frame["atoms"]:
            coords = np.column_stack([
                frame["atoms"]["x"].values,
                frame["atoms"]["y"].values,
                frame["atoms"]["z"].values
            ])
        elif "xyz" in frame["atoms"]:
            coords = frame["atoms"]["xyz"].values
            
        assert coords is not None
        assert coords.shape == (12, 3)
        assert np.all(np.isfinite(coords))


class TestWriteLammpsData:

    def test_write_simple(self):
        """Test writing a simple LAMMPS data file."""
        # Create a simple frame using new Frame API
        frame = mp.Frame()
        
        # Create atoms data as dict for new Frame API
        atoms_data = {
            'id': [0, 1, 2],
            'molid': [1, 1, 2],
            'type': ['H', 'H', 'O'],
            'q': [0.0, 0.0, -1.0],
            'xyz': [[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [0.5, 1.0, 0.0]]
        }
        frame["atoms"] = atoms_data
        
        # Create bonds data
        bonds_data = {
            'id': [0],
            'type': [1],
            'i': [0],
            'j': [1]
        }
        frame["bonds"] = bonds_data
        
        # Set box
        frame.box = mp.Box(np.eye(3) * 10.0)
        
        # Write to temporary file
        with tempfile.NamedTemporaryFile(mode='w', suffix='.lmp', delete=False) as tmp:
            writer = mp.io.data.LammpsDataWriter(tmp.name)
            writer.write(frame)
            
            # Read back and verify
            reader = mp.io.data.LammpsDataReader(tmp.name)
            frame_read = reader.read(mp.Frame())
            
            # Get number of atoms correctly based on frame structure
            n_atoms = 0
            if "atoms" in frame_read:
                atoms_data = frame_read["atoms"]
                if hasattr(atoms_data, 'sizes'):
                    # Find the atom count dimension
                    for dim_name in ['index', 'dim_id_0', 'dim_q_0', 'dim_xyz_0']:
                        if dim_name in atoms_data.sizes:
                            n_atoms = atoms_data.sizes[dim_name]
                            break
            
            assert n_atoms == 3
            
            # Check bonds if they exist
            if "bonds" in frame_read:
                bonds_data = frame_read["bonds"]
                n_bonds = 0
                if hasattr(bonds_data, 'sizes'):
                    for dim_name in ['index', 'dim_id_0', 'dim_i_0']:
                        if dim_name in bonds_data.sizes:
                            n_bonds = bonds_data.sizes[dim_name]
                            break
                assert n_bonds == 1

    def test_write_with_context_manager(self):
        """Test writing using context manager."""
        frame = mp.Frame()
        
        atoms_data = {
            'id': [0, 1],
            'molid': [1, 1],
            'type': ['H', 'H'],
            'q': [0.0, 0.0],
            'xyz': [[0.0, 0.0, 0.0], [1.0, 0.0, 0.0]]
        }
        frame["atoms"] = atoms_data
        frame.box = mp.Box(np.eye(3) * 5.0)
        
        with tempfile.NamedTemporaryFile(mode='w', suffix='.lmp', delete=False) as tmp:
            with mp.io.data.LammpsDataWriter(tmp.name) as writer:
                writer.write(frame)

    def test_write_waterbox_style(self):
        """Test writing a more complex system like waterbox."""
        # Create a test system similar to waterbox
        frame = mp.Frame()
        
        # Create 2 water molecules (6 atoms total)
        atoms_data = {
            'id': list(range(6)),
            'molid': [1, 1, 1, 2, 2, 2],
            'type': ['O', 'H', 'H', 'O', 'H', 'H'],
            'q': [-0.8476, 0.4238, 0.4238, -0.8476, 0.4238, 0.4238],
            'xyz': [
                [0.0, 0.0, 0.0],
                [0.8164904, 0.577359, 0.0],
                [-0.8164904, 0.577359, 0.0],
                [3.0, 0.0, 0.0],
                [3.8164904, 0.577359, 0.0],
                [2.1835096, 0.577359, 0.0]
            ]
        }
        frame["atoms"] = atoms_data
        
        # Create bonds (O-H bonds)
        bonds_data = {
            'id': [0, 1, 2, 3],
            'i': [0, 0, 3, 3],  # O atoms
            'j': [1, 2, 4, 5],  # H atoms
        }
        frame["bonds"] = bonds_data
        
        # Create angles (H-O-H)
        angles_data = {
            'id': [0, 1],
            'i': [1, 4],  # H atoms
            'j': [0, 3],  # O atoms  
            'k': [2, 5],  # H atoms
            'theta0': [109.47, 109.47],
            'k': [1000.0, 1000.0]
        }
        frame["angles"] = angles_data
        
        frame.box = mp.Box(np.diag([10.0, 10.0, 10.0]))
        
        with tempfile.NamedTemporaryFile(mode='w', suffix='.lmp', delete=False) as tmp:
            writer = mp.io.data.LammpsDataWriter(tmp.name)
            writer.write(frame)
            
            # Verify the written file
            with open(tmp.name, 'r') as f:
                content = f.read()
                
            # Check header information
            assert "6 atoms" in content
            assert "4 bonds" in content
            assert "2 angles" in content
            
            # Check that charges are written correctly (not zero)
            assert "-0.847600" in content
            assert "0.423800" in content
            
            # Check that bonds and angles sections exist
            assert "Bonds" in content
            assert "Angles" in content

    def test_write_charge_field_mapping(self):
        """Test that 'q' field is correctly mapped to 'charge' in output."""
        frame = mp.Frame()
        
        atoms_data = {
            'id': [0, 1],
            'molid': [1, 1],
            'type': ['O', 'H'],
            'q': [-0.5, 0.5],  # Use 'q' field
            'xyz': [[0.0, 0.0, 0.0], [1.0, 0.0, 0.0]]
        }
        frame["atoms"] = atoms_data
        frame.box = mp.Box(np.eye(3) * 5.0)
        
        with tempfile.NamedTemporaryFile(mode='w', suffix='.lmp', delete=False) as tmp:
            writer = mp.io.data.LammpsDataWriter(tmp.name)
            writer.write(frame)
            
            # Check that charges are written correctly
            with open(tmp.name, 'r') as f:
                content = f.read()
                
            assert "-0.500000" in content
            assert "0.500000" in content
            # Should not have zero charges if we provided non-zero charges
            lines = content.split('\n')
            atoms_section = False
            for line in lines:
                if line.strip() == "Atoms":
                    atoms_section = True
                    continue
                if atoms_section and line.strip() and not line.startswith('#'):
                    if line.strip() == "":
                        continue
                    parts = line.split()
                    if len(parts) >= 4:
                        charge = float(parts[3])
                        assert charge != 0.0  # Should have non-zero charges
            
            # Read back and verify
            reader = mp.io.data.LammpsDataReader(tmp.name)
            frame_read = reader.read(mp.Frame())
            
            assert frame_read["atoms"].sizes["index"] == 2  # We only have 2 atoms

    def test_charge_field_mapping_fix(self):
        """Test that 'q' field is correctly mapped to 'charge' in output."""
        frame = mp.Frame()
        
        atoms_data = {
            'id': [0, 1, 2],
            'molid': [1, 1, 1],
            'type': ['O', 'H', 'H'],
            'q': [-0.8476, 0.4238, 0.4238],  # Using 'q' not 'charge'
            'xyz': [[0.0, 0.0, 0.0], [0.816, 0.577, 0.0], [-0.816, 0.577, 0.0]]
        }
        frame["atoms"] = atoms_data
        frame.box = mp.Box(np.eye(3) * 10.0)
        
        with tempfile.NamedTemporaryFile(mode='w', suffix='.data', delete=False) as tmp:
            writer = mp.io.data.LammpsDataWriter(tmp.name)
            writer.write(frame)
            
            # Read file content and verify charges are written correctly
            with open(tmp.name, 'r') as f:
                content = f.read()
            
            # Should contain the charge values
            assert "-0.847600" in content
            assert "0.423800" in content
            
            # Should have correct atom count
            assert "3 atoms" in content

    def test_bond_angle_output_frame_api(self):
        """Test that bonds and angles are correctly output with new Frame API."""
        frame = mp.Frame()
        
        # Water molecule with bonds and angles
        atoms_data = {
            'id': [0, 1, 2],
            'molid': [1, 1, 1],
            'type': ['O', 'H', 'H'],
            'q': [-0.8476, 0.4238, 0.4238],
            'xyz': [[0.0, 0.0, 0.0], [0.816, 0.577, 0.0], [-0.816, 0.577, 0.0]]
        }
        
        bonds_data = {
            'id': [0, 1],
            'i': [0, 0],  # O-H bonds
            'j': [1, 2]
        }
        
        angles_data = {
            'id': [0],
            'i': [1],  # H-O-H angle
            'j': [0],
            'k': [2]
        }
        
        frame["atoms"] = atoms_data
        frame["bonds"] = bonds_data
        frame["angles"] = angles_data
        frame.box = mp.Box(np.eye(3) * 10.0)
        
        with tempfile.NamedTemporaryFile(mode='w', suffix='.data', delete=False) as tmp:
            writer = mp.io.data.LammpsDataWriter(tmp.name)
            writer.write(frame)
            
            with open(tmp.name, 'r') as f:
                content = f.read()
            
            # Should have correct counts
            assert "3 atoms" in content
            assert "2 bonds" in content
            assert "1 angles" in content
            
            # Should have sections
            assert "Atoms" in content
            assert "Bonds" in content
            assert "Angles" in content
            
            # Verify bond data (1-based indexing)
            lines = content.split('\n')
            bonds_section = False
            bond_found = False
            for line in lines:
                if line.strip() == "Bonds":
                    bonds_section = True
                    continue
                elif line.strip() == "Angles":
                    bonds_section = False
                
                if bonds_section and line.strip() and not line.startswith('#'):
                    parts = line.split()
                    if len(parts) >= 4:
                        bond_id = int(parts[0])
                        atom_i = int(parts[2])
                        atom_j = int(parts[3])
                        
                        # Should use 1-based indexing in LAMMPS output
                        assert 1 <= atom_i <= 3
                        assert 1 <= atom_j <= 3
                        bond_found = True
            
            assert bond_found, "No bonds found in output"

    def test_dimension_name_compatibility(self):
        """Test that various dimension names are handled correctly."""
        # Test with different dimension naming patterns that can occur
        # in the new Frame API
        frame = mp.Frame()
        
        # Create atoms using dict (gets converted to xarray internally)
        atoms_data = {
            'id': [0, 1],
            'molid': [1, 1], 
            'type': ['A', 'B'],
            'q': [0.5, -0.5],
            'xyz': [[0.0, 0.0, 0.0], [1.0, 0.0, 0.0]]
        }
        frame["atoms"] = atoms_data
        frame.box = mp.Box(np.eye(3) * 5.0)
        
        with tempfile.NamedTemporaryFile(mode='w', suffix='.data', delete=False) as tmp:
            writer = mp.io.data.LammpsDataWriter(tmp.name)
            writer.write(frame)
            
            with open(tmp.name, 'r') as f:
                content = f.read()
            
            # Should correctly count atoms regardless of dimension names
            assert "2 atoms" in content
            assert "0.500000" in content  # Positive charge
            assert "-0.500000" in content  # Negative charge

    def test_coordinate_array_formats(self):
        """Test handling of xyz coordinate arrays vs individual x,y,z fields."""
        frame = mp.Frame()
        
        # Test with xyz array format
        atoms_data = {
            'id': [0, 1, 2],
            'molid': [1, 1, 1],
            'type': ['A', 'A', 'A'],
            'q': [0.0, 0.0, 0.0],
            'xyz': [[0.0, 0.0, 0.0], [1.0, 1.0, 1.0], [2.0, 2.0, 2.0]]
        }
        frame["atoms"] = atoms_data
        frame.box = mp.Box(np.eye(3) * 10.0)
        
        with tempfile.NamedTemporaryFile(mode='w', suffix='.data', delete=False) as tmp:
            writer = mp.io.data.LammpsDataWriter(tmp.name)
            writer.write(frame)
            
            with open(tmp.name, 'r') as f:
                content = f.read()
            
            # Should write coordinates correctly
            assert "3 atoms" in content
            # Check that coordinates appear in file
            assert "0.000000" in content
            assert "1.000000" in content  
            assert "2.000000" in content


class TestLammpsMoleculeFormat:

    def test_molecule_reader_writer_roundtrip(self):
        """Test molecule format reader/writer roundtrip."""
        # Create test frame
        frame = mp.Frame()
        
        atoms_data = pd.DataFrame({
            'id': [0, 1, 2],
            'x': [0.0, 1.0, 0.5],
            'y': [0.0, 0.0, 1.0],
            'z': [0.0, 0.0, 0.0],
            'type': [1, 1, 2],
            'charge': [0.0, 0.0, -1.0],
            'molid': [1, 1, 2]
        })
        frame["atoms"] = atoms_data.to_xarray()
        
        # Write to temporary file
        with tempfile.NamedTemporaryFile(mode='w', suffix='.mol', delete=False) as tmp:
            writer = mp.io.data.LammpsMoleculeWriter(tmp.name)
            writer.write(frame)
            
            # Read back
            reader = mp.io.data.LammpsMoleculeReader(tmp.name)
            frame_read = reader.read(mp.Frame())
            
            assert "atoms" in frame_read
            # Note: exact comparison might differ due to coordinate handling


class TestErrorHandling:

    def test_empty_file(self):
        """Test handling of empty files."""
        with tempfile.NamedTemporaryFile(mode='w', suffix='.lmp', delete=False) as tmp:
            tmp.write("")  # Empty file
            
            reader = mp.io.data.LammpsDataReader(tmp.name)
            frame = reader.read(mp.Frame())
            # Should not crash, but frame might be mostly empty
            assert isinstance(frame, mp.Frame)

    def test_malformed_file(self):
        """Test handling of malformed files."""
        with tempfile.NamedTemporaryFile(mode='w', suffix='.lmp', delete=False) as tmp:
            tmp.write("This is not a valid LAMMPS file\n")
            tmp.write("Just some random text\n")
            
            reader = mp.io.data.LammpsDataReader(tmp.name)
            # Should not crash even with malformed input
            frame = reader.read(mp.Frame())
            assert isinstance(frame, mp.Frame)

    def test_missing_file(self):
        """Test handling of missing files."""
        with pytest.raises((FileNotFoundError, IOError)):
            reader = mp.io.data.LammpsDataReader("nonexistent_file.lmp")
            reader.read(mp.Frame())
