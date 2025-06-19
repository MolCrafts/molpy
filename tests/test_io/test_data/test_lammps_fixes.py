"""
Tests specifically for the LAMMPS data writer fixes.
This test file focuses on the specific issues that were addressed:
- Deep copy issues with AtomicStructure
- Charge field mapping from 'q' to 'charge' 
- Bond/Angle output in LAMMPS data files
- Atom counting and coordinate handling
"""

import pytest
import tempfile
import numpy as np
import molpy as mp
from pathlib import Path


class TestLammpsDataWriterFixes:
    """Test the specific fixes made to LAMMPS data writer."""

    def test_charge_field_mapping_q_to_charge(self):
        """Test that 'q' field is properly mapped to 'charge' in LAMMPS output."""
        # Create frame with 'q' field instead of 'charge'
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

    def test_bond_angle_output_with_frame_api(self):
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

    def test_dimension_name_handling(self):
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

    def test_coordinate_array_handling(self):
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

    def test_empty_structure_handling(self):
        """Test graceful handling of structures without bonds/angles."""
        frame = mp.Frame()
        
        # Just atoms, no bonds or angles
        atoms_data = {
            'id': [0],
            'molid': [1],
            'type': ['A'],
            'q': [0.0],
            'xyz': [[0.0, 0.0, 0.0]]
        }
        frame["atoms"] = atoms_data
        frame.box = mp.Box(np.eye(3) * 5.0)
        
        with tempfile.NamedTemporaryFile(mode='w', suffix='.data', delete=False) as tmp:
            writer = mp.io.data.LammpsDataWriter(tmp.name)
            writer.write(frame)
            
            with open(tmp.name, 'r') as f:
                content = f.read()
            
            # Should write correctly even without bonds/angles
            assert "1 atoms" in content
            # Note: Writer may not output "0 bonds" line if no bonds exist
            # This is acceptable behavior
            assert "1 atom types" in content


class TestWaterboxWorkflowFixes:
    """Test the complete waterbox workflow with the fixes."""

    def test_deep_copy_independence(self):
        """Test that deep copied molecules are truly independent."""
        # Create template
        template = mp.AtomicStructure("water_template")
        o = template.def_atom(name="O", type="O", q=-0.8476, xyz=[0.0, 0.0, 0.0])
        h1 = template.def_atom(name="H1", type="H", q=0.4238, xyz=[0.816, 0.577, 0.0])
        h2 = template.def_atom(name="H2", type="H", q=0.4238, xyz=[-0.816, 0.577, 0.0])
        template.def_bond(o, h1)
        template.def_bond(o, h2)
        
        # Create multiple copies
        copies = []
        for i in range(3):
            copy = template(molid=i+1)
            copies.append(copy)
        
        # Modify one copy
        copies[0].atoms[0].xyz = [10.0, 10.0, 10.0]
        
        # Other copies should be unchanged
        assert not np.allclose(copies[1].atoms[0].xyz, [10.0, 10.0, 10.0])
        assert not np.allclose(copies[2].atoms[0].xyz, [10.0, 10.0, 10.0])
        
        # Original template should be unchanged
        assert np.allclose(template.atoms[0].xyz, [0.0, 0.0, 0.0])

    def test_spatial_wrapper_with_copies(self):
        """Test SpatialWrapper works correctly with copied structures.""" 
        template = mp.AtomicStructure("template")
        template.def_atom(name="A", xyz=[0.0, 0.0, 0.0])
        template.def_atom(name="B", xyz=[1.0, 0.0, 0.0])
        
        wrapper = mp.SpatialWrapper(template)
        
        # Create copies and move them
        positions = []
        for i in range(3):
            copy_wrapper = wrapper(molid=i+1)
            copy_wrapper.move([i*3.0, 0.0, 0.0])
            
            # Extract coordinates
            coords = [atom.xyz for atom in copy_wrapper._wrapped.atoms]
            positions.append(coords)
        
        # Verify positions are different
        for i in range(3):
            for j in range(i+1, 3):
                # Positions should be different between copies
                assert not np.allclose(positions[i][0], positions[j][0])
        
        # Original should be unchanged
        assert np.allclose(template.atoms[0].xyz, [0.0, 0.0, 0.0])

    def test_complete_waterbox_export(self):
        """Test complete waterbox generation and LAMMPS export."""
        # Create water template
        template = mp.AtomicStructure("water", molid=1)
        o = template.def_atom(name="O", type="O", q=-0.8476, xyz=[0.0, 0.0, 0.0])
        h1 = template.def_atom(name="H1", type="H", q=0.4238, xyz=[0.816, 0.577, 0.0])
        h2 = template.def_atom(name="H2", type="H", q=0.4238, xyz=[-0.816, 0.577, 0.0])
        template.def_bond(o, h1)
        template.def_bond(o, h2)
        
        # Create system with multiple water molecules
        system = mp.System()
        system.def_box(np.diag([10.0, 10.0, 10.0]))
        
        wrapper = mp.SpatialWrapper(template)
        
        # Add 4 water molecules in 2x2 grid
        n_molecules = 0
        for i in range(2):
            for j in range(2):
                molid = n_molecules + 1
                
                water_wrapper = wrapper(molid=molid)
                # Update molids
                for atom in water_wrapper._wrapped.atoms:
                    atom['molid'] = molid
                
                # Position molecule
                water_wrapper.move([i*3.0, j*3.0, 0.0])
                
                # Add to system
                system.add_struct(water_wrapper._wrapped)
                n_molecules += 1
        
        # Convert to frame
        frame = system.to_frame()
        
        # Export to LAMMPS
        with tempfile.NamedTemporaryFile(mode='w', suffix='.data', delete=False) as tmp:
            writer = mp.io.data.LammpsDataWriter(tmp.name)
            writer.write(frame)
            
            with open(tmp.name, 'r') as f:
                content = f.read()
            
            # Verify structure
            assert "12 atoms" in content  # 3 atoms * 4 molecules
            assert "8 bonds" in content   # 2 bonds * 4 molecules
            
            # Verify charges
            assert "-0.847600" in content  # Oxygen charges
            assert "0.423800" in content   # Hydrogen charges
            
            # Verify sections exist
            assert "Atoms" in content
            assert "Bonds" in content
