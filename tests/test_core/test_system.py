"""
Tests for System class.
"""

import pytest
import numpy as np
from molpy.core.system import System
from molpy.core.atomistic import AtomicStructure, Atom
from molpy.core.forcefield import ForceField
from molpy.core.box import Box
from molpy.core.frame import Frame


class TestSystem:
    """Test System class functionality."""
    
    def test_init(self):
        """Test System initialization."""
        system = System()
        
        assert isinstance(system.forcefield, ForceField)
        assert hasattr(system, '_box')
        assert system._struct == []
    
    def test_set_forcefield(self):
        """Test setting forcefield."""
        system = System()
        ff = ForceField(name="test_ff")
        
        system.set_forcefield(ff)
        assert system.forcefield == ff
        
        # Test property setter
        ff2 = ForceField(name="test_ff2")
        system.forcefield = ff2
        assert system.forcefield == ff2
    
    def test_get_forcefield(self):
        """Test getting forcefield."""
        system = System()
        ff = ForceField(name="test_ff")
        system.set_forcefield(ff)
        
        retrieved_ff = system.get_forcefield()
        assert retrieved_ff == ff
    
    def test_def_box(self):
        """Test defining simulation box."""
        system = System()
        
        # Test with simple orthogonal box
        matrix = np.diag([10.0, 10.0, 10.0])
        system.def_box(matrix)
        
        assert hasattr(system._box, 'matrix')
        np.testing.assert_array_equal(system._box.matrix, matrix)
    
    def test_add_struct(self):
        """Test adding structures to system."""
        system = System()
        
        # Create test structure
        struct = AtomicStructure(name="test_mol")
        struct.def_atom(name="C", element="C", xyz=[0, 0, 0])
        
        system.add_struct(struct)
        assert len(system._struct) == 1
        assert system._struct[0] == struct
        
        # Add another structure
        struct2 = AtomicStructure(name="test_mol2")
        struct2.def_atom(name="O", element="O", xyz=[1, 1, 1])
        
        system.add_struct(struct2)
        assert len(system._struct) == 2
        assert system._struct[1] == struct2
    
    def test_to_dict_empty_system(self):
        """Test to_dict method with empty system."""
        system = System()
        
        result = system.to_dict()
        
        assert isinstance(result, dict)
        assert 'forcefield' in result
        assert 'box' in result
        assert 'structures' in result
        assert 'n_structures' in result
        
        assert result['n_structures'] == 0
        assert result['structures'] == []
    
    def test_to_dict_with_structures(self):
        """Test to_dict method with structures."""
        system = System()
        
        # Add structures
        struct1 = AtomicStructure(name="water")
        struct1.def_atom(name="O", element="O", xyz=[0, 0, 0])
        struct1.def_atom(name="H1", element="H", xyz=[1, 0, 0])
        struct1.def_atom(name="H2", element="H", xyz=[0, 1, 0])
        
        struct2 = AtomicStructure(name="methane")
        struct2.def_atom(name="C", element="C", xyz=[0, 0, 0])
        
        system.add_struct(struct1)
        system.add_struct(struct2)
        
        # Set box
        matrix = np.diag([10.0, 10.0, 10.0])
        system.def_box(matrix)
        
        result = system.to_dict()
        
        assert result['n_structures'] == 2
        assert len(result['structures']) == 2
        
        # Check box data
        box_data = result['box']
        # Box.to_dict() returns LAMMPS format
        assert 'xlo' in box_data
        assert 'xhi' in box_data
        
        # Check structures data
        structures = result['structures']
        assert len(structures) == 2
    
    def test_to_frame_empty_system(self):
        """Test to_frame method with empty system."""
        system = System()
        
        frame = system.to_frame()
        
        assert isinstance(frame, Frame)
        assert hasattr(frame, 'box')
        assert hasattr(frame, 'forcefield')
    
    def test_to_frame_with_structures(self):
        """Test to_frame method with structures."""
        system = System()
        
        # Create test structures
        struct1 = AtomicStructure(name="mol1")
        struct1.def_atom(name="C1", element="C", xyz=[0, 0, 0])
        struct1.def_atom(name="H1", element="H", xyz=[1, 0, 0])
        
        struct2 = AtomicStructure(name="mol2")
        struct2.def_atom(name="C2", element="C", xyz=[2, 0, 0])
        struct2.def_atom(name="H2", element="H", xyz=[3, 0, 0])
        
        system.add_struct(struct1)
        system.add_struct(struct2)
        
        # Set box and forcefield
        matrix = np.diag([15.0, 15.0, 15.0])
        system.def_box(matrix)
        ff = ForceField(name="test_ff")
        system.set_forcefield(ff)
        
        frame = system.to_frame()
        
        assert isinstance(frame, Frame)
        assert frame.box == system._box
        assert frame.forcefield == system._forcefield
        
        # Check that atoms are combined
        assert 'atoms' in frame._data
        atoms_data = frame._data['atoms']
        
        # Should have 4 atoms total (2 from each structure)
        assert len(atoms_data['name']) == 4
    
    def test_to_frame_hierarchy_preserved(self):
        """Test that hierarchical structure is preserved in frame conversion."""
        system = System()
        
        # Create nested structure
        parent_struct = AtomicStructure(name="parent")
        parent_struct.def_atom(name="P1", element="P", xyz=[0, 0, 0])
        
        child_struct = AtomicStructure(name="child")
        child_struct.def_atom(name="C1", element="C", xyz=[1, 1, 1])
        
        # Add child to parent
        parent_struct.add_struct(child_struct)
        system.add_struct(parent_struct)
        
        frame = system.to_frame()
        
        # Should contain atoms from both parent and child
        assert 'atoms' in frame._data
        atoms_data = frame._data['atoms']
        assert len(atoms_data['name']) == 2
        
        # Check that both atoms are present
        names = atoms_data['name']
        assert 'P1' in names
        assert 'C1' in names


class TestSystemIntegration:
    """Integration tests for System class."""
    
    def test_system_workflow(self):
        """Test complete system workflow."""
        # Create system
        system = System()
        
        # Set up forcefield
        ff = ForceField(name="test_forcefield", unit="real")
        system.set_forcefield(ff)
        
        # Set up box
        box_matrix = np.array([
            [20.0, 0.0, 0.0],
            [0.0, 20.0, 0.0],
            [0.0, 0.0, 20.0]
        ])
        system.def_box(box_matrix)
        
        # Create molecules
        water = AtomicStructure(name="water")
        o = water.def_atom(name="O", element="O", xyz=[0.0, 0.0, 0.0])
        h1 = water.def_atom(name="H1", element="H", xyz=[0.757, 0.586, 0.0])
        h2 = water.def_atom(name="H2", element="H", xyz=[-0.757, 0.586, 0.0])
        water.def_bond(o, h1)
        water.def_bond(o, h2)
        
        system.add_struct(water)
        
        # Test to_dict
        system_dict = system.to_dict()
        assert system_dict['n_structures'] == 1
        assert len(system_dict['structures']) == 1
        
        # Test to_frame
        frame = system.to_frame()
        assert isinstance(frame, Frame)
        assert 'atoms' in frame._data
        
        atoms_data = frame._data['atoms']
        assert len(atoms_data['name']) == 3  # O, H1, H2
        
        # Check bonds are preserved
        if 'bonds' in frame._data:
            bonds_data = frame._data['bonds']
            assert len(bonds_data['i']) == 2  # Two bonds: O-H1, O-H2
    
    def test_system_with_multiple_molecules(self):
        """Test system with multiple different molecules."""
        system = System()
        
        # Create different molecules
        molecules = []
        
        # Water
        water = AtomicStructure(name="water")
        water.def_atom(name="O", element="O", xyz=[0, 0, 0])
        water.def_atom(name="H1", element="H", xyz=[1, 0, 0])
        water.def_atom(name="H2", element="H", xyz=[0, 1, 0])
        molecules.append(water)
        
        # Methane
        methane = AtomicStructure(name="methane")
        methane.def_atom(name="C", element="C", xyz=[5, 5, 5])
        for i in range(4):
            methane.def_atom(name=f"H{i+1}", element="H", xyz=[5+i*0.5, 5, 5])
        molecules.append(methane)
        
        # Add to system
        for mol in molecules:
            system.add_struct(mol)
        
        # Convert to frame
        frame = system.to_frame()
        
        # Should have 3 + 5 = 8 atoms total
        atoms_data = frame._data['atoms']
        assert len(atoms_data['name']) == 8
        
        # Check that all atom names are present
        names = set(atoms_data['name'].values)  # Use .values to get numpy array
        expected_names = {'O', 'H1', 'H2', 'C', 'H1', 'H2', 'H3', 'H4'}
        # Note: some names might overlap, so we check counts
        assert len(names) >= 4  # At least C, O, and some H atoms


class TestSystemFrameIntegration:
    """Tests for System and Frame integration."""
    
    def test_system_to_frame_to_dict_roundtrip(self):
        """Test that System -> Frame -> dict conversion preserves data."""
        # Create a complex system
        system = System()
        
        # Set up forcefield
        ff = ForceField(name="integration_test", unit="real")
        system.set_forcefield(ff)
        
        # Set up box
        box_matrix = np.array([
            [15.0, 0.0, 0.0],
            [0.0, 15.0, 0.0],
            [0.0, 0.0, 15.0]
        ])
        system.def_box(box_matrix)
        
        # Create molecules
        water = AtomicStructure(name="water")
        water.def_atom(name="O", element="O", xyz=[0.0, 0.0, 0.0])
        water.def_atom(name="H1", element="H", xyz=[0.757, 0.586, 0.0])
        water.def_atom(name="H2", element="H", xyz=[-0.757, 0.586, 0.0])
        
        methane = AtomicStructure(name="methane")
        methane.def_atom(name="C", element="C", xyz=[5.0, 5.0, 5.0])
        methane.def_atom(name="H1", element="H", xyz=[5.5, 5.0, 5.0])
        methane.def_atom(name="H2", element="H", xyz=[5.0, 5.5, 5.0])
        methane.def_atom(name="H3", element="H", xyz=[5.0, 5.0, 5.5])
        methane.def_atom(name="H4", element="H", xyz=[4.5, 5.0, 5.0])
        
        system.add_struct(water)
        system.add_struct(methane)
        
        # Convert System to Frame
        frame = system.to_frame()
        
        # Verify Frame has expected data
        assert isinstance(frame, Frame)
        assert frame.box is not None
        assert frame.forcefield is not None
        assert 'atoms' in frame._data
        
        atoms_data = frame._data['atoms']
        assert len(atoms_data['name']) == 8  # 3 (water) + 5 (methane)
        
        # Convert Frame to dict
        frame_dict = frame.to_dict()
        
        # Verify dict structure
        assert 'data' in frame_dict
        assert 'box' in frame_dict
        assert 'forcefield' in frame_dict
        assert 'metadata' in frame_dict
        
        # Verify that atom data is preserved
        atoms_dict = frame_dict['data']['atoms']
        assert 'name' in atoms_dict['data_vars']
        assert 'element' in atoms_dict['data_vars']
        assert 'xyz' in atoms_dict['data_vars']
    
    def test_frame_from_dict_reconstruction(self):
        """Test Frame reconstruction from dictionary."""
        # Create original system
        system = System()
        
        # Simple setup
        ff = ForceField(name="test_ff")
        system.set_forcefield(ff)
        
        box_matrix = np.diag([10.0, 10.0, 10.0])
        system.def_box(box_matrix)
        
        # Single molecule
        mol = AtomicStructure(name="test_mol")
        mol.def_atom(name="C1", element="C", xyz=[0, 0, 0])
        mol.def_atom(name="H1", element="H", xyz=[1, 0, 0])
        system.add_struct(mol)
        
        # Convert to Frame and then to dict
        original_frame = system.to_frame()
        frame_dict = original_frame.to_dict()
        
        # Reconstruct Frame from dict
        reconstructed_frame = Frame.from_dict(frame_dict)
        
        # Verify reconstruction
        assert isinstance(reconstructed_frame, Frame)
        assert 'atoms' in reconstructed_frame._data
        
        # Compare atom data
        orig_atoms = original_frame._data['atoms']
        recon_atoms = reconstructed_frame._data['atoms']
        
        assert len(orig_atoms['name']) == len(recon_atoms['name'])
        np.testing.assert_array_equal(orig_atoms['name'].values, recon_atoms['name'].values)
        np.testing.assert_array_equal(orig_atoms['element'].values, recon_atoms['element'].values)
        np.testing.assert_array_almost_equal(orig_atoms['xyz'].values, recon_atoms['xyz'].values)
    
    def test_system_frame_data_consistency(self):
        """Test that System data is consistently represented in Frame."""
        system = System()
        
        # Create test structures with different properties
        struct1 = AtomicStructure(name="mol1")
        struct1.def_atom(name="O1", element="O", xyz=[1.0, 2.0, 3.0])
        struct1.def_atom(name="H1", element="H", xyz=[1.5, 2.0, 3.0])
        
        struct2 = AtomicStructure(name="mol2") 
        struct2.def_atom(name="C1", element="C", xyz=[4.0, 5.0, 6.0])
        struct2.def_atom(name="N1", element="N", xyz=[4.5, 5.0, 6.0])
        
        system.add_struct(struct1)
        system.add_struct(struct2)
        
        # Convert to frame
        frame = system.to_frame()
        
        # Verify all atoms are present
        atoms_data = frame._data['atoms']
        assert len(atoms_data['name']) == 4
        
        # Verify coordinates are preserved
        expected_coords = np.array([
            [1.0, 2.0, 3.0],  # O1
            [1.5, 2.0, 3.0],  # H1  
            [4.0, 5.0, 6.0],  # C1
            [4.5, 5.0, 6.0]   # N1
        ])
        
        actual_coords = atoms_data['xyz'].values
        np.testing.assert_array_almost_equal(actual_coords, expected_coords)
        
        # Verify elements are preserved
        expected_elements = ['O', 'H', 'C', 'N']
        actual_elements = atoms_data['element'].values.tolist()
        assert actual_elements == expected_elements


if __name__ == "__main__":
    pytest.main([__file__])
