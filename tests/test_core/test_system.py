"""Unit tests for system module."""

import pytest
import numpy as np
from numpy.testing import assert_array_almost_equal

from molpy.core.system import FrameSystem, StructSystem, PeriodicSystem
from molpy.core.frame import Frame, Block
from molpy.core.box import Box
from molpy.core.forcefield import ForceField


class TestFrameSystem:
    """Test cases for FrameSystem class."""
    
    def test_init_default(self):
        """Test FrameSystem initialization with default parameters."""
        system = FrameSystem()
        assert system.box is not None
        assert system.forcefield is not None
        assert isinstance(system.box, Box)
        assert isinstance(system.forcefield, ForceField)
    
    def test_init_with_frame(self):
        """Test FrameSystem initialization with Frame."""
        frame = Frame()
        box = Box.cubic(10.0)
        ff = ForceField()
        
        system = FrameSystem(frame=frame, box=box, forcefield=ff)
        assert system._wrapped is frame
        assert system.box is box
        assert system.forcefield is ff
    
    def test_properties(self):
        """Test FrameSystem properties."""
        box = Box.cubic(10.0)
        ff = ForceField()
        system = FrameSystem(box=box, forcefield=ff)
        
        assert system.box is box
        assert system.forcefield is ff


class TestStructSystem:
    """Test cases for StructSystem class."""
    
    def test_init(self):
        """Test StructSystem initialization."""
        from struct import Struct
        
        struct = Struct('if')
        box = Box.cubic(10.0)
        ff = ForceField()
        
        system = StructSystem(struct=struct, box=box, forcefield=ff)
        assert system._wrapped is struct
        assert system.box is box
        assert system.forcefield is ff


class TestPeriodicSystem:
    """Test cases for PeriodicSystem class."""
    
    def setup_method(self):
        """Set up test fixtures."""
        # Create a simple frame with some atoms
        self.frame = Frame()
        atom_block = Block({
            'xyz': np.array([[0.0, 0.0, 0.0], [1.0, 1.0, 1.0]]),
            'type': np.array(['C', 'O']),
            'id': np.array([1, 2])
        })
        self.frame['atoms'] = atom_block
        
        # Create a periodic box
        self.box = Box.cubic(5.0, pbc=np.ones(3, dtype=bool))
        self.ff = ForceField()
        
        # Create frame system
        self.frame_system = FrameSystem(frame=self.frame, box=self.box, forcefield=self.ff)
    
    def test_init_with_periodic_system(self):
        """Test PeriodicSystem initialization with periodic system."""
        periodic_system = PeriodicSystem(self.frame_system)
        assert periodic_system._wrapped is self.frame_system
        assert periodic_system.box is self.box
        assert periodic_system.forcefield is self.ff
    
    def test_init_with_non_periodic_system(self):
        """Test PeriodicSystem initialization with non-periodic system."""
        non_periodic_box = Box.cubic(5.0, pbc=np.zeros(3, dtype=bool))
        non_periodic_system = FrameSystem(box=non_periodic_box)
        
        with pytest.raises(ValueError, match="must have periodic boundary conditions"):
            PeriodicSystem(non_periodic_system)
    
    def test_make_supercell_simple(self):
        """Test supercell creation with simple diagonal matrix."""
        periodic_system = PeriodicSystem(self.frame_system)
        
        # Create 2x2x2 supercell
        transformation_matrix = [[2, 0, 0], [0, 2, 0], [0, 0, 2]]
        supercell = periodic_system.make_supercell(transformation_matrix)
        
        assert isinstance(supercell, PeriodicSystem)
        assert isinstance(supercell._wrapped, FrameSystem)
        
        # Check that atoms are replicated (2 original atoms * 8 cells = 16 atoms)
        new_frame = supercell._wrapped._wrapped
        assert len(new_frame['atoms']['xyz']) == 16
        assert len(new_frame['atoms']['type']) == 16
        assert len(new_frame['atoms']['id']) == 16
    
    def test_make_supercell_asymmetric(self):
        """Test supercell creation with asymmetric matrix."""
        periodic_system = PeriodicSystem(self.frame_system)
        
        # Create 4x2x3 supercell
        transformation_matrix = [[4, 0, 0], [0, 2, 0], [0, 0, 3]]
        supercell = periodic_system.make_supercell(transformation_matrix)
        
        assert isinstance(supercell, PeriodicSystem)
        
        # Check that atoms are replicated (2 original atoms * 24 cells = 48 atoms)
        new_frame = supercell._wrapped._wrapped
        assert len(new_frame['atoms']['xyz']) == 48
    
    def test_make_supercell_box_transformation(self):
        """Test that box is properly transformed."""
        periodic_system = PeriodicSystem(self.frame_system)
        
        # Create 2x2x2 supercell
        transformation_matrix = [[2, 0, 0], [0, 2, 0], [0, 0, 2]]
        supercell = periodic_system.make_supercell(transformation_matrix)
        
        # Original box is 5x5x5, new box should be 10x10x10
        original_lengths = self.box.lengths
        new_lengths = supercell.box.lengths
        
        assert_array_almost_equal(new_lengths, original_lengths * 2)
    
    def test_make_supercell_invalid_matrix(self):
        """Test error handling for invalid transformation matrices."""
        periodic_system = PeriodicSystem(self.frame_system)
        
        # Test invalid dimensions
        with pytest.raises(ValueError, match="must be 3x3"):
            periodic_system.make_supercell([[2, 0], [0, 2]])
        
        # Test negative determinant
        with pytest.raises(ValueError, match="positive determinant"):
            periodic_system.make_supercell([[1, 0, 0], [0, 1, 0], [0, 0, -1]])
    
    def test_position_replication(self):
        """Test that positions are correctly replicated and translated."""
        periodic_system = PeriodicSystem(self.frame_system)
        
        # Create 2x1x1 supercell
        transformation_matrix = [[2, 0, 0], [0, 1, 0], [0, 0, 1]]
        supercell = periodic_system.make_supercell(transformation_matrix)
        
        new_frame = supercell._wrapped._wrapped
        new_positions = new_frame['atoms']['xyz']
        
        # Should have 4 atoms (2 original * 2 cells)
        assert len(new_positions) == 4
        
        # Check that positions are correctly translated
        # Original positions: [0,0,0] and [1,1,1]
        # After replication: [0,0,0], [1,1,1], [5,0,0], [6,1,1]
        expected_positions = np.array([
            [0.0, 0.0, 0.0],
            [1.0, 1.0, 1.0],
            [5.0, 0.0, 0.0],  # translated by box length in x
            [6.0, 1.0, 1.0]   # translated by box length in x
        ])
        
        # Sort both arrays to compare (order might vary)
        new_positions_sorted = new_positions[np.lexsort(new_positions.T)]
        expected_positions_sorted = expected_positions[np.lexsort(expected_positions.T)]
        
        assert_array_almost_equal(new_positions_sorted, expected_positions_sorted)
    
    def test_property_replication(self):
        """Test that atomic properties are correctly replicated."""
        periodic_system = PeriodicSystem(self.frame_system)
        
        # Create 2x1x1 supercell
        transformation_matrix = [[2, 0, 0], [0, 1, 0], [0, 0, 1]]
        supercell = periodic_system.make_supercell(transformation_matrix)
        
        new_frame = supercell._wrapped._wrapped
        
        # Check that types are replicated
        new_types = new_frame['atoms']['type']
        assert len(new_types) == 4
        assert list(new_types) == ['C', 'O', 'C', 'O']
        
        # Check that IDs are replicated
        new_ids = new_frame['atoms']['id']
        assert len(new_ids) == 4
        assert list(new_ids) == [1, 2, 1, 2]
    
    def test_add_vacuum_default_z(self):
        """Test adding vacuum in the default z direction."""
        periodic_system = PeriodicSystem(self.frame_system)
        
        # Add 25.0 units of vacuum in z direction
        vacuum_system = periodic_system.add_vacuum(25.0)
        
        assert isinstance(vacuum_system, PeriodicSystem)
        
        # Check that box dimensions changed correctly
        original_box = self.box
        new_box = vacuum_system.box
        
        # x and y dimensions should remain the same
        assert new_box.lengths[0] == original_box.lengths[0]
        assert new_box.lengths[1] == original_box.lengths[1]
        
        # z dimension should be increased by 25.0
        assert new_box.lengths[2] == original_box.lengths[2] + 25.0
        
        # Atoms should remain in the same positions
        original_frame = self.frame_system._wrapped
        new_frame = vacuum_system._wrapped._wrapped
        
        np.testing.assert_array_equal(
            original_frame['atoms']['xyz'], 
            new_frame['atoms']['xyz']
        )
    
    def test_add_vacuum_x_direction(self):
        """Test adding vacuum in x direction."""
        periodic_system = PeriodicSystem(self.frame_system)
        
        # Add 10.0 units of vacuum in x direction
        vacuum_system = periodic_system.add_vacuum(10.0, direction='x')
        
        # Check that box dimensions changed correctly
        original_box = self.box
        new_box = vacuum_system.box
        
        # x dimension should be increased by 10.0
        assert new_box.lengths[0] == original_box.lengths[0] + 10.0
        
        # y and z dimensions should remain the same
        assert new_box.lengths[1] == original_box.lengths[1]
        assert new_box.lengths[2] == original_box.lengths[2]
    
    def test_add_vacuum_y_direction(self):
        """Test adding vacuum in y direction."""
        periodic_system = PeriodicSystem(self.frame_system)
        
        # Add 15.0 units of vacuum in y direction
        vacuum_system = periodic_system.add_vacuum(15.0, direction='y')
        
        # Check that box dimensions changed correctly
        original_box = self.box
        new_box = vacuum_system.box
        
        # y dimension should be increased by 15.0
        assert new_box.lengths[1] == original_box.lengths[1] + 15.0
        
        # x and z dimensions should remain the same
        assert new_box.lengths[0] == original_box.lengths[0]
        assert new_box.lengths[2] == original_box.lengths[2]
    
    def test_add_vacuum_invalid_direction(self):
        """Test error handling for invalid direction."""
        periodic_system = PeriodicSystem(self.frame_system)
        
        with pytest.raises(ValueError, match="Direction must be"):
            periodic_system.add_vacuum(10.0, direction='invalid')
    
    def test_add_vacuum_negative_thickness(self):
        """Test error handling for negative vacuum thickness."""
        periodic_system = PeriodicSystem(self.frame_system)
        
        with pytest.raises(ValueError, match="Vacuum thickness must be positive"):
            periodic_system.add_vacuum(-5.0)
    
    def test_add_vacuum_zero_thickness(self):
        """Test error handling for zero vacuum thickness."""
        periodic_system = PeriodicSystem(self.frame_system)
        
        with pytest.raises(ValueError, match="Vacuum thickness must be positive"):
            periodic_system.add_vacuum(0.0)
    
    def test_add_vacuum_preserves_properties(self):
        """Test that atomic properties are preserved when adding vacuum."""
        periodic_system = PeriodicSystem(self.frame_system)
        
        vacuum_system = periodic_system.add_vacuum(20.0)
        
        # Check that all atomic properties are preserved
        original_frame = self.frame_system._wrapped
        new_frame = vacuum_system._wrapped._wrapped
        
        # Check all properties in the atoms block
        for var_name in original_frame['atoms']:
            np.testing.assert_array_equal(
                original_frame['atoms'][var_name],
                new_frame['atoms'][var_name]
            )
    
    def test_add_vacuum_maintains_periodicity(self):
        """Test that periodicity is maintained after adding vacuum."""
        periodic_system = PeriodicSystem(self.frame_system)
        
        vacuum_system = periodic_system.add_vacuum(30.0)
        
        # Check that the system is still periodic
        assert vacuum_system.box.is_periodic
        
        # Check that PBC settings are preserved
        np.testing.assert_array_equal(
            periodic_system.box.pbc,
            vacuum_system.box.pbc
        )


class TestPeriodicSystemIntegration:
    """Integration tests for PeriodicSystem with various scenarios."""
    
    def test_zno_surface_example(self):
        """Test the original use case: ZnO surface supercell."""
        # Create a simple ZnO-like structure
        frame = Frame()
        
        # Simple 2x2 ZnO unit cell
        positions = np.array([
            [0.0, 0.0, 0.0],  # Zn
            [1.0, 0.0, 0.0],  # O
            [0.0, 1.0, 0.0],  # Zn
            [1.0, 1.0, 0.0],  # O
        ])
        
        elements = np.array(['Zn', 'O', 'Zn', 'O'])
        
        atom_block = Block({
            'xyz': positions,
            'element': elements,
            'id': np.arange(1, 5)
        })
        
        frame['atoms'] = atom_block
        
        # Create periodic box
        box = Box.orth([2.0, 2.0, 1.0], pbc=np.ones(3, dtype=bool))
        
        # Create systems
        frame_system = FrameSystem(frame=frame, box=box)
        zno_periodic = PeriodicSystem(frame_system)
        
        # Create 4x2x3 supercell as in the original request
        supercell = zno_periodic.make_supercell([[4, 0, 0], [0, 2, 0], [0, 0, 3]])
        
        # Check results
        new_frame = supercell._wrapped._wrapped
        assert len(new_frame['atoms']['xyz']) == 4 * 4 * 2 * 3  # 96 atoms
        
        # Check box transformation
        new_box = supercell.box
        expected_lengths = np.array([8.0, 4.0, 3.0])  # 2*4, 2*2, 1*3
        assert_array_almost_equal(new_box.lengths, expected_lengths)
    
    def test_chaining_supercells(self):
        """Test creating supercells from supercells."""
        frame = Frame()
        atom_block = Block({
            'xyz': np.array([[0.0, 0.0, 0.0]]),
            'type': np.array(['C'])
        })
        frame['atoms'] = atom_block
        
        box = Box.cubic(1.0, pbc=np.ones(3, dtype=bool))
        frame_system = FrameSystem(frame=frame, box=box)
        periodic_system = PeriodicSystem(frame_system)
        
        # Create 2x2x2 supercell
        supercell1 = periodic_system.make_supercell([[2, 0, 0], [0, 2, 0], [0, 0, 2]])
        
        # Create another 2x2x2 supercell from the first one
        supercell2 = supercell1.make_supercell([[2, 0, 0], [0, 2, 0], [0, 0, 2]])
        
        # Should have 64 atoms (1 * 8 * 8)
        new_frame = supercell2._wrapped._wrapped
        assert len(new_frame['atoms']['xyz']) == 64
        
        # Box should be 4x4x4
        assert_array_almost_equal(supercell2.box.lengths, [4.0, 4.0, 4.0])
    
    def test_zno_surface_with_vacuum(self):
        """Test the complete workflow: ZnO surface + supercell + vacuum."""
        # Create a simple ZnO-like structure
        frame = Frame()
        
        # Simple 2x2 ZnO unit cell
        positions = np.array([
            [0.0, 0.0, 0.0],  # Zn
            [1.0, 0.0, 0.0],  # O
            [0.0, 1.0, 0.0],  # Zn
            [1.0, 1.0, 0.0],  # O
        ])
        
        elements = np.array(['Zn', 'O', 'Zn', 'O'])
        
        atom_block = Block({
            'xyz': positions,
            'element': elements,
            'id': np.arange(1, 5)
        })
        
        frame['atoms'] = atom_block
        
        # Create periodic box
        box = Box.orth([2.0, 2.0, 1.0], pbc=np.ones(3, dtype=bool))
        
        # Create systems
        frame_system = FrameSystem(frame=frame, box=box)
        zno_periodic = PeriodicSystem(frame_system)
        
        # Create 4x2x3 supercell
        supercell = zno_periodic.make_supercell([[4, 0, 0], [0, 2, 0], [0, 0, 3]])
        
        # Add 25.0 units of vacuum in z direction
        final_system = supercell.add_vacuum(25.0)
        
        # Check results
        new_frame = final_system._wrapped._wrapped
        new_box = final_system.box
        
        # Should have same number of atoms as supercell
        assert len(new_frame['atoms']['xyz']) == 4 * 4 * 2 * 3  # 96 atoms
        
        # Box dimensions should be: [8.0, 4.0, 3.0 + 25.0]
        expected_lengths = np.array([8.0, 4.0, 28.0])
        assert_array_almost_equal(new_box.lengths, expected_lengths)
        
        # Atoms should be in the same positions as the supercell
        supercell_frame = supercell._wrapped._wrapped
        np.testing.assert_array_equal(
            supercell_frame['atoms']['xyz'],
            new_frame['atoms']['xyz']
        )
    
    def test_vacuum_and_supercell_chaining(self):
        """Test chaining vacuum and supercell operations."""
        frame = Frame()
        atom_block = Block({
            'xyz': np.array([[0.0, 0.0, 0.0], [1.0, 1.0, 0.5]]),
            'type': np.array(['A', 'B'])
        })
        frame['atoms'] = atom_block
        
        box = Box.cubic(2.0, pbc=np.ones(3, dtype=bool))
        frame_system = FrameSystem(frame=frame, box=box)
        periodic_system = PeriodicSystem(frame_system)
        
        # Chain operations: supercell -> vacuum -> supercell
        step1 = periodic_system.make_supercell([[2, 0, 0], [0, 1, 0], [0, 0, 1]])
        step2 = step1.add_vacuum(10.0, direction='z')
        step3 = step2.make_supercell([[1, 0, 0], [0, 2, 0], [0, 0, 1]])
        
        final_frame = step3._wrapped._wrapped
        final_box = step3.box
        
        # Should have 2 * 2 * 2 = 8 atoms
        assert len(final_frame['atoms']['xyz']) == 8
        
        # Box should be [4.0, 4.0, 12.0] (2*2, 1*2, (2+10)*1)
        expected_lengths = np.array([4.0, 4.0, 12.0])
        assert_array_almost_equal(final_box.lengths, expected_lengths)
