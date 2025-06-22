"""
Unit tests for the packer.target module.
Tests Target class functionality.
"""

import pytest
import numpy as np
import molpy as mp
from molpy.pack.target import Target
from molpy.pack import constraint as mpk_constraint


class TestTarget:
    """Test Target class functionality."""

    def setup_method(self):
        """Set up test fixtures."""
        # Create a simple frame with atoms
        self.frame = mp.Frame()
        atoms_data = {
            'id': [0, 1, 2],
            'name': ['C1', 'C2', 'C3'],
            'x': [0.0, 1.0, 2.0],
            'y': [0.0, 0.0, 0.0],
            'z': [0.0, 0.0, 0.0]
        }
        self.frame["atoms"] = atoms_data

        # Create a simple constraint
        self.constraint = mpk_constraint.InsideBoxConstraint([5.0, 5.0, 5.0], [0.0, 0.0, 0.0])

    def test_target_initialization(self):
        """Test Target initialization with basic parameters."""
        target = Target(
            frame=self.frame,
            number=2,
            constraint=self.constraint,
            name="test_target"
        )
        
        assert target.frame == self.frame
        assert target.number == 2
        assert target.constraint == self.constraint
        assert target.name == "test_target"
        assert target.is_fixed is False
        assert target.optimizer is None

    def test_target_initialization_with_all_parameters(self):
        """Test Target initialization with all parameters."""
        target = Target(
            frame=self.frame,
            number=3,
            constraint=self.constraint,
            is_fixed=True,
            optimizer="test_optimizer",
            name="full_target"
        )
        
        assert target.frame == self.frame
        assert target.number == 3
        assert target.constraint == self.constraint
        assert target.is_fixed is True
        assert target.optimizer == "test_optimizer"
        assert target.name == "full_target"

    def test_target_n_points_property(self):
        """Test n_points property calculation."""
        target = Target(
            frame=self.frame,
            number=4,
            constraint=self.constraint
        )
        
        # Frame has 3 atoms, target number is 4
        expected_points = 3 * 4
        assert target.n_points == expected_points

    def test_target_points_property_xyz_format(self):
        """Test points property with x,y,z coordinate format."""
        target = Target(
            frame=self.frame,
            number=1,
            constraint=self.constraint
        )
        
        points = target.points
        assert isinstance(points, np.ndarray)
        assert points.shape == (3, 3)  # 3 atoms * 1 copy, 3 coordinates each
        
        # Check first atom coordinates
        assert np.allclose(points[0], [0.0, 0.0, 0.0])
        assert np.allclose(points[1], [1.0, 0.0, 0.0])
        assert np.allclose(points[2], [2.0, 0.0, 0.0])

    def test_target_points_property_xyz_array_format(self):
        """Test points property with xyz array coordinate format."""
        # Create frame with xyz array format
        frame_xyz = mp.Frame()
        atoms_data = {
            'id': [0, 1],
            'name': ['A', 'B'],
            'xyz': [[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]]
        }
        frame_xyz["atoms"] = atoms_data
        
        target = Target(
            frame=frame_xyz,
            number=1,
            constraint=self.constraint
        )
        
        points = target.points
        assert isinstance(points, np.ndarray)
        assert points.shape == (2, 3)  # 2 atoms, 3 coordinates each
        
        # Check atom coordinates
        assert np.allclose(points[0], [1.0, 2.0, 3.0])
        assert np.allclose(points[1], [4.0, 5.0, 6.0])

    def test_target_points_property_missing_coordinates(self):
        """Test points property when coordinates are missing."""
        # Create frame without proper coordinates
        frame_bad = mp.Frame()
        atoms_data = {
            'id': [0, 1],
            'name': ['A', 'B']
            # Missing coordinates
        }
        frame_bad["atoms"] = atoms_data
        
        target = Target(
            frame=frame_bad,
            number=1,
            constraint=self.constraint
        )
        
        with pytest.raises(ValueError, match="Frame must contain either 'xyz' or 'x', 'y', 'z' coordinates"):
            _ = target.points

    def test_target_repr(self):
        """Test Target string representation."""
        target = Target(
            frame=self.frame,
            number=2,
            constraint=self.constraint,
            name="test_molecule"
        )
        
        repr_str = repr(target)
        assert "test_molecule" in repr_str
        assert "3 atoms" in repr_str  # Frame has 3 atoms

    def test_target_empty_name(self):
        """Test Target with empty name."""
        target = Target(
            frame=self.frame,
            number=1,
            constraint=self.constraint,
            name=""
        )
        
        repr_str = repr(target)
        assert "Target :" in repr_str  # Empty name should show as empty

    def test_target_with_different_constraints(self):
        """Test Target with different constraint types."""
        # Box constraint
        box_constraint = mpk_constraint.InsideBoxConstraint([2.0, 2.0, 2.0], [0.0, 0.0, 0.0])
        target_box = Target(
            frame=self.frame,
            number=1,
            constraint=box_constraint
        )
        assert target_box.constraint == box_constraint
        
        # Sphere constraint
        sphere_constraint = mpk_constraint.InsideSphereConstraint(1.5, [0.0, 0.0, 0.0])
        target_sphere = Target(
            frame=self.frame,
            number=1,
            constraint=sphere_constraint
        )
        assert target_sphere.constraint == sphere_constraint
        
        # Combined constraint
        combined_constraint = box_constraint & sphere_constraint
        target_combined = Target(
            frame=self.frame,
            number=1,
            constraint=combined_constraint
        )
        assert target_combined.constraint == combined_constraint

    def test_target_with_zero_atoms(self):
        """Test Target with frame containing zero atoms."""
        empty_frame = mp.Frame()
        empty_frame["atoms"] = {'id': [], 'x': [], 'y': [], 'z': []}
        
        target = Target(
            frame=empty_frame,
            number=5,
            constraint=self.constraint
        )
        
        assert target.n_points == 0  # 0 atoms * 5 = 0
        points = target.points
        assert points.shape == (0, 3)  # Empty array with correct shape

    def test_target_with_large_number(self):
        """Test Target with large number parameter."""
        target = Target(
            frame=self.frame,
            number=1000,
            constraint=self.constraint
        )
        
        # Should handle large numbers correctly
        assert target.number == 1000
        assert target.n_points == 3 * 1000  # 3 atoms * 1000

    def test_target_n_points_calculation_edge_cases(self):
        """Test n_points calculation with edge cases."""
        # Single atom, single copy
        single_atom_frame = mp.Frame()
        single_atom_frame["atoms"] = {'id': [0], 'x': [0.0], 'y': [0.0], 'z': [0.0]}
        
        target_single = Target(
            frame=single_atom_frame,
            number=1,
            constraint=self.constraint
        )
        assert target_single.n_points == 1
        
        # Single atom, multiple copies
        target_multiple = Target(
            frame=single_atom_frame,
            number=10,
            constraint=self.constraint
        )
        assert target_multiple.n_points == 10

    def test_target_immutability_of_frame(self):
        """Test that Target doesn't modify the original frame."""
        original_atoms = self.frame["atoms"].copy()
        
        target = Target(
            frame=self.frame,
            number=2,
            constraint=self.constraint
        )
        
        # Access points property (which processes the frame)
        _ = target.points
        
        # Original frame should be unchanged
        assert self.frame["atoms"].equals(original_atoms)

    def test_target_with_complex_frame(self):
        """Test Target with frame containing bonds, angles, etc."""
        complex_frame = mp.Frame()
        
        # Add atoms
        atoms_data = {
            'id': [0, 1, 2, 3],
            'name': ['C1', 'C2', 'O1', 'H1'],
            'x': [0.0, 1.0, 2.0, 3.0],
            'y': [0.0, 0.0, 1.0, 0.0],
            'z': [0.0, 0.0, 0.0, 0.0],
            'type': [1, 1, 2, 3]
        }
        complex_frame["atoms"] = atoms_data
        
        # Add bonds
        bonds_data = {
            'id': [0, 1, 2],
            'i': [0, 1, 2],
            'j': [1, 2, 3]
        }
        complex_frame["bonds"] = bonds_data
        
        target = Target(
            frame=complex_frame,
            number=2,
            constraint=self.constraint,
            name="complex_molecule"
        )
        
        assert target.n_points == 8  # 4 atoms * 2 copies
        points = target.points
        assert points.shape == (8, 3)  # 4 atoms * 2 copies, 3 coordinates
