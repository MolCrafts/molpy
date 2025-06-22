"""
Unit tests for the packer.constraint module.
Tests various constraint classes and their logic.
"""

import pytest
import numpy as np
import molpy as mp
from molpy.pack import constraint as mpk_constraint


class TestBaseConstraint:
    """Test the base Constraint class and logic operations."""

    def test_constraint_and_operation(self):
        """Test the & (and) operation between constraints."""
        box1 = mpk_constraint.InsideBoxConstraint([2.0, 2.0, 2.0], [0.0, 0.0, 0.0])
        box2 = mpk_constraint.InsideBoxConstraint([1.5, 1.5, 1.5], [0.5, 0.5, 0.5])
        
        combined = box1 & box2
        assert isinstance(combined, mpk_constraint.AndConstraint)
        assert combined.a == box1
        assert combined.b == box2

    def test_constraint_or_operation(self):
        """Test the | (or) operation between constraints."""
        box1 = mpk_constraint.InsideBoxConstraint([1.0, 1.0, 1.0], [0.0, 0.0, 0.0])
        box2 = mpk_constraint.InsideBoxConstraint([1.0, 1.0, 1.0], [2.0, 2.0, 2.0])
        
        combined = box1 | box2
        assert isinstance(combined, mpk_constraint.OrConstraint)
        assert combined.a == box1
        assert combined.b == box2


class TestAndConstraint:
    """Test AndConstraint logic."""

    def test_and_constraint_penalty(self):
        """Test that AndConstraint combines penalties correctly."""
        # Create two box constraints
        box1 = mpk_constraint.InsideBoxConstraint([2.0, 2.0, 2.0], [0.0, 0.0, 0.0])
        box2 = mpk_constraint.InsideBoxConstraint([1.5, 1.5, 1.5], [0.5, 0.5, 0.5])
        
        and_constraint = mpk_constraint.AndConstraint(box1, box2)
        
        # Test points inside both boxes (should have low penalty)
        points_inside = np.array([[1.0, 1.0, 1.0]])
        penalty_inside = and_constraint.penalty(points_inside)
        
        # Test points outside one box (should have higher penalty)
        points_outside = np.array([[3.0, 3.0, 3.0]])
        penalty_outside = and_constraint.penalty(points_outside)
        
        assert penalty_outside > penalty_inside

    def test_and_constraint_gradient(self):
        """Test that AndConstraint combines gradients correctly."""
        box1 = mpk_constraint.InsideBoxConstraint([2.0, 2.0, 2.0], [0.0, 0.0, 0.0])
        box2 = mpk_constraint.InsideBoxConstraint([1.5, 1.5, 1.5], [0.5, 0.5, 0.5])
        
        and_constraint = mpk_constraint.AndConstraint(box1, box2)
        
        points = np.array([[3.0, 3.0, 3.0]])
        grad_and = and_constraint.dpenalty(points)
        grad1 = box1.dpenalty(points)
        grad2 = box2.dpenalty(points)
        
        # Gradient should be sum of individual gradients
        assert np.allclose(grad_and, grad1 + grad2)


class TestOrConstraint:
    """Test OrConstraint logic."""

    def test_or_constraint_penalty(self):
        """Test that OrConstraint takes minimum penalty."""
        # Create two separate boxes
        box1 = mpk_constraint.InsideBoxConstraint([1.0, 1.0, 1.0], [0.0, 0.0, 0.0])
        box2 = mpk_constraint.InsideBoxConstraint([1.0, 1.0, 1.0], [3.0, 3.0, 3.0])
        
        or_constraint = mpk_constraint.OrConstraint(box1, box2)
        
        # Point inside first box but outside second
        points = np.array([[0.5, 0.5, 0.5]])
        penalty_or = or_constraint.penalty(points)
        penalty1 = box1.penalty(points)
        penalty2 = box2.penalty(points)
        
        # Should take minimum penalty
        assert penalty_or == min(penalty1, penalty2)

    def test_or_constraint_gradient(self):
        """Test that OrConstraint uses gradient from constraint with lower penalty."""
        box1 = mpk_constraint.InsideBoxConstraint([1.0, 1.0, 1.0], [0.0, 0.0, 0.0])
        box2 = mpk_constraint.InsideBoxConstraint([1.0, 1.0, 1.0], [3.0, 3.0, 3.0])
        
        or_constraint = mpk_constraint.OrConstraint(box1, box2)
        
        # Point closer to first box
        points = np.array([[0.5, 0.5, 0.5]])
        grad_or = or_constraint.dpenalty(points)
        grad1 = box1.dpenalty(points)
        
        # Should use gradient from constraint with lower penalty
        assert np.allclose(grad_or, grad1)


class TestInsideBoxConstraint:
    """Test InsideBoxConstraint functionality."""

    def test_inside_box_penalty_points_inside(self):
        """Test penalty for points inside the box."""
        constraint = mpk_constraint.InsideBoxConstraint([2.0, 2.0, 2.0], [0.0, 0.0, 0.0])
        
        # Points inside the box
        points_inside = np.array([
            [0.5, 0.5, 0.5],
            [1.0, 1.0, 1.0],
            [1.5, 1.5, 1.5]
        ])
        
        penalty = constraint.penalty(points_inside)
        assert penalty == 0.0  # No penalty for points inside

    def test_inside_box_penalty_points_outside(self):
        """Test penalty for points outside the box."""
        constraint = mpk_constraint.InsideBoxConstraint([2.0, 2.0, 2.0], [0.0, 0.0, 0.0])
        
        # Points outside the box
        points_outside = np.array([
            [-0.5, 0.5, 0.5],  # Outside in x
            [2.5, 1.0, 1.0],   # Outside in x
            [1.0, 1.0, 2.5]    # Outside in z
        ])
        
        penalty = constraint.penalty(points_outside)
        assert penalty == 3.0  # Three points outside

    def test_inside_box_gradient(self):
        """Test gradient calculation for inside box constraint."""
        constraint = mpk_constraint.InsideBoxConstraint([2.0, 2.0, 2.0], [0.0, 0.0, 0.0])
        
        # Point outside in positive x direction
        points = np.array([[2.5, 1.0, 1.0]])
        grad = constraint.dpenalty(points)
        
        # Should have negative gradient in x to push back inside
        assert grad[0, 0] < 0
        assert grad[0, 1] == 0
        assert grad[0, 2] == 0

    def test_inside_box_invert(self):
        """Test that inverting creates OutsideBoxConstraint."""
        constraint = mpk_constraint.InsideBoxConstraint([2.0, 2.0, 2.0], [0.0, 0.0, 0.0])
        inverted = ~constraint
        
        assert isinstance(inverted, mpk_constraint.OutsideBoxConstraint)
        assert np.allclose(inverted.origin, [0.0, 0.0, 0.0])


class TestOutsideBoxConstraint:
    """Test OutsideBoxConstraint functionality."""

    def test_outside_box_penalty_points_outside(self):
        """Test penalty for points outside the box (should be zero)."""
        constraint = mpk_constraint.OutsideBoxConstraint([0.0, 0.0, 0.0], [2.0, 2.0, 2.0])
        
        # Points outside the box
        points_outside = np.array([
            [-0.5, 0.5, 0.5],
            [2.5, 1.0, 1.0],
            [1.0, 1.0, 2.5]
        ])
        
        penalty = constraint.penalty(points_outside)
        assert penalty == 0.0  # No penalty for points outside

    def test_outside_box_penalty_points_inside(self):
        """Test penalty for points inside the box (should be positive)."""
        constraint = mpk_constraint.OutsideBoxConstraint([0.0, 0.0, 0.0], [2.0, 2.0, 2.0])
        
        # Points inside the box
        points_inside = np.array([
            [0.5, 0.5, 0.5],
            [1.0, 1.0, 1.0]
        ])
        
        penalty = constraint.penalty(points_inside)
        assert penalty == 2.0  # Two points inside

    def test_outside_box_invert(self):
        """Test that inverting creates InsideBoxConstraint."""
        constraint = mpk_constraint.OutsideBoxConstraint([0.0, 0.0, 0.0], [2.0, 2.0, 2.0])
        inverted = ~constraint
        
        assert isinstance(inverted, mpk_constraint.InsideBoxConstraint)


class TestInsideSphereConstraint:
    """Test InsideSphereConstraint functionality."""

    def test_inside_sphere_penalty_points_inside(self):
        """Test penalty for points inside the sphere."""
        constraint = mpk_constraint.InsideSphereConstraint(1.0, [0.0, 0.0, 0.0])
        
        # Points inside the sphere
        points_inside = np.array([
            [0.0, 0.0, 0.0],     # Center
            [0.5, 0.0, 0.0],     # Inside
            [0.0, 0.8, 0.0]      # Inside
        ])
        
        penalty = constraint.penalty(points_inside)
        assert penalty == 0.0  # No penalty for points inside

    def test_inside_sphere_penalty_points_outside(self):
        """Test penalty for points outside the sphere."""
        constraint = mpk_constraint.InsideSphereConstraint(1.0, [0.0, 0.0, 0.0])
        
        # Points outside the sphere
        points_outside = np.array([
            [1.5, 0.0, 0.0],     # Outside
            [0.0, 1.5, 0.0],     # Outside
            [1.0, 1.0, 0.0]      # Outside (distance > 1)
        ])
        
        penalty = constraint.penalty(points_outside)
        assert penalty == 3.0  # Three points outside

    def test_inside_sphere_gradient(self):
        """Test gradient calculation for inside sphere constraint."""
        constraint = mpk_constraint.InsideSphereConstraint(1.0, [0.0, 0.0, 0.0])
        
        # Point outside the sphere
        points = np.array([[1.5, 0.0, 0.0]])
        grad = constraint.dpenalty(points)
        
        # Should have gradient pointing toward center
        assert grad[0, 0] < 0  # Negative x direction
        assert grad[0, 1] == 0
        assert grad[0, 2] == 0

    def test_inside_sphere_invert(self):
        """Test that inverting creates OutsideSphereConstraint."""
        constraint = mpk_constraint.InsideSphereConstraint(1.0, [0.0, 0.0, 0.0])
        inverted = ~constraint
        
        assert isinstance(inverted, mpk_constraint.OutsideSphereConstraint)
        assert inverted.radius == 1.0
        assert np.allclose(inverted.center, [0.0, 0.0, 0.0])


class TestOutsideSphereConstraint:
    """Test OutsideSphereConstraint functionality."""

    def test_outside_sphere_penalty_points_outside(self):
        """Test penalty for points outside the sphere (should be zero)."""
        constraint = mpk_constraint.OutsideSphereConstraint(1.0, [0.0, 0.0, 0.0])
        
        # Points outside the sphere
        points_outside = np.array([
            [1.5, 0.0, 0.0],
            [0.0, 1.5, 0.0],
            [1.0, 1.0, 0.0]
        ])
        
        penalty = constraint.penalty(points_outside)
        assert penalty == 0.0  # No penalty for points outside

    def test_outside_sphere_penalty_points_inside(self):
        """Test penalty for points inside the sphere (should be positive)."""
        constraint = mpk_constraint.OutsideSphereConstraint(1.0, [0.0, 0.0, 0.0])
        
        # Points inside the sphere
        points_inside = np.array([
            [0.0, 0.0, 0.0],     # Center
            [0.5, 0.0, 0.0]      # Inside
        ])
        
        penalty = constraint.penalty(points_inside)
        assert penalty == 2.0  # Two points inside

    def test_outside_sphere_gradient(self):
        """Test gradient calculation for outside sphere constraint."""
        constraint = mpk_constraint.OutsideSphereConstraint(1.0, [0.0, 0.0, 0.0])
        
        # Point inside the sphere
        points = np.array([[0.5, 0.0, 0.0]])
        grad = constraint.dpenalty(points)
        
        # Should have gradient pointing outward from center
        assert grad[0, 0] < 0  # Negative x direction (away from center)
        assert grad[0, 1] == 0
        assert grad[0, 2] == 0

    def test_outside_sphere_invert(self):
        """Test that inverting creates InsideSphereConstraint."""
        constraint = mpk_constraint.OutsideSphereConstraint(1.0, [0.0, 0.0, 0.0])
        inverted = ~constraint
        
        assert isinstance(inverted, mpk_constraint.InsideSphereConstraint)
        assert inverted.radius == 1.0
        assert np.allclose(inverted.center, [0.0, 0.0, 0.0])


class TestMinDistanceConstraint:
    """Test MinDistanceConstraint functionality."""

    def test_min_distance_penalty_no_violations(self):
        """Test penalty when all distances are above minimum."""
        constraint = mpk_constraint.MinDistanceConstraint(1.0)
        
        # Points with sufficient distance
        points = np.array([
            [0.0, 0.0, 0.0],
            [1.5, 0.0, 0.0],
            [0.0, 1.5, 0.0]
        ])
        
        penalty = constraint.penalty(points)
        assert penalty == 0.0  # No violations

    def test_min_distance_penalty_with_violations(self):
        """Test penalty when some distances are below minimum."""
        constraint = mpk_constraint.MinDistanceConstraint(1.0)
        
        # Points too close together
        points = np.array([
            [0.0, 0.0, 0.0],
            [0.5, 0.0, 0.0],  # Distance 0.5 < 1.0
            [0.0, 0.5, 0.0]   # Distance sqrt(0.5) < 1.0
        ])
        
        penalty = constraint.penalty(points)
        assert penalty > 0.0  # Should have violations

    def test_min_distance_gradient(self):
        """Test gradient calculation for minimum distance constraint."""
        constraint = mpk_constraint.MinDistanceConstraint(1.0)
        
        # Two points too close
        points = np.array([
            [0.0, 0.0, 0.0],
            [0.5, 0.0, 0.0]
        ])
        
        grad = constraint.dpenalty(points)
        
        # Should have non-zero gradient
        assert not np.allclose(grad, 0.0)
        # Gradients should push points apart
        assert grad[0, 0] <= 0  # First point should move left
        assert grad[1, 0] >= 0  # Second point should move right


class TestConstraintCombinations:
    """Test combinations of constraints."""

    def test_complex_constraint_combination(self):
        """Test a complex combination of constraints."""
        # Create a constraint: inside box AND outside sphere
        box = mpk_constraint.InsideBoxConstraint([4.0, 4.0, 4.0], [-2.0, -2.0, -2.0])
        sphere = mpk_constraint.OutsideSphereConstraint(1.0, [0.0, 0.0, 0.0])
        
        combined = box & sphere
        
        # Point inside box but inside sphere (should have penalty)
        point_bad = np.array([[0.5, 0.0, 0.0]])
        penalty_bad = combined.penalty(point_bad)
        
        # Point inside box and outside sphere (should have no penalty)
        point_good = np.array([[1.5, 0.0, 0.0]])
        penalty_good = combined.penalty(point_good)
        
        assert penalty_bad > penalty_good

    def test_or_constraint_with_separate_regions(self):
        """Test OR constraint with completely separate regions."""
        # Two separate boxes
        box1 = mpk_constraint.InsideBoxConstraint([1.0, 1.0, 1.0], [0.0, 0.0, 0.0])
        box2 = mpk_constraint.InsideBoxConstraint([1.0, 1.0, 1.0], [3.0, 3.0, 3.0])
        
        or_constraint = box1 | box2
        
        # Point in middle (outside both boxes)
        point_middle = np.array([[1.5, 1.5, 1.5]])
        penalty_middle = or_constraint.penalty(point_middle)
        
        # Point in first box
        point_box1 = np.array([[0.5, 0.5, 0.5]])
        penalty_box1 = or_constraint.penalty(point_box1)
        
        # Point in second box  
        point_box2 = np.array([[3.5, 3.5, 3.5]])
        penalty_box2 = or_constraint.penalty(point_box2)
        
        assert penalty_middle > penalty_box1
        assert penalty_middle > penalty_box2
        assert penalty_box1 == 0.0
        assert penalty_box2 == 0.0
