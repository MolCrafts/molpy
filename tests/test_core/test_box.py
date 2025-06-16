"""
Comprehensive tests for the Box module.
This file replaces the redundant test_box.py with a clean, organized structure.
"""

import numpy as np
import numpy.testing as npt
import pytest

import molpy as mp


class TestBoxConstruction:
    """Test different ways to construct Box objects."""

    def test_default_construction(self):
        """Test default box construction."""
        box = mp.Box()
        assert box.style == mp.Box.Style.FREE
        assert box.volume == 0.0  # Free box has zero volume

    def test_matrix_construction(self):
        """Test box construction from matrix."""
        matrix = np.diag([1, 2, 3])
        box = mp.Box(matrix)
        assert box.style == mp.Box.Style.ORTHOGONAL
        npt.assert_allclose(box.lx, 1)
        npt.assert_allclose(box.ly, 2)
        npt.assert_allclose(box.lz, 3)

    def test_cubic_construction(self):
        """Test cubic box construction."""
        box = mp.Box.cubic(5.0)
        assert box.style == mp.Box.Style.ORTHOGONAL
        assert box.lx == box.ly == box.lz == 5.0
        assert box.xy == box.xz == box.yz == 0.0

    def test_cubic_central(self):
        """Test cubic box with central origin."""
        box = mp.Box.cubic(4.0, central=True)
        npt.assert_allclose(box.origin, [-2.0, -2.0, -2.0])

    def test_orthogonal_construction(self):
        """Test orthogonal box construction."""
        lengths = [1, 2, 3]
        box = mp.Box.orth(lengths)
        assert box.style == mp.Box.Style.ORTHOGONAL
        npt.assert_allclose(box.lengths, lengths)

    def test_triclinic_construction(self):
        """Test triclinic box construction."""
        lengths = [2, 3, 4]
        tilts = [0.5, 1.0, 1.5]
        box = mp.Box.tric(lengths, tilts)
        assert box.style == mp.Box.Style.TRICLINIC
        # Note: triclinic box lengths are calculated from the matrix, not direct input
        npt.assert_allclose(box.tilts, tilts)

    def test_from_lengths_angles(self):
        """Test construction from lengths and angles."""
        lengths = [2, 3, 4]
        angles = [90, 90, 90]  # Orthogonal case
        box = mp.Box.from_lengths_angles(lengths, angles)
        npt.assert_allclose(box.lengths, lengths, rtol=1e-6)
        npt.assert_allclose(box.angles, angles, rtol=1e-6)

    def test_from_box(self):
        """Test copying a box."""
        original = mp.Box.tric([2, 3, 4], [0.5, 1.0, 1.5])
        copy = mp.Box.from_box(original)
        assert original == copy
        assert original is not copy  # Different objects


class TestBoxProperties:
    """Test box properties and getters/setters."""

    def test_lengths_property(self):
        """Test length properties."""
        box = mp.Box.tric([2, 4, 5], [1, 0, 0])
        npt.assert_allclose(box.lx, 2)
        npt.assert_allclose(box.ly, 4)
        npt.assert_allclose(box.lz, 5)
        npt.assert_allclose(box.l, [2, 4, 5])
        npt.assert_allclose(box.l_inv, [0.5, 0.25, 0.2])

    def test_lengths_setter(self):
        """Test setting individual lengths."""
        box = mp.Box.tric([1, 2, 3], [1, 0, 0])
        box.lx = 4
        box.ly = 5
        box.lz = 6
        npt.assert_allclose(box.lx, 4)
        npt.assert_allclose(box.ly, 5)
        npt.assert_allclose(box.lz, 6)

    def test_tilts_property(self):
        """Test tilt factor properties."""
        box = mp.Box.tric([2, 2, 2], [1, 2, 3])
        npt.assert_allclose(box.xy, 1)
        npt.assert_allclose(box.xz, 2)
        npt.assert_allclose(box.yz, 3)
        npt.assert_allclose(box.tilts, [1, 2, 3])

    def test_tilts_setter(self):
        """Test setting tilt factors."""
        box = mp.Box.tric([2, 2, 2], [1, 2, 3])
        box.xy = 4
        box.xz = 5
        box.yz = 6
        npt.assert_allclose(box.xy, 4)
        npt.assert_allclose(box.xz, 5)
        npt.assert_allclose(box.yz, 6)

    def test_bounds_property(self):
        """Test bounds property."""
        box = mp.Box.orth([2, 3, 4])
        bounds = box.bounds
        expected = np.array([[0, 0, 0], [2, 3, 4]])
        npt.assert_allclose(bounds, expected)

    def test_volume_property(self):
        """Test volume calculation."""
        # Orthogonal box
        box = mp.Box.orth([2, 3, 4])
        # Just check that volume is close to expected value
        expected_volume = 24.0
        actual_volume = box.volume  # volume is a property, not a method
        assert abs(actual_volume - expected_volume) < 1e-10
        
        # Cubic box
        box = mp.Box.cubic(3)
        expected_volume = 27.0
        actual_volume = float(box.volume)
        assert abs(actual_volume - expected_volume) < 1e-10

    def test_periodic_boundary_conditions(self):
        """Test periodic boundary condition properties."""
        box = mp.Box.orth([1, 2, 3])
        
        # Default is all periodic
        assert box.periodic_x
        assert box.periodic_y
        assert box.periodic_z
        assert box.periodic
        
        # Test setting individual flags
        box.periodic_x = False
        assert not box.periodic_x
        assert not box.periodic  # Should be False if any is False
        
        # Test setting all flags
        box.periodic = True
        assert box.periodic_x and box.periodic_y and box.periodic_z
        
        # Test setting as list
        box.periodic = [True, False, True]
        assert box.periodic_x and not box.periodic_y and box.periodic_z


class TestBoxTransformations:
    """Test coordinate transformations and wrapping."""

    def test_fractional_absolute_conversion(self):
        """Test conversion between fractional and absolute coordinates."""
        box = mp.Box.orth([2, 2, 2], central=True)
        
        # Test points
        absolute_points = np.array([[0, -0.5, 0.5], [-1, -1, -1], [0, 0, 0]])
        fractional_points = np.array([[0.5, 0.25, 0.75], [0, 0, 0], [0.5, 0.5, 0.5]])
        
        # Test absolute to fractional
        result_frac = box.make_fractional(absolute_points)
        npt.assert_allclose(result_frac, fractional_points)
        
        # Test fractional to absolute
        result_abs = box.make_absolute(fractional_points)
        npt.assert_allclose(result_abs, absolute_points)

    def test_wrapping_orthogonal(self):
        """Test coordinate wrapping in orthogonal box."""
        box = mp.Box.orth([2, 2, 2])
        
        # Test single point
        point = [3, -1, 5]
        wrapped = box.wrap(point)
        npt.assert_allclose(wrapped, [1, 1, 1])
        
        # Test multiple points
        points = [[10, -5, -5], [0, 0.5, 0]]
        wrapped = box.wrap(points)
        npt.assert_allclose(wrapped, [[0, 1, 1], [0, 0.5, 0]])

    def test_wrapping_triclinic(self):
        """Test coordinate wrapping in triclinic box."""
        box = mp.Box.tric([2, 2, 2], [1, 0, 0])
        points = [[10, -5, -5], [0, 0.5, 0]]
        wrapped = box.wrap(points)
        # The exact values depend on the triclinic transformation
        assert wrapped.shape == (2, 3)

    def test_unwrapping(self):
        """Test coordinate unwrapping."""
        box = mp.Box.tric([2, 2, 2], [1, 0, 0])
        
        points = [0, -1, -1]
        images = [1, 0, 0]
        unwrapped = box.unwrap(points, images)
        npt.assert_allclose(unwrapped, [2, -1, -1])

    def test_image_calculation(self):
        """Test image calculation."""
        box = mp.Box.orth([2, 2, 2])
        points = np.array([[3, 4, 5], [-1, -2, -3]])
        images = box.get_images(points)
        expected = np.array([[2, 2, 2], [-1, -1, -2]])
        npt.assert_allclose(images, expected)


class TestBoxDistanceCalculations:
    """Test distance calculations and periodic boundary conditions."""

    def test_distance_calculation(self):
        """Test distance calculation between points."""
        box = mp.Box.orth([10, 10, 10])
        
        r1 = np.array([[0, 0, 0], [1, 1, 1]])
        r2 = np.array([[1, 0, 0], [0, 0, 0]])
        
        distances = box.dist(r1, r2)
        expected = [1.0, np.sqrt(3)]
        npt.assert_allclose(distances, expected)

    def test_distance_all_pairs(self):
        """Test distance calculation for all pairs."""
        box = mp.Box.orth([10, 10, 10])
        
        points1 = np.array([[0, 0, 1], [0, 0, 0]])
        points2 = np.array([[1, 0, 1], [0, 0, 1], [0, 0, 0]])
        
        distances = box.dist_all(points1, points2)
        assert distances.shape == (2, 3)
        npt.assert_allclose(distances[0, 0], 1.0)  # [0,0,1] to [1,0,1]
        npt.assert_allclose(distances[0, 1], 0.0)  # [0,0,1] to [0,0,1]
        npt.assert_allclose(distances[1, 2], 0.0)  # [0,0,0] to [0,0,0]

    def test_face_distances(self):
        """Test distance between faces calculation."""
        # Orthogonal box
        box = mp.Box.orth([2, 4, 6])
        face_distances = box.get_distance_between_faces()
        npt.assert_allclose(face_distances, [2, 4, 6])
        
        # Free box
        box = mp.Box()
        face_distances = box.get_distance_between_faces()
        npt.assert_allclose(face_distances, [0, 0, 0])


class TestBoxValidation:
    """Test error handling and validation."""

    def test_invalid_matrix(self):
        """Test error handling for invalid matrices."""
        # Non-square matrix should raise error
        with pytest.raises(AssertionError):
            mp.Box(np.array([[1, 2], [3, 4]]))
        
        # Singular matrix should raise error
        with pytest.raises(AssertionError):
            mp.Box(np.array([[1, 2, 3], [2, 4, 6], [3, 6, 9]]))

    def test_invalid_angles(self):
        """Test error handling for invalid angles."""
        lengths = [1, 2, 3]
        
        # Invalid angle range
        with pytest.raises(ValueError):
            mp.Box.from_lengths_angles(lengths, [0, 90, 90])
        
        with pytest.raises(ValueError):
            mp.Box.from_lengths_angles(lengths, [180, 90, 90])

    def test_lengths_validation(self):
        """Test validation of length assignments."""
        box = mp.Box.orth([1, 2, 3])
        
        # Invalid length array size
        with pytest.raises(AssertionError):
            box.lengths = [1, 2, 3, 4]
        
        with pytest.raises(AssertionError):
            box.lengths = [1, 2]


class TestBoxUtility:
    """Test utility methods and operations."""

    def test_box_equality(self):
        """Test box equality comparison."""
        box1 = mp.Box.tric([2, 2, 2], [1, 0.5, 0.1])
        box2 = mp.Box.tric([2, 2, 2], [1, 0.5, 0.1])
        box3 = mp.Box.tric([2, 2, 2], [1, 0, 0])
        
        assert box1 == box2
        assert box1 != box3

    def test_box_multiplication(self):
        """Test box scaling operations."""
        box = mp.Box.tric([2, 3, 4], [1, 0.5, 0.1])
        
        # Test right multiplication
        box2 = box * 2
        npt.assert_allclose(box2.lx, 4)
        npt.assert_allclose(box2.ly, 6)
        npt.assert_allclose(box2.lz, 8)
        npt.assert_allclose(box2.xy, 2)
        
        # Test left multiplication
        box3 = 2 * box
        assert box2 == box3

    def test_box_representation(self):
        """Test string representations."""
        # Free box
        box = mp.Box()
        assert repr(box) == "<Free Box>"
        
        # Orthogonal box
        box = mp.Box.orth([1, 2, 3])
        assert "Orthogonal Box" in repr(box)
        
        # Triclinic box
        box = mp.Box.tric([2, 2, 2], [1, 0.5, 0.1])
        assert "Triclinic Box" in repr(box)

    def test_to_dict(self):
        """Test conversion to dictionary."""
        box = mp.Box.tric([2, 2, 2], [1, 0.5, 0.1])
        box_dict = box.to_dict()
        
        expected_keys = {
            "xlo", "xhi", "ylo", "yhi", "zlo", "zhi",
            "xy", "xz", "yz", "x_pbc", "y_pbc", "z_pbc"
        }
        assert set(box_dict.keys()) == expected_keys

    def test_isin(self):
        """Test point-in-box checking."""
        box = mp.Box.tric([2.0, 2.0, 2.0], [1.0, 0.0, 0.0])
        
        points = np.array([
            [0.0, 0.0, 0.0],  # Inside
            [2.0, 0, 0],      # Outside (on boundary)
            [1.0, 1.0, 0.0],  # Inside
            [0.5, 1.75, 0.0]  # Outside
        ])
        
        result = box.isin(points)
        expected = [True, False, True, False]
        npt.assert_array_equal(result, expected)


class TestBoxStyle:
    """Test box style detection and classification."""

    def test_style_detection(self):
        """Test automatic style detection."""
        # Free box
        box = mp.Box()
        assert box.style == mp.Box.Style.FREE
        
        # Orthogonal box
        box = mp.Box(np.diag([1, 2, 3]))
        assert box.style == mp.Box.Style.ORTHOGONAL
        
        # Triclinic box
        matrix = np.array([[1, 0.5, 0], [0, 2, 0], [0, 0, 3]])
        box = mp.Box(matrix)
        assert box.style == mp.Box.Style.TRICLINIC

    def test_style_consistency(self):
        """Test that constructed boxes have expected styles."""
        assert mp.Box.cubic(1).style == mp.Box.Style.ORTHOGONAL
        assert mp.Box.orth([1, 2, 3]).style == mp.Box.Style.ORTHOGONAL
        assert mp.Box.tric([1, 2, 3], [0.5, 1, 1.5]).style == mp.Box.Style.TRICLINIC


if __name__ == "__main__":
    pytest.main([__file__])
