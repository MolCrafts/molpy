import numpy as np
import numpy.testing as npt
import pytest

from molpy.core.region import (
    AndRegion,
    BoxRegion,
    Cube,
    NotRegion,
    OrRegion,
    Region,
    SphereRegion,
)


class TestRegion:
    """Test the abstract Region class and its operators."""

    def test_region_operators(self):
        """Test logical operators for regions."""
        cube1 = Cube(2.0, origin=[0, 0, 0])
        cube2 = Cube(2.0, origin=[1, 1, 1])
        
        # Test AND operator
        and_region = cube1 & cube2
        assert isinstance(and_region, AndRegion)
        assert and_region.r1 == cube1
        assert and_region.r2 == cube2
        
        # Test OR operator
        or_region = cube1 | cube2
        assert isinstance(or_region, OrRegion)
        assert or_region.r1 == cube1
        assert or_region.r2 == cube2
        
        # Test NOT operator
        not_region = ~cube1
        assert isinstance(not_region, NotRegion)
        assert not_region.region == cube1


class TestCube:
    """Test the Cube region class."""

    def test_init(self):
        """Test cube initialization."""
        cube = Cube(2.0)
        assert cube.length == 2.0
        npt.assert_array_equal(cube.origin, [0, 0, 0])
        assert cube.name == "Cube"
        
        # Test with custom origin and name
        cube2 = Cube(3.0, origin=[1, 2, 3], name="TestCube")
        assert cube2.length == 3.0
        npt.assert_array_equal(cube2.origin, [1, 2, 3])
        assert cube2.name == "TestCube"

    def test_isin(self):
        """Test if points are inside the cube."""
        cube = Cube(2.0, origin=[0, 0, 0])
        
        # Points inside
        points_inside = np.array([[0.1, 0.1, 0.1], [1.0, 1.0, 1.0], [1.9, 1.9, 1.9]])
        result = cube.isin(points_inside)
        npt.assert_array_equal(result, [True, True, True])
        
        # Points outside
        points_outside = np.array([[-0.1, 0, 0], [0, -0.1, 0], [2.1, 1.0, 1.0]])
        result = cube.isin(points_outside)
        npt.assert_array_equal(result, [False, False, False])
        
        # Edge points (should be inside due to <= comparison)
        edge_points = np.array([[0, 0, 0], [2, 2, 2]])
        result = cube.isin(edge_points)
        npt.assert_array_equal(result, [True, True])

    def test_volume(self):
        """Test cube volume calculation."""
        cube = Cube(2.0)
        assert cube.volume() == 8.0
        
        cube = Cube(3.5)
        assert cube.volume() == 3.5**3

    def test_bounds(self):
        """Test cube bounds property."""
        cube = Cube(2.0, origin=[1, 2, 3])
        bounds = cube.bounds
        expected = np.array([[1, 3], [2, 4], [3, 5]]).T
        npt.assert_array_equal(bounds, expected)

    def test_properties(self):
        """Test cube coordinate properties."""
        cube = Cube(2.0, origin=[1, 2, 3])
        assert cube.xlo == 1
        assert cube.xhi == 3
        assert cube.ylo == 2
        assert cube.yhi == 4
        assert cube.zlo == 3
        assert cube.zhi == 5


class TestSphereRegion:
    """Test the SphereRegion class."""

    def test_init(self):
        """Test sphere initialization."""
        sphere = SphereRegion(1.0, [0, 0, 0])
        assert sphere.radius == 1.0
        npt.assert_array_equal(sphere.origin, [0, 0, 0])
        assert sphere.name == "Sphere"
        
        # Test with custom name
        sphere2 = SphereRegion(2.0, [1, 2, 3], name="TestSphere")
        assert sphere2.radius == 2.0
        npt.assert_array_equal(sphere2.origin, [1, 2, 3])
        assert sphere2.name == "TestSphere"

    def test_init_validation(self):
        """Test sphere initialization validation."""
        # Test invalid radius
        with pytest.raises(AssertionError):
            SphereRegion("invalid", [0, 0, 0])
        
        # Test invalid origin
        with pytest.raises(AssertionError):
            SphereRegion(1.0, [0, 0])  # Only 2D

    def test_isin(self):
        """Test if points are inside the sphere."""
        sphere = SphereRegion(1.0, [0, 0, 0])
        
        # Points inside
        points_inside = np.array([[0, 0, 0], [0.5, 0.5, 0], [0.1, 0.1, 0.1]])
        result = sphere.isin(points_inside)
        expected = np.array([True, True, True])
        npt.assert_array_equal(result, expected)
        
        # Points outside
        points_outside = np.array([[1.1, 0, 0], [0, 1.1, 0], [0.8, 0.8, 0.8]])
        result = sphere.isin(points_outside)
        expected = np.array([False, False, False])
        npt.assert_array_equal(result, expected)
        
        # Edge point (should be inside due to <= comparison)
        edge_point = np.array([[1, 0, 0]])
        result = sphere.isin(edge_point)
        npt.assert_array_equal(result, [True])

    def test_volume(self):
        """Test sphere volume calculation."""
        sphere = SphereRegion(1.0, [0, 0, 0])
        expected_volume = 4/3 * np.pi * 1.0**3
        assert np.isclose(sphere.volume(), expected_volume)
        
        sphere = SphereRegion(2.0, [0, 0, 0])
        expected_volume = 4/3 * np.pi * 2.0**3
        assert np.isclose(sphere.volume(), expected_volume)

    def test_bounds(self):
        """Test sphere bounds property."""
        sphere = SphereRegion(1.0, [1, 2, 3])
        bounds = sphere.bounds
        expected = np.array([[0, 1, 2], [2, 3, 4]])
        npt.assert_array_equal(bounds, expected)


class TestBoxRegion:
    """Test the BoxRegion class."""

    def test_init(self):
        """Test box initialization."""
        box = BoxRegion([2, 3, 4], [0, 0, 0])
        npt.assert_array_equal(box.lengths, [2, 3, 4])
        npt.assert_array_equal(box.origin, [0, 0, 0])
        npt.assert_array_equal(box.upper, [2, 3, 4])
        assert box.name == "Box"
        
        # Test with custom name
        box2 = BoxRegion([1, 2, 3], [1, 1, 1], name="TestBox")
        assert box2.name == "TestBox"

    def test_isin(self):
        """Test if points are inside the box."""
        box = BoxRegion([2, 3, 4], [0, 0, 0])
        
        # Points inside
        points_inside = np.array([[0.1, 0.1, 0.1], [1.0, 1.5, 2.0], [1.9, 2.9, 3.9]])
        result = box.isin(points_inside)
        npt.assert_array_equal(result, [True, True, True])
        
        # Points outside
        points_outside = np.array([[-0.1, 0, 0], [0, 3.1, 0], [1.0, 1.0, 4.1]])
        result = box.isin(points_outside)
        npt.assert_array_equal(result, [False, False, False])

    def test_volume(self):
        """Test box volume calculation."""
        box = BoxRegion([2, 3, 4], [0, 0, 0])
        assert box.volume() == 24.0

    def test_bounds(self):
        """Test box bounds property."""
        box = BoxRegion([2, 3, 4], [1, 2, 3])
        bounds = box.bounds
        expected = np.array([[1, 2, 3], [3, 5, 7]])
        npt.assert_array_equal(bounds, expected)


class TestAndRegion:
    """Test the AndRegion (intersection) class."""

    def test_init(self):
        """Test AND region initialization."""
        cube1 = Cube(2.0, origin=[0, 0, 0], name="Cube1")
        cube2 = Cube(2.0, origin=[1, 1, 1], name="Cube2")
        and_region = AndRegion(cube1, cube2)
        
        assert and_region.r1 is cube1
        assert and_region.r2 is cube2
        assert and_region.name == "(Cube1 & Cube2)"

    def test_isin(self):
        """Test intersection region contains logic."""
        cube1 = Cube(2.0, origin=[0, 0, 0])
        cube2 = Cube(2.0, origin=[1, 1, 1])
        and_region = AndRegion(cube1, cube2)
        
        # Point in intersection
        point_in_both = np.array([[1.5, 1.5, 1.5]])
        result = and_region.isin(point_in_both)
        npt.assert_array_equal(result, [True])
        
        # Point in only one cube
        point_in_one = np.array([[0.5, 0.5, 0.5]])
        result = and_region.isin(point_in_one)
        npt.assert_array_equal(result, [False])
        
        # Point in neither cube
        point_in_neither = np.array([[5, 5, 5]])
        result = and_region.isin(point_in_neither)
        npt.assert_array_equal(result, [False])

    def test_bounds(self):
        """Test intersection region bounds."""
        cube1 = Cube(2.0, origin=[0, 0, 0])
        cube2 = Cube(2.0, origin=[1, 1, 1])
        and_region = AndRegion(cube1, cube2)
        
        bounds = and_region.bounds
        # Intersection should be from [1,1,1] to [2,2,2]
        expected = np.array([[1, 1, 1], [2, 2, 2]])
        npt.assert_array_equal(bounds, expected)


class TestOrRegion:
    """Test the OrRegion (union) class."""

    def test_init(self):
        """Test OR region initialization."""
        cube1 = Cube(1.0, origin=[0, 0, 0], name="Cube1")
        cube2 = Cube(1.0, origin=[2, 2, 2], name="Cube2")
        or_region = OrRegion(cube1, cube2)
        
        assert or_region.r1 is cube1
        assert or_region.r2 is cube2
        assert or_region.name == "(Cube1 | Cube2)"

    def test_isin(self):
        """Test union region contains logic."""
        cube1 = Cube(1.0, origin=[0, 0, 0])
        cube2 = Cube(1.0, origin=[2, 2, 2])
        or_region = OrRegion(cube1, cube2)
        
        # Point in first cube
        point_in_first = np.array([[0.5, 0.5, 0.5]])
        result = or_region.isin(point_in_first)
        npt.assert_array_equal(result, [True])
        
        # Point in second cube
        point_in_second = np.array([[2.5, 2.5, 2.5]])
        result = or_region.isin(point_in_second)
        npt.assert_array_equal(result, [True])
        
        # Point in neither cube
        point_in_neither = np.array([[1.5, 1.5, 1.5]])
        result = or_region.isin(point_in_neither)
        npt.assert_array_equal(result, [False])

    def test_bounds(self):
        """Test union region bounds."""
        cube1 = Cube(1.0, origin=[0, 0, 0])
        cube2 = Cube(1.0, origin=[2, 2, 2])
        or_region = OrRegion(cube1, cube2)
        
        bounds = or_region.bounds
        # Union should be from [0,0,0] to [3,3,3]
        expected = np.array([[0, 0, 0], [3, 3, 3]])
        npt.assert_array_equal(bounds, expected)


class TestNotRegion:
    """Test the NotRegion (complement) class."""

    def test_init(self):
        """Test NOT region initialization."""
        cube = Cube(1.0, origin=[0, 0, 0], name="TestCube")
        not_region = NotRegion(cube)
        
        assert not_region.region is cube
        assert not_region.name == "(!TestCube)"

    def test_isin(self):
        """Test complement region contains logic."""
        cube = Cube(1.0, origin=[0, 0, 0])
        not_region = NotRegion(cube)
        
        # Point inside original cube (should be outside complement)
        point_inside = np.array([[0.5, 0.5, 0.5]])
        result = not_region.isin(point_inside)
        npt.assert_array_equal(result, [False])
        
        # Point outside original cube (should be inside complement)
        point_outside = np.array([[2, 2, 2]])
        result = not_region.isin(point_outside)
        npt.assert_array_equal(result, [True])

    def test_bounds(self):
        """Test complement region bounds (should be infinite)."""
        cube = Cube(1.0, origin=[0, 0, 0])
        not_region = NotRegion(cube)
        
        bounds = not_region.bounds
        expected = np.array([[-np.inf, -np.inf, -np.inf], [np.inf, np.inf, np.inf]])
        npt.assert_array_equal(bounds, expected)


class TestComplexRegions:
    """Test complex combinations of regions."""

    def test_nested_operations(self):
        """Test nested logical operations."""
        cube1 = Cube(2.0, origin=[0, 0, 0])
        cube2 = Cube(2.0, origin=[1, 1, 1])
        sphere = SphereRegion(1.5, [1, 1, 1])
        
        # Test (cube1 & cube2) | sphere
        intersection = cube1 & cube2
        complex_region = intersection | sphere
        
        # Point in intersection should be in complex region
        point_in_intersection = np.array([[1.5, 1.5, 1.5]])
        result = complex_region.isin(point_in_intersection)
        npt.assert_array_equal(result, [True])
        
        # Point only in sphere should be in complex region
        point_in_sphere_only = np.array([[1, 1, 0.5]])
        result = complex_region.isin(point_in_sphere_only)
        npt.assert_array_equal(result, [True])

    def test_not_of_intersection(self):
        """Test NOT of intersection."""
        cube1 = Cube(1.0, origin=[0, 0, 0])
        cube2 = Cube(1.0, origin=[0.5, 0.5, 0.5])
        intersection = cube1 & cube2
        complement = ~intersection
        
        # Point in intersection should not be in complement
        point_in_intersection = np.array([[0.75, 0.75, 0.75]])
        result = complement.isin(point_in_intersection)
        npt.assert_array_equal(result, [False])
        
        # Point outside intersection should be in complement
        point_outside_intersection = np.array([[2, 2, 2]])
        result = complement.isin(point_outside_intersection)
        npt.assert_array_equal(result, [True])
