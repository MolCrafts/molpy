import numpy as np

from molpy import Block
from molpy.core.region import (
    AndRegion,
    BoxRegion,
    Cube,
    NotRegion,
    OrRegion,
    SphereRegion,
)


class TestRegion:
    """Test the abstract Region class."""

    def test_region_is_mask_predicate(self):
        """Test that Region inherits from MaskPredicate."""
        box = BoxRegion(np.array([2.0, 2.0, 2.0]))

        # Should have MaskPredicate methods
        assert hasattr(box, "mask")
        assert hasattr(box, "__call__")
        assert hasattr(box, "__and__")
        assert hasattr(box, "__or__")
        assert hasattr(box, "__invert__")

    def test_region_mask_integration(self):
        """Test that Region.mask works with Block."""
        # Create a block with coordinates
        block = Block(
            {
                "xyz": np.array(
                    [
                        [0.5, 0.5, 0.5],  # Inside
                        [3.0, 3.0, 3.0],  # Outside
                        [1.0, 1.0, 1.0],  # Inside
                    ]
                ),
                "type": np.array([1, 2, 3]),
            }
        )

        box = BoxRegion(np.array([2.0, 2.0, 2.0]))
        mask = box.mask(block)

        expected = np.array([True, False, True])
        assert np.array_equal(mask, expected)

    def test_region_call_filters_block(self):
        """Test that Region.__call__ filters the Block."""
        block = Block(
            {
                "xyz": np.array(
                    [
                        [0.5, 0.5, 0.5],  # Inside
                        [3.0, 3.0, 3.0],  # Outside
                        [1.0, 1.0, 1.0],  # Inside
                    ]
                ),
                "type": np.array([1, 2, 3]),
            }
        )

        box = BoxRegion(np.array([2.0, 2.0, 2.0]))
        filtered = box(block)

        assert isinstance(filtered, Block)
        assert len(filtered["type"]) == 2
        assert np.array_equal(filtered["type"], np.array([1, 3]))


class TestBoxRegion:
    """Test BoxRegion implementation."""

    def test_box_region_init(self):
        """Test BoxRegion initialization."""
        lengths = np.array([2.0, 3.0, 4.0])
        origin = np.array([1.0, 1.0, 1.0])

        box = BoxRegion(lengths, origin, coord_field="xyz")

        assert np.array_equal(box.lengths, lengths)
        assert np.array_equal(box.origin, origin)
        assert box.coord_field == "xyz"

    def test_box_region_default_origin(self):
        """Test BoxRegion with default origin."""
        lengths = np.array([2.0, 3.0, 4.0])
        box = BoxRegion(lengths)

        assert np.array_equal(box.origin, np.zeros(3))

    def test_box_region_isin(self):
        """Test BoxRegion.isin method."""
        box = BoxRegion(np.array([2.0, 2.0, 2.0]), np.array([1.0, 1.0, 1.0]))

        points = np.array(
            [
                [1.5, 1.5, 1.5],  # Inside
                [0.5, 1.5, 1.5],  # Outside (x too small)
                [3.5, 1.5, 1.5],  # Outside (x too large)
                [2.0, 2.0, 2.0],  # On boundary (should be inside)
                [1.0, 1.0, 1.0],  # On boundary (origin, should be inside)
            ]
        )

        result = box.isin(points)
        expected = np.array([True, False, False, True, True])
        assert np.array_equal(result, expected)

    def test_box_region_bounds(self):
        """Test BoxRegion.bounds property."""
        lengths = np.array([2.0, 3.0, 4.0])
        origin = np.array([1.0, 2.0, 3.0])
        box = BoxRegion(lengths, origin)

        bounds = box.bounds
        expected_lower = np.array([1.0, 2.0, 3.0])
        expected_upper = np.array([3.0, 5.0, 7.0])

        assert np.array_equal(bounds[0], expected_lower)
        assert np.array_equal(bounds[1], expected_upper)


class TestSphereRegion:
    """Test SphereRegion implementation."""

    def test_sphere_region_init(self):
        """Test SphereRegion initialization."""
        radius = 2.5
        center = np.array([1.0, 2.0, 3.0])

        sphere = SphereRegion(radius, center, coord_field="coords")

        assert sphere.radius == radius
        assert np.array_equal(sphere.center, center)
        assert sphere.coord_field == "coords"

    def test_sphere_region_default_center(self):
        """Test SphereRegion with default center."""
        sphere = SphereRegion(1.0)
        assert np.array_equal(sphere.center, np.zeros(3))

    def test_sphere_region_isin(self):
        """Test SphereRegion.isin method."""
        sphere = SphereRegion(2.0, np.array([0.0, 0.0, 0.0]))

        points = np.array(
            [
                [0.0, 0.0, 0.0],  # Center (inside)
                [1.0, 0.0, 0.0],  # Inside
                [0.0, 2.0, 0.0],  # On boundary (inside)
                [1.0, 1.0, 1.0],  # Inside (distance = sqrt(3) ≈ 1.73 < 2)
                [2.0, 0.0, 0.0],  # On boundary (inside)
                [3.0, 0.0, 0.0],  # Outside
            ]
        )

        result = sphere.isin(points)
        distances_sq = np.sum(points**2, axis=1)
        expected = distances_sq <= 4.0  # radius^2 = 4

        assert np.array_equal(result, expected)

    def test_sphere_region_bounds(self):
        """Test SphereRegion.bounds property."""
        radius = 2.5
        center = np.array([1.0, 2.0, 3.0])
        sphere = SphereRegion(radius, center)

        bounds = sphere.bounds
        expected_lower = center - radius
        expected_upper = center + radius

        assert np.array_equal(bounds[0], expected_lower)
        assert np.array_equal(bounds[1], expected_upper)


class TestCube:
    """Test Cube implementation."""

    def test_cube_init(self):
        """Test Cube initialization."""
        edge = 3.0
        origin = np.array([1.0, 1.0, 1.0])

        cube = Cube(edge, origin, coord_field="positions")

        assert cube.edge == edge
        assert np.array_equal(cube.lengths, np.array([edge, edge, edge]))
        assert np.array_equal(cube.origin, origin)
        assert cube.coord_field == "positions"

    def test_cube_default_origin(self):
        """Test Cube with default origin."""
        cube = Cube(2.0)
        assert np.array_equal(cube.origin, np.zeros(3))

    def test_cube_inherits_box_behavior(self):
        """Test that Cube behaves like BoxRegion."""
        cube = Cube(2.0)

        points = np.array(
            [
                [1.0, 1.0, 1.0],  # Inside
                [2.5, 1.0, 1.0],  # Outside
            ]
        )

        result = cube.isin(points)
        expected = np.array([True, False])
        assert np.array_equal(result, expected)


class TestRegionComposition:
    """Test boolean composition of regions."""

    def setup_method(self):
        """Set up test data."""
        self.box = BoxRegion(np.array([2.0, 2.0, 2.0]))
        self.sphere = SphereRegion(1.5)

        self.points = np.array(
            [
                [0.5, 0.5, 0.5],  # Inside both
                [1.8, 0.1, 0.1],  # Inside box, outside sphere
                [0.1, 0.1, 1.8],  # Inside box, outside sphere
                [3.0, 3.0, 3.0],  # Outside both
            ]
        )

    def test_and_region(self):
        """Test AndRegion (intersection)."""
        and_region = self.box & self.sphere

        assert isinstance(and_region, AndRegion)
        assert and_region.a == self.box
        assert and_region.b == self.sphere

        result = and_region.isin(self.points)

        # Only points inside both regions
        box_mask = self.box.isin(self.points)
        sphere_mask = self.sphere.isin(self.points)
        expected = box_mask & sphere_mask

        assert np.array_equal(result, expected)

    def test_or_region(self):
        """Test OrRegion (union)."""
        or_region = self.box | self.sphere

        assert isinstance(or_region, OrRegion)
        assert or_region.a == self.box
        assert or_region.b == self.sphere

        result = or_region.isin(self.points)

        # Points inside either region
        box_mask = self.box.isin(self.points)
        sphere_mask = self.sphere.isin(self.points)
        expected = box_mask | sphere_mask

        assert np.array_equal(result, expected)

    def test_not_region(self):
        """Test NotRegion (complement)."""
        not_region = ~self.box

        assert isinstance(not_region, NotRegion)
        assert not_region.a == self.box

        result = not_region.isin(self.points)

        # Points outside the box
        box_mask = self.box.isin(self.points)
        expected = ~box_mask

        assert np.array_equal(result, expected)

    def test_complex_composition(self):
        """Test complex boolean composition."""
        cube = Cube(1.0)

        # (box | sphere) & (~cube)
        complex_region = (self.box | self.sphere) & (~cube)
        result = complex_region.isin(self.points)

        box_mask = self.box.isin(self.points)
        sphere_mask = self.sphere.isin(self.points)
        cube_mask = cube.isin(self.points)

        expected = (box_mask | sphere_mask) & (~cube_mask)
        assert np.array_equal(result, expected)

    def test_and_region_bounds(self):
        """Test AndRegion bounds calculation."""
        box = BoxRegion(np.array([4.0, 4.0, 4.0]), np.array([1.0, 1.0, 1.0]))
        sphere = SphereRegion(2.0, np.array([2.0, 2.0, 2.0]))

        and_region = box & sphere
        bounds = and_region.bounds

        # Intersection bounds should be overlap
        box_bounds = box.bounds
        sphere_bounds = sphere.bounds

        expected_lower = np.maximum(box_bounds[0], sphere_bounds[0])
        expected_upper = np.minimum(box_bounds[1], sphere_bounds[1])

        assert np.array_equal(bounds[0], expected_lower)
        assert np.array_equal(bounds[1], expected_upper)

    def test_or_region_bounds(self):
        """Test OrRegion bounds calculation."""
        box = BoxRegion(np.array([2.0, 2.0, 2.0]), np.array([1.0, 1.0, 1.0]))
        sphere = SphereRegion(1.0, np.array([4.0, 4.0, 4.0]))

        or_region = box | sphere
        bounds = or_region.bounds

        # Union bounds should encompass both
        box_bounds = box.bounds
        sphere_bounds = sphere.bounds

        expected_lower = np.minimum(box_bounds[0], sphere_bounds[0])
        expected_upper = np.maximum(box_bounds[1], sphere_bounds[1])

        assert np.array_equal(bounds[0], expected_lower)
        assert np.array_equal(bounds[1], expected_upper)


class TestRegionWithBlock:
    """Test Region integration with Block objects."""

    def test_custom_coord_field(self):
        """Test Region with custom coordinate field."""
        block = Block(
            {
                "positions": np.array(
                    [
                        [0.5, 0.5, 0.5],
                        [3.0, 3.0, 3.0],
                    ]
                ),
                "type": np.array([1, 2]),
            }
        )

        box = BoxRegion(np.array([2.0, 2.0, 2.0]), coord_field="positions")
        filtered = box(block)

        assert len(filtered["type"]) == 1
        assert filtered["type"][0] == 1

    def test_region_as_selection(self):
        """Test that Region works as a MaskPredicate/Selection."""
        from molpy.core.selector import AtomTypeSelector

        block = Block(
            {
                "xyz": np.array(
                    [
                        [0.5, 0.5, 0.5],  # Inside
                        [3.0, 3.0, 3.0],  # Outside
                        [1.0, 1.0, 1.0],  # Inside
                    ]
                ),
                "type": np.array([1, 1, 2]),
            }
        )

        box = BoxRegion(np.array([2.0, 2.0, 2.0]))
        type1 = AtomTypeSelector(1)

        # Test region and selection separately first
        box_filtered = box(block)
        type_filtered = type1(block)

        # Region should filter by spatial location
        assert len(box_filtered["type"]) == 2  # Inside atoms

        # Selection should filter by type
        assert len(type_filtered["type"]) == 2  # Type 1 atoms
        assert np.all(type_filtered["type"] == 1)

        # Note: Direct composition of Region & AtomTypeSelector requires
        # both to implement the same interface consistently
