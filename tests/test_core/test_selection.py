import numpy as np

from molpy import Block
from molpy.core.selector import AtomIndexSelector, AtomTypeSelector


class TestMaskPredicate:
    """Test the abstract MaskPredicate class and boolean operators."""

    def test_mask_predicate_composition(self):
        """Test that MaskPredicate can be combined with & | ~ operators."""
        # Create concrete instances for testing
        type1 = AtomTypeSelector(1)
        type2 = AtomTypeSelector(2)

        # Test AND operator
        and_pred = type1 & type2
        assert hasattr(and_pred, "mask")
        assert hasattr(and_pred, "__call__")

        # Test OR operator
        or_pred = type1 | type2
        assert hasattr(or_pred, "mask")
        assert hasattr(or_pred, "__call__")

        # Test NOT operator
        not_pred = ~type1
        assert hasattr(not_pred, "mask")
        assert hasattr(not_pred, "__call__")

    def test_mask_predicate_call(self):
        """Test that MaskPredicate.__call__ returns filtered Block."""
        block = Block(
            {
                "type": np.array([1, 2, 1, 3]),
                "id": np.array([0, 1, 2, 3]),
                "xyz": np.random.random((4, 3)),
            }
        )

        type1 = AtomTypeSelector(1)
        filtered_block = type1(block)

        assert isinstance(filtered_block, Block)
        assert len(filtered_block["type"]) == 2  # Only type 1 atoms
        assert np.all(filtered_block["type"] == 1)


class TestAtomTypeSelector:
    """Test AtomTypeSelector predicate."""

    def test_atom_type_init(self):
        """Test AtomTypeSelector initialization."""
        # Test with integer type
        pred1 = AtomTypeSelector(1)
        assert pred1.atom_type == 1
        assert pred1.field == "type"

        # Test with string type
        pred2 = AtomTypeSelector("C")
        assert pred2.atom_type == "C"
        assert pred2.field == "type"

        # Test with custom field
        pred3 = AtomTypeSelector(42, field="element")
        assert pred3.atom_type == 42
        assert pred3.field == "element"

    def test_atom_type_mask(self):
        """Test AtomTypeSelector.mask method."""
        block = Block({"type": np.array([1, 2, 1, 3, 1]), "id": np.arange(5)})

        pred = AtomTypeSelector(1)
        mask = pred.mask(block)

        expected = np.array([True, False, True, False, True])
        assert np.array_equal(mask, expected)

    def test_atom_type_with_string(self):
        """Test AtomTypeSelector with string values."""
        block = Block({"type": np.array(["C", "O", "C", "N"]), "id": np.arange(4)})

        pred = AtomTypeSelector("C")
        mask = pred.mask(block)

        expected = np.array([True, False, True, False])
        assert np.array_equal(mask, expected)

    def test_atom_type_custom_field(self):
        """Test AtomTypeSelector with custom field name."""
        block = Block(
            {
                "element": np.array([6, 8, 6, 7]),  # C, O, C, N atomic numbers
                "id": np.arange(4),
            }
        )

        pred = AtomTypeSelector(6, field="element")
        mask = pred.mask(block)

        expected = np.array([True, False, True, False])
        assert np.array_equal(mask, expected)


class TestAtomIndexSelector:
    """Test AtomIndexSelector predicate."""

    def test_atom_index_init(self):
        """Test AtomIndexSelector initialization."""
        # Test with list of indices
        pred1 = AtomIndexSelector([0, 2, 4])
        assert np.array_equal(pred1.indices, np.array([0, 2, 4]))
        assert pred1.id_field == "id"

        # Test with custom field
        pred2 = AtomIndexSelector([10, 20], id_field="atom_id")
        assert np.array_equal(pred2.indices, np.array([10, 20]))
        assert pred2.id_field == "atom_id"

    def test_atom_index_mask(self):
        """Test AtomIndexSelector.mask method."""
        block = Block({"id": np.array([10, 20, 30, 40, 50]), "type": np.ones(5)})

        pred = AtomIndexSelector([20, 40])
        mask = pred.mask(block)

        expected = np.array([False, True, False, True, False])
        assert np.array_equal(mask, expected)

    def test_atom_index_custom_field(self):
        """Test AtomIndexSelector with custom id field."""
        block = Block({"atom_id": np.array([100, 200, 300]), "type": np.ones(3)})

        pred = AtomIndexSelector([200], id_field="atom_id")
        mask = pred.mask(block)

        expected = np.array([False, True, False])
        assert np.array_equal(mask, expected)


class TestBooleanComposition:
    """Test boolean composition of predicates."""

    def setup_method(self):
        """Set up test data."""
        self.block = Block(
            {
                "type": np.array([1, 2, 1, 3, 2]),
                "id": np.array([10, 20, 30, 40, 50]),
                "xyz": np.random.random((5, 3)),
            }
        )

    def test_and_composition(self):
        """Test AND composition of predicates."""
        type1 = AtomTypeSelector(1)
        indices = AtomIndexSelector([10, 30])  # First and third atoms

        # Both type=1 AND id in [10, 30]
        combined = type1 & indices
        mask = combined.mask(self.block)

        # Only atoms 0 and 2 have type=1, and only 0,2 have id in [10,30]
        expected = np.array([True, False, True, False, False])
        assert np.array_equal(mask, expected)

    def test_or_composition(self):
        """Test OR composition of predicates."""
        type1 = AtomTypeSelector(1)
        type2 = AtomTypeSelector(2)

        # Either type=1 OR type=2
        combined = type1 | type2
        mask = combined.mask(self.block)

        # Atoms with type 1 or 2 (positions 0,1,2,4)
        expected = np.array([True, True, True, False, True])
        assert np.array_equal(mask, expected)

    def test_not_composition(self):
        """Test NOT composition of predicates."""
        type1 = AtomTypeSelector(1)

        # NOT type=1
        negated = ~type1
        mask = negated.mask(self.block)

        # All atoms except type=1 (positions 1,3,4)
        expected = np.array([False, True, False, True, True])
        assert np.array_equal(mask, expected)

    def test_complex_composition(self):
        """Test complex boolean expressions."""
        type1 = AtomTypeSelector(1)
        type2 = AtomTypeSelector(2)
        indices = AtomIndexSelector([20, 40])  # Second and fourth atoms

        # (type=1 OR type=2) AND id in [20, 40]
        combined = (type1 | type2) & indices
        mask = combined.mask(self.block)

        # type1|type2: [True, True, True, False, True]
        # indices: [False, True, False, True, False]
        # AND: [False, True, False, False, False]
        expected = np.array([False, True, False, False, False])
        assert np.array_equal(mask, expected)

    def test_filter_block_with_composition(self):
        """Test filtering Block with composed predicates."""
        type1 = AtomTypeSelector(1)
        filtered_block = type1(self.block)

        # Should get only atoms with type=1
        expected_types = np.array([1, 1])
        expected_ids = np.array([10, 30])

        assert np.array_equal(filtered_block["type"], expected_types)
        assert np.array_equal(filtered_block["id"], expected_ids)
        assert filtered_block["xyz"].shape == (2, 3)
