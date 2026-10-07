import numpy as np
import pytest

from molpy.core import AtomIndexSelector, AtomTypeSelector, Block, ElementSelector


class TestMaskPredicate:
    """Test the abstract MaskPredicate class and boolean operators."""

    def test_mask_predicate_composition(self):
        """Test that MaskPredicate can be combined with & | ~ operators."""
        # Create concrete instances for testing
        type1 = AtomTypeSelector(1, field="type_id")
        type2 = AtomTypeSelector(2, field="type_id")

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
                "type_id": np.array([1, 2, 1, 3]),
                "id": np.array([0, 1, 2, 3]),
                "vec": np.random.random((4, 3)),
            }
        )

        type1 = AtomTypeSelector(1, field="type_id")
        filtered_block = type1(block)

        assert isinstance(filtered_block, Block)
        assert len(filtered_block["type_id"]) == 2  # Only type 1 atoms
        assert np.all(filtered_block["type_id"] == 1)


class TestAtomTypeSelector:
    """Test AtomTypeSelector predicate."""

    def test_atom_type_init(self):
        """Test AtomTypeSelector initialization."""
        # Test with integer type
        pred1 = AtomTypeSelector(1, field="type_id")
        assert pred1.atom_type == 1
        assert pred1.field == "type_id"

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
        block = Block({"type_id": np.array([1, 2, 1, 3, 1]), "id": np.arange(5)})

        pred = AtomTypeSelector(1, field="type_id")
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
                # Atomic numbers belong in `atomic_number`; `element` is the
                # IUPAC symbol. The vocabulary keeps them apart.
                "atomic_number": np.array([6, 8, 6, 7], dtype=np.uint32),
                "id": np.arange(4),
            }
        )

        pred = AtomTypeSelector(6, field="atomic_number")
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
        block = Block(
            {
                "id": np.array([10, 20, 30, 40, 50]),
                "type_id": np.ones(5, dtype=np.uint32),
            }
        )

        pred = AtomIndexSelector([20, 40])
        mask = pred.mask(block)

        expected = np.array([False, True, False, True, False])
        assert np.array_equal(mask, expected)

    def test_atom_index_custom_field(self):
        """Test AtomIndexSelector with custom id field."""
        block = Block(
            {
                "atom_id": np.array([100, 200, 300]),
                "type_id": np.ones(3, dtype=np.uint32),
            }
        )

        pred = AtomIndexSelector([200], id_field="atom_id")
        mask = pred.mask(block)

        expected = np.array([False, True, False])
        assert np.array_equal(mask, expected)


class TestSelectorFailFast:
    """Selectors raise on a missing column instead of silently selecting nothing:
    a typo'd or absent field is a mistake to surface, not a selection of zero
    atoms.
    """

    def test_element_selector_missing_field_raises(self):
        block = Block({"x": np.array([0.0, 1.0])})  # no "element" column
        with pytest.raises(KeyError, match="ElementSelector"):
            ElementSelector("C").mask(block)

    def test_atom_index_selector_missing_field_raises(self):
        block = Block({"x": np.array([0.0, 1.0])})  # no "id" column
        with pytest.raises(KeyError, match="AtomIndexSelector"):
            AtomIndexSelector([1, 2]).mask(block)
