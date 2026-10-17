import numpy as np
import pytest

import molpy as mp
from molpy.core import AtomIndexSelector, Block, ColumnSelector, keys


def _atoms() -> Block:
    return Block(
        {
            "id": np.array([1, 2, 3, 4], dtype=np.uint32),  # LAMMPS-style 1-based
            "element": ["C", "H", "C", "O"],
            "type": ["c3", "hc", "c3", "oh"],
            "type_id": np.array([1, 2, 1, 3], dtype=np.uint32),
            "charge": np.array([-0.1, 0.1, -0.1, -0.5]),
            "x": np.array([0.0, 1.0, 2.0, 3.0]),
            "y": np.zeros(4),
            "z": np.zeros(4),
        }
    )


class TestColumnSelector:
    def test_defaults_to_the_element_key(self):
        sel = ColumnSelector("C")
        assert sel.column == keys.ELEMENT
        assert sel.mask(_atoms()).tolist() == [True, False, True, False]

    def test_a_keys_constant_names_the_column(self):
        assert ColumnSelector("c3", column=keys.TYPE).column == "type"
        assert ColumnSelector("c3", column=keys.TYPE).mask(_atoms()).tolist() == [
            True,
            False,
            True,
            False,
        ]
        assert ColumnSelector(2, column=keys.TYPE_ID).mask(_atoms()).tolist() == [
            False,
            True,
            False,
            False,
        ]

    def test_a_plain_column_name_works(self):
        assert ColumnSelector(3, column="type_id").mask(_atoms()).tolist() == [
            False,
            False,
            False,
            True,
        ]

    def test_call_returns_the_filtered_block(self):
        picked = ColumnSelector("c3", column=keys.TYPE)(_atoms())
        assert isinstance(picked, Block)
        assert picked["id"].tolist() == [1, 3]

    def test_an_integer_on_a_string_column_raises(self):
        # `type` is always a string label in molrs; 1 would match nothing.
        with pytest.raises(TypeError, match="'type' holds"):
            ColumnSelector(1, column=keys.TYPE).mask(_atoms())

    def test_a_string_on_a_numeric_column_raises(self):
        with pytest.raises(TypeError, match="'type_id' holds"):
            ColumnSelector("1", column=keys.TYPE_ID).mask(_atoms())

    def test_a_bool_on_an_integer_column_raises(self):
        with pytest.raises(TypeError):
            ColumnSelector(True, column=keys.TYPE_ID).mask(_atoms())

    def test_a_float_column_takes_a_number(self):
        assert ColumnSelector(-0.5, column=keys.CHARGE).mask(_atoms()).tolist() == [
            False,
            False,
            False,
            True,
        ]

    def test_a_non_scalar_value_raises(self):
        with pytest.raises(TypeError, match="scalar"):
            ColumnSelector(["C", "H"])  # type: ignore[arg-type]

    def test_a_missing_column_raises(self):
        block = Block({"x": np.array([0.0, 1.0])})
        with pytest.raises(KeyError, match="ColumnSelector"):
            ColumnSelector("C").mask(block)


class TestAtomIndexSelector:
    def test_selects_by_row_position_not_by_id(self):
        # ids are 1-based, so row 0 is id 1. Matching the id column would pick
        # only id 2 (row 1) instead.
        block = _atoms()
        sel = AtomIndexSelector([0, 2])
        assert sel.mask(block).tolist() == [True, False, True, False]
        assert sel(block)["id"].tolist() == [1, 3]

    def test_call_and_mask_agree(self):
        block = _atoms()
        sel = AtomIndexSelector(np.array([3, 1, 3]))
        assert sel.indices.tolist() == [1, 3]
        assert sel(block)["id"].tolist() == block[sel.mask(block)]["id"].tolist()

    def test_needs_no_id_column(self):
        block = Block({"x": np.array([0.0, 1.0, 2.0])})
        assert AtomIndexSelector([1]).mask(block).tolist() == [False, True, False]

    def test_an_out_of_range_index_raises(self):
        with pytest.raises(ValueError, match="out of range"):
            AtomIndexSelector([4]).mask(_atoms())
        with pytest.raises(ValueError, match="out of range"):
            AtomIndexSelector([4])(_atoms())

    def test_a_negative_index_raises(self):
        with pytest.raises(ValueError, match="non-negative"):
            AtomIndexSelector([-1])

    def test_non_integer_indices_raise(self):
        with pytest.raises(TypeError):
            AtomIndexSelector([0.5, 1.0])
        with pytest.raises(TypeError):
            AtomIndexSelector([[0, 1]])

    def test_an_empty_selection_is_empty(self):
        assert not AtomIndexSelector([]).mask(_atoms()).any()


class TestComposition:
    def test_and_or_not(self):
        block = _atoms()
        carbon = ColumnSelector("C")
        first_two = AtomIndexSelector([0, 1])
        assert (carbon & first_two).mask(block).tolist() == [True, False, False, False]
        assert (carbon | first_two).mask(block).tolist() == [True, True, True, False]
        assert (~carbon).mask(block).tolist() == [False, True, False, True]

    def test_a_region_on_either_side(self):
        block = _atoms()
        near = mp.core.Sphere([0.0, 0.0, 0.0], 1.5)  # rows 0, 1
        carbon = ColumnSelector("C")
        assert (carbon & near).mask(block).tolist() == [True, False, False, False]
        assert (near & carbon).mask(block).tolist() == [True, False, False, False]
        assert (near | carbon).mask(block).tolist() == [True, True, True, False]
        assert (carbon | near)(block)["id"].tolist() == [1, 2, 3]


def test_nullable_values_do_not_match_fill_values():
    import numpy as np
    from molpy.core import Block, ColumnSelector

    block = Block({"foo": np.array([0, 0], dtype=np.int32)})
    block.set_validity("foo", [False, True])
    assert ColumnSelector(0, column="foo").mask(block).tolist() == [False, True]


def test_unsigned_indices_cannot_wrap_to_negative():
    import numpy as np
    import pytest
    from molpy.core import AtomIndexSelector

    with pytest.raises(ValueError, match="range"):
        AtomIndexSelector(np.array([2**64 - 1], dtype=np.uint64))
