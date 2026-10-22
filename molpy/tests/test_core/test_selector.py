import numpy as np
import pytest

import molpy as mp


def _atoms() -> mp.Block:
    return mp.Block(
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
        sel = mp.core.ColumnSelector("C")
        assert sel.column == mp.core.keys.ELEMENT
        assert sel.mask(_atoms()).tolist() == [True, False, True, False]

    def test_a_keys_constant_names_the_column(self):
        assert mp.core.ColumnSelector("c3", column=mp.core.keys.TYPE).column == "type"
        assert mp.core.ColumnSelector("c3", column=mp.core.keys.TYPE).mask(
            _atoms()
        ).tolist() == [
            True,
            False,
            True,
            False,
        ]
        assert mp.core.ColumnSelector(2, column=mp.core.keys.TYPE_ID).mask(
            _atoms()
        ).tolist() == [
            False,
            True,
            False,
            False,
        ]

    def test_a_plain_column_name_works(self):
        assert mp.core.ColumnSelector(3, column="type_id").mask(_atoms()).tolist() == [
            False,
            False,
            False,
            True,
        ]

    def test_call_returns_the_filtered_block(self):
        picked = mp.core.ColumnSelector("c3", column=mp.core.keys.TYPE)(_atoms())
        assert isinstance(picked, mp.Block)
        assert picked["id"].tolist() == [1, 3]

    def test_an_integer_on_a_string_column_raises(self):
        # `type` is always a string label in molrs; 1 would match nothing.
        with pytest.raises(TypeError, match="'type' holds"):
            mp.core.ColumnSelector(1, column=mp.core.keys.TYPE).mask(_atoms())

    def test_a_string_on_a_numeric_column_raises(self):
        with pytest.raises(TypeError, match="'type_id' holds"):
            mp.core.ColumnSelector("1", column=mp.core.keys.TYPE_ID).mask(_atoms())

    def test_a_bool_on_an_integer_column_raises(self):
        with pytest.raises(TypeError):
            mp.core.ColumnSelector(True, column=mp.core.keys.TYPE_ID).mask(_atoms())

    def test_a_float_column_takes_a_number(self):
        assert mp.core.ColumnSelector(-0.5, column=mp.core.keys.CHARGE).mask(
            _atoms()
        ).tolist() == [
            False,
            False,
            False,
            True,
        ]

    def test_a_non_scalar_value_raises(self):
        with pytest.raises(TypeError, match="scalar"):
            mp.core.ColumnSelector(["C", "H"])  # type: ignore[arg-type]

    def test_a_missing_column_raises(self):
        block = mp.Block({"x": np.array([0.0, 1.0])})
        with pytest.raises(KeyError, match="ColumnSelector"):
            mp.core.ColumnSelector("C").mask(block)


class TestAtomIndexSelector:
    def test_selects_by_row_position_not_by_id(self):
        # ids are 1-based, so row 0 is id 1. Matching the id column would pick
        # only id 2 (row 1) instead.
        block = _atoms()
        sel = mp.core.AtomIndexSelector([0, 2])
        assert sel.mask(block).tolist() == [True, False, True, False]
        assert sel(block)["id"].tolist() == [1, 3]

    def test_call_and_mask_agree(self):
        block = _atoms()
        sel = mp.core.AtomIndexSelector(np.array([3, 1, 3]))
        assert sel.indices.tolist() == [1, 3]
        assert sel(block)["id"].tolist() == block[sel.mask(block)]["id"].tolist()

    def test_needs_no_id_column(self):
        block = mp.Block({"x": np.array([0.0, 1.0, 2.0])})
        assert mp.core.AtomIndexSelector([1]).mask(block).tolist() == [
            False,
            True,
            False,
        ]

    def test_an_out_of_range_index_raises(self):
        with pytest.raises(ValueError, match="out of range"):
            mp.core.AtomIndexSelector([4]).mask(_atoms())
        with pytest.raises(ValueError, match="out of range"):
            mp.core.AtomIndexSelector([4])(_atoms())

    def test_a_negative_index_raises(self):
        with pytest.raises(ValueError, match="non-negative"):
            mp.core.AtomIndexSelector([-1])

    def test_non_integer_indices_raise(self):
        with pytest.raises(TypeError):
            mp.core.AtomIndexSelector([0.5, 1.0])
        with pytest.raises(TypeError):
            mp.core.AtomIndexSelector([[0, 1]])

    def test_an_empty_selection_is_empty(self):
        assert not mp.core.AtomIndexSelector([]).mask(_atoms()).any()


class TestComposition:
    def test_and_or_not(self):
        block = _atoms()
        carbon = mp.core.ColumnSelector("C")
        first_two = mp.core.AtomIndexSelector([0, 1])
        assert (carbon & first_two).mask(block).tolist() == [True, False, False, False]
        assert (carbon | first_two).mask(block).tolist() == [True, True, True, False]
        assert (~carbon).mask(block).tolist() == [False, True, False, True]

    def test_a_region_on_either_side(self):
        block = _atoms()
        near = mp.core.Sphere([0.0, 0.0, 0.0], 1.5)  # rows 0, 1
        carbon = mp.core.ColumnSelector("C")
        assert (carbon & near).mask(block).tolist() == [True, False, False, False]
        assert (near & carbon).mask(block).tolist() == [True, False, False, False]
        assert (near | carbon).mask(block).tolist() == [True, True, True, False]
        assert (carbon | near)(block)["id"].tolist() == [1, 2, 3]


def test_nullable_values_do_not_match_fill_values():
    import numpy as np

    block = mp.Block({"foo": np.array([0, 0], dtype=np.int32)})
    block.set_validity("foo", [False, True])
    assert mp.core.ColumnSelector(0, column="foo").mask(block).tolist() == [False, True]


def test_unsigned_indices_cannot_wrap_to_negative():
    import numpy as np
    import pytest

    with pytest.raises(ValueError, match="range"):
        mp.core.AtomIndexSelector(np.array([2**64 - 1], dtype=np.uint64))
