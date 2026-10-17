"""``molrs.core.Block`` — the one column store, native end to end.

There is exactly one ``Block`` class: the PyO3 ``molrs._native.Block``. Every
capability below (construction from a mapping, schema-dtype adoption, row and
multi-column indexing, rename, deep copy, sort) is implemented once, in Rust.
"""

import pickle

import molrs
import numpy as np
import pytest
from molrs.core import Block


class _SubBlock(molrs.core.Block):
    """Module level, so pickle can find it."""


class TestOneClass:
    def test_molrs_block_is_the_native_class(self):
        assert molrs.core.Block is molrs._native.Block

    def test_a_block_can_be_subclassed(self):
        """Core data classes are extensible; a subclass is still a Block."""

        class Sub(molrs.core.Block):
            pass

        assert isinstance(Sub(), molrs.core.Block)

    def test_a_block_subclass_pickles_as_itself(self):
        sub = _SubBlock()
        sub.insert("x", np.array([1.0, 2.0]))
        sub.tag = "kept"
        back = pickle.loads(pickle.dumps(sub))
        assert type(back) is _SubBlock
        assert back.tag == "kept"
        assert back["x"].tolist() == [1.0, 2.0]


class TestBlockConstruction:
    def test_empty(self):
        b = Block()
        assert b.n_rows == 0
        assert len(b) == 0
        assert b.keys() == []

    def test_repr_empty(self):
        assert "Block" in repr(Block())

    def test_from_a_mapping(self):
        b = Block({"x": [1.0, 2.0, 3.0], "name": ["C", "H", "O"]})
        assert sorted(b.keys()) == ["name", "x"]
        assert b.n_rows == 3
        assert b["x"].dtype == np.float64
        np.testing.assert_array_equal(b["x"], [1.0, 2.0, 3.0])
        assert list(b["name"]) == ["C", "H", "O"]

    def test_the_mapping_ctor_adopts_the_schema_dtype(self):
        # `id` is declared uint by the Frame schema; numpy's default int64 for
        # [1, 2] is width, not meaning, so the constructor stores uint.
        b = Block({"id": np.array([1, 2], dtype=np.int64)})
        assert b.dtype("id") == "uint"
        np.testing.assert_array_equal(b["id"], [1, 2])

    def test_a_length_mismatch_reports_the_real_cause(self):
        with pytest.raises(ValueError) as exc:
            Block({"x": np.zeros(6), "y": np.zeros(4)})
        assert "array-like" not in str(exc.value)

    def test_a_non_array_like_value_is_named(self):
        with pytest.raises(ValueError, match="array-like"):
            Block({"x": object()})

    def test_an_existing_block_is_not_a_mapping_argument(self):
        # Copying is `block.copy()`; there is no wrap-an-existing-block form.
        with pytest.raises(TypeError, match="copy"):
            Block(Block({"x": [1.0]}))


class TestBlockInsert:
    def test_f64(self):
        b = Block()
        b.insert("x", np.array([1.0, 2.0], dtype=np.float64))
        assert b.n_rows == 2
        assert "x" in b

    def test_uint(self):
        b = Block()
        b.insert("id", np.array([10, 20, 30], dtype=np.uint32))
        assert b.n_rows == 3

    def test_bool(self):
        b = Block()
        b.insert("mask", np.array([True, False, True]))
        assert b.n_rows == 3

    def test_2d_array(self):
        b = Block()
        b.insert("pos", np.zeros((5, 3), dtype=np.float64))
        assert b.n_rows == 5

    def test_nrows_enforcement(self):
        b = Block()
        b.insert("x", np.array([1.0, 2.0], dtype=np.float64))
        with pytest.raises(ValueError):
            b.insert("y", np.array([1.0, 2.0, 3.0], dtype=np.float64))

    def test_int32_accepted(self):
        b = Block()
        b.insert("count", np.array([1, 2], dtype=np.int32))
        assert b.dtype("count") == "int"

    def test_insert_promotes_a_narrow_float(self):
        # There is one float column: `F` (f64). A float32 array widens on
        # the way in.
        b = Block()
        b.insert("x", np.array([1.0, 2.0], dtype=np.float32))
        assert b.dtype("x") == "float"
        assert b["x"].dtype == np.float64

    def test_overwrite_key(self):
        b = Block()
        b.insert("x", np.array([1.0, 2.0], dtype=np.float64))
        b.insert("x", np.array([3.0, 4.0], dtype=np.float64))
        np.testing.assert_allclose(b["x"], [3.0, 4.0])


class TestBlockGet:
    def test_roundtrip_f64(self):
        b = Block()
        b.insert("x", np.array([1.1, 2.2], dtype=np.float64))
        np.testing.assert_allclose(b["x"], [1.1, 2.2], atol=1e-12)

    def test_roundtrip_uint(self):
        b = Block()
        b.insert("id", np.array([10, 20], dtype=np.uint32))
        np.testing.assert_array_equal(b["id"], [10, 20])

    def test_roundtrip_bool(self):
        b = Block()
        b.insert("m", np.array([True, False]))
        np.testing.assert_array_equal(b["m"], [True, False])

    def test_missing_key_raises_key_error(self):
        with pytest.raises(KeyError):
            Block()["nonexistent"]

    def test_missing_column_lists_candidates(self):
        b = Block()
        b.insert("x", np.array([1.0, 2.0], dtype=np.float64))
        with pytest.raises(KeyError, match="x"):
            b["xx"]

    def test_roundtrip_2d(self):
        b = Block()
        data = np.array([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]], dtype=np.float64)
        b.insert("pos", data)
        np.testing.assert_allclose(b["pos"], data)

    def test_view_is_a_zero_copy_numpy_array(self):
        b = Block({"pos": np.arange(9.0).reshape(3, 3)})
        view = b["pos"]
        assert isinstance(view, np.ndarray)
        assert view.base is not None


class TestBlockKeys:
    def test_iteration_yields_the_column_names(self):
        b = Block({"x": [1.0, 2.0], "y": [3.0, 4.0]})
        assert sorted(iter(b)) == ["x", "y"]
        assert sorted(b) == ["x", "y"]

    def test_dict_of_a_block_is_its_columns(self):
        b = Block({"x": [1.0, 2.0], "y": [3.0, 4.0]})
        d = dict(b)
        assert set(d) == {"x", "y"}
        np.testing.assert_array_equal(d["y"], [3.0, 4.0])

    def test_contains(self):
        b = Block({"x": [1.0]})
        assert "x" in b
        assert "y" not in b
        assert 1 not in b

    def test_a_schema_key_is_a_column_name(self):
        b = Block()
        z = molrs.core.keys.ATOMIC_NUMBER
        b[z] = np.array([1, 8], dtype=np.int64)
        assert z in b
        assert "atomic_number" in b
        np.testing.assert_array_equal(b[z], [1, 8])
        np.testing.assert_array_equal(b[z], [1, 8])
        assert b.dtype(z) == "uint"
        del b[z]
        assert z not in b

    def test_del(self):
        b = Block({"x": [1.0]})
        del b["x"]
        assert "x" not in b
        assert len(b) == 0

    def test_del_missing_raises_key_error(self):
        b = Block()
        with pytest.raises(KeyError):
            del b["nonexistent"]

    def test_dtype(self):
        b = Block()
        b.insert("x", np.array([1.0], dtype=np.float64))
        b.insert("id", np.array([1], dtype=np.uint32))
        assert b.dtype("x") == "float"
        assert b.dtype("id") == "uint"

    def test_dtype_missing_raises_key_error(self):
        with pytest.raises(KeyError):
            Block().dtype("missing")

    def test_dtype_tells_float_from_uint(self):
        b = Block()
        b.insert("x", np.array([1.0], dtype=np.float64))
        b.insert("id", np.array([1], dtype=np.uint32))
        assert b.dtype("x") == "float"
        assert b.dtype("id") != "float"
        assert "missing" not in b

    def test_repr(self):
        b = Block()
        b.insert("x", np.array([1.0, 2.0], dtype=np.float64))
        assert "Block" in repr(b)
        assert "2" in repr(b)


class TestBlockSubscriptAssignment:
    def test_setitem_stores_a_column(self):
        b = Block()
        b["x"] = np.array([1.0, 2.0, 3.0], dtype=np.float64)
        np.testing.assert_array_equal(b["x"], [1.0, 2.0, 3.0])

    def test_setitem_accepts_a_python_list(self):
        b = Block()
        b["x"] = [1.0, 2.0]
        assert b.dtype("x") == "float"
        np.testing.assert_array_equal(b["x"], [1.0, 2.0])

    def test_setitem_adopts_the_schema_dtype(self):
        b = Block()
        b["id"] = np.array([1, 2], dtype=np.int64)
        assert b.dtype("id") == "uint"
        b["x"] = np.array([1.5, 2.5], dtype=np.float32)
        assert b.dtype("x") == "float"

    def test_setitem_refuses_values_the_schema_dtype_cannot_hold(self):
        b = Block()
        with pytest.raises(ValueError, match="schema"):
            b["id"] = np.array([-1, 2], dtype=np.int64)  # uint cannot hold -1
        assert "id" not in b
        with pytest.raises(ValueError, match="schema"):
            b["mol_id"] = np.array([1.5, 2.0])  # 1.5 is not an integer
        assert "mol_id" not in b

    def test_the_adopted_dtype_is_the_one_the_schema_reports(self):
        b = Block({"id": [1, 2], "mol_id": [1, 1], "ix": [0, 1], "x": [0, 1]})
        for key in ("id", "mol_id", "ix", "x"):
            assert b[key].dtype == np.dtype(molrs.core.schema.column(key).numpy_dtype)

    def test_an_unconstrained_key_keeps_its_dtype(self):
        b = Block()
        b["count"] = np.array([1, 2], dtype=np.int64)
        assert b.dtype("count") == "i64"

    def test_setitem_stores_a_string_column(self):
        b = Block()
        b["name"] = ["C", "H", "O"]
        assert list(b["name"]) == ["C", "H", "O"]

    def test_an_empty_string_column_is_still_a_string_array(self):
        b = Block()
        b["name"] = np.array([], dtype=str)
        assert b.dtype("name") == "string"
        assert b["name"].dtype.kind == "U"
        assert b["name"].shape == (0,)

    def test_a_string_column_indexes_like_a_numeric_one(self):
        b = Block({"name": ["C", "H", "O"]})
        out = b["name"]
        assert out.shape == (3,)
        assert out.dtype.kind == "U"
        assert out.tolist() == ["C", "H", "O"]
        # Numpy strings are fixed-width, so this is a copy: writing it does
        # not change the column.
        out[0] = "N"
        assert b["name"].tolist() == ["C", "H", "O"]

    def test_setitem_replaces_an_existing_column(self):
        b = Block()
        b["x"] = np.zeros(3, dtype=np.float64)
        b["x"] = np.ones(3, dtype=np.float64)
        np.testing.assert_array_equal(b["x"], np.ones(3))
        assert len(b) == 1

    def test_setitem_enforces_the_row_count(self):
        b = Block()
        b["x"] = np.zeros(3, dtype=np.float64)
        with pytest.raises(ValueError):
            b["y"] = np.zeros(2, dtype=np.float64)

    def test_setitem_rejects_an_object_column(self):
        b = Block()
        with pytest.raises(molrs.core.BlockDtypeError):
            b["bad"] = np.array([object(), object()], dtype=object)

    def test_setitem_refuses_a_scalar(self):
        b = Block({"x": np.array([1.0, 2.0])})
        with pytest.raises(ValueError, match="at least 1-D"):
            b["q"] = 1.0
        assert "q" not in b

    def test_roundtrip_through_all_three_dunders(self):
        b = Block()
        b["x"] = np.array([1.0], dtype=np.float64)
        assert "x" in b
        del b["x"]
        assert "x" not in b


class TestBlockMultiColumnIndexing:
    """``block["x", "y", "z"]`` — several equal-shaped columns, side by side."""

    @staticmethod
    def _xyz() -> Block:
        return Block(
            {
                "x": np.array([0.0, 1.0], dtype=np.float64),
                "y": np.array([2.0, 3.0], dtype=np.float64),
                "z": np.array([4.0, 5.0], dtype=np.float64),
                "id": np.array([1, 2], dtype=np.uint32),
            }
        )

    def test_tuple_and_list_keys_agree(self):
        block = self._xyz()
        np.testing.assert_array_equal(block["x", "y", "z"], block[["x", "y", "z"]])

    def test_columns_are_stacked_one_per_output_column(self):
        stacked = self._xyz()["x", "y", "z"]
        assert stacked.shape == (2, 3)
        np.testing.assert_allclose(stacked, [[0.0, 2.0, 4.0], [1.0, 3.0, 5.0]])

    def test_the_canonical_coordinate_keys_work_as_a_tuple(self):
        assert self._xyz()[molrs.core.keys.COORDS].shape == (2, 3)

    def test_a_missing_column_names_itself(self):
        with pytest.raises(KeyError, match="absent"):
            self._xyz()["x", "absent"]

    def test_an_empty_name_list_is_a_key_error(self):
        with pytest.raises(KeyError):
            self._xyz()[[]]

    def test_mixed_dtypes_are_refused(self):
        with pytest.raises(ValueError, match="dtype"):
            self._xyz()["x", "id"]

    def test_mixed_shapes_are_refused(self):
        block = Block({"a": np.zeros((2, 3)), "b": np.zeros((2, 2))})
        with pytest.raises(ValueError, match="shape"):
            block["a", "b"]

    def test_string_columns_stack_like_numeric_ones(self):
        block = Block({"a": ["C", "H"], "b": ["O", "N"]})
        stacked = block["a", "b"]
        assert stacked.shape == (2, 2)
        assert stacked.tolist() == [["C", "O"], ["H", "N"]]


class TestBlockMultiColumnAssignment:
    """``block["x", "y", "z"] = arr`` spreads an ``(N, k)`` array over k columns.

    Kept: the operator's backmap script writes coordinates this way
    (``backmap_pe_pma/backmap.py:19,59``).
    """

    @staticmethod
    def _xyz() -> Block:
        return Block(
            {
                "x": np.array([0.0, 1.0, 2.0], dtype=np.float64),
                "y": np.array([3.0, 4.0, 5.0], dtype=np.float64),
                "z": np.array([6.0, 7.0, 8.0], dtype=np.float64),
            }
        )

    @staticmethod
    def _snapshot(block: Block) -> dict[str, np.ndarray]:
        return {k: np.array(block[k], copy=True) for k in block}

    def _assert_unchanged(self, block: Block, before: dict[str, np.ndarray]) -> None:
        assert sorted(block.keys()) == sorted(before)
        for k, v in before.items():
            np.testing.assert_array_equal(block[k], v)

    def test_a_tuple_key_spreads_the_columns(self):
        block = self._xyz()
        arr = np.arange(9, dtype=np.float64).reshape(3, 3) + 10.0
        block["x", "y", "z"] = arr
        for i, name in enumerate(("x", "y", "z")):
            np.testing.assert_array_equal(block[name], arr[:, i])
            assert block[name].dtype == np.float64

    def test_a_list_key_spreads_the_columns(self):
        block = self._xyz()
        arr = np.array([[1.5, -1.5], [2.5, -2.5], [3.5, -3.5]], dtype=np.float64)
        block[["x", "y"]] = arr
        np.testing.assert_array_equal(block["x"], [1.5, 2.5, 3.5])
        np.testing.assert_array_equal(block["y"], [-1.5, -2.5, -3.5])
        np.testing.assert_array_equal(block["z"], [6.0, 7.0, 8.0])

    def test_reading_then_writing_the_same_key_is_identity(self):
        block = self._xyz()
        before = self._snapshot(block)
        block["x", "y", "z"] = block["x", "y", "z"]
        self._assert_unchanged(block, before)

    def test_a_column_count_mismatch_is_refused_without_writing(self):
        block = self._xyz()
        before = self._snapshot(block)
        with pytest.raises(ValueError, match="3"):
            block["x", "y", "z"] = np.zeros((3, 2), dtype=np.float64)
        self._assert_unchanged(block, before)

    def test_a_row_count_mismatch_is_refused_without_writing(self):
        block = self._xyz()
        before = self._snapshot(block)
        with pytest.raises(ValueError, match="4"):
            block["x", "y", "z"] = np.zeros((4, 3), dtype=np.float64)
        self._assert_unchanged(block, before)

    def test_a_schema_refusal_comes_before_the_first_write(self):
        block = Block({"id": np.array([1, 2], dtype=np.uint32), "x": [0.0, 0.0]})
        before = self._snapshot(block)
        with pytest.raises(ValueError, match="schema"):
            block["x", "id"] = np.array([[9.0, 1.0], [9.0, -1.0]])
        self._assert_unchanged(block, before)

    def test_an_empty_key_is_a_key_error(self):
        with pytest.raises(KeyError, match="Empty"):
            self._xyz()[()] = np.zeros((3, 0), dtype=np.float64)

    def test_a_repeated_name_is_refused_without_writing(self):
        block = self._xyz()
        before = self._snapshot(block)
        with pytest.raises(ValueError, match="x"):
            block["x", "x"] = np.zeros((3, 2), dtype=np.float64)
        self._assert_unchanged(block, before)


class TestBlockRowSelection:
    @staticmethod
    def _three() -> Block:
        return Block(
            {
                "x": np.array([0.0, 1.0, 2.0], dtype=np.float64),
                "id": np.array([10, 20, 30], dtype=np.uint32),
            }
        )

    def test_a_bool_mask_selects_its_true_rows(self):
        sub = self._three()[np.array([True, False, True])]
        assert type(sub) is Block
        np.testing.assert_array_equal(sub["x"], [0.0, 2.0])
        np.testing.assert_array_equal(sub["id"], [10, 30])

    def test_an_int_array_selects_in_order_and_wraps_negatives(self):
        sub = self._three()[np.array([2, -3])]
        np.testing.assert_array_equal(sub["x"], [2.0, 0.0])

    def test_a_slice_selects_rows(self):
        sub = self._three()[1:]
        assert type(sub) is Block
        np.testing.assert_array_equal(sub["id"], [20, 30])
        np.testing.assert_array_equal(self._three()[::-2]["x"], [2.0, 0.0])

    def test_a_selection_carries_the_validity_mask(self):
        b = Block()
        b.insert("tag", np.array([1, 2, 3], dtype=np.int32), [True, False, True])
        np.testing.assert_array_equal(
            b[np.array([1, 2])].validity("tag"), [False, True]
        )

    def test_a_mask_of_the_wrong_length_is_an_index_error(self):
        with pytest.raises(IndexError):
            self._three()[np.array([True, False])]

    def test_an_index_below_minus_n_is_an_index_error(self):
        with pytest.raises(IndexError):
            self._three()[np.array([-4])]

    def test_a_float_index_is_a_type_error(self):
        with pytest.raises(TypeError):
            self._three()[np.array([0.5])]

    def test_an_index_past_the_end_is_a_value_error(self):
        with pytest.raises(ValueError):
            self._three()[np.array([3])]

    def test_an_index_array_gathers_rows_in_its_order(self):
        np.testing.assert_array_equal(self._three()[np.array([2, 0])]["x"], [2.0, 0.0])


class TestBlockRename:
    def test_rename_moves_the_column(self):
        b = Block({"old": [1.0, 2.0]})
        b.rename("old", "new")
        assert "new" in b and "old" not in b

    def test_rename_missing_raises_key_error(self):
        with pytest.raises(KeyError):
            Block({"x": [1.0]}).rename("missing", "new")

    def test_rename_onto_a_canonical_key_adopts_its_dtype(self):
        # A format-native column carries the file's width; renaming it onto
        # `id` (uint) adopts the schema dtype when the values allow it.
        b = Block()
        b.insert("serial", np.array([1, 2], dtype=np.int64))
        b.rename("serial", molrs.core.keys.ID)
        assert "serial" not in b
        assert b.dtype("id") == "uint"
        np.testing.assert_array_equal(b["id"], [1, 2])

    def test_rename_keeps_the_column_in_place_across_adoption(self):
        b = Block()
        b.insert("serial", np.array([1, 2], dtype=np.int64))
        b.insert("x", np.array([0.5, 1.5]))
        b.rename("serial", "id")
        assert list(b.keys()) == ["id", "x"]

    def test_rename_onto_an_existing_column_names_both(self):
        b = Block({"x": [1.0], "y": [2.0]})
        with pytest.raises(KeyError, match="'x' to 'y'"):
            b.rename("x", "y")
        assert list(b.keys()) == ["x", "y"]

    def test_rename_keeps_the_validity_mask_across_adoption(self):
        b = Block()
        b.insert("serial", np.array([1, 0], dtype=np.int64), [True, False])
        b.rename("serial", "id")
        np.testing.assert_array_equal(b.validity("id"), [True, False])

    def test_rename_refuses_values_the_canonical_dtype_cannot_hold(self):
        b = Block()
        b.insert("serial", np.array([-1, 2], dtype=np.int64))
        with pytest.raises(ValueError, match="schema"):
            b.rename("serial", "id")
        assert "serial" in b and "id" not in b


class TestBlockCopy:
    def test_copy_does_not_share_buffers(self):
        b = Block({"x": np.array([1.0, 2.0, 3.0])})
        c = b.copy()
        assert not np.shares_memory(b["x"], c["x"])
        c["x"][0] = 9.0
        np.testing.assert_array_equal(b["x"], [1.0, 2.0, 3.0])

    def test_copy_of_a_numpy_backed_column_does_not_alias_the_source_array(self):
        source = np.array([1.0, 2.0, 3.0])
        b = Block()
        b.insert("x", source)
        c = b.copy()
        source[0] = 7.0
        np.testing.assert_array_equal(c["x"], [1.0, 2.0, 3.0])

    def test_copy_keeps_masks_and_shape(self):
        b = Block()
        b.insert(
            "tag", np.array([1, 0, 3, 0], dtype=np.int32), [True, False, True, False]
        )
        b.set_shape([2, 2])
        c = b.copy()
        np.testing.assert_array_equal(c.validity("tag"), [True, False, True, False])
        assert c.structural_shape == [2, 2]


class TestBlockSort:
    def test_sort_returns_a_new_block_and_leaves_the_original(self):
        b = Block({"x": [3.0, 1.0, 2.0], "y": [30.0, 10.0, 20.0]})
        s = b.sort("x")
        assert type(s) is Block
        np.testing.assert_allclose(s["x"], [1.0, 2.0, 3.0])
        np.testing.assert_allclose(s["y"], [10.0, 20.0, 30.0])
        np.testing.assert_allclose(b["x"], [3.0, 1.0, 2.0])

    def test_sort_reverse(self):
        s = Block({"x": [1.0, 2.0, 3.0]}).sort("x", reverse=True)
        np.testing.assert_allclose(s["x"], [3.0, 2.0, 1.0])

    def test_sort_by_a_missing_column_is_a_key_error(self):
        with pytest.raises(KeyError):
            Block({"x": [1.0]}).sort("y")
