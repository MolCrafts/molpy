"""Block numpy-only dtype admission contract (spec frame-block-sink-molrs-02).

The Rust Store only holds numpy-representable dtypes (float, int, bool, str).
Object / None-bearing / ragged columns are rejected at insert with a clear,
column-named ``mp.core.BlockDtypeError`` (fail-fast) — never silently routed to a
Python-side overflow dict. Supported dtypes coerce, store, and expose zero-copy
views unchanged.
"""

import molpy as mp
import numpy as np
import pytest

# --- ac-008: BlockDtypeError is a public, documented symbol ------------------


class TestErrorType:
    def test_importable(self):
        assert hasattr(mp.core, "BlockDtypeError")

    def test_subclasses_typeerror(self):
        # Fixed in docs as a TypeError subclass so callers can `except` it.
        assert issubclass(mp.core.BlockDtypeError, TypeError)


# --- ac-001 / ac-002 / ac-003: rejection triggers ---------------------------


class TestRejection:
    def test_object_dtype_raises_named(self):
        b = mp.core.Block()
        with pytest.raises(mp.core.BlockDtypeError) as exc:
            b.insert("mixed", np.array(["a", 1, None], dtype=object))
        msg = str(exc.value)
        assert "mixed" in msg  # column name
        assert "object" in msg.lower()  # detected dtype

    def test_none_bearing_raises_identified(self):
        b = mp.core.Block()
        with pytest.raises(mp.core.BlockDtypeError) as exc:
            b.insert("c", np.array([1.0, None]))
        msg = str(exc.value)
        assert "c" in msg
        assert "none" in msg.lower()  # identified as None-bearing

    def test_ragged_raises_named(self):
        b = mp.core.Block()
        with pytest.raises(mp.core.BlockDtypeError) as exc:
            b.insert("rag", np.array([[1, 2], [3]], dtype=object))
        assert "rag" in str(exc.value)

    def test_explicit_object_column_raises(self):
        b = mp.core.Block()
        with pytest.raises(mp.core.BlockDtypeError):
            b.insert("o", np.empty(3, dtype=object))

    def test_setitem_rejects_object(self):
        b = mp.core.Block()
        with pytest.raises(mp.core.BlockDtypeError):
            b["mixed"] = np.array(["a", 1, None], dtype=object)


# --- ac-004: supported dtypes coerce and store ------------------------------


class TestSupportedDtypes:
    @pytest.mark.parametrize(
        "arr,kind",
        [
            (np.array([1.0, 2.0, 3.0], dtype=np.float64), "float"),
            (np.array([10, 20, 30], dtype=np.int64), "i64"),
            (np.array([True, False, True]), "bool"),
            (np.array([[1.0, 2.0], [3.0, 4.0]], dtype=np.float64), "float"),
        ],
    )
    def test_round_trip(self, arr, kind):
        b = mp.core.Block()
        b.insert("value", arr)
        out = b["value"]
        np.testing.assert_array_equal(out, arr)
        assert b.dtype("value") == kind

    def test_str_round_trip(self):
        b = mp.core.Block()
        b.insert("name", ["a", "b", "c"])
        assert list(b["name"]) == ["a", "b", "c"]


# --- ac-005: zero-copy views ------------------------------------------------


class TestZeroCopy:
    def test_float_view_shares_memory(self):
        b = mp.core.Block()
        b.insert("x", np.array([1.0, 2.0, 3.0], dtype=np.float64))
        v = b["x"]
        assert v.base is not None  # Arc-backed window, not a defensive copy


# --- ac-006: empty supported-dtype column stores ----------------------------


class TestEmptyColumn:
    def test_empty_float_stores(self):
        b = mp.core.Block()
        b.insert("x", np.array([], dtype=np.float64))
        out = b["x"]
        assert out.shape == (0,)
        assert b.dtype("x") == "float"

    def test_empty_object_rejected(self):
        b = mp.core.Block()
        with pytest.raises(mp.core.BlockDtypeError):
            b.insert("x", np.array([], dtype=object))
