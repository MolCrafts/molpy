"""Verify ``molpy.Block`` collapses onto the canonical ``molrs.Block``.

After the ``frame-block-sink`` cutover, ``molpy.core.frame.Block is
molrs.Block`` — there is no molpy subclass and no Python-side object-column
overflow. All columns (numeric / bool / string) live in the Rust Store and are
visible to every ``molrs.*`` API; object / None / ragged columns are rejected
fail-fast.
"""

import molrs
import numpy as np
import pytest

from molpy.core.frame import Block


class TestBlockIdentityCollapse:
    """``molpy.Block`` IS ``molrs.Block``."""

    def test_block_is_molrs_block(self):
        assert Block is molrs.Block

    def test_instance_is_molrs_block(self):
        assert isinstance(Block(), molrs.Block)

    def test_block_accepted_by_molrs_frame(self):
        b = Block({"x": np.array([1.0, 2.0], dtype=np.float64)})
        frame = molrs.Frame()
        frame["atoms"] = b
        assert "atoms" in frame
        assert frame["atoms"].nrows == 2


class TestStringColumnsLiveInStore:
    """String columns are stored natively and ARE visible to molrs (no overflow)."""

    def test_string_column_round_trips_through_molrs_frame(self):
        b = Block(
            {
                "x": np.array([1.0, 2.0], dtype=np.float64),
                "symbol": np.array(["H", "O"]),  # native str -> molrs str column
            }
        )
        mf = molrs.Frame()
        mf["atoms"] = b
        rs_block = mf["atoms"]
        # Both columns survive — strings are first-class in the Store now.
        assert "x" in rs_block
        assert "symbol" in rs_block
        assert list(rs_block["symbol"]) == ["H", "O"]


class TestNumpyOnlyRejection:
    """Object / None columns are rejected (no _objects overflow)."""

    def test_object_column_rejected(self):
        b = Block()
        with pytest.raises(molrs.BlockDtypeError):
            b["symbol"] = np.array(["H", "O"], dtype=object)

    def test_none_bearing_column_rejected(self):
        b = Block()
        with pytest.raises(molrs.BlockDtypeError):
            b["c"] = np.array([1.0, None])

    def test_no_objects_overflow_attribute(self):
        b = Block({"x": np.array([1.0, 2.0], dtype=np.float64)})
        assert not hasattr(b, "_objects")
