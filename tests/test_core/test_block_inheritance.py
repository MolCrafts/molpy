"""Verify ``molpy.Block`` truly inherits from ``molrs.Block``.

These are contract tests for the inheritance refactor: after the change,
``molpy.Block`` must satisfy ``isinstance(b, molrs.Block)`` so any
``molrs.*`` API that takes a ``Block`` accepts a molpy block directly.

Python-only column state (object-dtype columns like string symbols)
stays on the Python ``__dict__`` and must stay invisible to the Rust
side — confirmed by round-tripping through a ``molrs.Frame``.
"""

import molrs
import numpy as np

import molpy
from molpy.core.frame import Block


class TestBlockInheritance:
    """``molpy.Block`` IS-A ``molrs.Block``."""

    def test_molpy_block_is_a_molrs_block(self):
        b = Block()
        assert isinstance(b, molrs.Block)
        assert isinstance(b, Block)

    def test_subclass_relation_holds_at_type_level(self):
        assert issubclass(Block, molrs.Block)

    def test_molpy_block_directly_accepted_by_molrs_frame(self):
        """A ``molpy.Block`` must extract through PyO3's ``Block``
        downcast — no copying, no conversion shim."""
        b = Block({"x": np.array([1.0, 2.0], dtype=np.float32)})
        frame = molrs.Frame()
        frame["atoms"] = b  # would raise TypeError pre-refactor
        assert "atoms" in frame
        assert frame["atoms"].nrows == 2


class TestObjectColumnsInvisibleToMolrs:
    """Object-dtype columns live on the Python side only.

    The Rust ``molrs.Block`` schema is numeric / bool / string-list
    only — Python ``object`` dtype has no Rust representation. We
    keep those columns on the Python ``__dict__`` (the historical
    ``_objects`` cache) and confirm a round-trip through a
    ``molrs.Frame`` does not surface them on the Rust side.
    """

    def test_object_columns_stored_python_side(self):
        b = Block(
            {
                "x": np.array([1.0, 2.0], dtype=np.float32),
                "symbol": np.array(["H", "O"], dtype=object),
            }
        )
        # Python side sees both columns
        assert "x" in b
        assert "symbol" in b
        assert b["symbol"].tolist() == ["H", "O"]

    def test_object_columns_invisible_to_molrs_roundtrip(self):
        b = Block(
            {
                "x": np.array([1.0, 2.0], dtype=np.float32),
                "symbol": np.array(["H", "O"], dtype=object),
            }
        )
        # Hand the molpy block to a molrs.Frame — PyO3 only sees the
        # Rust slot (the numeric columns).
        mf = molrs.Frame()
        mf["atoms"] = b
        # Retrieving from the molrs.Frame yields a bare molrs.Block,
        # which knows only what the Rust slot stored.
        rs_block = mf["atoms"]
        assert "x" in rs_block
        assert "symbol" not in rs_block
