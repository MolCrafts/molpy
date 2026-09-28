"""Field vocabulary: the native ``FieldFormatter`` re-exported by molpy."""

import numpy as np

import molrs

from molpy.core.fields import CHARGE, FieldFormatter


class _AcLike(FieldFormatter):
    _field_formatters = {"q": CHARGE}


class TestFieldFormatter:
    def test_canonicalize_and_localize_rename_in_place_and_invert(self):
        block = molrs.Block()
        block["q"] = np.array([0.1, -0.1])
        block["x"] = np.array([0.0, 1.0])
        _AcLike().canonicalize(block)
        assert sorted(block.keys()) == ["charge", "x"]
        _AcLike().localize(block)
        assert sorted(block.keys()) == ["q", "x"]

    def test_canonicalize_frame_walks_every_block(self):
        frame = molrs.Frame()
        frame["atoms"] = {"q": np.array([0.5]), "x": np.array([0.0])}
        assert _AcLike().canonicalize_frame(frame) is frame
        assert "charge" in frame["atoms"] and "q" not in frame["atoms"]

    def test_register_field_extends_the_mapping_at_runtime(self):
        class _Fmt(FieldFormatter):
            _field_formatters = {}

        _Fmt.register_field("qq", CHARGE)
        block = molrs.Block()
        block["qq"] = np.array([1.0])
        _Fmt().canonicalize(block)
        assert "charge" in block
