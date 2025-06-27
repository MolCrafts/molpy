import numpy as np
import pytest
from molpy.core.frame import Frame, Block

@pytest.fixture
def simple_frame() -> Frame:
    f = Frame()
    f["atoms", "xyz"] = np.arange(9).reshape(3, 3)
    f["atoms", "charge"] = np.array([-1.0, 0.5, 0.5])
    f["bonds", "i"] = np.arange(3)
    return f

class TestBlock:
    def test_block_set_and_get(self):
        blk = Block()
        blk["foo"] = np.arange(5)
        assert np.array_equal(blk["foo"], np.arange(5))

    def test_block_len_and_iter(self):
        blk = Block({"a": np.arange(2), "b": np.ones(2)})
        assert set(blk) == {"a", "b"}
        assert len(blk) == 2

    def test_block_to_from_dict(self):
        blk = Block({"a": np.arange(2), "b": np.ones(2)})
        dct = blk.to_dict()
        restored = Block.from_dict(dct)
        for k in ("a", "b"):
            assert np.array_equal(blk[k], restored[k])

class TestFrame:
    def test_set_and_get_variable(self, simple_frame):
        assert np.isclose(simple_frame["atoms", "charge"][0], -1.0)
        assert np.array_equal(simple_frame["bonds", "i"], np.arange(3))

    def test_setitem_creates_block(self):
        f = Frame()
        f["foo", "bar"] = np.ones(4)
        assert "foo" in list(f.blocks())
        assert np.array_equal(f["foo", "bar"], np.ones(4))

    def test_get_block(self, simple_frame):
        blk = simple_frame["atoms"]
        assert isinstance(blk, Block)
        assert set(blk) == {"xyz", "charge"}

    def test_variables(self, simple_frame):
        assert set(simple_frame.variables("atoms")) == {"xyz", "charge"}
        assert set(simple_frame.variables("bonds")) == {"i"}

    def test_blocks_iter_and_len(self, simple_frame):
        blocks = set(simple_frame.blocks())
        assert blocks == {"atoms", "bonds"}
        assert len(list(simple_frame.blocks())) == 2

    def test_delete_variable(self, simple_frame):
        del simple_frame["atoms"]["charge"]
        assert "charge" not in simple_frame.variables("atoms")

    def test_delete_block(self, simple_frame):
        del simple_frame._blocks["bonds"]
        assert "bonds" not in set(simple_frame.blocks())

    def test_to_from_dict_roundtrip(self, simple_frame):
        dct = simple_frame.to_dict()
        restored = Frame.from_dict(dct)
        for g in restored.blocks():
            for v in restored.variables(g):
                assert np.array_equal(restored[g, v], simple_frame[g, v])

    def test_forbid_assigning_non_block(self):
        f = Frame()
        with pytest.raises(ValueError, match="Value must be a Block instance"):
            f["invalid"] = np.array([1, 2, 3])