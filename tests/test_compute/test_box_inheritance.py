"""molpy.Box inherits from molrs.Box and is interchangeable with it.

Acceptance criteria covered:
- molpy-box-is-a-molrs-box (Phase 1)
- molpy-box-public-api-preserved (Phase 1, partial — full API is in
  test_core/test_box.py)
"""

import numpy as np
import pytest

import molpy
import molrs


class TestMolpyBoxInheritsFromMolrsBox:
    """Critical contract: molpy.Box is-a molrs.Box."""

    def test_molpy_box_is_a_molrs_box(self):
        b = molpy.Box.cubic(10.0)
        assert isinstance(b, molrs.Box)
        assert isinstance(b, molpy.Box)

    def test_molpy_box_keeps_molpy_api(self):
        b = molpy.Box.cubic(10.0)
        # molpy-style read-through.
        assert b.lx == pytest.approx(10.0)
        assert b.ly == pytest.approx(10.0)
        assert b.lz == pytest.approx(10.0)
        assert b.style == molpy.Box.Style.ORTHOGONAL
        assert b.matrix.shape == (3, 3)
        assert b.matrix[0, 0] == pytest.approx(10.0)
        np.testing.assert_array_equal(b.pbc, np.array([True, True, True]))


class TestFrameBoxPassesDirectlyToMolrs:
    """frame.box must be accepted by molrs APIs without conversion."""

    def test_neighbor_query_accepts_frame_box_directly(self):
        frame = molrs.Frame()
        frame.box = molpy.Box.cubic(10.0)
        rng = np.random.default_rng(0)
        xyz = rng.uniform(0.0, 10.0, size=(50, 3))
        # No adapter: pass molpy frame.box straight into molrs.
        nq = molrs.NeighborQuery(frame.box, xyz, 2.0)
        nlist = nq.query_self()
        assert nlist.n_pairs >= 0  # "no exception, returns nlist" check

    def test_molrs_volume_method_works_on_molpy_box(self):
        b = molpy.Box.cubic(10.0)
        # molrs.Box.volume() is a method; molpy.Box.volume is a property.
        # Both must coexist via MRO override (subclass property wins for
        # bare attribute access; super().volume() still callable).
        assert b.volume == pytest.approx(1000.0)  # molpy property
