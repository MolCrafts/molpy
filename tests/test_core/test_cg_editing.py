"""Tests for the Phase-1 editing primitives on CoarseGrain."""

from __future__ import annotations

import pytest

from molpy.core.cg import Bead, CGBond, CoarseGrain


def _three_beads() -> CoarseGrain:
    cg = CoarseGrain()
    b1 = cg.def_bead(type="A", x=0.0, y=0.0, z=0.0)
    b2 = cg.def_bead(type="B", x=1.0, y=0.0, z=0.0)
    b3 = cg.def_bead(type="A", x=2.0, y=0.0, z=0.0)
    cg.def_cgbond(b1, b2, type="A-B")
    cg.def_cgbond(b2, b3, type="A-B")
    return cg


class TestCGEditing:
    def test_del_bead_cascades(self):
        cg = _three_beads()
        b = next(iter(cg.beads))
        cg.del_bead(b)
        assert len(cg.beads) == 2
        # Incident bond removed by remove_entity
        assert len(cg.cgbonds) == 1

    def test_del_cgbond(self):
        cg = _three_beads()
        bond = next(iter(cg.cgbonds))
        cg.del_cgbond(bond)
        assert len(cg.cgbonds) == 1
        assert len(cg.beads) == 3

    def test_rename_bead_type(self):
        cg = _three_beads()
        n = cg.rename_type("A", "A_new")
        assert n == 2
        assert sum(1 for b in cg.beads if b.get("type") == "A_new") == 2

    def test_rename_cgbond_type(self):
        cg = _three_beads()
        n = cg.rename_type("A-B", "link", kind=CGBond)
        assert n == 2

    def test_set_property_callable(self):
        cg = _three_beads()
        n = cg.set_property(lambda b: b.get("type") == "B", "mass", 72.0)
        assert n == 1
        b_mass = [b.get("mass") for b in cg.beads if b.get("type") == "B"]
        assert b_mass == [72.0]

    def test_select_subset(self):
        cg = _three_beads()
        sub = cg.select(lambda b: b.get("type") == "A")
        assert isinstance(sub, CoarseGrain)
        assert sub is not cg
        assert len(sub.beads) == 2
        # No bonds: every bond in the parent crossed A <-> B
        assert len(sub.cgbonds) == 0

    def test_select_rejects_string(self):
        cg = _three_beads()
        with pytest.raises(TypeError):
            cg.select("[A]")
