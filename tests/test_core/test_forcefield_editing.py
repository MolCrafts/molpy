"""Tests for the Phase-1 editing primitives on ForceField."""

from __future__ import annotations

from molpy.core.forcefield import AtomStyle, AtomType, ForceField


def _simple_ff() -> ForceField:
    ff = ForceField(name="test", units="real")
    astyle = ff.def_style(AtomStyle("full"))
    astyle.def_type(name="CT", mass=12.011, charge=-0.18, element="C")
    astyle.def_type(name="HC", mass=1.008, charge=0.06, element="H")
    return ff


class TestRenameType:
    def test_rename_atomtype(self):
        ff = _simple_ff()
        n = ff.rename_type(AtomStyle, "CT", "opls_135")
        assert n == 1
        names = {t.name for t in ff.get_types(AtomType)}
        assert "CT" not in names
        assert "opls_135" in names

    def test_rename_missing_returns_zero(self):
        ff = _simple_ff()
        assert ff.rename_type(AtomStyle, "nonexistent", "foo") == 0


class TestRemoveType:
    def test_remove_atomtype(self):
        ff = _simple_ff()
        n = ff.remove_type(AtomStyle, "HC")
        assert n == 1
        names = {t.name for t in ff.get_types(AtomType)}
        assert names == {"CT"}


class TestRemoveStyle:
    def test_remove_style_by_name(self):
        ff = _simple_ff()
        assert ff.remove_style(AtomStyle, "full") is True
        assert ff.get_style_by_name("full", AtomStyle) is None

    def test_remove_style_missing_is_false(self):
        ff = _simple_ff()
        assert ff.remove_style(AtomStyle, "minimal") is False
