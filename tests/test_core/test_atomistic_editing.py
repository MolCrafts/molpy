"""Tests for the Phase-1 editing primitives on Atomistic."""

from __future__ import annotations

import pytest

from molpy import Angle, Atom, Atomistic, Bond, Dihedral


def _ethane() -> Atomistic:
    mol = Atomistic()
    c1 = mol.def_atom(symbol="C", type="CT", xyz=[0.0, 0.0, 0.0])
    c2 = mol.def_atom(symbol="C", type="CT", xyz=[1.54, 0.0, 0.0])
    h = [
        mol.def_atom(symbol="H", type="HC", xyz=[0.0, float(i), 0.0]) for i in range(6)
    ]
    mol.def_bond(c1, c2, type="CT-CT")
    for hh in h[:3]:
        mol.def_bond(c1, hh, type="CT-HC")
    for hh in h[3:]:
        mol.def_bond(c2, hh, type="CT-HC")
    mol.def_angle(h[0], c1, c2, type="HC-CT-CT")
    mol.def_dihedral(h[0], c1, c2, h[3], type="HC-CT-CT-HC")
    return mol


class TestDeleteLinks:
    def test_del_angle_removes_only_that_angle(self):
        mol = _ethane()
        angle = next(iter(mol.angles))
        mol.del_angle(angle)
        assert len(mol.angles) == 0
        assert len(mol.bonds) == 7
        assert len(mol.dihedrals) == 1

    def test_del_dihedral_does_not_touch_bonds(self):
        mol = _ethane()
        dih = next(iter(mol.dihedrals))
        mol.del_dihedral(dih)
        assert len(mol.dihedrals) == 0
        assert len(mol.bonds) == 7


class TestRenameType:
    def test_rename_atom_type(self):
        mol = _ethane()
        n = mol.rename_type("CT", "opls_135")
        assert n == 2
        assert all(a.get("type") != "CT" for a in mol.atoms if a.get("symbol") == "C")
        assert sum(1 for a in mol.atoms if a.get("type") == "opls_135") == 2

    def test_rename_bond_type(self):
        mol = _ethane()
        n = mol.rename_type("CT-HC", "opls_135-opls_140", kind=Bond)
        assert n == 6
        assert all(b.get("type") != "CT-HC" for b in mol.bonds)


class TestSetProperty:
    def test_set_property_callable(self):
        mol = _ethane()
        n = mol.set_property(lambda a: a.get("symbol") == "H", "charge", 0.06)
        assert n == 6
        assert all(a.get("charge") == 0.06 for a in mol.atoms if a.get("symbol") == "H")
        assert all("charge" not in a.data for a in mol.atoms if a.get("symbol") == "C")

    def test_set_property_rejects_string(self):
        mol = _ethane()
        with pytest.raises(TypeError):
            mol.set_property("[#1]", "charge", 0.06)


class TestSelect:
    def test_select_returns_new_atomistic(self):
        mol = _ethane()
        sub = mol.select(lambda a: a.get("symbol") == "C")
        assert isinstance(sub, Atomistic)
        assert sub is not mol
        assert len(sub.atoms) == 2
        # Bond between the two C atoms is preserved (both endpoints selected)
        assert len(sub.bonds) == 1

    def test_select_drops_cross_bonds(self):
        mol = _ethane()
        sub = mol.select(lambda a: a.get("symbol") == "H")
        assert len(sub.atoms) == 6
        # No bonds survive (every bond had a C endpoint that's excluded)
        assert len(sub.bonds) == 0

    def test_select_rejects_string(self):
        mol = _ethane()
        with pytest.raises(TypeError):
            mol.select("[#6]")
