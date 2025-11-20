#!/usr/bin/env python3
"""Unit tests for Reacter selector functions.

Tests cover:
- select_port_atom
- select_one_hydrogen
- select_all_hydrogens
- select_dummy_atoms (from selectors)
- select_hydroxyl_group
- select_none
"""

import pytest

from molpy import Atom, Atomistic, Bond
from molpy.core.wrappers.monomer import Monomer
from molpy.reacter.selectors import (
    select_none,
    select_port_atom,
    select_all_hydrogens,
    select_dummy_atoms,
    select_hydroxyl_group,
    select_one_hydrogen,
)


class TestPortAnchorSelector:
    """Test select_port_atom function."""

    def test_port_anchor_selector_basic(self):
        """Test selecting anchor from port."""
        mono = Monomer()
        c = Atom(symbol="C")
        mono.add_entity(c)

        mono.set_port("1", c)

        anchor = select_port_atom(mono, "1")

        assert anchor is c

    def test_port_anchor_selector_different_port(self):
        """Test selecting anchor from different port."""
        mono = Monomer()
        c1 = Atom(symbol="C")
        c2 = Atom(symbol="C")
        mono.add_entity(c1, c2)

        mono.set_port("head", c1)
        mono.set_port("tail", c2)

        anchor = select_port_atom(mono, "tail")

        assert anchor is c2

    def test_port_anchor_selector_missing_port(self):
        """Test selecting anchor from non-existent port raises error."""
        mono = Monomer()
        c = Atom(symbol="C")
        mono.add_entity(c)

        # Don't set port

        with pytest.raises(ValueError, match="Port '1' not found"):
            select_port_atom(mono, "1")


class TestRemoveOneH:
    """Test select_one_hydrogen function."""

    def test_remove_one_H_single(self):
        """Test removing one H when one exists."""
        mono = Monomer()
        c = Atom(symbol="C")
        h = Atom(symbol="H")
        mono.add_entity(c, h)
        mono.add_link(Bond(c, h))

        leaving = select_one_hydrogen(mono, c)

        assert len(leaving) == 1
        assert leaving[0] is h
        assert leaving[0].get("symbol") == "H"

    def test_remove_one_H_multiple(self):
        """Test removing one H when multiple exist."""
        mono = Monomer()
        c = Atom(symbol="C")
        h1 = Atom(symbol="H")
        h2 = Atom(symbol="H")
        h3 = Atom(symbol="H")
        mono.add_entity(c, h1, h2, h3)
        mono.add_link(Bond(c, h1), Bond(c, h2), Bond(c, h3))

        leaving = select_one_hydrogen(mono, c)

        # Should return only one H
        assert len(leaving) == 1
        assert leaving[0].get("symbol") == "H"
        assert leaving[0] in [h1, h2, h3]

    def test_remove_one_H_none(self):
        """Test removing one H when none exist."""
        mono = Monomer()
        c = Atom(symbol="C")
        o = Atom(symbol="O")
        mono.add_entity(c, o)
        mono.add_link(Bond(c, o))

        leaving = select_one_hydrogen(mono, c)

        assert len(leaving) == 0

    def test_remove_one_H_no_neighbors(self):
        """Test removing one H from isolated atom."""
        mono = Monomer()
        c = Atom(symbol="C")
        mono.add_entity(c)

        leaving = select_one_hydrogen(mono, c)

        assert len(leaving) == 0


class TestRemoveAllH:
    """Test select_all_hydrogens function."""

    def test_remove_all_H_single(self):
        """Test removing all H when one exists."""
        mono = Monomer()
        c = Atom(symbol="C")
        h = Atom(symbol="H")
        mono.add_entity(c, h)
        mono.add_link(Bond(c, h))

        leaving = select_all_hydrogens(mono, c)

        assert len(leaving) == 1
        assert leaving[0] is h

    def test_remove_all_H_multiple(self):
        """Test removing all H when multiple exist."""
        mono = Monomer()
        c = Atom(symbol="C")
        h1 = Atom(symbol="H")
        h2 = Atom(symbol="H")
        h3 = Atom(symbol="H")
        mono.add_entity(c, h1, h2, h3)
        mono.add_link(Bond(c, h1), Bond(c, h2), Bond(c, h3))

        leaving = select_all_hydrogens(mono, c)

        assert len(leaving) == 3
        assert all(h.get("symbol") == "H" for h in leaving)
        assert h1 in leaving
        assert h2 in leaving
        assert h3 in leaving

    def test_remove_all_H_none(self):
        """Test removing all H when none exist."""
        mono = Monomer()
        c = Atom(symbol="C")
        o = Atom(symbol="O")
        mono.add_entity(c, o)
        mono.add_link(Bond(c, o))

        leaving = select_all_hydrogens(mono, c)

        assert len(leaving) == 0

    def test_remove_all_H_mixed_neighbors(self):
        """Test removing all H when other neighbors exist."""
        mono = Monomer()
        c = Atom(symbol="C")
        h1 = Atom(symbol="H")
        h2 = Atom(symbol="H")
        o = Atom(symbol="O")
        mono.add_entity(c, h1, h2, o)
        mono.add_link(Bond(c, h1), Bond(c, h2), Bond(c, o))

        leaving = select_all_hydrogens(mono, c)

        # Should only return H atoms
        assert len(leaving) == 2
        assert all(h.get("symbol") == "H" for h in leaving)
        assert o not in leaving


class TestRemoveDummyAtoms:
    """Test select_dummy_atoms function (from selectors)."""

    def test_remove_dummy_atoms_single(self):
        """Test removing dummy atoms when one exists."""
        mono = Monomer()
        c = Atom(symbol="C")
        dummy = Atom(symbol="*")
        mono.add_entity(c, dummy)
        mono.add_link(Bond(c, dummy))

        leaving = select_dummy_atoms(mono, c)

        assert len(leaving) == 1
        assert leaving[0] is dummy
        assert leaving[0].get("symbol") == "*"

    def test_remove_dummy_atoms_multiple(self):
        """Test removing dummy atoms when multiple exist."""
        mono = Monomer()
        c = Atom(symbol="C")
        dummy1 = Atom(symbol="*")
        dummy2 = Atom(symbol="*")
        mono.add_entity(c, dummy1, dummy2)
        mono.add_link(Bond(c, dummy1), Bond(c, dummy2))

        leaving = select_dummy_atoms(mono, c)

        assert len(leaving) == 2
        assert all(d.get("symbol") == "*" for d in leaving)
        assert dummy1 in leaving
        assert dummy2 in leaving

    def test_remove_dummy_atoms_none(self):
        """Test removing dummy atoms when none exist."""
        mono = Monomer()
        c = Atom(symbol="C")
        h = Atom(symbol="H")
        mono.add_entity(c, h)
        mono.add_link(Bond(c, h))

        leaving = select_dummy_atoms(mono, c)

        assert len(leaving) == 0

    def test_remove_dummy_atoms_mixed_neighbors(self):
        """Test removing dummy atoms when other neighbors exist."""
        mono = Monomer()
        c = Atom(symbol="C")
        dummy = Atom(symbol="*")
        h = Atom(symbol="H")
        mono.add_entity(c, dummy, h)
        mono.add_link(Bond(c, dummy), Bond(c, h))

        leaving = select_dummy_atoms(mono, c)

        # Should only return dummy atoms
        assert len(leaving) == 1
        assert leaving[0] is dummy
        assert h not in leaving


class TestRemoveOH:
    """Test select_hydroxyl_group function."""

    def test_remove_OH_complete(self):
        """Test removing complete OH group."""
        mono = Monomer()
        c = Atom(symbol="C")
        o = Atom(symbol="O")
        h = Atom(symbol="H")
        mono.add_entity(c, o, h)
        mono.add_link(Bond(c, o, order=1), Bond(o, h, order=1))

        leaving = select_hydroxyl_group(mono, c)

        assert len(leaving) == 2
        assert o in leaving
        assert h in leaving

    def test_remove_OH_no_H(self):
        """Test removing OH when O has no H (just O)."""
        mono = Monomer()
        c = Atom(symbol="C")
        o = Atom(symbol="O")
        mono.add_entity(c, o)
        mono.add_link(Bond(c, o, order=1))
        # O has no H

        leaving = select_hydroxyl_group(mono, c)

        # Should return just O
        assert len(leaving) == 1
        assert leaving[0] is o

    def test_remove_OH_no_O(self):
        """Test removing OH when no O neighbor exists."""
        mono = Monomer()
        c = Atom(symbol="C")
        h = Atom(symbol="H")
        mono.add_entity(c, h)
        mono.add_link(Bond(c, h))

        leaving = select_hydroxyl_group(mono, c)

        assert len(leaving) == 0

    def test_remove_OH_multiple_O(self):
        """Test removing OH when multiple O neighbors exist."""
        mono = Monomer()
        c = Atom(symbol="C")
        o1 = Atom(symbol="O")  # hydroxyl oxygen
        o2 = Atom(symbol="O")  # carbonyl oxygen
        h = Atom(symbol="H")
        mono.add_entity(c, o1, o2, h)
        mono.add_link(
            Bond(c, o1, order=1),  # single bond = hydroxyl
            Bond(c, o2, order=2),  # double bond = carbonyl
            Bond(o1, h, order=1),
        )
        # Only o1 has H and single bond

        leaving = select_hydroxyl_group(mono, c)

        # Should return o1 (single-bonded) and its H
        assert len(leaving) == 2
        assert o1 in leaving
        assert h in leaving


class TestNoLeavingGroup:
    """Test select_none function."""

    def test_no_leaving_group_always_empty(self):
        """Test that select_none always returns empty list."""
        mono = Monomer()
        c = Atom(symbol="C")
        h = Atom(symbol="H")
        mono.add_entity(c, h)
        mono.add_link(Bond(c, h))

        leaving = select_none(mono, c)

        assert len(leaving) == 0

    def test_no_leaving_group_ignores_anchor(self):
        """Test that select_none ignores anchor parameter."""
        mono = Monomer()
        c = Atom(symbol="C")
        mono.add_entity(c)

        # Should work even with None or different anchor
        leaving = select_none(mono, c)

        assert len(leaving) == 0
