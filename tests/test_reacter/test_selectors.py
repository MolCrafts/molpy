#!/usr/bin/env python3
"""Unit tests for Reacter selector functions.

Tests cover:
- port_anchor_selector
- remove_one_H
- remove_all_H
- remove_dummy_atoms (from selectors)
- remove_OH
- remove_water
- no_leaving_group
"""

import pytest

from molpy import Atom, Atomistic, Bond
from molpy.core.wrappers.monomer import Monomer
from molpy.reacter.selectors import (
    no_leaving_group,
    port_anchor_selector,
    remove_all_H,
    remove_dummy_atoms,
    remove_OH,
    remove_one_H,
    remove_water,
)


class TestPortAnchorSelector:
    """Test port_anchor_selector function."""

    def test_port_anchor_selector_basic(self):
        """Test selecting anchor from port."""
        asm = Atomistic()
        c = Atom(symbol="C")
        asm.add_entity(c)

        mono = Monomer(asm)
        mono.set_port("1", c)

        anchor = port_anchor_selector(mono, "1")

        assert anchor is c

    def test_port_anchor_selector_different_port(self):
        """Test selecting anchor from different port."""
        asm = Atomistic()
        c1 = Atom(symbol="C")
        c2 = Atom(symbol="C")
        asm.add_entity(c1, c2)

        mono = Monomer(asm)
        mono.set_port("head", c1)
        mono.set_port("tail", c2)

        anchor = port_anchor_selector(mono, "tail")

        assert anchor is c2

    def test_port_anchor_selector_missing_port(self):
        """Test selecting anchor from non-existent port raises error."""
        asm = Atomistic()
        c = Atom(symbol="C")
        asm.add_entity(c)

        mono = Monomer(asm)
        # Don't set port

        with pytest.raises(ValueError, match="Port '1' not found"):
            port_anchor_selector(mono, "1")


class TestRemoveOneH:
    """Test remove_one_H function."""

    def test_remove_one_H_single(self):
        """Test removing one H when one exists."""
        asm = Atomistic()
        c = Atom(symbol="C")
        h = Atom(symbol="H")
        asm.add_entity(c, h)
        asm.add_link(Bond(c, h))

        mono = Monomer(asm)

        leaving = remove_one_H(mono, c)

        assert len(leaving) == 1
        assert leaving[0] is h
        assert leaving[0].get("symbol") == "H"

    def test_remove_one_H_multiple(self):
        """Test removing one H when multiple exist."""
        asm = Atomistic()
        c = Atom(symbol="C")
        h1 = Atom(symbol="H")
        h2 = Atom(symbol="H")
        h3 = Atom(symbol="H")
        asm.add_entity(c, h1, h2, h3)
        asm.add_link(Bond(c, h1), Bond(c, h2), Bond(c, h3))

        mono = Monomer(asm)

        leaving = remove_one_H(mono, c)

        # Should return only one H
        assert len(leaving) == 1
        assert leaving[0].get("symbol") == "H"
        assert leaving[0] in [h1, h2, h3]

    def test_remove_one_H_none(self):
        """Test removing one H when none exist."""
        asm = Atomistic()
        c = Atom(symbol="C")
        o = Atom(symbol="O")
        asm.add_entity(c, o)
        asm.add_link(Bond(c, o))

        mono = Monomer(asm)

        leaving = remove_one_H(mono, c)

        assert len(leaving) == 0

    def test_remove_one_H_no_neighbors(self):
        """Test removing one H from isolated atom."""
        asm = Atomistic()
        c = Atom(symbol="C")
        asm.add_entity(c)

        mono = Monomer(asm)

        leaving = remove_one_H(mono, c)

        assert len(leaving) == 0


class TestRemoveAllH:
    """Test remove_all_H function."""

    def test_remove_all_H_single(self):
        """Test removing all H when one exists."""
        asm = Atomistic()
        c = Atom(symbol="C")
        h = Atom(symbol="H")
        asm.add_entity(c, h)
        asm.add_link(Bond(c, h))

        mono = Monomer(asm)

        leaving = remove_all_H(mono, c)

        assert len(leaving) == 1
        assert leaving[0] is h

    def test_remove_all_H_multiple(self):
        """Test removing all H when multiple exist."""
        asm = Atomistic()
        c = Atom(symbol="C")
        h1 = Atom(symbol="H")
        h2 = Atom(symbol="H")
        h3 = Atom(symbol="H")
        asm.add_entity(c, h1, h2, h3)
        asm.add_link(Bond(c, h1), Bond(c, h2), Bond(c, h3))

        mono = Monomer(asm)

        leaving = remove_all_H(mono, c)

        assert len(leaving) == 3
        assert all(h.get("symbol") == "H" for h in leaving)
        assert h1 in leaving
        assert h2 in leaving
        assert h3 in leaving

    def test_remove_all_H_none(self):
        """Test removing all H when none exist."""
        asm = Atomistic()
        c = Atom(symbol="C")
        o = Atom(symbol="O")
        asm.add_entity(c, o)
        asm.add_link(Bond(c, o))

        mono = Monomer(asm)

        leaving = remove_all_H(mono, c)

        assert len(leaving) == 0

    def test_remove_all_H_mixed_neighbors(self):
        """Test removing all H when other neighbors exist."""
        asm = Atomistic()
        c = Atom(symbol="C")
        h1 = Atom(symbol="H")
        h2 = Atom(symbol="H")
        o = Atom(symbol="O")
        asm.add_entity(c, h1, h2, o)
        asm.add_link(Bond(c, h1), Bond(c, h2), Bond(c, o))

        mono = Monomer(asm)

        leaving = remove_all_H(mono, c)

        # Should only return H atoms
        assert len(leaving) == 2
        assert all(h.get("symbol") == "H" for h in leaving)
        assert o not in leaving


class TestRemoveDummyAtoms:
    """Test remove_dummy_atoms function (from selectors)."""

    def test_remove_dummy_atoms_single(self):
        """Test removing dummy atoms when one exists."""
        asm = Atomistic()
        c = Atom(symbol="C")
        dummy = Atom(symbol="*")
        asm.add_entity(c, dummy)
        asm.add_link(Bond(c, dummy))

        mono = Monomer(asm)

        leaving = remove_dummy_atoms(mono, c)

        assert len(leaving) == 1
        assert leaving[0] is dummy
        assert leaving[0].get("symbol") == "*"

    def test_remove_dummy_atoms_multiple(self):
        """Test removing dummy atoms when multiple exist."""
        asm = Atomistic()
        c = Atom(symbol="C")
        dummy1 = Atom(symbol="*")
        dummy2 = Atom(symbol="*")
        asm.add_entity(c, dummy1, dummy2)
        asm.add_link(Bond(c, dummy1), Bond(c, dummy2))

        mono = Monomer(asm)

        leaving = remove_dummy_atoms(mono, c)

        assert len(leaving) == 2
        assert all(d.get("symbol") == "*" for d in leaving)
        assert dummy1 in leaving
        assert dummy2 in leaving

    def test_remove_dummy_atoms_none(self):
        """Test removing dummy atoms when none exist."""
        asm = Atomistic()
        c = Atom(symbol="C")
        h = Atom(symbol="H")
        asm.add_entity(c, h)
        asm.add_link(Bond(c, h))

        mono = Monomer(asm)

        leaving = remove_dummy_atoms(mono, c)

        assert len(leaving) == 0

    def test_remove_dummy_atoms_mixed_neighbors(self):
        """Test removing dummy atoms when other neighbors exist."""
        asm = Atomistic()
        c = Atom(symbol="C")
        dummy = Atom(symbol="*")
        h = Atom(symbol="H")
        asm.add_entity(c, dummy, h)
        asm.add_link(Bond(c, dummy), Bond(c, h))

        mono = Monomer(asm)

        leaving = remove_dummy_atoms(mono, c)

        # Should only return dummy atoms
        assert len(leaving) == 1
        assert leaving[0] is dummy
        assert h not in leaving


class TestRemoveOH:
    """Test remove_OH function."""

    def test_remove_OH_complete(self):
        """Test removing complete OH group."""
        asm = Atomistic()
        c = Atom(symbol="C")
        o = Atom(symbol="O")
        h = Atom(symbol="H")
        asm.add_entity(c, o, h)
        asm.add_link(Bond(c, o), Bond(o, h))

        mono = Monomer(asm)

        leaving = remove_OH(mono, c)

        assert len(leaving) == 2
        assert o in leaving
        assert h in leaving

    def test_remove_OH_no_H(self):
        """Test removing OH when O has no H (just O)."""
        asm = Atomistic()
        c = Atom(symbol="C")
        o = Atom(symbol="O")
        asm.add_entity(c, o)
        asm.add_link(Bond(c, o))
        # O has no H

        mono = Monomer(asm)

        leaving = remove_OH(mono, c)

        # Should return just O
        assert len(leaving) == 1
        assert leaving[0] is o

    def test_remove_OH_no_O(self):
        """Test removing OH when no O neighbor exists."""
        asm = Atomistic()
        c = Atom(symbol="C")
        h = Atom(symbol="H")
        asm.add_entity(c, h)
        asm.add_link(Bond(c, h))

        mono = Monomer(asm)

        leaving = remove_OH(mono, c)

        assert len(leaving) == 0

    def test_remove_OH_multiple_O(self):
        """Test removing OH when multiple O neighbors exist."""
        asm = Atomistic()
        c = Atom(symbol="C")
        o1 = Atom(symbol="O")
        o2 = Atom(symbol="O")
        h = Atom(symbol="H")
        asm.add_entity(c, o1, o2, h)
        asm.add_link(Bond(c, o1), Bond(c, o2), Bond(o1, h))
        # Only o1 has H

        mono = Monomer(asm)

        leaving = remove_OH(mono, c)

        # Should return first O neighbor (o1) and its H
        # Note: remove_OH finds first O neighbor, then its H
        assert len(leaving) == 2
        assert o1 in leaving  # First O found
        assert h in leaving
        # o2 might be in leaving if it's found first, or not if o1 is found first
        # The behavior depends on iteration order, so we just check that we got O and H
        assert any(atom.get("symbol") == "O" for atom in leaving)
        assert any(atom.get("symbol") == "H" for atom in leaving)


class TestRemoveWater:
    """Test remove_water function."""

    def test_remove_water_same_as_remove_OH(self):
        """Test that remove_water is same as remove_OH."""
        asm = Atomistic()
        c = Atom(symbol="C")
        o = Atom(symbol="O")
        h = Atom(symbol="H")
        asm.add_entity(c, o, h)
        asm.add_link(Bond(c, o), Bond(o, h))

        mono = Monomer(asm)

        leaving = remove_water(mono, c)

        # Should behave same as remove_OH
        assert len(leaving) == 2
        assert o in leaving
        assert h in leaving


class TestNoLeavingGroup:
    """Test no_leaving_group function."""

    def test_no_leaving_group_always_empty(self):
        """Test that no_leaving_group always returns empty list."""
        asm = Atomistic()
        c = Atom(symbol="C")
        h = Atom(symbol="H")
        asm.add_entity(c, h)
        asm.add_link(Bond(c, h))

        mono = Monomer(asm)

        leaving = no_leaving_group(mono, c)

        assert len(leaving) == 0

    def test_no_leaving_group_ignores_anchor(self):
        """Test that no_leaving_group ignores anchor parameter."""
        asm = Atomistic()
        c = Atom(symbol="C")
        asm.add_entity(c)

        mono = Monomer(asm)

        # Should work even with None or different anchor
        leaving = no_leaving_group(mono, c)

        assert len(leaving) == 0
