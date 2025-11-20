#!/usr/bin/env python3
"""Unit tests for Reacter transformer functions.

Tests cover:
- form_single_bond
- form_double_bond
- form_triple_bond
- form_aromatic_bond
- create_bond_former
- skip_bond_formation
- break_bond
"""

from molpy import Atom, Atomistic, Bond
from molpy.reacter.transformers import (
    break_bond,
    form_aromatic_bond,
    create_bond_former,
    form_double_bond,
    form_single_bond,
    form_triple_bond,
    skip_bond_formation,
)
from molpy.reacter.utils import get_bond_between


class TestMakeSingleBond:
    """Test form_single_bond function."""

    def test_make_single_bond_new(self):
        """Test creating a new single bond."""
        asm = Atomistic()
        c1 = Atom(symbol="C")
        c2 = Atom(symbol="C")
        asm.add_entity(c1, c2)

        form_single_bond(asm, c1, c2)

        bond = get_bond_between(asm, c1, c2)
        assert bond is not None
        assert bond.get("order") == 1
        assert bond.get("kind") == "-"

    def test_make_single_bond_update_existing(self):
        """Test updating existing bond to single bond."""
        asm = Atomistic()
        c1 = Atom(symbol="C")
        c2 = Atom(symbol="C")
        bond = Bond(c1, c2, order=2)  # Start with double bond
        asm.add_entity(c1, c2)
        asm.add_link(bond)

        form_single_bond(asm, c1, c2)

        # Should update existing bond
        found = get_bond_between(asm, c1, c2)
        assert found is bond
        assert found.get("order") == 1
        assert found.get("kind") == "-"


class TestMakeDoubleBond:
    """Test form_double_bond function."""

    def test_make_double_bond_new(self):
        """Test creating a new double bond."""
        asm = Atomistic()
        c1 = Atom(symbol="C")
        c2 = Atom(symbol="C")
        asm.add_entity(c1, c2)

        form_double_bond(asm, c1, c2)

        bond = get_bond_between(asm, c1, c2)
        assert bond is not None
        assert bond.get("order") == 2
        assert bond.get("kind") == "="

    def test_make_double_bond_update_existing(self):
        """Test updating existing bond to double bond."""
        asm = Atomistic()
        c1 = Atom(symbol="C")
        c2 = Atom(symbol="C")
        bond = Bond(c1, c2, order=1)  # Start with single bond
        asm.add_entity(c1, c2)
        asm.add_link(bond)

        form_double_bond(asm, c1, c2)

        found = get_bond_between(asm, c1, c2)
        assert found is bond
        assert found.get("order") == 2
        assert found.get("kind") == "="


class TestMakeTripleBond:
    """Test form_triple_bond function."""

    def test_make_triple_bond_new(self):
        """Test creating a new triple bond."""
        asm = Atomistic()
        c1 = Atom(symbol="C")
        c2 = Atom(symbol="C")
        asm.add_entity(c1, c2)

        form_triple_bond(asm, c1, c2)

        bond = get_bond_between(asm, c1, c2)
        assert bond is not None
        assert bond.get("order") == 3
        assert bond.get("kind") == "#"

    def test_make_triple_bond_update_existing(self):
        """Test updating existing bond to triple bond."""
        asm = Atomistic()
        c1 = Atom(symbol="C")
        c2 = Atom(symbol="C")
        bond = Bond(c1, c2, order=1)
        asm.add_entity(c1, c2)
        asm.add_link(bond)

        form_triple_bond(asm, c1, c2)

        found = get_bond_between(asm, c1, c2)
        assert found is bond
        assert found.get("order") == 3
        assert found.get("kind") == "#"


class TestMakeAromaticBond:
    """Test form_aromatic_bond function."""

    def test_make_aromatic_bond_new(self):
        """Test creating a new aromatic bond."""
        asm = Atomistic()
        c1 = Atom(symbol="C")
        c2 = Atom(symbol="C")
        asm.add_entity(c1, c2)

        form_aromatic_bond(asm, c1, c2)

        bond = get_bond_between(asm, c1, c2)
        assert bond is not None
        assert bond.get("order") == 1.5
        assert bond.get("kind") == ":"
        assert bond.get("aromatic") is True

    def test_make_aromatic_bond_update_existing(self):
        """Test updating existing bond to aromatic bond."""
        asm = Atomistic()
        c1 = Atom(symbol="C")
        c2 = Atom(symbol="C")
        bond = Bond(c1, c2, order=1)
        asm.add_entity(c1, c2)
        asm.add_link(bond)

        form_aromatic_bond(asm, c1, c2)

        found = get_bond_between(asm, c1, c2)
        assert found is bond
        assert found.get("order") == 1.5
        assert found.get("kind") == ":"
        assert found.get("aromatic") is True


class TestMakeBondByOrder:
    """Test create_bond_former factory function."""

    def test_make_bond_by_order_single(self):
        """Test factory creating single bond maker."""
        bond_former = create_bond_former(1)

        asm = Atomistic()
        c1 = Atom(symbol="C")
        c2 = Atom(symbol="C")
        asm.add_entity(c1, c2)

        bond_former(asm, c1, c2)

        bond = get_bond_between(asm, c1, c2)
        assert bond is not None
        assert bond.get("order") == 1
        assert bond.get("kind") == "-"

    def test_make_bond_by_order_double(self):
        """Test factory creating double bond maker."""
        bond_former = create_bond_former(2)

        asm = Atomistic()
        c1 = Atom(symbol="C")
        c2 = Atom(symbol="C")
        asm.add_entity(c1, c2)

        bond_former(asm, c1, c2)

        bond = get_bond_between(asm, c1, c2)
        assert bond.get("order") == 2
        assert bond.get("kind") == "="

    def test_make_bond_by_order_triple(self):
        """Test factory creating triple bond maker."""
        bond_former = create_bond_former(3)

        asm = Atomistic()
        c1 = Atom(symbol="C")
        c2 = Atom(symbol="C")
        asm.add_entity(c1, c2)

        bond_former(asm, c1, c2)

        bond = get_bond_between(asm, c1, c2)
        assert bond.get("order") == 3
        assert bond.get("kind") == "#"

    def test_make_bond_by_order_aromatic(self):
        """Test factory creating aromatic bond maker."""
        bond_former = create_bond_former(1.5)

        asm = Atomistic()
        c1 = Atom(symbol="C")
        c2 = Atom(symbol="C")
        asm.add_entity(c1, c2)

        bond_former(asm, c1, c2)

        bond = get_bond_between(asm, c1, c2)
        assert bond.get("order") == 1.5
        assert bond.get("kind") == ":"
        assert bond.get("aromatic") is True

    def test_make_bond_by_order_update_existing(self):
        """Test factory bond maker updates existing bond."""
        bond_former = create_bond_former(2)

        asm = Atomistic()
        c1 = Atom(symbol="C")
        c2 = Atom(symbol="C")
        bond = Bond(c1, c2, order=1)
        asm.add_entity(c1, c2)
        asm.add_link(bond)

        bond_former(asm, c1, c2)

        found = get_bond_between(asm, c1, c2)
        assert found is bond
        assert found.get("order") == 2


class TestNoNewBond:
    """Test skip_bond_formation function."""

    def test_no_new_bond_does_nothing(self):
        """Test that skip_bond_formation doesn't create any bond."""
        asm = Atomistic()
        c1 = Atom(symbol="C")
        c2 = Atom(symbol="C")
        asm.add_entity(c1, c2)

        result = skip_bond_formation(asm, c1, c2)

        # Should return None
        assert result is None

        # No bond should be created
        bond = get_bond_between(asm, c1, c2)
        assert bond is None

    def test_no_new_bond_with_existing_bond(self):
        """Test that skip_bond_formation doesn't modify existing bond."""
        asm = Atomistic()
        c1 = Atom(symbol="C")
        c2 = Atom(symbol="C")
        bond = Bond(c1, c2, order=1)
        asm.add_entity(c1, c2)
        asm.add_link(bond)

        skip_bond_formation(asm, c1, c2)

        # Bond should remain unchanged
        found = get_bond_between(asm, c1, c2)
        assert found is bond
        assert found.get("order") == 1


class TestBreakBond:
    """Test break_bond function."""

    def test_break_bond_exists(self):
        """Test breaking an existing bond."""
        asm = Atomistic()
        c1 = Atom(symbol="C")
        c2 = Atom(symbol="C")
        bond = Bond(c1, c2, order=1)
        asm.add_entity(c1, c2)
        asm.add_link(bond)

        break_bond(asm, c1, c2)

        # Bond should be removed
        found = get_bond_between(asm, c1, c2)
        assert found is None

        # Atoms should still exist
        assert c1 in asm.atoms
        assert c2 in asm.atoms

    def test_break_bond_not_exists(self):
        """Test breaking a bond that doesn't exist (should do nothing)."""
        asm = Atomistic()
        c1 = Atom(symbol="C")
        c2 = Atom(symbol="C")
        asm.add_entity(c1, c2)
        # No bond between them

        # Should not raise error
        break_bond(asm, c1, c2)

        # Still no bond
        found = get_bond_between(asm, c1, c2)
        assert found is None

    def test_break_bond_reverse_order(self):
        """Test breaking bond works in reverse order."""
        asm = Atomistic()
        c1 = Atom(symbol="C")
        c2 = Atom(symbol="C")
        bond = Bond(c1, c2, order=1)
        asm.add_entity(c1, c2)
        asm.add_link(bond)

        # Try reverse order
        break_bond(asm, c2, c1)

        found = get_bond_between(asm, c1, c2)
        assert found is None
