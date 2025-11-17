#!/usr/bin/env python3
"""Unit tests for Reacter utility functions.

Tests cover:
- find_neighbors
- get_bond_between
- count_bonds
- remove_dummy_atoms
"""

from molpy import Atom, Atomistic, Bond
from molpy.reacter.utils import (
    count_bonds,
    find_neighbors,
    get_bond_between,
    remove_dummy_atoms,
)


class TestFindNeighbors:
    """Test find_neighbors function."""

    def test_find_all_neighbors(self):
        """Test finding all neighbors of an atom."""
        asm = Atomistic()
        c = Atom(symbol="C")
        h1 = Atom(symbol="H")
        h2 = Atom(symbol="H")
        o = Atom(symbol="O")

        asm.add_entity(c, h1, h2, o)
        asm.add_link(Bond(c, h1), Bond(c, h2), Bond(c, o))

        neighbors = find_neighbors(asm, c)

        assert len(neighbors) == 3
        assert h1 in neighbors
        assert h2 in neighbors
        assert o in neighbors

    def test_find_neighbors_with_element_filter(self):
        """Test finding neighbors filtered by element."""
        asm = Atomistic()
        c = Atom(symbol="C")
        h1 = Atom(symbol="H")
        h2 = Atom(symbol="H")
        o = Atom(symbol="O")

        asm.add_entity(c, h1, h2, o)
        asm.add_link(Bond(c, h1), Bond(c, h2), Bond(c, o))

        # Find only H neighbors
        h_neighbors = find_neighbors(asm, c, element="H")

        assert len(h_neighbors) == 2
        assert all(n.get("symbol") == "H" for n in h_neighbors)
        assert h1 in h_neighbors
        assert h2 in h_neighbors
        assert o not in h_neighbors

    def test_find_neighbors_with_element_filter_no_match(self):
        """Test finding neighbors with element filter that matches nothing."""
        asm = Atomistic()
        c = Atom(symbol="C")
        h1 = Atom(symbol="H")
        h2 = Atom(symbol="H")

        asm.add_entity(c, h1, h2)
        asm.add_link(Bond(c, h1), Bond(c, h2))

        # Find O neighbors (none exist)
        o_neighbors = find_neighbors(asm, c, element="O")

        assert len(o_neighbors) == 0

    def test_find_neighbors_no_neighbors(self):
        """Test finding neighbors of isolated atom."""
        asm = Atomistic()
        c = Atom(symbol="C")
        asm.add_entity(c)

        neighbors = find_neighbors(asm, c)

        assert len(neighbors) == 0

    def test_find_neighbors_multiple_bonds(self):
        """Test finding neighbors with multiple bonds to same atom."""
        asm = Atomistic()
        c1 = Atom(symbol="C")
        c2 = Atom(symbol="C")
        h = Atom(symbol="H")

        asm.add_entity(c1, c2, h)
        # Multiple bonds between c1 and c2 (shouldn't happen, but test robustness)
        asm.add_link(Bond(c1, c2), Bond(c1, h))

        neighbors = find_neighbors(asm, c1)

        # Should find each neighbor once
        assert len(neighbors) == 2
        assert c2 in neighbors
        assert h in neighbors


class TestGetBondBetween:
    """Test get_bond_between function."""

    def test_get_bond_between_exists(self):
        """Test getting bond between two atoms that are bonded."""
        asm = Atomistic()
        c1 = Atom(symbol="C")
        c2 = Atom(symbol="C")

        bond = Bond(c1, c2, order=1)
        asm.add_entity(c1, c2)
        asm.add_link(bond)

        found = get_bond_between(asm, c1, c2)

        assert found is bond
        assert found.get("order") == 1

    def test_get_bond_between_reverse_order(self):
        """Test getting bond works in reverse order."""
        asm = Atomistic()
        c1 = Atom(symbol="C")
        c2 = Atom(symbol="C")

        bond = Bond(c1, c2, order=2)
        asm.add_entity(c1, c2)
        asm.add_link(bond)

        # Try reverse order
        found = get_bond_between(asm, c2, c1)

        assert found is bond

    def test_get_bond_between_not_exists(self):
        """Test getting bond between atoms that are not bonded."""
        asm = Atomistic()
        c1 = Atom(symbol="C")
        c2 = Atom(symbol="C")
        h = Atom(symbol="H")

        asm.add_entity(c1, c2, h)
        asm.add_link(Bond(c1, h))  # c1-h bond, but no c1-c2 bond

        found = get_bond_between(asm, c1, c2)

        assert found is None

    def test_get_bond_between_with_different_bond_orders(self):
        """Test getting bond with different bond orders."""
        asm = Atomistic()
        c1 = Atom(symbol="C")
        c2 = Atom(symbol="C")

        bond = Bond(c1, c2, order=2)
        asm.add_entity(c1, c2)
        asm.add_link(bond)

        found = get_bond_between(asm, c1, c2)

        assert found is bond
        assert found.get("order") == 2


class TestCountBonds:
    """Test count_bonds function."""

    def test_count_bonds_single(self):
        """Test counting bonds for atom with one bond."""
        asm = Atomistic()
        c = Atom(symbol="C")
        h = Atom(symbol="H")

        asm.add_entity(c, h)
        asm.add_link(Bond(c, h))

        count = count_bonds(asm, c)

        assert count == 1

    def test_count_bonds_multiple(self):
        """Test counting bonds for atom with multiple bonds."""
        asm = Atomistic()
        c = Atom(symbol="C")
        h1 = Atom(symbol="H")
        h2 = Atom(symbol="H")
        h3 = Atom(symbol="H")

        asm.add_entity(c, h1, h2, h3)
        asm.add_link(Bond(c, h1), Bond(c, h2), Bond(c, h3))

        count = count_bonds(asm, c)

        assert count == 3

    def test_count_bonds_zero(self):
        """Test counting bonds for isolated atom."""
        asm = Atomistic()
        c = Atom(symbol="C")
        asm.add_entity(c)

        count = count_bonds(asm, c)

        assert count == 0

    def test_count_bonds_with_double_bond(self):
        """Test counting bonds - double bond counts as one."""
        asm = Atomistic()
        c1 = Atom(symbol="C")
        c2 = Atom(symbol="C")
        h = Atom(symbol="H")

        asm.add_entity(c1, c2, h)
        asm.add_link(Bond(c1, c2, order=2), Bond(c1, h))

        count = count_bonds(asm, c1)

        # Should count 2 bonds (one double, one single)
        assert count == 2


class TestRemoveDummyAtoms:
    """Test remove_dummy_atoms function."""

    def test_remove_dummy_atoms_by_symbol(self):
        """Test removing dummy atoms with symbol='*'."""
        asm = Atomistic()
        c = Atom(symbol="C")
        dummy1 = Atom(symbol="*")
        dummy2 = Atom(symbol="*")
        h = Atom(symbol="H")

        asm.add_entity(c, dummy1, dummy2, h)
        asm.add_link(Bond(c, dummy1), Bond(c, h))

        removed = remove_dummy_atoms(asm)

        assert len(removed) == 2
        assert dummy1 in removed
        assert dummy2 in removed

        # Check atoms remaining
        atoms = list(asm.atoms)
        assert len(atoms) == 2
        assert c in atoms
        assert h in atoms
        assert dummy1 not in atoms
        assert dummy2 not in atoms

    def test_remove_dummy_atoms_by_element(self):
        """Test removing dummy atoms with element='*'."""
        asm = Atomistic()
        c = Atom(symbol="C", element="C")
        dummy = Atom(symbol="X", element="*")  # element='*' but symbol is not
        h = Atom(symbol="H")

        asm.add_entity(c, dummy, h)

        removed = remove_dummy_atoms(asm)

        assert len(removed) == 1
        assert dummy in removed

        # Check atoms remaining
        atoms = list(asm.atoms)
        assert len(atoms) == 2
        assert c in atoms
        assert h in atoms

    def test_remove_dummy_atoms_none(self):
        """Test removing dummy atoms when none exist."""
        asm = Atomistic()
        c = Atom(symbol="C")
        h1 = Atom(symbol="H")
        h2 = Atom(symbol="H")

        asm.add_entity(c, h1, h2)

        removed = remove_dummy_atoms(asm)

        assert len(removed) == 0

        # All atoms should remain
        atoms = list(asm.atoms)
        assert len(atoms) == 3

    def test_remove_dummy_atoms_drops_bonds(self):
        """Test that removing dummy atoms drops incident bonds."""
        asm = Atomistic()
        c = Atom(symbol="C")
        dummy = Atom(symbol="*")

        bond = Bond(c, dummy)
        asm.add_entity(c, dummy)
        asm.add_link(bond)

        removed = remove_dummy_atoms(asm)

        assert len(removed) == 1

        # Bond should be removed
        bonds = list(asm.bonds)
        assert len(bonds) == 0
