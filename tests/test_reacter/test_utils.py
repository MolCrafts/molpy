#!/usr/bin/env python3
"""Unit tests for Reacter utility functions.

Tests cover:
- find_neighbors
- get_bond_between
- count_bonds
- remove_dummy_atoms
- build_adjacency + adjacency-accelerated query equivalence
"""

from molpy import Atom, Atomistic, Bond
from molpy.reacter.utils import (
    count_bonds,
    create_atom_mapping,
    find_neighbors,
    get_bond_between,
    remove_dummy_atoms,
)


class TestFindNeighbors:
    """Test find_neighbors function."""

    def test_find_all_neighbors(self):
        """Test finding all neighbors of an atom."""
        asm = Atomistic()
        c = Atom(element="C")
        h1 = Atom(element="H")
        h2 = Atom(element="H")
        o = Atom(element="O")

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
        c = Atom(element="C")
        h1 = Atom(element="H")
        h2 = Atom(element="H")
        o = Atom(element="O")

        asm.add_entity(c, h1, h2, o)
        asm.add_link(Bond(c, h1), Bond(c, h2), Bond(c, o))

        # Find only H neighbors
        h_neighbors = find_neighbors(asm, c, element="H")

        assert len(h_neighbors) == 2
        assert all(n.get("element") == "H" for n in h_neighbors)
        assert h1 in h_neighbors
        assert h2 in h_neighbors
        assert o not in h_neighbors

    def test_find_neighbors_with_element_filter_no_match(self):
        """Test finding neighbors with element filter that matches nothing."""
        asm = Atomistic()
        c = Atom(element="C")
        h1 = Atom(element="H")
        h2 = Atom(element="H")

        asm.add_entity(c, h1, h2)
        asm.add_link(Bond(c, h1), Bond(c, h2))

        # Find O neighbors (none exist)
        o_neighbors = find_neighbors(asm, c, element="O")

        assert len(o_neighbors) == 0

    def test_find_neighbors_no_neighbors(self):
        """Test finding neighbors of isolated atom."""
        asm = Atomistic()
        c = Atom(element="C")
        asm.add_entity(c)

        neighbors = find_neighbors(asm, c)

        assert len(neighbors) == 0

    def test_find_neighbors_multiple_bonds(self):
        """Test finding neighbors with multiple bonds to same atom."""
        asm = Atomistic()
        c1 = Atom(element="C")
        c2 = Atom(element="C")
        h = Atom(element="H")

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
        c1 = Atom(element="C")
        c2 = Atom(element="C")

        bond = Bond(c1, c2, order=1)
        asm.add_entity(c1, c2)
        asm.add_link(bond)

        found = get_bond_between(asm, c1, c2)

        assert found is bond
        assert found.get("order") == 1

    def test_get_bond_between_reverse_order(self):
        """Test getting bond works in reverse order."""
        asm = Atomistic()
        c1 = Atom(element="C")
        c2 = Atom(element="C")

        bond = Bond(c1, c2, order=2)
        asm.add_entity(c1, c2)
        asm.add_link(bond)

        # Try reverse order
        found = get_bond_between(asm, c2, c1)

        assert found is bond

    def test_get_bond_between_not_exists(self):
        """Test getting bond between atoms that are not bonded."""
        asm = Atomistic()
        c1 = Atom(element="C")
        c2 = Atom(element="C")
        h = Atom(element="H")

        asm.add_entity(c1, c2, h)
        asm.add_link(Bond(c1, h))  # c1-h bond, but no c1-c2 bond

        found = get_bond_between(asm, c1, c2)

        assert found is None

    def test_get_bond_between_with_different_bond_orders(self):
        """Test getting bond with different bond orders."""
        asm = Atomistic()
        c1 = Atom(element="C")
        c2 = Atom(element="C")

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
        c = Atom(element="C")
        h = Atom(element="H")

        asm.add_entity(c, h)
        asm.add_link(Bond(c, h))

        count = count_bonds(asm, c)

        assert count == 1

    def test_count_bonds_multiple(self):
        """Test counting bonds for atom with multiple bonds."""
        asm = Atomistic()
        c = Atom(element="C")
        h1 = Atom(element="H")
        h2 = Atom(element="H")
        h3 = Atom(element="H")

        asm.add_entity(c, h1, h2, h3)
        asm.add_link(Bond(c, h1), Bond(c, h2), Bond(c, h3))

        count = count_bonds(asm, c)

        assert count == 3

    def test_count_bonds_zero(self):
        """Test counting bonds for isolated atom."""
        asm = Atomistic()
        c = Atom(element="C")
        asm.add_entity(c)

        count = count_bonds(asm, c)

        assert count == 0

    def test_count_bonds_with_double_bond(self):
        """Test counting bonds - double bond counts as one."""
        asm = Atomistic()
        c1 = Atom(element="C")
        c2 = Atom(element="C")
        h = Atom(element="H")

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
        c = Atom(element="C")
        dummy1 = Atom(element="*")
        dummy2 = Atom(element="*")
        h = Atom(element="H")

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
        h = Atom(element="H")

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
        c = Atom(element="C")
        h1 = Atom(element="H")
        h2 = Atom(element="H")

        asm.add_entity(c, h1, h2)

        removed = remove_dummy_atoms(asm)

        assert len(removed) == 0

        # All atoms should remain
        atoms = list(asm.atoms)
        assert len(atoms) == 3

    def test_remove_dummy_atoms_drops_bonds(self):
        """Test that removing dummy atoms drops incident bonds."""
        asm = Atomistic()
        c = Atom(element="C")
        dummy = Atom(element="*")

        bond = Bond(c, dummy)
        asm.add_entity(c, dummy)
        asm.add_link(bond)

        removed = remove_dummy_atoms(asm)

        assert len(removed) == 1

        # Bond should be removed
        bonds = list(asm.bonds)
        assert len(bonds) == 0


class TestAdjacencyQueries:
    """build_adjacency + adjacency= fast paths match the full-scan fallback.

    Planned perf API (spec builder-reacter-05-perf): ``build_adjacency``
    maps each atom to ``[(neighbor, bond), ...]`` in one pass over bonds;
    ``find_neighbors`` / ``get_bond_between`` / ``count_bonds`` accept an
    ``adjacency=`` keyword for O(degree) lookups, with ``None`` falling
    back to the current full bond scan.
    """

    def _make_ethanol(self) -> tuple[Atomistic, dict[str, Atom]]:
        """8-atom ethanol-like structure: H3C-CH2-O-H."""
        asm = Atomistic()
        atoms = {
            "c1": Atom(element="C"),
            "c2": Atom(element="C"),
            "o": Atom(element="O"),
            "h11": Atom(element="H"),
            "h12": Atom(element="H"),
            "h13": Atom(element="H"),
            "h21": Atom(element="H"),
            "ho": Atom(element="H"),
        }
        asm.add_entity(*atoms.values())
        asm.add_link(
            Bond(atoms["c1"], atoms["c2"]),
            Bond(atoms["c1"], atoms["h11"]),
            Bond(atoms["c1"], atoms["h12"]),
            Bond(atoms["c1"], atoms["h13"]),
            Bond(atoms["c2"], atoms["h21"]),
            Bond(atoms["c2"], atoms["o"]),
            Bond(atoms["o"], atoms["ho"]),
        )
        return asm, atoms

    def test_build_adjacency_maps_atoms_to_neighbor_bond_pairs(self):
        """build_adjacency yields (neighbor, bond) pairs per atom."""
        from molpy.reacter.utils import build_adjacency

        asm, atoms = self._make_ethanol()
        adjacency = build_adjacency(asm)

        c1_entries = adjacency[atoms["c1"]]
        assert len(c1_entries) == 4
        bonds_in_asm = list(asm.bonds)
        for neighbor, bond in c1_entries:
            assert any(neighbor is n for n in find_neighbors(asm, atoms["c1"]))
            assert any(bond is b for b in bonds_in_asm)
            assert any(ep is atoms["c1"] for ep in bond.endpoints)
            assert any(ep is neighbor for ep in bond.endpoints)

    def test_find_neighbors_with_adjacency_matches_fallback(self):
        """adjacency= lookup returns the same neighbor objects as full scan."""
        from molpy.reacter.utils import build_adjacency

        asm, atoms = self._make_ethanol()
        adjacency = build_adjacency(asm)

        for atom in atoms.values():
            fallback = find_neighbors(asm, atom)
            fast = find_neighbors(asm, atom, adjacency=adjacency)
            assert len(fast) == len(fallback)
            assert {id(n) for n in fast} == {id(n) for n in fallback}

    def test_find_neighbors_with_adjacency_element_filter_matches_fallback(self):
        """element= filter behaves identically with and without adjacency."""
        from molpy.reacter.utils import build_adjacency

        asm, atoms = self._make_ethanol()
        adjacency = build_adjacency(asm)

        for atom in atoms.values():
            for element in ("H", "C", "O", "N"):
                fallback = find_neighbors(asm, atom, element=element)
                fast = find_neighbors(asm, atom, element=element, adjacency=adjacency)
                assert {id(n) for n in fast} == {id(n) for n in fallback}, (
                    f"element={element!r} filter diverged for "
                    f"{atom.get('element')} atom"
                )

    def test_get_bond_between_with_adjacency_matches_fallback(self):
        """adjacency= returns the identical Bond object (or None)."""
        from molpy.reacter.utils import build_adjacency

        asm, atoms = self._make_ethanol()
        adjacency = build_adjacency(asm)

        atom_list = list(atoms.values())
        for i in atom_list:
            for j in atom_list:
                fallback = get_bond_between(asm, i, j)
                fast = get_bond_between(asm, i, j, adjacency=adjacency)
                assert fast is fallback

    def test_count_bonds_with_adjacency_matches_fallback(self):
        """adjacency= bond counts equal full-scan counts for every atom."""
        from molpy.reacter.utils import build_adjacency

        asm, atoms = self._make_ethanol()
        adjacency = build_adjacency(asm)

        for atom in atoms.values():
            assert count_bonds(asm, atom, adjacency=adjacency) == count_bonds(asm, atom)
