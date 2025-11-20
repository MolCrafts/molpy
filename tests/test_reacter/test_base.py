#!/usr/bin/env python3
"""Unit tests for Reacter base classes and core functionality.

Tests cover:
- ReactionProduct dataclass
- Reacter class initialization
- Reacter.run() method with various scenarios
- Error handling
"""

import pytest

from molpy import Atom, Atomistic, Bond
from molpy.core.wrappers.monomer import Monomer
from molpy.reacter import (
    ReactionProduct,
    Reacter,
    form_double_bond,
    form_single_bond,
    form_triple_bond,
    select_none,
    select_port_atom,
    select_all_hydrogens,
    select_one_hydrogen,
)


class TestProductSet:
    """Test ReactionProduct dataclass."""

    def test_productset_creation(self):
        """Test creating a ReactionProduct with product and notes."""
        asm = Atomistic()
        c = Atom(symbol="C")
        asm.add_entity(c)

        notes = {
            "reaction_name": "test_reaction",
            "n_eliminated": 0,
        }

        product = ReactionProduct(product=asm, notes=notes)

        assert product.product is asm
        assert product.notes == notes
        assert product.notes["reaction_name"] == "test_reaction"

    def test_productset_default_notes(self):
        """Test ReactionProduct with default empty notes."""
        asm = Atomistic()
        product = ReactionProduct(product=asm)

        assert product.product is asm
        assert product.notes == {}


class TestReacter:
    """Test Reacter class."""

    def test_reacter_initialization(self):
        """Test Reacter initialization with all components."""
        reacter = Reacter(
            name="test_reaction",
            port_selector_left=select_port_atom,
            port_selector_right=select_port_atom,
            leaving_selector_left=select_one_hydrogen,
            leaving_selector_right=select_one_hydrogen,
            bond_former=form_single_bond,
        )

        assert reacter.name == "test_reaction"
        assert reacter.port_selector_left is select_port_atom
        assert reacter.port_selector_right is select_port_atom
        assert reacter.leaving_selector_left is select_one_hydrogen
        assert reacter.leaving_selector_right is select_one_hydrogen
        assert reacter.bond_former is form_single_bond

    def test_reacter_repr(self):
        """Test Reacter string representation."""
        reacter = Reacter(
            name="test_reaction",
            port_selector_left=select_port_atom,
            port_selector_right=select_port_atom,
            leaving_selector_left=select_one_hydrogen,
            leaving_selector_right=select_one_hydrogen,
            bond_former=form_single_bond,
        )

        repr_str = repr(reacter)
        assert "Reacter" in repr_str
        assert "test_reaction" in repr_str

    def test_reacter_run_basic(self):
        """Test basic Reacter.run() execution."""
        # Create left monomer: C-H
        mono_L = Monomer()
        c_L = Atom(symbol="C")
        h_L = Atom(symbol="H")
        mono_L.add_entity(c_L, h_L)
        mono_L.add_link(Bond(c_L, h_L))
        mono_L.set_port("1", c_L)

        # Create right monomer: C-H
        mono_R = Monomer()
        c_R = Atom(symbol="C")
        h_R = Atom(symbol="H")
        mono_R.add_entity(c_R, h_R)
        mono_R.add_link(Bond(c_R, h_R))
        mono_R.set_port("2", c_R)

        # Create reaction
        reacter = Reacter(
            name="C-C_coupling",
            port_selector_left=select_port_atom,
            port_selector_right=select_port_atom,
            leaving_selector_left=select_one_hydrogen,
            leaving_selector_right=select_one_hydrogen,
            bond_former=form_single_bond,
        )

        # Run reaction
        product = reacter.run(mono_L, mono_R, port_L="1", port_R="2")

        # Validate ReactionProduct
        assert isinstance(product, ReactionProduct)
        assert isinstance(product.product, Atomistic)

        # Check atoms (2 C, 0 H)
        atoms = list(product.product.atoms)
        assert len(atoms) == 2
        assert all(a.get("symbol") == "C" for a in atoms)

        # Check bonds (1 C-C bond)
        bonds = list(product.product.bonds)
        assert len(bonds) == 1
        assert bonds[0].get("order") == 1

        # Check notes
        assert product.notes["reaction_name"] == "C-C_coupling"
        assert product.notes["n_eliminated"] == 2
        assert len(product.notes["eliminated_atoms"]) == 2
        assert product.notes["port_name_L"] == "1"
        assert product.notes["port_name_R"] == "2"
        assert product.notes["port_L"] == c_L
        assert product.notes["port_R"] == c_R
        assert product.notes["requires_retype"] is True

    def test_reacter_run_with_double_bond(self):
        """Test Reacter.run() with double bond formation."""
        # Create monomers
        mono_L = Monomer()
        c_L = Atom(symbol="C")
        h_L = Atom(symbol="H")
        mono_L.add_entity(c_L, h_L)
        mono_L.add_link(Bond(c_L, h_L))
        mono_L.set_port("1", c_L)

        mono_R = Monomer()
        c_R = Atom(symbol="C")
        h_R = Atom(symbol="H")
        mono_R.add_entity(c_R, h_R)
        mono_R.add_link(Bond(c_R, h_R))
        mono_R.set_port("2", c_R)

        # Create reaction with double bond
        reacter = Reacter(
            name="C=C_coupling",
            port_selector_left=select_port_atom,
            port_selector_right=select_port_atom,
            leaving_selector_left=select_one_hydrogen,
            leaving_selector_right=select_one_hydrogen,
            bond_former=form_double_bond,
        )

        product = reacter.run(mono_L, mono_R, port_L="1", port_R="2")

        # Check bond order
        bonds = list(product.product.bonds)
        assert len(bonds) == 1
        assert bonds[0].get("order") == 2
        assert bonds[0].get("kind") == "="

    def test_reacter_run_with_triple_bond(self):
        """Test Reacter.run() with triple bond formation."""
        # Create monomers
        mono_L = Monomer()
        c_L = Atom(symbol="C")
        h_L = Atom(symbol="H")
        mono_L.add_entity(c_L, h_L)
        mono_L.add_link(Bond(c_L, h_L))
        mono_L.set_port("1", c_L)

        mono_R = Monomer()
        c_R = Atom(symbol="C")
        h_R = Atom(symbol="H")
        mono_R.add_entity(c_R, h_R)
        mono_R.add_link(Bond(c_R, h_R))
        mono_R.set_port("2", c_R)

        # Create reaction with triple bond
        reacter = Reacter(
            name="C#C_coupling",
            port_selector_left=select_port_atom,
            port_selector_right=select_port_atom,
            leaving_selector_left=select_one_hydrogen,
            leaving_selector_right=select_one_hydrogen,
            bond_former=form_triple_bond,
        )

        product = reacter.run(mono_L, mono_R, port_L="1", port_R="2")

        # Check bond order
        bonds = list(product.product.bonds)
        assert len(bonds) == 1
        assert bonds[0].get("order") == 3
        assert bonds[0].get("kind") == "#"

    def test_reacter_run_no_leaving_groups(self):
        """Test Reacter.run() with no leaving groups."""
        # Create monomers
        mono_L = Monomer()
        c_L = Atom(symbol="C")
        mono_L.add_entity(c_L)
        mono_L.set_port("1", c_L)

        mono_R = Monomer()
        c_R = Atom(symbol="C")
        mono_R.add_entity(c_R)
        mono_R.set_port("2", c_R)

        # Addition reaction
        reacter = Reacter(
            name="addition",
            port_selector_left=select_port_atom,
            port_selector_right=select_port_atom,
            leaving_selector_left=select_none,
            leaving_selector_right=select_none,
            bond_former=form_single_bond,
        )

        product = reacter.run(mono_L, mono_R, port_L="1", port_R="2")

        # No atoms removed
        assert product.notes["n_eliminated"] == 0
        assert len(product.notes["eliminated_atoms"]) == 0

        # 2 atoms, 1 bond
        assert len(list(product.product.atoms)) == 2
        assert len(list(product.product.bonds)) == 1

    def test_reacter_run_with_all_H_removal(self):
        """Test Reacter.run() removing all H from one side."""
        # Create left: C-H-H-H
        mono_L = Monomer()
        c_L = Atom(symbol="C")
        h_L1 = Atom(symbol="H")
        h_L2 = Atom(symbol="H")
        h_L3 = Atom(symbol="H")
        mono_L.add_entity(c_L, h_L1, h_L2, h_L3)
        mono_L.add_link(Bond(c_L, h_L1), Bond(c_L, h_L2), Bond(c_L, h_L3))
        mono_L.set_port("1", c_L)

        # Create right: C-H
        mono_R = Monomer()
        c_R = Atom(symbol="C")
        h_R = Atom(symbol="H")
        mono_R.add_entity(c_R, h_R)
        mono_R.add_link(Bond(c_R, h_R))
        mono_R.set_port("2", c_R)

        # Remove all H from left, one H from right
        reacter = Reacter(
            name="asymmetric",
            port_selector_left=select_port_atom,
            port_selector_right=select_port_atom,
            leaving_selector_left=select_all_hydrogens,
            leaving_selector_right=select_one_hydrogen,
            bond_former=form_single_bond,
        )

        product = reacter.run(mono_L, mono_R, port_L="1", port_R="2")

        # Should remove 4 H total
        assert product.notes["n_eliminated"] == 4
        assert len(product.notes["eliminated_atoms"]) == 4

        # Only 2 C atoms remain
        atoms = list(product.product.atoms)
        assert len(atoms) == 2
        assert all(a.get("symbol") == "C" for a in atoms)

    def test_reacter_run_with_compute_topology_false(self):
        """Test Reacter.run() with compute_topology=False."""
        # Create monomers
        mono_L = Monomer()
        c_L = Atom(symbol="C")
        h_L = Atom(symbol="H")
        mono_L.add_entity(c_L, h_L)
        mono_L.add_link(Bond(c_L, h_L))
        mono_L.set_port("1", c_L)

        mono_R = Monomer()
        c_R = Atom(symbol="C")
        h_R = Atom(symbol="H")
        mono_R.add_entity(c_R, h_R)
        mono_R.add_link(Bond(c_R, h_R))
        mono_R.set_port("2", c_R)

        reacter = Reacter(
            name="test",
            port_selector_left=select_port_atom,
            port_selector_right=select_port_atom,
            leaving_selector_left=select_one_hydrogen,
            leaving_selector_right=select_one_hydrogen,
            bond_former=form_single_bond,
        )

        product = reacter.run(
            mono_L, mono_R, port_L="1", port_R="2", compute_topology=False
        )

        # Should still work
        assert isinstance(product, ReactionProduct)
        assert len(list(product.product.atoms)) == 2

    def test_reacter_run_with_record_intermediates(self):
        """Test Reacter.run() with record_intermediates=True."""
        # Create monomers
        mono_L = Monomer()
        c_L = Atom(symbol="C")
        h_L = Atom(symbol="H")
        mono_L.add_entity(c_L, h_L)
        mono_L.add_link(Bond(c_L, h_L))
        mono_L.set_port("1", c_L)

        mono_R = Monomer()
        c_R = Atom(symbol="C")
        h_R = Atom(symbol="H")
        mono_R.add_entity(c_R, h_R)
        mono_R.add_link(Bond(c_R, h_R))
        mono_R.set_port("2", c_R)

        reacter = Reacter(
            name="test",
            port_selector_left=select_port_atom,
            port_selector_right=select_port_atom,
            leaving_selector_left=select_one_hydrogen,
            leaving_selector_right=select_one_hydrogen,
            bond_former=form_single_bond,
        )

        product = reacter.run(
            mono_L, mono_R, port_L="1", port_R="2", record_intermediates=True
        )

        # Check intermediates are recorded
        assert "intermediates" in product.notes
        intermediates = product.notes["intermediates"]
        assert isinstance(intermediates, list)
        assert len(intermediates) > 0

        # Check intermediate structure
        for inter in intermediates:
            assert "step" in inter
            assert "description" in inter

    def test_reacter_run_missing_port_left(self):
        """Test Reacter.run() raises error when left port is missing."""
        mono_L = Monomer()
        c_L = Atom(symbol="C")
        mono_L.add_entity(c_L)
        # Don't set port

        mono_R = Monomer()
        c_R = Atom(symbol="C")
        mono_R.add_entity(c_R)
        mono_R.set_port("2", c_R)

        reacter = Reacter(
            name="test",
            port_selector_left=select_port_atom,
            port_selector_right=select_port_atom,
            leaving_selector_left=select_none,
            leaving_selector_right=select_none,
            bond_former=form_single_bond,
        )

        with pytest.raises(ValueError, match="Port '1' not found"):
            reacter.run(mono_L, mono_R, port_L="1", port_R="2")

    def test_reacter_run_missing_port_right(self):
        """Test Reacter.run() raises error when right port is missing."""
        mono_L = Monomer()
        c_L = Atom(symbol="C")
        mono_L.add_entity(c_L)
        mono_L.set_port("1", c_L)

        mono_R = Monomer()
        c_R = Atom(symbol="C")
        mono_R.add_entity(c_R)
        # Don't set port

        reacter = Reacter(
            name="test",
            port_selector_left=select_port_atom,
            port_selector_right=select_port_atom,
            leaving_selector_left=select_none,
            leaving_selector_right=select_none,
            bond_former=form_single_bond,
        )

        with pytest.raises(ValueError, match="Port '2' not found"):
            reacter.run(mono_L, mono_R, port_L="1", port_R="2")
