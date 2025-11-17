#!/usr/bin/env python3
"""Unit tests for Reacter base classes and core functionality.

Tests cover:
- ProductSet dataclass
- Reacter class initialization
- Reacter.run() method with various scenarios
- Error handling
"""

import pytest

from molpy import Atom, Atomistic, Bond
from molpy.core.wrappers.monomer import Monomer
from molpy.reacter import (
    ProductSet,
    Reacter,
    make_double_bond,
    make_single_bond,
    make_triple_bond,
    no_leaving_group,
    port_anchor_selector,
    remove_all_H,
    remove_one_H,
)


class TestProductSet:
    """Test ProductSet dataclass."""

    def test_productset_creation(self):
        """Test creating a ProductSet with product and notes."""
        asm = Atomistic()
        c = Atom(symbol="C")
        asm.add_entity(c)

        notes = {
            "reaction_name": "test_reaction",
            "removed_count": 0,
        }

        product = ProductSet(product=asm, notes=notes)

        assert product.product is asm
        assert product.notes == notes
        assert product.notes["reaction_name"] == "test_reaction"

    def test_productset_default_notes(self):
        """Test ProductSet with default empty notes."""
        asm = Atomistic()
        product = ProductSet(product=asm)

        assert product.product is asm
        assert product.notes == {}


class TestReacter:
    """Test Reacter class."""

    def test_reacter_initialization(self):
        """Test Reacter initialization with all components."""
        reacter = Reacter(
            name="test_reaction",
            anchor_left=port_anchor_selector,
            anchor_right=port_anchor_selector,
            leaving_left=remove_one_H,
            leaving_right=remove_one_H,
            bond_maker=make_single_bond,
        )

        assert reacter.name == "test_reaction"
        assert reacter.anchor_left is port_anchor_selector
        assert reacter.anchor_right is port_anchor_selector
        assert reacter.leaving_left is remove_one_H
        assert reacter.leaving_right is remove_one_H
        assert reacter.bond_maker is make_single_bond

    def test_reacter_repr(self):
        """Test Reacter string representation."""
        reacter = Reacter(
            name="test_reaction",
            anchor_left=port_anchor_selector,
            anchor_right=port_anchor_selector,
            leaving_left=remove_one_H,
            leaving_right=remove_one_H,
            bond_maker=make_single_bond,
        )

        repr_str = repr(reacter)
        assert "Reacter" in repr_str
        assert "test_reaction" in repr_str

    def test_reacter_run_basic(self):
        """Test basic Reacter.run() execution."""
        # Create left monomer: C-H
        asm_L = Atomistic()
        c_L = Atom(symbol="C")
        h_L = Atom(symbol="H")
        asm_L.add_entity(c_L, h_L)
        asm_L.add_link(Bond(c_L, h_L))
        mono_L = Monomer(asm_L)
        mono_L.set_port("1", c_L)

        # Create right monomer: C-H
        asm_R = Atomistic()
        c_R = Atom(symbol="C")
        h_R = Atom(symbol="H")
        asm_R.add_entity(c_R, h_R)
        asm_R.add_link(Bond(c_R, h_R))
        mono_R = Monomer(asm_R)
        mono_R.set_port("2", c_R)

        # Create reaction
        reacter = Reacter(
            name="C-C_coupling",
            anchor_left=port_anchor_selector,
            anchor_right=port_anchor_selector,
            leaving_left=remove_one_H,
            leaving_right=remove_one_H,
            bond_maker=make_single_bond,
        )

        # Run reaction
        product = reacter.run(mono_L, mono_R, port_L="1", port_R="2")

        # Validate ProductSet
        assert isinstance(product, ProductSet)
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
        assert product.notes["removed_count"] == 2
        assert len(product.notes["removed_atoms"]) == 2
        assert product.notes["port_L"] == "1"
        assert product.notes["port_R"] == "2"
        assert product.notes["needs_retypification"] is True

    def test_reacter_run_with_double_bond(self):
        """Test Reacter.run() with double bond formation."""
        # Create monomers
        asm_L = Atomistic()
        c_L = Atom(symbol="C")
        h_L = Atom(symbol="H")
        asm_L.add_entity(c_L, h_L)
        asm_L.add_link(Bond(c_L, h_L))
        mono_L = Monomer(asm_L)
        mono_L.set_port("1", c_L)

        asm_R = Atomistic()
        c_R = Atom(symbol="C")
        h_R = Atom(symbol="H")
        asm_R.add_entity(c_R, h_R)
        asm_R.add_link(Bond(c_R, h_R))
        mono_R = Monomer(asm_R)
        mono_R.set_port("2", c_R)

        # Create reaction with double bond
        reacter = Reacter(
            name="C=C_coupling",
            anchor_left=port_anchor_selector,
            anchor_right=port_anchor_selector,
            leaving_left=remove_one_H,
            leaving_right=remove_one_H,
            bond_maker=make_double_bond,
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
        asm_L = Atomistic()
        c_L = Atom(symbol="C")
        h_L = Atom(symbol="H")
        asm_L.add_entity(c_L, h_L)
        asm_L.add_link(Bond(c_L, h_L))
        mono_L = Monomer(asm_L)
        mono_L.set_port("1", c_L)

        asm_R = Atomistic()
        c_R = Atom(symbol="C")
        h_R = Atom(symbol="H")
        asm_R.add_entity(c_R, h_R)
        asm_R.add_link(Bond(c_R, h_R))
        mono_R = Monomer(asm_R)
        mono_R.set_port("2", c_R)

        # Create reaction with triple bond
        reacter = Reacter(
            name="C#C_coupling",
            anchor_left=port_anchor_selector,
            anchor_right=port_anchor_selector,
            leaving_left=remove_one_H,
            leaving_right=remove_one_H,
            bond_maker=make_triple_bond,
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
        asm_L = Atomistic()
        c_L = Atom(symbol="C")
        asm_L.add_entity(c_L)
        mono_L = Monomer(asm_L)
        mono_L.set_port("1", c_L)

        asm_R = Atomistic()
        c_R = Atom(symbol="C")
        asm_R.add_entity(c_R)
        mono_R = Monomer(asm_R)
        mono_R.set_port("2", c_R)

        # Addition reaction
        reacter = Reacter(
            name="addition",
            anchor_left=port_anchor_selector,
            anchor_right=port_anchor_selector,
            leaving_left=no_leaving_group,
            leaving_right=no_leaving_group,
            bond_maker=make_single_bond,
        )

        product = reacter.run(mono_L, mono_R, port_L="1", port_R="2")

        # No atoms removed
        assert product.notes["removed_count"] == 0
        assert len(product.notes["removed_atoms"]) == 0

        # 2 atoms, 1 bond
        assert len(list(product.product.atoms)) == 2
        assert len(list(product.product.bonds)) == 1

    def test_reacter_run_with_all_H_removal(self):
        """Test Reacter.run() removing all H from one side."""
        # Create left: C-H-H-H
        asm_L = Atomistic()
        c_L = Atom(symbol="C")
        h_L1 = Atom(symbol="H")
        h_L2 = Atom(symbol="H")
        h_L3 = Atom(symbol="H")
        asm_L.add_entity(c_L, h_L1, h_L2, h_L3)
        asm_L.add_link(Bond(c_L, h_L1), Bond(c_L, h_L2), Bond(c_L, h_L3))
        mono_L = Monomer(asm_L)
        mono_L.set_port("1", c_L)

        # Create right: C-H
        asm_R = Atomistic()
        c_R = Atom(symbol="C")
        h_R = Atom(symbol="H")
        asm_R.add_entity(c_R, h_R)
        asm_R.add_link(Bond(c_R, h_R))
        mono_R = Monomer(asm_R)
        mono_R.set_port("2", c_R)

        # Remove all H from left, one H from right
        reacter = Reacter(
            name="asymmetric",
            anchor_left=port_anchor_selector,
            anchor_right=port_anchor_selector,
            leaving_left=remove_all_H,
            leaving_right=remove_one_H,
            bond_maker=make_single_bond,
        )

        product = reacter.run(mono_L, mono_R, port_L="1", port_R="2")

        # Should remove 4 H total
        assert product.notes["removed_count"] == 4
        assert len(product.notes["removed_atoms"]) == 4

        # Only 2 C atoms remain
        atoms = list(product.product.atoms)
        assert len(atoms) == 2
        assert all(a.get("symbol") == "C" for a in atoms)

    def test_reacter_run_with_compute_topology_false(self):
        """Test Reacter.run() with compute_topology=False."""
        # Create monomers
        asm_L = Atomistic()
        c_L = Atom(symbol="C")
        h_L = Atom(symbol="H")
        asm_L.add_entity(c_L, h_L)
        asm_L.add_link(Bond(c_L, h_L))
        mono_L = Monomer(asm_L)
        mono_L.set_port("1", c_L)

        asm_R = Atomistic()
        c_R = Atom(symbol="C")
        h_R = Atom(symbol="H")
        asm_R.add_entity(c_R, h_R)
        asm_R.add_link(Bond(c_R, h_R))
        mono_R = Monomer(asm_R)
        mono_R.set_port("2", c_R)

        reacter = Reacter(
            name="test",
            anchor_left=port_anchor_selector,
            anchor_right=port_anchor_selector,
            leaving_left=remove_one_H,
            leaving_right=remove_one_H,
            bond_maker=make_single_bond,
        )

        product = reacter.run(
            mono_L, mono_R, port_L="1", port_R="2", compute_topology=False
        )

        # Should still work
        assert isinstance(product, ProductSet)
        assert len(list(product.product.atoms)) == 2

    def test_reacter_run_with_record_intermediates(self):
        """Test Reacter.run() with record_intermediates=True."""
        # Create monomers
        asm_L = Atomistic()
        c_L = Atom(symbol="C")
        h_L = Atom(symbol="H")
        asm_L.add_entity(c_L, h_L)
        asm_L.add_link(Bond(c_L, h_L))
        mono_L = Monomer(asm_L)
        mono_L.set_port("1", c_L)

        asm_R = Atomistic()
        c_R = Atom(symbol="C")
        h_R = Atom(symbol="H")
        asm_R.add_entity(c_R, h_R)
        asm_R.add_link(Bond(c_R, h_R))
        mono_R = Monomer(asm_R)
        mono_R.set_port("2", c_R)

        reacter = Reacter(
            name="test",
            anchor_left=port_anchor_selector,
            anchor_right=port_anchor_selector,
            leaving_left=remove_one_H,
            leaving_right=remove_one_H,
            bond_maker=make_single_bond,
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
        asm_L = Atomistic()
        c_L = Atom(symbol="C")
        asm_L.add_entity(c_L)
        mono_L = Monomer(asm_L)
        # Don't set port

        asm_R = Atomistic()
        c_R = Atom(symbol="C")
        asm_R.add_entity(c_R)
        mono_R = Monomer(asm_R)
        mono_R.set_port("2", c_R)

        reacter = Reacter(
            name="test",
            anchor_left=port_anchor_selector,
            anchor_right=port_anchor_selector,
            leaving_left=no_leaving_group,
            leaving_right=no_leaving_group,
            bond_maker=make_single_bond,
        )

        with pytest.raises(ValueError, match="Port '1' not found"):
            reacter.run(mono_L, mono_R, port_L="1", port_R="2")

    def test_reacter_run_missing_port_right(self):
        """Test Reacter.run() raises error when right port is missing."""
        asm_L = Atomistic()
        c_L = Atom(symbol="C")
        asm_L.add_entity(c_L)
        mono_L = Monomer(asm_L)
        mono_L.set_port("1", c_L)

        asm_R = Atomistic()
        c_R = Atom(symbol="C")
        asm_R.add_entity(c_R)
        mono_R = Monomer(asm_R)
        # Don't set port

        reacter = Reacter(
            name="test",
            anchor_left=port_anchor_selector,
            anchor_right=port_anchor_selector,
            leaving_left=no_leaving_group,
            leaving_right=no_leaving_group,
            bond_maker=make_single_bond,
        )

        with pytest.raises(ValueError, match="Port '2' not found"):
            reacter.run(mono_L, mono_R, port_L="1", port_R="2")
