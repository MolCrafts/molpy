#!/usr/bin/env python3
"""Unit tests for Reacter base classes and core functionality.

Tests cover:
- ReactionResult dataclass
- Reacter class initialization
- Reacter.run() method with various scenarios
- Error handling
"""

import pytest

from molpy.core.atomistic import Atom, Atomistic, Bond
from molpy.reacter import (
    Reacter,
    ReactionResult,
    form_double_bond,
    form_single_bond,
    form_triple_bond,
    select_all_hydrogens,
    select_port,
    select_none,
    select_one_hydrogen,
)


class TestProductSet:
    """Test ReactionResult dataclass."""

    def test_reactionresult_creation(self):
        """Test creating a ReactionResult with new structure."""
        asm = Atomistic()
        c = Atom(element="C")
        asm.add_entity(c)

        merged = Atomistic()
        merged.add_entity(c)

        result = ReactionResult(
            product=asm,
            reactants=merged,
            reaction_name="test_reaction",
        )

        # Test new structure
        assert result.product is asm
        assert result.reaction_name == "test_reaction"
        assert result.reactants is merged


class TestReacter:
    """Test Reacter class."""

    def test_reacter_initialization(self):
        """Test Reacter initialization with all components."""
        reacter = Reacter(
            name="test_reaction",
            anchor_selector_left=select_port,
            anchor_selector_right=select_port,
            leaving_selector_left=select_one_hydrogen,
            leaving_selector_right=select_one_hydrogen,
            bond_former=form_single_bond,
        )

        assert reacter.name == "test_reaction"
        assert reacter.anchor_selector_left is select_port
        assert reacter.anchor_selector_right is select_port
        assert reacter.leaving_selector_left is select_one_hydrogen
        assert reacter.leaving_selector_right is select_one_hydrogen
        assert reacter.bond_former is form_single_bond

    def test_reacter_repr(self):
        """Test Reacter string representation."""
        reacter = Reacter(
            name="test_reaction",
            anchor_selector_left=select_port,
            anchor_selector_right=select_port,
            leaving_selector_left=select_one_hydrogen,
            leaving_selector_right=select_one_hydrogen,
            bond_former=form_single_bond,
        )

        repr_str = repr(reacter)
        assert "Reacter" in repr_str
        assert "test_reaction" in repr_str

    def test_reacter_run_basic(self):
        """Test basic Reacter.run() execution."""
        # Create left structure: C-H
        struct_L = Atomistic()
        c_L = Atom(element="C")
        h_L = Atom(element="H")
        struct_L.add_entity(c_L, h_L)
        struct_L.add_link(Bond(c_L, h_L))
        c_L["port"] = "1"

        # Create right structure: C-H
        struct_R = Atomistic()
        c_R = Atom(element="C")
        h_R = Atom(element="H")
        struct_R.add_entity(c_R, h_R)
        struct_R.add_link(Bond(c_R, h_R))
        c_R["port"] = "2"

        # Create reaction
        reacter = Reacter(
            name="C-C_coupling",
            anchor_selector_left=select_port,
            anchor_selector_right=select_port,
            leaving_selector_left=select_one_hydrogen,
            leaving_selector_right=select_one_hydrogen,
            bond_former=form_single_bond,
        )

        # Run reaction
        from molpy.reacter.selectors import find_port_atom

        port_atom_L = find_port_atom(struct_L, "1")
        port_atom_R = find_port_atom(struct_R, "2")
        result = reacter.run(
            struct_L, struct_R, port_atom_L=port_atom_L, port_atom_R=port_atom_R
        )

        # Validate ReactionResult
        assert isinstance(result, ReactionResult)
        assert isinstance(result.product, Atomistic)

        # Check atoms (2 C, 0 H)
        atoms = list(result.product.atoms)
        assert len(atoms) == 2
        assert all(a.get("element") == "C" for a in atoms)

        # Check bonds (1 C-C bond)
        bonds = list(result.product.bonds)
        assert len(bonds) == 1
        assert bonds[0].get("order") == 1

        # Check ReactionResult attributes
        assert result.reaction_name == "C-C_coupling"
        assert len(result.removed_atoms) == 2
        assert result.anchor_L is not None
        assert result.anchor_R is not None
        assert result.anchor_L.get("element") == "C"
        assert result.anchor_R.get("element") == "C"
        assert result.requires_retype is True

    def test_reacter_run_with_double_bond(self):
        """Test Reacter.run() with double bond formation."""
        # Create structures
        struct_L = Atomistic()
        c_L = Atom(element="C")
        h_L = Atom(element="H")
        struct_L.add_entity(c_L, h_L)
        struct_L.add_link(Bond(c_L, h_L))
        c_L["port"] = "1"

        struct_R = Atomistic()
        c_R = Atom(element="C")
        h_R = Atom(element="H")
        struct_R.add_entity(c_R, h_R)
        struct_R.add_link(Bond(c_R, h_R))
        c_R["port"] = "2"

        # Create reaction with double bond
        reacter = Reacter(
            name="C=C_coupling",
            anchor_selector_left=select_port,
            anchor_selector_right=select_port,
            leaving_selector_left=select_one_hydrogen,
            leaving_selector_right=select_one_hydrogen,
            bond_former=form_double_bond,
        )

        from molpy.reacter.selectors import find_port_atom

        port_atom_L = find_port_atom(struct_L, "1")
        port_atom_R = find_port_atom(struct_R, "2")
        result = reacter.run(
            struct_L, struct_R, port_atom_L=port_atom_L, port_atom_R=port_atom_R
        )

        # Check bond order
        bonds = list(result.product.bonds)
        assert len(bonds) == 1
        assert bonds[0].get("order") == 2
        assert bonds[0].get("kind") == "="

    def test_reacter_run_with_triple_bond(self):
        """Test Reacter.run() with triple bond formation."""
        # Create structures
        struct_L = Atomistic()
        c_L = Atom(element="C")
        h_L = Atom(element="H")
        struct_L.add_entity(c_L, h_L)
        struct_L.add_link(Bond(c_L, h_L))
        c_L["port"] = "1"

        struct_R = Atomistic()
        c_R = Atom(element="C")
        h_R = Atom(element="H")
        struct_R.add_entity(c_R, h_R)
        struct_R.add_link(Bond(c_R, h_R))
        c_R["port"] = "2"

        # Create reaction with triple bond
        reacter = Reacter(
            name="C#C_coupling",
            anchor_selector_left=select_port,
            anchor_selector_right=select_port,
            leaving_selector_left=select_one_hydrogen,
            leaving_selector_right=select_one_hydrogen,
            bond_former=form_triple_bond,
        )

        from molpy.reacter.selectors import find_port_atom

        port_atom_L = find_port_atom(struct_L, "1")
        port_atom_R = find_port_atom(struct_R, "2")
        result = reacter.run(
            struct_L, struct_R, port_atom_L=port_atom_L, port_atom_R=port_atom_R
        )

        # Check bond order
        bonds = list(result.product.bonds)
        assert len(bonds) == 1
        assert bonds[0].get("order") == 3
        assert bonds[0].get("kind") == "#"

    def test_reacter_run_no_leaving_groups(self):
        """Test Reacter.run() with no leaving groups."""
        # Create structures
        struct_L = Atomistic()
        c_L = Atom(element="C")
        struct_L.add_entity(c_L)
        c_L["port"] = "1"

        struct_R = Atomistic()
        c_R = Atom(element="C")
        struct_R.add_entity(c_R)
        c_R["port"] = "2"

        # Addition reaction
        reacter = Reacter(
            name="addition",
            anchor_selector_left=select_port,
            anchor_selector_right=select_port,
            leaving_selector_left=select_none,
            leaving_selector_right=select_none,
            bond_former=form_single_bond,
        )

        from molpy.reacter.selectors import find_port_atom

        port_atom_L = find_port_atom(struct_L, "1")
        port_atom_R = find_port_atom(struct_R, "2")
        result = reacter.run(
            struct_L, struct_R, port_atom_L=port_atom_L, port_atom_R=port_atom_R
        )

        # No atoms removed
        assert len(result.removed_atoms) == 0

        # 2 atoms, 1 bond
        assert len(list(result.product.atoms)) == 2
        assert len(list(result.product.bonds)) == 1

    def test_reacter_run_with_all_H_removal(self):
        """Test Reacter.run() removing all H from one side."""
        # Create left: C-H-H-H
        struct_L = Atomistic()
        c_L = Atom(element="C")
        h_L1 = Atom(element="H")
        h_L2 = Atom(element="H")
        h_L3 = Atom(element="H")
        struct_L.add_entity(c_L, h_L1, h_L2, h_L3)
        struct_L.add_link(Bond(c_L, h_L1), Bond(c_L, h_L2), Bond(c_L, h_L3))
        c_L["port"] = "1"

        # Create right: C-H
        struct_R = Atomistic()
        c_R = Atom(element="C")
        h_R = Atom(element="H")
        struct_R.add_entity(c_R, h_R)
        struct_R.add_link(Bond(c_R, h_R))
        c_R["port"] = "2"

        # Remove all H from left, one H from right
        reacter = Reacter(
            name="asymmetric",
            anchor_selector_left=select_port,
            anchor_selector_right=select_port,
            leaving_selector_left=select_all_hydrogens,
            leaving_selector_right=select_one_hydrogen,
            bond_former=form_single_bond,
        )

        from molpy.reacter.selectors import find_port_atom

        port_atom_L = find_port_atom(struct_L, "1")
        port_atom_R = find_port_atom(struct_R, "2")
        result = reacter.run(
            struct_L, struct_R, port_atom_L=port_atom_L, port_atom_R=port_atom_R
        )

        # Should remove 4 H total
        assert len(result.removed_atoms) == 4

        # Only 2 C atoms remain
        atoms = list(result.product.atoms)
        assert len(atoms) == 2
        assert all(a.get("element") == "C" for a in atoms)

    def test_reacter_run_with_compute_topology_false(self):
        """Test Reacter.run() with compute_topology=False."""
        # Create structures
        struct_L = Atomistic()
        c_L = Atom(element="C")
        h_L = Atom(element="H")
        struct_L.add_entity(c_L, h_L)
        struct_L.add_link(Bond(c_L, h_L))
        c_L["port"] = "1"

        struct_R = Atomistic()
        c_R = Atom(element="C")
        h_R = Atom(element="H")
        struct_R.add_entity(c_R, h_R)
        struct_R.add_link(Bond(c_R, h_R))
        c_R["port"] = "2"

        reacter = Reacter(
            name="test",
            anchor_selector_left=select_port,
            anchor_selector_right=select_port,
            leaving_selector_left=select_one_hydrogen,
            leaving_selector_right=select_one_hydrogen,
            bond_former=form_single_bond,
        )

        from molpy.reacter.selectors import find_port_atom

        port_atom_L = find_port_atom(struct_L, "1")
        port_atom_R = find_port_atom(struct_R, "2")
        result = reacter.run(
            struct_L,
            struct_R,
            port_atom_L=port_atom_L,
            port_atom_R=port_atom_R,
            compute_topology=False,
        )

        # Should still work
        assert isinstance(result, ReactionResult)
        assert len(list(result.product.atoms)) == 2

    def test_reacter_run_with_record_intermediates(self):
        """Test Reacter.run() with record_intermediates=True."""
        # Create structures
        struct_L = Atomistic()
        c_L = Atom(element="C")
        h_L = Atom(element="H")
        struct_L.add_entity(c_L, h_L)
        struct_L.add_link(Bond(c_L, h_L))
        c_L["port"] = "1"

        struct_R = Atomistic()
        c_R = Atom(element="C")
        h_R = Atom(element="H")
        struct_R.add_entity(c_R, h_R)
        struct_R.add_link(Bond(c_R, h_R))
        c_R["port"] = "2"

        reacter = Reacter(
            name="test",
            anchor_selector_left=select_port,
            anchor_selector_right=select_port,
            leaving_selector_left=select_one_hydrogen,
            leaving_selector_right=select_one_hydrogen,
            bond_former=form_single_bond,
        )

        from molpy.reacter.selectors import find_port_atom

        port_atom_L = find_port_atom(struct_L, "1")
        port_atom_R = find_port_atom(struct_R, "2")
        result = reacter.run(
            struct_L,
            struct_R,
            port_atom_L=port_atom_L,
            port_atom_R=port_atom_R,
            record_intermediates=True,
        )

        # Check intermediates are recorded
        assert isinstance(result.intermediates, list)
        assert len(result.intermediates) > 0

        # Check intermediate structure
        for inter in result.intermediates:
            assert "step" in inter
            assert "description" in inter

    def test_reacter_run_missing_port_left(self):
        """Test Reacter.run() raises error when left port is missing."""
        struct_L = Atomistic()
        c_L = Atom(element="C")
        struct_L.add_entity(c_L)
        # Don't set port

        struct_R = Atomistic()
        c_R = Atom(element="C")
        struct_R.add_entity(c_R)
        c_R["port"] = "2"

        reacter = Reacter(
            name="test",
            anchor_selector_left=select_port,
            anchor_selector_right=select_port,
            leaving_selector_left=select_none,
            leaving_selector_right=select_none,
            bond_former=form_single_bond,
        )

        from molpy.reacter.selectors import find_port_atom

        with pytest.raises(ValueError, match="Port '1' not found"):
            port_atom_L = find_port_atom(struct_L, "1")  # noqa: F841

    def test_reacter_run_missing_port_right(self):
        """Test Reacter.run() raises error when right port is missing."""
        struct_L = Atomistic()
        c_L = Atom(element="C")
        struct_L.add_entity(c_L)
        c_L["port"] = "1"

        struct_R = Atomistic()
        c_R = Atom(element="C")
        struct_R.add_entity(c_R)
        # Don't set port

        reacter = Reacter(
            name="test",
            anchor_selector_left=select_port,
            anchor_selector_right=select_port,
            leaving_selector_left=select_none,
            leaving_selector_right=select_none,
            bond_former=form_single_bond,
        )

        from molpy.reacter.selectors import find_port_atom

        with pytest.raises(ValueError, match="Port '2' not found"):
            port_atom_R = find_port_atom(struct_R, "2")  # noqa: F841
