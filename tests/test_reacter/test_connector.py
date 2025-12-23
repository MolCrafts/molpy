#!/usr/bin/env python3
"""Unit tests for MonomerLinker class.

Tests cover:
- MonomerLinker initialization
- connect method
- _select_reacter method
- get_history method
- get_all_modified_atoms method
- needs_retypification method
- clear_history method
"""

import pytest

from molpy.core.atomistic import Atom, Atomistic, Bond
from molpy.reacter import (
    MonomerLinker,
    Reacter,
    form_single_bond,
    select_port,
    select_none,
    select_one_hydrogen,
)


class TestMonomerLinker:
    """Test MonomerLinker class."""

    def test_connector_initialization(self):
        """Test MonomerLinker initialization."""
        default_reacter = Reacter(
            name="default",
            anchor_selector_left=select_port,
            anchor_selector_right=select_port,
            leaving_selector_left=select_one_hydrogen,
            leaving_selector_right=select_one_hydrogen,
            bond_former=form_single_bond,
        )

        connector = MonomerLinker(default_reaction=default_reacter)

        assert connector.default_reaction is default_reacter
        assert connector.specialized_reactions == {}

    def test_connector_initialization_with_overrides(self):
        """Test MonomerLinker initialization with overrides."""
        default_reacter = Reacter(
            name="default",
            anchor_selector_left=select_port,
            anchor_selector_right=select_port,
            leaving_selector_left=select_one_hydrogen,
            leaving_selector_right=select_one_hydrogen,
            bond_former=form_single_bond,
        )

        special_reacter = Reacter(
            name="special",
            anchor_selector_left=select_port,
            anchor_selector_right=select_port,
            leaving_selector_left=select_none,
            leaving_selector_right=select_none,
            bond_former=form_single_bond,
        )

        overrides = {("A", "B"): special_reacter}

        connector = MonomerLinker(
            default_reaction=default_reacter, specialized_reactions=overrides
        )

        assert connector.default_reaction is default_reacter
        assert connector.specialized_reactions == overrides
        assert connector.specialized_reactions[("A", "B")] is special_reacter

    def test_connector_connect_basic(self):
        """Test basic connector.connect() execution."""
        # Create default reacter
        default_reacter = Reacter(
            name="default",
            anchor_selector_left=select_port,
            anchor_selector_right=select_port,
            leaving_selector_left=select_one_hydrogen,
            leaving_selector_right=select_one_hydrogen,
            bond_former=form_single_bond,
        )

        connector = MonomerLinker(default_reaction=default_reacter)

        # Create structures
        struct_L = Atomistic()
        c_L = Atom(symbol="C")
        h_L = Atom(symbol="H")
        struct_L.add_entity(c_L, h_L)
        struct_L.add_link(Bond(c_L, h_L))
        c_L["port"] = "1"

        struct_R = Atomistic()
        c_R = Atom(symbol="C")
        h_R = Atom(symbol="H")
        struct_R.add_entity(c_R, h_R)
        struct_R.add_link(Bond(c_R, h_R))
        c_R["port"] = "2"

        # Connect
        product = connector.connect(
            left=struct_L, right=struct_R, port_L="1", port_R="2"
        )

        assert isinstance(product, Atomistic)
        assert len(list(product.atoms)) == 2
        assert len(list(product.bonds)) == 1

    def test_connector_connect_with_type_override(self):
        """Test connector.connect() using override reacter."""
        default_reacter = Reacter(
            name="default",
            anchor_selector_left=select_port,
            anchor_selector_right=select_port,
            leaving_selector_left=select_one_hydrogen,
            leaving_selector_right=select_one_hydrogen,
            bond_former=form_single_bond,
        )

        special_reacter = Reacter(
            name="special",
            anchor_selector_left=select_port,
            anchor_selector_right=select_port,
            leaving_selector_left=select_none,
            leaving_selector_right=select_none,
            bond_former=form_single_bond,
        )

        overrides = {("A", "B"): special_reacter}
        connector = MonomerLinker(
            default_reaction=default_reacter, specialized_reactions=overrides
        )

        # Create monomers
        struct_L = Atomistic()
        c_L = Atom(symbol="C")
        h_L = Atom(symbol="H")
        struct_L.add_entity(c_L, h_L)
        struct_L.add_link(Bond(c_L, h_L))
        c_L["port"] = "1"

        struct_R = Atomistic()
        c_R = Atom(symbol="C")
        h_R = Atom(symbol="H")
        struct_R.add_entity(c_R, h_R)
        struct_R.add_link(Bond(c_R, h_R))
        c_R["port"] = "2"

        # Connect with types that match override
        product = connector.connect(
            left=struct_L,
            right=struct_R,
            port_L="1",
            port_R="2",
            left_type="A",
            right_type="B",
        )

        # Should use special reacter (no leaving groups)
        assert isinstance(product, Atomistic)
        # Should have 4 atoms (2 C + 2 H, none removed)
        assert len(list(product.atoms)) == 4

    def test_connector_connect_with_reverse_override(self):
        """Test connector.connect() using reverse direction override."""
        default_reacter = Reacter(
            name="default",
            anchor_selector_left=select_port,
            anchor_selector_right=select_port,
            leaving_selector_left=select_one_hydrogen,
            leaving_selector_right=select_one_hydrogen,
            bond_former=form_single_bond,
        )

        special_reacter = Reacter(
            name="special",
            anchor_selector_left=select_port,
            anchor_selector_right=select_port,
            leaving_selector_left=select_none,
            leaving_selector_right=select_none,
            bond_former=form_single_bond,
        )

        overrides = {("A", "B"): special_reacter}
        connector = MonomerLinker(
            default_reaction=default_reacter, specialized_reactions=overrides
        )

        # Create monomers
        struct_L = Atomistic()
        c_L = Atom(symbol="C")
        h_L = Atom(symbol="H")
        struct_L.add_entity(c_L, h_L)
        struct_L.add_link(Bond(c_L, h_L))
        c_L["port"] = "1"

        struct_R = Atomistic()
        c_R = Atom(symbol="C")
        h_R = Atom(symbol="H")
        struct_R.add_entity(c_R, h_R)
        struct_R.add_link(Bond(c_R, h_R))
        c_R["port"] = "2"

        # Connect with reverse types (B, A) should match (A, B) override
        product = connector.connect(
            left=struct_L,
            right=struct_R,
            port_L="1",
            port_R="2",
            left_type="B",
            right_type="A",
        )

        # Should use special reacter
        assert isinstance(product, Atomistic)
        assert len(list(product.atoms)) == 4

    def test_connector_connect_missing_port_left(self):
        """Test connector.connect() raises error when left port is missing."""
        default_reacter = Reacter(
            name="default",
            anchor_selector_left=select_port,
            anchor_selector_right=select_port,
            leaving_selector_left=select_none,
            leaving_selector_right=select_none,
            bond_former=form_single_bond,
        )

        connector = MonomerLinker(default_reaction=default_reacter)

        struct_L = Atomistic()
        c_L = Atom(symbol="C")
        struct_L.add_entity(c_L)
        # Don't set port

        struct_R = Atomistic()
        c_R = Atom(symbol="C")
        struct_R.add_entity(c_R)
        c_R["port"] = "2"

        with pytest.raises(ValueError, match="Port '1' not found"):
            connector.connect(struct_L, struct_R, port_L="1", port_R="2")

    def test_connector_connect_missing_port_right(self):
        """Test connector.connect() raises error when right port is missing."""
        default_reacter = Reacter(
            name="default",
            anchor_selector_left=select_port,
            anchor_selector_right=select_port,
            leaving_selector_left=select_none,
            leaving_selector_right=select_none,
            bond_former=form_single_bond,
        )

        connector = MonomerLinker(default_reaction=default_reacter)

        struct_L = Atomistic()
        c_L = Atom(symbol="C")
        struct_L.add_entity(c_L)
        c_L["port"] = "1"

        struct_R = Atomistic()
        c_R = Atom(symbol="C")
        struct_R.add_entity(c_R)
        # Don't set port

        with pytest.raises(ValueError, match="Port '2' not found"):
            connector.connect(struct_L, struct_R, port_L="1", port_R="2")

    def test_connector_get_history(self):
        """Test connector.get_history() method."""
        default_reacter = Reacter(
            name="default",
            anchor_selector_left=select_port,
            anchor_selector_right=select_port,
            leaving_selector_left=select_none,
            leaving_selector_right=select_none,
            bond_former=form_single_bond,
        )

        connector = MonomerLinker(default_reaction=default_reacter)

        # Initially empty
        history = connector.get_history()
        assert len(history) == 0

        # Create and connect structures
        struct_L = Atomistic()
        c_L = Atom(symbol="C")
        struct_L.add_entity(c_L)
        c_L["port"] = "1"

        struct_R = Atomistic()
        c_R = Atom(symbol="C")
        struct_R.add_entity(c_R)
        c_R["port"] = "2"

        connector.connect(struct_L, struct_R, port_L="1", port_R="2")

        # Should have one entry
        history = connector.get_history()
        assert len(history) == 1
        assert isinstance(history[0].product_info.product, Atomistic)

    def test_connector_get_history_copy(self):
        """Test that get_history() returns a copy."""
        default_reacter = Reacter(
            name="default",
            anchor_selector_left=select_port,
            anchor_selector_right=select_port,
            leaving_selector_left=select_none,
            leaving_selector_right=select_none,
            bond_former=form_single_bond,
        )

        connector = MonomerLinker(default_reaction=default_reacter)

        history1 = connector.get_history()
        history2 = connector.get_history()

        # Should be different objects (copy)
        assert history1 is not history2
        # But same content
        assert history1 == history2

    def test_connector_get_all_modified_atoms(self):
        """Test connector.get_all_modified_atoms() method."""
        default_reacter = Reacter(
            name="default",
            anchor_selector_left=select_port,
            anchor_selector_right=select_port,
            leaving_selector_left=select_none,
            leaving_selector_right=select_none,
            bond_former=form_single_bond,
        )

        connector = MonomerLinker(default_reaction=default_reacter)

        # Initially empty
        modified = connector.get_all_modified_atoms()
        assert len(modified) == 0

        # Create and connect structures
        struct_L = Atomistic()
        c_L = Atom(symbol="C")
        struct_L.add_entity(c_L)
        c_L["port"] = "1"

        struct_R = Atomistic()
        c_R = Atom(symbol="C")
        struct_R.add_entity(c_R)
        c_R["port"] = "2"

        connector.connect(struct_L, struct_R, port_L="1", port_R="2")

        # Check modified atoms (may be empty if notes don't include it)
        modified = connector.get_all_modified_atoms()
        assert isinstance(modified, set)

    def test_connector_needs_retypification(self):
        """Test connector.needs_retypification() method."""
        default_reacter = Reacter(
            name="default",
            anchor_selector_left=select_port,
            anchor_selector_right=select_port,
            leaving_selector_left=select_none,
            leaving_selector_right=select_none,
            bond_former=form_single_bond,
        )

        connector = MonomerLinker(default_reaction=default_reacter)

        # Initially False (no reactions yet)
        assert connector.needs_retypification() is False

        # Create and connect structures
        struct_L = Atomistic()
        c_L = Atom(symbol="C")
        struct_L.add_entity(c_L)
        c_L["port"] = "1"

        struct_R = Atomistic()
        c_R = Atom(symbol="C")
        struct_R.add_entity(c_R)
        c_R["port"] = "2"

        connector.connect(struct_L, struct_R, port_L="1", port_R="2")

        # Should be True (default reacter sets needs_retypification=True)
        assert connector.needs_retypification() is True

    def test_connector_clear_history(self):
        """Test connector.clear_history() method."""
        default_reacter = Reacter(
            name="default",
            anchor_selector_left=select_port,
            anchor_selector_right=select_port,
            leaving_selector_left=select_none,
            leaving_selector_right=select_none,
            bond_former=form_single_bond,
        )

        connector = MonomerLinker(default_reaction=default_reacter)

        # Create and connect structures
        struct_L = Atomistic()
        c_L = Atom(symbol="C")
        struct_L.add_entity(c_L)
        c_L["port"] = "1"

        struct_R = Atomistic()
        c_R = Atom(symbol="C")
        struct_R.add_entity(c_R)
        c_R["port"] = "2"

        connector.connect(struct_L, struct_R, port_L="1", port_R="2")

        # Should have history
        assert len(connector.get_history()) == 1

        # Clear history
        connector.clear_history()

        # Should be empty
        assert len(connector.get_history()) == 0
        assert connector.needs_retypification() is False
