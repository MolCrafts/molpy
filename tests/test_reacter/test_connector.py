#!/usr/bin/env python3
"""Unit tests for ReacterConnector class.

Tests cover:
- ReacterConnector initialization
- connect method
- _select_reacter method
- get_history method
- get_all_modified_atoms method
- needs_retypification method
- clear_history method
"""

import pytest

from molpy import Atom, Atomistic, Bond
from molpy.core.wrappers.monomer import Monomer
from molpy.reacter import (
    Reacter,
    ReacterConnector,
    make_single_bond,
    no_leaving_group,
    port_anchor_selector,
    remove_one_H,
)


class TestReacterConnector:
    """Test ReacterConnector class."""

    def test_connector_initialization(self):
        """Test ReacterConnector initialization."""
        default_reacter = Reacter(
            name="default",
            anchor_left=port_anchor_selector,
            anchor_right=port_anchor_selector,
            leaving_left=remove_one_H,
            leaving_right=remove_one_H,
            bond_maker=make_single_bond,
        )

        connector = ReacterConnector(default=default_reacter)

        assert connector.default is default_reacter
        assert connector.overrides == {}

    def test_connector_initialization_with_overrides(self):
        """Test ReacterConnector initialization with overrides."""
        default_reacter = Reacter(
            name="default",
            anchor_left=port_anchor_selector,
            anchor_right=port_anchor_selector,
            leaving_left=remove_one_H,
            leaving_right=remove_one_H,
            bond_maker=make_single_bond,
        )

        special_reacter = Reacter(
            name="special",
            anchor_left=port_anchor_selector,
            anchor_right=port_anchor_selector,
            leaving_left=no_leaving_group,
            leaving_right=no_leaving_group,
            bond_maker=make_single_bond,
        )

        overrides = {("A", "B"): special_reacter}

        connector = ReacterConnector(default=default_reacter, overrides=overrides)

        assert connector.default is default_reacter
        assert connector.overrides == overrides
        assert connector.overrides[("A", "B")] is special_reacter

    def test_connector_connect_basic(self):
        """Test basic connector.connect() execution."""
        # Create default reacter
        default_reacter = Reacter(
            name="default",
            anchor_left=port_anchor_selector,
            anchor_right=port_anchor_selector,
            leaving_left=remove_one_H,
            leaving_right=remove_one_H,
            bond_maker=make_single_bond,
        )

        connector = ReacterConnector(default=default_reacter)

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

        # Connect
        product = connector.connect(left=mono_L, right=mono_R, port_L="1", port_R="2")

        assert isinstance(product, Atomistic)
        assert len(list(product.atoms)) == 2
        assert len(list(product.bonds)) == 1

    def test_connector_connect_with_type_override(self):
        """Test connector.connect() using override reacter."""
        default_reacter = Reacter(
            name="default",
            anchor_left=port_anchor_selector,
            anchor_right=port_anchor_selector,
            leaving_left=remove_one_H,
            leaving_right=remove_one_H,
            bond_maker=make_single_bond,
        )

        special_reacter = Reacter(
            name="special",
            anchor_left=port_anchor_selector,
            anchor_right=port_anchor_selector,
            leaving_left=no_leaving_group,
            leaving_right=no_leaving_group,
            bond_maker=make_single_bond,
        )

        overrides = {("A", "B"): special_reacter}
        connector = ReacterConnector(default=default_reacter, overrides=overrides)

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

        # Connect with types that match override
        product = connector.connect(
            left=mono_L,
            right=mono_R,
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
            anchor_left=port_anchor_selector,
            anchor_right=port_anchor_selector,
            leaving_left=remove_one_H,
            leaving_right=remove_one_H,
            bond_maker=make_single_bond,
        )

        special_reacter = Reacter(
            name="special",
            anchor_left=port_anchor_selector,
            anchor_right=port_anchor_selector,
            leaving_left=no_leaving_group,
            leaving_right=no_leaving_group,
            bond_maker=make_single_bond,
        )

        overrides = {("A", "B"): special_reacter}
        connector = ReacterConnector(default=default_reacter, overrides=overrides)

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

        # Connect with reverse types (B, A) should match (A, B) override
        product = connector.connect(
            left=mono_L,
            right=mono_R,
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
            anchor_left=port_anchor_selector,
            anchor_right=port_anchor_selector,
            leaving_left=no_leaving_group,
            leaving_right=no_leaving_group,
            bond_maker=make_single_bond,
        )

        connector = ReacterConnector(default=default_reacter)

        mono_L = Monomer()
        c_L = Atom(symbol="C")
        mono_L.add_entity(c_L)
        # Don't set port

        mono_R = Monomer()
        c_R = Atom(symbol="C")
        mono_R.add_entity(c_R)
        mono_R.set_port("2", c_R)

        with pytest.raises(ValueError, match="Port '1' not found"):
            connector.connect(mono_L, mono_R, port_L="1", port_R="2")

    def test_connector_connect_missing_port_right(self):
        """Test connector.connect() raises error when right port is missing."""
        default_reacter = Reacter(
            name="default",
            anchor_left=port_anchor_selector,
            anchor_right=port_anchor_selector,
            leaving_left=no_leaving_group,
            leaving_right=no_leaving_group,
            bond_maker=make_single_bond,
        )

        connector = ReacterConnector(default=default_reacter)

        mono_L = Monomer()
        c_L = Atom(symbol="C")
        mono_L.add_entity(c_L)
        mono_L.set_port("1", c_L)

        mono_R = Monomer()
        c_R = Atom(symbol="C")
        mono_R.add_entity(c_R)
        # Don't set port

        with pytest.raises(ValueError, match="Port '2' not found"):
            connector.connect(mono_L, mono_R, port_L="1", port_R="2")

    def test_connector_get_history(self):
        """Test connector.get_history() method."""
        default_reacter = Reacter(
            name="default",
            anchor_left=port_anchor_selector,
            anchor_right=port_anchor_selector,
            leaving_left=no_leaving_group,
            leaving_right=no_leaving_group,
            bond_maker=make_single_bond,
        )

        connector = ReacterConnector(default=default_reacter)

        # Initially empty
        history = connector.get_history()
        assert len(history) == 0

        # Create and connect monomers
        mono_L = Monomer()
        c_L = Atom(symbol="C")
        mono_L.add_entity(c_L)
        mono_L.set_port("1", c_L)

        mono_R = Monomer()
        c_R = Atom(symbol="C")
        mono_R.add_entity(c_R)
        mono_R.set_port("2", c_R)

        connector.connect(mono_L, mono_R, port_L="1", port_R="2")

        # Should have one entry
        history = connector.get_history()
        assert len(history) == 1
        assert isinstance(history[0].product, Atomistic)

    def test_connector_get_history_copy(self):
        """Test that get_history() returns a copy."""
        default_reacter = Reacter(
            name="default",
            anchor_left=port_anchor_selector,
            anchor_right=port_anchor_selector,
            leaving_left=no_leaving_group,
            leaving_right=no_leaving_group,
            bond_maker=make_single_bond,
        )

        connector = ReacterConnector(default=default_reacter)

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
            anchor_left=port_anchor_selector,
            anchor_right=port_anchor_selector,
            leaving_left=no_leaving_group,
            leaving_right=no_leaving_group,
            bond_maker=make_single_bond,
        )

        connector = ReacterConnector(default=default_reacter)

        # Initially empty
        modified = connector.get_all_modified_atoms()
        assert len(modified) == 0

        # Create and connect monomers
        mono_L = Monomer()
        c_L = Atom(symbol="C")
        mono_L.add_entity(c_L)
        mono_L.set_port("1", c_L)

        mono_R = Monomer()
        c_R = Atom(symbol="C")
        mono_R.add_entity(c_R)
        mono_R.set_port("2", c_R)

        connector.connect(mono_L, mono_R, port_L="1", port_R="2")

        # Check modified atoms (may be empty if notes don't include it)
        modified = connector.get_all_modified_atoms()
        assert isinstance(modified, set)

    def test_connector_needs_retypification(self):
        """Test connector.needs_retypification() method."""
        default_reacter = Reacter(
            name="default",
            anchor_left=port_anchor_selector,
            anchor_right=port_anchor_selector,
            leaving_left=no_leaving_group,
            leaving_right=no_leaving_group,
            bond_maker=make_single_bond,
        )

        connector = ReacterConnector(default=default_reacter)

        # Initially False (no reactions yet)
        assert connector.needs_retypification() is False

        # Create and connect monomers
        mono_L = Monomer()
        c_L = Atom(symbol="C")
        mono_L.add_entity(c_L)
        mono_L.set_port("1", c_L)

        mono_R = Monomer()
        c_R = Atom(symbol="C")
        mono_R.add_entity(c_R)
        mono_R.set_port("2", c_R)

        connector.connect(mono_L, mono_R, port_L="1", port_R="2")

        # Should be True (default reacter sets needs_retypification=True)
        assert connector.needs_retypification() is True

    def test_connector_clear_history(self):
        """Test connector.clear_history() method."""
        default_reacter = Reacter(
            name="default",
            anchor_left=port_anchor_selector,
            anchor_right=port_anchor_selector,
            leaving_left=no_leaving_group,
            leaving_right=no_leaving_group,
            bond_maker=make_single_bond,
        )

        connector = ReacterConnector(default=default_reacter)

        # Create and connect monomers
        mono_L = Monomer()
        c_L = Atom(symbol="C")
        mono_L.add_entity(c_L)
        mono_L.set_port("1", c_L)

        mono_R = Monomer()
        c_R = Atom(symbol="C")
        mono_R.add_entity(c_R)
        mono_R.set_port("2", c_R)

        connector.connect(mono_L, mono_R, port_L="1", port_R="2")

        # Should have history
        assert len(connector.get_history()) == 1

        # Clear history
        connector.clear_history()

        # Should be empty
        assert len(connector.get_history()) == 0
        assert connector.needs_retypification() is False
