"""Tests for the unified Connector class."""

import pytest

from molpy import Atomistic
from molpy.builder.polymer.connectors import Connector, ConnectorContext
from molpy.builder.polymer.errors import AmbiguousPortsError
from molpy.core.atomistic import Atom
from molpy.reacter import Reacter
from molpy.reacter.utils import form_single_bond


def _make_reacter():
    return Reacter(
        name="test",
        anchor_selector_left=lambda a, port_atom: port_atom,
        anchor_selector_right=lambda a, port_atom: port_atom,
        leaving_selector_left=lambda a, anchor: [],
        leaving_selector_right=lambda a, anchor: [],
        bond_former=form_single_bond,
    )


def _make_port_atom(port_name: str) -> Atom:
    a = Atom(symbol="C")
    a["port"] = port_name
    return a


@pytest.fixture
def reacter():
    return _make_reacter()


@pytest.fixture
def ctx():
    return ConnectorContext(step=0, left_label="A", right_label="B")


class TestConnectorInit:
    def test_default_empty_port_map(self, reacter):
        c = Connector(reacter=reacter)
        assert c.port_map == {}
        assert c.overrides == {}

    def test_with_port_map(self, reacter):
        pm = {("A", "B"): (">", "<")}
        c = Connector(reacter=reacter, port_map=pm)
        assert c.port_map == pm

    def test_with_overrides(self, reacter):
        other = _make_reacter()
        c = Connector(reacter=reacter, overrides={("X", "Y"): other})
        assert c.get_reacter("X", "Y") is other
        assert c.get_reacter("A", "B") is reacter


class TestSelectPortsExplicit:
    def test_explicit_port_map(self, reacter, ctx):
        c = Connector(reacter=reacter, port_map={("A", "B"): (">", "<")})
        left_ports = {">": [_make_port_atom(">")]}
        right_ports = {"<": [_make_port_atom("<")]}
        result = c.select_ports(Atomistic(), Atomistic(), left_ports, right_ports, ctx)
        assert result[0] == ">"
        assert result[2] == "<"

    def test_explicit_port_not_found_raises(self, reacter, ctx):
        c = Connector(reacter=reacter, port_map={("A", "B"): (">", "<")})
        left_ports = {"x": [_make_port_atom("x")]}
        right_ports = {"<": [_make_port_atom("<")]}
        with pytest.raises(AmbiguousPortsError):
            c.select_ports(Atomistic(), Atomistic(), left_ports, right_ports, ctx)


class TestSelectPortsCompatibility:
    def test_directional_gt_lt(self, reacter, ctx):
        """Left's > connects to right's <."""
        c = Connector(reacter=reacter)
        left_ports = {">": [_make_port_atom(">")]}
        right_ports = {"<": [_make_port_atom("<")]}
        result = c.select_ports(Atomistic(), Atomistic(), left_ports, right_ports, ctx)
        assert result[0] == ">"
        assert result[2] == "<"

    def test_both_ports_selects_correct_direction(self, reacter, ctx):
        """When both sides have < and >, selects > on left, < on right."""
        c = Connector(reacter=reacter)
        left_ports = {
            "<": [_make_port_atom("<")],
            ">": [_make_port_atom(">")],
        }
        right_ports = {
            "<": [_make_port_atom("<")],
            ">": [_make_port_atom(">")],
        }
        result = c.select_ports(Atomistic(), Atomistic(), left_ports, right_ports, ctx)
        assert result[0] == ">"
        assert result[2] == "<"


class TestSelectPortsSinglePort:
    def test_single_port_each_side(self, reacter, ctx):
        c = Connector(reacter=reacter)
        left_ports = {"$": [_make_port_atom("$")]}
        right_ports = {"$": [_make_port_atom("$")]}
        result = c.select_ports(Atomistic(), Atomistic(), left_ports, right_ports, ctx)
        assert result[0] == "$"
        assert result[2] == "$"


class TestSelectPortsAmbiguous:
    def test_ambiguous_raises(self, reacter, ctx):
        c = Connector(reacter=reacter)
        left_ports = {
            "a": [_make_port_atom("a")],
            "b": [_make_port_atom("b")],
        }
        right_ports = {
            "x": [_make_port_atom("x")],
            "y": [_make_port_atom("y")],
        }
        with pytest.raises(AmbiguousPortsError):
            c.select_ports(Atomistic(), Atomistic(), left_ports, right_ports, ctx)
