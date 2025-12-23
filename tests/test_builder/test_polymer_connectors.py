"""Tests for polymer connector classes."""

import pytest

from molpy import Atomistic
from molpy.builder.polymer.connectors import (
    AutoConnector,
    CallbackConnector,
    ChainConnector,
    ConnectorContext,
    ReacterConnector,
    TableConnector,
)
from molpy.builder.polymer.errors import AmbiguousPortsError, MissingConnectorRule
from molpy.builder.polymer.port_utils import PortInfo
from molpy.reacter import Reacter, form_single_bond


def create_monomer_with_ports(port_names, roles=None):
    """Helper to create a monomer with specified ports."""
    asm = Atomistic()
    atom = asm.def_atom(element="C")
    atom["port"] = port_names[0]
    if roles:
        atom[f"port_role_{port_names[0]}"] = roles[0]
    return asm


@pytest.fixture
def simple_context():
    """Create a simple connector context."""
    return ConnectorContext(
        step=0,
        sequence=["A", "B"],
        left_label="A",
        right_label="B",
        audit=[],
    )


class TestAutoConnector:
    """Tests for AutoConnector."""

    def test_select_ports_role_based_right_left(self, simple_context):
        """Test role-based selection: left.right -> right.left."""
        connector = AutoConnector()

        left_asm = create_monomer_with_ports([">"], roles=["right"])
        right_asm = create_monomer_with_ports(["<"], roles=["left"])

        left_ports = {">": [PortInfo(">", left_asm.atoms[0], role="right")]}
        right_ports = {"<": [PortInfo("<", right_asm.atoms[0], role="left")]}

        result = connector.select_ports(
            left_asm, right_asm, left_ports, right_ports, simple_context
        )

        assert result[0] == ">"  # left_port_name
        assert result[2] == "<"  # right_port_name

    def test_select_ports_role_based_left_right(self, simple_context):
        """Test role-based selection: left.left -> right.right."""
        connector = AutoConnector()

        left_asm = create_monomer_with_ports(["<"], roles=["left"])
        right_asm = create_monomer_with_ports([">"], roles=["right"])

        left_ports = {"<": [PortInfo("<", left_asm.atoms[0], role="left")]}
        right_ports = {">": [PortInfo(">", right_asm.atoms[0], role="right")]}

        result = connector.select_ports(
            left_asm, right_asm, left_ports, right_ports, simple_context
        )

        assert result[0] == "<"
        assert result[2] == ">"

    def test_select_ports_same_name(self, simple_context):
        """Test selection when both sides have same port name."""
        connector = AutoConnector()

        left_asm = create_monomer_with_ports(["$"])
        right_asm = create_monomer_with_ports(["$"])

        left_ports = {"$": [PortInfo("$", left_asm.atoms[0])]}
        right_ports = {"$": [PortInfo("$", right_asm.atoms[0])]}

        result = connector.select_ports(
            left_asm, right_asm, left_ports, right_ports, simple_context
        )

        assert result[0] == "$"
        assert result[2] == "$"

    def test_select_ports_ambiguous_raises(self, simple_context):
        """Test that ambiguous port selection raises error."""
        connector = AutoConnector()

        left_asm = Atomistic()
        left_atom1 = left_asm.def_atom(element="C")
        left_atom1["port"] = "port1"
        left_atom2 = left_asm.def_atom(element="C")
        left_atom2["port"] = "port2"

        right_asm = Atomistic()
        right_atom = right_asm.def_atom(element="C")
        right_atom["port"] = "port3"

        left_ports = {
            "port1": [PortInfo("port1", left_atom1)],
            "port2": [PortInfo("port2", left_atom2)],
        }
        right_ports = {
            "port3": [PortInfo("port3", right_atom)],
        }

        with pytest.raises(AmbiguousPortsError):
            connector.select_ports(
                left_asm, right_asm, left_ports, right_ports, simple_context
            )


class TestTableConnector:
    """Tests for TableConnector."""

    def test_select_ports_from_table(self, simple_context):
        """Test port selection using table lookup."""
        rules = {
            ("A", "B"): ("port1", "port2"),
        }
        connector = TableConnector(rules)

        left_asm = create_monomer_with_ports(["port1"])
        right_asm = create_monomer_with_ports(["port2"])

        left_ports = {"port1": [PortInfo("port1", left_asm.atoms[0])]}
        right_ports = {"port2": [PortInfo("port2", right_asm.atoms[0])]}

        result = connector.select_ports(
            left_asm, right_asm, left_ports, right_ports, simple_context
        )

        assert result[0] == "port1"
        assert result[2] == "port2"

    def test_select_ports_with_bond_kind(self, simple_context):
        """Test port selection with bond kind override."""
        from molpy.builder.polymer.connectors import BondKind

        rules = {
            ("A", "B"): ("port1", "port2", "="),  # type: ignore
        }
        connector = TableConnector(rules)

        left_asm = create_monomer_with_ports(["port1"])
        right_asm = create_monomer_with_ports(["port2"])

        left_ports = {"port1": [PortInfo("port1", left_asm.atoms[0])]}
        right_ports = {"port2": [PortInfo("port2", right_asm.atoms[0])]}

        result = connector.select_ports(
            left_asm, right_asm, left_ports, right_ports, simple_context
        )

        assert result[0] == "port1"
        assert result[2] == "port2"
        assert result[4] == "="

    def test_select_ports_with_fallback(self, simple_context):
        """Test port selection with fallback connector."""
        rules = {
            ("X", "Y"): ("portX", "portY"),
        }
        fallback = AutoConnector()
        connector = TableConnector(rules, fallback=fallback)

        left_asm = create_monomer_with_ports([">"], roles=["right"])
        right_asm = create_monomer_with_ports(["<"], roles=["left"])

        left_ports = {">": [PortInfo(">", left_asm.atoms[0], role="right")]}
        right_ports = {"<": [PortInfo("<", right_asm.atoms[0], role="left")]}

        # Should use fallback since ("A", "B") not in rules
        result = connector.select_ports(
            left_asm, right_asm, left_ports, right_ports, simple_context
        )

        assert result[0] == ">"
        assert result[2] == "<"

    def test_select_ports_missing_rule_raises(self, simple_context):
        """Test that missing rule raises error when no fallback."""
        rules = {
            ("X", "Y"): ("portX", "portY"),
        }
        connector = TableConnector(rules)

        left_asm = create_monomer_with_ports(["port1"])
        right_asm = create_monomer_with_ports(["port2"])

        left_ports = {"port1": [PortInfo("port1", left_asm.atoms[0])]}
        right_ports = {"port2": [PortInfo("port2", right_asm.atoms[0])]}

        with pytest.raises(MissingConnectorRule):
            connector.select_ports(
                left_asm, right_asm, left_ports, right_ports, simple_context
            )


class TestCallbackConnector:
    """Tests for CallbackConnector."""

    def test_select_ports_with_callback(self, simple_context):
        """Test port selection using callback function."""

        def my_selector(left, right, left_ports, right_ports, ctx):
            return ("port1", 0, "port2", 0)  # type: ignore

        connector = CallbackConnector(my_selector)  # type: ignore

        left_asm = create_monomer_with_ports(["port1"])
        right_asm = create_monomer_with_ports(["port2"])

        left_ports = {"port1": [PortInfo("port1", left_asm.atoms[0])]}
        right_ports = {"port2": [PortInfo("port2", right_asm.atoms[0])]}

        result = connector.select_ports(
            left_asm, right_asm, left_ports, right_ports, simple_context
        )

        assert result[0] == "port1"
        assert result[2] == "port2"

    def test_select_ports_with_bond_kind_callback(self, simple_context):
        """Test callback returning bond kind."""

        def my_selector(left, right, left_ports, right_ports, ctx):
            return ("port1", 0, "port2", 0, "=")  # type: ignore

        connector = CallbackConnector(my_selector)  # type: ignore

        left_asm = create_monomer_with_ports(["port1"])
        right_asm = create_monomer_with_ports(["port2"])

        left_ports = {"port1": [PortInfo("port1", left_asm.atoms[0])]}
        right_ports = {"port2": [PortInfo("port2", right_asm.atoms[0])]}

        result = connector.select_ports(
            left_asm, right_asm, left_ports, right_ports, simple_context
        )

        assert result[4] == "="


class TestChainConnector:
    """Tests for ChainConnector."""

    def test_select_ports_first_connector_succeeds(self, simple_context):
        """Test that first successful connector is used."""
        table_connector = TableConnector({("A", "B"): ("port1", "port2")})
        auto_connector = AutoConnector()

        connector = ChainConnector([table_connector, auto_connector])

        left_asm = create_monomer_with_ports(["port1"])
        right_asm = create_monomer_with_ports(["port2"])

        left_ports = {"port1": [PortInfo("port1", left_asm.atoms[0])]}
        right_ports = {"port2": [PortInfo("port2", right_asm.atoms[0])]}

        result = connector.select_ports(
            left_asm, right_asm, left_ports, right_ports, simple_context
        )

        assert result[0] == "port1"
        assert result[2] == "port2"

    def test_select_ports_falls_back_to_second(self, simple_context):
        """Test that second connector is tried if first fails."""
        table_connector = TableConnector({("X", "Y"): ("portX", "portY")})
        auto_connector = AutoConnector()

        connector = ChainConnector([table_connector, auto_connector])

        left_asm = create_monomer_with_ports([">"], roles=["right"])
        right_asm = create_monomer_with_ports(["<"], roles=["left"])

        left_ports = {">": [PortInfo(">", left_asm.atoms[0], role="right")]}
        right_ports = {"<": [PortInfo("<", right_asm.atoms[0], role="left")]}

        result = connector.select_ports(
            left_asm, right_asm, left_ports, right_ports, simple_context
        )

        assert result[0] == ">"
        assert result[2] == "<"

    def test_select_ports_all_fail_raises(self, simple_context):
        """Test that error is raised when all connectors fail."""
        table1 = TableConnector({("X", "Y"): ("portX", "portY")})
        table2 = TableConnector({("Z", "W"): ("portZ", "portW")})

        connector = ChainConnector([table1, table2])

        left_asm = create_monomer_with_ports(["port1"])
        right_asm = create_monomer_with_ports(["port2"])

        left_ports = {"port1": [PortInfo("port1", left_asm.atoms[0])]}
        right_ports = {"port2": [PortInfo("port2", right_asm.atoms[0])]}

        with pytest.raises(AmbiguousPortsError):
            connector.select_ports(
                left_asm, right_asm, left_ports, right_ports, simple_context
            )


class TestReacterConnector:
    """Tests for ReacterConnector."""

    def test_select_ports_from_port_map(self, simple_context):
        """Test port selection using port_map."""
        reacter = Reacter(
            name="test",
            anchor_selector_left=lambda a, port_atom: port_atom,
            anchor_selector_right=lambda a, port_atom: port_atom,
            leaving_selector_left=lambda a, anchor: [],
            leaving_selector_right=lambda a, anchor: [],
            bond_former=form_single_bond,
        )
        port_map = {("A", "B"): ("port1", "port2")}
        connector = ReacterConnector(default=reacter, port_map=port_map)

        left_asm = create_monomer_with_ports(["port1"])
        right_asm = create_monomer_with_ports(["port2"])

        left_ports = {"port1": [PortInfo("port1", left_asm.atoms[0])]}
        right_ports = {"port2": [PortInfo("port2", right_asm.atoms[0])]}

        result = connector.select_ports(
            left_asm, right_asm, left_ports, right_ports, simple_context
        )

        assert result[0] == "port1"
        assert result[2] == "port2"

    def test_select_ports_missing_mapping_raises(self, simple_context):
        """Test that missing port mapping raises error."""
        reacter = Reacter(
            name="test",
            anchor_selector_left=lambda a, port_atom: port_atom,
            anchor_selector_right=lambda a, port_atom: port_atom,
            leaving_selector_left=lambda a, anchor: [],
            leaving_selector_right=lambda a, anchor: [],
            bond_former=form_single_bond,
        )
        port_map = {("X", "Y"): ("portX", "portY")}
        connector = ReacterConnector(default=reacter, port_map=port_map)

        left_asm = create_monomer_with_ports(["port1"])
        right_asm = create_monomer_with_ports(["port2"])

        left_ports = {"port1": [PortInfo("port1", left_asm.atoms[0])]}
        right_ports = {"port2": [PortInfo("port2", right_asm.atoms[0])]}

        with pytest.raises(ValueError, match="No port mapping defined"):
            connector.select_ports(
                left_asm, right_asm, left_ports, right_ports, simple_context
            )

    def test_select_ports_invalid_port_raises(self, simple_context):
        """Test that invalid port name raises error."""
        reacter = Reacter(
            name="test",
            anchor_selector_left=lambda a, port_atom: port_atom,
            anchor_selector_right=lambda a, port_atom: port_atom,
            leaving_selector_left=lambda a, anchor: [],
            leaving_selector_right=lambda a, anchor: [],
            bond_former=form_single_bond,
        )
        port_map = {("A", "B"): ("port1", "port2")}
        connector = ReacterConnector(default=reacter, port_map=port_map)

        left_asm = create_monomer_with_ports(["port1"])
        right_asm = create_monomer_with_ports(["port3"])  # Wrong port name

        left_ports = {"port1": [PortInfo("port1", left_asm.atoms[0])]}
        right_ports = {"port3": [PortInfo("port3", right_asm.atoms[0])]}

        with pytest.raises(ValueError, match="not found in right structure"):
            connector.select_ports(
                left_asm, right_asm, left_ports, right_ports, simple_context
            )

    def test_get_reacter_default(self):
        """Test getting default reacter."""
        default_reacter = Reacter(
            name="default",
            anchor_selector_left=lambda a, port_atom: port_atom,
            anchor_selector_right=lambda a, port_atom: port_atom,
            leaving_selector_left=lambda a, anchor: [],
            leaving_selector_right=lambda a, anchor: [],
            bond_former=form_single_bond,
        )
        port_map = {("A", "B"): ("port1", "port2")}
        connector = ReacterConnector(default=default_reacter, port_map=port_map)

        reacter = connector.get_reacter("A", "B")
        assert reacter == default_reacter

    def test_get_reacter_override(self):
        """Test getting override reacter."""
        default_reacter = Reacter(
            name="default",
            anchor_selector_left=lambda a, port_atom: port_atom,
            anchor_selector_right=lambda a, port_atom: port_atom,
            leaving_selector_left=lambda a, anchor: [],
            leaving_selector_right=lambda a, anchor: [],
            bond_former=form_single_bond,
        )
        override_reacter = Reacter(
            name="override",
            anchor_selector_left=lambda a, port_atom: port_atom,
            anchor_selector_right=lambda a, port_atom: port_atom,
            leaving_selector_left=lambda a, anchor: [],
            leaving_selector_right=lambda a, anchor: [],
            bond_former=form_single_bond,
        )
        port_map = {("A", "B"): ("port1", "port2")}
        connector = ReacterConnector(
            default=default_reacter,
            port_map=port_map,
            overrides={("A", "B"): override_reacter},
        )

        reacter = connector.get_reacter("A", "B")
        assert reacter == override_reacter
