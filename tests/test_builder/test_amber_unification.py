"""Tests for Amber builder unification with the standard polymer API.

Tests verify:
- AmberPolymerBuilderConfig accepts reaction_preset
- AmberPolymerBuilder uses port utilities for validation
- AmberPolymerBuilder uses preset leaving selectors
- BuildPolymerAmber Compute class basics
- DSL backend="amber" dispatch

Note: Tests that require actual AmberTools executables are marked
@pytest.mark.external.
"""

import pytest

from molpy.core.atomistic import Atom, Atomistic, Bond


# ---- Config tests ----


class TestAmberPolymerBuilderInit:
    def test_default_no_preset(self):
        from molpy.builder.polymer.ambertools import AmberPolymerBuilder

        builder = AmberPolymerBuilder(library={})
        assert builder.reaction_preset is None
        assert builder.force_field == "gaff2"
        assert builder.charge_method == "bcc"

    def test_with_preset(self):
        from molpy.builder.polymer.ambertools import AmberPolymerBuilder

        builder = AmberPolymerBuilder(library={}, reaction_preset="dehydration")
        assert builder.reaction_preset == "dehydration"

    def test_with_condensation_preset(self):
        from molpy.builder.polymer.ambertools import AmberPolymerBuilder

        builder = AmberPolymerBuilder(library={}, reaction_preset="condensation")
        assert builder.reaction_preset == "condensation"


# ---- Port utility integration tests ----


def _make_test_monomer() -> Atomistic:
    """Create a minimal monomer with ports for testing."""
    monomer = Atomistic()
    c1 = Atom(symbol="C", element="C", name="C1", port="<")
    c2 = Atom(symbol="C", element="C", name="C2")
    o1 = Atom(symbol="O", element="O", name="O1", port=">")
    h1 = Atom(symbol="H", element="H", name="H1")
    h2 = Atom(symbol="H", element="H", name="H2")

    monomer.add_entity(c1)
    monomer.add_entity(c2)
    monomer.add_entity(o1)
    monomer.add_entity(h1)
    monomer.add_entity(h2)

    monomer.add_link(Bond(c1, c2, order=1), include_endpoints=False)
    monomer.add_link(Bond(c2, o1, order=1), include_endpoints=False)
    monomer.add_link(Bond(c1, h1, order=1), include_endpoints=False)
    monomer.add_link(Bond(o1, h2, order=1), include_endpoints=False)

    return monomer


class TestAmberBuilderPortUtilities:
    def test_validate_ir_uses_port_utils(self):
        """AmberPolymerBuilder._validate_ir should use get_all_ports."""
        from molpy.builder.polymer.ambertools import AmberPolymerBuilder
        from molpy.parser.smiles import parse_cgsmiles

        monomer = _make_test_monomer()
        builder = AmberPolymerBuilder(library={"M": monomer})
        ir = parse_cgsmiles("{[#M]|3}")

        # Should not raise
        builder._validate_ir(ir)

    def test_validate_ir_missing_port_raises(self):
        """Should raise if monomer lacks port annotations."""
        from molpy.builder.polymer.ambertools import AmberPolymerBuilder
        from molpy.parser.smiles import parse_cgsmiles

        # Monomer without ports
        monomer = Atomistic()
        monomer.add_entity(Atom(symbol="C", element="C", name="C1"))

        builder = AmberPolymerBuilder(library={"M": monomer})
        ir = parse_cgsmiles("{[#M]}")

        with pytest.raises(ValueError, match="port annotations"):
            builder._validate_ir(ir)


# ---- Leaving group detection tests ----


class TestAmberBuilderLeavingGroups:
    def test_omit_hydrogens_fallback(self):
        """Without preset, auto-detects H atoms as leaving groups."""
        from molpy.builder.polymer.ambertools.amber_builder import AmberPolymerBuilder

        monomer = _make_test_monomer()
        builder = AmberPolymerBuilder(library={"M": monomer}, reaction_preset=None)

        # Port "<" is on C1, which has H1 bonded
        omit = builder._get_omit_names(monomer, "<")
        assert "H1" in omit

        # Port ">" is on O1, which has H2 bonded
        omit = builder._get_omit_names(monomer, ">")
        assert "H2" in omit

    def test_omit_from_preset(self):
        """With preset, uses preset leaving selectors."""
        from molpy.builder.polymer.ambertools.amber_builder import AmberPolymerBuilder

        monomer = _make_test_monomer()
        builder = AmberPolymerBuilder(
            library={"M": monomer}, reaction_preset="dehydration"
        )

        # Dehydration: select_self as site, select_hydrogens(1) as leaving
        # Port "<" on C1 → site=C1, leaving=H1
        omit = builder._get_omit_names(monomer, "<")
        assert "H1" in omit

    def test_omit_no_port_returns_empty(self):
        """If port atom not found, return empty list."""
        from molpy.builder.polymer.ambertools.amber_builder import AmberPolymerBuilder

        monomer = Atomistic()
        monomer.add_entity(Atom(symbol="C", element="C", name="C1"))

        builder = AmberPolymerBuilder(library={"M": monomer})
        omit = builder._get_omit_names(monomer, "<")
        assert omit == []


# ---- BuildPolymerAmber compute tests ----


class TestBuildPolymerAmber:
    def test_is_tool_subclass(self):
        from molpy.tool.polymer import BuildPolymerAmber
        from molpy.tool.base import Tool

        assert issubclass(BuildPolymerAmber, Tool)

    def test_not_compute_subclass(self):
        from molpy.tool.polymer import BuildPolymerAmber
        from molpy.tool.base import Compute

        assert not issubclass(BuildPolymerAmber, Compute)

    def test_registered_in_registry(self):
        from molpy.tool.polymer import BuildPolymerAmber
        from molpy.tool.base import ToolRegistry

        assert ToolRegistry.get("BuildPolymerAmber") is BuildPolymerAmber

    def test_frozen(self):
        from molpy.tool.polymer import BuildPolymerAmber

        tool = BuildPolymerAmber()
        with pytest.raises(AttributeError):
            tool.force_field = "gaff"

    def test_defaults(self):
        from molpy.tool.polymer import BuildPolymerAmber

        tool = BuildPolymerAmber()
        assert tool.reaction_preset == "dehydration"
        assert tool.force_field == "gaff2"
        assert tool.charge_method == "bcc"


# ---- DSL backend dispatch tests ----


class TestDSLBackendDispatch:
    def test_amber_backend_requires_library(self):
        from molpy.tool.polymer import polymer

        with pytest.raises(TypeError, match="requires 'library'"):
            polymer("{[#EO]|10}", backend="amber")

    def test_default_backend_unchanged(self):
        """Verify default backend still works as before."""
        from molpy.tool.polymer import _detect_notation

        # These should still work regardless of backend param
        assert _detect_notation("{[#EO]|10}") == "cgsmiles"
        assert _detect_notation("{[<]CCO[>]}|10|") == "gbigsmiles"


# ---- Exports tests ----


class TestAmberExports:
    def test_builder_init_exports(self):
        """builder/__init__.py should export the ambertools version."""
        from molpy.builder import AmberPolymerBuilder

        assert AmberPolymerBuilder is not None

    def test_tool_exports_amber(self):
        """molpy.tool should export BuildPolymerAmber."""
        from molpy.tool import BuildPolymerAmber

        assert BuildPolymerAmber is not None

    def test_no_old_amber_in_polymer_all(self):
        """Old AmberPolymerBuilder should NOT be in polymer.__all__."""
        from molpy.builder.polymer import __all__

        assert "AmberPolymerBuilder" not in __all__
