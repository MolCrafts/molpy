#!/usr/bin/env python3
"""Unit tests for LayeredTypingEngine class.

Tests cover:
- LayeredTypingEngine initialization
- typify method
- _resolve_level method
- _resolve_circular method
- get_explain_data method
"""

from molpy.parser.smarts import SmartsParser
from molpy.typifier.graph import SMARTSGraph
from molpy.typifier.layered_engine import LayeredTypingEngine


class TestLayeredTypingEngine:
    """Test LayeredTypingEngine class."""

    def test_layered_engine_initialization(self):
        """Test LayeredTypingEngine initialization."""
        parser = SmartsParser()
        patterns = {
            "opls_135": SMARTSGraph(
                smarts_string="[C]", parser=parser, atomtype_name="opls_135", priority=0
            ),
        }

        engine = LayeredTypingEngine(patterns)

        assert engine.patterns == patterns
        assert hasattr(engine, "analyzer")
        assert hasattr(engine, "matcher")

    def test_layered_engine_typify_simple(self):
        """Test typify with simple patterns."""
        parser = SmartsParser()
        patterns = {
            "opls_135": SMARTSGraph(
                smarts_string="[C]", parser=parser, atomtype_name="opls_135", priority=0
            ),
        }

        engine = LayeredTypingEngine(patterns)

        # Create simple molecule graph
        from molpy import Atom, Atomistic
        from molpy.typifier.adapter import build_mol_graph

        asm = Atomistic()
        c = Atom(symbol="C")
        asm.add_entity(c)

        mol_graph, vs_to_atomid, _atomid_to_vs = build_mol_graph(asm)

        result = engine.typify(mol_graph, vs_to_atomid)

        assert isinstance(result, dict)
        # Should assign type to carbon
        assert len(result) > 0

    def test_layered_engine_typify_with_dependencies(self):
        """Test typify with dependent patterns."""
        parser = SmartsParser()

        # Level 0 pattern
        pattern0 = SMARTSGraph(
            smarts_string="[C]", parser=parser, atomtype_name="opls_135", priority=0
        )
        pattern0.dependencies = set()

        # Level 1 pattern (depends on opls_135)
        pattern1 = SMARTSGraph(
            smarts_string="[C]", parser=parser, atomtype_name="opls_136", priority=0
        )
        pattern1.dependencies = {"opls_135"}

        patterns = {
            "opls_135": pattern0,
            "opls_136": pattern1,
        }

        engine = LayeredTypingEngine(patterns)

        # Create molecule graph
        from molpy import Atom, Atomistic
        from molpy.typifier.adapter import build_mol_graph

        asm = Atomistic()
        c = Atom(symbol="C")
        asm.add_entity(c)

        mol_graph, vs_to_atomid, _atomid_to_vs = build_mol_graph(asm)

        result = engine.typify(mol_graph, vs_to_atomid)

        assert isinstance(result, dict)

    def test_layered_engine_get_explain_data(self):
        """Test get_explain_data method."""
        parser = SmartsParser()
        patterns = {
            "opls_135": SMARTSGraph(
                smarts_string="[C]", parser=parser, atomtype_name="opls_135", priority=0
            ),
        }

        engine = LayeredTypingEngine(patterns)

        # Create molecule graph
        from molpy import Atom, Atomistic
        from molpy.typifier.adapter import build_mol_graph

        asm = Atomistic()
        c = Atom(symbol="C")
        asm.add_entity(c)

        mol_graph, vs_to_atomid, _atomid_to_vs = build_mol_graph(asm)

        explain_data = engine.get_explain_data(mol_graph, vs_to_atomid)

        assert isinstance(explain_data, dict)
        assert "levels" in explain_data
        assert "circular_groups" in explain_data
        assert "final_assignments" in explain_data

    def test_layered_engine_empty_patterns(self):
        """Test engine with empty patterns."""
        patterns = {}

        engine = LayeredTypingEngine(patterns)

        # Create molecule graph
        from molpy import Atom, Atomistic
        from molpy.typifier.adapter import build_mol_graph

        asm = Atomistic()
        c = Atom(symbol="C")
        asm.add_entity(c)

        mol_graph, vs_to_atomid, _atomid_to_vs = build_mol_graph(asm)

        result = engine.typify(mol_graph, vs_to_atomid)

        # Should return empty dict
        assert isinstance(result, dict)
        assert len(result) == 0
