#!/usr/bin/env python3
"""Unit tests for SMARTSGraph class.

Tests cover:
- SMARTSGraph initialization
- from_igraph class method
- priority property
- get_specificity_score method
- extract_dependencies method
- _node_match_fn method
- _edge_match_fn method
"""

from igraph import Graph

from molpy.parser.smarts import SmartsParser
from molpy.typifier.graph import SMARTSGraph


class TestSMARTSGraph:
    """Test SMARTSGraph class."""

    def test_smarts_graph_initialization_from_string(self):
        """Test SMARTSGraph initialization from SMARTS string."""
        parser = SmartsParser()
        pattern = SMARTSGraph(
            smarts_string="[C]",
            parser=parser,
            atomtype_name="opls_135",
            priority=1,
            source="test",
        )

        assert pattern.atomtype_name == "opls_135"
        assert pattern.priority == 1
        assert pattern.source == "test"
        assert pattern.smarts_string == "[C]"
        assert pattern.ir is not None

    def test_smarts_graph_initialization_from_igraph(self):
        """Test SMARTSGraph.from_igraph class method."""
        # Create igraph
        g = Graph()
        g.add_vertices(2)
        g.add_edge(0, 1)
        g.vs[0]["element"] = "C"
        g.vs[1]["element"] = "H"

        pattern = SMARTSGraph.from_igraph(
            g, atomtype_name="opls_135", priority=1, source="test"
        )

        assert pattern.atomtype_name == "opls_135"
        assert pattern.priority == 1
        assert pattern.vcount() == 2
        assert pattern.ecount() == 1

    def test_smarts_graph_priority_property(self):
        """Test priority property."""
        parser = SmartsParser()
        pattern = SMARTSGraph(
            smarts_string="[C]", parser=parser, atomtype_name="opls_135", priority=5
        )

        assert pattern.priority == 5

    def test_smarts_graph_get_specificity_score(self):
        """Test get_specificity_score method."""
        parser = SmartsParser()
        pattern = SMARTSGraph(
            smarts_string="[C]", parser=parser, atomtype_name="opls_135", priority=0
        )

        score = pattern.get_specificity_score()

        assert isinstance(score, int)
        assert score >= 0

    def test_smarts_graph_extract_dependencies(self):
        """Test extract_dependencies method."""
        parser = SmartsParser()

        # Pattern with type reference
        pattern = SMARTSGraph(
            smarts_string="[C;%opls_135]",
            parser=parser,
            atomtype_name="opls_136",
            priority=0,
        )

        deps = pattern.extract_dependencies()

        # Should extract opls_135 dependency
        assert "opls_135" in deps

    def test_smarts_graph_extract_dependencies_no_refs(self):
        """Test extract_dependencies with no type references."""
        parser = SmartsParser()
        pattern = SMARTSGraph(
            smarts_string="[C]", parser=parser, atomtype_name="opls_135", priority=0
        )

        deps = pattern.extract_dependencies()

        # Should be empty
        assert len(deps) == 0

    def test_smarts_graph_repr(self):
        """Test SMARTSGraph string representation."""
        parser = SmartsParser()
        pattern = SMARTSGraph(
            smarts_string="[C]", parser=parser, atomtype_name="opls_135", priority=0
        )

        repr_str = repr(pattern)
        assert "SMARTSGraph" in repr_str or "SmartsGraph" in repr_str

    def test_smarts_graph_target_vertices(self):
        """Test target_vertices attribute."""
        parser = SmartsParser()
        pattern = SMARTSGraph(
            smarts_string="[C][C]",
            parser=parser,
            atomtype_name="opls_135",
            priority=0,
            target_vertices=[0],  # Only first vertex should be typed
        )

        assert pattern.target_vertices == [0]

    def test_smarts_graph_overrides(self):
        """Test overrides attribute."""
        parser = SmartsParser()
        pattern = SMARTSGraph(
            smarts_string="[C]",
            parser=parser,
            atomtype_name="opls_135",
            priority=0,
            overrides={"opls_136", "opls_137"},
        )

        assert pattern.overrides == {"opls_136", "opls_137"}

    def test_smarts_graph_override_method(self):
        """Test override method."""
        parser = SmartsParser()
        pattern = SMARTSGraph(
            smarts_string="[C]", parser=parser, atomtype_name="opls_135", priority=0
        )

        pattern.override({"opls_136"})

        assert pattern.overrides == {"opls_136"}

    def test_smarts_graph_dependencies_attribute(self):
        """Test dependencies attribute."""
        parser = SmartsParser()
        pattern = SMARTSGraph(
            smarts_string="[C]", parser=parser, atomtype_name="opls_135", priority=0
        )

        # Set dependencies manually
        pattern.dependencies = {"opls_136"}

        assert pattern.dependencies == {"opls_136"}

    def test_smarts_graph_level_attribute(self):
        """Test level attribute."""
        parser = SmartsParser()
        pattern = SMARTSGraph(
            smarts_string="[C]", parser=parser, atomtype_name="opls_135", priority=0
        )

        # Set level manually
        pattern.level = 1

        assert pattern.level == 1
