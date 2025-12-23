#!/usr/bin/env python3
"""Unit tests for DependencyAnalyzer class.

Tests cover:
- DependencyAnalyzer initialization
- _build_dependency_graph
- _compute_levels
- _detect_circular_groups
- get_patterns_by_level
- get_max_level
- has_circular_dependencies
"""

from molpy.parser.smarts import SmartsParser
from molpy.typifier.dependency_analyzer import DependencyAnalyzer
from molpy.typifier.graph import SMARTSGraph


class TestDependencyAnalyzer:
    """Test DependencyAnalyzer class."""

    def test_dependency_analyzer_initialization(self):
        """Test DependencyAnalyzer initialization."""
        parser = SmartsParser()
        patterns = {
            "opls_135": SMARTSGraph(
                smarts_string="[C]", parser=parser, atomtype_name="opls_135", priority=0
            ),
            "opls_136": SMARTSGraph(
                smarts_string="[C]", parser=parser, atomtype_name="opls_136", priority=0
            ),
        }

        analyzer = DependencyAnalyzer(patterns)

        assert analyzer.patterns == patterns
        assert hasattr(analyzer, "dependency_graph")
        assert hasattr(analyzer, "levels")
        assert hasattr(analyzer, "circular_groups")

    def test_dependency_analyzer_no_dependencies(self):
        """Test analyzer with patterns that have no dependencies."""
        parser = SmartsParser()
        patterns = {
            "opls_135": SMARTSGraph(
                smarts_string="[C]", parser=parser, atomtype_name="opls_135", priority=0
            ),
            "opls_136": SMARTSGraph(
                smarts_string="[C]", parser=parser, atomtype_name="opls_136", priority=0
            ),
        }

        analyzer = DependencyAnalyzer(patterns)

        # All should be at level 0
        assert analyzer.levels["opls_135"] == 0
        assert analyzer.levels["opls_136"] == 0
        assert analyzer.has_circular_dependencies() is False

    def test_dependency_analyzer_with_dependencies(self):
        """Test analyzer with patterns that have dependencies."""
        parser = SmartsParser()

        # Create pattern that depends on another
        pattern1 = SMARTSGraph(
            smarts_string="[C]", parser=parser, atomtype_name="opls_135", priority=0
        )
        pattern1.dependencies = set()  # No dependencies

        # Pattern that depends on opls_135
        pattern2 = SMARTSGraph(
            smarts_string="[C;%opls_135]",
            parser=parser,
            atomtype_name="opls_136",
            priority=0,
        )
        pattern2.dependencies = {"opls_135"}

        patterns = {
            "opls_135": pattern1,
            "opls_136": pattern2,
        }

        analyzer = DependencyAnalyzer(patterns)

        # opls_135 should be at level 0, opls_136 at level 1
        assert analyzer.levels["opls_135"] == 0
        assert analyzer.levels["opls_136"] == 1
        assert analyzer.has_circular_dependencies() is False

    def test_dependency_analyzer_get_patterns_by_level(self):
        """Test get_patterns_by_level method."""
        parser = SmartsParser()
        patterns = {
            "opls_135": SMARTSGraph(
                smarts_string="[C]", parser=parser, atomtype_name="opls_135", priority=0
            ),
            "opls_136": SMARTSGraph(
                smarts_string="[C]", parser=parser, atomtype_name="opls_136", priority=0
            ),
        }

        analyzer = DependencyAnalyzer(patterns)

        level_0 = analyzer.get_patterns_by_level(0)
        assert len(level_0) == 2
        assert all(p.atomtype_name in ["opls_135", "opls_136"] for p in level_0)

    def test_dependency_analyzer_get_max_level(self):
        """Test get_max_level method."""
        parser = SmartsParser()
        patterns = {
            "opls_135": SMARTSGraph(
                smarts_string="[C]", parser=parser, atomtype_name="opls_135", priority=0
            ),
        }

        analyzer = DependencyAnalyzer(patterns)

        assert analyzer.get_max_level() == 0

    def test_dependency_analyzer_empty_patterns(self):
        """Test analyzer with empty patterns."""
        patterns = {}

        analyzer = DependencyAnalyzer(patterns)

        assert analyzer.get_max_level() == -1
        assert analyzer.has_circular_dependencies() is False

    def test_dependency_analyzer_circular_dependencies(self):
        """Test analyzer detects circular dependencies."""
        parser = SmartsParser()

        # Create circular dependency: A depends on B, B depends on A
        pattern_a = SMARTSGraph(
            smarts_string="[C]", parser=parser, atomtype_name="A", priority=0
        )
        pattern_a.dependencies = {"B"}

        pattern_b = SMARTSGraph(
            smarts_string="[C]", parser=parser, atomtype_name="B", priority=0
        )
        pattern_b.dependencies = {"A"}

        patterns = {
            "A": pattern_a,
            "B": pattern_b,
        }

        analyzer = DependencyAnalyzer(patterns)

        # Should detect circular dependency
        assert analyzer.has_circular_dependencies() is True
        assert len(analyzer.circular_groups) > 0

    def test_dependency_analyzer_multiple_levels(self):
        """Test analyzer with multiple dependency levels."""
        parser = SmartsParser()

        # Level 0: no dependencies
        pattern0 = SMARTSGraph(
            smarts_string="[C]", parser=parser, atomtype_name="opls_135", priority=0
        )
        pattern0.dependencies = set()

        # Level 1: depends on level 0
        pattern1 = SMARTSGraph(
            smarts_string="[C]", parser=parser, atomtype_name="opls_136", priority=0
        )
        pattern1.dependencies = {"opls_135"}

        # Level 2: depends on level 1
        pattern2 = SMARTSGraph(
            smarts_string="[C]", parser=parser, atomtype_name="opls_137", priority=0
        )
        pattern2.dependencies = {"opls_136"}

        patterns = {
            "opls_135": pattern0,
            "opls_136": pattern1,
            "opls_137": pattern2,
        }

        analyzer = DependencyAnalyzer(patterns)

        assert analyzer.levels["opls_135"] == 0
        assert analyzer.levels["opls_136"] == 1
        assert analyzer.levels["opls_137"] == 2
        assert analyzer.get_max_level() == 2
