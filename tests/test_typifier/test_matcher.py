#!/usr/bin/env python3
"""Unit tests for matcher classes and functions.

Tests cover:
- Candidate dataclass
- ScoringPolicy
- SmartsMatcher class
"""

from molpy.parser.smarts import SmartsParser
from molpy.typifier.graph import SMARTSGraph
from molpy.typifier.matcher import (
    Candidate,
    ScoringPolicy,
    SmartsMatcher,
)


class TestCandidate:
    """Test Candidate dataclass."""

    def test_candidate_creation(self):
        """Test creating a Candidate."""
        cand = Candidate(
            atom_id=1,
            atomtype="opls_135",
            source="test",
            priority=0,
            score=5,
            pattern_size=(2, 1),
            definition_order=0,
        )

        assert cand.atom_id == 1
        assert cand.atomtype == "opls_135"
        assert cand.priority == 0
        assert cand.score == 5

    def test_candidate_sorting(self):
        """Test Candidate sorting by sort_key."""
        cand1 = Candidate(
            atom_id=1,
            atomtype="opls_135",
            source="test",
            priority=1,  # Higher priority
            score=5,
            pattern_size=(2, 1),
            definition_order=0,
        )

        cand2 = Candidate(
            atom_id=1,
            atomtype="opls_136",
            source="test",
            priority=0,  # Lower priority
            score=10,  # Higher score, but priority wins
            pattern_size=(3, 2),
            definition_order=1,
        )

        # cand1 should sort before cand2 (higher priority)
        assert cand1 < cand2

    def test_candidate_repr(self):
        """Test Candidate string representation."""
        cand = Candidate(
            atom_id=1,
            atomtype="opls_135",
            source="test",
            priority=0,
            score=5,
            pattern_size=(2, 1),
            definition_order=0,
        )

        repr_str = repr(cand)
        assert "Candidate" in repr_str
        assert "opls_135" in repr_str


class TestScoringPolicy:
    """Test ScoringPolicy class."""

    def test_scoring_policy_default(self):
        """Test default scoring policy."""
        parser = SmartsParser()
        pattern = SMARTSGraph(
            smarts_string="[C]", parser=parser, atomtype_name="opls_135", priority=0
        )

        score = ScoringPolicy.default(pattern)

        # Score should be based on pattern size
        assert isinstance(score, int)
        assert score >= 0

    def test_scoring_policy_custom(self):
        """Test custom scoring policy."""
        parser = SmartsParser()
        pattern = SMARTSGraph(
            smarts_string="[C]", parser=parser, atomtype_name="opls_135", priority=0
        )

        score = ScoringPolicy.custom(pattern, vertex_weight=1.0, edge_weight=1.5)

        assert isinstance(score, int)
        assert score >= 0


class TestSmartsMatcher:
    """Test SmartsMatcher class."""

    def test_smarts_matcher_initialization(self):
        """Test SmartsMatcher initialization."""
        parser = SmartsParser()
        patterns = [
            SMARTSGraph(
                smarts_string="[C]", parser=parser, atomtype_name="opls_135", priority=0
            ),
        ]

        matcher = SmartsMatcher(patterns)

        assert matcher.patterns == patterns
        assert hasattr(matcher, "scoring")

    def test_smarts_matcher_find_candidates_simple(self):
        """Test finding candidates in simple molecule."""
        parser = SmartsParser()
        patterns = [
            SMARTSGraph(
                smarts_string="[C]", parser=parser, atomtype_name="opls_135", priority=0
            ),
        ]

        matcher = SmartsMatcher(patterns)

        # Create simple molecule graph
        from molpy import Atom, Atomistic
        from molpy.typifier.adapter import build_mol_graph

        asm = Atomistic()
        c = Atom(symbol="C")
        asm.add_entity(c)

        mol_graph, vs_to_atomid, _atomid_to_vs = build_mol_graph(asm)

        candidates = matcher.find_candidates(mol_graph, vs_to_atomid)

        assert len(candidates) > 0
        assert all(isinstance(c, Candidate) for c in candidates)

    def test_smarts_matcher_resolve(self):
        """Test resolving candidates."""
        parser = SmartsParser()
        patterns = [
            SMARTSGraph(
                smarts_string="[C]", parser=parser, atomtype_name="opls_135", priority=0
            ),
        ]

        matcher = SmartsMatcher(patterns)

        # Create candidates
        candidates = [
            Candidate(
                atom_id=1,
                atomtype="opls_135",
                source="test",
                priority=0,
                score=5,
                pattern_size=(1, 0),
                definition_order=0,
            ),
            Candidate(
                atom_id=1,
                atomtype="opls_136",
                source="test",
                priority=1,  # Higher priority
                score=3,
                pattern_size=(1, 0),
                definition_order=1,
            ),
        ]

        result = matcher.resolve(candidates)

        # Should pick higher priority
        assert result[1] == "opls_136"

    def test_smarts_matcher_resolve_multiple_atoms(self):
        """Test resolving candidates for multiple atoms."""
        parser = SmartsParser()
        patterns = [
            SMARTSGraph(
                smarts_string="[C]", parser=parser, atomtype_name="opls_135", priority=0
            ),
        ]

        matcher = SmartsMatcher(patterns)

        # Create candidates for different atoms
        candidates = [
            Candidate(
                atom_id=1,
                atomtype="opls_135",
                source="test",
                priority=0,
                score=5,
                pattern_size=(1, 0),
                definition_order=0,
            ),
            Candidate(
                atom_id=2,
                atomtype="opls_136",
                source="test",
                priority=0,
                score=5,
                pattern_size=(1, 0),
                definition_order=0,
            ),
        ]

        result = matcher.resolve(candidates)

        assert len(result) == 2
        assert result[1] == "opls_135"
        assert result[2] == "opls_136"

    def test_smarts_matcher_explain(self):
        """Test explain method."""
        parser = SmartsParser()
        patterns = [
            SMARTSGraph(
                smarts_string="[C]", parser=parser, atomtype_name="opls_135", priority=0
            ),
        ]

        matcher = SmartsMatcher(patterns)

        candidates = [
            Candidate(
                atom_id=1,
                atomtype="opls_135",
                source="test",
                priority=0,
                score=5,
                pattern_size=(1, 0),
                definition_order=0,
            ),
        ]

        explain_data = matcher.explain(candidates)

        assert 1 in explain_data
        assert "winner" in explain_data[1]
        assert "candidates" in explain_data[1]
        assert explain_data[1]["winner"] == "opls_135"
