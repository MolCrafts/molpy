"""SMARTS matcher for atomtyping with conflict resolution.

This module provides the main SmartsMatcher class that:
1. Finds all candidate atom type assignments from SMARTS patterns
2. Resolves conflicts using deterministic priority rules
3. Provides explain functionality for debugging
"""

from dataclasses import dataclass, field
from typing import Any

from igraph import Graph

from .graph import SMARTSGraph

# ===================================================================
#                     Candidate Data Class
# ===================================================================


@dataclass(order=True)
class Candidate:
    """Represents a candidate atom type assignment.

    The ordering is designed for conflict resolution:
    1. Higher priority wins
    2. Higher specificity score wins
    3. Larger pattern size wins
    4. Later definition order wins
    """

    # Sorting key (first field determines primary sort order)
    sort_key: tuple[int, int, int, int] = field(init=False, repr=False)

    # Data fields
    atom_id: int
    atomtype: str
    source: str
    priority: int
    score: int  # specificity score
    pattern_size: tuple[int, int]  # (n_vertices, n_edges)
    definition_order: int  # Index in pattern list

    def __post_init__(self):
        """Compute sort key for deterministic ordering."""
        # Higher values win, so negate for descending sort
        self.sort_key = (
            -self.priority,  # Higher priority first (negate to reverse)
            -self.score,  # Higher score first
            -(self.pattern_size[0] + self.pattern_size[1]),  # Larger pattern first
            -self.definition_order,  # Later definition first
        )

    def __repr__(self) -> str:
        return (
            f"Candidate(atom={self.atom_id}, type={self.atomtype!r}, "
            f"prio={self.priority}, score={self.score}, "
            f"size={self.pattern_size})"
        )


# ===================================================================
#                     Scoring Policy
# ===================================================================


class ScoringPolicy:
    """Policy for computing pattern specificity scores.

    The default policy uses predicate weights:
        +0 per element predicate (baseline)
        +1 per charge/degree/hyb constraint
        +2 per aromatic/in_ring constraint
        +3 per bond order predicate
        +4 per custom predicate
    """

    @staticmethod
    def default(pattern: SMARTSGraph) -> int:
        """Compute default specificity score for a pattern.

        Uses pattern size (vertices + edges) as specificity metric.
        Larger patterns are considered more specific.

        Args:
            pattern: SmartsGraph pattern

        Returns:
            Specificity score (higher = more specific)
        """
        return pattern.vcount() + pattern.ecount()

    @staticmethod
    def custom(
        pattern: SMARTSGraph, vertex_weight: float = 1.0, edge_weight: float = 1.5
    ) -> int:
        """Compute custom specificity score.

        Args:
            pattern: SmartsGraph pattern
            vertex_weight: Multiplier for vertex predicate weights
            edge_weight: Multiplier for edge predicate weights

        Returns:
            Specificity score
        """
        score = 0

        # Score vertex predicates
        for v in pattern.vs:
            if "preds" in v.attributes():
                preds = v["preds"]
                for pred in preds:
                    if hasattr(pred, "meta"):
                        score += int(pred.meta.weight * vertex_weight)

        # Score edge predicates
        for e in pattern.es:
            if "preds" in e.attributes():
                preds = e["preds"]
                for pred in preds:
                    if hasattr(pred, "meta"):
                        score += int(pred.meta.weight * edge_weight)

        return score


# ===================================================================
#                     SMARTS Matcher
# ===================================================================


class SmartsMatcher:
    """Main matcher for atomtyping with SMARTS patterns.

    This class finds all candidate atom type assignments from a list
    of SMARTS patterns and resolves conflicts using a deterministic
    priority system.

    Example:
        >>> patterns = [pattern1, pattern2, pattern3]
        >>> matcher = SmartsMatcher(patterns)
        >>> mol_graph = build_mol_graph(structure)
        >>> candidates = matcher.find_candidates(mol_graph, vs_to_atomid)
        >>> result = matcher.resolve(candidates)
    """

    def __init__(
        self, patterns: list[SMARTSGraph], scoring: ScoringPolicy | None = None
    ):
        """Initialize matcher with patterns.

        Args:
            patterns: List of SmartsGraph patterns
            scoring: Scoring policy (default: ScoringPolicy.default)
        """
        self.patterns = patterns
        self.scoring = scoring or ScoringPolicy()

        # Store definition order
        for i, pattern in enumerate(self.patterns):
            if not hasattr(pattern, "definition_order"):
                pattern.definition_order = i

    def find_candidates(
        self,
        mol_graph: Graph,
        vs_to_atomid: dict[int, int],
        type_assignments: dict[int, str] | None = None,
    ) -> list[Candidate]:
        """Find all candidate atom type assignments.

        Args:
            mol_graph: Molecule graph with vertex/edge attributes
            vs_to_atomid: Mapping from vertex index to atom ID
            type_assignments: Current type assignments (for reference checking)

        Returns:
            List of Candidate objects
        """
        candidates = []

        for pattern_idx, pattern in enumerate(self.patterns):
            # Run VF2 subgraph isomorphism with type-aware matching
            matches = self._find_pattern_matches(pattern, mol_graph, type_assignments)

            # Get pattern metadata
            atomtype = pattern.atomtype_name
            priority = pattern.priority
            score = self.scoring.default(pattern)
            pattern_size = (pattern.vcount(), pattern.ecount())
            source = pattern.source or f"pattern_{pattern_idx}"
            definition_order = getattr(pattern, "definition_order", pattern_idx)

            # Determine which vertices should be typed
            target_vertices = pattern.target_vertices
            if not target_vertices:
                # Default: all matched vertices
                target_vertices = list(range(pattern.vcount()))

            # Create candidates for each match
            for match in matches:
                # match is a list [mol_v0, mol_v1, ...] where index is pattern vertex
                for pattern_v in target_vertices:
                    if pattern_v < len(match):
                        mol_v = match[pattern_v]
                        atom_id = vs_to_atomid.get(mol_v)
                        if atom_id is not None:
                            candidates.append(
                                Candidate(
                                    atom_id=atom_id,
                                    atomtype=atomtype,
                                    source=source,
                                    priority=priority,
                                    score=score,
                                    pattern_size=pattern_size,
                                    definition_order=definition_order,
                                )
                            )

        return candidates

    def _find_pattern_matches(
        self,
        pattern: SMARTSGraph,
        mol_graph: Graph,
        type_assignments: dict[int, str] | None = None,
    ) -> list[list[int]]:
        """Find all matches of pattern in molecule graph.

        Args:
            pattern: SmartsGraph pattern
            mol_graph: Molecule graph
            type_assignments: Current type assignments (for reference checking)

        Returns:
            List of matches, where each match is a list of vertex indices
        """

        # Build node and edge compatibility functions
        def node_compat(g1, g2, v1, v2):
            """Check if mol vertex v1 matches pattern vertex v2."""
            return pattern._node_match_fn(g1, g2, v1, v2)

        def edge_compat(g1, g2, e1, e2):
            """Check if mol edge e1 matches pattern edge e2."""
            return pattern._edge_match_fn(g1, g2, e1, e2)

        # Find all subgraph isomorphisms
        matches = mol_graph.get_subisomorphisms_vf2(
            pattern, node_compat_fn=node_compat, edge_compat_fn=edge_compat
        )

        return matches

    def resolve(self, candidates: list[Candidate]) -> dict[int, str]:
        """Resolve conflicts and return final atom type assignments.

        Args:
            candidates: List of Candidate objects

        Returns:
            Dict mapping atom_id -> atomtype
        """
        # Group candidates by atom_id
        by_atom: dict[int, list[Candidate]] = {}
        for cand in candidates:
            if cand.atom_id not in by_atom:
                by_atom[cand.atom_id] = []
            by_atom[cand.atom_id].append(cand)

        # Resolve conflicts for each atom
        result = {}
        for atom_id, atom_candidates in by_atom.items():
            # Sort by priority (uses Candidate.__lt__ via sort_key)
            atom_candidates.sort()

            # Pick the best (first after sorting)
            winner = atom_candidates[0]
            result[atom_id] = winner.atomtype

        return result

    def explain(self, candidates: list[Candidate]) -> dict[int, Any]:
        """Generate explain data for debugging.

        Args:
            candidates: List of Candidate objects

        Returns:
            Dict mapping atom_id -> explain data with all candidates
            and their ordering keys
        """
        # Group candidates by atom_id
        by_atom: dict[int, list[Candidate]] = {}
        for cand in candidates:
            if cand.atom_id not in by_atom:
                by_atom[cand.atom_id] = []
            by_atom[cand.atom_id].append(cand)

        explain_data = {}
        for atom_id, atom_candidates in by_atom.items():
            # Sort candidates
            atom_candidates.sort()

            # Build explain entry
            explain_data[atom_id] = {
                "winner": atom_candidates[0].atomtype if atom_candidates else None,
                "candidates": [
                    {
                        "atomtype": c.atomtype,
                        "source": c.source,
                        "priority": c.priority,
                        "score": c.score,
                        "pattern_size": c.pattern_size,
                        "definition_order": c.definition_order,
                        "rank": i + 1,
                    }
                    for i, c in enumerate(atom_candidates)
                ],
            }

        return explain_data
