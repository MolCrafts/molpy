"""Layered typing engine for dependency-aware SMARTS matching."""

from igraph import Graph

from molpy.typifier.dependency_analyzer import DependencyAnalyzer
from molpy.typifier.graph import SMARTSGraph
from molpy.typifier.matcher import SmartsMatcher


class LayeredTypingEngine:
    """Orchestrates level-by-level atom typing with dependency resolution.

    This engine handles:
    1. Dependency analysis and topological sorting
    2. Layered matching (level 0 first, then level 1, etc.)
    3. Conflict resolution within each level
    4. Fixed-point iteration for circular dependencies

    Attributes:
        patterns: Dictionary mapping atom type names to SMARTSGraph patterns
        matcher: SmartsMatcher instance for pattern matching
        analyzer: DependencyAnalyzer for computing levels
    """

    def __init__(self, patterns: dict[str, SMARTSGraph]) -> None:
        """Initialize layered typing engine.

        Args:
            patterns: Dictionary of {atom_type_name: SMARTSGraph}
        """
        self.patterns = patterns
        self.analyzer = DependencyAnalyzer(patterns)

        # Create matcher with patterns sorted by level
        pattern_list = self._get_sorted_patterns()
        self.matcher = SmartsMatcher(pattern_list)

    def _get_sorted_patterns(self) -> list[SMARTSGraph]:
        """Get patterns sorted by level and priority.

        Returns:
            List of SMARTSGraph patterns
        """
        max_level = self.analyzer.get_max_level()
        sorted_patterns = []

        for level in range(max_level + 1):
            level_patterns = self.analyzer.get_patterns_by_level(level)
            # Sort by priority within level (higher first)
            level_patterns.sort(key=lambda p: -p.priority)
            sorted_patterns.extend(level_patterns)

        return sorted_patterns

    def typify(
        self, mol_graph: Graph, vs_to_atomid: dict[int, int], max_iterations: int = 10
    ) -> dict[int, str]:
        """Perform layered atom typing with dependency resolution.

        Args:
            mol_graph: Molecule graph with vertex/edge attributes
            vs_to_atomid: Mapping from vertex index to atom ID
            max_iterations: Maximum iterations for circular dependency resolution

        Returns:
            Dictionary mapping atom_id -> atom_type
        """
        type_assignments: dict[int, str] = {}
        max_level = self.analyzer.get_max_level()

        # Process each level in order
        for level in range(max_level + 1):
            level_patterns = self.analyzer.get_patterns_by_level(level)

            if not level_patterns:
                continue

            # Check if this level has circular dependencies
            is_circular = any(
                self.patterns[p.atomtype_name] in group
                for group in self.analyzer.circular_groups
                for p in level_patterns
            )

            if is_circular:
                # Use fixed-point iteration
                type_assignments = self._resolve_circular(
                    level_patterns,
                    mol_graph,
                    vs_to_atomid,
                    type_assignments,
                    max_iterations,
                )
            else:
                # Normal level-by-level matching
                type_assignments = self._resolve_level(
                    level_patterns, mol_graph, vs_to_atomid, type_assignments
                )

        return type_assignments

    def _resolve_level(
        self,
        level_patterns: list[SMARTSGraph],
        mol_graph: Graph,
        vs_to_atomid: dict[int, int],
        current_assignments: dict[int, str],
    ) -> dict[int, str]:
        """Resolve atom types for a single level.

        Args:
            level_patterns: Patterns at this level
            mol_graph: Molecule graph
            vs_to_atomid: Vertex to atom ID mapping
            current_assignments: Current type assignments

        Returns:
            Updated type assignments
        """
        # Add current type assignments to graph vertices
        # This allows IR mode patterns to access type information
        atomid_to_vs = {aid: vs for vs, aid in vs_to_atomid.items()}
        for atom_id, atomtype in current_assignments.items():
            if atom_id in atomid_to_vs:
                vs_idx = atomid_to_vs[atom_id]
                mol_graph.vs[vs_idx]["atomtype"] = atomtype

        # Create temporary matcher for this level
        level_matcher = SmartsMatcher(level_patterns)

        # Find candidates with current type assignments
        candidates = level_matcher.find_candidates(
            mol_graph, vs_to_atomid, current_assignments
        )

        # Resolve conflicts
        new_assignments = level_matcher.resolve(candidates)

        # Merge with current assignments (new assignments override)
        result = current_assignments.copy()
        result.update(new_assignments)

        return result

    def _resolve_circular(
        self,
        level_patterns: list[SMARTSGraph],
        mol_graph: Graph,
        vs_to_atomid: dict[int, int],
        current_assignments: dict[int, str],
        max_iterations: int,
    ) -> dict[int, str]:
        """Resolve circular dependencies using fixed-point iteration.

        Args:
            level_patterns: Patterns with circular dependencies
            mol_graph: Molecule graph
            vs_to_atomid: Vertex to atom ID mapping
            current_assignments: Current type assignments
            max_iterations: Maximum iterations

        Returns:
            Updated type assignments
        """
        assignments = current_assignments.copy()

        for _iteration in range(max_iterations):
            # Save previous state
            prev_assignments = assignments.copy()

            # Run matching with current assignments
            assignments = self._resolve_level(
                level_patterns, mol_graph, vs_to_atomid, assignments
            )

            # Check for convergence
            if assignments == prev_assignments:
                break

        return assignments

    def get_explain_data(self, mol_graph: Graph, vs_to_atomid: dict[int, int]) -> dict:
        """Generate detailed explanation of typing process.

        Args:
            mol_graph: Molecule graph
            vs_to_atomid: Vertex to atom ID mapping

        Returns:
            Dictionary with detailed typing information
        """
        type_assignments = {}
        max_level = self.analyzer.get_max_level()

        explain_data = {
            "levels": {},
            "circular_groups": [list(g) for g in self.analyzer.circular_groups],
            "final_assignments": {},
        }

        for level in range(max_level + 1):
            level_patterns = self.analyzer.get_patterns_by_level(level)

            if not level_patterns:
                continue

            # Create matcher for this level
            level_matcher = SmartsMatcher(level_patterns)
            candidates = level_matcher.find_candidates(
                mol_graph, vs_to_atomid, type_assignments
            )

            # Get explanation
            level_explain = level_matcher.explain(candidates)
            explain_data["levels"][level] = {
                "patterns": [p.atomtype_name for p in level_patterns],
                "assignments": level_explain,
            }

            # Update assignments
            new_assignments = level_matcher.resolve(candidates)
            type_assignments.update(new_assignments)

        explain_data["final_assignments"] = type_assignments
        return explain_data
