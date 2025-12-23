"""Dependency analysis for SMARTS patterns with type references."""

from collections import defaultdict, deque

from molpy.typifier.graph import SMARTSGraph


class DependencyAnalyzer:
    """Analyzes dependencies between SMARTS patterns and computes matching levels.

    Attributes:
        patterns: Dictionary mapping atom type names to their SMARTSGraph patterns
        dependency_graph: Adjacency list of dependencies (type -> depends_on)
        levels: Dictionary mapping atom type names to their topological levels
        circular_groups: List of sets containing types with circular dependencies
    """

    def __init__(self, patterns: dict[str, SMARTSGraph]) -> None:
        """Initialize dependency analyzer.

        Args:
            patterns: Dictionary of {atom_type_name: SMARTSGraph}
        """
        self.patterns = patterns
        self.dependency_graph: dict[str, set[str]] = defaultdict(set)
        self.levels: dict[str, int] = {}
        self.circular_groups: list[set[str]] = []

        self._build_dependency_graph()
        self._compute_levels()

    def _build_dependency_graph(self) -> None:
        """Build dependency graph from pattern dependencies."""
        for type_name, pattern in self.patterns.items():
            # Get dependencies from the pattern
            deps = pattern.dependencies

            # Filter to only include dependencies that exist in our pattern set
            valid_deps = {dep for dep in deps if dep in self.patterns}
            self.dependency_graph[type_name] = valid_deps

    def _compute_levels(self) -> None:
        """Compute topological levels using Kahn's algorithm.

        Assigns level 0 to patterns with no dependencies, level 1 to patterns
        that only depend on level 0, etc. Detects circular dependencies.
        """
        # In-degree is the number of dependencies each type has
        # (i.e., how many types it depends on)
        in_degree = {
            type_name: len(deps) for type_name, deps in self.dependency_graph.items()
        }

        # Initialize queue with nodes having no dependencies
        queue = deque(
            [
                type_name
                for type_name, deps in self.dependency_graph.items()
                if len(deps) == 0
            ]
        )

        current_level = 0
        processed = set()

        while queue:
            # Process all nodes at current level
            level_size = len(queue)
            for _ in range(level_size):
                type_name = queue.popleft()
                self.levels[type_name] = current_level
                self.patterns[type_name].level = current_level
                processed.add(type_name)

                # For all types that depend on this type,
                # decrease their in-degree
                for other_type, deps in self.dependency_graph.items():
                    if type_name in deps:
                        in_degree[other_type] -= 1
                        if in_degree[other_type] == 0:
                            queue.append(other_type)

            current_level += 1

        # Detect circular dependencies
        unprocessed = set(self.patterns.keys()) - processed
        if unprocessed:
            self._detect_circular_groups(unprocessed)
            # Assign max level + 1 to circular groups
            max_level = max(self.levels.values()) if self.levels else -1
            for group in self.circular_groups:
                for type_name in group:
                    self.levels[type_name] = max_level + 1
                    self.patterns[type_name].level = max_level + 1
                    self.patterns[type_name].level = max_level + 1

    def _detect_circular_groups(self, unprocessed: set[str]) -> None:
        """Detect strongly connected components (circular dependency groups).

        Args:
            unprocessed: Set of type names that weren't assigned a level
        """
        # Use Tarjan's algorithm to find strongly connected components
        index_counter = [0]
        stack = []
        lowlinks = {}
        index = {}
        on_stack = defaultdict(bool)

        def strongconnect(node):
            index[node] = index_counter[0]
            lowlinks[node] = index_counter[0]
            index_counter[0] += 1
            stack.append(node)
            on_stack[node] = True

            # Consider successors (nodes that depend on this node)
            for other_node in self.dependency_graph.get(node, set()):
                if other_node not in unprocessed:
                    continue
                if other_node not in index:
                    strongconnect(other_node)
                    lowlinks[node] = min(lowlinks[node], lowlinks[other_node])
                elif on_stack[other_node]:
                    lowlinks[node] = min(lowlinks[node], index[other_node])

            # If node is a root, pop the stack to get SCC
            if lowlinks[node] == index[node]:
                component = set()
                while True:
                    w = stack.pop()
                    on_stack[w] = False
                    component.add(w)
                    if w == node:
                        break
                if len(component) > 1:  # Only record if actually circular
                    self.circular_groups.append(component)

        for node in unprocessed:
            if node not in index:
                strongconnect(node)

    def get_patterns_by_level(self, level: int) -> list[SMARTSGraph]:
        """Get all patterns at a specific level.

        Args:
            level: Topological level number

        Returns:
            List of SMARTSGraph patterns at that level
        """
        return [
            pattern
            for type_name, pattern in self.patterns.items()
            if self.levels.get(type_name) == level
        ]

    def get_max_level(self) -> int:
        """Get the maximum level number.

        Returns:
            Maximum level, or -1 if no patterns
        """
        return max(self.levels.values()) if self.levels else -1

    def has_circular_dependencies(self) -> bool:
        """Check if there are any circular dependencies.

        Returns:
            True if circular dependencies exist
        """
        return len(self.circular_groups) > 0
