"""Lightweight DAG orchestration for Compute nodes.

Zero non-stdlib dependencies (uses ``graphlib.TopologicalSorter``).
"""

from __future__ import annotations

from graphlib import CycleError, TopologicalSorter
from typing import Any


class WorkflowError(Exception):
    """Base exception for Workflow errors."""


class WorkflowDuplicateNodeError(WorkflowError):
    """A node with the same name is already registered."""


class WorkflowCycleError(WorkflowError):
    """Adding this edge would create a cycle in the DAG."""


class WorkflowMissingInputError(WorkflowError):
    """Required external inputs were not provided to run()."""

    def __init__(self, missing: set[str]):
        self.missing = missing
        super().__init__(f"Missing external inputs: {sorted(missing)!r}")


class Workflow:
    """Compose Compute nodes into a DAG and execute them in topological order.

    Parameters are bound by name: each node is called as ``node(**resolved)``
    where *resolved* maps its parameter names to upstream results or
    externally-supplied values. The Workflow never inspects node signatures.
    """

    def __init__(self):
        self._nodes: dict[str, Any] = {}
        self._edges: dict[str, dict[str, str]] = {}
        self._graph: dict[str, set[str]] = {}

    # ------------------------------------------------------------------
    # Registration
    # ------------------------------------------------------------------

    def add(self, name: str, compute: Any, inputs: dict[str, str] | None = None) -> str:
        """Register a compute node.

        Args:
            name: Unique node name.
            compute: Callable object called as ``compute(**resolved)``.
            inputs: Mapping from the node's parameter names to source names.
                    Each source is either a registered node name or an
                    external input name.

        Returns:
            *name*, for fluent chaining.

        Raises:
            WorkflowDuplicateNodeError: *name* already registered.
            WorkflowCycleError: Adding this node creates a cycle.
        """
        if name in self._nodes:
            raise WorkflowDuplicateNodeError(f"Node {name!r} is already registered")

        inputs = dict(inputs) if inputs is not None else {}

        # Classify sources: registered nodes → predecessors, rest → externals
        new_predecessors: set[str] = set()
        for src in inputs.values():
            if src in self._nodes:
                new_predecessors.add(src)

        # Back-propagate: any existing node that has *this* new node as a
        # source in _edges now gains a real predecessor edge.
        back_edges: dict[str, str] = {}  # existing_node → param_name
        for node_name, node_inputs in self._edges.items():
            for param, src in node_inputs.items():
                if src == name:
                    back_edges[node_name] = param

        # Tentatively apply new state
        self._nodes[name] = compute
        self._edges[name] = inputs
        self._graph[name] = new_predecessors
        for ename in back_edges:
            self._graph[ename].add(name)

        # Cycle check on the tentative state
        try:
            TopologicalSorter(self._graph).prepare()
        except CycleError as exc:
            # Rollback
            del self._nodes[name]
            del self._edges[name]
            del self._graph[name]
            for ename in back_edges:
                self._graph[ename].discard(name)
            raise WorkflowCycleError(
                f"Adding node {name!r} creates a cycle: {exc}"
            ) from exc

        return name

    # ------------------------------------------------------------------
    # Introspection
    # ------------------------------------------------------------------

    def topological_order(self) -> list[str]:
        """Return nodes in topological execution order."""
        ts = TopologicalSorter(self._graph)
        return list(ts.static_order())

    def predecessors(self, name: str) -> set[str]:
        """Return the set of node-name predecessors for *name*.

        External inputs are excluded.
        """
        return self._graph.get(name, set())

    @property
    def nodes(self) -> list[str]:
        """Registered node names in insertion order."""
        return list(self._nodes.keys())

    @property
    def external_inputs(self) -> set[str]:
        """All source names that are not registered nodes."""
        result: set[str] = set()
        for inputs_ in self._edges.values():
            for src in inputs_.values():
                if src not in self._nodes:
                    result.add(src)
        return result

    # ------------------------------------------------------------------
    # Execution
    # ------------------------------------------------------------------

    def run(self, **externals: Any) -> dict[str, Any]:
        """Execute all nodes in topological order.

        Args:
            **externals: Values for every external input.

        Returns:
            ``{node_name: result}`` for every registered node.

        Raises:
            WorkflowMissingInputError: One or more external inputs are absent.
        """
        # Validate external inputs before any execution
        needed = self.external_inputs
        provided = set(externals.keys())
        missing = needed - provided
        if missing:
            raise WorkflowMissingInputError(missing)

        order = self.topological_order()
        results: dict[str, Any] = {}

        for name in order:
            param_sources = self._edges.get(name, {})
            resolved = {
                param: results[src] if src in results else externals[src]
                for param, src in param_sources.items()
            }
            results[name] = self._nodes[name](**resolved)

        return results
