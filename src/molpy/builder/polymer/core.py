"""Core polymer builder module.

This module consolidates the core PolymerBuilder class, type definitions,
and exception classes for polymer assembly.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, field
from typing import Any, Protocol, runtime_checkable

from molpy.core.atomistic import Atom, Atomistic
from molpy.parser.smiles import parse_cgsmiles
from molpy.parser.smiles.cgsmiles_ir import (
    CGSmilesGraphIR,
    CGSmilesIR,
    CGSmilesNodeIR,
)

# ============================================================================
# Type Protocol
# ============================================================================


@runtime_checkable
class TypifierProtocol(Protocol):
    """Protocol for typifiers usable with PolymerBuilder.

    Any class implementing a `typify()` method with this signature can be used
    as a typifier, without needing to inherit from TypifierBase.
    """

    def typify(self, struct: Atomistic) -> Atomistic:
        """Typify an atomistic structure.

        Args:
            struct: Structure to typify

        Returns:
            The typified structure
        """
        ...


# ============================================================================
# Exception Classes
# ============================================================================


class AssemblyError(Exception):
    """Base exception for polymer assembly errors."""

    pass


class SequenceError(AssemblyError):
    """Invalid sequence (e.g., too short, unknown labels)."""

    pass


class AmbiguousPortsError(AssemblyError):
    """Cannot uniquely determine which ports to connect."""

    pass


class MissingConnectorRule(AssemblyError):
    """No connector rule found for a given monomer pair."""

    pass


class NoCompatiblePortsError(AssemblyError):
    """No compatible port pair found between two monomers."""

    pass


class BondKindConflictError(AssemblyError):
    """Conflicting bond kind specifications."""

    pass


class PortReuseError(AssemblyError):
    """Attempt to reuse a consumed port (multiplicity = 0)."""

    pass


class GeometryError(AssemblyError):
    """Base exception for geometry-related errors."""

    pass


class OrientationUnavailableError(GeometryError):
    """Cannot infer orientation for a port (no neighbors, no role info)."""

    pass


class PositionMissingError(GeometryError):
    """Entity is missing required 3D position data."""

    pass


from molpy.reacter.base import ReactionResult
from .polymer_builder import PolymerBuildResult


# ============================================================================
# PolymerBuilder
# ============================================================================


class PolymerBuilder:
    """Build polymers from CGSmiles notation with support for arbitrary topologies.

    This builder parses CGSmiles strings and constructs polymers using a graph-based
    approach, supporting:
    - Linear chains: ``{[#A][#B][#C]}``
    - Branched structures: ``{[#A]([#B])[#C]}``
    - Cyclic structures: ``{[#A]1[#B][#C]1}``
    - Repeat operators: ``{[#A]|10}``

    Example:
        >>> builder = PolymerBuilder(
        ...     library={"EO": eo_monomer},
        ...     connector=Connector(rules={("EO", "EO"): (">", "<")}, reacter=rxn),
        ... )
        >>> result = builder.build("{[#EO]|10}")
    """

    def __init__(
        self,
        library: Mapping[str, Atomistic],
        connector=None,
        *,
        reacter=None,
        typifier: TypifierProtocol | None = None,
        placer=None,
    ):
        """Initialize the polymer builder.

        Provide **either** ``connector`` or ``reacter``, not both.
        """
        if connector is not None and reacter is not None:
            raise TypeError("Provide either 'connector' or 'reacter', not both")
        if connector is None and reacter is None:
            raise TypeError("One of 'connector' or 'reacter' is required")

        if reacter is not None:
            from .connectors import Connector

            connector = Connector(reacter=reacter)

        self.library = library
        self.connector = connector
        self.typifier = typifier
        self.placer = placer

    def build(self, cgsmiles: str) -> PolymerBuildResult:
        """Build a polymer from a CGSmiles string.

        Args:
            cgsmiles: CGSmiles notation string (e.g., "{[#EO]|10}")

        Returns:
            PolymerBuildResult containing the assembled polymer and metadata
        """
        ir = parse_cgsmiles(cgsmiles)
        self._validate_ir(ir)

        polymer, history = self._build_from_graph(ir.base_graph)

        from .port_utils import cleanup_build_markers

        cleanup_build_markers(polymer)

        return PolymerBuildResult(
            polymer=polymer,
            connection_history=history,
            total_steps=len(history),
        )

    def _validate_ir(self, ir: CGSmilesIR) -> None:
        """Validate CGSmiles IR."""
        graph = ir.base_graph

        if not graph.nodes:
            raise ValueError("CGSmiles graph is empty")

        missing_labels = {
            node.label for node in graph.nodes if node.label not in self.library
        }

        if missing_labels:
            available = list(self.library.keys())
            raise SequenceError(
                f"Labels {sorted(missing_labels)} not found in library. "
                f"Available labels: {available}"
            )

    def _build_from_graph(
        self, graph: CGSmilesGraphIR
    ) -> tuple[Atomistic, list[ReactionResult]]:
        """Build polymer from CGSmiles graph using iterative DFS traversal."""
        if not graph.nodes:
            raise ValueError("Cannot build from empty graph")

        adjacency = self._build_adjacency_list(graph)
        node_index = {node.id: node for node in graph.nodes}

        monomers: dict[int, Atomistic] = {}
        for node in graph.nodes:
            monomers[node.id] = self._create_monomer(node)

        connection_history: list[ReactionResult] = []
        visited_edges: set[tuple[int, int]] = set()

        # Iterative DFS using explicit stack
        root_node = graph.nodes[0]
        stack = [root_node]
        visited_nodes: set[int] = set()

        while stack:
            node = stack.pop()
            if node.id in visited_nodes:
                continue
            visited_nodes.add(node.id)

            for neighbor_id, bond_order in adjacency.get(node.id, []):
                edge_key = (min(node.id, neighbor_id), max(node.id, neighbor_id))
                if edge_key in visited_edges:
                    continue
                visited_edges.add(edge_key)

                neighbor_node = node_index[neighbor_id]

                current_monomer = monomers[node.id]
                neighbor_monomer = monomers[neighbor_id]

                if current_monomer is neighbor_monomer:
                    # Ring closure
                    merged = self._connect_monomers(
                        left=current_monomer,
                        right=neighbor_monomer,
                        left_label=node.label,
                        right_label=neighbor_node.label,
                        left_node_id=node.id,
                        right_node_id=neighbor_id,
                        bond_order=bond_order,
                        connection_history=connection_history,
                    )
                    for nid in monomers:
                        if monomers[nid] is current_monomer:
                            monomers[nid] = merged
                else:
                    current_group = [
                        nid for nid, mon in monomers.items() if mon is current_monomer
                    ]
                    neighbor_group = [
                        nid for nid, mon in monomers.items() if mon is neighbor_monomer
                    ]

                    merged = self._connect_monomers(
                        left=current_monomer,
                        right=neighbor_monomer,
                        left_label=node.label,
                        right_label=neighbor_node.label,
                        left_node_id=node.id,
                        right_node_id=neighbor_id,
                        bond_order=bond_order,
                        connection_history=connection_history,
                    )

                    for nid in current_group + neighbor_group:
                        monomers[nid] = merged

                # Push neighbor for further traversal
                if neighbor_id not in visited_nodes:
                    stack.append(neighbor_node)

        return monomers[root_node.id], connection_history

    def _build_adjacency_list(
        self, graph: CGSmilesGraphIR
    ) -> dict[int, list[tuple[int, int]]]:
        """Build adjacency list from bonds."""
        adjacency: dict[int, list[tuple[int, int]]] = {
            node.id: [] for node in graph.nodes
        }

        for bond in graph.bonds:
            i_id = bond.node_i.id
            j_id = bond.node_j.id
            order = bond.order
            adjacency[i_id].append((j_id, order))
            adjacency[j_id].append((i_id, order))

        return adjacency

    def _create_monomer(self, node: CGSmilesNodeIR) -> Atomistic:
        """Create monomer copy from library and mark atoms with node ID."""
        monomer = self.library[node.label].copy()

        for atom in monomer.atoms:
            atom["monomer_node_id"] = node.id

        return monomer

    def _connect_monomers(
        self,
        left: Atomistic,
        right: Atomistic,
        left_label: str,
        right_label: str,
        left_node_id: int,
        right_node_id: int,
        bond_order: int,
        connection_history: list[ReactionResult],
    ) -> Atomistic:
        """Connect two monomers using node IDs for precise port location."""
        from .connectors import ConnectorContext
        from .port_utils import get_ports_on_node

        left_ports = get_ports_on_node(left, left_node_id)
        right_ports = get_ports_on_node(right, right_node_id)

        if not left_ports:
            raise NoCompatiblePortsError(
                f"Node {left_node_id} (label '{left_label}') has no available ports"
            )
        if not right_ports:
            raise NoCompatiblePortsError(
                f"Node {right_node_id} (label '{right_label}') has no available ports"
            )

        ctx = ConnectorContext(
            step=len(connection_history),
            left_label=left_label,
            right_label=right_label,
            sequence=[left_label, right_label],
        )

        left_port_name, left_port_idx, right_port_name, right_port_idx, _ = (
            self.connector.select_ports(left, right, left_ports, right_ports, ctx)
        )

        left_port_atom = left_ports[left_port_name][left_port_idx]
        right_port_atom = right_ports[right_port_name][right_port_idx]

        if self.placer is not None:
            self.placer.place_monomer(left, right, left_port_atom, right_port_atom)

        # Save port atoms for transfer after reaction
        left_port_targets: dict[str, list[Atom]] = dict(left_ports)
        right_port_targets: dict[str, list[Atom]] = dict(right_ports)

        connection_result = self.connector.connect(
            left,
            right,
            left_label,
            right_label,
            left_port_atom,
            right_port_atom,
            typifier=self.typifier,
        )

        product = connection_result.product
        connection_history.append(connection_result)

        entity_map = self._build_entity_map(connection_result)
        self._transfer_unused_ports(
            product,
            entity_map,
            left_port_targets,
            right_port_targets,
            left_port_name,
            left_port_idx,
            right_port_name,
            right_port_idx,
            left_node_id,
            right_node_id,
        )
        self._preserve_node_ids(product, entity_map, (left_node_id, right_node_id))
        self._cleanup_stale_ports(product)

        return product

    def _build_entity_map(self, result: ReactionResult) -> dict[Atom, Atom]:
        """Build combined entity map from reaction result."""
        entity_map: dict[Atom, Atom] = {}
        if result.entity_maps:
            for emap in result.entity_maps:
                entity_map.update(emap)
        return entity_map

    def _transfer_unused_ports(
        self,
        product: Atomistic,
        entity_map: dict[Atom, Atom],
        left_port_targets: dict[str, list[Atom]],
        right_port_targets: dict[str, list[Atom]],
        left_port_name: str,
        left_port_idx: int,
        right_port_name: str,
        right_port_idx: int,
        left_node_id: int,
        right_node_id: int,
    ) -> None:
        """Transfer unused ports from reactants to product."""
        atoms_in_product = set(product.atoms)

        for port_name, original_targets in left_port_targets.items():
            for idx, original_target in enumerate(original_targets):
                if port_name == left_port_name and idx == left_port_idx:
                    continue
                new_target = entity_map.get(original_target)
                if new_target is not None and new_target in atoms_in_product:
                    new_target["monomer_node_id"] = left_node_id
                    new_target["port"] = port_name

        for port_name, original_targets in right_port_targets.items():
            for idx, original_target in enumerate(original_targets):
                if port_name == right_port_name and idx == right_port_idx:
                    continue
                new_target = entity_map.get(original_target)
                if new_target is not None and new_target in atoms_in_product:
                    new_target["monomer_node_id"] = right_node_id
                    new_target["port"] = port_name

    def _preserve_node_ids(
        self,
        product: Atomistic,
        entity_map: dict[Atom, Atom],
        connected_node_ids: tuple[int, int],
    ) -> None:
        """Preserve node IDs and ports from original atoms through entity map.

        Ports for the two connected nodes are NOT restored here — they were
        already handled (with consumed-port filtering) by _transfer_unused_ports.
        """
        reverse_map: dict[Atom, Atom] = {v: k for k, v in entity_map.items()}

        for atom in product.atoms:
            original_atom = reverse_map.get(atom)
            if original_atom is None:
                continue

            original_node_id = original_atom.get("monomer_node_id")
            if original_node_id is not None:
                atom["monomer_node_id"] = original_node_id

            original_port = original_atom.get("port")
            if original_port and atom.get("port") is None:
                if original_node_id in connected_node_ids:
                    continue
                atom["port"] = original_port

    def _cleanup_stale_ports(self, product: Atomistic) -> None:
        """Remove port markers from O atoms that have no bonded H."""
        from molpy.reacter.utils import find_neighbors as _find_neighbors

        for atom in product.atoms:
            if atom.get("element") == "O" and atom.get("port"):
                h_neighbors = _find_neighbors(product, atom, element="H")
                if not h_neighbors:
                    del atom["port"]
