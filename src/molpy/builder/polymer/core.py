"""Core polymer builder module.

This module consolidates the core PolymerBuilder class, type definitions,
and exception classes for polymer assembly.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Protocol, runtime_checkable

from molpy.core.atomistic import Atom, Atomistic
from molpy.parser.smiles import parse_cgsmiles
from molpy.parser.smiles.cgsmiles_ir import (
    CGSmilesGraphIR,
    CGSmilesIR,
    CGSmilesNodeIR,
)
from molpy.reacter.base import ReactionResult

from .errors import (
    AmbiguousPortsError,
    AssemblyError,
    BondKindConflictError,
    GeometryError,
    MissingConnectorRule,
    NoCompatiblePortsError,
    OrientationUnavailableError,
    PortReuseError,
    PositionMissingError,
    SequenceError,
)

if TYPE_CHECKING:
    from molpy.typifier.cache import RetypeCache

__all__ = [
    "AmbiguousPortsError",
    "AssemblyError",
    "BondKindConflictError",
    "GeometryError",
    "MissingConnectorRule",
    "NoCompatiblePortsError",
    "OrientationUnavailableError",
    "PolymerBuilder",
    "PolymerBuildResult",
    "PortReuseError",
    "PositionMissingError",
    "SequenceError",
    "TypifierProtocol",
]

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
# Build Result
# ============================================================================


@dataclass
class PolymerBuildResult:
    """Result of building a polymer."""

    polymer: Atomistic
    connection_history: list[ReactionResult] = field(default_factory=list)
    total_steps: int = 0


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

    Attributes:
        library: Mapping from monomer label to port-annotated Atomistic
            template.
        connector: :class:`Connector` choosing ports and reactions between
            adjacent monomers.
        typifier: Optional typifier applied during assembly.
        placer: Optional :class:`Placer` for geometric placement (distances
            in Å).

    Example::

        builder = PolymerBuilder(
            library={"EO": eo_monomer},
            connector=Connector(port_map={("EO", "EO"): (">", "<")}, reacter=rxn),
        )
        result = builder.build("{[#EO]|10}")
        chain = result.polymer
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

        # One shared retype cache for the whole build: structurally identical
        # junctions recurring along the chain dedupe to a single hash key, so the
        # underlying atom typing runs once per distinct junction environment
        # (O(#distinct)) instead of once per connection (O(N)). Built only when
        # the typifier supports region typing; otherwise ``None`` leaves each
        # connection on its unchanged per-call path.
        retype_cache = self._make_retype_cache()
        polymer, history = self._build_from_graph(ir.base_graph, retype_cache)

        from .port_utils import cleanup_build_markers

        cleanup_build_markers(polymer)

        return PolymerBuildResult(
            polymer=polymer,
            connection_history=history,
            total_steps=len(history),
        )

    def _make_retype_cache(self) -> "RetypeCache | None":
        """Build the per-build shared cache when region typing is available.

        Returns ``None`` unless a typifier is set *and* it exposes
        ``typify_region`` — the capability the region-scoped retype path needs.
        Without one, connections fall back to the unchanged per-call behaviour.
        """
        typifier = self.typifier
        if typifier is not None and hasattr(typifier, "typify_region"):
            from molpy.typifier.cache import RetypeCache

            return RetypeCache(typifier)
        return None

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
        self, graph: CGSmilesGraphIR, retype_cache: "RetypeCache | None" = None
    ) -> tuple[Atomistic, list[ReactionResult]]:
        """Build polymer from CGSmiles graph using iterative DFS traversal.

        ``retype_cache`` is the build-wide shared cache threaded into every
        connection so recurring junctions type once; ``None`` keeps each
        connection on its per-call path.
        """
        if not graph.nodes:
            raise ValueError("Cannot build from empty graph")

        adjacency = self._build_adjacency_list(graph)
        node_index = {node.id: node for node in graph.nodes}

        # Group bookkeeping (replaces per-edge identity scans over all
        # monomers): each node maps to a group id; the group's current
        # structure is stored once per group. Live port atoms are kept
        # in a registry so connections never rescan the growing chain.
        from .port_utils import get_ports_on_node

        group_id: dict[int, int] = {}
        members: dict[int, list[int]] = {}
        struct_of: dict[int, Atomistic] = {}
        ports_registry: dict[int, dict[str, list[Atom]]] = {}

        for node in graph.nodes:
            monomer = self._create_monomer(node)
            group_id[node.id] = node.id
            members[node.id] = [node.id]
            struct_of[node.id] = monomer
            ports_registry[node.id] = get_ports_on_node(monomer, node.id)

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
                gid_left = group_id[node.id]
                gid_right = group_id[neighbor_id]

                merged, entity_map, consumed = self._connect_monomers(
                    left=struct_of[gid_left],
                    right=struct_of[gid_right],
                    left_label=node.label,
                    right_label=neighbor_node.label,
                    left_node_id=node.id,
                    right_node_id=neighbor_id,
                    left_ports=ports_registry.get(node.id, {}),
                    right_ports=ports_registry.get(neighbor_id, {}),
                    bond_order=bond_order,
                    connection_history=connection_history,
                    retype_cache=retype_cache,
                )

                if gid_left == gid_right:
                    # Ring closure: group membership unchanged
                    struct_of[gid_left] = merged
                    affected = members[gid_left]
                else:
                    # Union, smaller group folds into the larger one
                    if len(members[gid_left]) < len(members[gid_right]):
                        gid_left, gid_right = gid_right, gid_left
                    for nid in members[gid_right]:
                        group_id[nid] = gid_left
                    members[gid_left].extend(members[gid_right])
                    del members[gid_right]
                    del struct_of[gid_right]
                    struct_of[gid_left] = merged
                    affected = members[gid_left]

                self._remap_ports_registry(
                    ports_registry, affected, entity_map, consumed
                )
                self._cleanup_stale_ports(merged, ports_registry, affected)

                # Push neighbor for further traversal
                if neighbor_id not in visited_nodes:
                    stack.append(neighbor_node)

        return struct_of[group_id[root_node.id]], connection_history

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
        left_ports: dict[str, list[Atom]],
        right_ports: dict[str, list[Atom]],
        # bond_order is parsed from CGSmiles and reserved for future
        # bond-order-aware connection; connectors currently ignore it.
        bond_order: int,
        connection_history: list[ReactionResult],
        retype_cache: "RetypeCache | None" = None,
    ) -> tuple[Atomistic, dict[Atom, Atom], set[tuple[int, str, int]]]:
        """Connect two monomers using registry-provided port atoms.

        Port atoms come from the caller's per-build registry, so this
        method never rescans the (growing) accumulated structure.

        Returns:
            Tuple of (product, entity_map, consumed) where ``consumed``
            holds the (node_id, port_name, index) entries used by this
            connection.
        """
        from .connectors import ConnectorContext

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
        left_port_targets: dict[str, list[Atom]] = {
            name: list(atoms) for name, atoms in left_ports.items()
        }
        right_port_targets: dict[str, list[Atom]] = {
            name: list(atoms) for name, atoms in right_ports.items()
        }

        connection_result = self.connector.connect(
            left,
            right,
            left_label,
            right_label,
            left_port_atom,
            right_port_atom,
            typifier=self.typifier,
            retype_cache=retype_cache,
        )
        product = connection_result.product
        connection_history.append(connection_result)

        entity_map = self._build_entity_map(connection_result)
        self._transfer_unused_ports(
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
        self._preserve_node_ids(entity_map, (left_node_id, right_node_id))

        consumed = {
            (left_node_id, left_port_name, left_port_idx),
            (right_node_id, right_port_name, right_port_idx),
        }
        return product, entity_map, consumed

    @staticmethod
    def _remap_ports_registry(
        ports_registry: dict[int, dict[str, list[Atom]]],
        node_ids: list[int],
        entity_map: dict[Atom, Atom],
        consumed: set[tuple[int, str, int]],
    ) -> None:
        """Remap live port atoms through the connection's entity map.

        Bounded by the number of live port atoms in the affected groups,
        not by chain length: consumed ports are dropped, surviving port
        atoms are replaced by their product-side copies.
        """
        for nid in node_ids:
            node_ports = ports_registry.get(nid)
            if not node_ports:
                continue
            for port_name in list(node_ports):
                remapped: list[Atom] = []
                for idx, atom in enumerate(node_ports[port_name]):
                    if (nid, port_name, idx) in consumed:
                        continue
                    mapped = entity_map.get(atom)
                    if mapped is not None:
                        remapped.append(mapped)
                if remapped:
                    node_ports[port_name] = remapped
                else:
                    del node_ports[port_name]

    def _build_entity_map(self, result: ReactionResult) -> dict[Atom, Atom]:
        """Build combined entity map from reaction result."""
        entity_map: dict[Atom, Atom] = {}
        if result.entity_maps:
            for emap in result.entity_maps:
                entity_map.update(emap)
        return entity_map

    def _transfer_unused_ports(
        self,
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
        """Transfer unused ports from reactants to product.

        ``entity_map`` values are product members by construction (the
        base reacter filters on product membership), so no product-wide
        set is built here.
        """
        for port_name, original_targets in left_port_targets.items():
            for idx, original_target in enumerate(original_targets):
                if port_name == left_port_name and idx == left_port_idx:
                    continue
                new_target = entity_map.get(original_target)
                if new_target is not None:
                    new_target["monomer_node_id"] = left_node_id
                    new_target["port"] = port_name

        for port_name, original_targets in right_port_targets.items():
            for idx, original_target in enumerate(original_targets):
                if port_name == right_port_name and idx == right_port_idx:
                    continue
                new_target = entity_map.get(original_target)
                if new_target is not None:
                    new_target["monomer_node_id"] = right_node_id
                    new_target["port"] = port_name

    def _preserve_node_ids(
        self,
        entity_map: dict[Atom, Atom],
        connected_node_ids: tuple[int, int],
    ) -> None:
        """Preserve node IDs and ports from original atoms through entity map.

        Ports for the two connected nodes are NOT restored here — they were
        already handled (with consumed-port filtering) by _transfer_unused_ports.
        """
        # Iterate the entity map directly: equivalent to scanning the
        # product with a reverse map, since only mapped atoms can match.
        for original_atom, atom in entity_map.items():
            original_node_id = original_atom.get("monomer_node_id")
            if original_node_id is not None:
                atom["monomer_node_id"] = original_node_id

            original_port = original_atom.get("port")
            if original_port and atom.get("port") is None:
                if original_node_id in connected_node_ids:
                    continue
                atom["port"] = original_port

    def _cleanup_stale_ports(
        self,
        product: Atomistic,
        ports_registry: dict[int, dict[str, list[Atom]]],
        node_ids: list[int],
    ) -> None:
        """Remove port markers from O atoms that have no bonded H.

        Operates on the live port atoms tracked in the registry (bounded
        by port count) instead of scanning the whole product.
        """
        from molpy.reacter.utils import find_neighbors as _find_neighbors

        for nid in node_ids:
            node_ports = ports_registry.get(nid)
            if not node_ports:
                continue
            for port_name in list(node_ports):
                kept: list[Atom] = []
                for atom in node_ports[port_name]:
                    if atom.get("element") == "O" and atom.get("port"):
                        h_neighbors = _find_neighbors(product, atom, element="H")
                        if not h_neighbors:
                            del atom["port"]
                            continue
                    kept.append(atom)
                if kept:
                    node_ports[port_name] = kept
                else:
                    del node_ports[port_name]
