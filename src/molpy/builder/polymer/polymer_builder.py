"""
PolymerBuilder for constructing polymers from CGSmiles notation.

This module provides a builder class that can construct polymers with arbitrary
topologies (linear, branched, cyclic) directly from CGSmiles strings.
"""

from __future__ import annotations

from collections.abc import Mapping

from molpy.core.atomistic import Atomistic
from molpy.parser.smiles import parse_cgsmiles
from molpy.parser.smiles.cgsmiles_ir import (
    CGSmilesGraphIR,
    CGSmilesIR,
    CGSmilesNodeIR,
)
from molpy.typifier.atomistic import TypifierBase

from .connectors import ConnectorContext, ReacterConnector
from .errors import NoCompatiblePortsError, SequenceError
from .placer import Placer
from .port_utils import (
    get_all_port_info,
    set_port_metadata,
    PortInfo,
    get_port_metadata,
)
from .types import ConnectionMetadata, PolymerBuildResult


class PolymerBuilder:
    """
    Build polymers from CGSmiles notation with support for arbitrary topologies.

    This builder parses CGSmiles strings and constructs polymers using a graph-based
    approach, supporting:
    - Linear chains: {[#A][#B][#C]}
    - Branched structures: {[#A]([#B])[#C]}
    - Cyclic structures: {[#A]1[#B][#C]1}
    - Repeat operators: {[#A]|10}

    Example:
        >>> builder = PolymerBuilder(
        ...     library={"EO2": eo2_monomer, "PS": ps_monomer},
        ...     connector=connector,
        ...     typifier=typifier,
        ... )
        >>> result = builder.build("{[#EO2]|8[#PS]}")
    """

    def __init__(
        self,
        library: Mapping[str, Atomistic],
        connector: ReacterConnector,
        typifier: TypifierBase | None = None,
        placer: Placer | None = None,
    ):
        """
        Initialize the polymer builder.

        Args:
            library: Mapping from CGSmiles labels to Atomistic monomer structures
            connector: ReacterConnector for port selection and chemical reactions
            typifier: Optional typifier for automatic retypification
            placer: Optional Placer for positioning structures before connection
        """
        self.library = library
        self.connector = connector
        self.typifier = typifier
        self.placer = placer

    def build(self, cgsmiles: str) -> PolymerBuildResult:
        """
        Build a polymer from a CGSmiles string.

        Args:
            cgsmiles: CGSmiles notation string (e.g., "{[#EO2]|8[#PS]}")

        Returns:
            PolymerBuildResult containing the assembled polymer and metadata

        Raises:
            ValueError: If CGSmiles is invalid
            SequenceError: If labels in CGSmiles are not found in library
        """
        # Parse CGSmiles
        ir = parse_cgsmiles(cgsmiles)

        # Validate
        self._validate_ir(ir)

        # Build polymer from graph
        polymer, history = self._build_from_graph(ir.base_graph)

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

        # Check all labels exist in library
        missing_labels = set()
        for node in graph.nodes:
            if node.label not in self.library:
                missing_labels.add(node.label)

        if missing_labels:
            available = list(self.library.keys())
            raise SequenceError(
                f"Labels {sorted(missing_labels)} not found in library. "
                f"Available labels: {available}"
            )

    def _build_from_graph(
        self, graph: CGSmilesGraphIR
    ) -> tuple[Atomistic, list[ConnectionMetadata]]:
        """
        Build polymer from CGSmiles graph using DFS traversal with node tracking.

        Strategy (Approach B - Graph Building + Delayed Merging):
        1. Create independent monomer copies for each graph node
        2. Mark each atom with its originating node ID for topology tracking
        3. Connect monomers sequentially using DFS, but use node IDs to
           precisely locate ports even after merging (enables correct branching)
        4. Preserve node IDs through all merge operations
        """
        if not graph.nodes:
            raise ValueError("Cannot build from empty graph")

        # Build adjacency list
        adjacency = self._build_adjacency_list(graph)

        # Create independent monomer copies for each node
        monomers: dict[int, Atomistic] = {}
        for node in graph.nodes:
            monomers[node.id] = self._create_monomer(node)

        # Track connection state
        connection_history: list[ConnectionMetadata] = []
        visited_edges: set[tuple[int, int]] = set()

        # Start DFS from first node
        root_node = graph.nodes[0]

        # Connect all edges using DFS
        self._dfs_connect(
            graph=graph,
            node=root_node,
            adjacency=adjacency,
            monomers=monomers,
            visited_edges=visited_edges,
            connection_history=connection_history,
        )

        # Return the monomer of the root node (which now contains the entire polymer)
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
        """Create monomer copy from library and mark atoms with node ID.

        This allows tracking which atoms belong to which original graph node
        even after multiple merge operations. This is critical for:
        - Branching: Finding the correct node when connecting branches
        - Ring closure: Identifying which atoms belong to closure points
        - Port management: Locating ports on specific nodes after merging
        """
        monomer = self.library[node.label].copy()

        # Mark all atoms with the node ID for topology tracking
        # This allows us to find atoms belonging to a specific node
        # even after the monomer has been merged with others
        for atom in monomer.atoms:
            atom["monomer_node_id"] = node.id

        return monomer

    def _find_ports_on_node(
        self, polymer: Atomistic, node_id: int
    ) -> dict[str, list[PortInfo]]:
        """Find all ports belonging to a specific node in a merged polymer.

        This method searches through all atoms in the polymer and returns
        only those ports that belong to the specified node_id. This is
        essential for branch connections where we need to connect to a
        specific node in an already-merged structure.

        Args:
            polymer: Merged polymer structure
            node_id: ID of the node to find ports for

        Returns:
            Dictionary mapping port name -> list of PortInfo for ports on that node
        """
        ports: dict[str, list[PortInfo]] = {}
        for atom in polymer.atoms:
            # Only consider atoms that belong to the specified node
            if atom.get("monomer_node_id") == node_id:
                port_name = atom.get("port")
                if port_name:
                    metadata = get_port_metadata(atom, port_name)
                    if port_name not in ports:
                        ports[port_name] = []
                    ports[port_name].append(PortInfo(port_name, atom, **metadata))
        return ports

    def _dfs_connect(
        self,
        graph: CGSmilesGraphIR,
        node: CGSmilesNodeIR,
        adjacency: dict[int, list[tuple[int, int]]],
        monomers: dict[int, Atomistic],
        visited_edges: set[tuple[int, int]],
        connection_history: list[ConnectionMetadata],
    ) -> None:
        """DFS traversal to connect all neighbors using node-based port lookup.

        Key improvement: Uses node IDs to precisely locate ports even after
        monomers have been merged. This allows correct branch and ring connections.
        """
        node_id = node.id
        neighbors = adjacency.get(node_id, [])

        for neighbor_id, bond_order in neighbors:
            # Skip if edge already visited
            sorted_nodes = sorted([node_id, neighbor_id])
            edge_key = (sorted_nodes[0], sorted_nodes[1])
            if edge_key in visited_edges:
                continue
            visited_edges.add(edge_key)

            # Get neighbor node
            neighbor_node = next(n for n in graph.nodes if n.id == neighbor_id)

            # Get the current monomers
            current_monomer = monomers[node_id]
            neighbor_monomer = monomers[neighbor_id]

            # Handle case where monomers are already the same object (already merged)
            # This can happen in cyclic structures when DFS returns to a node
            # that has already been merged with the current node
            # In this case, we need to form the ring closure bond if it doesn't exist
            if current_monomer is neighbor_monomer:
                # Already merged - this is a ring closure
                # Use connector.connect() to handle ring closure through reacter
                # This ensures bond creation and typification are handled consistently
                merged = self._connect_monomers(
                    left=current_monomer,
                    right=neighbor_monomer,
                    left_label=node.label,
                    right_label=neighbor_node.label,
                    left_node_id=node_id,
                    right_node_id=neighbor_id,
                    bond_order=bond_order,
                    connection_history=connection_history,
                )

                # Update all nodes pointing to current_monomer to point to merged result
                # (though for ring closure, merged should be the same as current_monomer)
                for nid in monomers:
                    if monomers[nid] is current_monomer:
                        monomers[nid] = merged

                # Continue DFS
                self._dfs_connect(
                    graph=graph,
                    node=neighbor_node,
                    adjacency=adjacency,
                    monomers=monomers,
                    visited_edges=visited_edges,
                    connection_history=connection_history,
                )
                continue

            # Find all nodes that currently point to current_monomer
            current_group = [
                nid for nid, mon in monomers.items() if mon is current_monomer
            ]

            # Find all nodes that currently point to neighbor_monomer
            neighbor_group = [
                nid for nid, mon in monomers.items() if mon is neighbor_monomer
            ]

            # Connect them using node IDs for precise port location
            merged = self._connect_monomers(
                left=current_monomer,
                right=neighbor_monomer,
                left_label=node.label,
                right_label=neighbor_node.label,
                left_node_id=node_id,
                right_node_id=neighbor_id,
                bond_order=bond_order,
                connection_history=connection_history,
            )

            # Update ALL nodes in both groups to point to the merged result
            for nid in current_group + neighbor_group:
                monomers[nid] = merged

            # Recursively connect neighbors
            self._dfs_connect(
                graph=graph,
                node=neighbor_node,
                adjacency=adjacency,
                monomers=monomers,
                visited_edges=visited_edges,
                connection_history=connection_history,
            )

    def _connect_monomers(
        self,
        left: Atomistic,
        right: Atomistic,
        left_label: str,
        right_label: str,
        left_node_id: int,
        right_node_id: int,
        bond_order: int,
        connection_history: list[ConnectionMetadata],
    ) -> Atomistic:
        """Connect two monomers using node IDs for precise port location.

        This is the core improvement: By using node IDs, we can find the
        correct ports even when monomers have been merged with others.
        This enables correct branch and ring connections.

        Args:
            left: Left monomer structure (may already be merged with others)
            right: Right monomer structure (may already be merged with others)
            left_label: Label of left monomer type
            right_label: Label of right monomer type
            left_node_id: Graph node ID for the left monomer
            right_node_id: Graph node ID for the right monomer
            bond_order: Bond order for the connection
            connection_history: List to append connection metadata to

        Returns:
            Merged polymer structure
        """
        # Find ports on the specific nodes using node IDs
        # This is critical: even if left/right are merged polymers,
        # we can still find the correct ports for the specific nodes
        left_ports = self._find_ports_on_node(left, left_node_id)
        right_ports = self._find_ports_on_node(right, right_node_id)

        if not left_ports:
            raise NoCompatiblePortsError(
                f"Node {left_node_id} (label '{left_label}') has no available ports"
            )
        if not right_ports:
            raise NoCompatiblePortsError(
                f"Node {right_node_id} (label '{right_label}') has no available ports"
            )

        # Build context
        ctx = ConnectorContext(
            step=len(connection_history),
            sequence=[left_label, right_label],
            left_label=left_label,
            right_label=right_label,
            audit=[],
        )

        # Select ports (now returns indices for when multiple ports share the same name)
        left_port_name, left_port_idx, right_port_name, right_port_idx, _ = (
            self.connector.select_ports(
                left,
                right,
                left_ports,
                right_ports,
                ctx,
            )
        )

        # Get the specific port info from the lists using the indices
        left_port = left_ports[left_port_name][left_port_idx]
        right_port = right_ports[right_port_name][right_port_idx]

        # Position monomer
        if self.placer is not None:
            self.placer.place_monomer(
                left,
                right,
                left_port,
                right_port,
            )

        # Save port targets for transfer of unused ports after reaction
        # We need to know which atoms had ports before, so we can transfer to product
        left_port_targets: dict[str, list[Atom]] = {
            name: [port.target for port in port_list]
            for name, port_list in left_ports.items()
        }
        right_port_targets: dict[str, list[Atom]] = {
            name: [port.target for port in port_list]
            for name, port_list in right_ports.items()
        }

        # Find port atoms directly by node_id
        from molpy.reacter.selectors import find_port_atom_by_node

        left_port_atom = find_port_atom_by_node(left, left_port_name, left_node_id)
        right_port_atom = find_port_atom_by_node(right, right_port_name, right_node_id)

        # Execute connection with explicit port atoms
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
        metadata = connection_result.metadata

        # Store metadata
        connection_history.append(metadata)

        # Transfer unused ports using entity map from reaction
        # The entity_maps in metadata contain the mapping from original atoms to product atoms
        from molpy.core.atomistic import Atom

        atoms_in_product = set(product.atoms)

        # Build combined entity map from metadata
        entity_map: dict[Atom, Atom] = {}
        if metadata.entity_maps:
            for emap in metadata.entity_maps:
                entity_map.update(emap)

        # Build reverse map for node ID preservation
        reverse_entity_map: dict[Atom, Atom] = {}
        for original_atom, product_atom in entity_map.items():
            reverse_entity_map[product_atom] = original_atom

        # Transfer unused left ports (only for the specific node)
        for port_name, original_targets in left_port_targets.items():
            # Skip the port name that was used (but only the specific instance)
            for idx, original_target in enumerate(original_targets):
                # Skip the specific port that was used
                if port_name == left_port_name and idx == left_port_idx:
                    continue

                # Find the corresponding atom in product using entity map
                new_target = entity_map.get(original_target)

                # Only add port if we found a valid target in product
                if new_target is not None and new_target in atoms_in_product:
                    # Preserve the node ID on the atom
                    new_target["monomer_node_id"] = left_node_id
                    # Get original port metadata
                    original_port = left_ports[port_name][idx]
                    # Mark port on atom
                    new_target["port"] = port_name
                    # Set port metadata
                    set_port_metadata(
                        new_target,
                        port_name,
                        role=original_port.role,
                        bond_kind=original_port.bond_kind,
                        compat=original_port.compat,
                        priority=original_port.priority,
                    )

        # Transfer unused right ports (only for the specific node)
        for port_name, original_targets in right_port_targets.items():
            # Skip the port name that was used (but only the specific instance)
            for idx, original_target in enumerate(original_targets):
                # Skip the specific port that was used
                if port_name == right_port_name and idx == right_port_idx:
                    continue

                # Find the corresponding atom in product using entity map
                new_target = entity_map.get(original_target)

                # Only add port if we found a valid target in product
                if new_target is not None and new_target in atoms_in_product:
                    # Preserve the node ID on the atom
                    new_target["monomer_node_id"] = right_node_id
                    # Get original port metadata
                    original_port = right_ports[port_name][idx]
                    # Mark port on atom
                    new_target["port"] = port_name
                    # Set port metadata
                    set_port_metadata(
                        new_target,
                        port_name,
                        role=original_port.role,
                        bond_kind=original_port.bond_kind,
                        compat=original_port.compat,
                        priority=original_port.priority,
                    )

        # Preserve node IDs for all atoms in the product
        # Map atoms from left and right to their node IDs
        for atom in product.atoms:
            # Try to find original atom in reverse entity map
            original_atom = reverse_entity_map.get(atom)
            if original_atom is not None:
                # Get node ID from original atom if available
                original_node_id = original_atom.get("monomer_node_id")
                if original_node_id is not None:
                    atom["monomer_node_id"] = original_node_id
            # If not found in entity map, check if it already has a node_id
            # (from port transfer above)
            elif atom.get("monomer_node_id") is None:
                # This shouldn't happen for atoms from input structures,
                # but if it does, we leave it unmarked
                pass

        # CRITICAL: Also preserve ports from OTHER nodes in merged structures
        # The above loops only handle ports for left_node_id and right_node_id.
        # When left/right are already merged polymers, we need to preserve
        # ports from other nodes that weren't directly involved in this connection.
        for atom in product.atoms:
            original_atom = reverse_entity_map.get(atom)
            if original_atom is not None:
                # Check if original atom had a port that wasn't already transferred
                original_port = original_atom.get("port")
                original_node_id = original_atom.get("monomer_node_id")

                # Only transfer if port exists and isn't already set on product atom
                if original_port and atom.get("port") is None:
                    atom["port"] = original_port
                    # Also transfer port metadata
                    from molpy.builder.polymer.port_utils import get_port_metadata

                    try:
                        metadata = get_port_metadata(original_atom, original_port)
                        set_port_metadata(
                            atom,
                            original_port,
                            role=metadata.get("role"),
                            bond_kind=metadata.get("bond_kind"),
                            compat=metadata.get("compat", set()),
                            priority=metadata.get("priority", 0),
                        )
                    except (KeyError, AttributeError):
                        # If metadata not available, just set port name
                        pass

        # Post-process: Clear ports from O atoms that have no bonded H
        # This handles dehydration-type reactions where -OH or -H is removed
        # The port atom's H was consumed, so the port should not be reused
        from molpy.reacter.selectors import find_neighbors as _find_neighbors

        for atom in product.atoms:
            if atom.get("symbol") == "O" and atom.get("port"):
                h_neighbors = _find_neighbors(product, atom, element="H")
                if not h_neighbors:
                    # This O had its H consumed - clear the port
                    del atom["port"]

        return product
