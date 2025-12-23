"""Module for SMARTSGraph and SMARTS matching logic."""

import itertools
from collections import OrderedDict

from igraph import Graph, plot

from molpy.core.element import Element
from molpy.parser.smarts import (
    AtomExpressionIR,
    AtomPrimitiveIR,
    SmartsIR,
    SmartsParser,
)


class SMARTSGraph(Graph):
    """A graph representation of a SMARTS pattern.

    This class supports two modes of construction:
    1. From SMARTS string (legacy mode)
    2. From predicates (new predicate-based mode)

    Attributes
    ----------
    atomtype_name : str
        The atom type this pattern assigns
    priority : int
        Priority for conflict resolution (higher wins)
    target_vertices : list[int]
        Which pattern vertices should receive the atom type (empty = all)
    source : str
        Source identifier for debugging
    smarts_string : str | None
        The SMARTS string (if constructed from string)
    ir : SmartsIR | None
        The intermediate representation (if constructed from string)

    Notes
    -----
    SMARTSGraph inherits from igraph.Graph

    Vertex attributes:
        - preds: list[Callable] - list of predicates that must all pass

    Edge attributes:
        - preds: list[Callable] - list of predicates that must all pass

    Graph attributes:
        - atomtype_name: str
        - priority: int
        - target_vertices: list[int]
        - source: str
        - specificity_score: int (computed)
    """

    def __init__(
        self,
        smarts_string: str | None = None,
        parser: SmartsParser | None = None,
        name: str | None = None,
        atomtype_name: str | None = None,
        priority: int = 0,
        target_vertices: list[int] | None = None,
        source: str = "",
        overrides: set | None = None,
        *args,
        **kwargs,
    ):
        super().__init__(*args, **kwargs)

        # Metadata
        self.atomtype_name = atomtype_name or name or ""
        self._priority = priority  # Use _priority as internal storage
        self.target_vertices = target_vertices or []
        self.source = source
        self.overrides = overrides

        # Dependency tracking
        self.dependencies: set[str] = (
            set()
        )  # Set of atom type names this pattern depends on
        self.level: int | None = None  # Topological level (0 = no deps, 1+ = has deps)

        # Legacy support
        self.smarts_string = smarts_string
        self.ir: SmartsIR | None = None

        if smarts_string is not None:
            # Legacy mode: construct from SMARTS string
            if parser is None:
                self.ir = SmartsParser().parse_smarts(smarts_string)
            else:
                self.ir = parser.parse_smarts(smarts_string)

            self._atom_indices = OrderedDict()
            self._add_nodes()
            self._add_edges()

            # Extract dependencies from SMARTS string
            self.dependencies = self.extract_dependencies()

        self._graph_matcher = None
        self._specificity_score: int | None = None

    @property
    def priority(self) -> int:
        """Get priority for conflict resolution (higher wins)."""
        return self._priority

    @classmethod
    def from_igraph(
        cls,
        graph: Graph,
        atomtype_name: str,
        priority: int = 0,
        target_vertices: list[int] | None = None,
        source: str = "",
    ) -> "SMARTSGraph":
        """Create SmartsGraph from an existing igraph.Graph.

        Args:
            graph: igraph.Graph with vertex/edge predicates
            atomtype_name: Atom type this pattern assigns
            priority: Priority for conflict resolution
            target_vertices: Which vertices should be typed (empty = all)
            source: Source identifier

        Returns:
            SMARTSGraph instance
        """
        # Create empty instance
        instance = cls(
            atomtype_name=atomtype_name,
            priority=priority,
            target_vertices=target_vertices or [],
            source=source,
        )

        # Copy graph structure and attributes
        instance.add_vertices(graph.vcount())
        if graph.ecount() > 0:
            instance.add_edges(graph.get_edgelist())

        # Copy vertex attributes
        for attr in graph.vs.attributes():
            instance.vs[attr] = graph.vs[attr]

        # Copy edge attributes
        for attr in graph.es.attributes():
            instance.es[attr] = graph.es[attr]

        return instance

    def __repr__(self):
        if self.smarts_string:
            return f"<SmartsGraph({self.smarts_string})>"
        return f"<SmartsGraph(name={self.atomtype_name}, vertices={self.vcount()}, edges={self.ecount()})>"

    def get_specificity_score(self) -> int:
        """Compute specificity score for this pattern.

        Scoring heuristic:
            +0 per element predicate (baseline)
            +1 per charge/degree/hyb constraint
            +2 per aromatic/in_ring constraint
            +3 per bond order predicate
            +4 per custom predicate

        Returns:
            Specificity score (higher = more specific)
        """
        if self._specificity_score is not None:
            return self._specificity_score

        score = 0

        # Score vertex predicates
        for v in self.vs:
            preds = v["preds"] if "preds" in v.attributes() else []
            for pred in preds:
                if hasattr(pred, "meta"):
                    score += pred.meta.weight

        # Score edge predicates
        for e in self.es:
            preds = e["preds"] if "preds" in e.attributes() else []
            for pred in preds:
                if hasattr(pred, "meta"):
                    score += pred.meta.weight

        self._specificity_score = score
        return score

    def plot(self, *args, **kwargs):
        """Plot the SMARTS graph."""
        graph = Graph(edges=self.get_edgelist())
        graph.vs["label"] = [v.index for v in self.vs]
        return plot(graph, *args, **kwargs)

    def override(self, overrides):
        """Set the priority of this SMART"""
        self.overrides = overrides
        # Legacy behavior: compute priority from overrides
        # Now priority is set explicitly, but keep this for compatibility
        if hasattr(self, "_priority"):
            # New mode: use explicit priority
            pass
        else:
            # Legacy mode: compute from overrides
            if self.overrides:
                self._priority = max([override.priority for override in overrides]) + 1

    def get_priority(self) -> int:
        """Get priority value (supports both new and legacy modes)."""
        if hasattr(self, "_priority"):
            return self._priority
        # Legacy: compute from overrides
        if self.overrides is None:
            return 0
        return max([override.priority for override in self.overrides]) + 1

    def extract_dependencies(self) -> set[str]:
        """Extract type references from SMARTS IR.

        Finds all has_label primitives that reference atom types (e.g., %opls_154).
        These are parsed by Lark as AtomPrimitiveIR(type="has_label", value="%opls_154").

        Returns:
            Set of referenced atom type names (e.g., {'opls_154', 'opls_135'})
        """
        if not self.ir or not self.ir.atoms:
            return set()

        dependencies = set()

        def extract_from_expr(expr):
            """Recursively extract dependencies from expression."""
            if isinstance(expr, AtomPrimitiveIR):
                if expr.type == "has_label" and isinstance(expr.value, str):
                    # has_label value is like "%opls_154"
                    label = expr.value
                    if label.startswith("%opls_"):
                        # Strip the % to get "opls_154"
                        dependencies.add(label[1:])
            elif isinstance(expr, AtomExpressionIR):
                for child in expr.children:
                    extract_from_expr(child)

        # Extract from all atoms
        for atom in self.ir.atoms:
            extract_from_expr(atom.expression)

        return dependencies

    def _add_nodes(self):
        """Add all atoms in the SMARTS IR as nodes in the graph."""
        atoms = self.ir.atoms
        self.add_vertices(len(atoms), {"atom": atoms})
        for i, atom in enumerate(atoms):
            self._atom_indices[id(atom)] = i

    def _add_edges(self):
        """Add all bonds in the SMARTS IR as edges in the graph."""
        atom_indices = self._atom_indices
        for bond in self.ir.bonds:
            start_idx = atom_indices[id(bond.start)]
            end_idx = atom_indices[id(bond.end)]
            self.add_edge(start_idx, end_idx, bond_type=bond.bond_type)

    def _node_match_fn(self, g1, g2, v1, v2):
        """Determine if two graph nodes are equal.

        This method supports both legacy (SMARTS IR) and new (predicate) modes.
        """
        host = g1.vs[v1]
        pattern = g2.vs[v2]

        # New predicate mode
        if "preds" in pattern.attributes():
            preds = pattern["preds"]
            host_attrs = host.attributes()
            return all(pred(host_attrs) for pred in preds)

        # Legacy mode (SMARTS IR)
        if "atom" in pattern.attributes():
            atom_ir = pattern["atom"]
            neighbors = g1.neighbors(v1)
            result = self._atom_expr_matches(atom_ir.expression, host, neighbors, g1)
            return result

        # No constraints - match anything
        return True

    def _edge_match_fn(self, g1, g2, e1, e2):
        """Determine if two graph edges are equal.

        This method supports both legacy (bond_type) and new (predicate) modes.
        """
        host_edge = g1.es[e1]
        pattern_edge = g2.es[e2]

        # New predicate mode
        if "preds" in pattern_edge.attributes():
            preds = pattern_edge["preds"]
            host_attrs = host_edge.attributes()
            return all(pred(host_attrs) for pred in preds)

        # Legacy mode (bond_type)
        if "bond_type" in pattern_edge.attributes():
            # Simple bond type matching for now
            # TODO: Implement full bond type matching logic
            return True

        # No constraints - match anything
        return True

    def _atom_expr_matches(
        self, atom_expr: AtomExpressionIR | AtomPrimitiveIR, atom, bond_partners, graph
    ):
        """Evaluate SMARTS IR expressions."""
        # Handle AtomPrimitiveIR directly
        if isinstance(atom_expr, AtomPrimitiveIR):
            return self._atom_primitive_matches(atom_expr, atom, bond_partners, graph)

        # Handle AtomExpressionIR
        if atom_expr.op == "not":
            return not self._atom_expr_matches(
                atom_expr.children[0], atom, bond_partners, graph
            )
        elif atom_expr.op in ("and", "weak_and"):
            result = True
            for child in atom_expr.children:
                result = result and self._atom_expr_matches(
                    child, atom, bond_partners, graph
                )
            return result
        elif atom_expr.op == "or":
            for child in atom_expr.children:
                if self._atom_expr_matches(child, atom, bond_partners, graph):
                    return True
            return False
        elif atom_expr.op == "primitive":
            if atom_expr.children:
                return self._atom_expr_matches(
                    atom_expr.children[0], atom, bond_partners, graph
                )
            return True
        else:
            raise TypeError(f"Unexpected atom expression op: {atom_expr.op}")

    @staticmethod
    def _atom_primitive_matches(
        atom_primitive: AtomPrimitiveIR, atom, bond_partners, graph
    ):
        """Compare atomic primitives against atom properties."""
        atomic_num = atom.attributes().get("number", None)
        atom_name = atom.attributes().get("name", None)
        atom_idx = atom.index
        assert atomic_num or atom_name, f"Atom {atom_idx} has no atomic number or name."

        if atom_primitive.type == "atomic_num":
            assert isinstance(atom_primitive.value, int)
            return atomic_num == atom_primitive.value
        elif atom_primitive.type == "symbol":
            symbol_val = str(atom_primitive.value)
            if symbol_val == "*":
                return True
            elif symbol_val.startswith("_"):
                # Store non-element elements in .name
                return atom_name == symbol_val
            else:
                # Handle lowercase (aromatic) symbols
                if symbol_val.islower():
                    # Check both element and aromaticity
                    # For now, just check element
                    return atomic_num == Element(symbol_val.upper()).number
                return atomic_num == Element(symbol_val).number
        elif atom_primitive.type == "wildcard":
            return True
        elif atom_primitive.type == "has_label":
            # Type reference (e.g., %opls_154)
            label = str(atom_primitive.value)
            if label.startswith("%opls_"):
                # This is a type reference - check if atom has this type assigned
                required_type = label[1:]  # Strip % to get "opls_154"
                assigned_type = atom.attributes().get("atomtype")
                return assigned_type == required_type
            else:
                # Legacy behavior: check if label is in type attribute
                label = label[1:]  # Strip the % sign
                vertex = graph.vs[atom_idx]
                vertex_type = vertex["type"] if "type" in vertex.attributes() else []
                return label in vertex_type
        elif atom_primitive.type == "neighbor_count":
            assert isinstance(atom_primitive.value, int)
            return len(bond_partners) == atom_primitive.value
        elif atom_primitive.type == "ring_size":
            assert isinstance(atom_primitive.value, int)
            cycle_len = atom_primitive.value
            vertex = graph.vs[atom_idx]
            cycles = vertex["cycles"] if "cycles" in vertex.attributes() else []
            return any(len(cycle) == cycle_len for cycle in cycles)
        elif atom_primitive.type == "ring_count":
            assert isinstance(atom_primitive.value, int)
            vertex = graph.vs[atom_idx]
            cycles = vertex["cycles"] if "cycles" in vertex.attributes() else []
            n_cycles = len(cycles)
            return n_cycles == atom_primitive.value
        elif atom_primitive.type == "hydrogen_count":
            # Explicit hydrogen count (H2): count H atoms bonded to this atom
            assert isinstance(atom_primitive.value, int)
            h_count = 0
            for partner_idx in bond_partners:
                partner_vertex = graph.vs[partner_idx]
                partner_num = partner_vertex.get("number", None)
                if partner_num == 1:  # Hydrogen has atomic number 1
                    h_count += 1
            return h_count == atom_primitive.value
        elif atom_primitive.type == "implicit_hydrogen_count":
            # Implicit hydrogen count (h2): this is typically used for SMILES
            # In SMARTS context, we treat it similarly to explicit hydrogen count
            # but this might need refinement based on actual usage
            assert isinstance(atom_primitive.value, int)
            h_count = 0
            for partner_idx in bond_partners:
                partner_vertex = graph.vs[partner_idx]
                partner_num = partner_vertex.get("number", None)
                if partner_num == 1:  # Hydrogen has atomic number 1
                    h_count += 1
            return h_count == atom_primitive.value
        elif atom_primitive.type == "matches_smarts":
            raise NotImplementedError(
                "Recursive SMARTS (matches_smarts) is not yet implemented"
            )
        else:
            raise ValueError(f"Unknown atom primitive type: {atom_primitive.type}")

    def find_matches(self, graph):
        """Return sets of atoms that match this SMARTS pattern in a topology.

        Parameters
        ----------
        structure : TopologyGraph
            The topology that we are trying to atomtype.
        typemap : dict
            The target typemap being used/edited

        Notes
        -----
        When this function gets used in atomtyper.py, we actively modify the
        white- and blacklists of the atoms in `topology` after finding a match.
        This means that between every successive call of
        `subgraph_isomorphisms_iter()`, the topology against which we are
        matching may have actually changed. Currently, we take advantage of this
        behavior in some edges cases (e.g. see `test_hexa_coordinated` in
        `test_smarts.py`).

        """

        self.calc_signature(graph)

        self._graph_matcher = SMARTSMatcher(
            graph,
            self,
            node_match_fn=self._node_match_fn,
            edge_match_fn=self._edge_match_fn,
        )

        matches = self._graph_matcher.subgraph_isomorphisms()
        match_index = set([match[0] for match in matches])
        return match_index

    def calc_signature(self, graph):
        """Calculate graph signatures for pattern matching."""

        # Check if any atoms have ring-related properties
        def check_expr_for_rings(expr):
            """Recursively check expression for ring-related primitives."""
            if isinstance(expr, AtomPrimitiveIR):
                return expr.type in ("ring_size", "ring_count")
            if isinstance(expr, AtomExpressionIR):
                return any(check_expr_for_rings(child) for child in expr.children)
            return False

        has_ring_rules = any(
            check_expr_for_rings(atom.expression) for atom in self.ir.atoms
        )

        if has_ring_rules:
            graph.vs["cycles"] = [set() for _ in graph.vs]
            all_cycles = _find_chordless_cycles(graph, max_cycle_size=6)
            for i, cycles in enumerate(all_cycles):
                for cycle in cycles:
                    graph.vs[i]["cycles"].add(tuple(cycle))


class SMARTSMatcher:
    """Inherits and implements VF2 for a SMARTSGraph."""

    def __init__(self, G1: Graph, G2: Graph, node_match_fn, edge_match_fn=None):
        self.G1 = G1
        self.G2 = G2
        self.node_match_fn = node_match_fn
        self.edge_match_fn = edge_match_fn

    @property
    def is_isomorphic(self):
        """Return True if the two graphs are isomorphic."""
        return self.G1.isomorphic(self.G2)

    def subgraph_isomorphisms(self):
        """Iterate over all subgraph isomorphisms between G1 and G2."""
        # Build edge compatibility function if provided
        edge_compat_fn = None
        if self.edge_match_fn is not None:

            def edge_compat_fn(g1, g2, e1, e2):
                return self.edge_match_fn(g1, g2, e1, e2)

        matches = self.G1.get_subisomorphisms_vf2(
            self.G2, node_compat_fn=self.node_match_fn, edge_compat_fn=edge_compat_fn
        )
        results = []
        for sgi in matches:
            sg = self.G1.subgraph(sgi)
            if sg.get_isomorphisms_vf2(self.G2):
                results.append(sgi)
        return results

    def candidate_pairs_iter(self):
        """Iterate over candidate pairs of nodes in G1 and G2."""
        # All computations are done using the current state!
        G2_nodes = self.G2_nodes

        # First we compute the inout-terminal sets.
        T1_inout = set(self.inout_1.keys()) - set(self.core_1.keys())
        T2_inout = set(self.inout_2.keys()) - set(self.core_2.keys())

        # If T1_inout and T2_inout are both nonempty.
        # P(s) = T1_inout x {min T2_inout}
        if T1_inout and T2_inout:
            for node in T1_inout:
                yield node, min(T2_inout)
        else:
            # First we determine the candidate node for G2
            other_node = min(G2_nodes - set(self.core_2))
            host_nodes = self.valid_nodes if other_node == 0 else self.G1.nodes()
            for node in host_nodes:
                if node not in self.core_1:
                    yield node, other_node

        # For all other cases, we don't have any candidate pairs.


def _find_chordless_cycles(graph, max_cycle_size):
    """Find all chordless cycles (i.e. rings) in the bond graph.

    Traverses the bond graph to determine all cycles (i.e. rings) each
    atom is contained within. Algorithm has been adapted from:
    https://stackoverflow.com/questions/4022662/find-all-chordless-cycles-in-an-undirected-graph/4028855#4028855
    """
    cycles = [[] for _ in graph.vs]

    """
    For all nodes we need to find the cycles that they are included within.
    """
    for i, node in enumerate(graph.vs):
        node_idx = node.index
        neighbors = list(graph.neighbors(node_idx))
        pairs = list(itertools.combinations(neighbors, 2))
        """
        Loop over all pairs of neighbors of the node. We will see if a ring
        exists that includes these branches.
        """
        for pair in pairs:
            """
            We need to store all node sequences that could be rings. We will
            update this as we traverse the graph.
            """
            connected = False
            possible_rings = []

            last_node = pair[0]
            ring = [last_node, node_idx, pair[1]]
            possible_rings.append(ring)

            if graph.are_adjacent(last_node, pair[1]):
                cycles[i].append(ring)
                connected = True

            while not connected:
                """
                Branch and create a new list of possible rings
                """
                new_possible_rings = []
                for possible_ring in possible_rings:
                    next_neighbors = graph.neighbors(possible_ring[-1])
                    for next_neighbor in next_neighbors:
                        if next_neighbor != possible_ring[-2]:
                            new_possible_rings.append([*possible_ring, next_neighbor])
                possible_rings = new_possible_rings

                for possible_ring in possible_rings:
                    if graph.are_adjacent(possible_ring[-1], last_node):
                        if any(
                            [
                                graph.are_adjacent(possible_ring[-1], internal_node)
                                for internal_node in possible_ring[1:-2]
                            ]
                        ):
                            pass
                        else:
                            cycles[i].append(possible_ring)
                            connected = True

                if not possible_rings or len(possible_rings[0]) == max_cycle_size:
                    break

    return cycles
