"""Adapter utilities for converting molecules to graphs for matching.

This module provides functions to convert Atomistic structures into
igraph.Graph representations suitable for SMARTS pattern matching.
"""

from typing import Any

from igraph import Graph

from molpy.core import Atom, Atomistic, Bond


def build_mol_graph(
    structure: "Atomistic",
) -> tuple[Graph, dict[int, int], dict[int, int]]:
    """Convert Atomistic structure to igraph.Graph for matching.

    Args:
        structure: Atomistic structure with atoms and bonds

    Returns:
        Tuple of (graph, vs_to_atomid, atomid_to_vs) where:
        - graph: igraph.Graph with vertex/edge attributes
        - vs_to_atomid: mapping from vertex index to atom ID
        - atomid_to_vs: mapping from atom ID to vertex index

    Vertex attributes set:
        - element: str (e.g., "C", "N", "O")
        - number: int (atomic number)
        - is_aromatic: bool
        - charge: int
        - degree: int (number of bonds)
        - hyb: int | None (1=sp, 2=sp2, 3=sp3)
        - in_ring: bool
        - cycles: set of tuples (ring membership)

    Edge attributes set:
        - order: int | str (1, 2, 3, or ":")
        - is_aromatic: bool
        - is_in_ring: bool
    """

    # Type check
    if not isinstance(structure, Atomistic):
        raise TypeError(f"Expected Atomistic structure, got {type(structure).__name__}")

    atoms = list(structure.atoms)
    bonds = list(structure.bonds)

    # Create vertex index mappings
    # Use stable ordering based on entity storage order
    vs_to_atomid = {}
    atomid_to_vs = {}
    atom_to_vs = {}

    for i, atom in enumerate(atoms):
        atom_id = id(atom)  # Use Python id as stable identifier
        vs_to_atomid[i] = atom_id
        atomid_to_vs[atom_id] = i
        atom_to_vs[atom] = i

    # Create graph
    g = Graph(n=len(atoms), directed=False)

    # Set vertex attributes
    for i, atom in enumerate(atoms):
        attrs = _extract_atom_attributes(atom, structure)
        # Add atom_id for type assignment lookup
        attrs["atom_id"] = id(atom)
        for key, value in attrs.items():
            g.vs[i][key] = value

    # Add edges
    edge_list = []
    edge_attrs_list = []

    for bond in bonds:
        itom_idx = atom_to_vs.get(bond.itom)
        jtom_idx = atom_to_vs.get(bond.jtom)

        if itom_idx is not None and jtom_idx is not None:
            edge_list.append((itom_idx, jtom_idx))
            edge_attrs = _extract_bond_attributes(bond, structure)
            edge_attrs_list.append(edge_attrs)

    if edge_list:
        g.add_edges(edge_list)

        # Set edge attributes
        for eid, attrs in enumerate(edge_attrs_list):
            for key, value in attrs.items():
                g.es[eid][key] = value

    # Compute derived properties
    _compute_derived_properties(g, structure, atom_to_vs)

    return g, vs_to_atomid, atomid_to_vs


def _extract_atom_attributes(atom: "Atom", structure: "Atomistic") -> dict[str, Any]:
    """Extract attributes from an Atom entity.

    Args:
        atom: Atom entity
        structure: Parent Atomistic structure

    Returns:
        Dict of attributes for graph vertex
    """
    attrs = {}

    # Element symbol
    symbol = atom.get("symbol", atom.get("element", None))
    if symbol:
        attrs["element"] = str(symbol).upper()

        # Try to get atomic number
        try:
            from molpy.core.element import Element

            elem = Element(symbol)
            attrs["number"] = elem.number
        except:
            # If element lookup fails, use symbol as-is
            attrs["number"] = None
    else:
        # Check for atomic number directly
        number = atom.get("number", atom.get("atomic_number", None))
        if number is not None:
            attrs["number"] = int(number)
            try:
                from molpy.core.element import Element

                elem = Element(number)
                attrs["element"] = elem.symbol
            except:
                attrs["element"] = "*"
        else:
            attrs["element"] = "*"
            attrs["number"] = None

    # Aromaticity
    attrs["is_aromatic"] = bool(atom.get("is_aromatic", atom.get("aromatic", False)))

    # Charge
    charge_val = atom.get("charge", atom.get("formal_charge", 0))
    attrs["charge"] = int(charge_val) if charge_val is not None else 0

    # Hybridization
    hyb = atom.get("hyb", atom.get("hybridization", None))
    if hyb is not None:
        # Normalize to int (1=sp, 2=sp2, 3=sp3)
        if isinstance(hyb, str):
            hyb_map = {"sp": 1, "sp1": 1, "sp2": 2, "sp3": 3}
            hyb = hyb_map.get(hyb.lower(), None)
        attrs["hyb"] = int(hyb) if hyb is not None else None
    else:
        attrs["hyb"] = None

    # Initialize degree (will be computed later)
    attrs["degree"] = 0

    # Initialize ring properties (will be computed later)
    attrs["in_ring"] = False
    attrs["cycles"] = set()

    return attrs


def _extract_bond_attributes(bond: "Bond", structure: "Atomistic") -> dict[str, Any]:
    """Extract attributes from a Bond entity.

    Args:
        bond: Bond entity
        structure: Parent Atomistic structure

    Returns:
        Dict of attributes for graph edge
    """
    attrs = {}

    # Bond order
    order = bond.get("order", bond.get("bond_order", 1))
    if isinstance(order, str):
        # Handle aromatic bonds
        if order in (":", "aromatic", "ar"):
            attrs["order"] = ":"
            attrs["is_aromatic"] = True
        else:
            try:
                attrs["order"] = int(order)
                attrs["is_aromatic"] = False
            except ValueError:
                attrs["order"] = 1
                attrs["is_aromatic"] = False
    else:
        attrs["order"] = int(order)
        attrs["is_aromatic"] = bool(
            bond.get("is_aromatic", bond.get("aromatic", False))
        )

    # Ring membership (will be computed later)
    attrs["is_in_ring"] = False

    return attrs


def _compute_derived_properties(
    g: Graph, structure: "Atomistic", atom_to_vs: dict
) -> None:
    """Compute derived properties like degree and ring membership.

    Args:
        g: Graph with basic attributes
        structure: Atomistic structure
        atom_to_vs: Mapping from Atom to vertex index
    """
    # Compute degree
    for v in g.vs:
        v["degree"] = g.degree(v.index)

    # Compute ring membership using simple cycle detection
    _detect_rings(g)


def _detect_rings(g: Graph, max_ring_size: int = 8) -> None:
    """Detect rings and mark ring membership.

    This uses a simple algorithm to find small rings (up to max_ring_size).
    For each atom, we find all cycles it participates in.

    Args:
        g: Graph to analyze
        max_ring_size: Maximum ring size to detect
    """
    # Find all simple cycles up to max_ring_size
    # This is a simplified version - for production use a proper cycle detection

    # Mark atoms in any cycle
    for v in g.vs:
        # Try to find cycles containing this vertex
        cycles = _find_cycles_containing_vertex(g, v.index, max_ring_size)
        if cycles:
            v["in_ring"] = True
            v["cycles"] = set(tuple(sorted(cycle)) for cycle in cycles)
        else:
            v["in_ring"] = False
            v["cycles"] = set()

    # Mark edges in rings
    for e in g.es:
        source_cycles = (
            g.vs[e.source]["cycles"]
            if "cycles" in g.vs[e.source].attributes()
            else set()
        )
        target_cycles = (
            g.vs[e.target]["cycles"]
            if "cycles" in g.vs[e.target].attributes()
            else set()
        )

        # Edge is in ring if both endpoints share a cycle
        shared = source_cycles & target_cycles
        e["is_in_ring"] = len(shared) > 0


def _find_cycles_containing_vertex(g: Graph, v: int, max_size: int) -> list[list[int]]:
    """Find all cycles containing vertex v up to max_size.

    This is a simplified DFS-based cycle finder.

    Args:
        g: Graph
        v: Vertex index
        max_size: Maximum cycle size

    Returns:
        List of cycles (each cycle is a list of vertex indices)
    """
    cycles = []
    visited_paths = set()

    def dfs(path: list[int]):
        """DFS to find cycles."""
        if len(path) > max_size:
            return

        current = path[-1]
        neighbors = g.neighbors(current)

        for neighbor in neighbors:
            if neighbor == v and len(path) >= 3:
                # Found a cycle back to start
                # Normalize cycle to avoid duplicates
                cycle = path[:]
                cycle_tuple = tuple(sorted(cycle))
                if cycle_tuple not in visited_paths:
                    visited_paths.add(cycle_tuple)
                    cycles.append(cycle)
            elif neighbor not in path:
                # Continue DFS
                dfs([*path, neighbor])

    # Start DFS from v
    dfs([v])

    return cycles
