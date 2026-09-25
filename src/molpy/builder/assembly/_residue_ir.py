"""Residue topology IR for polymer assembly.

Pure dataclasses — nodes are monomer labels, edges are residue adjacency.
Built directly or by the :mod:`molpy.builder.assembly._residue_graph`
constructors (``linear_topology`` / ``ring_topology`` / ``star_topology``).
This module never parses text: chemistry notation is parsed by the native core.
"""

from __future__ import annotations

import itertools
from dataclasses import dataclass, field

_node_ids = itertools.count(1)


@dataclass(eq=False)
class ResidueNode:
    """One residue (one monomer copy) in a residue topology.

    Two nodes are equal only when they share an ``id``; two ``"EO"`` nodes
    are distinct residues. Construct as ``ResidueNode(label)`` or
    ``ResidueNode(label, id=...)`` (``id`` is keyword-only).

    Attributes:
        label: Monomer library key this residue expands to.
        id: Process-unique node identity; auto-assigned from a process-wide
            counter when omitted. It is an identity, not a residue number:
            the residue number stamped as ``RES_ID`` is the node's 1-based
            position in :attr:`ResidueTopology.nodes`.
    """

    label: str
    id: int = field(default_factory=_node_ids.__next__, kw_only=True)

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, ResidueNode):
            return NotImplemented
        return self.id == other.id

    def __hash__(self) -> int:
        return self.id


@dataclass(eq=False)
class ResidueBond:
    """Undirected edge between two topology nodes.

    Two bonds are equal when they join the same pair of node ids, in either
    order.

    Attributes:
        node_i: One endpoint.
        node_j: The other endpoint.
    """

    node_i: ResidueNode
    node_j: ResidueNode

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, ResidueBond):
            return NotImplemented
        return {self.node_i.id, self.node_j.id} == {other.node_i.id, other.node_j.id}

    def __hash__(self) -> int:
        return hash(frozenset((self.node_i.id, self.node_j.id)))


@dataclass(eq=True)
class ResidueTopology:
    """Residue graph: nodes are monomers, bonds are adjacency for assembly.

    Validated at construction; equality compares nodes and bonds in order.

    Attributes:
        nodes: Residues, in residue-id order.
        bonds: Residue adjacencies; every endpoint is one of ``nodes``.

    Raises:
        ValueError: if a node label is empty, two nodes share an id, a bond
            joins a node to itself, or a bond endpoint is not in ``nodes``.
    """

    nodes: list[ResidueNode] = field(default_factory=list)
    bonds: list[ResidueBond] = field(default_factory=list)

    def __post_init__(self) -> None:
        ids: set[int] = set()
        for node in self.nodes:
            if not node.label:
                raise ValueError(f"residue node {node.id} has an empty label")
            if node.id in ids:
                raise ValueError(f"residue node id {node.id} appears twice")
            ids.add(node.id)
        for bond in self.bonds:
            i, j = bond.node_i.id, bond.node_j.id
            if i == j:
                raise ValueError(f"residue bond joins node {i} to itself")
            missing = {i, j} - ids
            if missing:
                raise ValueError(
                    f"residue bond {i}-{j} references node(s) "
                    f"{sorted(missing)} outside the topology"
                )
