"""Coarse-grained residue topology IR for polymer assembly.

These are pure dataclasses — no grammar. Topology is built by
:mod:`molpy.builder.assembly._residue_graph` constructors, not by parsing a
chemistry notation string; molrs owns SMILES/SMARTS.
"""

from __future__ import annotations

from dataclasses import dataclass, field

_id_counter = 0


def _generate_id() -> int:
    global _id_counter
    _id_counter += 1
    return _id_counter


@dataclass(eq=False)
class CGSmilesNodeIR:
    """One coarse-grained residue node (label is a library key)."""

    id: int = field(default_factory=_generate_id)
    label: str = ""
    annotations: dict[str, str] = field(default_factory=dict)

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, CGSmilesNodeIR):
            return False
        return self.id == other.id

    def __hash__(self) -> int:
        return self.id


@dataclass(eq=True)
class CGSmilesBondIR:
    """Undirected edge between two topology nodes."""

    node_i: CGSmilesNodeIR = field(compare=False)
    node_j: CGSmilesNodeIR = field(compare=False)
    order: int = 1
    id: int = field(default_factory=_generate_id, compare=False)

    def __hash__(self) -> int:
        return self.id


@dataclass(eq=True)
class CGSmilesGraphIR:
    """Residue graph: nodes are monomers, bonds are adjacency for assembly."""

    nodes: list[CGSmilesNodeIR] = field(default_factory=list)
    bonds: list[CGSmilesBondIR] = field(default_factory=list)
