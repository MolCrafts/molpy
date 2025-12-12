"""CGSmiles structural graph IR.

This module defines the intermediate representation for CGSmiles strings.
It represents coarse-grained molecular structures with labeled nodes and fragments.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Literal

BondOrder = Literal[0, 1, 2, 3, 4]  # 0=none, 1=single, 2=double, 3=triple, 4=aromatic

# Global counter for generating unique IDs
_id_counter = 0


def _generate_id() -> int:
    """Generate a unique ID for nodes and bonds."""
    global _id_counter
    _id_counter += 1
    return _id_counter


@dataclass(eq=False)
class CGSmilesNodeIR:
    """Intermediate representation for a CGSmiles node.
    
    A coarse-grained node with a label (e.g., "PEO", "PMA") and optional annotations.
    """

    id: int = field(default_factory=_generate_id)
    label: str = ""  # e.g., "PEO", "PMA"
    annotations: dict[str, str] = field(default_factory=dict)  # e.g., {"q": "1"}

    def __eq__(self, other):
        if not isinstance(other, CGSmilesNodeIR):
            return False
        return self.id == other.id

    def __hash__(self):
        return self.id

    def __repr__(self):
        attrs = [f"id={self.id}"]
        if self.label:
            attrs.append(f"label={self.label!r}")
        if self.annotations:
            attrs.append(f"annotations={self.annotations}")
        return f"CGSmilesNodeIR({', '.join(attrs)})"


@dataclass(eq=True)
class CGSmilesBondIR:
    """Intermediate representation for a CGSmiles bond.

    Bonds directly reference NodeIR objects, not just IDs.
    """

    node_i: CGSmilesNodeIR = field(compare=False)
    node_j: CGSmilesNodeIR = field(compare=False)
    order: BondOrder = 1
    id: int = field(default_factory=_generate_id, compare=False)

    def __hash__(self):
        return self.id

    def __repr__(self):
        attrs = [
            f"id={self.id}",
            f"node_i={self.node_i.label!r}",
            f"node_j={self.node_j.label!r}",
            f"order={self.order!r}",
        ]
        return f"CGSmilesBondIR({', '.join(attrs)})"


@dataclass(eq=True)
class CGSmilesGraphIR:
    """Coarse-grained graph representation.

    Represents a molecular graph with CG nodes and bonds.
    """

    nodes: list[CGSmilesNodeIR] = field(default_factory=list)
    bonds: list[CGSmilesBondIR] = field(default_factory=list)

    def __repr__(self):
        return f"CGSmilesGraphIR(nodes={len(self.nodes)}, bonds={len(self.bonds)})"


@dataclass(eq=True)
class CGSmilesFragmentIR:
    """Fragment definition.
    
    Maps a fragment name to its SMILES or CGSmiles representation.
    """

    name: str = ""  # e.g., "OH", "PEO"
    body: str = ""  # SMILES or CG fragment string

    def __repr__(self):
        return f"CGSmilesFragmentIR(name={self.name!r}, body={self.body!r})"


@dataclass(eq=True)
class CGSmilesIR:
    """Root-level IR for CGSmiles parser.

    Represents a complete CGSmiles string with base graph and fragment definitions.
    This is the output of the CGSmiles parser.
    """

    base_graph: CGSmilesGraphIR = field(default_factory=CGSmilesGraphIR)
    fragments: list[CGSmilesFragmentIR] = field(default_factory=list)

    def __repr__(self):
        return (
            f"CGSmilesIR("
            f"base_graph={self.base_graph}, "
            f"fragments={len(self.fragments)})"
        )
