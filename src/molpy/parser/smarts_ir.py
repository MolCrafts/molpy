"""Intermediate Representation (IR) for SMARTS patterns.

Defines the data structures for SMARTS pattern representation:
- AtomPrimitiveIR: Single primitive atom pattern
- AtomExpressionIR: Logical combination of primitives
- SmartsAtomIR: Complete SMARTS atom with expression and label
- SmartsBondIR: Bond between two SMARTS atoms
- SmartsIR: Complete SMARTS pattern graph
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Literal


@dataclass(eq=True)
class AtomPrimitiveIR:
    """Represents a single primitive atom pattern in SMARTS."""

    type: Literal[
        "symbol",
        "atomic_num",
        "neighbor_count",
        "ring_connectivity",
        "ring_size",
        "ring_count",
        "has_label",
        "matches_smarts",
        "wildcard",
        "hydrogen_count",
        "implicit_hydrogen_count",
        "degree",
        "valence",
        "charge",
        "aromatic",
        "aliphatic",
        "chirality",
        "isotope",
        "atom_class",
    ]
    value: str | int | SmartsIR | None = None
    id: int = field(
        default_factory=lambda: id(AtomPrimitiveIR), compare=False, repr=False
    )

    def __hash__(self):
        return self.id

    def __repr__(self):
        if self.type == "wildcard":
            return "AtomPrimitiveIR(type='wildcard')"
        return f"AtomPrimitiveIR(type={self.type!r}, value={self.value!r})"


@dataclass(eq=True)
class AtomExpressionIR:
    """Represents logical expressions combining atom primitives.

    Operators:
        - 'and' (&): high-priority AND
        - 'or' (,): OR
        - 'weak_and' (;): low-priority AND
        - 'not' (!): negation
    """

    op: Literal["and", "or", "weak_and", "not", "primitive"]
    children: list[AtomPrimitiveIR | AtomExpressionIR] = field(default_factory=list)
    id: int = field(
        default_factory=lambda: id(AtomExpressionIR), compare=False, repr=False
    )

    def __hash__(self):
        return self.id

    def __repr__(self):
        if self.op == "primitive" and len(self.children) == 1:
            return f"AtomExpressionIR({self.children[0]!r})"
        return f"AtomExpressionIR(op={self.op!r}, children={self.children!r})"


@dataclass(eq=True)
class SmartsAtomIR:
    """Represents a complete SMARTS atom with expression and optional label."""

    expression: AtomExpressionIR | AtomPrimitiveIR
    label: int | None = None
    id: int = field(default_factory=lambda: id(SmartsAtomIR), compare=False, repr=False)

    def __hash__(self):
        return self.id

    def __repr__(self):
        if self.label is not None:
            return f"SmartsAtomIR(expression={self.expression!r}, label={self.label})"
        return f"SmartsAtomIR(expression={self.expression!r})"


@dataclass(eq=True)
class SmartsBondIR:
    """Represents a bond between two SMARTS atoms."""

    itom: SmartsAtomIR
    jtom: SmartsAtomIR
    bond_type: str = "implicit"

    def __repr__(self):
        expr_start = getattr(self.itom.expression, "value", str(self.itom.expression))
        expr_end = getattr(self.jtom.expression, "value", str(self.jtom.expression))
        return f"SmartsBondIR({expr_start!r}, {expr_end!r}, {self.bond_type!r})"


@dataclass(eq=True)
class SmartsIR:
    """Complete SMARTS pattern intermediate representation."""

    atoms: list[SmartsAtomIR] = field(default_factory=list)
    bonds: list[SmartsBondIR] = field(default_factory=list)

    def __repr__(self):
        return f"SmartsIR(atoms={self.atoms!r}, bonds={self.bonds!r})"
