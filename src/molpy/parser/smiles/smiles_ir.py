"""SMILES structural graph IR.

This module defines the intermediate representation for SMILES strings.
It represents only the chemical structure graph (atoms and bonds).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Literal

BondOrder = Literal[1, 2, 3, "ar"]

# Global counter for generating unique IDs
_id_counter = 0


def _generate_id() -> int:
    """Generate a unique ID for atoms and bonds."""
    global _id_counter
    _id_counter += 1
    return _id_counter


@dataclass(eq=True)
class SmilesAtomIR:
    """Intermediate representation for a SMILES atom."""

    id: int = field(default_factory=_generate_id, compare=False)
    element: str | None = None
    aromatic: bool = False
    formal_charge: int | None = None
    hydrogens: int | None = None
    extras: dict[str, Any] = field(default_factory=dict)

    def __hash__(self):
        return self.id

    def __repr__(self):
        attrs = [f"id={self.id}"]
        if self.element is not None:
            attrs.append(f"element={self.element!r}")
        if self.aromatic:
            attrs.append("aromatic=True")
        if self.formal_charge is not None:
            attrs.append(f"formal_charge={self.formal_charge}")
        if self.hydrogens is not None:
            attrs.append(f"hydrogens={self.hydrogens}")
        if self.extras:
            attrs.append(f"extras={self.extras}")
        return f"SmilesAtomIR({', '.join(attrs)})"


@dataclass(eq=True)
class SmilesBondIR:
    """Intermediate representation for a SMILES bond.

    Bonds directly reference AtomIR objects, not just IDs.
    """

    itom: SmilesAtomIR = field(compare=False)
    jtom: SmilesAtomIR = field(compare=False)
    order: BondOrder = 1
    stereo: Literal["/", "\\"] | None = None
    id: int = field(default_factory=_generate_id, compare=False)

    def __hash__(self):
        return self.id

    def __repr__(self):
        attrs = [
            f"id={self.id}",
            f"itom={self.itom.element!r}",
            f"jtom={self.jtom.element!r}",
            f"order={self.order!r}",
        ]
        if self.stereo is not None:
            attrs.append(f"stereo={self.stereo!r}")
        return f"SmilesBondIR({', '.join(attrs)})"


@dataclass(eq=True)
class SmilesGraphIR:
    """Root-level IR for SMILES parser.

    Represents a molecular graph with atoms and bonds.
    This is the output of the SMILES parser.
    """

    atoms: list[SmilesAtomIR] = field(default_factory=list)
    bonds: list[SmilesBondIR] = field(default_factory=list)

    def __repr__(self):
        return f"SmilesGraphIR(atoms={len(self.atoms)}, bonds={len(self.bonds)})"
