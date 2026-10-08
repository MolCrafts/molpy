"""Selectors: boolean row masks over a :class:`~molpy.Block` by column value.

A selector picks rows by what a column holds (a type, an element, an atom
id). Selecting by *where* an atom is — a slab, a sphere, a shell — is a
geometric region's job: every native region (``mp.core.Cuboid``, ``mp.core.Sphere``,
``mp.core.HalfSpace``, their ``&`` / ``|`` / ``~`` compositions) has the same
``mask(block)`` and ``region(block)`` surface, and a selector composes with
one through ``&`` / ``|``.
"""

from abc import ABC, abstractmethod
from typing import TYPE_CHECKING

import numpy as np

if TYPE_CHECKING:
    from molrs.core import Block

__all__ = [
    "AtomIndexSelector",
    "AtomTypeSelector",
    "ElementSelector",
    "MaskPredicate",
]


class MaskPredicate(ABC):
    """Boolean mask producer combinable with &, |, ~."""

    @abstractmethod
    def mask(self, block: "Block") -> np.ndarray: ...

    def __call__(self, block: "Block") -> "Block":
        return block[self.mask(block)]

    # Compositional logic
    def __and__(self, other: "MaskPredicate") -> "MaskPredicate":
        return _And(self, other)

    def __or__(self, other: "MaskPredicate") -> "MaskPredicate":
        return _Or(self, other)

    def __invert__(self) -> "MaskPredicate":
        return _Not(self)

    __rand__ = __and__
    __ror__ = __or__


class AtomTypeSelector(MaskPredicate):
    """Select atoms by their type (integer or string)."""

    def __init__(self, atom_type: int | str, field: str = "type") -> None:
        """
        Initialize atom type Selector.

        Args:
            atom_type: The atom type to select (integer or string)
            field: The field name containing atom types (default: "type")
        """
        self.atom_type = atom_type
        self.field = field

    def mask(self, block: "Block") -> np.ndarray:
        if self.field not in block:
            raise KeyError(f"Field '{self.field}' not found in block")
        return block[self.field] == self.atom_type


class AtomIndexSelector(MaskPredicate):
    """Select atoms by their indices."""

    def __init__(self, indices: list[int] | np.ndarray, id_field: str = "id") -> None:
        """
        Initialize atom index Selector.

        Args:
            indices: List or array of atom indices to select
            id_field: The field name containing atom IDs (default: "id")
        """
        # Convert to numpy array and validate
        if isinstance(indices, list):
            self.indices = np.array(indices, dtype=int)
        elif isinstance(indices, np.ndarray):
            self.indices = indices.astype(int)
        else:
            raise TypeError("indices must be a list[int] or np.ndarray")

        self.id_field = id_field

    def mask(self, block: "Block") -> np.ndarray:
        if self.id_field not in block:
            raise KeyError(
                f"AtomIndexSelector: block has no '{self.id_field}' column "
                f"(available: {list(block.keys())})"
            )
        return np.isin(block[self.id_field], self.indices)


class ElementSelector(MaskPredicate):
    """Select atoms by their element symbol."""

    def __init__(self, element: str, field: str = "element"):
        """
        Initialize element Selector.

        Args:
            element: The element symbol to select (e.g., "C", "H", "O")
            field: The field name containing element symbols (default: "element")
        """
        if not isinstance(element, str):
            raise TypeError("element must be a string")
        self.element = element
        self.field = field

    def mask(self, block: "Block") -> np.ndarray:
        if self.field not in block:
            raise KeyError(
                f"ElementSelector: block has no '{self.field}' column "
                f"(available: {list(block.keys())})"
            )
        return block[self.field] == self.element


# ------------------------------------------------------------------ combinators
class _And(MaskPredicate):
    """Logical AND combination of two predicates."""

    def __init__(self, a: MaskPredicate, b: MaskPredicate):
        self.a = a
        self.b = b

    def mask(self, block: "Block") -> np.ndarray:
        return self.a.mask(block) & self.b.mask(block)


class _Or(MaskPredicate):
    """Logical OR combination of two predicates."""

    def __init__(self, a: MaskPredicate, b: MaskPredicate):
        self.a = a
        self.b = b

    def mask(self, block: "Block") -> np.ndarray:
        return self.a.mask(block) | self.b.mask(block)


class _Not(MaskPredicate):
    """Logical NOT of a predicate."""

    def __init__(self, a: MaskPredicate):
        self.a = a

    def mask(self, block: "Block") -> np.ndarray:
        return ~self.a.mask(block)
