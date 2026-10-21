"""Selectors: boolean row masks over a :class:`~molpy.Block`.

A selector picks rows by what a column holds (:class:`ColumnSelector`) or by
their row position (:class:`AtomIndexSelector`). Selecting by *where* an atom
is — a slab, a sphere, a shell — is a geometric region's job: every native
region (``mp.core.Cuboid``, ``mp.core.Sphere``, ``mp.core.HalfSpace``, their
``&`` / ``|`` / ``~`` compositions) has the same ``mask(block)`` surface, and a
selector composes with one through ``&`` / ``|`` on either side.
"""

from abc import ABC, abstractmethod
from typing import TYPE_CHECKING, Protocol

import numpy as np
from molrs.core import keys

if TYPE_CHECKING:
    from molrs.core import Block

__all__ = [
    "AtomIndexSelector",
    "ColumnSelector",
    "MaskPredicate",
]

#: A scalar a column can be compared with.
_Scalar = str | int | float | bool | np.generic


class _Masks(Protocol):
    """Anything with a row mask over a block: a selector or a molrs region."""

    def mask(self, block: "Block") -> np.ndarray: ...


class MaskPredicate(ABC):
    """Boolean mask producer combinable with ``&``, ``|``, ``~``.

    The other operand of ``&`` / ``|`` may be another selector or a molrs
    region (``mp.core.Sphere``, …): anything with ``mask(block)``.
    """

    @abstractmethod
    def mask(self, block: "Block") -> np.ndarray: ...

    def __call__(self, block: "Block") -> "Block":
        return block[self.mask(block)]

    def __and__(self, other: _Masks) -> "MaskPredicate":
        return _And(self, other)

    def __or__(self, other: _Masks) -> "MaskPredicate":
        return _Or(self, other)

    def __invert__(self) -> "MaskPredicate":
        return _Not(self)

    def __rand__(self, other: _Masks) -> "MaskPredicate":
        return _And(other, self)

    def __ror__(self, other: _Masks) -> "MaskPredicate":
        return _Or(other, self)


def _column_name(column: "str | keys.Key") -> str:
    """The plain column name of a ``str`` or a :class:`molrs.core.keys.Key`."""
    if isinstance(column, keys.Key):
        return column.key
    if isinstance(column, str):
        return column
    raise TypeError(f"column must be a str or a keys.Key, got {type(column).__name__}")


def _accepts(kind: str, value: _Scalar) -> bool:
    """Whether ``value`` has the type a column of numpy dtype ``kind`` holds."""
    if kind in "US":
        return isinstance(value, (str, np.str_))
    if kind == "b":
        return isinstance(value, (bool, np.bool_))
    if isinstance(value, (bool, np.bool_)):
        return False
    if kind in "iu":
        return isinstance(value, (int, np.integer))
    if kind == "f":
        return isinstance(value, (int, float, np.integer, np.floating))
    return False


class ColumnSelector(MaskPredicate):
    """Select rows whose ``column`` equals ``value``.

    The column defaults to :data:`molrs.core.keys.ELEMENT`; pass another
    ``keys`` constant for another column (``keys.TYPE`` for the string type
    label, ``keys.TYPE_ID`` for the numeric one, ``keys.RES_NAME``, …).

    The value must have the column's type: ``ColumnSelector(1,
    column=keys.TYPE)`` on a string ``type`` column raises ``TypeError``
    instead of silently selecting nothing. A missing column raises
    ``KeyError``. A null cell never matches its stored fill value. Masks are
    ordinary booleans, so negating a selector includes those unmatched rows.

    Args:
        value: The value to match (a string, integer, float or bool scalar).
        column: The column to compare, a ``keys`` constant or its name.
    """

    def __init__(self, value: _Scalar, column: "str | keys.Key" = keys.ELEMENT) -> None:
        if not isinstance(value, (str, int, float, bool, np.generic)):
            raise TypeError(
                f"ColumnSelector: value must be a scalar, got {type(value).__name__}"
            )
        self.value = value
        self.column = _column_name(column)

    def mask(self, block: "Block") -> np.ndarray:
        if self.column not in block:
            raise KeyError(
                f"ColumnSelector: block has no '{self.column}' column "
                f"(available: {list(block.keys())})"
            )
        data = block[self.column]
        if not _accepts(data.dtype.kind, self.value):
            raise TypeError(
                f"ColumnSelector: column '{self.column}' holds {data.dtype}, "
                f"not {type(self.value).__name__} ({self.value!r})"
            )
        matched = data == self.value
        validity = block.validity(self.column)
        return matched if validity is None else matched & validity

    def __repr__(self) -> str:
        return f"ColumnSelector({self.value!r}, column={self.column!r})"


class AtomIndexSelector(MaskPredicate):
    """Select rows by their 0-based row index.

    This is the row position in the block, not the ``id`` column (molrs's
    ``id`` is a label, never an index). Calling the selector on a block is
    ``block.select_rows(indices)``; ``mask`` marks the same rows.

    Args:
        indices: Non-negative row indices. Duplicates collapse and the
            selected rows keep the block's order.

    Raises:
        TypeError: If ``indices`` is not a 1-D sequence of integers.
        ValueError: If an index is negative, or (on use) past the block's end.
    """

    def __init__(self, indices: "list[int] | np.ndarray") -> None:
        array = np.asarray(indices)
        if array.ndim != 1 or (array.size and array.dtype.kind not in "iu"):
            raise TypeError("AtomIndexSelector: indices must be a 1-D integer sequence")
        if array.size and array.min() < 0:
            raise ValueError("AtomIndexSelector: row indices must be non-negative")
        if array.size and int(array.max()) > np.iinfo(np.int64).max:
            raise ValueError("AtomIndexSelector: row index exceeds int64 range")
        self.indices = np.unique(array.astype(np.int64))

    def _check(self, block: "Block") -> None:
        if self.indices.size and (
            self.indices[0] < 0 or self.indices[-1] >= block.n_rows
        ):
            raise ValueError(
                f"AtomIndexSelector: row index {int(self.indices[-1])} out of range "
                f"(n_rows={block.n_rows})"
            )

    def mask(self, block: "Block") -> np.ndarray:
        self._check(block)
        selected = np.zeros(block.n_rows, dtype=bool)
        selected[self.indices] = True
        return selected

    def __call__(self, block: "Block") -> "Block":
        self._check(block)
        return block[self.indices]

    def __repr__(self) -> str:
        return f"AtomIndexSelector({self.indices.tolist()!r})"


# ------------------------------------------------------------------ combinators
class _And(MaskPredicate):
    """Logical AND of two masks (selectors or regions)."""

    def __init__(self, a: _Masks, b: _Masks) -> None:
        self.a = a
        self.b = b

    def mask(self, block: "Block") -> np.ndarray:
        return np.asarray(self.a.mask(block)) & np.asarray(self.b.mask(block))


class _Or(MaskPredicate):
    """Logical OR of two masks (selectors or regions)."""

    def __init__(self, a: _Masks, b: _Masks) -> None:
        self.a = a
        self.b = b

    def mask(self, block: "Block") -> np.ndarray:
        return np.asarray(self.a.mask(block)) | np.asarray(self.b.mask(block))


class _Not(MaskPredicate):
    """Logical NOT of a predicate."""

    def __init__(self, a: _Masks) -> None:
        self.a = a

    def mask(self, block: "Block") -> np.ndarray:
        return ~np.asarray(self.a.mask(block))
