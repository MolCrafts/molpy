"""Build numpy-representable Block columns from sparse per-entity values.

The molrs Block Store is numpy-only (float / int / bool / str): object, None,
and ragged columns are rejected fail-fast. ``Atomistic`` / ``CoarseGrain``
attributes are sparse — not every atom/bead/link carries every key — so a naive
``np.array([atom.get(k) for ...])`` produces ``None``-bearing object arrays.

:func:`to_numpy_column` collapses a value list to a single numpy column, filling
gaps with a type-appropriate default (``0`` / ``0.0`` / ``""`` / ``False``), and
returns ``None`` when the values are not numpy-representable (all-None, ragged,
or heterogeneous object) so the caller can skip the column rather than overflow.
"""

from __future__ import annotations

from typing import Any

import numpy as np


def _is_bool(v: Any) -> bool:
    return isinstance(v, (bool, np.bool_))


def _is_int(v: Any) -> bool:
    return isinstance(v, (int, np.integer)) and not isinstance(v, bool)


def _is_float(v: Any) -> bool:
    return isinstance(v, (float, np.floating))


def to_numpy_column(values: list[Any]) -> np.ndarray | None:
    """Convert a per-entity value list to a numpy column, or ``None`` if unrepresentable.

    Missing entries (``None``) are filled with a typed default inferred from the
    present values: ``False`` for bool, ``0`` for int, ``0.0`` for float, ``""``
    for str. Mixed int/float promotes to float. Vector values (e.g. ``xyz``) are
    accepted only when dense and uniformly shaped. Returns ``None`` for all-None,
    ragged, or otherwise object-dtype columns so the caller can skip them.

    Args:
        values: One entry per entity; ``None`` marks a missing key.

    Returns:
        A numpy array (1-D or 2-D) of a Store-representable dtype, or ``None``.
    """
    present = [v for v in values if v is not None]
    if not present:
        return None

    if all(_is_bool(v) for v in present):
        return np.array(
            [bool(v) if v is not None else False for v in values], dtype=bool
        )
    if all(_is_int(v) for v in present):
        return np.array(
            [int(v) if v is not None else 0 for v in values], dtype=np.int64
        )
    if all(_is_int(v) or _is_float(v) for v in present):
        return np.array(
            [float(v) if v is not None else 0.0 for v in values], dtype=np.float64
        )
    if all(isinstance(v, str) for v in present):
        return np.array([str(v) if v is not None else "" for v in values])

    # Vector / array-valued column (e.g. xyz): representable only if every entry
    # is present and they stack to a NUMERIC dtype. Anything that numpy renders
    # as object or stringifies (e.g. a tuple of Entity handles → '<U…') is not a
    # real numeric column and is reported unrepresentable so the caller skips it.
    if any(v is None for v in values):
        return None
    try:
        arr = np.asarray(values)
    except (ValueError, TypeError):
        return None
    return arr if arr.dtype.kind in "fiub" else None
