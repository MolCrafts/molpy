"""Exact-dtype metadata operations for the canonical :mod:`molrs` Frame.

The native ``Frame.meta`` getter returns a snapshot dictionary.  Updating an
entry therefore means replacing the map with a new dictionary of
``molrs.MetaValue`` objects; plain Python values are deliberately rejected by
molrs and are never coerced here.
"""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

from molrs import Frame, MetaValue


def get_frame_meta(frame: Frame, key: str, default: Any = None) -> Any:
    """Return the Python value of one typed metadata entry."""
    entry = frame.meta.get(key)
    return default if entry is None else entry.value


def update_frame_meta(frame: Frame, entries: Mapping[str, MetaValue]) -> None:
    """Replace ``frame.meta`` with its current entries plus *entries*."""
    frame.meta = {**frame.meta, **entries}
