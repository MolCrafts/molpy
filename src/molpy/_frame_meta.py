"""Frame metadata helpers — thin wrappers around the dict-like ``Frame.meta``."""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

from molrs import Frame


def get_frame_meta(frame: Frame, key: str, default: Any = None) -> Any:
    """Return one metadata entry, or *default* when the key is absent."""
    return frame.meta.get(key, default)


def update_frame_meta(frame: Frame, entries: Mapping[str, Any]) -> None:
    """Merge *entries* into ``frame.meta`` (write-through)."""
    frame.meta.update(entries)
