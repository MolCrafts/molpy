"""Frame metadata helpers — thin wrappers around the dict-like ``Frame.meta``."""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

from molrs import Frame, MetaValue


def _unwrap_meta(value: Any) -> Any:
    """molrs 0.14 stores ``Frame.meta`` as ``MetaValue``; IO wants the Python payload."""
    if isinstance(value, MetaValue):
        return value.value
    return value


def get_frame_meta(frame: Frame, key: str, default: Any = None) -> Any:
    """Return one metadata payload, or *default* when the key is absent.

    ``Frame.meta[key]`` is a ``MetaValue`` on molrs 0.14. Callers of this
    helper (writers, format façades) receive the unwrapped Python value.
    """
    if key not in frame.meta:
        return default
    return _unwrap_meta(frame.meta[key])


def update_frame_meta(frame: Frame, entries: Mapping[str, Any]) -> None:
    """Merge *entries* into ``frame.meta``.

    molrs 0.14 ``Frame.meta`` is a cloned ``dict[str, MetaValue]``; in-place
    ``update`` does not write through. Replace the map instead.
    """
    current = dict(frame.meta)
    current.update(entries)
    frame.meta = current
