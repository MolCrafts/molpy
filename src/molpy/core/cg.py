from __future__ import annotations

from typing import Any

from .entity import Entity, Link


class Bead(Entity):
    """Coarse-grain bead; may include {"type": "X", "xyz": [...], "members": list[Entity] | None}."""


class Bond(Link):
    def __init__(self, a: Entity, b: Entity, /, **attrs: Any):
        super().__init__([a, b], **attrs)
