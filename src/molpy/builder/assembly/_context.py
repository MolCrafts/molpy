"""What a selector is handed: the world and the sites already matched.

The assembler matches the reaction's reactant patterns **once**, in linear time,
and passes the result down. Pairing those matches is the selector's job and the
only thing that varies between assembly jobs, so the O(sites²) work — if a
selector needs any — lives there, never in the kernel.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

from molpy.builder.assembly._selector import Binding

if TYPE_CHECKING:
    from molpy.core.atomistic import Atomistic


@dataclass(frozen=True)
class MatchContext:
    """Immutable view of the reaction's matches against one world.

    Attributes:
        world: The graph about to be edited (read-only for a selector).
        occurrences: One list of bindings per reactant component of the reaction.
        map_a: Map number of the forming bond's first endpoint.
        map_b: Map number of the forming bond's second endpoint.
        comp_a: Index into ``occurrences`` of the component holding ``map_a``.
        comp_b: Index into ``occurrences`` of the component holding ``map_b``.
    """

    world: Atomistic
    occurrences: list[list[Binding]]
    map_a: int
    map_b: int
    comp_a: int
    comp_b: int

    def sites(self, component: int, map_number: int) -> list[Binding]:
        """Deduplicated representative occurrence per site atom, ordered by handle."""
        representative: dict[int, Binding] = {}
        for occurrence in self.occurrences[component]:
            handle = occurrence[map_number]
            representative.setdefault(handle, occurrence)
        return [representative[handle] for handle in sorted(representative)]
