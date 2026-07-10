"""The one thing that differs between assembly jobs: which sites pair up.

Growing a polymer, crosslinking a melt and closing a macrocycle are three jobs
that ask MolPy for the same three inputs — which atoms may react, what the
reaction does, and which marked sites actually pair up. Only the third differs,
so only the third is polymorphic.

A :class:`Selector` receives the world and the sites the reaction already
matched (the assembler matches **once**, in linear time) and yields the bindings
it wants bonded. It never scans the system and never edits the graph.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import Iterator
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from molpy.builder.assembly._context import MatchContext

#: One matched occurrence of a reactant pattern: ``{map_number: atom handle}``.
type Binding = dict[int, int]


class Selector(ABC):
    """Choose which matched sites react — the assembler's only variation point.

    Subclasses see the world read-only through a
    :class:`~molpy.builder.assembly._context.MatchContext`, which already carries
    the reaction's matches. Pairing is theirs; matching is not.
    """

    @abstractmethod
    def select(self, context: MatchContext) -> Iterator[Binding]:
        """Yield ``{map_number: handle}`` bindings to react.

        The bindings must be **pairwise disjoint** in the atoms they name: every
        edit is applied to the same world, and two edits that share an atom would
        make the second act on handles the first invalidated. The assembler
        asserts this rather than trusting it.
        """

    @staticmethod
    def _atoms_of(binding: Binding) -> frozenset[int]:
        return frozenset(binding.values())
