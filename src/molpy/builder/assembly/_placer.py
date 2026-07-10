"""Geometry: where the pieces sit before the reaction joins them.

Placement answers "do these two components have meaningful relative
coordinates?", which is a fact about the *input*, not about which builder you
reached for. A packed melt already has them and must not be disturbed; fresh
template copies land on top of one another and must be laid out. So a placer is a
constructor argument, not a subclass.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from molpy.builder.assembly._selector import Binding
    from molpy.core.atomistic import Atomistic


class Placer(ABC):
    """Position the components a set of bindings is about to join.

    Called once, before any reaction is applied, with the bindings the selector
    chose. Implementations mutate ``world``'s coordinates in place.
    """

    @abstractmethod
    def place(self, world: Atomistic, bindings: list[Binding]) -> None:
        """Move components so each binding's two site atoms sit at bonding range.

        The distance used is an initial guess — the sum of the two atoms' covalent
        radii — because the equilibrium bond length is the force field's answer
        and the force field has not been applied yet. A geometry optimisation
        downstream converges it. Guess the value, never the identity: an unknown
        element raises rather than defaulting to carbon.
        """
