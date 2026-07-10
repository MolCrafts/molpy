"""A typifier's receptive field, and the two radii it implies.

A region retype has **two** radii, and they are not independent degrees of
freedom — they are ``reach`` and (essentially) ``2 * reach``.

Let ``reach`` be the neighbourhood radius (in bonds) a typifier needs in order to
decide one atom's type.

*Write-back set.* An edit lands on ``touched``. Atom ``a``'s type can change iff
the edit falls inside ``a``'s deciding neighbourhood, i.e. ``touched`` intersects
``ball(a, reach)``. Distance is symmetric, so the atoms that must be retyped are
exactly ``ball(touched, reach)``.

*Extraction radius.* Every atom in that write-back set still needs its own
``reach``-ball in view. The farthest one sits ``reach`` hops out, and its ball
reaches ``reach`` hops further — so the extracted ball has radius ``2 * reach``.

:class:`TypeScope` is the **only** place in molpy that does this arithmetic.
"""

from __future__ import annotations

from collections.abc import Iterable
from dataclasses import dataclass
from typing import TYPE_CHECKING, ClassVar

if TYPE_CHECKING:
    from molpy.core.affected_region import AffectedRegion
    from molpy.core.atomistic import Atom, Atomistic


@dataclass(frozen=True)
class TypeScope:
    """The receptive field of a local typifier, in bonds.

    Attributes:
        reach: Neighbourhood radius (bond hops) that decides one atom's type.

    Example::

        scope = TypeScope(reach=2)
        scope.interior_reach   # -> 2
        scope.extract_radius   # -> 4
    """

    #: A dihedral / improper is a 4-body term, so a term containing a newly
    #: formed bond spans at most ``4 - 2 = 2`` hops from that bond. Those atoms
    #: must be typed for the term to be looked up, which floors the write-back
    #: radius. Not a magic number: it is the arity of the widest bonded term
    #: :class:`~molpy.core.atomistic.Atomistic` carries, minus two.
    TERM_REACH: ClassVar[int] = 2

    reach: int

    def __post_init__(self) -> None:
        if self.reach < 1:
            raise ValueError(f"reach must be >= 1, got {self.reach}")

    @property
    def interior_reach(self) -> int:
        """Radius of the write-back set: ``max(reach, TERM_REACH)``.

        Atoms within this distance of ``touched`` may change type, or are needed
        to type a bonded term the edit created.
        """
        return max(self.reach, self.TERM_REACH)

    @property
    def extract_radius(self) -> int:
        """Radius of the extracted ball: ``interior_reach + reach``.

        The outermost written-back atom still needs a full ``reach``-ball of
        context around itself.
        """
        return self.interior_reach + self.reach

    def region(self, graph: Atomistic, touched: Iterable[Atom | int]) -> AffectedRegion:
        """Extract the region this typifier needs around ``touched``.

        Args:
            graph: The parent graph an edit just modified.
            touched: Seed atoms (views or molrs handles) the edit reported.

        Returns:
            An :class:`~molpy.core.affected_region.AffectedRegion` whose
            ``interior`` is ``ball(touched, interior_reach)`` — the atoms whose
            types are written back — inside an extracted ball of radius
            ``extract_radius``.

        Raises:
            ValueError: if ``touched`` is empty or names a handle that is not a
                live atom of ``graph``.
        """
        from molpy.core.affected_region import AffectedRegion

        return AffectedRegion._from(
            graph,
            touched,
            extract_radius=self.extract_radius,
            interior_reach=self.interior_reach,
        )
