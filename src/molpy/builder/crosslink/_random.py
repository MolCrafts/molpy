"""Random crosslinker (crosslink-02) — Flory-Stockmayer style network growth.

``RandomCrosslinker`` overrides only :meth:`select`: it shuffles the base-class
candidate pairs with a seeded ``numpy.random.RandomState`` and consumes them up
to a target ``conversion``, honouring ``cutoff`` (inherited), ``exclude_*`` and
``max_per_molecule``. Everything else — copy, molrs matching, distance filtering,
graph edit — is inherited unchanged.
"""

from __future__ import annotations

from collections.abc import Iterator
from typing import TYPE_CHECKING

import numpy

from ._crosslinker import Candidate, Crosslinker, SelectionContext

if TYPE_CHECKING:
    from molpy.typifier.region import RegionTypifier


class RandomCrosslinker(Crosslinker):
    """Random crosslinker to a target conversion, reproducible via ``seed``."""

    def __init__(
        self,
        reaction: str,
        *,
        conversion: float = 1.0,
        seed: int | None = None,
        cutoff: float | None = None,
        exclude_same_molecule: bool = False,
        exclude_same_match: bool = False,
        max_per_molecule: tuple[str, int] | int | None = None,
        typifier: RegionTypifier | None = None,
    ) -> None:
        super().__init__(reaction, cutoff=cutoff, typifier=typifier)
        self._conversion = conversion
        self._seed = seed
        self._exclude_same_molecule = exclude_same_molecule
        self._exclude_same_match = exclude_same_match
        self._max_per_molecule = max_per_molecule

    def select(self, context: SelectionContext) -> Iterator[dict[int, int]]:
        rng = numpy.random.RandomState(self._seed)
        order = rng.permutation(len(context.candidates))
        target = self._target_reactions(context)
        components = context.components
        consumed: set[int] = set()
        per_molecule: dict[int, int] = {}
        reacted = 0
        # The loop is bounded by the (finite) candidate list, so a conversion
        # that can never be reached simply stops when candidates run out.
        for index in order:
            if reacted >= target:
                break
            candidate = context.candidates[int(index)]
            if self._skip(candidate, consumed, components, per_molecule):
                continue
            yield self._binding(candidate)
            self._mark(candidate, consumed, components, per_molecule)
            reacted += 1

    # -- conversion target ---------------------------------------------------

    def _target_reactions(self, context: SelectionContext) -> float:
        """``conversion`` x limiting-reactant sites (A x B -> min, A x A -> half)."""
        occurrences = context.occurrences
        sites_a = {oa[self._map_a] for oa in occurrences[self._comp_a]}
        sites_b = {ob[self._map_b] for ob in occurrences[self._comp_b]}
        if sites_a and sites_a == sites_b:
            limiting = len(sites_a) / 2.0  # self-reaction: two sites per bond
        else:
            limiting = float(min(len(sites_a), len(sites_b)))
        return self._conversion * limiting

    # -- per-candidate gate --------------------------------------------------

    def _skip(
        self,
        candidate: Candidate,
        consumed: set[int],
        components: dict[int, int],
        per_molecule: dict[int, int],
    ) -> bool:
        atoms = self._occ_atoms(candidate.occ_a) | self._occ_atoms(candidate.occ_b)
        if atoms & consumed:
            return True
        if self._exclude_same_match and self._same_match(candidate):
            return True
        if self._exclude_same_molecule and self._same_molecule(components, candidate):
            return True
        return self._over_limit(candidate, components, per_molecule)

    def _limit(self) -> int | None:
        if self._max_per_molecule is None:
            return None
        if isinstance(self._max_per_molecule, tuple):
            return self._max_per_molecule[1]
        return self._max_per_molecule

    def _site_increments(
        self, candidate: Candidate, components: dict[int, int]
    ) -> dict[int, int]:
        """Sites (one per bonding atom) this reaction consumes, keyed by molecule."""
        ha, hb = self._bond_atoms(candidate)
        increments: dict[int, int] = {}
        for handle in (ha, hb):
            root = components.get(handle, handle)
            increments[root] = increments.get(root, 0) + 1
        return increments

    def _over_limit(
        self,
        candidate: Candidate,
        components: dict[int, int],
        per_molecule: dict[int, int],
    ) -> bool:
        limit = self._limit()
        if limit is None:
            return False
        for root, increment in self._site_increments(candidate, components).items():
            if per_molecule.get(root, 0) + increment > limit:
                return True
        return False

    def _mark(
        self,
        candidate: Candidate,
        consumed: set[int],
        components: dict[int, int],
        per_molecule: dict[int, int],
    ) -> None:
        consumed.update(self._occ_atoms(candidate.occ_a))
        consumed.update(self._occ_atoms(candidate.occ_b))
        for root, increment in self._site_increments(candidate, components).items():
            per_molecule[root] = per_molecule.get(root, 0) + increment
