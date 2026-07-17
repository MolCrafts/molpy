"""Random pairing to a target conversion — Flory–Stockmayer style networks."""

from __future__ import annotations

import math
from collections.abc import Iterator
from typing import TYPE_CHECKING

import numpy as np

from molpy.builder.assembly._proximity import Candidate, ProximitySelector
from molpy.builder.assembly._selector import Binding

if TYPE_CHECKING:
    from molpy.builder.assembly._context import MatchContext


class RandomSelector(ProximitySelector):
    """Shuffle the candidates and consume them to a target ``conversion``.

    ``conversion`` is the fraction of the *limiting* reactant's sites that react,
    so ``conversion=1.0`` means every site of the scarcer species is consumed.
    A given ``seed`` reproduces a network exactly.

    Everything else — matching, cutoff filtering, disjointness — is inherited.
    This class supplies a sampling rule and nothing else.
    """

    def __init__(
        self,
        *,
        conversion: float = 1.0,
        seed: int | None = None,
        cutoff: float | None = None,
        max_per_molecule: int | None = None,
        **kwargs: object,
    ) -> None:
        if not 0.0 <= conversion <= 1.0:
            raise ValueError(f"conversion must be in [0, 1], got {conversion}")
        super().__init__(cutoff=cutoff, **kwargs)  # type: ignore[arg-type]
        self._conversion = conversion
        self._seed = seed
        self._max_per_molecule = max_per_molecule

    def choose(
        self, context: MatchContext, candidates: list[Candidate]
    ) -> Iterator[Binding]:
        rng = np.random.RandomState(self._seed)
        order = rng.permutation(len(candidates))
        shuffled = [candidates[i] for i in order]
        return self._consume(
            self._within_molecule_limit(context, shuffled),
            limit=self._target_reactions(context),
        )

    def _target_reactions(self, context: MatchContext) -> int:
        """Sites of the limiting reactant that must react."""
        n_a = len({oa[context.map_a] for oa in context.occurrences[context.comp_a]})
        n_b = len({ob[context.map_b] for ob in context.occurrences[context.comp_b]})
        return math.ceil(self._conversion * min(n_a, n_b))

    def _within_molecule_limit(
        self, context: MatchContext, candidates: list[Candidate]
    ) -> Iterator[Candidate]:
        if self._max_per_molecule is None:
            yield from candidates
            return
        components = self._components(context.world)
        counts: dict[int, int] = {}
        for candidate in candidates:
            ha = candidate.occ_a[context.map_a]
            hb = candidate.occ_b[context.map_b]
            roots = {components[ha], components[hb]}
            if any(counts.get(r, 0) >= self._max_per_molecule for r in roots):
                continue
            for r in roots:
                counts[r] = counts.get(r, 0) + 1
            yield candidate
