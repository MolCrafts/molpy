"""Deterministic crosslinker (crosslink-01) — no RNG, no conversion target.

Three deterministic ways to pick crosslink points:

- **exhaustive** (default): react every candidate once, in a reproducible order
  (100% of what topology / ``cutoff`` allows);
- **spacing=K**: thin each molecule's sites to every ``K``-th along the backbone
  (uniform crosslink points, pure topology);
- **pairs**: form exactly the named ``(site_a, site_b)`` bonds.
"""

from __future__ import annotations

from collections.abc import Iterator, Sequence
from typing import TYPE_CHECKING

from ._crosslinker import Candidate, Crosslinker

if TYPE_CHECKING:
    from molpy.typifier.atomistic import ForceFieldTypifier


class DeterministicCrosslinker(Crosslinker):
    """Reproducible crosslinker with no randomness and no conversion cutoff."""

    def __init__(
        self,
        reaction: str,
        *,
        cutoff: float | None = None,
        spacing: int | None = None,
        pairs: Sequence[tuple[int, int]] | None = None,
        exclude_same_molecule: bool = False,
        exclude_same_match: bool = False,
        typifier: ForceFieldTypifier | None = None,
    ) -> None:
        super().__init__(reaction, cutoff=cutoff, typifier=typifier)
        self._spacing = spacing
        self._pairs = pairs
        self._exclude_same_molecule = exclude_same_molecule
        self._exclude_same_match = exclude_same_match

    def select(self, graph, candidates: list[Candidate]) -> Iterator[dict[int, int]]:
        if self._pairs is not None:
            yield from self._select_explicit(graph)
            return

        eligible = candidates
        if self._spacing is not None:
            keep = self._regular_sites(graph, self._spacing)
            eligible = [
                c
                for c in candidates
                if all(handle in keep for handle in self._bond_atoms(c))
            ]

        yield from self._select_exhaustive(graph, eligible)

    # -- exhaustive: consume every candidate once, deterministically ---------

    def _select_exhaustive(
        self, graph, candidates: list[Candidate]
    ) -> Iterator[dict[int, int]]:
        components = self._components(graph)
        consumed: set[int] = set()
        for candidate in sorted(candidates, key=self._sort_key):
            atoms = self._occ_atoms(candidate.occ_a) | self._occ_atoms(candidate.occ_b)
            if atoms & consumed:
                continue
            if self._exclude_same_match and self._same_match(candidate):
                continue
            if self._exclude_same_molecule and self._same_molecule(
                components, candidate
            ):
                continue
            yield self._binding(candidate)
            consumed |= atoms

    # -- explicit: form exactly the named site pairs -------------------------

    def _select_explicit(self, graph) -> Iterator[dict[int, int]]:
        assert self._pairs is not None
        sites_a = self._ordered_sites(graph, self._comp_a, self._map_a)
        sites_b = self._ordered_sites(graph, self._comp_b, self._map_b)
        for i, j in self._pairs:
            candidate = Candidate(
                sites_a[i], sites_b[j], self._comp_a, self._comp_b, 0.0
            )
            yield self._binding(candidate)
