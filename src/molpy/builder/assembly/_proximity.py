"""Selectors that pair sites by how close they are.

This is where crosslinking lives. A :class:`ProximitySelector` builds the
candidate pairs — the only O(sites²) step in assembly, and the reason the
assembler does not build them for everyone — and a subclass decides which of
them react.

Splitting the old ``DeterministicCrosslinker(spacing=…, pairs=…)`` into one class
per rule makes the illegal states unrepresentable: there is no longer a
constructor that accepts two mutually exclusive knobs and silently prefers one.
"""

from __future__ import annotations

import math
from abc import abstractmethod
from collections.abc import Iterator, Sequence
from dataclasses import dataclass
from typing import TYPE_CHECKING

import numpy as np

import molrs
from molpy.builder.assembly._selector import Binding, Selector

if TYPE_CHECKING:
    from molpy.builder.assembly._context import MatchContext
    from molpy.core.atomistic import Atomistic

# Non-periodic padding (Angstrom) so every point sits strictly inside the
# synthesized bounding box used for the neighbor search.
_BOX_MARGIN = 1.0


@dataclass(frozen=True)
class Candidate:
    """One cross-component occurrence pairing considered for reaction.

    ``distance`` is the separation of the two bonding atoms, or ``None`` when the
    graph carries no coordinates. ``None`` is not ``0.0``: a fake distance that
    looks real would silently collapse every distance ordering.
    """

    occ_a: Binding
    occ_b: Binding
    distance: float | None

    def binding(self) -> Binding:
        return {**self.occ_a, **self.occ_b}


class ProximitySelector(Selector):
    """Pair sites within ``cutoff`` of one another; subclasses choose which react.

    With ``cutoff=None`` every cross-component pairing is a candidate and the
    graph needs no coordinates (topological pairing) — a named mode, not a
    fallback. With a ``cutoff`` the graph **must** have coordinates.
    """

    def __init__(
        self,
        *,
        cutoff: float | None = None,
        exclude_same_molecule: bool = False,
        exclude_same_match: bool = True,
    ) -> None:
        self._cutoff = cutoff
        self._exclude_same_molecule = exclude_same_molecule
        self._exclude_same_match = exclude_same_match

    # -- the subclass hook ---------------------------------------------------

    @abstractmethod
    def choose(
        self, context: MatchContext, candidates: list[Candidate]
    ) -> Iterator[Binding]:
        """Turn the candidate pairings into the bindings that react.

        Subclasses order or filter ``candidates`` and hand them to
        :meth:`_consume`, which enforces the disjointness the assembler requires.
        """

    # -- template ------------------------------------------------------------

    def select(self, context: MatchContext) -> Iterator[Binding]:
        yield from self.choose(context, self._candidates(context))

    def _consume(
        self,
        candidates: Iterator[Candidate] | list[Candidate],
        limit: int | None = None,
    ) -> Iterator[Binding]:
        """Yield bindings for candidates that share no atom with an earlier one.

        A site atom may only be consumed once — two edits that share an atom
        would make the second act on handles the first invalidated.
        """
        used: set[int] = set()
        reacted = 0
        for candidate in candidates:
            if limit is not None and reacted >= limit:
                return
            atoms = self._atoms_of(candidate.occ_a) | self._atoms_of(candidate.occ_b)
            if atoms & used:
                continue
            used |= atoms
            reacted += 1
            yield candidate.binding()

    # -- candidate construction (the O(sites^2) step, owned here) ------------

    def _candidates(self, context: MatchContext) -> list[Candidate]:
        occ_a = context.occurrences[context.comp_a]
        occ_b = context.occurrences[context.comp_b]
        if not occ_a or not occ_b:
            return []
        raw = (
            self._cutoff_candidates(context, occ_a, occ_b)
            if self._cutoff is not None
            else self._all_candidates(context, occ_a, occ_b)
        )
        components = (
            self._components(context.world) if self._exclude_same_molecule else {}
        )
        kept: list[Candidate] = []
        for candidate in raw:
            ha, hb = candidate.occ_a[context.map_a], candidate.occ_b[context.map_b]
            if ha == hb:
                continue
            if self._exclude_same_match and self._atoms_of(
                candidate.occ_a
            ) == self._atoms_of(candidate.occ_b):
                continue
            if self._exclude_same_molecule and components.get(ha) == components.get(hb):
                continue
            kept.append(candidate)
        kept.sort(key=self._sort_key)
        return kept

    def _all_candidates(
        self, context: MatchContext, occ_a: list[Binding], occ_b: list[Binding]
    ) -> list[Candidate]:
        out: list[Candidate] = []
        for oa in occ_a:
            for ob in occ_b:
                distance = self._pair_distance(
                    context.world, oa[context.map_a], ob[context.map_b]
                )
                out.append(Candidate(oa, ob, distance))
        return out

    def _cutoff_candidates(
        self, context: MatchContext, occ_a: list[Binding], occ_b: list[Binding]
    ) -> list[Candidate]:
        coords_a = [self._xyz(context.world, oa[context.map_a]) for oa in occ_a]
        coords_b = [self._xyz(context.world, ob[context.map_b]) for ob in occ_b]
        if any(c is None for c in coords_a) or any(c is None for c in coords_b):
            raise ValueError(
                "cutoff-based pairing requires atom coordinates, but the graph has "
                "atoms without x/y/z; drop cutoff for topological pairing"
            )
        points_a = np.asarray(coords_a, dtype=float)
        points_b = np.asarray(coords_b, dtype=float)
        box = self._bounding_box(np.vstack([points_a, points_b]))
        nlist = molrs.NeighborQuery(box, points_b, self._cutoff).query(points_a)
        pairs = nlist.pairs()
        distances = nlist.distances
        return [
            Candidate(
                occ_a[int(pairs[row, 0])],
                occ_b[int(pairs[row, 1])],
                float(distances[row]),
            )
            for row in range(pairs.shape[0])
        ]

    def _bounding_box(self, points: np.ndarray) -> molrs.Box:
        margin = (self._cutoff or 0.0) + _BOX_MARGIN
        lo = points.min(axis=0) - margin
        hi = points.max(axis=0) + margin
        return molrs.Box.ortho(hi - lo, lo, np.array([False, False, False]))

    # -- ordering ------------------------------------------------------------

    @staticmethod
    def _sort_key(candidate: Candidate) -> tuple[int, float, tuple[int, ...]]:
        """Distance first when it exists; otherwise a deterministic handle order.

        A ``None`` distance sorts as its own class, so a coordinate-free graph
        yields the same bindings on every run instead of an arbitrary tie-break
        among fake zero distances.
        """
        handles = tuple(sorted({*candidate.occ_a.values(), *candidate.occ_b.values()}))
        if candidate.distance is None:
            return (1, 0.0, handles)
        return (0, candidate.distance, handles)

    # -- geometry ------------------------------------------------------------

    @staticmethod
    def _xyz(graph: Atomistic, handle: int) -> tuple[float, float, float] | None:
        x, y, z = (graph.get(handle, k) for k in ("x", "y", "z"))
        if x is None or y is None or z is None:
            return None
        return (float(x), float(y), float(z))

    def _pair_distance(self, graph: Atomistic, ha: int, hb: int) -> float | None:
        pa, pb = self._xyz(graph, ha), self._xyz(graph, hb)
        if pa is None or pb is None:
            return None
        return math.dist(pa, pb)

    # -- topology ------------------------------------------------------------

    @staticmethod
    def _adjacency(graph: Atomistic) -> tuple[list[int], dict[int, list[int]]]:
        handles = [atom.handle for atom in graph.atoms]
        adjacency: dict[int, list[int]] = {h: [] for h in handles}
        for bond in graph.bonds:
            i, j = bond.itom.handle, bond.jtom.handle
            adjacency.setdefault(i, []).append(j)
            adjacency.setdefault(j, []).append(i)
        return handles, adjacency

    def _components(self, graph: Atomistic) -> dict[int, int]:
        """Connected-component root per atom handle (BFS over bonds)."""
        handles, adjacency = self._adjacency(graph)
        root: dict[int, int] = {}
        for start in handles:
            if start in root:
                continue
            root[start] = start
            stack = [start]
            while stack:
                current = stack.pop()
                for neighbor in adjacency.get(current, ()):
                    if neighbor not in root:
                        root[neighbor] = start
                        stack.append(neighbor)
        return root


class ExhaustiveSelector(ProximitySelector):
    """React every candidate pairing, nearest first. No RNG, no conversion."""

    def choose(
        self, context: MatchContext, candidates: list[Candidate]
    ) -> Iterator[Binding]:
        return self._consume(candidates)


class ExplicitPairSelector(ProximitySelector):
    """React exactly the named ``(handle_a, handle_b)`` bonding-atom pairs.

    This is the rule a polymer builder degenerates to once its topology is
    resolved to atom handles — see
    :class:`~molpy.builder.assembly._topology.TopologySelector`.
    """

    def __init__(self, pairs: Sequence[tuple[int, int]], **kwargs: object) -> None:
        super().__init__(**kwargs)  # type: ignore[arg-type]
        self._pairs = [tuple(p) for p in pairs]

    def choose(
        self, context: MatchContext, candidates: list[Candidate]
    ) -> Iterator[Binding]:
        by_bond = {
            frozenset(
                (candidate.occ_a[context.map_a], candidate.occ_b[context.map_b])
            ): candidate
            for candidate in candidates
        }
        missing = {frozenset(p) for p in self._pairs} - set(by_bond)
        if missing:
            raise ValueError(
                f"explicit pair(s) {sorted(sorted(m) for m in missing)} are not "
                "matched site pairings of this reaction"
            )
        return self._consume(by_bond[frozenset(pair)] for pair in self._pairs)


class SpacingSelector(ProximitySelector):
    """Thin each molecule's sites to every ``spacing``-th along its backbone.

    Sites are ordered by topological distance from a chain end, so the surviving
    crosslink points are uniformly spaced. Pure topology — coordinates are never
    read for the thinning itself.
    """

    def __init__(self, spacing: int, **kwargs: object) -> None:
        if spacing < 1:
            raise ValueError(f"spacing must be >= 1, got {spacing}")
        super().__init__(**kwargs)  # type: ignore[arg-type]
        self._spacing = spacing

    def choose(
        self, context: MatchContext, candidates: list[Candidate]
    ) -> Iterator[Binding]:
        keep = self._regular_sites(context)
        return self._consume(
            candidate
            for candidate in candidates
            if candidate.occ_a[context.map_a] in keep
            and candidate.occ_b[context.map_b] in keep
        )

    def _regular_sites(self, context: MatchContext) -> set[int]:
        sites = {oa[context.map_a] for oa in context.occurrences[context.comp_a]}
        sites |= {ob[context.map_b] for ob in context.occurrences[context.comp_b]}
        _, adjacency = self._adjacency(context.world)
        components = self._components(context.world)

        by_molecule: dict[int, list[int]] = {}
        for handle in sites:
            by_molecule.setdefault(components[handle], []).append(handle)

        keep: set[int] = set()
        for root, site_handles in by_molecule.items():
            ordered = self._backbone_order(
                context.world, adjacency, components, root, site_handles
            )
            keep.update(ordered[:: self._spacing])
        return keep

    @staticmethod
    def _backbone_order(
        graph: Atomistic,
        adjacency: dict[int, list[int]],
        components: dict[int, int],
        root: int,
        site_handles: Sequence[int],
    ) -> list[int]:
        molecule_atoms = [h for h, r in components.items() if r == root]
        # Chain end = lowest-degree atom (deterministic tie-break on handle).
        end = min(molecule_atoms, key=lambda h: (len(adjacency.get(h, [])), h))
        distances = {h: d for h, d in graph.topo_distances(end)}
        return sorted(site_handles, key=lambda h: (distances.get(h, 0), h))
