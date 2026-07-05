"""Offline crosslinker base class — orchestrates the molrs reaction engine.

:class:`Crosslinker` is a house-style structural transform (mirroring
:class:`~molpy.builder.virtualsite.VirtualSiteBuilder`): the constructor takes a
Daylight reaction SMARTS string plus an optional ``cutoff``; the single verb
:meth:`apply` copies the input graph, matches the reactant patterns via molrs,
loops the subclass :meth:`select` hook to choose bindings, applies each via
``molrs.Reaction.apply``, and returns the new graph — the input is never touched.

**All chemistry is molrs**: SMARTS matching (``SmartsPattern.find_matches_mapped``),
the SMIRKS graph edit (``Reaction.apply``), distances (``NeighborQuery``), and
backbone ordering (``Atomistic.topo_distances``). molpy holds no SMARTS engine,
no match store, no graph-edit primitives — matches are just molrs's
``list[dict[map_number, handle]]``, paired in plain Python.
"""

from __future__ import annotations

import math
from abc import ABC, abstractmethod
from collections.abc import Iterator, Sequence
from dataclasses import dataclass
from typing import TypeVar

import numpy as np

import molrs
from molpy.core.affected_region import AffectedRegion, region_radius
from molpy.core.atomistic import Atomistic

# ``apply`` returns the same subclass it was handed (Atomistic -> Atomistic, and
# any user subclass -> itself), matching the immutable ``.copy()`` contract.
GraphT = TypeVar("GraphT", bound=Atomistic)

# Non-periodic padding (Angstrom) so every point sits strictly inside the
# synthesized bounding box used for the neighbor search.
_BOX_MARGIN = 1.0


@dataclass(frozen=True)
class Candidate:
    """One cross-component occurrence pairing considered for reaction.

    ``occ_a`` / ``occ_b`` are molrs occurrences (``{map_number: handle}``) drawn
    from the two reactant components joined by the primary forming bond;
    ``distance`` is the geometric separation of the two bonding atoms (``0.0``
    when the graph carries no coordinates).
    """

    occ_a: dict[int, int]
    occ_b: dict[int, int]
    comp_a: int
    comp_b: int
    distance: float


class Crosslinker(ABC):
    """Immutable offline crosslinker driven by a molrs reaction.

    Subclasses override the single :meth:`select` hook to decide *which* of the
    ``cutoff``-filtered candidate pairings actually react (exhaustive, spacing,
    explicit, random, ...). Everything else — copy, match, apply — is shared.
    """

    def __init__(self, reaction: str, *, cutoff: float | None = None) -> None:
        self._reaction = molrs.Reaction(reaction)
        self._cutoff = cutoff
        # Affected regions from the most recent ``apply`` — one per formed
        # crosslink, seeded by the atoms ``molrs.Reaction.apply`` reports as
        # touched. Consumed by the incremental-typify layer (spec 02); ``apply``
        # still returns the graph, so this is a backward-compatible side channel.
        self._last_regions: list[AffectedRegion] = []
        forming = self._reaction.forming_bonds
        label_sets = self._component_label_sets()
        if forming:
            self._map_a, self._map_b = forming[0]
            self._comp_a = self._find_component(label_sets, self._map_a)
            self._comp_b = self._find_component(label_sets, self._map_b)
        else:
            # No forming bond -> no crosslink candidates.
            self._map_a = self._map_b = -1
            self._comp_a = self._comp_b = -1

    # -- template method -----------------------------------------------------

    def apply(self, graph: GraphT) -> GraphT:
        """Return a new crosslinked graph (same subclass); ``graph`` untouched.

        Each ``molrs.Reaction.apply`` returns the atom handles it touched; the
        radius-``region_radius`` ball around them is captured as an
        :class:`AffectedRegion` in :attr:`last_regions` for the retype layer.
        """
        work = graph.copy()
        occurrences = self._match_occurrences(work)
        candidates = self._candidate_pairs(work, occurrences)
        radius = region_radius()
        regions: list[AffectedRegion] = []
        for binding in self.select(work, candidates):
            touched = self._reaction.apply(work, binding)
            regions.append(AffectedRegion._from(work, touched, radius))
        self._last_regions = regions
        return work

    @property
    def last_regions(self) -> list[AffectedRegion]:
        """Affected regions built by the most recent :meth:`apply` call."""
        return self._last_regions

    def _match_occurrences(self, graph: Atomistic) -> list[list[dict[int, int]]]:
        """Per-component site occurrences ``{map_number: handle}`` for the reaction.

        Defaults to molrs SMARTS matching (one list per reactant component).
        Override to swap in an alternate site front-end (e.g. reading modelled
        ``port`` markers via :class:`PortMatcher`) — the molrs edit engine that
        consumes the occurrences stays the same.
        """
        return [
            pattern.find_matches_mapped(graph)
            for pattern in self._reaction.reactant_patterns
        ]

    @abstractmethod
    def select(
        self, graph: Atomistic, candidates: list[Candidate]
    ) -> Iterator[dict[int, int]]:
        """Yield ``{map_number: handle}`` bindings to react (subclass mode)."""

    # -- reaction wiring -----------------------------------------------------

    def _component_label_sets(self) -> list[set[int]]:
        """Map-number set of each reactant component (for locating forming bonds)."""
        sets: list[set[int]] = []
        for pattern in self._reaction.reactant_patterns:
            labels = {
                lbl
                for i in range(pattern.num_query_atoms)
                if (lbl := pattern.map_label(i)) is not None
            }
            sets.append(labels)
        return sets

    @staticmethod
    def _find_component(label_sets: list[set[int]], map_number: int) -> int:
        for index, labels in enumerate(label_sets):
            if map_number in labels:
                return index
        return 0

    # -- candidate construction ---------------------------------------------

    def _candidate_pairs(
        self, graph: Atomistic, occurrences: list[list[dict[int, int]]]
    ) -> list[Candidate]:
        """Cross-component occurrence pairs, filtered by ``cutoff`` when set."""
        if self._comp_a < 0 or not self._reaction.forming_bonds:
            return []
        occ_a = occurrences[self._comp_a]
        occ_b = occurrences[self._comp_b]
        if self._cutoff is not None:
            return self._cutoff_candidates(graph, occ_a, occ_b)
        return self._all_candidates(graph, occ_a, occ_b)

    def _all_candidates(
        self,
        graph: Atomistic,
        occ_a: list[dict[int, int]],
        occ_b: list[dict[int, int]],
    ) -> list[Candidate]:
        candidates: list[Candidate] = []
        for oa in occ_a:
            ha = oa[self._map_a]
            for ob in occ_b:
                distance = self._pair_distance(graph, ha, ob[self._map_b])
                candidates.append(
                    Candidate(oa, ob, self._comp_a, self._comp_b, distance)
                )
        return candidates

    def _cutoff_candidates(
        self,
        graph: Atomistic,
        occ_a: list[dict[int, int]],
        occ_b: list[dict[int, int]],
    ) -> list[Candidate]:
        coords_a = [self._xyz(graph, oa[self._map_a]) for oa in occ_a]
        coords_b = [self._xyz(graph, ob[self._map_b]) for ob in occ_b]
        if any(c is None for c in coords_a) or any(c is None for c in coords_b):
            raise ValueError(
                "cutoff-based crosslinking requires atom coordinates, but the "
                "graph has atoms without x/y/z; drop cutoff for topological pairing"
            )
        if not coords_a or not coords_b:
            return []
        points_a = np.asarray(coords_a, dtype=float)
        points_b = np.asarray(coords_b, dtype=float)
        box = self._bounding_box(np.vstack([points_a, points_b]))
        nlist = molrs.NeighborQuery(box, points_b, self._cutoff).query(points_a)
        pairs = nlist.pairs()
        distances = nlist.distances
        candidates: list[Candidate] = []
        for row in range(pairs.shape[0]):
            i = int(pairs[row, 0])
            j = int(pairs[row, 1])
            candidates.append(
                Candidate(
                    occ_a[i],
                    occ_b[j],
                    self._comp_a,
                    self._comp_b,
                    float(distances[row]),
                )
            )
        return candidates

    def _bounding_box(self, points: np.ndarray) -> molrs.Box:
        margin = (self._cutoff or 0.0) + _BOX_MARGIN
        lo = points.min(axis=0) - margin
        hi = points.max(axis=0) + margin
        return molrs.Box.ortho(hi - lo, lo, np.array([False, False, False]))

    # -- geometry helpers ----------------------------------------------------

    @staticmethod
    def _xyz(graph: Atomistic, handle: int) -> tuple[float, float, float] | None:
        x = graph.get(handle, "x")
        y = graph.get(handle, "y")
        z = graph.get(handle, "z")
        if x is None or y is None or z is None:
            return None
        return (float(x), float(y), float(z))

    def _pair_distance(self, graph: Atomistic, ha: int, hb: int) -> float:
        pa = self._xyz(graph, ha)
        pb = self._xyz(graph, hb)
        if pa is None or pb is None:
            return 0.0
        return math.dist(pa, pb)

    # -- binding / occurrence helpers ---------------------------------------

    def _binding(self, candidate: Candidate) -> dict[int, int]:
        """Merge the two occurrences into one ``{map_number: handle}`` binding."""
        return {**candidate.occ_a, **candidate.occ_b}

    @staticmethod
    def _occ_atoms(occurrence: dict[int, int]) -> frozenset[int]:
        return frozenset(occurrence.values())

    def _bond_atoms(self, candidate: Candidate) -> tuple[int, int]:
        return candidate.occ_a[self._map_a], candidate.occ_b[self._map_b]

    def _same_match(self, candidate: Candidate) -> bool:
        return self._occ_atoms(candidate.occ_a) == self._occ_atoms(candidate.occ_b)

    def _same_molecule(self, components: dict[int, int], candidate: Candidate) -> bool:
        ha, hb = self._bond_atoms(candidate)
        return components.get(ha) == components.get(hb)

    def _sort_key(
        self, candidate: Candidate
    ) -> tuple[float, tuple[int, ...], tuple[int, ...]]:
        return (
            candidate.distance,
            tuple(sorted(candidate.occ_a.values())),
            tuple(sorted(candidate.occ_b.values())),
        )

    # -- topology helpers ----------------------------------------------------

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
        """Connected-component root per atom handle (union by BFS over bonds)."""
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

    def _regular_sites(self, graph: Atomistic, spacing: int) -> set[int]:
        """Bonding atoms kept when thinning each molecule's sites every ``spacing``.

        Sites are ordered along the backbone by ``topo_distances`` from a chain
        end; keeping every ``spacing``-th yields uniformly spaced crosslink
        points. Pure topology — coordinates are never read.
        """
        occurrences = [
            pattern.find_matches_mapped(graph)
            for pattern in self._reaction.reactant_patterns
        ]
        sites = {oa[self._map_a] for oa in occurrences[self._comp_a]}
        sites |= {ob[self._map_b] for ob in occurrences[self._comp_b]}
        handles, adjacency = self._adjacency(graph)
        components = self._components(graph)

        by_molecule: dict[int, list[int]] = {}
        for handle in sites:
            by_molecule.setdefault(components[handle], []).append(handle)

        keep: set[int] = set()
        for root, site_handles in by_molecule.items():
            ordered = self._backbone_order(
                graph, adjacency, components, root, site_handles
            )
            keep.update(ordered[::spacing])
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

    def _ordered_sites(
        self, graph: Atomistic, component: int, map_number: int
    ) -> list[dict[int, int]]:
        """Deduplicated representative occurrences per site, ordered by handle."""
        occurrences = self._reaction.reactant_patterns[component].find_matches_mapped(
            graph
        )
        representative: dict[int, dict[int, int]] = {}
        for occurrence in occurrences:
            handle = occurrence[map_number]
            if handle not in representative:
                representative[handle] = occurrence
        return [representative[handle] for handle in sorted(representative)]
