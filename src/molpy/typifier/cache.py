"""Hash-keyed retype cache: type each distinct affected-region once.

:class:`RetypeCache` keys :class:`~molpy.typifier.region.RegionTypes` by the
region's isomorphism-invariant structural hash
(:meth:`~molpy.typifier.affected_region.AffectedRegion.__hash__`). During polymer
growth the many structurally-identical junctions collapse to one hash → the
region is typed **once** and every later identical junction is a cache hit,
turning the reacter's O(N²) whole-graph retype pass into O(#distinct junction
environments).

A hash bucket holds ``(region, types)`` pairs so a rare hash collision is
resolved by ``region == cached_region`` — which is the molrs graph-equality
(``is_isomorphic``) check, not identity. On a hit the cached snapshot came from a
*different* but isomorphic region; :meth:`apply` lines its canonical positions up
against the new region's own :meth:`canonical_order` and reaches the parent atoms
through the region's ``entity_map``.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from molpy.typifier.affected_region import AffectedRegion
    from molpy.typifier.region import RegionTypes, RegionTypifier


class RetypeCache:
    """Deduplicate region typing by structural hash + isomorphism confirm."""

    def __init__(self, typifier: RegionTypifier) -> None:
        self._typifier = typifier
        self._buckets: dict[int, list[tuple[AffectedRegion, RegionTypes]]] = {}

    def retype(self, region: AffectedRegion) -> RegionTypes:
        """Return ``region``'s types, typing it only on a genuine cache miss."""
        key = hash(region)
        for cached_region, types in self._buckets.get(key, ()):
            if region == cached_region:  # is_isomorphic confirm
                return types
        types = self._typifier.typify_region(region)
        self._buckets.setdefault(key, []).append((region, types))
        return types

    def retype_and_apply(self, region: AffectedRegion) -> RegionTypes:
        """Type ``region`` (cached) and write its interior types onto the parent.

        The write-back maps each stored canonical position onto ``region``'s own
        canonical order + ``entity_map``, so a snapshot captured from a different
        but isomorphic region still lands correctly. Atoms outside the write-back
        set are never touched.
        """
        types = self.retype(region)
        types.apply_to(region)
        return types
