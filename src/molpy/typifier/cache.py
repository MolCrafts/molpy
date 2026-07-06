"""Hash-keyed retype cache: type each distinct affected-region once.

:class:`RetypeCache` keys :class:`~molpy.typifier.region.RegionTypes` by the
region's isomorphism-invariant structural hash
(:meth:`~molpy.core.affected_region.AffectedRegion.__hash__`). During polymer
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

from molpy.typifier.region import apply_region_types

if TYPE_CHECKING:
    from molpy.core.affected_region import AffectedRegion
    from molpy.typifier.atomistic import ForceFieldTypifier
    from molpy.typifier.region import RegionTypes


class RetypeCache:
    """Deduplicate region typing by structural hash + isomorphism confirm."""

    def __init__(self, typifier: ForceFieldTypifier) -> None:
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

    def apply(self, region_types: RegionTypes, region: AffectedRegion) -> None:
        """Write ``region_types`` onto ``region``'s parent interior atoms.

        Delegates to :func:`~molpy.typifier.region.apply_region_types`, which
        maps each stored canonical position onto ``region``'s atoms via its own
        canonical order + ``entity_map`` (boundary atoms are never touched).
        """
        apply_region_types(region_types, region)

    def retype_and_apply(self, region: AffectedRegion) -> RegionTypes:
        """Convenience: :meth:`retype` then :meth:`apply` onto the parent."""
        types = self.retype(region)
        self.apply(types, region)
        return types
