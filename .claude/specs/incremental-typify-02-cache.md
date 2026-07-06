---
title: Incremental-typify 2/3 — region-scoped typing + hash-keyed RetypeCache
status: done
created: 2026-07-05
depends_on: incremental-typify-01-region
---

# Region-scoped typing + RetypeCache (dedup identical junctions)

> Type an `AffectedRegion` as a standalone graph (`typify_region`), and cache the result by the
> region's **structural hash** so identical polymer junctions retype **once**. Rewire the reacter's
> `_incremental_typify` to use region-scoped typing + the cache — killing the O(N²) whole-graph
> atom-typing pass. Architecture: `.claude/notes/incremental-typification-design.md`.

## Summary

- **`typifier.typify_region(region: AffectedRegion) -> RegionTypes`** — type the region as a
  standalone `Atomistic` (the boundary shell gives complete ring/degree/SMARTS context), return type
  info for the **interior** atoms + their incident bonded terms, keyed by the region's
  **canonical order** (molrs, `region-support-01`) so it maps onto any isomorphic region.
- **`RetypeCache`** — `dict[region_hash -> RegionTypes]`. `retype(region)`: on hash hit (confirmed by
  `is_isomorphic`), reuse; else `typify_region` + store. Types written back to the parent via
  `entity_map` + canonical order.
- **Rewire `_incremental_typify`** (`reacter/base.py:546`): replace the whole-structure atom-typing
  pass (`:582-586`) with `RetypeCache.retype(result.region)` mapped back onto the product. Bonded-term
  logic already local — fold into the region result.

## Domain basis

- **The O(N²) pain**: `_incremental_typify` re-runs `atom_typifier.typify(assembly)` on the whole
  structure every bond (`base.py:582-586`). Region-scoped + cached typing makes the retype pass
  O(#distinct junction environments), i.e. O(1) amortized during regular polymer growth.
- **Correctness**: typing needs full context — the region already carries the retype-safe shell
  (`incremental-typify-01`). `canonical_order` lets a cached region's types line up with a new
  isomorphic region deterministically.
- **Typifier surface**: `ForceFieldTypifier.typify(struct)->struct` (`typifier/atomistic.py:580`) types
  a whole graph; the per-element typifiers (`PairTypifier`/`ForceFieldBondTypifier`/… `atomistic.py:157,
  216,274,338`) already type single elements (what `_incremental_typify` uses). `typify_region` = type
  the region graph, then read off interior types.

## Design

### 1. `typify_region`

```python
def typify_region(self, region: AffectedRegion) -> RegionTypes:
    typed = self.typify(region)                    # region IS a standalone Atomistic (shell = context)
    order = region.canonical_order()               # molrs
    return RegionTypes(interior atom types + incident bond/angle/dihedral types, keyed by canonical order)
```
`RegionTypes` = frozen dataclass of type strings/params in canonical order (hashable/cacheable, no
live Entity refs).

### 2. `RetypeCache`

```python
class RetypeCache:
    def retype(self, region: AffectedRegion) -> RegionTypes:
        h = hash(region)
        for cached_region, types in self._by_hash.get(h, ()):     # collision bucket
            if region == cached_region: return types              # is_isomorphic confirm
        types = region._typifier.typify_region(region)
        self._by_hash.setdefault(h, []).append((region, types)); return types
```
Write-back: map `RegionTypes` (canonical order) onto the parent's interior atoms via
`region.entity_map` + `region.canonical_order()`.

### 3. Rewire `_incremental_typify`

Replace the whole-graph atom-typing block with `RetypeCache.retype(result.region)` applied to the
product's interior atoms; keep the existing local bonded-term generation but source its types from the
region result. Fall back to the old path if `result.region is None` (no molrs support / disabled).

## Files to create or modify

- `src/molpy/typifier/region.py` (new) — `typify_region` mixin/method + `RegionTypes`
- `src/molpy/typifier/atomistic.py` — add `typify_region` to `ForceFieldTypifier` (+ `context_radius`)
- `src/molpy/typifier/cache.py` (new) — `RetypeCache`
- `src/molpy/reacter/base.py` — `_incremental_typify` uses region + cache (fallback to old path)
- `tests/test_typifier/test_region_cache.py` — region typing correctness + cache hit + O(N) growth

## Tasks

- [ ] **T1**: `RegionTypes` + `typify_region` — type region as standalone Atomistic, read interior types in canonical order
- [ ] **T2**: `RetypeCache` — hash bucket + `is_isomorphic` confirm + write-back via entity_map/canonical order
- [ ] **T3**: rewire `_incremental_typify` to region+cache; fallback when `region is None`
- [ ] **T4**: `context_radius` on OPLS/FF typifier (max SMARTS pattern depth)
- [ ] **T5**: tests + quality gate — ruff/ty/pytest 全绿；`tests/test_reacter/` 无回归

## Testing strategy

- **region typing == whole typing** — for a small structure, region-scoped interior types match the
  types the whole-graph `typify` assigns to those atoms (boundary shell makes it exact).
- **cache hit** — retyping two identical junctions calls the underlying typifier **once** (spy/counter);
  second is a cache hit.
- **growth cost** — building an N-junction regular chain, the typifier invocation count is O(#distinct
  junctions) not O(N) — assert the counter is bounded as N grows.
- **correctness vs baseline** — a fully typed product via the region+cache path equals the product via
  the old whole-graph `_incremental_typify` (same types).
- No regression: `tests/test_reacter/` green (fallback path intact when region absent).

## Out of scope

- **builder junction dedup / crosslink retype hook / AmberTools caching** — `incremental-typify-03`
- **AffectedRegion / molrs hash** — `incremental-typify-01` / molrs
