---
title: Incremental-typify 1/3 — AffectedRegion (hashable MolGraph subgraph)
status: done
created: 2026-07-05
depends_on: "molrs: region-support-01-graph-hash, region-support-02-reaction-touched"
---

# AffectedRegion — a hashable MolGraph subgraph produced by graph edits

> A first-class **`AffectedRegion`** (subclass of `Atomistic`/`CoarseGrain`, so it IS a MolGraph):
> the radius-N ball around the atoms a graph edit touched, extracted at a **retype-safe radius**,
> carrying `interior`/`boundary`/`entity_map` and a **structural `__hash__`/`__eq__`** (via the
> molrs graph hash). Producers (`Reacter`, `Crosslinker`) return it. It replaces the flat
> `ReactionResult.modified_atoms`, is passable to AmberTools (it's an `Atomistic`), and dedupes
> identical polymer junctions (`incremental-typify-02` caches by its hash).
> Architecture: `.claude/notes/incremental-typification-design.md`.

## Summary

`AffectedRegion` is built by `graph.extract_subgraph(touched, radius)` (already exists,
`core/atomistic.py:536`) where `touched` = the seed atoms a graph edit reports and `radius` is
**auto-derived** from the typifier's `context_radius` (floor 4). It adds:

- `interior` — the changed atoms (to be retyped); `boundary` — the context-only shell (never
  retyped); `entity_map` — region atom → parent atom (map types back);
- `__hash__` = `self.structural_hash()` (molrs, `region-support-01`); `__eq__` =
  `self.is_isomorphic(other)` — so two identical junctions compare equal → cache hit.

Because it IS-A `Atomistic`/`CoarseGrain`, `AmberTools.parameterize(region)` / the
`Atomistic→PDB` bridge consume it unchanged.

## Domain basis

- **`extract_subgraph(center_entities, radius) -> (Atomistic, boundary_atoms)`**
  (`core/atomistic.py:536`) already: BFS radius-ball via molrs `topo_distances`, induced-clone of
  bonds/angles/dihedrals/impropers, returns boundary (edge) atoms. Proven in BondReact
  (`bond_react.py:395`, radius 4).
- **molrs `structural_hash()`/`canonical_order()`/`is_isomorphic()`** (`region-support-01`) — the
  isomorphism-invariant dedup key on the region.
- **`Reaction.apply` returns touched handles** (`region-support-02`); molpy `Reacter._execute_reaction`
  (`base.py:447`) knows anchors/leaving/new-bonds. Both feed `touched`.
- **Retype-safe radius**: atom typing needs full ring/degree context (`typifier/adapter.py` ring
  perception is whole-graph), so a bare changed-atom set types the boundary wrong. Radius must
  cover the typifier's SMARTS depth → the region includes a complete context shell.

## Design

### 1. `AffectedRegion`

```python
class AffectedRegion(Atomistic):          # + a CoarseGrain variant (shared mixin _RegionView)
    """Radius-N ball around edited atoms; a MolGraph you can hash, type, or hand to AmberTools."""
    @classmethod
    def _from(cls, parent, touched, radius): ...   # extract_subgraph + record interior/boundary/map
    interior: tuple[Atom, ...]
    boundary: tuple[Atom, ...]
    entity_map: dict[Atom, Atom]          # region atom → parent atom
    def __hash__(self) -> int: return self.structural_hash()          # molrs
    def __eq__(self, other) -> bool: return isinstance(other, AffectedRegion) and self.is_isomorphic(other)
```

- `interior` = the seed `touched` atoms (that changed); `boundary` = `extract_subgraph`'s returned
  edge atoms; middle-shell atoms are context but valid. `entity_map` inverts the clone mapping.
- Note: `AffectedRegion` overrides hashing at the **region** level only; its member `Entity`/`Link`
  keep identity hashing (unchanged core contract).

### 2. Radius policy

`region_radius(typifier) = max(getattr(typifier, "context_radius", 0), _FLOOR)` with `_FLOOR = 4`.
Typifiers gain an optional `context_radius` property (max path length over their SMARTS patterns);
absent → floor. Overridable per call.

### 3. Producers build it

- `Crosslinker` (`builder/crosslink/_crosslinker.py`): `molrs.Reaction.apply(work, binding)` now
  returns touched handles → after the loop (or per edit) build `AffectedRegion._from(work, touched,
  radius)`. `apply` gains an optional return/attr exposing the regions (kept backward-compatible —
  default behavior unchanged, regions available for the typify layer in 02).
- `Reacter` (`reacter/base.py`): `_execute_reaction` returns `(new_bond, removed_atoms)`; wrap the
  touched atoms (anchors + new-bond endpoints + leaving-group neighbors) into an `AffectedRegion`
  and attach to `ReactionResult` (new field `region: AffectedRegion | None`), superseding
  `modified_atoms`.

This spec only **produces** the region; region-scoped typing + cache is `incremental-typify-02`.

## Files to create or modify

- `src/molpy/core/region.py` (new) — `AffectedRegion` (+ CG variant / shared mixin)
- `src/molpy/core/__init__.py` — export `AffectedRegion`
- `src/molpy/reacter/base.py` — `ReactionResult.region`; build region in `_execute_reaction`/`run`
- `src/molpy/builder/crosslink/_crosslinker.py` — consume `Reaction.apply` touched handles → regions
- `tests/test_core/test_region.py` — extraction, interior/boundary/map, structural hash/eq, AmberTools-shape

## Tasks

- [x] **T1**: `AffectedRegion` (`core/affected_region.py`, renamed to avoid the existing spatial-mask `core/region.py`) — `_from(parent, touched, radius)` via `Atomistic._extract_mapped` (refactor of `extract_subgraph`); `interior`/`boundary`/`entity_map`; `__hash__`/`__eq__` via molrs `structural_hash`/`is_isomorphic`
- [x] **T2**: `region_radius(typifier)` helper + optional `context_radius` on `ForceFieldTypifier` (floor 4)
- [x] **T3**: molrs already provides `structural_hash`/`is_isomorphic`/touched-return (workspace venv); `Crosslinker` builds `last_regions` from `Reaction.apply` touched handles
- [x] **T4**: `ReactionResult.region`; `Reacter.run` builds it from anchors + new-bond endpoints (supersedes `modified_atoms`, which is kept for back-compat)
- [x] **T5**: tests (`tests/test_core/test_affected_region.py`) + quality gate — ruff/ty/pytest全绿

## Testing strategy

- **extraction** — a known edit → region contains the changed atoms (`interior`) + a radius-N shell;
  `boundary` atoms have a neighbor outside; `entity_map` round-trips region→parent.
- **structural hash/eq** — two identical junctions (same radius-N env) → equal hash, `==`; a
  different junction → `!=`.
- **AmberTools shape** — `isinstance(region, Atomistic)`; `AmberTools`'s `Atomistic→PDB` path accepts it.
- **producers** — crosslink one bond → an `AffectedRegion` around it; reacter run → `result.region` set.
- **immutable/core** — member `Entity` still identity-hashed; only the region object is structurally hashed.

## Out of scope

- **`typify_region` + RetypeCache** — `incremental-typify-02`
- **builder junction dedup / AmberTools caching / crosslink retype wiring** — `incremental-typify-03`
- **molrs graph hash + touched return** — molrs `region-support-01/02`
