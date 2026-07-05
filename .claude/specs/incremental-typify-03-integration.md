---
title: Incremental-typify 3/3 — wire builder / crosslink / AmberTools onto the region cache
status: draft
created: 2026-07-05
depends_on: incremental-typify-02-cache
---

# Wire polymer builder, crosslink, and AmberTools onto the region cache

> Consume the `AffectedRegion` + `RetypeCache` (01/02) at the three call sites that motivated it:
> the polymer builder (junction dedup — the O(N²) fix lands here), the crosslinker (which today
> does not retype at all), and the AmberTools wrapper (type a region subgraph, cached by hash).
> Architecture: `.claude/notes/incremental-typification-design.md`.

## Summary

Three integrations, no new engine:

1. **Polymer builder** — `PolymerBuilder`/`Connector` already forward a typifier into `Reacter.run`
   (`core.py:374`, `connectors.py:161`); with 02 rewiring `_incremental_typify`, per-bond retype
   already caches. This spec adds a **shared `RetypeCache` across the whole build** (one cache per
   `PolymerBuilder.build`) so identical junctions dedupe across all connections → chain retype cost
   O(#distinct junctions).
2. **Crosslink** — `Crosslinker.apply` (`builder/crosslink/_crosslinker.py`) currently returns an
   untyped graph. Add an optional `typifier=` that, per applied reaction, retypes the
   `AffectedRegion` (from 01) via a shared `RetypeCache`. Off by default (crosslink stays pure-topology
   unless a typifier is supplied).
3. **AmberTools** — `AmberTools`/`AntechamberWrapper` type a whole molecule via `Atomistic→PDB`.
   Add a **region path**: type an `AffectedRegion` subgraph (it IS an `Atomistic`) through antechamber,
   cached by the region hash, mapping GAFF types back via `entity_map`. The wrapper's `Atomistic→PDB`
   bridge is unchanged; only a region-aware caller + cache is added.

## Domain basis

- **Builder O(N²)**: `_connect_monomers`→`Connector.connect`→`Reacter.run` per edge (`core.py:374`);
  02 makes each retype region-scoped+cached, but the cache must be **shared across the build** to
  dedupe junctions that recur along the chain. Today no cache is threaded through the build loop.
- **Crosslink no-retype**: `Crosslinker.apply` (`_crosslinker.py:80`) does the molrs edit and returns
  the graph untyped (verified: zero typify in the package). A typifier hook + region cache adds
  optional local retyping.
- **AmberTools contract**: `AntechamberWrapper.atomtype_assign` + `write_antechamber_input_pdb`
  (`wrapper/antechamber.py:49,105`) already take an `Atomistic`; a region is an `Atomistic`.
  `AmberTools.parameterize(struct)` (`builder/ambertools.py:69`) is the whole-molecule path.

## Design

### 1. Shared `RetypeCache` through the build

`PolymerBuilder.build` creates one `RetypeCache`; threads it into `Connector.connect`→`Reacter.run`
(new optional `retype_cache=` param, default a fresh cache). Junctions recurring along the chain hit
the shared cache. No change to placement/sequence logic.

### 2. Crosslink retype hook

`Crosslinker.__init__(..., typifier=None)`; when set, `apply` builds each `AffectedRegion` (01) and
`RetypeCache.retype`s it, writing interior types back to `work`. Default `None` → unchanged
pure-topology behavior.

### 3. AmberTools region path

`AmberTools.parameterize_region(region: AffectedRegion, *, net_charge, name)` (or a `region=` branch):
write the region `Atomistic`→PDB, run antechamber, map GAFF atom types back onto the parent via
`entity_map`; cache by `hash(region)` so recurring junctions skip the subprocess. Whole-molecule path
untouched.

## Files to create or modify

- `src/molpy/builder/polymer/core.py` + `connectors.py` — thread a shared `RetypeCache` through the build
- `src/molpy/builder/crosslink/_crosslinker.py` — optional `typifier=` + region retype via cache
- `src/molpy/builder/crosslink/__init__.py` — (no new public type; param only)
- `src/molpy/builder/ambertools.py` + `src/molpy/wrapper/antechamber.py` — region path + hash cache
- `tests/test_builder/test_polymer/` — junction dedup across a chain (typifier call count bounded)
- `tests/test_builder/test_crosslink/` — crosslink with typifier retypes interior
- `tests/test_wrapper/` (or `@pytest.mark.external`) — AmberTools region typing + cache

## Tasks

- [ ] **T1**: thread a shared `RetypeCache` through `PolymerBuilder.build` → `Connector.connect` → `Reacter.run`
- [ ] **T2**: `Crosslinker(typifier=None)` — region retype via shared cache when set; default unchanged
- [ ] **T3**: `AmberTools` region path + hash-keyed cache (map GAFF types back via entity_map); external-marked test
- [ ] **T4**: tests — chain junction dedup (bounded typifier calls), crosslink retype, AmberTools region cache
- [ ] **T5**: quality gate — ruff/ty/pytest 全绿；builder/crosslink 现有测试无回归

## Testing strategy

- **builder dedup** — grow an N-monomer regular chain; the FF typifier's underlying invocation count is
  bounded (≈distinct junctions), not O(N); product types match a whole-chain typify baseline.
- **crosslink retype** — `Crosslinker(rxn, typifier=ff).apply(g)` → interior atoms of each crosslink carry
  correct types; without `typifier` the graph is untyped (unchanged).
- **AmberTools region** — (external) type a junction region via antechamber; a recurring identical region is a
  cache hit (no second subprocess); GAFF types land on the parent via `entity_map`.
- No regression: `tests/test_builder/ -m "not external"` green; `PolymerBuilder` product unchanged besides
  faster/cached typing.

## Out of scope

- **AffectedRegion / typify_region / RetypeCache internals** — 01 / 02
- **molrs graph hash + touched return** — molrs `region-support-01/02`
- **Rewriting the builder's bonding onto molrs Reaction** — separate `crosslink-03` follow-up
