# Incremental Typification — Affected-Region Design (v1)

Status: draft | Author: Claude Code | Date: 2026-07-05

> Every graph edit (bond formation, elimination, crosslink) returns a **hashable
> `MolGraph` affected-region**. Retypification is **region-scoped and cached by structural
> hash** — so during polymer growth the many identical junctions retype **once** and reuse.
> The region *is* an `Atomistic`/`CoarseGrain` subgraph, so it hands straight to third-party
> typifiers (AmberTools). One structure, four consumers: reacter, crosslink, polymer builder,
> AmberTools.
>
> **Repo split (per the standing rule — shared components in molrs):** the **structural graph
> hash** lives in molrs (native, AA+CG, next to the BFS/SMARTS/topology kernels and the
> O(N²)-hot matcher). Everything else (region extraction, region typing, cache, integration)
> is molpy.

---

## 0. Principles

```
图编辑 → 返回受影响区域       任何成键/消除/交联都返回一个 AffectedRegion（MolGraph 子图）
区域是 MolGraph              AffectedRegion IS-A Atomistic/CoarseGrain → 可直接喂 AmberTools
结构可哈希                    同构不变的结构哈希（WL/Morgan）→ 去重键
retype = 区域内 + 缓存        只对区域内重新分型，按结构哈希缓存；相同 junction 只分型一次
哈希在 molrs                  结构哈希是 native（AA+CG 共用，紧邻 BFS/SMARTS/topology 内核）
半径自适应                    抽取半径 ≥ typifier 的 SMARTS 深度（保证边界原子环境完整）
```

---

## 1. The problem (verified current state)

- **Retype is scattered and mostly whole-graph.** `_incremental_typify` (`reacter/base.py:546`)
  types bonded terms locally but **re-runs SMARTS atom-typing over the *entire* structure**
  every bond (`:582-586`, comment "graph matching needs full structure"). In polymer growth
  that is **O(N²)** SMARTS matching — the builder avoids O(N²) elsewhere (`core.py` group
  bookkeeping / port registry), then the retype path defeats it.
- **No shared affected-region object.** `ReactionResult` (`reacter/base.py:22`) is a flat
  dataclass; `modified_atoms` is only the two anchors. Not a graph, not hashable, not reused.
- **Crosslink doesn't retype at all**; `molrs.Reaction.apply` returns `()`.
- **`extract_subgraph(centers, radius) → (subgraph, boundary_atoms)`** already exists
  (`core/atomistic.py:536`, proven in BondReact template gen) but is **not** wired into typing.
- **No structural graph hash exists in either repo** — the one missing primitive for dedup.
- **AmberTools already consumes `Atomistic → PDB → antechamber`** (`wrapper/antechamber.py:49,105`)
  — so a region that *is* an `Atomistic` needs no wrapper change.

---

## 2. Architecture

```
  molrs (native)                                   molpy
  ──────────────                                   ─────
  MolGraph structural hash  ──graph_hash()──────▶  AffectedRegion(Atomistic|CoarseGrain)
    (WL/Morgan + canonical order + iso check)        = extract_subgraph(touched, radius)
  Reaction.apply → touched  ──touched handles────▶     + boundary + entity_map
    atom handles                                       + __hash__/__eq__ (molrs hash)
                                                              │
                                                              ▼
                                            typifier.typify_region(region)  ── region-scoped typing
                                            RetypeCache[hash] → types        ── dedup identical junctions
                                                              │
                     producers ─────────────────────────────┤
                       Reacter.run / Crosslinker.apply       │  consumers
                       (return AffectedRegion)               ▼
                                            _incremental_typify · builder · AmberTools
```

Only new types: `AffectedRegion` (molpy subgraph subclass), `RetypeCache` (molpy), and molrs's
`graph_hash`/iso primitive. No new chemical concepts.

---

## 3. `AffectedRegion` — a hashable MolGraph subgraph

A **subclass** of the graph it came from (`Atomistic` or `CoarseGrain`), so it *is* a MolGraph
(passable to AmberTools) but adds region semantics:

```python
class AffectedRegion(Atomistic):          # (and a CoarseGrain variant / a shared mixin)
    interior: tuple[Atom, ...]            # atoms that actually changed → to be retyped
    boundary: tuple[Atom, ...]            # context-only shell (NOT retyped)
    entity_map: dict[Atom, Atom]          # region atom → parent atom (map types back)
    def __hash__(self) -> int: ...        # molrs structural hash (isomorphism-invariant)
    def __eq__(self, other) -> bool: ...  # molrs graph-equality (resolves hash collisions)
```

- Built by `extract_subgraph(touched_atoms, radius)` — the radius-R ball around the changed
  atoms; `extract_subgraph` already returns the boundary/edge atoms.
- **Overrides `Entity`'s identity hashing** at the *region* level: two structurally-identical
  junctions (same radius-R environment + labels) hash equal → cache hit.
- Because it IS-A `Atomistic`, `AmberTools.parameterize(region)` / the `Atomistic→PDB` bridge
  work unchanged.

---

## 4. molrs structural hash (the dedup key)

A **Weisfeiler–Lehman / Morgan-style** isomorphism-invariant hash on `MolGraph`, native:

- Node labels: element (or bead type) + degree + formal charge + aromatic flag; edge labels:
  bond order. Iterate WL to convergence (or a fixed R ≥ region radius) → a 64-bit hash.
- Also produce a **canonical node ordering** (the WL refinement's stable order) — needed to
  map cached interior types onto a new region's atoms deterministically.
- A companion **`is_isomorphic(a, b)`** (reuse the existing backtracking/VF2 matcher machinery)
  to resolve the rare hash collision before trusting a cache hit.
- Lives on molrs `MolGraph` (serves AA+CG; sits next to `topo_distances`/`generate_topology`/
  the SMARTS matcher), exposed to Python (`Atomistic.structural_hash() -> int`,
  `Atomistic.canonical_order() -> list[handle]`, `graph_is_isomorphic(a, b) -> bool`).

Why molrs: it's a shared component (AA+CG), and the cost it eliminates (SMARTS atom-typing) is
Rust-side, so keeping the dedup key native avoids a Python round-trip per junction.

---

## 5. Region-scoped typing + retype cache

- **`typifier.typify_region(region: AffectedRegion) -> dict[Atom, type-info]`** — types the
  region as a **standalone `Atomistic`** (the boundary shell gives complete ring/degree/SMARTS
  context), returns type info for the **interior** atoms + incident bonded terms, mapped back
  via `entity_map`. This is correct precisely because the region includes the retype-safe shell
  (§6); typing a bare changed-atom set would give wrong boundary types.
- **`RetypeCache`** keyed by `AffectedRegion.__hash__`: `hash → (interior atom types, bonded-term
  types)` in canonical order. Consulted before invoking any typifier (native OR AmberTools).
  During polymer growth, identical junctions → same hash → **type once, reuse** → the retype pass
  drops from **O(N²) → O(N)** (really O(#distinct junction environments)).
- Cache hit application uses the molrs **canonical order** to line up cached types with the new
  region's interior atoms.

---

## 6. Retype-safe radius (auto-derived)

Atom typing needs full neighbor/ring context, so a truncated subgraph gives **wrong boundary
types**. The extraction radius must cover the typifier's SMARTS depth:

- The typifier exposes a **`context_radius`** (max path length over its SMARTS patterns; OPLS/MMFF
  are a few bonds). `AffectedRegion` radius = `max(typifier.context_radius, FLOOR=4)` (BondReact's
  proven default). Overridable.
- Boundary atoms (returned by `extract_subgraph`) are **context only** — never retyped, and their
  types are discarded on write-back.

---

## 7. Producers & consumers

- **Producers (return an `AffectedRegion`):**
  - `molrs.Reaction.apply` → returns the **touched atom handles** (the atoms it formed/broke/added
    bonds on / deleted-neighbors of). Small molrs change (currently returns `()`).
  - molpy `Reacter.run` / `_execute_reaction` and `Crosslinker.apply` → build the `AffectedRegion`
    from those touched atoms via `extract_subgraph`, replacing `ReactionResult.modified_atoms`.
- **Consumers (retype from a region):**
  - `_incremental_typify` (`reacter/base.py:546`) → region-scoped typing + cache lookup (kills the
    whole-graph atom-typing pass).
  - `PolymerBuilder._connect_monomers` / `Connector.connect` → dedup identical junction regions
    across the chain (the O(N²) fix lands here).
  - `Crosslinker.apply` → gains a retype hook (today it doesn't retype).
  - `AmberTools`/`AntechamberWrapper` → consume the region `Atomistic` via the existing
    `Atomistic→PDB` bridge, cached by region hash.

---

## 8. Spec chain (2 molrs + 3 molpy)

| Spec | Repo | Scope |
|------|------|-------|
| `region-support-01-graph-hash` | molrs | WL/Morgan structural hash + canonical order + `is_isomorphic` on `MolGraph`, exposed to Python (AA+CG) |
| `region-support-02-reaction-touched` | molrs | `Reaction.apply` returns touched atom handles |
| `incremental-typify-01-region` | molpy | `AffectedRegion` subclass (extract at auto radius, boundary, entity_map, structural `__hash__`/`__eq__`); producers build it |
| `incremental-typify-02-cache` | molpy | `typifier.typify_region` + `RetypeCache` (hash-keyed); rewire `_incremental_typify` |
| `incremental-typify-03-integration` | molpy | wire into polymer builder (junction dedup), crosslink (retype hook), AmberTools (region → antechamber, cached) |

---

## 9. What we deliberately do NOT do

- **Put the structural hash in molpy.** It's a shared MolGraph primitive → molrs (native, AA+CG).
- **Retype the boundary shell.** Boundary atoms are context only; their types are discarded.
- **Type a bare changed-atom set.** Always extract the retype-safe shell first (else wrong types).
- **Change the AmberTools wrapper.** The region is an `Atomistic`; the existing `Atomistic→PDB`
  bridge consumes it unchanged.
- **Break `Entity` identity hashing.** Only `AffectedRegion` (the region object) is structurally
  hashed; individual `Entity`/`Link` keep identity hashing.
