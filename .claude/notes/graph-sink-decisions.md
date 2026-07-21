# Graph sink decisions (Atomistic / CoarseGrain → molrs)

**Date:** 2026-07-11
**Status:** locked
**Scope:** finish sinking molecular-graph engine to molrs; molpy keeps ergonomic constructors only.

Companion to the migration plan discussed in session. Aligns with architecture iron laws 2 (system policy stays in molpy), 3 (experimental: no compat shims), 6 (users never import molpy as mp).

## Locked decisions

### A — `copy()` = molrs handle-preserving clone

- Canonical: `MolGraph` / leaf `Clone` (and Python `molrs.Atomistic.copy`).
- **Handles are preserved** in the clone (same generational keys).
- molpy must **not** re-implement copy via `def_atom` + re-spawn (that invents new handles).
- After copy, molpy intern tables start empty; views are re-interned lazily against the new world.
- Callers that held views into the *old* world keep pointing at the old world — never silently rebound.

### B — `merge()` = molrs structural merge (handle remap)

- Canonical: `MolGraph::merge` — nodes/relations transferred; **all handles remapped**.
- molpy must **not** keep identity-preserving merge (`_detach` + re-spawn same Python objects) as the public `merge`.
- Cross-graph identity is **handle-based**, not Python object identity. Code that needs “the same atom after merge” must use a returned `old_handle → new_handle` map (add to molrs API if missing) or re-query topology.
- No `merge_identity` shim (iron law 3). If a caller breaks, fix the caller to track handles.

### C — API split: molrs primitives / molpy `def_*`

| Layer | Owns | Does not own |
|---|---|---|
| **molrs** | Storage, ECS get/set/spawn/despawn/add_relation, domain leaves, graph algorithms (`extract_*`, topology, spatial, `copy`/`merge`, chemistry kernels) | Python pending-Entity lifecycle, `def_atom` sugar |
| **molpy** | Thin subclass + `def_atom` / `def_bond` / …, `Atom`/`Bond`/… views, interning for `bond.itom is atom` | Re-implementing graph algorithms in Python |

- molrs public surface: handle + `get`/`set`/`has`/`delete` (+ relation equivalents), builders like `add_atom_bare` / `add_bond`.
- molpy public surface: `mol.def_atom(element="C", xyz=[…])` etc., built **on top of** those primitives.
- End-state thickness target: `atomistic.py` / `cg.py` each ≲ 200 LOC of sugar + adopt/from_frame wrappers.

## Non-goals (unchanged)

- Do not sink ForceField, AffectedRegion, assembler policy, typifier strategy.
- Do not require pure re-export (`molpy.Atomistic is molrs.Atomistic`) in the first cut — subclass + `def_*` is intentional.
- Do not implement pyo3-native Entity views until path B is thin and stable.

## Implementation order (binding)

Specs (both repos, `status: approved`):

| Order | Repo | Slug |
|---|---|---|
| 1 | molrs | `graph-sink-01-extract` |
| 2 | molrs | `graph-sink-02-copy-merge` |
| 3 | molrs | `graph-sink-03-python-bind` (depends 01+02) |
| 4 | molrs | `graph-sink-04-hydrogens-coords` |
| 5 | molpy | `graph-sink-01-wire-algorithms` (blocked on molrs 01–03 release) |
| 6 | molpy | `graph-sink-02-def-sugar` |
| 7 | molpy | `graph-sink-03-valence` (blocked on molrs 04) |

Start with: `/mol:impl graph-sink-01-extract` in the molrs repo.
