---
title: "graph-sink 2/3 (molpy) — def_* sugar only; thin Atomistic/CoarseGrain leaves"
status: done
created: 2026-07-11
chain: graph-sink
depends_on: "graph-sink-01-wire-algorithms"
---

# API split C: molrs primitives, molpy def_* ergonomics

> Chain **graph-sink** molpy 2/3.
> Decision **C**: molrs owns get/set/spawn; molpy owns `def_atom` etc.
> After 01, algorithms are gone — this spec enforces the **thickness** and **primitive-only**
> construction path.

## Summary

1. Every `def_*` / `add_*` path creates storage **only** via molrs primitives
   (`spawn` / `add_relation` / `set` / leaf `add_atom`/`add_bond` if used).
2. Target size: `atomistic.py` ≲ **250 LOC**, `cg.py` ≲ **200 LOC** (sugar + view properties +
   adopt/from_frame/to_frame wrappers). Excess helpers move to small modules or delete.
3. `Entity` / `Link` / `_GraphViews` stay in `entity.py` as the view layer — not reimplemented
   in Rust.
4. Document the split in `docs/developer/molrs-backend.md` (or equivalent developer note).

## Domain basis

### Allowed molrs primitives (non-exhaustive)

- Node: `spawn`, `despawn`, `get`, `set`, `has`, `delete`, `entities`, `node_keys`, `column`
- Relation: `register_kind`, `add_relation`, `remove_relation`, `relation_nodes`,
  `set_relation_prop`, `get_relation_prop`, `relation_ids`, `incident_relations`
- Leaf: `add_atom*` / `add_bond` / `generate_topology` / `to_frame` / `from_frame` / `adopt` /
  `copy` / `merge` / `extract_subgraph` / spatial free functions

### def_* contract

```python
def def_atom(self, mapping=None, /, **attrs) -> Atom:
    atom = Atom(mapping, **attrs)          # pending Entity (plain dict)
    return self._spawn_entity(atom)        # spawn + _attach → set components
```

`_spawn_entity` must use `self.spawn()` + component sets (already does). Do **not** call a
parallel pure-Python store.

### What stays on the leaf (minimal)

- Domain view types: `Atom`, `Bond`, `Angle`, `Dihedral`, `Improper`, `VirtualSite*`, `Bead`, `CGBond`
- Collection properties: `.atoms`, `.bonds`, …
- `def_*` / `add_*` / `del_*` thin wrappers
- `adopt` / `from_frame` / `to_frame` (frame upgrade to molpy `Frame` if still needed)
- `__repr__`, `__len__`, `__add__` / `__iadd__` if still desired (merge-based)
- `select` / `rename_type` / `set_property` — may stay as thin Python filters (not engine)

### Move or delete

| Item | Action |
|---|---|
| Remaining Python geometry helpers only used by align | use molrs; delete local copies |
| Duplicate batch `def_atoms` boilerplate | keep if short; else shared helper on `_GraphViews` |
| Dead comments referring to Struct/Mixin | delete |

## Design

### 1. LOC budget (binding soft gate)

```
wc -l src/molpy/core/atomistic.py   # ≤ 250
wc -l src/molpy/core/cg.py          # ≤ 200
```

If over budget after 01, extract pure helpers (e.g. `_frame_export.py`) without reintroducing
algorithms.

### 2. Construction invariant test

```python
def test_def_atom_uses_molrs_columns():
    m = Atomistic()
    a = m.def_atom(element="C", x=1.0, y=2.0, z=3.0)
    assert m.get(a.handle, "element") == "C"
    assert m.column("x")[0] == 1.0
```

### 3. No second graph store

```
grep -rn 'self\._atoms|self\._nodes|OrderedDict' src/molpy/core/ → no private atom lists
```

Intern tables (`WeakValueDictionary`) are views only — allowed.

### 4. Developer docs

State in one place:

- Users write `mol.def_atom(...)`.
- Extensions that skip views may use `mol.spawn()` / `mol.set(h, k, v)`.
- Users never `import molrs` (iron law 6) — document re-exports if any new symbols needed.

## Files to create or modify

- `src/molpy/core/atomistic.py`
- `src/molpy/core/cg.py`
- `src/molpy/core/entity.py` (only if shared sugar moves here)
- `tests/test_core/test_atomistic_editing.py` / new construction tests
- `docs/developer/molrs-backend.md` (or create short section)

## Tasks

- [x] **T1**: Audit def_*/_spawn_* paths — only molrs primitives
- [x] **T2**: Hit LOC budgets (refactor/extract)
- [x] **T3**: Construction invariant tests
- [x] **T4**: Grep guard against private atom lists
- [x] **T5**: Developer doc paragraph for API split C
- [x] **T6**: pytest green

## Testing strategy

- Existing editing tests remain green.
- New tests: def_atom → column visible; def_bond endpoints match handles; pending Entity
  binds on add.
- LOC check can be a test or CI note in acceptance (runtime `wc`).

## Out of scope

- pyo3-native Atom views
- pure re-export identity
- valence → 03
- ForceField / typifier changes
