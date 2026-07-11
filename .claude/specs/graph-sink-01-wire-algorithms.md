---
title: "graph-sink 1/3 (molpy) — wire copy / merge / extract / CG spatial to molrs"
status: done
created: 2026-07-11
chain: graph-sink
depends_on: "molrs:graph-sink-01-extract; molrs:graph-sink-02-copy-merge; molrs:graph-sink-03-python-bind"
blocked_on: "molrs graph-sink-01 + 02 + 03 shipped and molcrafts-molrs pin available"
---

# Delete Python graph algorithms; delegate to molrs

> Chain **graph-sink** molpy 1/3.
> Locked decisions: `.claude/notes/graph-sink-decisions.md` (A/B/C).
> **BLOCKED** until molrs 01–03 are released and pin-able.

## Summary

Replace molpy reimplementations of graph engine operations with thin calls into
`molrs.Atomistic` / `molrs.CoarseGrain`:

| molpy method | New behaviour |
|---|---|
| `Atomistic.copy` / `CoarseGrain.copy` | molrs handle-preserving clone + fresh intern tables / props |
| `merge` | molrs structural merge; return or discard handle map; **no** `_detach` identity merge |
| `extract_subgraph` / `_extract_mapped` | molrs `extract_subgraph`; re-intern views; preserve `AffectedRegion` out_cls path |
| `move` / `rotate` / `scale` / `align` | already mostly molrs; **CG scale/align** must match AA (no per-bead Python Rodrigues) |
| `get_topo` | already delegates `generate_topology` — keep thin |

`def_*` and Entity views **unchanged** in this spec (→ 02).

## Domain basis

### What exists today

- `core/atomistic.py`: ~850 LOC; Python `copy`/`merge`/`_extract_mapped`/`align` partial.
- `core/cg.py`: CG `scale`/`align` still pure Python loops.
- molrs free fns: `translate` / `rotate` / `scale`.
- Consumers: `typifier` region extract, `builder/assembly` merge, crosslink copy paths,
  tests under `tests/test_core/test_atomistic*.py`, `test_copy_behavior.py`.

### Contract changes (breaking, iron law 3)

1. **copy**: handles preserved → update tests that assumed re-spawned handles.
2. **merge**: views into `other` are **not** rebound; `other` emptied; use handle map if
   needed. Public signature:

```python
def merge(self, other: Self) -> Self:
    """Structural merge (molrs). other is emptied. Handle identity is not preserved."""
```

Optional for advanced callers (same method or attribute):

```python
def merge_mapped(self, other: Self) -> dict[int, int]:
    """Like merge but returns other_handle -> self_handle."""
```

Prefer **one** public method that returns `Self` for chaining (current style) and stash the
map only if a caller needs it — check assembly: `world.merge(copy)` ignores return map.
If nothing needs the map in-tree, `merge` can ignore the dict (still call molrs for remap
semantics). **Grep before delete**; if a test relies on identity, rewrite to handles.

### `_extract_mapped` for AffectedRegion

Must keep the return shape used by typifier:

```python
(subgraph: G, boundary: list[Atom], region_to_parent: dict[Atom, Atom], hops: dict[int, int])
```

Implementation sketch:

```python
res = molrs.Atomistic.extract_subgraph(self, [c.handle for c in centers], radius,
                                       regenerate_topology=regenerate_topology)
sub = out_cls.adopt(res.graph)  # or construct out_cls + adopt storage
# intern boundary / maps via handles
```

`out_cls` may be `AffectedRegion` — adopt into empty instance of that subclass.

## Design

### 1. copy

```python
def copy(self) -> Self:
    cloned = molrs.Atomistic.copy(self)  # or type(self).__bases__ path
    new = type(self).adopt(cloned) if not isinstance(cloned, type(self)) else ...
    new._props = dict(self._props)
    return new
```

Careful with pyo3 subclass: `self` is already molrs leaf; `molrs.Atomistic.copy(self)` may
return bare `molrs.Atomistic`. Always `type(self).adopt(...)` or equivalent zero-copy take
into molpy subclass. Clear intern dicts on the new instance (`_GraphViews.__init__` path).

### 2. merge

```python
def merge(self, other: Self) -> Self:
    _map = molrs.Atomistic.merge(self, other)  # empties other
    # Drop any cached interns that pointed at moved-from other; self interns stay
    # (new handles are new — not in intern tables until accessed)
    return self
```

Delete `_detach` transfer loops.

### 3. CG spatial parity

```python
def scale(self, factor, about=None, ...):
    o = about or [0,0,0]
    molrs.scale(self, [factor, factor, factor], o)
    return self
```

`align` uses `molrs.rotate` + `move` like Atomistic.

### 4. Pin

Bump `molcrafts-molrs` in `pyproject.toml` / `uv.lock` to the release that contains
graph-sink 01–03.

## Files to create or modify

- `src/molpy/core/atomistic.py` — copy/merge/extract
- `src/molpy/core/cg.py` — copy/merge/scale/align/extract if any
- `src/molpy/core/entity.py` — only if merge touched intern helpers
- `src/molpy/typifier/**` / `builder/assembly/**` — only if identity-merge assumed
- `tests/test_core/test_copy_behavior.py`, `test_atomistic*.py`, `test_cg*.py`
- `pyproject.toml` + lockfile

## Tasks

- [x] **T1**: Pin molrs version that ships 01–03
- [x] **T2**: `copy` via molrs + adopt; update tests for handle preservation
- [x] **T3**: `merge` via molrs; delete identity path; fix callers
- [x] **T4**: `_extract_mapped` / `extract_subgraph` via molrs
- [x] **T5**: CG scale/align via molrs
- [x] **T6**: full `pytest tests/ -m "not external"` green
- [x] **T7**: changelog entry for breaking copy/merge semantics

## Testing strategy

- `test_copy_behavior`: assert handle equality pre/post copy for a known atom.
- merge two molecules: n_atoms sum; old view from `other` is unbound or world is empty.
- extract ball: same counts as golden small fixture (compare to pre-change snapshot once).
- typifier region tests still pass (incremental-typify suite).
- No `for atom in self.atoms: def_atom` inside `copy` (grep guard).

## Out of scope

- Thinning `def_*` LOC budget → `graph-sink-02-def-sugar`
- `complete_valence` → `graph-sink-03-valence`
- Pure re-export (`is molrs.Atomistic`)
