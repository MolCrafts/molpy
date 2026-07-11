---
title: "graph-sink 3/3 (molpy) — complete_valence → molrs add_hydrogens"
status: done
created: 2026-07-11
chain: graph-sink
depends_on: "graph-sink-01-wire-algorithms; molrs:graph-sink-04-hydrogens-coords"
blocked_on: "molrs graph-sink-04 shipped (geometric add_hydrogens)"
---

# Unify valence completion with molrs

> Chain **graph-sink** molpy 3/3 (terminal).
> Depends on wire-up (01) and molrs geometric `add_hydrogens` (04).

## Summary

1. `Atomistic.complete_valence` / `molpy.core.capping.complete_valence` become a **thin
   forward** to `molrs.add_hydrogens` (+ adopt into molpy `Atomistic`).
2. Delete the Python valence table / tetrahedral placement once molrs 04 matches scientific
   acceptance (counts + coords).
3. `CoarseGrain.complete_valence` stays identity (already correct — beads have no valence).
4. Final pin bump; changelog; grep for dead capping helpers.

## Domain basis

### Why this is engine, not system policy

Filling implicit H from element valence + bond orders is a **structure/chemistry fact**
(same class as aromaticity / Gasteiger). Typifiers *consume* a completed graph; they do not
define valence.

### Behaviour to preserve for typify-region

- Under-coordinated fragment gains explicit H.
- When fragment has xyz, H has coordinates (antechamber / sqm path).
- Original heavy atoms keep **handle identity** under decision A (clone-based):
  first `n_heavy` handles stable; new H appended with new handles.
- Completing an already-complete molecule is identity on connectivity (may still clone).

### Mapping

| molpy today | molrs |
|---|---|
| `complete_valence(struct)` | `Atomistic.adopt(add_hydrogens(struct))` or in-place API if added |
| `_VALENCE` / charge tweaks | `implicit_h_count` (RDKit-like charge rule) |
| `_cap_directions` | molrs 04 placement |

**Charge rule differences** between old molpy table and molrs are resolved **in favor of
molrs** (decision: engine owns chemistry). Update any test that hardcoded the old O⁻/N⁺
edge cases to match molrs.

## Design

### 1. Implementation

```python
# core/capping.py
def complete_valence(struct: Atomistic) -> Atomistic:
    import molrs
    return Atomistic.adopt(molrs.add_hydrogens(struct))
```

If `add_hydrogens` returns a bare molrs graph, always `adopt`.
If it mutates in place in a future API, still return a molpy `Atomistic` without double-wrap.

### 2. Method on class

```python
def complete_valence(self) -> Atomistic:
    return complete_valence(self)
```

Keep method for typifier polymorphism (`graph.complete_valence()`).

### 3. Delete

- `_VALENCE`, `_CAP_LEN`, `_cap_directions`, and related private helpers in `capping.py`
  once unused.
- Prefer deleting `capping.py` body down to the forwarder (file may remain for stable import
  path `molpy.core.capping.complete_valence`).

### 4. Pin + close chain

- Bump `molcrafts-molrs` to version including graph-sink-04.
- Changelog: valence completion delegated to molrs.
- INDEX: mark graph-sink molpy chain complete when acceptance verified.

## Files to create or modify

- `src/molpy/core/capping.py`
- `src/molpy/core/atomistic.py` (method only if needed)
- `tests/test_core/test_capping.py`
- `pyproject.toml` / lockfile
- `CHANGELOG.md`

## Tasks

- [x] **T1**: Pin molrs with geometric add_hydrogens
- [x] **T2**: Forward complete_valence; delete Python geometry/valence tables
- [x] **T3**: Update capping tests to molrs chemistry rules + coord assertions
- [x] **T4**: typifier region tests still pass (capped fragments)
- [x] **T5**: pytest -m "not external" green
- [x] **T6**: changelog

## Testing strategy

- Bare C → 4 H with distances ~1.09 Å (scientific).
- Complete methane → no extra H (connectivity identity).
- Fragment from extract_subgraph then complete_valence → no under-valent heavies.
- CoarseGrain.complete_valence is copy/identity without calling add_hydrogens.

## Out of scope

- Changing typifier reach policy
- RDKit hydrogen policies beyond molrs
- New VirtualSite behaviour
