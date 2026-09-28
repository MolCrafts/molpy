---
title: "molrs 0.15 alignment 02 — core contract: box images; Atomistic, CoarseGrain and Perceive become native identities"
status: code-complete
created: 2026-09-27
---

# molrs 0.15 alignment 02 — core contract: box images; Atomistic, CoarseGrain and Perceive become native identities

## Summary

This link aligns molpy's core with the molrs 0.15 contract:
- `Box.unwrap` hands int32 image arrays to the native kernel.
- `mp.Atomistic is molrs.Atomistic`, `mp.CoarseGrain is molrs.CoarseGrain` and `mp.Perceive is molrs.perceive.Perceive` (operator ruling R2).

Once the same-name shadows are deleted, no molpy member is a real addition:
- the shadows with different semantics or a different signature: `adopt`, `merge`, `extract_subgraph`, `replicate`, `to_frame`, `def_bond`;
- the unused `__post_init__` hook and its `__init__` override;
- the aliases `positions`, `xyz`, `get_topo`, `get_topo_neighbors` and `get_topo_distances`.

What remains falls into three kinds, and none earns a subclass (law 13):
- a shadow that only re-classes a native result (`copy`, `from_frame`);
- an alias of a native primitive (the `del_*` and batch `def_*` families, `__iadd__`/`__add__`, `__len__`, `symbols`);
- a one-line caller composition (`select`, `rename_type`, `set_property`, `get_neighbors`, `beads_of`, `__repr__`).

So every native door (`copy`, `from_frame`, `Fragment.to_atomistic`, `CGSmilesIR.to_atomistic`/`to_coarsegrain`, perception, conformer) already returns the molpy type, and no `adopt` door exists anywhere. `NotPublic` goes. molpy source callers are ported, and so are test callers of removed members (including every `.xyz` use). Tests of now-native behaviour are deleted. Depends on link 01.

## Design

### `src/molpy/core/box.py:515-518` — `Box.unwrap`
- Converts `image` to `np.int32`; molrs 0.15 rejects int64 with `TypeError`.
- An image value outside the int32 range raises `ValueError` rather than wrapping silently (law 14).

### The identity decision, member by member

This table was checked against the editable molrs source: `molrs-python/python/molrs/views.py:474-1062` and `molrs-python/src/core/system/molgraph.rs:420-530, 840-1066, 1539-1760`. Task 1 re-checks it live before anything is deleted.

| molpy member (Atomistic / CoarseGrain) | Kind | What callers use instead |
|---|---|---|
| `adopt` (staticmethod returning a new object) | shadow, different semantics | nothing: native doors already return `molrs.Atomistic` / `molrs.CoarseGrain`; `g.adopt(other)` is the native instance move |
| `merge` (returns `self`, drops the map) | shadow, different semantics | native `merge(other) -> {old_handle: new_handle}` |
| `extract_subgraph` (tuple return), `_extract_mapped`, `_merge_map` | shadow, different signature | native `extract_subgraph(handles, radius, *, regenerate_topology, max_ring_size) -> ExtractedSubgraph` |
| `replicate(n, transform)` | shadow, different signature | native `replicate(template, rotations, translations, frag_ids)` |
| `to_frame(fields)` | shadow, different signature | native `to_frame()` (no field selection; select columns on the returned frame) |
| `def_bond`, `def_atom`, `def_virtual_site`, `def_angle`/`def_dihedral`/`def_improper`, `def_bead`, `def_cgbond`, `del_atom`, `del_bead`, `remove_link`, the collection properties, `_node_cls`, `_relation_classes` | re-declared native | the native members (native `def_bond` also stamps `bond_type`/`bond_number`) |
| `add_*` `NotPublic` attributes, `remove_entity` | dead | — |
| `copy`, `from_frame` | shadow that only re-classes | native `copy()`, `Atomistic.from_frame(frame)` |
| `__init__` + `__post_init__` hook | no user outside its own test | native `__init__` |
| `positions`, `xyz` | alias | `g.atoms["x", "y", "z"]` → (N, 3) (`views.py:400-402`) |
| `symbols` | alias; per R2 it must raise `KeyError` on a missing column, which is exactly the native behaviour | `g.column(fields.ELEMENT)` raises `KeyError` when the column is absent or has holes (`molgraph.rs:440-467`) |
| `get_topo` | alias | `g.generate_topology(gen_angle=, gen_dihedral=, gen_improper=, clear_existing=)` (in place; returns counts) |
| `get_topo_neighbors`, `get_topo_distances` | alias | `g.topo_distances(handle, max_hops=)` |
| `__len__` | alias | `g.n_nodes` (`n_atoms` / `n_beads`) |
| `def_atoms`/`def_bonds`/`def_angles`/`def_dihedrals`, `def_beads`/`def_cgbonds` | alias (a loop) | a loop over `def_atom` / `def_bond` / … |
| `del_bond`/`del_angle`/`del_dihedral`/`del_improper`, `del_cgbond` | alias | `remove_link(*links)` |
| `__iadd__`, `__add__` | alias | `merge`; `copy()` then `merge` |
| `select(pred)`, CG `_subset` | caller composition | `g.induced_subgraph([a.handle for a in g.atoms if pred(a)])[0]` |
| `rename_type`, `set_property`, `_items_of_kind` | caller composition | a loop over `g.atoms` or a relation view with item assignment |
| `get_neighbors`, `beads_of` | caller composition, no `src/` caller | `g.incident_relations(h, "bonds")`; `cg.beads_of_atom(atom.handle)` (handles) |
| `__repr__` | cosmetic | native repr |

**Native gaps that are not worked around.** They have no molpy `src/` consumer after the deferred builder link and are recorded as a molrs ask in the deferred close link:
- native `copy()` does not carry the Python-side `props`, or `CoarseGrain._member_world`, so `bead["atoms"]` on a copied CoarseGrain resolves to `()`;
- native `merge()` leaves `other`'s interned view tables populated.

### `src/molpy/core/atomistic.py` and `src/molpy/core/cg.py`
- Each becomes an identity re-export list from `molrs.views`, with no class and no function:
  - atomistic: `Angle`, `Atom`, `Atomistic`, `Bond`, `Dihedral`, `DrudeParticle`, `Improper`, `MasslessSite`, `VirtualSite`;
  - cg: `Bead`, `CGBond`, `CoarseGrain`.
- The modules stay so the internal `from molpy.core.atomistic import …` paths keep resolving.

### `src/molpy/core/perceive.py` — delete
Every method was `super()` plus an adopt. `Perceive` joins the root `from molrs.perceive import` band (`__init__.py:299-303`) and `core/__init__.py`.

### `src/molpy/core/entity.py`
Delete `NotPublic` and the private `_GraphViews` import and `__all__` entry. The `Entity`/`Link`/`Entities` aliases are routed in the deferred close link.

### molpy `src/` callers, ported here
- `conformer/__init__.py:63-66`: return the native `generate` result directly. The docstrings (`:3-7`, `:43-58`) drop "re-adopt". The class keeps only its empty-molecule guard (routed in the deferred close link).
- `io/readers.py:444-455`: `return ir.to_atomistic()`. The multi-component message drops "and adopt each".
- `io/data/smiles.py`:
  - `Perceive` comes from `molrs.perceive` (`:123`);
  - `get_topo` (`:129`) becomes `out.generate_topology(gen_angle=True, gen_dihedral=True)`, in place;
  - `atom.get('element', 'X')` (`:133`) becomes `atom["element"]` (law 14: no guessed element);
  - `:140`, `:158` become `return ir.to_atomistic()`;
  - the docstrings `:9`, `:32` name `mp.Perceive`.
- `adapter/rdkit.py:254`: `atomistic.atoms["x", "y", "z"]`.
- `builder/_finalize.py`: `Perceive` from `molrs.perceive` (`:10`); `:46-47` become `graph.remove_link(*graph.angles, *graph.dihedrals)`.
- `parser/__init__.py:35-36` docstring table: `ir.to_atomistic()` and `ir.components()`, with no adopt.

### Test callers of removed members, ported here
Every use of `.xyz`, `symbols`, `get_topo`, batch `def_*` and `del_*` in kept tests moves to the native door from the table:
- `tests/test_compute/test_reorientation.py:29` → `mol.to_frame()`; `get_topo()` with no flags was a no-op;
- `tests/test_compute/test_distribution.py:29` → `generate_topology(gen_angle=True, gen_dihedral=True)`, then `to_frame()`;
- `tests/test_builder/test_crystal.py:169-191` → `structure.atoms["element"]`; `:220`, `:264`, `:277` (`structure.xyz`) → `structure.atoms["x", "y", "z"]`;
- `tests/test_builder/test_nanostructure/test_carbon_tube.py:33` (`tube.xyz[:, :2]`) and `:50` (`builder.build().xyz[0, :2]`) → `atoms["x", "y", "z"]`;
- `tests/test_builder/test_finalize.py:14` → a `def_bond` loop, and `:39-40` → `remove_link`;
- `tests/test_adapter/test_rdkit.py:86` → `list(m.atoms["element"])`; `:61`, `:107`, `:118`, `:134-135` (`m.xyz` / `out.xyz`) → `atoms["x", "y", "z"]`.

The gate skips test_rdkit.py (rdkit is optional), so ac-008 checks its port by search. The builder tests import `molpy.builder`, which imports only after the deferred builder link, so the deferred builder link ac-005 runs them.

**Ported in other links (explicit and owned):**
- Moltemplate (`parser/moltemplate/builder.py:641, 1006, 1232, 1240`; `py_emitter.py:530, 654, 700`) is ported by the deferred moltemplate link, which rewrites both files. Between links 02 and 03 moltemplate stays runtime-broken, as it already is after link 01.
- `typifier/affected_region.py:189` goes with the typifier in the deferred typifier link; `builder/assembly/` goes in the deferred builder link.

### Reuse decision

- **reuse `molrs.Atomistic`, `molrs.CoarseGrain`, `molrs.perceive.Perceive` and the `molrs.views` node/relation views** — identity (librarian: subclass shrink, taken to its end).
- **delete** every molpy member in the table. Callers use the native door named there.
- **new: none.**

### Laws

- **Law 1:** one graph model, with no same-name shadow of different meaning.
- **Law 3:** the `__post_init__` hook has no user.
- **Law 7:** no convenience aliases as separate contracts.
- **Law 13:** identity when molpy adds no real behaviour.
- **Law 14:** the int32 range raises; `symbols` is replaced by the raising native column; no `'X'` element guess.
- **Law 17:** tests of now-native behaviour are deleted.

## Files to create or modify

- src/molpy/core/box.py
- src/molpy/core/atomistic.py
- src/molpy/core/cg.py
- src/molpy/core/entity.py
- src/molpy/core/perceive.py (delete)
- src/molpy/core/__init__.py
- src/molpy/__init__.py
- src/molpy/conformer/__init__.py
- src/molpy/io/readers.py
- src/molpy/io/data/smiles.py
- src/molpy/adapter/rdkit.py
- src/molpy/builder/_finalize.py
- src/molpy/parser/__init__.py
- tests/test_core/test_box.py
- tests/test_io/test_readers.py (new)
- tests/test_core/test_atomistic.py (delete)
- tests/test_core/test_cg.py (delete)
- tests/test_core/test_atomistic_editing.py (delete)
- tests/test_core/test_copy_behavior.py (delete)
- tests/test_core/test_entity.py (delete)
- tests/test_core/test_entity_column_access.py (delete)
- tests/test_core/test_perceive.py (delete)
- tests/test_compute/test_reorientation.py
- tests/test_compute/test_distribution.py
- tests/test_builder/test_crystal.py
- tests/test_builder/test_nanostructure/test_carbon_tube.py
- tests/test_builder/test_finalize.py
- tests/test_adapter/test_rdkit.py

## Tasks

- [x] Verify the identity decision live with a one-off script, run with the editable molrs from a scratch directory outside the repo (not a suite test). Stop and report if any check fails:
  - `molrs.Atomistic` has `generate_topology`, `topo_distances`, `induced_subgraph`, `extract_subgraph`, `remove_link`, `from_frame`;
  - `merge` returns a dict, and `copy()` returns a `molrs.Atomistic`;
  - `g.column("element")` raises `KeyError` on a graph without elements;
  - `g.atoms["x","y","z"]` has shape (N, 3);
  - `molrs.CoarseGrain` has `beads_of_atom` and `induced_subgraph`.
- [x] Write failing tests for `Box.unwrap` (tests/test_core/test_box.py → TestBoxOps):
  - `test_unwrap_accepts_int64_images`, with the expected coordinates hand-computed as xyz + image·L on an orthogonal box;
  - `test_unwrap_rejects_image_outside_int32` (`ValueError`).
- [x] Implement the int32 image conversion with a range check in `Box.unwrap` (src/molpy/core/box.py).
- [x] Write a failing test in tests/test_io/test_readers.py (new) → `TestReadSmiles::test_multi_component_names_the_components_door`. It moves `test_rejects_a_multi_component_smiles` out of test_perceive.py: `mp.io.read_smiles("[Li+].[F-]")` raises `ValueError` whose message contains `components()` and not `adopt`.
- [x] Replace src/molpy/core/atomistic.py and src/molpy/core/cg.py with the `molrs.views` identity lists, delete src/molpy/core/perceive.py, and delete `NotPublic` and `_GraphViews` from src/molpy/core/entity.py. Take `Perceive` from `molrs.perceive` in src/molpy/core/__init__.py and src/molpy/__init__.py.
- [x] Port the molpy callers listed in Design: src/molpy/conformer/__init__.py, src/molpy/io/readers.py, src/molpy/io/data/smiles.py (including `:133`), src/molpy/adapter/rdkit.py, src/molpy/builder/_finalize.py, src/molpy/parser/__init__.py.
- [x] Delete the tests whose subject is now native behaviour: tests/test_core/test_atomistic.py, test_cg.py, test_atomistic_editing.py, test_copy_behavior.py, test_entity.py, test_entity_column_access.py and test_perceive.py.
- [x] Port every remaining test caller of a removed member, as listed in Design ("Test callers of removed members"):
  - tests/test_compute/test_reorientation.py:29 and tests/test_compute/test_distribution.py:29;
  - tests/test_builder/test_crystal.py:169-191, `:220`, `:264`, `:277`;
  - tests/test_builder/test_nanostructure/test_carbon_tube.py:33, `:50`;
  - tests/test_builder/test_finalize.py:14, `:39-40`;
  - tests/test_adapter/test_rdkit.py:61, `:86`, `:107`, `:118`, `:134-135`.
- [x] Run the named tests via test_single: tests/test_core/test_box.py, tests/test_io/test_readers.py, tests/test_compute/test_reorientation.py, tests/test_compute/test_distribution.py, tests/test_io/test_data/test_smiles.py, tests/test_conformer/test_physical_sanity.py. The ported builder tests import `molpy.builder`, so they run in the deferred builder link. test_rdkit.py needs the optional rdkit and is checked by search (ac-008). Full check and suite are deferred to the deferred close link.

## Testing strategy

- **New unit tests:**
  - `tests/test_core/test_box.py::TestBoxOps` targets `Box.unwrap`. Happy path: an orthogonal 10 Å box with image `np.array([[1, 0, -1]], dtype=np.int64)` gives `xyz + [10, 0, -10]`. Edge case: an image of `2**31` raises `ValueError`.
  - `tests/test_io/test_readers.py::TestReadSmiles` targets `read_smiles`'s own multi-component check and its message.
- **Deleted tests.** Every test whose subject is now a native member goes: the graph API, views, copy, merge, perception, `NotPublic`, the `__post_init__` hook. Those behaviours are proven in molrs (law 17; testing.md:19-21).
- **Ported tests** keep their molpy subject (compute, crystal, carbon tube, finalize, adapter). They only swap removed conveniences for the native door.
- **Not tested here.** The identity checks are one-off acceptance scripts. No regression example (law 17). No physics.

## Out of scope

- Moltemplate's graph calls (deferred moltemplate link).
- The `Fragment`/`Port` re-exports (link 03).
- The native copy/merge view-state gaps: a molrs ask in the deferred close link.
- The `Conformer` subclass, reduced to its empty guard: routed in the deferred close link.
- The `Entity`/`Link`/`Entities` aliases in `core/entity.py`: routed in the deferred close link.
- Updating architecture.md Graph sink decision C and Non-goals `:536`: the deferred close link.
