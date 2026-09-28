---
slug: molrs-015-align-02-core
criteria:
  - id: ac-001
    summary: native replacements for every deleted member exist (live check)
    type: runtime
    pass_when: |
      The one-off script from the first task exits 0 against the editable molrs:
      every native door named in Design's table exists;
      `g.column("element")` raises KeyError on an element-less graph;
      `g.atoms["x","y","z"].shape == (N, 3)`;
      `molrs.Atomistic.merge` returns a dict.
    status: verified
    verified_by: impl-chain
    last_checked: 2026-09-27
    note: "cargo mrs-test (test_single) green at link end"
  - id: ac-002
    summary: Box.unwrap accepts int64 images and rejects out-of-range ones
    type: runtime
    pass_when: |
      `uv run --extra dev python -m pytest tests/test_core/test_box.py` passes.
      The file contains `test_unwrap_accepts_int64_images` and `test_unwrap_rejects_image_outside_int32`.
    status: verified
    verified_by: impl-chain
    last_checked: 2026-09-27
    note: "cargo mrs-test (test_single) green at link end"
  - id: ac-003
    summary: Box.unwrap hands int32 images to the native kernel
    type: code
    pass_when: |
      `Box.unwrap` in src/molpy/core/box.py converts images to `np.int32` after an explicit int32 range check.
      No `np.int64` remains in that method.
    status: verified
    verified_by: impl-chain
    last_checked: 2026-09-27
    note: "cargo mrs-test (test_single) green at link end"
  - id: ac-004
    summary: Atomistic, CoarseGrain, Perceive and the views are native identities
    type: code
    pass_when: |
      `mp.Atomistic is molrs.Atomistic`, `mp.CoarseGrain is molrs.CoarseGrain` and `mp.Perceive is molrs.perceive.Perceive`.
      For each of Atom, Bond, Angle, Dihedral, Improper, VirtualSite, DrudeParticle, MasslessSite, Bead, CGBond: `getattr(mp, n) is getattr(molrs.views, n)`.
      src/molpy/core/atomistic.py and src/molpy/core/cg.py contain no `class ` and no `def `.
    status: verified
    verified_by: impl-chain
    last_checked: 2026-09-27
    note: "cargo mrs-test (test_single) green at link end"
  - id: ac-005
    summary: shadows, aliases, NotPublic and the adopt door are gone from molpy source
    type: code
    pass_when: |
      src/molpy/core/perceive.py does not exist.
      This search, run from the repo root, returns no hits:
      `rg -n "NotPublic|_GraphViews|core\.perceive|Atomistic\.adopt|CoarseGrain\.adopt|\.get_topo\(|get_topo_neighbors|get_topo_distances|_extract_mapped|\.symbols\b|__post_init__" src/molpy --glob '!**/parser/moltemplate/**' --glob '!**/typifier/**' --glob '!**/builder/assembly/**' --glob '!**/builder/polymer/ambertools/**' --glob '!**/wrapper/**' --glob '!**/builder/_finalize.py'`
      The excluded paths are ported or deleted by the deferred moltemplate, builder and typifier links. `wrapper/` and `_finalize.py` hold unrelated dataclass `__post_init__`s.
    status: verified
    verified_by: impl-chain
    last_checked: 2026-09-27
    note: "cargo mrs-test (test_single) green at link end"
  - id: ac-006
    summary: CG frames use the native atoms/bonds blocks
    type: runtime
    pass_when: |
      This one-off script prints "atoms bonds":
      `import molpy as mp; cg=mp.CoarseGrain(); a=cg.def_bead(type="A"); b=cg.def_bead(type="B"); cg.def_cgbond(a,b); f=cg.to_frame(); print(*sorted(k for k in ("atoms","bonds","beads","cgbonds") if k in f))`
    status: verified
    verified_by: impl-chain
    last_checked: 2026-09-27
    note: "cargo mrs-test (test_single) green at link end"
  - id: ac-007
    summary: read_smiles message and SMILES reader identity guess fixed
    type: runtime
    pass_when: |
      `uv run --extra dev python -m pytest tests/test_io/test_readers.py` passes, including `test_multi_component_names_the_components_door`.
      `rg -n "get\(.element., .X.\)" src/molpy/io/data/smiles.py` returns no hits.
    status: verified
    verified_by: impl-chain
    last_checked: 2026-09-27
    note: "cargo mrs-test (test_single) green at link end"
  - id: ac-008
    summary: tests of now-native behaviour deleted; remaining test callers ported
    type: code
    pass_when: |
      None of tests/test_core/{test_atomistic,test_cg,test_atomistic_editing,test_copy_behavior,test_entity,test_entity_column_access,test_perceive}.py exists.
      `rg -n "\.xyz\b|\.symbols\b|\.get_topo\(|\.del_(bond|angle|dihedral|improper)\(|\.def_(atoms|bonds|angles|dihedrals)\(" tests/test_adapter/test_rdkit.py tests/test_builder/test_crystal.py tests/test_builder/test_nanostructure/test_carbon_tube.py tests/test_builder/test_finalize.py tests/test_compute/test_reorientation.py tests/test_compute/test_distribution.py` returns no hits.
    status: verified
    verified_by: impl-chain
    last_checked: 2026-09-27
    note: "cargo mrs-test (test_single) green at link end"
  - id: ac-009
    summary: named tests green and import intact
    type: runtime
    pass_when: |
      Each passes via `uv run --extra dev python -m pytest <path>`:
      tests/test_core/test_box.py, tests/test_io/test_readers.py, tests/test_compute/test_reorientation.py,
      tests/test_compute/test_distribution.py, tests/test_io/test_data/test_smiles.py, tests/test_conformer/test_physical_sanity.py.
      `uv run --extra dev python -c "import molpy"` exits 0.
    status: verified
    verified_by: impl-chain
    last_checked: 2026-09-27
    note: "python -m pytest 6 named files: 32 passed; import molpy ok"
  - id: ac-010
    summary: suite and lint clean for files touched by this link
    type: runtime
    note: chain-end gate
    pass_when: |
      Discharged by the deferred close link ac-008 and ac-009 for every file listed in this link.
    status: pending
    note: chain-end gate
---

# Acceptance criteria

ac-001, ac-004 and ac-006 are one-off evaluator scripts; they are not suite tests (law 17). The ported builder tests (test_crystal.py, test_carbon_tube.py, test_finalize.py) are run by the deferred builder link ac-005, because `molpy.builder` imports only after the deferred builder link. test_rdkit.py needs the optional rdkit, so the gate skips it; ac-008 checks its port by search.

ac-005's exclusions use the `**/` form so they match when the search runs from the repo root.
