---
slug: molrs-015-align-05-builder
criteria:
  - id: ac-001
    summary: finalizer has exactly two stages and no typifier dependency
    type: runtime
    pass_when: |
      `uv run --extra dev python -m pytest tests/test_builder/test_finalize.py` passes, including `test_stages_are_atoms_and_topology`.
      `rg -n "BONDED|bonded=|ForceFieldParams" src/molpy/builder` returns no hits.
    status: verified
    verified_by: impl-chain
    last_checked: 2026-09-28
    note: "cargo mrs-test (test_single) green at link end"
  - id: ac-002
    summary: assembly package and its tests deleted as directories
    type: code
    pass_when: |
      `git ls-files src/molpy/builder/assembly tests/test_builder/test_assembly` prints nothing, neither directory exists on disk, and tests/test_builder/test_init.py does not exist.
      `uv run --extra dev python -c "import molpy.builder.assembly"` fails with `ModuleNotFoundError`.
      This search, run from the repo root, returns no hits:
      `rg -n "builder\.assembly|GraphAssembler|PolymerBuilder|MonomerLibrary|SiteMap|TracePlacer" src/molpy --glob '!**/io/data/smiles.py'`
      The one excluded file is fixed and tested by molrs-015-align-08-io-reexport ac-001.
    status: verified
    verified_by: impl-chain
    last_checked: 2026-09-28
    note: "cargo mrs-test (test_single) green at link end"
  - id: ac-003
    summary: AmberPolymerBuilder and build_polymer deleted
    type: code
    pass_when: |
      `git ls-files src/molpy/builder/polymer/ambertools tests/test_builder/test_polymer/test_ambertools` prints nothing, and neither directory exists on disk.
      `rg -n "AmberPolymerBuilder|build_polymer|_polymer_builders|AmberBuildResult" src/molpy` returns no hits, including builder/ambertools.py:184.
    status: verified
    verified_by: impl-chain
    last_checked: 2026-09-28
    note: "cargo mrs-test (test_single) green at link end"
  - id: ac-004
    summary: molpy.builder imports cleanly with no typifier import
    type: runtime
    pass_when: |
      `uv run --extra dev python -c "import molpy.builder"` exits 0.
      `rg -n "molpy\.typifier" src/molpy/builder` returns no hits.
    status: verified
    verified_by: impl-chain
    last_checked: 2026-09-28
    note: "cargo mrs-test (test_single) green at link end"
  - id: ac-005
    summary: remaining builder tests green (including link 02's ported ones)
    type: runtime
    pass_when: |
      Each passes via test_single:
      tests/test_builder/test_nanostructure/test_carbon_tube.py, tests/test_builder/test_polymer/{test_sequences,test_system,test_distributions}.py and tests/test_builder/{test_crystal,test_symmetry,test_finalize}.py.
    status: verified
    verified_by: impl-chain
    last_checked: 2026-09-28
    note: "126 passed: test_nanostructure, test_polymer, test_crystal, test_symmetry, test_finalize"
  - id: ac-006
    summary: suite and lint clean for files touched by this link
    type: runtime
    note: chain-end gate
    pass_when: |
      Discharged by molrs-015-align-10-close ac-008 and ac-009 for every file listed in this link.
    status: pending
    note: chain-end gate
---

# Acceptance criteria

The molpack examples and the lab-new project scripts break at import after this link. They are named and routed in molrs-015-align-10-close and are not edited in this chain.

ac-002's exclusion uses the `**/` form so it matches when the search runs from the repo root. The directory checks use `git ls-files` plus absence on disk, because ignored `__pycache__/` folders survive a `git rm`.
