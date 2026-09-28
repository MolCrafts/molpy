---
title: "molrs 0.15 alignment 05 — delete the assembly stack and AmberPolymerBuilder"
status: code-complete
created: 2026-09-27
---

# molrs 0.15 alignment 05 — delete the assembly stack and AmberPolymerBuilder

## Summary

By ruling (ports only, no facades, the user composes native primitives), the molpy assembly stack is deleted wholesale. It was built on the retired `SITE` key, `SiteMap`, `Placer` and molpy typifier retyping. The stack comprises:
- `GraphAssembler`, `PolymerBuilder` and `MonomerLibrary`;
- the Selector family, `ResidueTopology`, `Replicas` and `AssemblyFinalizer`;
- the placer/orienter/`SiteMap`/`Trace` re-exports.

`AmberPolymerBuilder`, its package and `AmberTools.build_polymer` go with it. The `BONDED` finalization stage loses its typifier dependency. After this link, `import molpy.builder` succeeds and no builder module imports the typifier. Depends on links 01 and 02.

## Design

**Delete `src/molpy/builder/assembly/`** (13 files) as a whole directory, including its ignored `__pycache__/`. A leftover directory would still import as an empty namespace package.
- Composition moves to caller code and docs, built from native primitives that molpy re-exports:
  - `CGSmilesIR.to_coarsegrain`/`to_atomistic` (link 08);
  - `SubgraphMatcher.find` (link 07);
  - `center`/`translate`/`merge`;
  - `Fragment.def_port`/`link` (link 07).
- No molpy facade replaces it.

**Delete `src/molpy/builder/polymer/ambertools/`** (`__init__`, `amber_builder`, `amber_leap`, `amber_utils`, `types`) as a whole directory.
- This removes `AmberPolymerBuilder` and `AmberBuildResult`.
- It also removes the dead `amber_leap`/`amber_utils` and the accidental `molpy.builder.polymer.ambertools.amber_builder.CGSmilesIR` path (librarian (b)).

The test directories `tests/test_builder/test_assembly/` and `tests/test_builder/test_polymer/test_ambertools/` are removed the same way.

**`src/molpy/builder/ambertools.py`:**
- Delete `AmberTools.build_polymer` (`:241-301`), `_polymer_builder_key` (`:304-...`), the `_polymer_builders` cache attribute (`:78`) and the docstring bullet (`:11-12`).
- The `minimize` error message (`:184`) drops "/ build_polymer" and becomes "(produced by parameterize)".
- `AmberTools` keeps `parameterize`, `minimize` and `parameterize_ion`.

**`src/molpy/builder/_finalize.py`:**
- `Finalization` = `ATOMS | TOPOLOGY`. `StructureFinalizer` loses `bonded` and the `ForceFieldParams` import (`:11`).
- The class stays: it has two real users (`GrapheneBuilder`, `CarbonTubeBuilder`) and owns the stage and aromaticity decision (law 12). Its `Perceive` import and `remove_link` call were ported in link 02.
- Bonded-parameter assignment for typed graphs is recorded as a molrs ask (link 10).

**`src/molpy/builder/nanostructure/graphene.py` and `carbon_tube.py`:** drop the `bonded` keyword and the `ForceFieldParams` import (`:17`).

**`src/molpy/builder/__init__.py`:**
- Remove the assembly and `AmberPolymerBuilder` imports and their `__all__` entries.
- Rewrite the docstring (`:7-14`): no `fields.SITE` and no `PolymerBuilder`. Assembly is caller composition, with a pointer to the docs.
- The region re-exports (`:18`) stay until link 07.

**`src/molpy/builder/polymer/__init__.py:3-5`:** the docstring no longer points to assembly. Sequences, distributions and `SystemPlanner` stay; they have no assembly dependency.

**Owned elsewhere.** The `SmilesReader` brace-refusal message (`io/data/smiles.py:143-148`) still names `molpy.builder.assembly` after this link. Link 08 rewrites it to the `mp.CGSmilesIR` doors, with its test, because that root export lands there. This link's acceptance excludes that one file.

### Reuse decision

- **delete** `GraphAssembler`, `PolymerBuilder`, `MonomerLibrary`, `Expansion`, `AssemblyFinalizer`, the Selector family, `Replicas`, `ResidueTopology` + IR, `MatchContext`/`Binding`/`Candidate`, and the placer/orienter/`SiteMap`/`Trace` re-exports (ruling).
- **delete** `AmberPolymerBuilder`, `AmberBuildResult`, `amber_leap`, `amber_utils` and `AmberTools.build_polymer` (grill 2). Link 10 routes an operator confirmation against the active peo-tg campaign.
- **keep** `_finalize` minus `BONDED`: the nanostructures are its two users.
- **reuse (docs only)** the native composition primitives, re-exported in links 07 and 08.
- **new: none.**

### Laws

- **Law 1:** no second assembly model beside molrs.
- **Law 3:** no retained unused hooks.
- **Law 7:** primitives, not a workflow facade.
- **Law 10:** the typifier edge is cut before the typifier is deleted; the smiles.py message is owned by link 08.

## Files to create or modify

- src/molpy/builder/_finalize.py
- src/molpy/builder/nanostructure/graphene.py
- src/molpy/builder/nanostructure/carbon_tube.py
- src/molpy/builder/ambertools.py
- src/molpy/builder/__init__.py
- src/molpy/builder/polymer/__init__.py
- src/molpy/builder/assembly/__init__.py (delete)
- src/molpy/builder/assembly/_assembler.py (delete)
- src/molpy/builder/assembly/_context.py (delete)
- src/molpy/builder/assembly/_finalize.py (delete)
- src/molpy/builder/assembly/_library.py (delete)
- src/molpy/builder/assembly/_polymer.py (delete)
- src/molpy/builder/assembly/_proximity.py (delete)
- src/molpy/builder/assembly/_random.py (delete)
- src/molpy/builder/assembly/_replicas.py (delete)
- src/molpy/builder/assembly/_residue_graph.py (delete)
- src/molpy/builder/assembly/_residue_ir.py (delete)
- src/molpy/builder/assembly/_selector.py (delete)
- src/molpy/builder/assembly/_topology.py (delete)
- src/molpy/builder/polymer/ambertools/__init__.py (delete)
- src/molpy/builder/polymer/ambertools/amber_builder.py (delete)
- src/molpy/builder/polymer/ambertools/amber_leap.py (delete)
- src/molpy/builder/polymer/ambertools/amber_utils.py (delete)
- src/molpy/builder/polymer/ambertools/types.py (delete)
- tests/test_builder/test_finalize.py
- tests/test_builder/test_init.py (delete)
- tests/test_builder/test_assembly/conftest.py (delete)
- tests/test_builder/test_assembly/test_assembler.py (delete)
- tests/test_builder/test_assembly/test_context.py (delete)
- tests/test_builder/test_assembly/test_finalize.py (delete)
- tests/test_builder/test_assembly/test_init.py (delete)
- tests/test_builder/test_assembly/test_library.py (delete)
- tests/test_builder/test_assembly/test_placer.py (delete)
- tests/test_builder/test_assembly/test_polymer.py (delete)
- tests/test_builder/test_assembly/test_proximity.py (delete)
- tests/test_builder/test_assembly/test_random.py (delete)
- tests/test_builder/test_assembly/test_replicas.py (delete)
- tests/test_builder/test_assembly/test_residue_graph.py (delete)
- tests/test_builder/test_assembly/test_residue_ir.py (delete)
- tests/test_builder/test_assembly/test_selector.py (delete)
- tests/test_builder/test_assembly/test_sites.py (delete)
- tests/test_builder/test_assembly/test_topology.py (delete)
- tests/test_builder/test_polymer/test_ambertools/test_amber_builder.py (delete)
- tests/test_builder/test_polymer/test_ambertools/test_amber_leap.py (delete)
- tests/test_builder/test_polymer/test_ambertools/test_amber_utils.py (delete)
- tests/test_builder/test_polymer/test_ambertools/test_types.py (delete)

## Tasks

- [x] Write a failing test for the two-stage finalizer (tests/test_builder/test_finalize.py → `TestFinalization::test_stages_are_atoms_and_topology`). In the same file:
  - delete `test_bonded_stage_requires_a_parameter_assigner`, the `bonded=` rejection test and `test_bonded_stage_assigns_relation_types`;
  - remove the `ForceFieldParams` import.
- [x] Remove the `BONDED` stage and the `bonded` field from src/molpy/builder/_finalize.py, and the `bonded` keyword and `ForceFieldParams` import from src/molpy/builder/nanostructure/graphene.py and src/molpy/builder/nanostructure/carbon_tube.py.
- [x] Delete src/molpy/builder/assembly/ (13 files), tests/test_builder/test_assembly/ (16 files) and tests/test_builder/test_init.py. Remove both directories as a whole: `git rm` the files, then delete the directories with their `__pycache__/`.
- [x] Delete src/molpy/builder/polymer/ambertools/ (5 files) and tests/test_builder/test_polymer/test_ambertools/ (4 files), removing both directories as a whole the same way.
- [x] In src/molpy/builder/ambertools.py, delete `AmberTools.build_polymer`, `_polymer_builder_key`, the `_polymer_builders` cache and the docstring bullet, and fix the `minimize` message at `:184`.
- [x] Update the imports, `__all__` and docstring in src/molpy/builder/__init__.py, and the docstring in src/molpy/builder/polymer/__init__.py.
- [x] Verify that `uv run --extra dev python -c "import molpy.builder"` exits 0. Then run the named tests via test_single: tests/test_builder/test_finalize.py, tests/test_builder/test_nanostructure/test_carbon_tube.py (ported in link 02), tests/test_builder/test_polymer/test_sequences.py, tests/test_builder/test_polymer/test_system.py, tests/test_builder/test_polymer/test_distributions.py, tests/test_builder/test_crystal.py (ported in link 02), tests/test_builder/test_symmetry.py. Full check and suite are deferred to link 10.

## Testing strategy

- **Unit test.** `tests/test_builder/test_finalize.py::TestFinalization` targets only `Finalization`/`StructureFinalizer.apply`.
  - The stage values are exactly `["atoms", "topology"]`.
  - The existing `ATOMS`/`TOPOLOGY` behaviour tests stay.
- The nanostructure test guards the dropped keyword. test_crystal.py, test_carbon_tube.py and test_finalize.py also guard link 02's ports (including every `.xyz` → `atoms["x","y","z"]`), since `molpy.builder` now imports.
- **Deleted tests** exercised deleted code.
- **No end-to-end backmap test** (grill 7). No regression example (law 17). No physics.

## Out of scope

- The re-exports of the backmap primitives (links 07 and 08).
- The `SmilesReader` brace-refusal message (link 08).
- Polymer planning primitives (kept unchanged).
- Rewriting the assembly docs and examples (link 09).
- Sibling and lab-new consumers (routed in link 10, not edited).
