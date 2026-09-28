---
title: "molrs 0.15 alignment 03 — backmap re-exports: Fragment, SubgraphMatcher, CGSmilesIR and mp.op"
status: code-complete
created: 2026-09-27
---

# molrs 0.15 alignment 03 — backmap re-exports: Fragment, SubgraphMatcher, CGSmilesIR and mp.op

## Summary

After this link, a CG→all-atom backmap script can be written against `molpy` alone. The root re-exports by identity `Fragment`, `Port`, `FrameMeta`, `MetaDocument`, `SubgraphMatcher`, `CGSmilesIR` and `SmilesError`. `mp.parser` carries the whole CGSmilesIR family. `mp.op` is a lazy, verbatim namespace over `molrs.op`. `SmilesReader` still refuses brace notation, but its message now points to `mp.CGSmilesIR(text).to_coarsegrain()` / `.to_atomistic()` instead of the deleted `molpy.builder.assembly`. The link adds no molpy behaviour beyond that message and no molpy class. Identity is checked by one-off acceptance scripts, not suite tests. The whole backmap surface is checked for reachability once, and the check also proves that neither `molpy.builder` nor `molpy.typifier` gets imported. It depends on links 01 and 02. Everything else from the former core and io re-export links is deferred.

## Design

Line numbers refer to the tree before the chain. Links 01 and 02 shift them, so locate each edit by its band or text.

### (a) Root identity band (`src/molpy/__init__.py`, `from molrs import` band `:194-219`, `__all__` `:409-492`)

- Add `Fragment`, `Port`, `FrameMeta` and `MetaDocument` to the `from molrs import (...)` band and to the molrs identity section of `__all__`.
- The region family is not added here, and neither is `Region`: they need `molpy.builder` (`Lattice.build`), which does not import until the deferred builder link. `mp.Region` stays molpy's own `MaskPredicate` model in `core/region.py`. That parallel model already exists and is owned by the deferred core re-export link; this link does not introduce it (law 10: named, not silent).
- `Fragment.to_atomistic()` returns `molrs.Atomistic`, which is `mp.Atomistic` since link 02. There is no wrapping door.
- There is no `Fragment` subclass: molpy adds no behaviour that would earn one (law 3).
- Rewrite the comment at `:173-176`: subclasses with real molpy additions are `Box`, `Trajectory`, `Conformer` and `UnitSystem`; `Atomistic`, `CoarseGrain` and `Perceive` are native identities; the base `Region` is molpy's own predicate model, while its concrete regions (`BoxRegion`, `SphereRegion`, the combinators) subclass the native shapes. No link or spec numbers appear in source text.

### (b) SubgraphMatcher

- `SubgraphMatcher` joins the root `from molrs.perceive import (...)` block (`:299-303`), which after link 02 also holds `Perceive`.
- Add it to `__all__`.

### (c) `mp.op` — a verbatim namespace (the `md` pattern)

- New `src/molpy/op/__init__.py`: a module docstring naming the native numeric base (weighted superposition, centroids), then `from molrs.op import *` and `from molrs.op import __all__ as __all__`.
- Register it lazily in the four places in `src/molpy/__init__.py`:
  - the `TYPE_CHECKING` import (`:19-31`);
  - `_LAZY_SUBMODULES` (`:37-50`);
  - `__dir__` (`:59`), which is covered by the set and needs no separate edit;
  - `__all__` (`:306-316`).
- `.claude/notes/architecture.md` § Import-direction rules (custom annotation, outside the `/mol:map` managed block): the line `md/ optimize/ potential/ → application code only (pure molrs re-exports)` gains `op/` and drops `potential/` (deleted by link 01). `op/__init__.py` carries `# noqa: F403` on its star import, like `md/__init__.py:7`. The managed Layer-roles table is not hand-edited.

### (d) CGSmiles (`src/molpy/parser/__init__.py`, the root, `src/molpy/io/data/smiles.py`)

- `parser/__init__.py` re-exports by identity from `molrs.io`: `CGSmilesIR`, `CGGraph`, `CGNode`, `CGEdge`, `CGFragmentDef`, `ResolvedPair`, `PairEnd`, `BondingDescriptor`, `SmilesError`. Add each to `__all__`.
- The `parser/__init__.py` docstring says CGsmiles is parsed by the type (`CGSmilesIR(text)`, then `.to_coarsegrain()`, `.to_fragment()` or `.to_atomistic()`), and that there is no molpy `CGSmilesReader`.
- The root imports `CGSmilesIR` and `SmilesError` beside `SmilesIR` (`:294`). The trailing comment becomes "parser types and their error, not file I/O entries". Add both to `__all__`.
- `io/data/smiles.py`:
  - the brace refusal in `SmilesReader._parse_graph` (`:143-148`, which still names `molpy.builder.assembly`) becomes a `ValueError` saying that molpy's SMILES reader does not parse BigSMILES / CGsmiles brace notation, and that CGsmiles is parsed with `mp.CGSmilesIR(text).to_coarsegrain()` (bead graph) or `mp.CGSmilesIR(text).to_atomistic()` (all-atom graph);
  - the class docstring bullet at `:31` ("Leading `{` → rejected (use assembly topology helpers)") names the same `mp.CGSmilesIR` doors.

### Reuse decision

- **reuse `molrs.Fragment`, `molrs.Port`, `molrs.FrameMeta`, `molrs.MetaDocument`**: identity on the root.
- **reuse `molrs.perceive.SubgraphMatcher`**: identity on the root.
- **reuse `molrs.op`**: a verbatim namespace, `mp.op`.
- **reuse the `molrs.io` CGSmilesIR family and `SmilesError`**: identity in `mp.parser`, with `CGSmilesIR` and `SmilesError` also on the root.
- **new: none.** The only molpy-owned change is the text of an existing refusal message. Naming follows the existing root identity bands and the `md` verbatim-namespace pattern.

### Laws

- **Law 1:** one fragment model, one CG pattern matcher and one CGsmiles parser, all native. The pre-existing molpy `Region` model is named above and left to its owning link.
- **Law 3:** no `Fragment` subclass and no molpy `CGSmilesReader`.
- **Law 7:** the backmap stays the caller's composition. No backmap façade is added.
- **Law 10:** the new `op/` package is recorded in the import-direction rule, and the deferred region model is named.
- **Law 12:** the backmap reachability check is a script, not a helper.
- **Law 13:** everything is an identity re-export or a verbatim namespace. Nothing forwards.
- **Law 17:** re-exports get no molpy test. Identity and reachability are one-off acceptance scripts. The only new test covers molpy-owned behaviour, the refusal message.

### (e) Deferred work gets a persisted owner (`.claude/notes/notes.md`)

The operator narrowed this chain (2026-09-27: "按收窄方案走，先把脚本跑通，然后再去做其他部分收尾") to links 01–03, so the backmap script can run first. What the narrowed chain leaves undone is recorded as one notes entry, "## 2026-09-27 — molrs-015-align narrowed to 01–03; deferred alignment (routed `/mol:spec`)", with § VII-style fields (Law, Evidence, Scope, Removal condition, Owner operator, Route):
- `molpy.builder` and `molpy.typifier` do not import (typifier subclassing and assembly on retired molrs APIs) — removal: the deferred builder and typifier links land;
- moltemplate (`parser/moltemplate/*`) and the LAMMPS force-field writers/emitter/engine are broken at runtime — removal: the deferred moltemplate and LAMMPS links land (design round-3 findings kept with them);
- tests still using removed API, owned by deferred links: `tests/test_typifier/*`, `tests/test_core/test_ops/test_scale_lj.py` (`AtomisticForcefield`, `def_atomstyle`/`def_pairstyle`), `tests/test_builder/test_finalize.py:52,64` (`AtomisticForcefield` for bonded typing), `tests/test_io/test_forcefield/test_lammps.py` (deleted style names), and callers of graph members removed by link 02 in tests that import builder/typifier: `tests/test_io/test_data/test_lammps_bond_react.py:125,160` (`get_neighbors`, `del_bond`), `tests/test_io/test_data/test_lammps_drude_flags.py:42,74` (`get_topo`, `def_atoms`), `tests/test_io/test_forcefield/test_xml.py:254` (`get_topo`), `tests/test_builder/test_polymer/test_ambertools/test_amber_builder.py:24` (`def_bonds`), `tests/test_typifier/*` — removal: their owning deferred links land;
- link 02's spec table names `atoms["x","y","z"]` for `adapter/rdkit.py` `_positions`, but that view returns `None` on a coordinate hole; the port uses `np.stack` of native `column()` (raises `KeyError`, law 14). Removal: the deferred close link amends the table;
- stale developer docs and CLAUDE.md sections naming `LammpsForceFieldFormatter`, `BondMorseStyle`, `ForceFieldFormatter`, `SITE`/`Q0`, the `potential` package — removal: the deferred docs and close links;
- the full suite and `tox -e lint` have not run for this chain; the builder/typifier/assembly tests fail at collection — removal: the deferred close link runs the gate green;
- `mp.Region` is still molpy's predicate `Region` beside the native shapes (law 1, parallel model) — removal: the deferred region migration;
- `molpy.op` (native numeric base) sits beside `molpy.core.ops` (LJ scaling helpers), a near-homonym — removal: the deferred deletion of `core/ops` into `mp.ff`;
- the deferred links are the staged `molrs-015-align` links renumbered from 04; they drop what this link now owns (`Fragment`/`Port`/`FrameMeta`/`MetaDocument`/`SubgraphMatcher`/`mp.op`/the CGSmilesIR family and SmilesReader message).

## Files to create or modify

- src/molpy/__init__.py
- src/molpy/op/__init__.py (new)
- src/molpy/parser/__init__.py
- src/molpy/io/data/smiles.py
- tests/test_io/test_data/test_smiles.py
- .claude/notes/architecture.md
- .claude/notes/notes.md

## Tasks

- [x] Write a failing test for the brace refusal (tests/test_io/test_data/test_smiles.py → free function `test_smiles_reader_brace_notation_points_to_cgsmilesir`, matching the file's free-function style). `SmilesReader("{[#EO]|3}").read()` must raise a `ValueError` whose message contains `mp.CGSmilesIR(` and `.to_coarsegrain()`.
- [x] Update the brace-refusal message in `SmilesReader._parse_graph` and the class docstring bullet in src/molpy/io/data/smiles.py so both name `mp.CGSmilesIR(text).to_coarsegrain()` / `.to_atomistic()`, with no mention of `molpy.builder.assembly`.
- [x] Re-export `CGSmilesIR`, `CGGraph`, `CGNode`, `CGEdge`, `CGFragmentDef`, `ResolvedPair`, `PairEnd`, `BondingDescriptor` and `SmilesError` from `molrs.io` in src/molpy/parser/__init__.py, updating `__all__` and the docstring (CGsmiles is parsed by the type; there is no `CGSmilesReader`). Add `CGSmilesIR` and `SmilesError` beside `SmilesIR` on the root in src/molpy/__init__.py and in `__all__`.
- [x] In src/molpy/__init__.py:
  - add `Fragment`, `Port`, `FrameMeta` and `MetaDocument` to the `from molrs import` band;
  - add `SubgraphMatcher` to the `from molrs.perceive import` block;
  - add all five names to `__all__`;
  - rewrite the subclass comment as given in Design (a).
- [x] Create src/molpy/op/__init__.py as a verbatim `molrs.op` namespace and register `op` lazily in src/molpy/__init__.py: in the `TYPE_CHECKING` import, in `_LAZY_SUBMODULES` (which `__dir__` covers) and in `__all__`. Add `op/` to the `md/ optimize/ potential/` line of § Import-direction rules in .claude/notes/architecture.md, outside the `/mol:map` managed block.
- [x] Verify identities and the backmap surface with a one-off script outside the suite. Run it from the session scratchpad directory (outside the repo, not bare /tmp) with `uv run --project <molpy> --extra dev python <script>`. Stop and report if any check fails. The script checks:
  - the identities of ac-002, ac-003 and ac-004;
  - every attribute listed in ac-005;
  - that `"molpy.builder"` and `"molpy.typifier"` are not in `sys.modules` at the end.
- [x] Record the deferred-work entry of Design (e) in .claude/notes/notes.md.
- [x] Verify that `python -c "import molpy, molpy.op, molpy.parser, molpy.io"` exits 0. Run tests/test_io/test_data/test_smiles.py via test_single. The full check and suite are deferred to the chain's close link.

## Testing strategy

- **New unit test.** `tests/test_io/test_data/test_smiles.py::test_smiles_reader_brace_notation_points_to_cgsmilesir` targets `SmilesReader.read` on the hand-written input `"{[#EO]|3}"`. The refusal happens before any native parse, so the test depends only on molpy's own check. It asserts `ValueError` with the fixed substrings `mp.CGSmilesIR(` and `.to_coarsegrain()`. The existing free-function tests in the file stay as they are.
- **Edge cases.** The refusal triggers on leading whitespace before `{` (the check is `lstrip().startswith("{")`). The test's plain `"{...}"` input covers the branch; plain SMILES inputs are already covered by the existing tests in the file.
- **No tests for re-exports** (testing.md:19-21; law 17). That covers `Fragment`, `Port`, `FrameMeta`, `MetaDocument`, `SubgraphMatcher`, the CGSmilesIR family, `SmilesError` and `mp.op`. They are proven by the one-off acceptance scripts (ac-002 to ac-005).
- **No end-to-end backmap test.** The backmap is caller composition and belongs in docs (law 11). **No regression example:** law 17 forbids `regressions/`. **No domain validation:** there is no physics change.

## Out of scope

- `FrameMeta`, `MetaDocument` and `mp.op` are not on the backmap surface; they are included because they are one-line identity re-exports in the same bands this link edits (law 13), and deferring them would re-open the same bands later.

- The native region family, `Region` identity, deleting `core/region.py` and the `Lattice.build` migration: they need `molpy.builder` (deferred core re-export link).
- `mp.ff`, deleting `core/ops/`, `mp.stream`, the compute gaps, deleting `compute/signal.py` and the native-parity compute test (deferred core re-export link).
- The io re-exports, forwarder identities, removal of the unused `frame` parameter, `read_top` / `read_amber` removal, `read_ac`, XML FF I/O and the verbatim `mp.io.mrec` (deferred io re-export link). The backmap only needs `read_lammps_data`, `read_smiles`, `write_gro` and `mrec.write_frame`, all of which are already reachable.
- Registering `conformer` as a lazy submodule. `mp.conformer` is reachable at runtime through the eager `Conformer` import, and `mp.Conformer` is the declared spelling.
- `molpy.builder` and `molpy.typifier` imports (links deferred).
- The `/mol:map` managed Layer-roles row for `op`, which comes from the `/mol:map` re-run routed in the close link.
