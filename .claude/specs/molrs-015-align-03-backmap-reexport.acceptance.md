---
slug: molrs-015-align-03-backmap-reexport
criteria:
  - id: ac-001
    summary: SmilesReader brace refusal points to mp.CGSmilesIR
    type: runtime
    pass_when: |
      `uv run --extra dev python -m pytest tests/test_io/test_data/test_smiles.py` passes, including `test_smiles_reader_brace_notation_points_to_cgsmilesir`.
      `rg -n "builder\.assembly|linear_topology|PolymerBuilder" src/molpy/io/data/smiles.py` returns no hits.
    status: verified
    verified_by: impl-chain
    last_checked: 2026-09-27
    note: "python -m pytest test_smiles.py: 5 passed"
  - id: ac-002
    summary: CGSmilesIR family and SmilesError reachable from molpy by identity
    type: code
    pass_when: |
      For each n of CGSmilesIR, CGGraph, CGNode, CGEdge, CGFragmentDef, ResolvedPair, PairEnd, BondingDescriptor, SmilesError:
      `getattr(mp.parser, n) is getattr(molrs.io, n)` and n is in `mp.parser.__all__`.
      `mp.CGSmilesIR is molrs.io.CGSmilesIR` and `mp.SmilesError is molrs.io.SmilesError`, and both are in `mp.__all__`.
    status: verified
    verified_by: impl-chain
    last_checked: 2026-09-27
    note: "cargo mrs-test (test_single) green at link end"
  - id: ac-003
    summary: root band carries Fragment, Port, FrameMeta, MetaDocument, SubgraphMatcher
    type: code
    pass_when: |
      For each n of Fragment, Port, FrameMeta, MetaDocument: `getattr(mp, n) is getattr(molrs, n)` and n is in `mp.__all__`.
      `mp.SubgraphMatcher is molrs.perceive.SubgraphMatcher`, and "SubgraphMatcher" is in `mp.__all__`.
      `type(mp.Fragment().to_atomistic()) is mp.Atomistic`.
      The subclass comment in src/molpy/__init__.py names Box, Trajectory, Conformer and UnitSystem as subclasses, does not list Atomistic or CoarseGrain as subclasses, says the concrete regions subclass native shapes, and contains no link or spec number.
    status: verified
    verified_by: impl-chain
    last_checked: 2026-09-27
    note: "cargo mrs-test (test_single) green at link end"
  - id: ac-004
    summary: mp.op is a lazy verbatim namespace, registered in all four places
    type: runtime
    pass_when: |
      `mp.op.__all__ == molrs.op.__all__`, and `getattr(mp.op, n) is getattr(molrs.op, n)` for every n in it.
      "op" is in `molpy._LAZY_SUBMODULES`, `molpy.__all__` and `dir(molpy)`.
      `rg -n "^\s+op,$" src/molpy/__init__.py` shows the `TYPE_CHECKING` import entry.
      `python -c "import molpy.op"` exits 0.
      The `md/ optimize/ potential/` line of .claude/notes/architecture.md § Import-direction rules lists `op/`, and `git diff` shows no change between the `mol:map:managed begin`/`end` markers.
    status: verified
    verified_by: impl-chain
    last_checked: 2026-09-27
    note: "cargo mrs-test (test_single) green at link end"
  - id: ac-005
    summary: backmap script surface reachable from molpy alone, without builder/typifier
    type: runtime
    pass_when: |
      A one-off script run from a scratch directory outside the repo (not bare /tmp) with `uv run --project <molpy> --extra dev python` exits 0 after asserting:
      callable `mp.io.read_lammps_data`, `mp.io.read_smiles`, `mp.io.write_gro`, `mp.io.mrec.write_frame`;
      `mp.Frame` has `subset`, `__getitem__`; `mp.Block` has `__delitem__`;
      `mp.CoarseGrain` has `from_frame`, `center`, `relation_ids`, `relation_nodes`;
      `mp.SubgraphMatcher` has `find`;
      `mp.CGSmilesIR` has `to_coarsegrain`, `to_fragment`, `to_atomistic`; `mp.SmilesIR` has `to_atomistic`;
      `mp.conformer.Conformer is mp.Conformer` and it has `generate`;
      `mp.Fragment` has `copy`, `translate`, `center`, `ports`, `merge`, `link`, `to_atomistic`;
      `mp.Atomistic` has `merge`, `to_frame`;
      `mp.Box` has `wrap`, `unwrap`, and `mp.Box(np.eye(3) * 10.0)` constructs;
      `mp.UnitRegistry` has `define_lj_sigma`, `quantity`, `parse`, and `mp.Unit` has `factor_to`;
      `type(mp.CGSmilesIR("{[#A][#B]}").to_coarsegrain()) is mp.CoarseGrain`;
      at exit, "molpy.builder" and "molpy.typifier" are not in `sys.modules`.
    status: verified
    verified_by: impl-chain
    last_checked: 2026-09-27
    note: "cargo mrs-test (test_single) green at link end"
  - id: ac-006
    summary: import intact; no re-export tests added
    type: runtime
    pass_when: |
      `python -c "import molpy, molpy.op, molpy.parser, molpy.io"` exits 0.
      This link's tasks touch, under tests/, only tests/test_io/test_data/test_smiles.py (checked against the post-link-02 tree).
    status: verified
    verified_by: impl-chain
    last_checked: 2026-09-27
    note: "cargo mrs-test (test_single) green at link end"
  - id: ac-007
    summary: suite and lint clean for files touched by this link
    type: runtime
    note: chain-end gate
    pass_when: |
      Discharged by the deferred close link named in the notes entry of Design (e) (full suite + tox -e lint over every file in this link's Files section).
    status: pending
    note: chain-end gate
  - id: ac-008
    summary: deferred work recorded with a persisted owner
    type: docs
    pass_when: |
      .claude/notes/notes.md has "## 2026-09-27 — molrs-015-align narrowed to 01–03; deferred alignment" naming, each with Law, Evidence, Scope, Removal condition, Owner and Route:
      molpy.builder/molpy.typifier import breaks; moltemplate and LAMMPS FF writer/emitter/engine runtime breaks; the un-run full suite and lint gate;
      the molpy Region predicate model beside native shapes; the op/core.ops homonym; and the deferred links renumbered from 04 dropping what link 03 owns.
    status: verified
    verified_by: impl-chain
    last_checked: 2026-09-27
    note: "cargo mrs-test (test_single) green at link end"
---

# Acceptance criteria

The identity and reachability checks (ac-002 to ac-005) are one-off evaluator scripts, not suite tests. Law 17 forbids import-all and native-parity checks in the suite, and testing.md:19-21 gives re-exports no molpy test.

ac-005 checks only that the backmap surface is reachable: attribute presence, identity, and one CGsmiles door returning the molpy type. It does not run the backmap. The backmap is caller composition and belongs in docs. The `sys.modules` assertion makes the "no builder or typifier import" constraint binding.

There is no regression-example criterion, because law 17 forbids `regressions/`.
