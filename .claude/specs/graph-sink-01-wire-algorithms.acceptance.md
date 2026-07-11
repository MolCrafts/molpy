---
slug: graph-sink-01-wire-algorithms
created: 2026-07-11
criteria:
  - id: ac-001
    summary: copy preserves handles
    type: code
    status: verified
    last_checked: 2026-07-11
    pass_when: |
      a = mol.def_atom(element="C", x=0,y=0,z=0); h = a.handle; c = mol.copy();
      any(atom.handle == h for atom in c.atoms); mol is not c.
    evidence: smoke + test_copy_behavior suite
  - id: ac-002
    summary: no Python re-spawn copy implementation
    type: code
    status: verified
    last_checked: 2026-07-11
    pass_when: |
      Atomistic.copy / CoarseGrain.copy call molrs copy + adopt only.
  - id: ac-003
    summary: merge is structural; other emptied; no identity rebind
    type: code
    status: verified
    last_checked: 2026-07-11
    pass_when: |
      n0 = len(a); n1 = len(b); a.merge(b); len(a) == n0+n1; len(b) == 0.
  - id: ac-004
    summary: extract_subgraph delegates to molrs; AffectedRegion path works
    type: code
    status: verified
    last_checked: 2026-07-11
    pass_when: |
      _extract_mapped uses molrs.Atomistic.extract_subgraph; typifier/builder suites green.
    evidence: tests/test_core + test_typifier + test_builder 738 passed
  - id: ac-005
    summary: CG scale/align use molrs free functions
    type: code
    status: verified
    last_checked: 2026-07-11
    pass_when: |
      CoarseGrain.scale/align call molrs.scale/rotate; no _rodrigues_rotate loop.
  - id: ac-006
    summary: test suite green
    type: runtime
    status: verified
    last_checked: 2026-07-11
    pass_when: |
      pytest tests/test_core tests/test_typifier tests/test_builder -m "not external" exit 0
    evidence: 738 passed
  - id: ac-007
    summary: changelog records breaking copy/merge semantics
    type: docs
    status: verified
    last_checked: 2026-07-11
    pass_when: |
      CHANGELOG Unreleased mentions handle-preserving copy and structural merge.
out_of_scope:
  - def_* thinning LOC budget (see 02)
---

# Acceptance — graph-sink-01-wire-algorithms

Done means molpy no longer owns graph copy/merge/extract/spatial kernels.
