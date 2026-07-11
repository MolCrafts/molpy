---
slug: graph-sink-03-valence
created: 2026-07-11
criteria:
  - id: ac-001
    summary: complete_valence delegates to molrs.add_hydrogens
    type: code
    status: verified
    last_checked: 2026-07-11
    pass_when: |
      capping.complete_valence calls molrs.add_hydrogens; no local _VALENCE tables.
  - id: ac-002
    summary: geometric caps when xyz present
    type: scientific
    status: verified
    last_checked: 2026-07-11
    pass_when: |
      C–H distance ~1.09 Å after complete_valence on under-valent C with xyz.
    evidence: tests/test_core/test_capping.py::test_cap_bond_length_matches_element
  - id: ac-003
    summary: complete molecule is connectivity-stable
    type: code
    status: verified
    last_checked: 2026-07-11
    evidence: TestCompletionIsIdentityOnCompleteMolecules
  - id: ac-004
    summary: CoarseGrain.complete_valence stays identity-like
    type: code
    status: verified
    last_checked: 2026-07-11
  - id: ac-005
    summary: typify-region path still green
    type: runtime
    status: verified
    last_checked: 2026-07-11
    evidence: tests/test_typifier + test_builder green in wire-up run
  - id: ac-006
    summary: full non-external suite + changelog
    type: runtime
    status: verified
    last_checked: 2026-07-11
    pass_when: |
      core/typifier/builder green; CHANGELOG notes valence → molrs.
out_of_scope:
  - force-field minimization of H positions
---

# Acceptance — graph-sink-03-valence
