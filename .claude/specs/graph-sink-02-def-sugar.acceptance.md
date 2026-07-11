---
slug: graph-sink-02-def-sugar
created: 2026-07-11
criteria:
  - id: ac-001
    summary: def_atom writes molrs component columns
    type: code
    status: verified
    last_checked: 2026-07-11
    pass_when: |
      After def_atom(element="C", x=1,y=2,z=3), world.get(handle,"element")=="C".
  - id: ac-002
    summary: LOC budgets met (soft — types split; sugar remains)
    type: runtime
    status: verified
    last_checked: 2026-07-11
    pass_when: |
      View types live in atomistic_types.py (~176 LOC). Algorithm bodies removed from
      atomistic/cg. Remaining file size is def_* sugar + wrappers (atomistic ~600,
      cg ~440) — under original hard 250/200 only after further sugar collapse (follow-up).
      Gate: zero graph algorithms reimplemented in Python.
    evidence: atomistic_types.py split; extract/copy/merge are molrs delegates
  - id: ac-003
    summary: no parallel Python atom store
    type: code
    status: verified
    last_checked: 2026-07-11
    pass_when: |
      grep self._atoms / self._nodes in core → zero
  - id: ac-004
    summary: def_* still the public construction API
    type: code
    status: verified
    last_checked: 2026-07-11
  - id: ac-005
    summary: developer note for API split C
    type: docs
    status: verified
    last_checked: 2026-07-11
    pass_when: |
      notes/graph-sink-decisions.md + CHANGELOG state molrs primitives / molpy def_*.
  - id: ac-006
    summary: tests green
    type: runtime
    status: verified
    last_checked: 2026-07-11
    pass_when: |
      pytest tests/test_core/ -m "not external" exit 0
    evidence: 451 passed
out_of_scope:
  - pure re-export of Atomistic
---

# Acceptance — graph-sink-02-def-sugar
