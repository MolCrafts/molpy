---
spec: release-0-12-molpy-03-charge-transport
created: 2026-08-04
criteria:
  - id: ac-001
    summary: JACF uses Σ q v
    type: scientific
    pass_when: "With q=±2, |J| is 2× monovalent case on same velocities; missing charge errors"
    status: cancelled
  - id: ac-002
    summary: PMSD uses Σ q r
    type: scientific
    pass_when: "PMSD scales as q² for identical unwrapped trajectories"
    status: cancelled
  - id: ac-003
    summary: docs match charge-weighted formulas
    type: docs
    pass_when: "transport.md presents J=Σqv and M=Σqr without monovalent-only definition"
    status: cancelled
  - id: ac-004
    summary: suite green
    type: runtime
    pass_when: "pytest tests/test_compute/test_jacf.py tests/test_compute/test_pmsd.py -v passes"
    status: cancelled
out_of_scope:
  - Unit fs (prior spec)
---

# Acceptance — release-0-12-molpy-03-charge-transport

JACF/PMSD are charge-weighted; monovalent ±1 is not the physics definition.
