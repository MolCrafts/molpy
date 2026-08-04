---
spec: release-0-12-molpy-04-compute-tests
created: 2026-08-04
criteria:
  - id: ac-001
    summary: signal/hbond/van_hove/spectra/voronoi have unit tests
    type: runtime
    pass_when: "pytest discovers and passes tests for each of the five modules"
    status: cancelled
  - id: ac-002
    summary: tests assert no frame mutation and Compute subclass where applicable
    type: code
    pass_when: "each new test module checks issubclass/Compute and/or no-mutation pattern"
    status: cancelled
out_of_scope:
  - Full science oracles
---

# Acceptance — release-0-12-molpy-04-compute-tests

Public compute thin shells have mirrored unit tests.
