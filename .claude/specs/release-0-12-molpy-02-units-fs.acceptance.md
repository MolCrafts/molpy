---
spec: release-0-12-molpy-02-units-fs
created: 2026-08-04
criteria:
  - id: ac-001
    summary: JACF prefactor uses fs
    type: scientific
    pass_when: "JACF SI path uses 1e-15 time conversion; docs say dt in fs"
    status: cancelled
  - id: ac-002
    summary: dielectric docs/prefactors use fs
    type: scientific
    pass_when: "dielectric.py and IonicConductivity document and convert dt as fs"
    status: cancelled
  - id: ac-003
    summary: no LAMMPS-real=ps claim in compute docs
    type: docs
    pass_when: "docs/compute and compute/*.py do not claim LAMMPS real time is ps"
    status: cancelled
  - id: ac-004
    summary: suite green
    type: runtime
    pass_when: "pytest tests/test_compute -v passes"
    status: cancelled
out_of_scope:
  - Charge-weighted J construction
---

# Acceptance — release-0-12-molpy-02-units-fs

Analysis trunk uses fs consistently with molrs science SSOT.
