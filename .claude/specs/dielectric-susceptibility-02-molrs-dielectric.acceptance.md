---
slug: dielectric-susceptibility-02-molrs-dielectric
criteria:
  - id: ac-001
    summary: compute_dipole_moment produces correct value for two-point charge
    type: scientific
    pass_when: |
      q=+1 at (0,0,0), q=−1 at (2,0,0) → M = (2.0, 0, 0) to machine precision.
    status: pending
  - id: ac-002
    summary: compute_current_density shape and NaN first row
    type: code
    pass_when: |
      3-frame input → (3,3) output; row 0 is NaN; rows 1+ finite.
    status: verified
    last_checked: 2026-05-16
  - id: ac-003
    summary: static_dielectric_constant returns epsilon_inf for zero fluctuation
    type: scientific
    pass_when: |
      All frames same dipole → ε(0) == ε_inf to 1e-12.
    status: pending
  - id: ac-004
    summary: einstein_helfand_spectrum importable from Rust and Python
    type: code
    pass_when: |
      `from molrs.dielectric import einstein_helfand_spectrum` succeeds.
    status: verified
    last_checked: 2026-05-16
  - id: ac-005
    summary: green_kubo_spectrum importable from Rust and Python
    type: code
    pass_when: |
      `from molrs.dielectric import green_kubo_spectrum` succeeds.
    status: verified
    last_checked: 2026-05-16
  - id: ac-006
    summary: decompose_current preserves total current
    type: scientific
    pass_when: |
      50/50 water/ion split: J_water + J_ion == J_total within 1e-12.
    status: pending
  - id: ac-007
    summary: Input arrays not mutated by any dielectric function
    type: runtime
    pass_when: |
      All six functions leave input arrays byte-identical.
    status: verified
    last_checked: 2026-05-16
  - id: ac-008
    summary: All six functions reject dimension mismatches
    type: code
    pass_when: |
      Each function raises ComputeError::DimensionMismatch for wrong-shaped inputs.
    status: verified
    last_checked: 2026-05-16
  - id: ac-009
    summary: Python bindings exist for all six functions
    type: code
    pass_when: |
      `dir(molrs.dielectric)` contains all six function names.
    status: verified
    last_checked: 2026-05-16
  - id: ac-010
    summary: EH zero-frequency limit matches static dielectric constant
    type: scientific
    pass_when: |
      einstein_helfand_spectrum at ω→0 matches static_dielectric_constant within 5%.
    status: pending
  - id: ac-011
    summary: Full test suite green
    type: code
    pass_when: |
      `cargo test --all-features -p molrs-compute && pytest molrs-python/tests/test_dielectric.py -v` exits 0.
    status: verified
    last_checked: 2026-05-16
