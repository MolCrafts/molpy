---
slug: clpol-02-damping-potentials
criteria:
  - id: ac-001
    summary: Thole and TangToennies importable from molpy.potential.pair
    type: code
    pass_when: |
      `from molpy.potential.pair import Thole, TangToennies` succeeds and both
      appear in molpy.potential.pair.__all__.
    status: verified
    last_checked: 2026-06-10
  - id: ac-002
    summary: Both evaluators expose calc_energy returning finite scalar
    type: code
    pass_when: |
      Thole().calc_energy(...) and TangToennies().calc_energy(...) on a sample
      pair list return a Python float that is finite (not nan/inf).
    status: verified
    last_checked: 2026-06-10
  - id: ac-003
    summary: Both evaluators expose calc_forces with shape (n_atoms, 3)
    type: code
    pass_when: |
      calc_forces(...) for both classes returns an np.ndarray of shape
      (n_atoms, 3) with all-finite values; empty pair list returns zeros.
    status: verified
    last_checked: 2026-06-10
  - id: ac-004
    summary: Thole T_ij reproduces closed-form damping at sample r
    type: scientific
    evaluator_hint: "marker: thole"
    pass_when: |
      Thole damping factor at sampled r equals
      1 - (1 + s_ij*r/2)*exp(-s_ij*r) with s_ij = a_ij/(alpha_i*alpha_j)^(1/6),
      a_ij=(a_i+a_j)/2, default a=2.6, within abs tol 1e-10.
    status: pending
  - id: ac-005
    summary: Tang-Toennies f_n reproduces closed-form damping at sample r
    type: scientific
    evaluator_hint: "marker: tang_toennies"
    pass_when: |
      TangToennies damping factor at sampled r equals
      1 - c*exp(-b*r)*sum_{k=0..4}(b*r)^k/k! with n=4, b=4.5, c=1.0,
      within abs tol 1e-10.
    status: pending
  - id: ac-006
    summary: Analytic force equals finite-difference of energy for both
    type: scientific
    pass_when: |
      For both Thole and TangToennies, calc_forces matches the central
      finite-difference gradient of calc_energy within rel tol 1e-5.
    status: pending
  - id: ac-007
    summary: Damping factors approach 1 at long range
    type: scientific
    pass_when: |
      As r -> infinity, Thole T_ij -> 1 and TT f_n -> 1 (no damping),
      verified at large r within abs tol 1e-6.
    status: pending
  - id: ac-008
    summary: Damping is strong at short range
    type: scientific
    pass_when: |
      As r -> 0, both Thole T_ij and TT f_n are well below 1 and decrease
      monotonically toward strong damping over a sampled short-range grid.
    status: pending
  - id: ac-009
    summary: Full check + test suite passes
    type: runtime
    pass_when: |
      `ruff format src tests && ruff check src tests && ty check src/molpy/`
      and `pytest tests/ -m "not external" -v` all exit 0.
    status: verified
    last_checked: 2026-06-10
---

# Acceptance criteria

- ac-001..003 (code): public import, presence and finiteness of `calc_energy`/`calc_forces`, and correct force array shape for both `Thole` and `TangToennies`.
- ac-004/005 (scientific): each damping factor reproduces its verified closed form (Thole 1981; Tang-Toennies 1984) at sampled distances.
- ac-006 (scientific): analytic force is the negative gradient of the damped energy, validated against finite difference.
- ac-007/008 (scientific): correct physical limits — no damping at long range, strong damping at short range.
- ac-009 (runtime): project format/lint/type/test gate is green.
