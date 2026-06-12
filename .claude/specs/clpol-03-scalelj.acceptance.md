---
slug: clpol-03-scalelj
criteria:
  - id: ac-001
    summary: scale_lj operator is importable from molpy.core.ops
    type: code
    pass_when: |
      `from molpy.core.ops import scale_lj, compute_k_ij,
      load_fragment_scaling_data, FragmentScaling` succeeds without error.
    status: verified
    last_checked: 2026-06-10
  - id: ac-002
    summary: scale_lj returns a new ForceField; input FF is unmutated
    type: code
    pass_when: |
      Returned FF is not the input object; every PairType epsilon/sigma in
      the input FF is unchanged after the call (test fixture asserts equality
      against a pre-call snapshot).
    status: verified
    last_checked: 2026-06-10
  - id: ac-003
    summary: sigma and atomic charges unchanged after default scaling
    type: code
    pass_when: |
      With scale_sigma=False, every returned PairType sigma equals its input
      sigma, and every atom charge in the struct equals its pre-call value.
    status: verified
    last_checked: 2026-06-10
  - id: ac-004
    summary: epsilon strictly reduced — every k_ij in (0, 1]
    type: code
    pass_when: |
      For the fixture system, each returned PairType epsilon equals
      k_ij * input_epsilon with the computed k_ij satisfying 0 < k_ij <= 1.
    status: verified
    last_checked: 2026-06-10
  - id: ac-005
    summary: k_ij reproduces the closed form with C0/C1 for a known pair
    type: scientific
    pass_when: |
      compute_k_ij for a fixture fragment pair (known q, mu, alpha, r) equals
      1/(1 + 0.254952*r^2*q^2/alpha + 0.106906*mu^2/alpha) within rtol 1e-6.
    status: verified
    last_checked: 2026-06-11
  - id: ac-006
    summary: mu^2 term carries no r^2 prefactor
    type: scientific
    pass_when: |
      Holding q, mu, alpha fixed and varying r, the mu^2/alpha contribution to
      the denominator of k_ij stays constant while only the q^2 term scales
      with r^2 (asserted by comparing two r values within rtol 1e-9).
    status: verified
    last_checked: 2026-06-11
  - id: ac-007
    summary: clandpol reference epsilon scaling matches for a known case
    type: scientific
    evaluator_hint: "skip if reference value unobtainable; mark via fixture"
    pass_when: |
      For a fully-nonpolarizable->polarizable fragment pair transcribed from
      clandpol fragment.ff, scaled epsilon matches the clandpol scaleLJ
      reference value within rtol 1e-4 (if reference obtainable).
    status: skipped
    note: clandpol reference epsilon values not obtainable; closed-form k_ij
      verification (ac-005) and code-level invariants (ac-006) provide
      sufficient scientific coverage.
  - id: ac-008
    summary: fragment scaling data file committed and loadable
    type: code
    pass_when: |
      src/molpy/data/forcefield/clpol_fragments.ff exists and
      load_fragment_scaling_data returns FragmentScaling entries exposing
      q, mu, alpha, polarizable for each fragment.
    status: verified
    last_checked: 2026-06-10
  - id: ac-009
    summary: full check + test suite passes
    type: runtime
    pass_when: |
      `ruff format --check src tests && ruff check src tests &&
      ty check src/molpy/ && pytest tests/ -m "not external"` all exit 0.
    status: verified
    last_checked: 2026-06-10
---

# Acceptance criteria

- ac-001..ac-004, ac-008 (code): operator surface, copy-not-mutate semantics,
  sigma/charge invariants, strict epsilon reduction, committed data file.
- ac-005..ac-007 (scientific): closed-form k_ij reproduction, the explicit
  no-r^2-prefactor-on-mu^2 check, and an optional clandpol reference match.
- ac-009 (runtime): repo-wide format/lint/type/test gate.
