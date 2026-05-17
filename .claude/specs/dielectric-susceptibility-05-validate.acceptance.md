---
slug: dielectric-susceptibility-05-validate
criteria:
  - id: ac-001
    summary: molrs.validate.kramers_kronig_check exists and callable
    type: code
    pass_when: |
      hasattr(molrs.validate, 'kramers_kronig_check'); returns dict with keys passed, mae, eps_real_recovered.
    status: verified
    last_checked: 2026-05-16
  - id: ac-002
    summary: molrs.validate.conductivity_sum_rule_check callable
    type: code
    pass_when: |
      hasattr(molrs.validate, 'conductivity_sum_rule_check'); returns dict with passed, relative_error, integral, expected.
    status: verified
    last_checked: 2026-05-16
  - id: ac-003
    summary: molrs.validate.route_agreement_check callable
    type: code
    pass_when: |
      hasattr(molrs.validate, 'route_agreement_check'); returns dict with passed, pairwise_rms.
    status: verified
    last_checked: 2026-05-16
  - id: ac-004
    summary: kramers_kronig_check passes on Debye-derived spectra
    type: scientific
    pass_when: |
      Synthetic Debye spectrum (ε_s=70, ε_inf=2, τ=9ps, 200 frequency points) → passed=True, mae < 1e-3.
    status: pending
  - id: ac-005
    summary: conductivity_sum_rule_check passes on consistent data
    type: scientific
    pass_when: |
      Data satisfying sum rule within 5% → passed=True, |relative_error| < 0.05.
    status: pending
  - id: ac-006
    summary: route_agreement_check passes for identical spectra
    type: scientific
    pass_when: |
      Two identical ε(ω) arrays → passed=True, pairwise_rms["eh_vs_gk"] < 1e-10.
    status: pending
  - id: ac-007
    summary: make_debye_dipole_timeseries generates Debye-like ACF
    type: scientific
    pass_when: |
      Fixture produces (n_frames, 3) array; fitted τ within 5% of input τ.
    status: pending
  - id: ac-008
    summary: compare_to_literature rejects out-of-range values
    type: code
    pass_when: |
      compare_to_literature({"eps_0": 150, "tau_D": 100}, "spce") → passed=False with failures.
    status: verified
    last_checked: 2026-05-16
  - id: ac-009
    summary: End-to-end pipeline — all 3 validations pass on Debye data
    type: scientific
    pass_when: |
      Debye → ACFAnalyzer → SpectralAnalyzer → DielectricSusceptibility →
      kramers_kronig_check + conductivity_sum_rule_check + route_agreement_check → all passed=True.
    status: pending
  - id: ac-010
    summary: EH vs GK routes agree within 10% in 0.1–10 THz
    type: scientific
    pass_when: |
      Same synthetic Debye trajectory, EH and GK ε(ω) RMS relative difference < 10%.
    status: pending
  - id: ac-011
    summary: Full integration test suite passes
    type: runtime
    pass_when: |
      pytest tests/test_compute/test_dielectric_integration.py -v exits 0.
    status: verified
    last_checked: 2026-05-16
