---
title: dielectric-susceptibility-05-validate — Domain validation
status: code-complete
created: 2026-05-13
depends_on:
  - dielectric-susceptibility-01-molrs-signal
  - dielectric-susceptibility-02-molrs-dielectric
  - dielectric-susceptibility-03-python-types
  - dielectric-susceptibility-04-python-compute
---

# dielectric-susceptibility-05-validate

## Summary

Provide domain validation functions and integration tests. Validation computation in molrs (Rust), synthetic generators and literature comparison in Python test files (not production code).

### molrs validation functions

Three pure functions in `molrs.validate`:
- `kramers_kronig_check` — Hilbert transform ε'' → recovered ε', compare to input
- `conductivity_sum_rule_check` — ∫σ(ω)dω vs (π/2)⟨J²⟩/(3V k_B T)
- `route_agreement_check` — pairwise RMS between route spectra

### Python test helpers (test-only, NOT in src/)

- `make_debye_dipole_timeseries` — synthetic Debye dipole generator (pytest fixture)
- `compare_to_literature` — compare ε(0), τ_D against SPC/E and TIP4P/2005 ranges

### Integration tests

End-to-end: synthetic Debye trajectory → ACFAnalyzer → SpectralAnalyzer → DielectricSusceptibility → all 3 validation checks pass + literature comparison passes.

## Domain basis

**Kramers-Kronig relation**: ε'_recovered(ω_i) = ε_∞ + (2/π) Σ_{j≠i} [ε''(ω_j)·ω_j/(ω_j²−ω_i²)]·Δω_j. Ref: Jackson, Classical Electrodynamics, 3rd ed., Wiley, 1999.

**Conductivity sum rule**: ∫₀^∞ σ(ω)dω = (π/2)·⟨J²⟩/(3V k_B T). Ref: McQuarrie, Statistical Mechanics, 2000; Kubo, Rep. Prog. Phys. 29, 255 (1966).

**Debye model**: ⟨M(t)·M(0)⟩/⟨M²(0)⟩ = exp(−t/τ), ε*(ω) = ε_∞ + (ε_s−ε_∞)/(1+iωτ). Ref: Debye, Phys. Z. 13, 97 (1912).

**Literature targets** (bulk water, 298 K, non-polarizable):

| Property | SPC/E | TIP4P/2005 |
|----------|-------|------------|
| ε(0) | 68–72 | 60–65 |
| τ_D (ps) | 8–10 | 6–8 |
| ε_∞ | 1 | 1 |
| f_D (GHz) | ~20 | ~20 |

## Design

### molrs.validate (Rust)

```rust
pub fn kramers_kronig_check(frequency: &Array1<f64>, eps_real: &Array1<f64>, eps_imag: &Array1<f64>, eps_inf: f64) -> KramersKronigResult;
pub fn conductivity_sum_rule_check(frequency: &Array1<f64>, conductivity: &Array1<f64>, current_sq_mean: f64, volume: f64, temperature: f64) -> SumRuleResult;
pub fn route_agreement_check(results: HashMap<String, Array1<f64>>) -> RouteAgreementResult;
```

### Python bindings

```python
molrs.validate.kramers_kronig_check(frequency, eps_real, eps_imag, eps_inf) -> dict
molrs.validate.conductivity_sum_rule_check(frequency, conductivity, current_sq_mean, volume, temperature) -> dict
molrs.validate.route_agreement_check(results: dict[str, np.ndarray]) -> dict
```

Each returns `{passed: bool, ...}` with relevant metrics.

### Python test file: `tests/test_compute/test_dielectric_integration.py`

- `make_debye_dipole_timeseries(n_frames, dt, eps_s, eps_inf, tau, temperature, volume, seed)` — fixture generating (n_frames, 3) dipole array whose ACF follows exp(−t/τ)
- `compare_to_literature(computed, target)` — returns `{passed: bool, failures: list[str]}`
- Four test classes: `TestKramersKronig`, `TestSumRule`, `TestRouteAgreement`, `TestEndToEnd`

## Files

- `molrs/molrs-python/src/validate.rs` (new)
- `molrs/molrs-python/src/lib.rs` (modify — register validate submodule)
- `molpy/tests/test_compute/test_dielectric_integration.py` (new)

## Tasks

- [x] Write failing Python tests for all 3 validation functions with synthetic arrays
- [x] Implement `kramers_kronig_check` in validate.rs; wire as `molrs.validate.kramers_kronig_check`
- [x] Implement `conductivity_sum_rule_check` in validate.rs; wire binding
- [x] Implement `route_agreement_check` in validate.rs; wire binding
- [x] Write `make_debye_dipole_timeseries` fixture and `compare_to_literature` helper in test file
- [x] Write end-to-end integration test: synthetic Debye → pipeline → all validations pass + literature comparison
- [x] Write route comparison test: EH vs GK agree within 10% in 0.1–10 THz
- [x] Run full check + test suite

## Testing strategy

- **Unit (validation functions)**: perfect input → passed=True; deliberately corrupted → passed=False; single frequency point edge case; zero arrays
- **Unit (test helpers)**: Debye ACF fits exp(−t/τ) within 5%; literature comparison passes/fails correctly
- **Integration (end-to-end)**: full pipeline on Debye data, all 3 validations pass, SPC/E literature check passes
- **Integration (route agreement)**: EH and GK on same Debye data, pairwise RMS < 10% in 0.1–10 THz

## Out of scope

- Non-Debye relaxation models (Cole-Cole, etc.)
- Real MD trajectory I/O — tests use purely synthetic data
- Statistical error estimation (bootstrap, block averaging)
- Performance optimization of validation functions
