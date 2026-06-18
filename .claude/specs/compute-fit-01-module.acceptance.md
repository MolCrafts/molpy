---
slug: compute-fit-01-module
criteria:
  - id: ac-001
    summary: Fit trait defined alongside Compute with GAT input + ComputeResult output
    type: code
    evaluator_hint: ""
    pass_when: |
      compute::traits exposes a public `Fit` trait with an associated GAT
      `Input<'a>`, an associated `Output: ComputeResult + Clone + Send + Sync + 'static`,
      and a method `fit(&self, input: Self::Input<'a>) -> Result<Self::Output, ComputeError>`.
      An in-crate trivial impl over `&Array1<f64>` compiles and `cargo check --all-features` passes.
    status: pending
  - id: ac-002
    summary: LinearFit recovers known slope/intercept/r2 on synthetic linear data
    type: code
    evaluator_hint: ""
    pass_when: |
      For y[i] = 3.0*x[i] + 2.0 sampled on a uniform grid, LinearFit { window: (0.0, 1.0) }.fit
      returns slope within 1e-12 of 3.0, intercept within 1e-12 of 2.0, r2 within 1e-12 of 1.0,
      and fit_start == 0, fit_end == last index.
    status: pending
  - id: ac-003
    summary: LinearFit OLS matches the existing dielectric inline fit
    type: code
    evaluator_hint: ""
    pass_when: |
      On the same lag_times/msd array and (fit_start_frac, fit_end_frac) used by
      einstein_helfand_conductivity, LinearFit returns a slope equal to that function's
      reported slope to within 1e-12 (same OLS, lifted not reimplemented).
    status: pending
  - id: ac-004
    summary: RunningIntegral (trapezoid) recovers a known integral
    type: code
    evaluator_hint: ""
    pass_when: |
      For a constant curve c on step dt, RunningIntegral.fit returns a curve whose k-th element
      equals c*k*dt within 1e-12; and its output equals the running trapezoid integral computed
      inline in jacf.rs (green_kubo_conductivity) on the same JACF + dt to within 1e-12.
    status: pending
  - id: ac-005
    summary: Plateau returns windowed mean over a curve
    type: code
    evaluator_hint: ""
    pass_when: |
      For a curve that is constant value P over the fractional window (a, b), Plateau { window: (a, b) }.fit
      returns a plateau value equal to P within 1e-12 and a sample count equal to the number of points
      inside [round(a*(n-1)), round(b*(n-1))].
    status: pending
  - id: ac-006
    summary: PowerSpectrum Fit reproduces old power_spectrum output bit-for-bit
    type: code
    evaluator_hint: ""
    pass_when: |
      Given the raw velocity ACF that power_spectrum computes internally, the PowerSpectrum Fit applied
      to that ACF with the same dt and window/FFT params returns frequencies_cm1 and intensities arrays
      bitwise-equal (==) to the legacy power_spectrum output on the same velocity input.
    status: pending
  - id: ac-007
    summary: IRSpectrum and RamanSpectrum Fits reproduce old spectra output bit-for-bit
    type: code
    evaluator_hint: ""
    pass_when: |
      For matched raw ACFs and identical window/FFT/temperature/incident-frequency params, the IRSpectrum
      Fit equals legacy ir_spectrum output and the RamanSpectrum Fit equals legacy raman_spectrum output
      (isotropic, anisotropic, parallel, perpendicular) bitwise-equal (==) on the same inputs.
    status: pending
  - id: ac-008
    summary: Spectral Fits compose molrs::signal window primitives, not reimplemented windows
    type: code
    evaluator_hint: ""
    pass_when: |
      compute/fit/spectral.rs contains no local cosine/Hann/Blackman window coefficient computation;
      windowing routes through molrs::signal::apply_window (or the shared helper that calls it), and the
      legacy spectra::window_and_fft / dielectric::windowed_acf_spectrum now delegate to the shared helper.
    status: pending
  - id: ac-009
    summary: EinsteinConductivity raw MSD equals ConductivityResult.msd
    type: code
    evaluator_hint: ""
    pass_when: |
      For the same translational dipole series, dt, and max_correlation_time, EinsteinConductivity.compute
      returns lag_times and a raw msd curve element-wise equal to einstein_helfand_conductivity's
      ConductivityResult.lag_times and .msd to within 1e-12, and carries no fitted sigma/slope field.
    status: pending
  - id: ac-010
    summary: GreenKuboConductivity raw current ACF equals JacfResult.jacf
    type: code
    evaluator_hint: ""
    pass_when: |
      For the same current series, dt, and max_correlation_time, GreenKuboConductivity.compute returns
      lag_times and a raw jacf curve element-wise equal to green_kubo_conductivity's JacfResult.lag_times
      and .jacf to within 1e-12, and carries no fitted sigma field.
    status: pending
  - id: ac-011
    summary: VACF raw curve equals the internal acf_sum of power_spectrum
    type: code
    evaluator_hint: ""
    pass_when: |
      For the same velocity series, dt, and resolution, VACF.compute returns an unnormalized velocity ACF
      element-wise equal (within 1e-12) to the acf_sum that power_spectrum builds before windowing.
    status: pending
  - id: ac-012
    summary: EinsteinDiffusion delegates to the existing MSD primitive
    type: code
    evaluator_hint: ""
    pass_when: |
      EinsteinDiffusion.compute over a frame slice returns an MSD curve element-wise equal (within 1e-12)
      to MSD::windowed().compute over the same frames; no MSD math is re-derived in raw_computes.rs.
    status: pending
  - id: ac-013
    summary: DebyeRelaxation carries unnormalized ACF plus zero-lag variance and V/T/BC metadata
    type: code
    evaluator_hint: ""
    pass_when: |
      DebyeRelaxation's result type exposes a non-normalized dipole ACF, a zero-lag variance field
      <M(0)^2> equal to the ACF value at lag 0, plus scalar fields for volume, temperature, and the
      Ewald boundary condition; constructing the result without these fields fails to compile.
    status: pending
  - id: ac-014
    summary: Raw max_lag sufficiency is enforced (invariant a)
    type: code
    evaluator_hint: ""
    pass_when: |
      A Fit/transform asked to consume a lag range exceeding the raw curve's length returns
      Err(ComputeError::OutOfRange) (or EmptyInput for empty/too-short curves) rather than silently
      truncating.
    status: pending
  - id: ac-015
    summary: Einstein and Green-Kubo conductivity reconstruct legacy sigma from raw + Fit
    type: scientific
    evaluator_hint: ""
    pass_when: |
      For a representative dipole/current series: LinearFit slope on EinsteinConductivity.msd divided by
      (6*V*k_B*T) (with the documented MD->SI prefactor) equals einstein_helfand_conductivity's sigma to a
      relative tolerance of 1e-9; and RunningIntegral on GreenKuboConductivity.jacf scaled by
      1/(3*V*k_B*T) equals green_kubo_conductivity's sigma to a relative tolerance of 1e-9.
    status: pending
  - id: ac-016
    summary: Degenerate fit window and shape errors return typed ComputeError
    type: code
    evaluator_hint: ""
    pass_when: |
      LinearFit with window (a, b) where a >= b, or where all x in the window are equal, returns
      Err(ComputeError::OutOfRange); a spectral Fit given a non-1D ACF or Raman given a non-6-component
      input returns Err(ComputeError::DimensionMismatch).
    status: pending
  - id: ac-017
    summary: Full check + test suite passes
    type: runtime
    evaluator_hint: ""
    pass_when: |
      cargo fmt --all --check && cargo clippy --all-targets --all-features -- -D warnings && cargo test --all-features
      all succeed with the new compute::fit module and raw computes compiled in (BLAS env per project notes
      for --all-features).
    status: pending
---

# Acceptance criteria

- **ac-001** — `Fit` trait exists and type-checks alongside `Compute`.
- **ac-002 / ac-003** — `LinearFit` is correct on synthetic data and equivalent to the lifted dielectric OLS.
- **ac-004 / ac-005** — `RunningIntegral` and `Plateau` recover known integrals/plateaus and match the lifted jacf trapezoid.
- **ac-006 / ac-007 / ac-008** — spectral `Fit` impls reproduce the legacy `power_spectrum`/`ir_spectrum`/`raman_spectrum` (and dielectric spectrum) outputs bit-for-bit and compose `molrs::signal` windows rather than reimplementing them.
- **ac-009 / ac-010 / ac-011 / ac-012** — new raw computes equal the raw portion of today's bundled results / the existing MSD primitive.
- **ac-013** — `DebyeRelaxation` carries the unnormalized ACF + ⟨M(0)²⟩ + V/T/Ewald-BC metadata (invariants b, c).
- **ac-014** — raw `max_lag` sufficiency enforced (invariant a).
- **ac-015** — raw + `Fit` pipeline reconstructs legacy σ (scientific parity).
- **ac-016** — typed errors on degenerate windows and bad shapes.
- **ac-017** — full check + test suite green.
