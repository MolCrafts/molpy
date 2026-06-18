---
slug: compute-fit-02-repoint
criteria:
  - id: ac-001
    summary: New raw-compute + fit classes are importable from molrs
    type: code
    evaluator_hint: ""
    pass_when: |
      After rebuilding the maturin wheel, `import molrs` exposes
      EinsteinConductivity, GreenKuboConductivity, EinsteinDiffusion,
      GreenKuboDiffusion, VACF, DebyeRelaxation, LinearFit, RunningIntegral,
      Plateau, DebyeFit, PowerSpectrum, IRSpectrum, RamanSpectrum as
      constructible classes (each `hasattr(molrs, name)` is True and callable).
    status: pending
  - id: ac-002
    summary: Raw-compute returns only its curve, no fitted coefficient
    type: code
    evaluator_hint: ""
    pass_when: |
      Calling .compute(...) on EinsteinConductivity / GreenKuboConductivity /
      VACF returns a raw curve (ndarray or dict of arrays) and does NOT contain
      a fitted scalar (no 'sigma'/'D' key on the raw result); the coefficient
      appears only after composing the matching fit class.
    status: pending
  - id: ac-003
    summary: Legacy free-function bindings emit DeprecationWarning, unchanged output
    type: runtime
    evaluator_hint: "pytest.warns(DeprecationWarning)"
    pass_when: |
      Each of dielectric_einstein_helfand_conductivity,
      dielectric_einstein_helfand_spectrum, dielectric_green_kubo_spectrum,
      transport_green_kubo_conductivity, and the power/ir/raman spectrum
      bindings emits exactly one DeprecationWarning when called, and the
      returned dict/scalar is byte-for-byte identical to the pre-change output
      on the same fixture.
    status: pending
  - id: ac-004
    summary: New pipeline reproduces old conductivity/diffusion coefficient
    type: scientific
    evaluator_hint: ""
    pass_when: |
      On a fixed trajectory fixture, σ from (raw EinsteinConductivity →
      LinearFit) equals σ from legacy dielectric_einstein_helfand_conductivity,
      and σ from (raw GreenKuboConductivity → RunningIntegral → Plateau) equals
      σ from legacy transport_green_kubo_conductivity; likewise D from the
      Einstein/Green-Kubo diffusion path equals the legacy bundled D — each
      within rtol <= 1e-12 (identical arithmetic) or the documented float
      tolerance where summation order differs.
    status: pending
  - id: ac-005
    summary: New spectral transforms reproduce old VDOS/IR/Raman output
    type: scientific
    evaluator_hint: ""
    pass_when: |
      PowerSpectrum / IRSpectrum / RamanSpectrum applied to a fixed velocity /
      dipole / polarizability fixture reproduce the legacy power_spectrum /
      ir_spectrum / raman_spectrum frequencies_cm1 and intensities within
      rtol <= 1e-12 (or documented float tolerance), with matching array shapes.
    status: pending
  - id: ac-006
    summary: molpy wrappers preserve public output through delegation
    type: scientific
    evaluator_hint: ""
    pass_when: |
      For dielectric.py, jacf.py, msd.py, onsager.py, mcd.py, persist.py,
      decomposition.py, each migrated wrapper's public output equals its
      pre-migration output on the shared fixture within the wrapper's existing
      tolerance; and molpy DebyeFit (now delegating) matches direct molrs
      DebyeFit on the same input within rtol <= 1e-12.
    status: pending
  - id: ac-007
    summary: No fitting/spectra math reimplemented in molpy
    type: code
    evaluator_hint: ""
    pass_when: |
      After migration, molpy/src/molpy/compute/{dielectric,jacf,msd,onsager,
      mcd,persist,decomposition,result}.py contain no numeric fitting/spectra
      implementation (no local least-squares / FFT / running-integral loop);
      they only construct and call molrs raw-compute + fit classes. result.py
      no longer defines an ad-hoc DebyeFit body.
    status: pending
  - id: ac-008
    summary: maturin wheel rebuilt and molpy integration tests green against it
    type: runtime
    evaluator_hint: ""
    pass_when: |
      The maturin wheel is rebuilt from the modified molrs-python sources and
      installed, then `pytest molpy/tests/compute` passes with zero failures
      and zero errors against the freshly built wheel.
    status: pending
  - id: ac-009
    summary: Bench reference-library equality holds via raw+fit path
    type: scientific
    evaluator_hint: "marker: parity"
    pass_when: |
      In bm-molrs-molpy, the conductivity/diffusion/spectra benches call
      raw-compute then fit and their equality checks vs the named reference
      libraries pass: conductivity vs LAMMPS reference, diffusion vs
      MDAnalysis/freud, VDOS/IR/Raman vs scipy.
    status: pending
  - id: ac-010
    summary: freud-parity two-budget floors preserved, not tightened
    type: performance
    evaluator_hint: "marker: freud-parity"
    pass_when: |
      The freud-parity benches in bm-molrs-molpy still pass their existing
      two-budget (float32-aware) assertions; the assertion thresholds are
      unchanged from before the repoint (no switch to a single value-range
      atol).
    status: pending
  - id: ac-011
    summary: Full check + test suite passes
    type: runtime
    evaluator_hint: ""
    pass_when: |
      `cargo fmt --all --check && cargo clippy -- -D warnings && cargo check`
      passes for molrs-python, and the molrs + molpy + bench test suites run
      green after the repoint.
    status: pending
---

# Acceptance criteria

- **ac-001 / ac-002 / ac-003** — binding surface: new classes import and compose; raw computes stay raw; legacy bindings stay alive but warn with unchanged output. These gate the non-breaking contract of phase 02.
- **ac-004 / ac-005** — the regression lock, the core of this phase: the explicit raw→fit pipeline numerically reproduces the old bundled σ / D / spectrum.
- **ac-006 / ac-007 / ac-008** — molpy thin-inheritance: public outputs preserved, no math reimplemented in molpy, and the wheel-rebuild prerequisite is enforced so a stale wheel can't mask the repoint.
- **ac-009 / ac-010** — bench layer: reference-library equality holds through the new path, and the deliberate float32 two-budget floors are preserved (not tightened).
- **ac-011** — the standard full check + test gate.
