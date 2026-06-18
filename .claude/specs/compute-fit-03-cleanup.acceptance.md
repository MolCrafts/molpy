---
slug: compute-fit-03-cleanup
criteria:
  - id: ac-001
    summary: ConductivityResult/JacfResult expose only raw fields
    type: code
    evaluator_hint: ""
    pass_when: |
      ConductivityResult has no sigma/slope/fit_start/fit_end and JacfResult
      has no sigma/sigma_running (or both bundled types are deleted); the only
      remaining public fields are the raw observables {lag_times, msd} and
      {lag_times, jacf} respectively, verified by the new unit tests in
      molrs/src/compute/dielectric.rs and molrs/src/compute/jacf.rs.
    status: pending
  - id: ac-002
    summary: Every public compute entry point is a Compute struct
    type: code
    evaluator_hint: ""
    pass_when: |
      onsager_correlation, power_spectrum, ir_spectrum, raman_spectrum, and any
      remaining einstein_helfand_*/green_kubo_* free function no longer exist as
      free functions; OnsagerCorrelation and the spectra computes are structs
      implementing molrs::compute::Compute, re-exported from compute/mod.rs.
    status: pending
  - id: ac-003
    summary: Spectra computes return raw unwindowed ACFs
    type: code
    evaluator_hint: ""
    pass_when: |
      power/ir/raman computes return raw (unwindowed, untransformed) ACF output;
      window_and_fft, acf_to_spectrum, and acf_to_intensities are removed from
      compute/spectra/mod.rs and the baked-in windowed output is gone, asserted
      by the new spectra unit tests.
    status: pending
  - id: ac-004
    summary: Deprecated PyO3 bindings and transition shims removed
    type: code
    evaluator_hint: ""
    pass_when: |
      molrs-python/src/{dielectric.rs,transport.rs} contain no bindings calling
      removed free functions or removed derived fields; register_dielectric/
      register_transport in lib.rs reference only surviving functions; the
      transition shims are gone from molrs-python/python/molrs/{dielectric,transport}.py
      and the molpy transport/dielectric wrappers.
    status: pending
  - id: ac-005
    summary: Grep-clean — zero references to removed symbols
    type: runtime
    evaluator_hint: "ripgrep over molrs/ molrs-python/ molpy/ benches/"
    pass_when: |
      grep across molrs, molrs-python, molpy, and benches finds zero references
      to sigma/slope/fit_start/fit_end on the old result types, to
      einstein_helfand_conductivity/green_kubo_conductivity/einstein_helfand_spectrum/
      green_kubo_spectrum/onsager_correlation/power_spectrum/ir_spectrum/raman_spectrum
      as free functions, or to window_and_fft/windowed_acf_spectrum/acf_to_spectrum/sigma_running.
    status: pending
  - id: ac-006
    summary: Full Rust check + test suite green
    type: runtime
    evaluator_hint: ""
    pass_when: |
      cargo test --all-features passes, cargo clippy --all-targets --all-features
      -- -D warnings is clean, and cargo fmt --all --check is clean.
    status: pending
  - id: ac-007
    summary: molpy + freud-parity bench suites green on rebuilt wheel
    type: runtime
    evaluator_hint: ""
    pass_when: |
      the maturin wheel is rebuilt and installed, the molpy test suite passes
      against it, and the freud-parity bench suite passes with its parity floors
      intact (no regression).
    status: pending
  - id: ac-008
    summary: Transport/dielectric physics unchanged after cleanup
    type: scientific
    evaluator_hint: "marker: transport,dielectric regression"
    pass_when: |
      EinsteinConductivity+LinearFit reproduces Nernst-Einstein within the prior
      ≤0.13 ensemble tolerance; GreenKuboConductivity+RunningIntegral reproduces
      the previous sigma/sigma_running values; EH/GK dielectric spectra recover
      the Neumann static limit and Debye equivalence; power-spectrum sine peak is
      ~333.56 cm⁻¹ (±20). All within prior tolerances vs. pre-cleanup outputs.
    status: pending
  - id: ac-009
    summary: Breaking SemVer bump recorded, not published
    type: docs
    evaluator_hint: ""
    pass_when: |
      molcrafts-molrs version is bumped to the next breaking SemVer in Cargo.toml
      with a note that downstream pins (molpack, molpy exact-pin) must be updated;
      no publish/tag is performed.
    status: pending
---

# Acceptance criteria

- **ac-001 / ac-002 / ac-003** (code) — the structural core of the BREAKING change: raw structs carry only raw observables, every entry point is an OOP `Compute`, and spectra return raw ACFs. Verified by the new RED-first unit tests plus type/signature inspection.
- **ac-004** (code) — the binding/shim surface is purged of the deprecated paths added for the 02 transition.
- **ac-005** (runtime) — the grep-clean gate; the single strongest anti-drift signal that no caller still reaches a removed field/function.
- **ac-006 / ac-007** (runtime) — the full Rust gate and the cross-language (wheel + molpy + freud-parity bench) gate.
- **ac-008** (scientific) — physics-unchanged regression: separation must not perturb any numeric output; tolerances are the pre-existing ones.
- **ac-009** (docs) — the breaking version bump is recorded (requirement only; no publish).
