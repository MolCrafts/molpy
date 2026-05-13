---
title: dielectric-susceptibility-02-molrs-dielectric — molrs dielectric computation
status: approved
created: 2026-05-13
depends_on: [dielectric-susceptibility-01-molrs-signal]
---

# dielectric-susceptibility-02-molrs-dielectric

## Summary

Add a `dielectric` submodule to `molrs-compute` exposing six single-responsibility functions for dielectric response computation from MD trajectory data. Each function does exactly one thing:

- `compute_dipole_moment(charges, positions)` → M = Σ q_i * r_i, returns (3,) vector
- `compute_current_density(dipole_moments, dt, volume)` → J = ΔM/(V*Δt), returns (n_frames, 3), first row NaN
- `static_dielectric_constant(dipole_moments, volume, temperature, epsilon_inf)` → ε(0) via Neumann fluctuation formula
- `einstein_helfand_spectrum(dipole_moments, dt, volume, temperature, epsilon_inf, max_correlation_time, window_type)` → full EH route: composes acf_fft → apply_window → frequency_grid → FT → ε*(ω)
- `green_kubo_spectrum(current_density, dt, volume, temperature, epsilon_inf, max_correlation_time, window_type)` → full GK route: acf_fft → apply_window → frequency_grid → σ(ω) → ε*(ω)
- `decompose_current(per_particle_current, water_mask)` → split J into (J_water, J_ion)

All computation in Rust. Python gets thin pyo3 bindings. Functions compose molrs.signal primitives (acf_fft, apply_window, frequency_grid) — do NOT reimplement them.

## Domain basis

**Dipole moment**: M = Σ_i q_i * r_i. Units: e·Å.

**Current density**: J(t) = (M(t+Δt) − M(t)) / (V * Δt). First row NaN. Units: e/(Å²·ps).

**Neumann fluctuation formula** (Neumann, Mol. Phys. 50(4), 841-858, 1983):

ε(0) = ε_∞ + (4π/3) * κ * (⟨M²⟩ − |⟨M⟩|²) / (V * k_B * T)

where κ = 332.0637 kcal·Å·mol⁻¹·e⁻² (Coulomb constant), k_B = 1.987204e-3 kcal/(mol·K).

**Einstein-Helfand route**: ε*(ω) − ε_∞ = (4π/3) * κ / (V * k_B * T) * FT[w(τ) * C_M(τ)]

**Green-Kubo route**: σ(ω) = (1/3V k_B T) * FT[w(τ) * C_J(τ)], then ε*(ω) = ε_∞ + iσ(ω)/(ε_0 ω)

## Design

### Rust API (`molrs-compute::dielectric`)

```rust
pub fn compute_dipole_moment(charges: &Array1<f64>, positions: &Array2<f64>) -> Result<Array1<f64>, ComputeError>;
pub fn compute_current_density(dipole_moments: &Array2<f64>, dt: f64, volume: f64) -> Result<Array2<f64>, ComputeError>;
pub fn static_dielectric_constant(dipole_moments: &Array2<f64>, volume: f64, temperature: f64, epsilon_inf: f64) -> Result<f64, ComputeError>;
pub fn einstein_helfand_spectrum(dipole_moments: &Array2<f64>, dt: f64, volume: f64, temperature: f64, epsilon_inf: f64, max_correlation_time: usize, window_type: &str) -> Result<DielectricSpectrum, ComputeError>;
pub fn green_kubo_spectrum(current_density: &Array2<f64>, dt: f64, volume: f64, temperature: f64, epsilon_inf: f64, max_correlation_time: usize, window_type: &str) -> Result<DielectricSpectrum, ComputeError>;
pub fn decompose_current(per_particle_current: &Array3<f64>, water_mask: &Array1<bool>) -> Result<(Array2<f64>, Array2<f64>), ComputeError>;

pub struct DielectricSpectrum {
    pub frequencies: Array1<f64>,
    pub epsilon_real: Array1<f64>,
    pub epsilon_imag: Array1<f64>,
    pub n_frames: usize,
    pub n_correlation_steps: usize,
}
```

### Python bindings

```python
molrs.dielectric.compute_dipole_moment(charges, positions) -> np.ndarray  # (3,)
molrs.dielectric.compute_current_density(dipole_moments, dt, volume) -> np.ndarray  # (n_frames, 3)
molrs.dielectric.static_dielectric_constant(dipole_moments, volume, temperature, epsilon_inf) -> float
molrs.dielectric.einstein_helfand_spectrum(dipole_moments, dt, volume, temperature, epsilon_inf, max_correlation_time, window_type) -> DielectricSpectrum
molrs.dielectric.green_kubo_spectrum(current_density, dt, volume, temperature, epsilon_inf, max_correlation_time, window_type) -> DielectricSpectrum
molrs.dielectric.decompose_current(per_particle_current, water_mask) -> (np.ndarray, np.ndarray)
```

### Composition pattern

`einstein_helfand_spectrum` and `green_kubo_spectrum` compose molrs.signal primitives:
1. Call `molrs.signal.acf_fft()` for autocorrelation
2. Call `molrs.signal.apply_window()` for windowing
3. Call `molrs.signal.frequency_grid()` for frequency axis
4. Compute Fourier integral + normalization internally

They do NOT reimplement FFT or window generation.

## Files

- `molrs/molrs-compute/src/dielectric.rs` (new)
- `molrs/molrs-compute/src/lib.rs` (modify — add `pub mod dielectric`)
- `molrs/molrs-compute/Cargo.toml` (modify — add `num-complex` dep)
- `molrs/molrs-compute/tests/test_dielectric.rs` (new)
- `molrs/molrs-python/src/dielectric.rs` (new)
- `molrs/molrs-python/src/lib.rs` (modify — register submodule + 6 functions)
- `molrs/molrs-python/tests/test_dielectric.py` (new)

## Tasks

- [ ] Create empty `dielectric.rs` module stubs returning `todo!()`; register `pub mod dielectric` in lib.rs; add `num-complex` dep
- [ ] Write failing tests for `compute_dipole_moment` and `compute_current_density` in `test_dielectric.rs`
- [ ] Implement `compute_dipole_moment` and `compute_current_density` in `dielectric.rs`
- [ ] Write failing tests for `static_dielectric_constant` in `test_dielectric.rs`
- [ ] Implement `static_dielectric_constant` in `dielectric.rs`
- [ ] Write failing tests for `einstein_helfand_spectrum` and `green_kubo_spectrum` in `test_dielectric.rs`
- [ ] Implement `einstein_helfand_spectrum` and `green_kubo_spectrum` in `dielectric.rs`
- [ ] Write failing tests for `decompose_current` in `test_dielectric.rs`
- [ ] Implement `decompose_current` in `dielectric.rs`
- [ ] Create Python bindings in `molrs-python/src/dielectric.rs`; register in lib.rs; write Python tests
- [ ] Run `cargo test --all-features -p molrs-compute && pytest molrs-python/tests/test_dielectric.py -v`

## Testing strategy

- **Shape validation**: each function rejects dimension mismatches and non-positive scalars with `ComputeError`
- **Known-value**: two opposite charges at (±d,0,0) → M = (2d*e, 0, 0); constant dipole → J=[NaN,0,0,…]
- **Static dielectric sanity**: zero fluctuation → ε(0) = ε_inf
- **Spectrum convergence**: EH at ω→0 matches `static_dielectric_constant` within 5%
- **Decomposition**: J_water + J_ion = J_total to 1e-12
- **Immutability**: all inputs unchanged after each function call
- **Python parity**: each `molrs.dielectric.*` call matches Rust counterpart on identical inputs

## Out of scope

- Signal primitives (acf_fft, apply_window, frequency_grid) — sub-spec 01
- Cross-correlation between species (multi-component decomposition beyond water/ion binary split)
- PBC unwrapping (caller must unwrap before passing positions)
- Integration with molpy Block/Frame/Trajectory (sub-spec 04)
