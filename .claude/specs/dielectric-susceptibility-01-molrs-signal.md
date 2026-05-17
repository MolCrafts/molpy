---
title: dielectric-susceptibility-01-molrs-signal — molrs signal processing primitives
status: code-complete
created: 2026-05-13
depends_on: []
---

# dielectric-susceptibility-01-molrs-signal

## Summary

Implement three single-responsibility signal processing functions in a new `molrs-signal` Rust crate, exposed to Python via a `molrs.signal` submodule. Each function does exactly one thing:

- `acf_fft` — raw FFT-based autocorrelation via Wiener-Khinchin theorem, returns un-normalized ACF
- `apply_window` — apply Hann or Blackman window to array along specified axis
- `frequency_grid` — generate angular frequency array [0, ω_Nyq] with correct spacing

All computation is in Rust. Python gets thin pyo3 bindings only. These are pure, stateless, input-immutable functions operating on NumPy arrays — they serve as building blocks for dielectric susceptibility computation in subsequent sub-specs.

## Domain basis

**ACF via Wiener-Khinchin theorem**: For a stationary process x(t), the autocorrelation R(τ) = ⟨x(t)x(t+τ)⟩ is the inverse Fourier transform of the power spectral density S(ω) = |F{x}(ω)|². The un-normalized ACF is: R_raw(τ) = F⁻¹(|F{x}|²). Normalization is the caller's responsibility.

**Hann window**: w[n] = 0.5 * (1 − cos(2πn/(N−1))), n = 0,…,N−1. Ref: Harris, Proc. IEEE 66(1):51-83, 1978.

**Blackman window**: w[n] = 0.42 − 0.5cos(2πn/(N−1)) + 0.08cos(4πn/(N−1)). Ref: Blackman & Tukey, The Measurement of Power Spectra, Dover, 1958.

**Nyquist frequency**: ω_Nyq = π/Δt. Frequency grid: ω_k = 2πk/(N_fft * Δt), k = 0,…,⌊N_fft/2⌋.

## Design

### Rust API (`molrs-signal` crate)

```rust
pub fn acf_fft(data: &Array1<f64>, max_lag: usize) -> Result<Array1<f64>, SignalError>;
pub fn apply_window(data: &ArrayD<f64>, window: WindowType, axis: usize) -> Result<ArrayD<f64>, SignalError>;
pub fn frequency_grid(n_fft: usize, dt: f64) -> Array1<f64>;

pub enum WindowType { Hann, Blackman }
```

- `acf_fft`: zero-pad to next power of 2 ≥ 2*max_lag, forward FFT, |X|², inverse FFT, extract first max_lag+1 entries. Returns raw un-normalized ACF.
- `apply_window`: create 1-D window array, broadcast across other dims, elementwise multiply. Returns new array.
- `frequency_grid`: returns `[0, Δω, 2Δω, …, ω_Nyq]` with Δω = 2π/(N_fft * Δt), length ⌊N_fft/2⌋+1.

### Python bindings

```python
molrs.signal.acf_fft(data: np.ndarray, max_lag: int) -> np.ndarray
molrs.signal.apply_window(data: np.ndarray, window_type: str, axis: int = 0) -> np.ndarray
molrs.signal.frequency_grid(n_fft: int, dt: float) -> np.ndarray
```

### Key constraints

- `acf_fft` returns **un-normalized** ACF. Caller normalizes.
- All functions are **pure**: no state, no mutation, no I/O.
- Input: NumPy arrays only (not molpy Frame/Trajectory).

## Files

- `molrs/molrs-signal/Cargo.toml` (new)
- `molrs/molrs-signal/src/lib.rs` (new)
- `molrs/molrs-signal/src/acf.rs` (new)
- `molrs/molrs-signal/src/window.rs` (new)
- `molrs/molrs-signal/src/grid.rs` (new)
- `molrs/molrs-python/src/signal.rs` (new)
- `molrs/molrs-python/tests/test_signal.py` (new)
- `molrs/Cargo.toml` (modify — add workspace member)
- `molrs/molrs/Cargo.toml` (modify — optional dep + feature)
- `molrs/molrs/src/lib.rs` (modify — conditional re-export)
- `molrs/molrs-python/Cargo.toml` (modify — add dep)
- `molrs/molrs-python/src/lib.rs` (modify — register submodule)
- `molrs/molrs-python/python/molrs/__init__.py` (modify)
- `molrs/molrs-python/python/molrs/molrs.pyi` (modify)

## Tasks

- [x] Write failing Rust tests for `acf_fft` in `molrs-signal/src/acf.rs`
- [x] Implement `acf_fft` in `molrs-signal/src/acf.rs`
- [x] Write failing Rust tests for `apply_window` in `molrs-signal/src/window.rs`
- [x] Implement `apply_window` with `WindowType` enum in `molrs-signal/src/window.rs`
- [x] Write failing Rust tests for `frequency_grid` in `molrs-signal/src/grid.rs`
- [x] Implement `frequency_grid` in `molrs-signal/src/grid.rs`
- [x] Write failing Python tests for `molrs.signal` in `molrs-python/tests/test_signal.py`
- [x] Add PyO3 bindings in `molrs-python/src/signal.rs`
- [x] Wire signal module into workspace (Cargo.toml chain, lib.rs registrations, Python imports)
- [x] Run full check + test suite (`cargo test --all-features -p molrs-signal && pytest molrs-python/tests/test_signal.py -v`)

## Testing strategy

- **Rust unit tests (cfg(test))**: ACF of constant signal; ACF of white noise (peak at lag 0); ACF of sine wave (oscillatory decay); ACF at max_lag=0; single-element input; max_lag > data length → error. Hann/Blackman window on 1D/2D arrays matches analytical formula; window does not mutate input. Frequency grid: correct length/spacing/endpoints; n_fft=2 minimal case.
- **Python integration tests**: Import `molrs.signal` success; `acf_fft` returns correct shape; `apply_window` with "hann"/"blackman" maps correctly, unknown string → ValueError; `frequency_grid` returns float64 with correct spacing; all functions do not mutate input; round-trip: FFT-ACF vs numpy reference.

## Out of scope

- Normalization or truncation of ACF (caller's responsibility)
- Window types beyond Hann and Blackman
- Multi-dimensional FFT or convolution
- Performance optimization (SIMD, etc.)
- Any Python-side compute logic
