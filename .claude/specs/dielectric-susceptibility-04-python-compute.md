---
title: dielectric-susceptibility-04-python-compute — Python Compute glue layer
status: approved
created: 2026-05-13
depends_on:
  - dielectric-susceptibility-01-molrs-signal
  - dielectric-susceptibility-02-molrs-dielectric
  - dielectric-susceptibility-03-python-types
---

# dielectric-susceptibility-04-python-compute

## Summary

Add three single-responsibility Compute classes to `src/molpy/compute/dielectric.py` that bridge molpy Trajectory to molrs computational functions. These are THIN GLUE LAYERS — zero NumPy physics computation in Python. All computation is delegated to molrs.

### Compute classes

1. **`ACFAnalyzer(Compute[Trajectory, ACFResult])`** — extract per-atom data from Trajectory, unwrap coordinates, call `molrs.signal.acf_fft()`, return ACF. Single responsibility: autocorrelation computation.

2. **`SpectralAnalyzer(Compute[ACFResult, SpectralResult])`** — apply window to ACF via `molrs.signal.apply_window()`, generate frequency grid via `molrs.signal.frequency_grid()`, Fourier transform ACF→spectrum. Single responsibility: time→frequency conversion.

3. **`DielectricSusceptibility(Compute[Trajectory, DielectricSusceptibilityResult])`** — compute dipole moment and current density time series from Trajectory, call `molrs.dielectric.einstein_helfand_spectrum()`, `molrs.dielectric.green_kubo_spectrum()`, `molrs.dielectric.static_dielectric_constant()`, decompose by component. Assembles results into `DielectricSusceptibilityResult`. Single responsibility: dielectric physics composition.

### Design note: Trajectory

`Trajectory` is currently pure Python. A future infrastructure spec will move `Trajectory` to inherit from `molrs.Trajectory` (following the existing `Box` inherits `molrs.Box` / `Block` composes `molrs.Block` pattern). The Compute classes in this spec accept `Trajectory` and are agnostic to whether it's Python-backed or molrs-backed — they extract data via the public API (`frame["atoms"]["x"]`, `frame.box`, etc.).

## Design

### ACFAnalyzer

```python
class ACFAnalyzer(Compute[Trajectory, ACFResult]):
    def __init__(self, columns: list[str], max_lag: int, unwrap: bool = True, **config_kwargs):
        ...
```

- Extract specified columns (e.g., `["x","y","z"]`) from `frame["atoms"]` for every frame
- If `unwrap=True`, unwrap coordinates via `Box.diff_dr` (requires `frame.box`)
- Build (n_frames, n_atoms, n_dim) array
- Call `molrs.signal.acf_fft()` → raw ACF
- Normalize ACF (divide by zero-lag value so ACF[0] = 1.0)
- Package into `ACFResult(time=lag_times, acf=normalized_acf, n_lags=max_lag)`

Input validation:
- Frames must contain all required columns → `ValueError` with column name
- If `unwrap=True`, frames must have non-free `box` → `ValueError`
- At least 2 frames required → `ValueError`

### SpectralAnalyzer

```python
class SpectralAnalyzer(Compute[ACFResult, SpectralResult]):
    def __init__(self, dt: float, window_type: str = "hann", **config_kwargs):
        ...
```

- Apply window to ACF via `molrs.signal.apply_window(acf, window_type)`
- Generate frequency grid via `molrs.signal.frequency_grid(n_lags, dt)`
- Fourier transform windowed ACF to frequency domain (one-sided cosine/sine transform)
- Package into `SpectralResult(time=frequency, frequency=frequency, spectrum=spectrum)`

### DielectricSusceptibility

```python
class DielectricSusceptibility(Compute[Trajectory, DielectricSusceptibilityResult]):
    def __init__(self, dt: float, temperature: float,
                 max_correlation_time: float, epsilon_inf: float = 1.0,
                 window_type: str = "hann",
                 routes: list[str] = ["einstein-helfand", "green-kubo"],
                 components: list[str] = ["full"],
                 water_residue_names: tuple[str, ...] = ("SOL", "H2O", "WAT", "TIP3", "TIP4", "TIP5"),
                 **config_kwargs):
        ...
```

- Iterate Trajectory once: extract positions, velocities, charges, box volume per frame
- Unwrap coordinates via `Box.diff_dr`
- Compute dipole moment time series M(t) = Σ q_i * r_i (calls `molrs.dielectric.compute_dipole_moment` per frame or in batch)
- Compute current density time series J(t) (calls `molrs.dielectric.compute_current_density`)
- For each enabled route:
  - "einstein-helfand": `molrs.dielectric.einstein_helfand_spectrum(M_series, ...)` → DielectricResult
  - "green-kubo": `molrs.dielectric.green_kubo_spectrum(J_series, ...)` → DielectricResult
- Compute `molrs.dielectric.static_dielectric_constant(M_series, ...)` → ε(0)
- For water/ion decomposition: `molrs.dielectric.decompose_current(J_per_atom, water_mask)` → separate spectra
- Assemble all results into `DielectricSusceptibilityResult(results={...}, metadata={...})`

Input validation:
- Same as ACFAnalyzer (columns, box, min frames)
- `dt > 0`, `temperature > 0` → `ValueError`

### Zero Python computation (CRITICAL)

The file `src/molpy/compute/dielectric.py` must contain:
- NO `np.fft` calls
- NO `np.sum` for physical quantities (reshaping/indexing is OK)
- NO manual formula implementation (ε = ε_inf + (4π/3) * κ * …)
- All physics computation via `molrs.signal.*` and `molrs.dielectric.*` calls

## Files

- `src/molpy/compute/dielectric.py` (new) — three Compute classes
- `tests/test_compute/test_dielectric.py` (new) — unit tests
- `src/molpy/compute/__init__.py` (modify) — exports

## Tasks

- [ ] Write failing tests for `ACFAnalyzer` — construction, column extraction, unwrap, molrs call delegation, ACFResult structure
- [ ] Implement `ACFAnalyzer` in `src/molpy/compute/dielectric.py`
- [ ] Write failing tests for `SpectralAnalyzer` — window application, frequency grid, FT, SpectralResult structure
- [ ] Implement `SpectralAnalyzer` in `src/molpy/compute/dielectric.py`
- [ ] Write failing tests for `DielectricSusceptibility` — dipole/current computation delegation, EH/GK route calls, static constant, decomposition, result assembly
- [ ] Implement `DielectricSusceptibility` in `src/molpy/compute/dielectric.py`
- [ ] Write immutability tests — all three classes do not mutate input Trajectory
- [ ] Verify zero Python physics computation — source review: no np.fft, no manual formula
- [ ] Export all 3 classes from `src/molpy/compute/__init__.py`
- [ ] Run full check + test suite

## Testing strategy

- **ACFAnalyzer**: mock `molrs.signal.acf_fft`, verify called with correct arguments; verify column extraction and coordinate unwrapping; missing column → ValueError; missing box → ValueError; too few frames → ValueError
- **SpectralAnalyzer**: mock `molrs.signal.apply_window` and `molrs.signal.frequency_grid`, verify calls; verify output SpectralResult shape
- **DielectricSusceptibility**: mock all `molrs.dielectric.*` functions; verify correct arguments for EH and GK routes; verify result dict keys match enabled routes and components; verify static constant computed
- **Immutability**: snapshot frame contents before/after, verify unchanged
- **Zero Python physics**: grep for `np.fft`, `np.sum` in dielectric.py — none found for physics computation

## Out of scope

- `Trajectory` molrs-backend migration (separate infrastructure spec)
- GROMACS file I/O (input must be molpy Trajectory)
- Plotting or visualization
- Integration with `Workflow` DAG (compatible but not required)
- Performance optimization or parallelization
