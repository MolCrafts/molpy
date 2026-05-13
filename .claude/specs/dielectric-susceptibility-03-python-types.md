---
title: dielectric-susceptibility-03-python-types — Python data containers
status: done
created: 2026-05-13
depends_on: [dielectric-susceptibility-01-molrs-signal, dielectric-susceptibility-02-molrs-dielectric]
---

# dielectric-susceptibility-03-python-types

## Summary

Add frozen dataclasses to `src/molpy/compute/result.py` for the dielectric susceptibility computation chain. These are PURE DATA CONTAINERS — zero computation logic. All computation is in molrs.

New types:
- `ACFResult(TimeSeriesResult)` — holds ACF array from molrs.signal.acf_fft
- `SpectralResult(TimeSeriesResult)` — holds frequency-domain spectrum
- `DielectricResult(TimeSeriesResult)` — holds single-route dielectric spectrum
- `DielectricSusceptibilityResult(Result)` — aggregates multiple DielectricResult instances

Follows existing `Result` → `TimeSeriesResult` → `MCDResult`/`PMSDResult` pattern.

## Design

### Class hierarchy

```
Result (existing)
  └─ TimeSeriesResult (existing, adds time: NDArray)
       ├─ MCDResult (existing)
       ├─ PMSDResult (existing)
       ├─ ACFResult (NEW, frozen)
       ├─ SpectralResult (NEW, frozen)
       └─ DielectricResult (NEW, frozen)
  └─ DielectricSusceptibilityResult (NEW, frozen)
```

### ACFResult (frozen dataclass)

Extends `TimeSeriesResult`. Fields:
- `acf: NDArray[np.float64]` — autocorrelation values at each lag (n_lags,)
- `n_lags: int` — number of time lags

### SpectralResult (frozen dataclass)

Extends `TimeSeriesResult`. Fields:
- `frequency: NDArray[np.float64]` — angular frequency grid (THz)
- `spectrum: NDArray[np.float64]` — spectral density

### DielectricResult (frozen dataclass)

Extends `TimeSeriesResult`. Fields:
- `frequency: NDArray[np.float64]` — angular frequencies (THz)
- `epsilon_real: NDArray[np.float64]` — ε'(ω)
- `epsilon_imag: NDArray[np.float64]` — ε''(ω)
- `epsilon_static: float` — ε(0)
- `epsilon_inf: float` — ε_∞
- `route: str` — "einstein-helfand" or "green-kubo"
- `component: str` — "full", "water", "ion"
- `conductivity: NDArray[np.float64] | None` — σ(ω), GK route only

### DielectricSusceptibilityResult (frozen dataclass)

Extends `Result`. Fields:
- `results: dict[str, DielectricResult]` — keyed by e.g. "EH-full", "GK-water"
- `metadata: dict[str, Any]` — trajectory parameters (dt, temperature, volume, …)

Fields `results` values use `to_dict()` recursively in serialization.

### Key constraints

- All classes `@dataclass(frozen=True)`
- NO computation methods — only `to_dict()` serialization
- All NDArray fields typed as `np.float64`
- Optional fields use `None` default

## Files

- `src/molpy/compute/result.py` (modify) — add 4 dataclasses
- `src/molpy/compute/__init__.py` (modify) — export new classes
- `tests/test_compute/test_result.py` (new) — tests

## Tasks

- [x] Write failing tests for `ACFResult` and `SpectralResult` construction, field types, frozen immutability, defaults
- [x] Implement `ACFResult` and `SpectralResult` in `src/molpy/compute/result.py`
- [x] Write failing tests for `DielectricResult` construction, field types, frozen immutability, optional field defaults
- [x] Implement `DielectricResult` in `src/molpy/compute/result.py`
- [x] Write failing tests for `DielectricSusceptibilityResult` construction, nested `to_dict()`, frozen immutability
- [x] Implement `DielectricSusceptibilityResult` with `to_dict()` override for nested serialization
- [x] Export all 4 classes from `src/molpy/compute/__init__.py`
- [x] Run full check + test suite

## Testing strategy

- **Happy path**: construct each dataclass with valid fields, verify field values and types
- **Defaults**: optional fields (`conductivity`) default to `None`
- **Frozen immutability**: attribute assignment raises `FrozenInstanceError`
- **Inheritance**: `isinstance(result, TimeSeriesResult)` and `isinstance(result, Result)` pass
- **Serialization**: `to_dict()` includes all fields; nested `DielectricResult` in `DielectricSusceptibilityResult.results` recursively serialized

## Out of scope

- Computation logic — all in molrs
- Conversion to/from JSON, HDF5, etc. beyond `to_dict()`
- Validation of physical plausibility (negative ε, NaN, etc.)
