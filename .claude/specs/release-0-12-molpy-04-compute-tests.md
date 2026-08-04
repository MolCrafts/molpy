---

> **SUPERSEDED** by `release-0-12-molpy-compute-sink`: do not implement science in molpy.

slug: release-0-12-molpy-04-compute-tests
status: cancelled
created: 2026-08-04
grilled: false
---

# release-0-12-molpy-04-compute-tests — thin-shell tests for public compute

## Summary

Add mirrored unit tests for public compute modules that currently have zero coverage: `signal`, `hbond`, `van_hove`, `spectra`, `voronoi`. Pattern matches existing molrs wrapper tests.

## Domain basis

N/A (test surface). Science correctness of kernels owned by molrs; molpy tests assert subclass, parity, no mutation.

## Design

- Follow `tests/test_compute/test_molrs_wrappers.py` pattern.
- One module per source file (or clearly partitioned classes).
- Hard-coded tiny fixtures only.

### Reuse decision

- `reuse` `test_molrs_wrappers.py` checklist
- `reuse` production thin shells as-is
- `new` — none

## Files to create or modify

- `tests/test_compute/test_signal.py` (new)
- `tests/test_compute/test_hbond.py` (new)
- `tests/test_compute/test_van_hove.py` (new)
- `tests/test_compute/test_spectra.py` (new)
- `tests/test_compute/test_voronoi.py` (new)
- optionally extend `test_molrs_wrappers.py`

## Tasks

- [ ] Write TestSignal thin-shell tests
- [ ] Write TestHBonds thin-shell tests
- [ ] Write TestVanHove thin-shell tests
- [ ] Write TestSpectra thin-shell tests
- [ ] Write TestVoronoi thin-shell tests
- [ ] Add regression checklist `regressions/release-0-12-molpy-04-compute-tests.md`
- [ ] Run full check + test suite

## Testing strategy

Unit only under `tests/test_compute/`. No e2e trajectories from network.

## Out of scope

- Moving docs e2e out of tests/
- Numerical freud parity (belongs bm suite)
