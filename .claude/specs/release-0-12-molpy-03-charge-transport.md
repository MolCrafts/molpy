---

> **SUPERSEDED** by `release-0-12-molpy-compute-sink`: do not implement science in molpy.

slug: release-0-12-molpy-03-charge-transport
status: cancelled
created: 2026-08-04
grilled: false
depends_on:
  - release-0-12-molpy-02-units-fs
---

# release-0-12-molpy-03-charge-transport — charge-weighted JACF/PMSD

## Summary

JACF and PMSD assemble collective current / polarization with real atomic charges \(q_a\), matching `IonicConductivity` and Green–Kubo physics. Monovalent ±1 hard-coding is deleted.

## Domain basis

\[
\mathbf{J}(t)=\sum_a q_a\mathbf{v}_a(t),\qquad
\mathbf{M}(t)=\sum_a q_a\mathbf{r}_a^{\mathrm{u}}(t)
\]

Missing `charge` column → **error**. Type masks only select which atoms enter the sum.

Refs: Green–Kubo ionic conductivity; MDAnalysis / molrs dielectric charge weighting.

## Design

- Generalize `JACF.__call__` current assembly using frame charges (pattern: `IonicConductivity`).
- Generalize `PMSDCompute` polarization the same way; keep molrs window MSD for lag series.
- Update docstrings; delete “unit charges ±1” as the definition.
- Tests: \(q=\pm2\) vs \(q=\pm1\); missing charge raises.

### Reuse decision

- `reuse` `IonicConductivity` charge extract + unwrap pattern
- `reuse` molrs `GreenKuboConductivity` + `MSD` kernels
- `generalize` JACF/PMSD assembly only
- `new` — none

## Files to create or modify

- `src/molpy/compute/jacf.py`
- `src/molpy/compute/pmsd.py`
- `tests/test_compute/test_jacf.py`
- `tests/test_compute/test_pmsd.py`
- `docs/compute/transport.md` (formula section)

## Tasks

- [ ] Write failing tests for charge-weighted J/M and missing-charge error
- [ ] Implement charge-weighted JACF current
- [ ] Implement charge-weighted PMSD polarization
- [ ] Update transport.md formulas and monovalent caveats (delete wrong formulas)
- [ ] Add regression `regressions/release-0-12-molpy-03-charge-transport.py`
- [ ] Run full check + test suite

## Testing strategy

Hard-coded synthetic frames with known charges/velocities. No third-party oracles.

## Out of scope

- Persist triclinic fix
- SpectralAnalyzer rename
