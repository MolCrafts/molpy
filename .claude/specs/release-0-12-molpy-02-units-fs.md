---

> **SUPERSEDED** by `release-0-12-molpy-compute-sink`: do not implement science in molpy.

slug: release-0-12-molpy-02-units-fs
status: cancelled
created: 2026-08-04
grilled: false
depends_on:
  - molrs release-0-12-04 (published or path-dep with fs kernels)
---

# release-0-12-molpy-02-units-fs — analysis time = fs

## Summary

Align all molpy transport/dielectric analysis APIs and SI prefactors with project unit **fs** (matching molrs 0.12 science decision and LAMMPS `real`). Remove “LAMMPS real uses ps” language.

## Domain basis

| Quantity | Unit |
|---|---|
| Analysis `dt` | **fs** |
| Velocity (LAMMPS real) | Å/fs |
| Spectra `dt_fs` | already correct — pattern |

SI conductivity: convert with \(10^{-15}\) s/fs, not \(10^{-12}\).

## Design

- Update `jacf.py`, `dielectric.py`, `pmsd.py`, `onsager.py`, `persist.py`, `result.py` docs and prefactors.
- Keep spectra `dt_fs` naming; prefer explicit unit in docs for bare `dt`.
- Do not invent dual unit APIs.

### Reuse decision

- `reuse` `spectra.py` `dt_fs` convention
- `reuse` `core/unit.py` real preset as documentation source
- `generalize` JACF/dielectric prefactors from ps → fs
- `new` — none

## Files to create or modify

- `src/molpy/compute/jacf.py`
- `src/molpy/compute/dielectric.py`
- `src/molpy/compute/pmsd.py`
- `src/molpy/compute/onsager.py`
- `src/molpy/compute/persist.py`
- `src/molpy/compute/result.py`
- related tests / docs strings in `docs/compute/*`

## Tasks

- [ ] Write failing unit tests for SI prefactor scaling with fs (hard-coded)
- [ ] Migrate JACF Green–Kubo prefactor to fs
- [ ] Migrate dielectric / IonicConductivity prefactors to fs
- [ ] Update PMSD/Onsager/Persist/result time unit docs
- [ ] Grep-gate docs/compute for “LAMMPS real” + ps contradictions
- [ ] Add regression `regressions/release-0-12-molpy-02-units-fs.md`
- [ ] Run full check + test suite

## Testing strategy

Synthetic conductivity with known integral; assert SI value with fs conversion. No live MD engines.

## Out of scope

- Charge weighting (03)
- Full README rewrite (05)
