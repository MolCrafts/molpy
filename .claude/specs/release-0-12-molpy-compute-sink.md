---
slug: release-0-12-molpy-compute-sink
status: done
created: 2026-08-04
closed: 2026-08-04
---

# release-0-12-molpy-compute-sink — compute = molrs re-export only

## Summary

molpy.compute must not invent recipe classes or free-function physics. Every
public analysis operator is the **same type** as the corresponding
`molrs.compute.*` Compute (identity re-export or zero-logic pass-through like
current `MSD` / `VanHove` / `HBonds` when a molpy `Compute` base is required for
registration).

## Design

- **Source of truth:** molrs `Compute` trait + exported classes
  (`GreenKuboConductivity`, `EinsteinConductivity`, `MSD`, `VACF`, …).
- **Forbidden:** molpy-side `Σ q v` / SI prefactors / second ACF implementation /
  free-function flux helpers / parallel class hierarchy (`JACF`, `PMSDCompute`
  as distinct types).
- **Allowed:** identity aliases for transitional imports
  (`JACF = GreenKuboConductivity`) until docs/benchmarks switch; no new logic.
- Fat modules (`dielectric.py` IonicConductivity, `onsager.py` unwrap loops)
  must be reduced to the same rule in follow-up tasks — same PR preference.

## Tasks

- [x] Delete molrs `flux` free-function experiment
- [x] JACF / PMSDCompute are identity re-exports of molrs Computes
- [x] Dielectric / Onsager / Persist fat shells → molrs-only re-exports
- [x] Benchmarks use molrs array API (`GreenKuboConductivity`, …)
- [x] Tests assert identity for JACF/PMSD/Onsager/Persist/Dielectric/Einstein/GreenKubo
- [x] Docs (`docs/compute/transport.md` etc.) teach compose API (not recipe classes)

## Out of scope

- Atomiverse
- Inventing new molrs free functions for assembly
