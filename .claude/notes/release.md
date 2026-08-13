# Release

1. **molrs first** — master + tag `vX.Y.Z` + Publish green (PyPI must include Pyodide wheel if browsers matter).
2. Bump molpy to the same **major.minor**, pin `molcrafts-molrs>=X.Y.0,<X.(Y+1)`.
3. Tag molpy `vX.Y.Z` → Release workflow (trusted publishing).

No publish helper scripts; workflows only.

## v0.13.1 (2026-08-13)

Tracks molrs 0.13.1 (`>=0.13.1,<0.14`).

- `Frame.meta` is dict-like (molrs `FrameMeta`): `frame.meta["timestep"] = 0`.
- `mp.io.write_smarts(mol, atom)` — local environment SMARTS (molrs io).
- `UnitSystem`: `k_B`, `openmm` preset, `factor(source, target)`.
