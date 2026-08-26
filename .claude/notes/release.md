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

## v0.13.2 (2026-08-19)

Tracks molrs 0.13.2 (`>=0.13.2,<0.14`).

## v0.14.0 (untagged — wait for molrs v0.14.0 on PyPI before tagging)

Tracks molrs 0.14.0 (`>=0.14.0,<0.15`).

- Pin and runtime check on the 0.14 minor line.
- Public record type is `molpy.Record` (molrs identity re-export; was `MolRec`).
- Identity columns are `uint64` (`molrs.types.Idx`); numpy widths are preserved.
