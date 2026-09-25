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

## v0.14.0 (2026-09-20)

Tracks molrs 0.14.0 (`>=0.14.0,<0.15`). Shipped on PyPI with molrs 0.14.0.

- Pin and runtime check on the 0.14 minor line.
- No public `molpy.Record`. Scientific-record I/O is `molpy.io.mrec.write_frame` / `write_system` / `write_trajectory`; schema checks bind `molrs::io::mrec::schema`.
- Identity columns are `uint64` (`molrs.types.Idx`); numpy widths are preserved.

## v0.15.0 (untagged — pre-0.15, same line as molrs)

molpy `0.15.0` and molrs `0.15.0` are the same unreleased line. Neither tag
exists. The pin is `molcrafts-molrs>=0.15.0,<0.16`, resolved by
`[tool.uv.sources]` path `../molrs/molrs-python` until molrs 0.15.0 is on PyPI.
Drop that path override in the change that tags `v0.15.0` (and the README
"until molrs 0.15.0 is published" install note plus the matching section of
`docs/getting-started/migration-0-15.md` with it).

User-facing migration guide: `docs/getting-started/migration-0-15.md` (in the
Tutorials → Get Running nav). Breaking changes it covers:

- Rigid-body verbs are native `translate` / `rotate` / `scale`, all returning
  self; `move(delta, entity_type=...)` → `translate(delta)`; scalar
  `scale(s)` → per-axis `scale([sx, sy, sz], about=None)`;
  `Atomistic.align` / `CoarseGrain.align` removed (use `rotate` or a
  `LineOrienter` / `TangOrienter`).
- Placement is opt-in: `PolymerBuilder(..., placer=None)` places nothing;
  `ResiduePlacer` → native `TracePlacer` (parent-relative BFS; ring-closing
  bond formed but not placed; `with_trace(Trace)` for path/ring, branched
  raises). `Trace` / `Placer` / orienters exported from `molpy.builder`.
- `PolymerBuilder.build` takes a `ResidueTopology` (string → `TypeError`);
  `CGSmiles*IR` residue types → `ResidueTopology` / `ResidueNode(label, *, id)`
  / `ResidueBond`; `linear_topology` / `ring_topology(label, n)` /
  `star_topology`. `AmberPolymerBuilder` still takes a CGSmiles string or a
  linear topology.
- `MonomerLibrary.expand` → `Expansion(world, pairing)`.
- `SiteMap` is the native class (atom views or int handles in, int handles
  out); `fields.SITE` / `fields.Q0` are native keys.
- `IonicConductivity`, `DielectricSusceptibility`, `ConductivityResult`,
  `DielectricSusceptibilityResult` and `molpy.compute.dielectric` removed;
  compose `EinsteinConductivity` → `LinearFit` → prefactor. Units trap:
  `IonicConductivity` took `dt` in ps, the composed route and the pmsd
  prefactor are fs.
- `io.emit`: `EMITTERS` / `emit` / `register` → `emitters.names()` /
  `.emit()` / `.register()`.
- moltemplate: `emit_python` → `PythonScriptEmitter(base_dir=...).emit(doc,
  dest)`; free `build_forcefield` / `build_system` →
  `MolTemplateBuilder(doc, base_dir=...).build_forcefield()` /
  `.build_system(ff=None, *, auto_topology=True)`.
