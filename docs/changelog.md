# Changelog

## 0.4.1 - 2026-06-14

Maintenance release on top of `0.4.0`. Requires `molcrafts-molrs == 0.1.2`.

### Removed

- **`molpy.legacy` is gone.** The pure-NumPy `MSD` / `DisplacementCorrelation`
  operators (and the `molpy.legacy` submodule) were removed. Use the
  molrs-backed trunk in `molpy.compute` instead — `molpy.compute.MSD` and
  `molpy.compute.MCDCompute`.

### Fixed

- **AMBER prmtop reader.** Atom-connectivity index columns
  (`atomi`/`atomj`/`atomk`/`atoml`/`id` on bonds/angles/dihedrals and the atom
  `id`) are now emitted as unsigned `uint32` to match molrs's `UInt` index
  columns (`get_uint`); mask/sentinel columns that use `-1` stay signed.

### Internal

- Pinned `molcrafts-molrs == 0.1.2` (was `0.1.1`); kept the CI install comment in
  sync.
- Collapsed the three near-identical PDB-writer required-field tests into a
  single parametrized test.

## 0.4.0 - 2026-06-11

This release lands the **molrs Rust backend** as the foundation of `Frame` /
`Block` / `Box` / `compute`, and a five-stage **builder/reacter overhaul**
(dead-code consolidation, fix bond/react serialization moved into the io layer,
REACTER template scientific-correctness fixes, a consolidated public API with
executable docs, and a behavior-preserving build-loop performance pass).
A follow-up consolidation wave completes the molrs sink: topology perception,
trajectory storage, the force-field/potential model, and box/region geometry
now all execute in the Rust backend, a **fail-fast sweep** removes every known
silent-failure path, and a new **CL&P ionic-liquid force field** plus the
foundations of the **CL&Pol polarizable stack** land in the typifier/builder
layer.
Requires `molcrafts-molrs == 0.1.0`.

### Builder / Reacter

- **`BondReactTemplate.write()` / `write_map()` were removed.**
  `BondReactTemplate` is now a pure data object; all fix bond/react
  serialization lives in the io layer. Write a complete reactive system
  (data + ff + templates, with type IDs unified across all files) with
  `mp.io.write_lammps_bond_react_system(workdir, frame, ff, templates=...)`;
  write just the `.map` file with `mp.io.write_bond_react_map(template, stem)`.
  The single-template `write()` path produced template-local type IDs that
  were inconsistent with the system data file and has no replacement.
- **Importing a molpy subpackage no longer eagerly loads the rest.**
  Top-level submodules (`molpy.io`, `molpy.engine`, `molpy.adapter`, …) are
  now lazy (PEP 562): `import molpy.reacter` initializes only `core` (and
  `potential`). `mp.io.…` attribute access and `import molpy.io` behave as
  before. `molpy.builder` / `reacter` / `pack` / `compute` are reachable as
  lazy top-level attributes (`mp.builder.…`).
- **Builder/reacter terminology and API consolidation.**
  `polymer()` / `polymer_system()` are the documented one-call entry points;
  `PolymerBuilder` + `Connector` remain the step-by-step API. Agent-only Tool
  classes moved to `molpy.builder.polymer.tools` (out of the user `__all__`);
  `ReactionPresets` / `ReactionPresetSpec` are now public extension points.
  `ReactionPresetSpec.site_selector_*` → `anchor_selector_*`;
  `molpy.reacter.find_port_atom` → `find_port`. No deprecation shims
  (experimental stage); see the repo-root `CHANGELOG.md`.
- **REACTER template correctness.** `BondReactReacter` post templates now carry
  impropers (sp2 planarity terms survive `fix bond/react`), `InitiatorIDs` are
  deterministic and validated (exactly 2, never on the template boundary), edge
  atoms are checked for identical pre/post type and charge, total charge is
  checked for conservation (`CHARGE_CONSERVATION_TOL = 1e-6` e), and `run()` no
  longer mutates caller-owned `left` / `right` structures.
- **molrs is now a required dependency.** `molcrafts-molrs` moved from an
  optional extra into the core `dependencies`. The `molpy[molrs]` extra key was
  removed — installing molpy always installs molrs. `Frame`, `Block`, and `Box`
  are backed by (and inherit from) molrs types, and the `compute` operators
  forward directly into the Rust kernel. There is no pure-Python fallback.
- **The RDKit-backed compute node was removed.** `molpy.compute.rdkit`
  (`Generate3D` / `OptimizeGeometry` over `RDKitAdapter`) is gone.
  `molpy.compute.Generate3D` is now the molrs-backed trunk operator, taking an
  `Atomistic` graph and returning a fresh 3D structure. The RDKit adapter
  (`molpy.adapter.rdkit`) is retained as an
  **optional** external backend; `rdkit` remains an optional extra, not a
  required dependency.
- **`Frame` / `Block` are the canonical molrs types, not molpy subclasses.**
  `molpy.core.frame.Frame is molrs.Frame` and `Block is molrs.Block` (thin
  re-exports). The Python-side object-column overflow (`_objects`) is gone:
  columns are **numpy-only** (float / int / bool / str). Assigning an
  object / `None` / ragged column now raises `molrs.BlockDtypeError` at write
  time instead of being silently stored on the Python side. `frame.box` returns
  a `molrs.Box` (carrying `is_free` / `style` / `volume()`); molpy's richer box
  geometry stays available as `molpy.Box`, upgradable via `Box.from_box(frame.box)`.

### Typifiers & Force Fields

- **One public typifier class per force field.** The dual atom-only/atomistic
  hierarchy collapsed into a single full-pipeline class per force field:
  `OplsTypifier` and `GaffTypifier` (atom → pair → bond → angle → dihedral).
  The orchestrator base was renamed `ForceFieldAtomisticTypifier` →
  `ForceFieldTypifier`; per-FF atom typifiers are now private
  (`_OplsAtomTypifier` / `_GaffAtomTypifier`); the
  `Opls{Bond,Angle,Dihedral}Typifier` classes and the
  `GaffTypifier = GaffAtomisticTypifier` alias were removed. Behavior is
  unchanged.
- **New CL&P ionic-liquid force field.** `ClpTypifier(OplsTypifier)` types
  ionic liquids through the full pipeline from a new built-in `clp.xml`
  (imidazolium cation + BF4 / PF6 / NTf2 / dca anions), generated from the
  authoritative paduagroup CL&P `il.ff` with exact charges/LJ and hand-authored
  SMARTS for ring-position discrimination (CR/CW/NA). It is read through the
  existing OPLS reader; `oplsaa.xml` is untouched. `[C4C1im]+` sums to +1,
  each anion to −1.
- **CL&Pol polarizable-force-field foundations.**
  `VirtualSite` / `DrudeParticle` / `MasslessSite` are new data-only `Atom`
  subclasses carrying a persistent `vsite` marker (identity lives in the field,
  so it survives the molrs store; `Atom.is_virtual` reads it). The
  `VirtualSiteBuilder` ABC (copy → select → build sites → redistribute, no
  input mutation) ships with `DrudeBuilder` (the CL&Pol polarizer, driven by
  the bundled `alpha.ff` polarizability data) and `Tip4pBuilder` (rigid M-site).
- **CL&Pol scaleLJ SAPT epsilon scaling.** `molpy.core.ops.scale_lj` scales
  cross-fragment LJ epsilon by the SAPT-derived factor k_ij (matching
  paduagroup/clandpol scaleLJ), deep-copying the `ForceField` and touching only
  inter-fragment pair epsilon — sigma and charges are unchanged. Fragment data
  ships as `clpol_fragments.ff`; `FragmentScaling`, `compute_k_ij`, and
  `load_fragment_scaling_data` are public.
- Interim pure-Python **Thole and Tang–Toennies damping evaluators** were
  developed and validated against paduagroup/clandpol and LAMMPS
  `pair_thole` / `pair_coul_tt` during this cycle, but were superseded by the
  force-field collapse onto molrs before release — they are **not** part of the
  0.4.0 surface and will return as molrs-native kernels.

### molrs Consolidation

- **The force-field model now lives entirely in molrs; `molpy.potential` is a
  facade.** `molpy.core.forcefield` is a thin re-export of the native molrs
  `ForceField` / Style / Type hierarchy (plus the `AtomisticForcefield` alias
  and named specialized Style classes). The parallel Python kernels and energy
  math under `potential/` were deleted; `molpy.potential[.bond|.angle|…]`
  re-exports the molrs-backed Style classes and `Potentials` so users never
  import molrs. Energy/forces are evaluated via
  `ff.to_potentials(frame).calc_energy(frame)` / `.calc_forces(frame)`;
  `optimize.ForceFieldPotential` wraps this (the per-kernel
  `potential_wrappers` are gone). I/O formatters now dispatch by style/kernel
  name (no per-kernel `Type` classes), and `def_type` parameters are
  keyword-only.
- **Topology perception runs on the molrs graph kernel; the parallel igraph
  engine is deleted.** `get_topo` angle/dihedral perception delegates to
  `molrs.Atomistic.generate_topology`; `get_topo_neighbors` /
  `get_topo_distances` and `extract_subgraph` use the molrs BFS
  (`topo_distances`) and adjacency kernels. `core/topology.py` and the public
  `molpy.Topology` / `molpy.core.Topology` exports were removed — `get_topo`
  now **always returns an `Atomistic`** (no flags = plain copy), fixing a
  latent bug where the no-flags path could leak a raw graph object. igraph
  remains only inside the SMARTS typifier. Relation enumeration uses the
  authoritative molrs `relation_ids()` — the Python-side `_rel_handles` shadow
  and its handle-range probing heuristic are gone.
- **`Trajectory` is now a `molrs.Trajectory` subclass.** The eager container,
  lazy reading, and LAMMPS/XYZ trajectory parsing all live in molrs
  (`read_lammps_trajectory` / `read_xyz_trajectory`); molpy's duplicate readers
  and the mmap-index infrastructure were deleted. molpy keeps the split
  extensions (`SplitStrategy` / `TrajectorySplitter`), topology/slice/map
  conveniences, the XYZ writer, and the HDF5 path. `TimeIntervalStrategy` reads
  the native `.time` array (Python `frame.metadata` does not round-trip the
  molrs store).
- **`Compute` is a plain class.** `__init__` takes configuration, `__call__`
  takes inputs, `dump()` persists. The single-input `_compute` hook, the molexp
  `execute()` / `input_key` / `output_key` shim, and the `Compute[InT, OutT]`
  generic were removed, along with 17 dead `_compute` stubs across the
  molrs-backed operator wrappers.
- **Box and region geometry delegate to the Rust kernels.**
  `Box.make_fractional` / `make_absolute` / `isin` forward to the inherited
  molrs `to_frac` / `to_cart` / `isin`; `volume`, lengths/tilts, wrapping, and
  minimum-image differences use the molrs properties and `Box.wrap` /
  `Box.delta`. `SphereRegion` and `BoxRegion` / `Cube` point-membership tests
  run on `molrs.Sphere.contains` / `molrs.Cuboid.contains` — molpy keeps only
  the boolean-algebra / `MaskPredicate` layer. The Python-side `_is_free` flag
  is gone: free-box state derives from molrs `cell_defined`, so a non-periodic
  bounding box (`from_bounds`) now correctly reports a real cell with
  volume/lengths. `Atomistic.scale` and `align` use `molrs.scale` /
  `molrs.rotate` (the per-atom Rodrigues loop is gone). Ortho + triclinic
  parity with the previous NumPy paths was verified before each sink.
- **Canonical field registry comes from `molrs.fields`.** `molpy.core.fields`
  no longer defines a parallel `FieldSpec` / `FieldFormatter` set — it
  re-exports the single canonical registry (`molpy.core.fields.CHARGE` *is*
  `molrs.fields.CHARGE`), keeping only the FF-specific `ForceFieldFormatter`
  on top. This resolves the drifted triple-duplicated registry.

### Fail-Fast

- **Selectors raise on missing columns.** `AtomIndexSelector` /
  `ElementSelector` / `CoordinateRangeSelector` / `DistanceSelector` now raise
  `KeyError` naming the missing column instead of silently matching zero atoms
  on a typo'd field.
- **`ForceField.to_potentials()` no longer silently drops styles.**
  Legitimately-empty styles (no types) are skipped explicitly; any real failure
  (unknown kernel, a type missing required params) propagates.
- **Typifier inputs are validated.** A SMARTS pattern that fails to parse now
  raises instead of being warned and dropped (which silently un-typed that atom
  type); an invalid element symbol or atomic number raises `ValueError` instead
  of degrading to a match-anything wildcard (`*` stays reserved for the
  explicit no-element/no-number case).
- **The LAMMPS data reader keeps force-field coefficients.**
  Pair/Bond/Angle/Dihedral/Improper Coeffs sections are now stored on the
  metadata `ForceField` (positionally-keyed, style-arity aware) so read → write
  round-trips them; previously they were parsed and discarded. Malformed
  (non-numeric) coeff lines raise `ValueError` instead of being swallowed.
- The dead `molpy.op` package (unused geometry helpers) was deleted.

### Added

- `molpy.compute.NeighborList` — linked-cell neighbor search (molrs backend).
- `molpy.compute.RDF` — radial distribution function over one or more frames.
- `molpy.Box` inherits `molrs.Box`, so a molpy box is accepted by any molrs API
  with no conversion.
- molrs analyses exposed as molpy operators: `MSD`, `Cluster`, `ClusterCenters`,
  `CenterOfMass`, `GyrationTensor`, `InertiaTensor`, `RadiusOfGyration`, `Pca`,
  `KMeans`.
- `molpy.typifier.ClpTypifier` + built-in `clp.xml` — CL&P ionic-liquid force
  field (imidazolium + BF4/PF6/NTf2/dca) on top of the OPLS pipeline.
- `VirtualSite` / `DrudeParticle` / `MasslessSite` atoms, `Atom.is_virtual`,
  and `VirtualSiteBuilder` / `DrudeBuilder` / `Tip4pBuilder` (CL&Pol Drude
  polarizer + TIP4P M-site), with bundled `alpha.ff` polarizability data.
- `molpy.core.ops.scale_lj` (+ `FragmentScaling`, `compute_k_ij`,
  `load_fragment_scaling_data`, `clpol_fragments.ff`) — CL&Pol SAPT
  cross-fragment LJ epsilon scaling.
- `UnitSystem.register_preset(name, base_units, *, overwrite=False)` — register
  custom LAMMPS-style unit presets usable via `preset()`; the preset table is
  no longer a closed dict.

### Changed

- `compute.mcd` and `compute.pmsd` now compute minimum-image displacements via
  molrs `Box.delta(minimum_image=True)`; public signatures are unchanged.

### Migration

- Replace `pip install "molcrafts-molpy[molrs]"` with `pip install molcrafts-molpy`.
- If you imported `from molpy.compute.rdkit import Generate3D`, switch to
  `from molpy.compute import Generate3D` (molrs-backed, `Atomistic -> Atomistic`)
  or, for the RDKit adapter flow, `from molpy.adapter import Generate3D`.
- Build `Block` columns from numpy-representable data: replace
  `np.array([...], dtype=object)` string columns with native `np.array([...])`
  (numpy infers a `U` dtype). Sparse per-entity attributes can no longer be
  `None`-filled into a column — use a typed default or omit the column.
  `Atomistic.to_frame()` / `CoarseGrain.to_frame()` now drop columns that cannot
  be numpy-represented (e.g. a CG bead's ragged `atoms` mapping) rather than
  emitting object arrays.
- `molpy.Topology` / `molpy.core.Topology` no longer exist; `get_topo()` always
  returns an `Atomistic`. For k-hop graph queries use
  `get_topo_neighbors` / `get_topo_distances` (molrs BFS backed).
- Typifiers: replace `OplsAtomTypifier` / `GaffAtomTypifier` /
  `Opls{Bond,Angle,Dihedral}Typifier` / `GaffAtomisticTypifier` with
  `OplsTypifier` / `GaffTypifier`; a custom orchestrator should subclass
  `ForceFieldTypifier` (renamed from `ForceFieldAtomisticTypifier`).
- Potentials: the Python kernel classes under `molpy.potential` are gone —
  import the molrs-backed Style classes from the same paths and evaluate via
  `ff.to_potentials(frame).calc_energy(frame)`. Parameter names follow molrs
  (`k`, not `k0`); look up styles with `ff.get_style(category, name)`;
  `def_type` parameters are keyword-only.
- `Compute` subclasses: configuration goes to `__init__`, data to `__call__`.
  The `execute()` / `input_key` / `output_key` molexp shim and the `_compute`
  hook were removed.
- Code that relied on selectors returning an empty mask for a missing column,
  on unparseable SMARTS being skipped, or on `to_potentials()` ignoring broken
  styles must now handle the raised `KeyError` / `ValueError` (or fix the
  input — these were silent-failure bugs).

See [the molrs backend developer guide](developer/molrs-backend.md) for details.
