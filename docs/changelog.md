# Changelog

Release notes for MolPy, newest first. MolPy and molrs share one version line
and release as a pair — every entry lists the `molcrafts-molrs` version it
requires. Tagged releases and installable artifacts live on
[GitHub Releases](https://github.com/MolCrafts/molpy/releases).

## 0.9.2 — 2026-07-23

Requires `molcrafts-molrs==0.9.2`.

### Fixed

- **Cloudflare Pages / Zensical docs build.** Preload `molrs` for mkdocstrings
  so `molpy.Frame` / `molpy.Block` re-export aliases resolve; document top-level
  symbols as `molpy.Frame` while user code keeps `import molpy as mp`.
- **Frame/Block import path** uses `molrs.frame` so static analysis can resolve
  the pure-Python rich layer without a broken alias chain.

## Unreleased


### Added

- **Compile-first polymer assembly and deferred finalization.** The selector's
  complete binding set is compiled into rooted local product motifs before the
  assembled graph is edited. Distinct motifs are typed once and cache scalar
  per-atom patches only; one batch reaction then builds the product.
  `Finalization.ATOMS`, `TOPOLOGY` (default), and `BONDED` split atom write-back,
  topology generation, and `ForceFieldParams` assignment so large MD systems
  can defer the latter two stages until export.
- **Carbon nanotube topology builder.** `CarbonTubeBuilder.build(n, m, ...)`
  covers zigzag, armchair, and general chiral tubes, open or axially periodic.
  Exact integer lattice quotienting closes seams without distance-guessed
  bonds; immutable compiled geometry/connectivity is cached internally.
- **`molpy.core` is now a Python surface over molrs, not a second kernel.**
  `Frame`/`Block`, entity refs, `Element`, and force-field primitives preserve
  molrs object identity; graph, Box, Region, trajectory, selector, and unit
  conveniences are native subclasses or Python call-shape sugar. Element/unit
  truth, PBC geometry, alignment/replication, and CL&Pol scaleLJ all execute in
  Rust, with an architecture manifest and AST gate guarding that ownership.
- **Polymer topology assembly helpers** on `PolymerBuilder`: `build_linear`,
  `build_sequence`, `build_ring`, `build_star` format CGSmiles and call the sole
  entry `build(cgsmiles)`. Supporting types: `SiteMap` (site labels + leaving-H
  charge fold), `Replicas` (grid / linear multi-chain worlds), and
  `linear_cgsmiles` / `ring_cgsmiles` / `star_cgsmiles` formatters.
- **User-guide section [Polymer Topologies](user-guide/topology/index.md)** with
  eleven pages paired 1:1 to `examples/topology/01_*.py` … `11_*.py` (linear,
  block, ring, star, comb, telechelic, exhaustive/random gel, end-linked, dual
  network, prepolymer + agent). Smoke with `python examples/topology/run_all.py`.

### BREAKING

- **`frame.simbox` is renamed to `frame.box` (paired with molrs).** The cell
  attribute is once again `frame.box` (`molrs.Box | None`). There is no
  `simbox` alias or shim — assign and read `frame.box` only. C / CXX / WASM
  entry points follow the same rename (`molrs_frame_set_box`, `frame_set_box`,
  …). On-disk MolStore layout still uses the historical `simbox/` group name.
- **Public storage types live on the molpy facade again.**
  `from molpy import Frame, Block, Element, MetaValue` is the supported path;
  they are identity re-exports of the molrs types (`molpy.Frame is molrs.Frame`).
  `molpy.core` does **not** re-export them, and there is still no
  `molpy.core.frame` / `molpy.core.element` module. User code must not import
  molrs directly.
- **Amber polymer construction now uses MolPy chemistry directly.**
  `AmberPolymerBuilder(library, reaction, ...)` and
  `AmberTools.build_polymer(..., reaction=...)` require a `molpy.Reaction` over
  `fields.SITE`. The private `port` language, `reaction_preset`, and
  one-hydrogen leaving-group guess are removed; the backend translates the
  shared compiled residue product into prepgen/tleap inputs.

- **Pint and the runtime CL&Pol fragment file are removed.** `UnitSystem` uses
  molrs' native unit registry; only the native `lj` compatibility context is
  accepted. Custom scaleLJ data must be supplied as `FragmentScaling` values
  instead of a parameter-file path.
- **`TypeBucket` and `molpy.core.utils` are removed.** They belonged to the old
  detached Python entity model; graph collections now come from molrs live
  handle views, and CSV input is `Block.from_csv`.
- **The typifier package has one contract: `Typifier` is `MolGraph -> MolGraph`.**
  `typify()` is concrete — copy, `match()`, write the annotations back — and
  `match()` is the single abstract method. Typifiers are generic over the graph,
  so an `Atomistic` typifier and a `CoarseGrain` one share the same pipeline.
  Removed: `ForceFieldTypifier`, `RegionTypifier`, `TypeScope`, `typify_region`,
  `retype_region`, `relaxed()`, `atomtype_matches`, `skip_atom_typing` and the
  four other `skip_*_typing` flags, and the public `PairTypifier`. Deleted
  modules: `typifier/{base(old),bond,angle,dihedral,pair,mmff,atomistic,scope}.py`
  — six of them were dead code that shadowed live names (`TypifierBase`,
  `PairTypifier` and `atomtype_matches` were each defined twice in the package).
- **`molpy.typifier.ForceFieldParams` replaces `Atomistic.assign_bonded_types`.**
  Assigning parameters to a graph whose types are already known is not a
  typifier; it is the second half of one. The old method lived in `core` (a
  force-field judgment in the data-model layer), matched force-field types by
  splitting their names on `"-"` — no wildcards, no atom classes, no overlay
  layers — and left a term it could not match *silently* unlabelled.
  `ForceFieldParams(ff).assign(graph)` returns a new graph, matches by
  specificity, and raises on an unparameterised term unless `strict=False`.
- **`GraphAssembler(reaction, typifier=..., reach=N)`** — `reach` is now a
  required argument alongside a typifier, and `AmberToolsTypifier(amber)` no
  longer takes one. `TypeScope` dissolved into
  `AffectedRegion.around(graph, touched, reach=N)`, the only place that derives
  the write-back radius (`max(reach, 2)`) and the extraction radius
  (`interior_reach + reach`) from the one number they share.

  *A defence moved from the type system to the test suite.* `graph-assembler-01`
  established that the radius is decided by the typifier and is "never a
  setting", because a wrong `reach` silently mistypes — measured, `reach`
  one too small mistyped 22 of 46 written-back atoms of a PEO junction. With
  `reach` supplied to the assembler, nothing rejects a wrong value at
  construction; it is caught by the oracle test (region typing == whole-graph
  typing) instead. Note that `AmberToolsTypifier(amber, reach=2)` already took
  `reach` from the user; this only moves where it is passed.

### Fixed

- **Hydrogen perception preserves angles, dihedrals and impropers.** The old
  `complete_valence()` facade copied atoms and bonds only. It has been removed;
  cut sites now call `Perceive.find_hydrogens` directly, which returns a
  non-mutating graph with every existing relation kind preserved.
- **A region typed from a raw slice could not be typed at all.** Because the
  extracted ball is `interior_reach + reach` wide, an interior atom's receptive
  field reaches exactly to the boundary atoms, and a raw cut leaves those with
  unfilled valences — radicals, to a SMARTS matcher. Only the AmberTools path
  completed them. Measured on p-xylene with OPLS-AA at `reach = 2`, **12 of 19
  raw slices are rejected outright** by the typifier (a truncated aromatic
  carbon types as something no bonded term covers); PEO and methyl acrylate
  happen to survive a raw cut. `RegionTypes.of` now completes every region
  before typing it — every region *is* a cut, so there is no condition to get
  wrong. A typifier itself never guesses whether its graph is a fragment:
  truncation is a fact about provenance, not something readable off a graph's
  valences.
- **An unparameterised bonded term was written back as `None`.**
  `RegionTypes._capture_links` recorded terms whose type was undecided, and
  `apply_to` then erased whatever the parent's term already carried. Undecided
  terms are now skipped.

- **`molpy.reacter` is removed.** Reaction semantics live in `molpy.Reaction`
  (a re-export of the molrs SMIRKS engine); chemistry lives in the reaction
  SMARTS itself. `Reacter`, `ReactionResult`, `TopologyDetector`, the 14
  anchor/leaving selectors, `BondReactReacter` and `ReactionPresets` are gone.
- **`molpy.builder.crosslink` is removed.** `Crosslinker`,
  `DeterministicCrosslinker` and `RandomCrosslinker` held a selector and
  forwarded `apply` to `assemble`. Crosslinking is now
  `GraphAssembler(rxn).assemble(melt, RandomSelector(...))`. The
  `crosslink_gel()` / `write_lammps()` recipes are documentation, not library.
- **`PolymerBuilder` is rebuilt** on the assembly kernel:
  `PolymerBuilder(MonomerLibrary({...}), reaction, typifier=..., placer=...)`
  `.build(cgsmiles)`. `build_sequence`, `PolymerBuildResult`, `Connector` and
  the `connector=` / `reacter=` dual constructor are gone. A repeat unit is a
  molecule with `fields.SITE` marked on the atoms that may react — there is no
  port system and no `<` / `>` direction.
- **`molpy.core.AffectedRegion` moved** to `molpy.typifier.affected_region`.
  It is not a data-model type: it is the ball a graph edit disturbed, and its
  radius is decided by the typifier's `TypeScope`.
- **`molpy.core.region_radius` is removed** along with the `_FLOOR = 4`
  fallback and the three `context_radius` declarations. A typifier declares a
  `TypeScope(reach)`; `AmberToolsTypifier` now requires `reach=`.
- `BondReactTemplate` moved from `molpy.reacter.bond_react` to
  `molpy.io.data.lammps_bond_react` (it is an IO artifact). The public
  `write_bond_react_map` / `write_lammps_bond_react_system` are unchanged.


- **Region retyping wrote wrong atom types.** The extraction radius and the
  write-back radius were conflated into one `radius`; correctness required
  `reach <= 1`, which no real typifier satisfies. Measured against whole-graph
  typing, `AmberToolsTypifier`'s default mistyped 22 of 46 written-back atoms
  of a PEO junction. The guard that should have caught it was gated on a
  `strict` flag read through a two-level `getattr(..., False)`, so for any
  typifier without an `atom_typifier` attribute it never fired.
- **Malformed reaction SMIRKS silently paired the wrong sites.**
  `_find_component` returned component `0` when a forming-bond map number
  appeared in no reactant pattern; it now raises.
- Polymer assembly was `O(N^2)` in chain length from four independent sources.

### Added

- `molpy.Reaction`, `molpy.SmartsPattern`, `molpy.NeighborQuery`, `molpy.Graph`,
  `molpy.perceive_aromaticity`, `molpy.find_rings` — re-exports, so user code
  never imports molrs.
- `molpy.core.fields.SITE`; `Entity` subscripting accepts a `FieldSpec`.
- `molpy.builder.assembly`: `GraphAssembler`, `Selector`, `TopologySelector`,
  `ProximitySelector` (`Exhaustive` / `Spacing` / `ExplicitPair` / `Random`),
  `MonomerLibrary`, `PolymerBuilder`, `Placer` / `ResiduePlacer`.

## 0.7.0 - 2026-07-08

Requires `molcrafts-molrs == 0.7.0` (molpy and molrs release as a pair).

### Breaking

- **OPLS-AA and MMFF typifiers moved to molrs.** `molpy.typifier` now re-exports
  `OPLSAATypifier` and `MMFFTypifier` from `molrs.typifier`; the molpy-side
  typifier internals (`adapter`, `graph`, `matcher`, `layered_engine`,
  `dependency_analyzer`) are removed. Construct via
  `OPLSAATypifier(source, strict=…)` and call `.typify(mol)` (returns a typed
  `molrs.Atomistic`; use `.to_frame()` for tabular access). The old
  `OplsTypifier(ff, strict_typing=…)` constructor is gone.

### Added

- **CL&P ionic-liquid typifier** (`molpy.typifier.ClpTypifier`) — now functional,
  implemented as an OPLS-AA overlay (molrs SMARTS assigns CL&P atom types from
  `clp.xml`; molpy assigns parameters). Matches the CL&P reference charges/σ/ε.
- **Offline network crosslinking** over the molrs SMARTS/SMIRKS engine.
- **Distance-based `SoftPotential` + force-field-free relax** in `molpy.optimize`.
- **Region-scoped typing with a hash-keyed `RetypeCache`** — only the subgraph
  affected by a graph edit (`AffectedRegion`) is re-typed, keyed by the molrs
  isomorphism-invariant structural graph hash.

### Changed

- Covalent radii merged into `Element`; `SoftPotential` reads radii via `Element`.

## 0.6.0 - 2026-07-03

Requires `molcrafts-molrs == 0.6.0`.

### Breaking

- **`molpy.compute` selection API** now reads atom-tuple selections (pairs /
  triplets / orientation axes) from the frame's core topology blocks
  (`bonds` / `angles` / `dihedrals` / `orientations`); the `groups` argument is
  removed.
- **`Molpack` removed** — use the `Packmol` wrapper directly.
- **`LammpsSystem` removed** — use `molpy.io.write_lammps_system(...)`.

### Added

- Deterministic copolymer sequence generators (`AlternatingSequenceGenerator`,
  `BlockSequenceGenerator`).
- `AmberTools` facade owning the GAFF2/AM1-BCC workflow (`parameterize`,
  `parameterize_ion`, `build_polymer`).
- `SmilesReader` — OOP SMILES → 3D `Atomistic` reader.

### Fixed

- O(N²) blow-ups in packing and LAMMPS writing (string columns hoisted to numpy
  once); LAMMPS force-field writer de-duplicates coefficients by type name.

### Changed

- Pin `molcrafts-molrs==0.6.0`; docs config migrated `mkdocs.yml` → `zensical.toml`.

## 0.5.1 - 2026-07-01

Requires `molcrafts-molrs == 0.5.1` (molpy and molrs now release as a pair).

### Added

- **analysis-parity compute operators** in `molpy.compute`: angular/dihedral/
  distance and combined distribution functions, the spatial distribution
  function, the Van Hove correlation `G(r, t)`, Legendre reorientational
  correlations, hydrogen-bond detection, radical Voronoi tessellation with
  domain/void/charge analysis, and vibrational spectra (VDOS, IR, Raman, VCD,
  ROA, resonance Raman).
- `molpy.version.check_molrs_version()` — run on `import molpy`; import fails
  when the exact paired `molcrafts-molrs` version is missing or does not match.

### Changed

- Documentation now builds with **Zensical**; user-guide notebooks are
  pre-rendered to Markdown. The compute section documents every operator with
  textbook-style guides and a full API reference.

## 0.5.0 - 2026-06-21

Requires `molcrafts-molrs == 0.1.5`.

### Removed

- **SMARTS GAFF typifier removed.** `molpy.typifier.GaffTypifier` (and the
  internal `_GaffAtomTypifier`), the `typifier/gaff.py` module, and the bundled
  `data/forcefield/gaff.xml` are gone. GAFF atom types and AM1-BCC charges now
  come exclusively from AmberTools (`antechamber` / `prepgen`) through the
  unchanged AmberTools wrapper; the 41 generic SMARTS-matcher tests are kept.

### Fixed

- **tip3p water `theta0` is now expressed in radians**, matching molrs's
  angles-internal-radians convention.

### Added

- PEO polymer-electrolyte workflow examples.

## 0.4.2 - 2026-06-18

Additive release — no API renames or breaking changes.
Requires `molcrafts-molrs == 0.1.4`.

### Added

- **GROMACS TRR / XTC and DCD trajectory readers** (`read_trr_trajectory`,
  `read_xtc_trajectory`, `read_dcd_trajectory`) and TRR / XTC writers
  (`write_trr`, `write_xtc`) — thin delegations to the molrs backend.

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
- **molrs is now an exact required dependency.** `molcrafts-molrs==0.7.0` is in
  the core `dependencies`; missing package metadata or any version mismatch is
  an import error. `Frame` and `Block` are owned only by molrs. Molpy's `Box`
  remains geometry sugar over the native box, and compute operators execute in
  Rust. There is no pure-Python fallback.
- **The RDKit-backed compute node was removed.** `molpy.compute.rdkit`
  (`Generate3D` / `OptimizeGeometry` over `RDKitAdapter`) is gone.
  `molpy.compute.Generate3D` is now the molrs-backed trunk operator, taking an
  `Atomistic` graph and returning a fresh 3D structure. The RDKit adapter
  (`molpy.adapter.rdkit`) is retained as an
  **optional** external backend; `rdkit` remains an optional extra, not a
  required dependency.
- **`Frame` / `Block` are imported directly from molrs.** Molpy's former
  top-level, `molpy.core`, and `molpy.core.frame` compatibility exports were
  deleted; use `from molrs import Frame, Block`. The Python-side object-column
  overflow (`_objects`) is gone:
  columns are **numpy-only** (float / int / bool / str). Assigning an
  object / `None` / ragged column now raises `molrs.BlockDtypeError` at write
  time instead of being silently stored on the Python side. `frame.simbox` is
  the only cell attribute and returns `molrs.Box`; `frame.box` is deleted.
  Exact-dtype metadata uses `frame.meta` + `molrs.MetaValue`; untyped
  `frame.metadata` is deleted. No aliases or compatibility shims remain.

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
  the native `.time` array; exact-dtype snapshot metadata uses `frame.meta`.
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
  `read_lammps_data` returns `LammpsDataResult`; coefficient sections live on
  `result.forcefield`, while `result.frame` remains pure Frame state. Writers
  accept `forcefield=` and `type_labels=` explicitly. Malformed coefficient
  lines raise `ValueError` instead of being swallowed.
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
