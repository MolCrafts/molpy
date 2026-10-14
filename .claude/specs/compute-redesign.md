---
title: Compute redesign — one contract, one entry per quantity, one implementation per algorithm
slug: compute-redesign
status: approved
created: 2026-10-14
chain: compute-redesign (01-contract → 02-time-series → 03-spectra-dielectric → 04-histogram-grid → 05-dynamics → 06-order-environment → 07-groups → 08-voronoi → 09-retire-legacy → 10-bindings → 11-catalog → 12-molpy)
repos: molrs (01–11), molvis (11, consumer), Atomiverse (10, consumer), molpy (12)
---

# Compute redesign

## Summary

A four-axis review of `molrs::compute` (Rust core ≈27k lines), its Python, WASM
and C++ bindings (≈10k lines), `molrs::signal`, and molpy's mirror found the
module is not OOP-composable and is not single-sourced:

- **Duplication.** The time-correlation estimator exists six times, the windowed
  MSD three times, the trapezoid rule four times, the finite-difference
  derivative four times, and cos² windows three times. Histogram binning exists
  about eight times and grid/voxel spreading five times. The per-cluster
  reductions exist four or five times, and the four PMFTs are about 80%
  copy-paste. Steinhardt/Hexatic and SolidLiquid/ContinuousCoordination repeat
  the same bond-harmonic kernel. Voronoi carries two private cell lists, PCA a
  private power iteration, and cluster/environment-match private
  connected-components.
- **Multiple entries for one thing.** These are five public paths to an ACF:
  `Acf`, `autocorrelation`, `Vacf`, `GreenKuboDiffusion`, `signal::acf_fft`.
  Further duplicates:
  - Three byte-identical spectrum `Fit`s: Power, IR and VCD.
  - Alias types that only forward: `RoaSpectrum`, `ResonanceRamanSpectrum`,
    `ResonanceRamanTensor`, `ClusterCenters`, `RadiusOfGyration`.
  - A god-struct, `ClusterProperties`.
  - Four `Rdf` entry methods.
  - 23 public free functions in `compute`, and all of `signal`.
- **The contract forces anti-patterns.**
  - `Compute::compute(&self, frames, args)` forces a `frames` slice on about 20
    kernels that ignore it; tests pass `&[]` 43 times.
  - No kernel can read a field other than `atoms.x/y/z`, so velocities, charges,
    dipoles and orientations arrive as positional tuples such as
    `(&Array2, f64, usize)`.
  - `Fit` and `Check` take bare arrays and cannot consume a compute's output.
  - Accumulators share no trait.
  - Pipelines are broken: descriptor → PCA → KMeans because `Kmeans` hardcodes
    `n_dims = 2`, and partitions are four incompatible label types.
- **Silent fallbacks.**
  - `max_lag` is silently clamped, and fit windows silently widen or collapse.
  - A missing mass becomes `1.0`, and a zero quaternion becomes the identity.
  - A missing Voronoi unit tag means "Bohr".
  - PCA returns an unconverged vector as if converged.
  - NaN reaches histogram bin 0.
  - The C++ API turns every error into an empty `Vec`/`NaN`/`false`.
- **Bindings fork behaviour.**
  - Defaults differ across core, Python, WASM and `catalog.rs`. For example,
    Cubatic is 1.0/0.95/500/4 in core but 5.0/0.9/100/10 in the catalog, and
    the GreenKubo window is `hann` in Python but `cosine_sq` in WASM.
  - Enum parsing is re-implemented in each binding.
  - WASM rewrites Rdf normalisation.
  - Python and WASM take orientation pairs in opposite orders.

This chain replaces the contract and collapses every duplicate into one owner.
When it lands:

1. **One verb.** Every analysis, fit and check is a type implementing one trait
   with one method, `compute`.
2. **One entry per quantity.** No public free functions, no alias types, and no
   second method that computes the same thing. Chemistry-named surfaces such as
   `Msd` and `Vacf` are the entries for their quantities. They are thin
   configurations over the one estimator, `Correlator` (D4).
3. **One implementation per algorithm.** The algorithm table below names each
   algorithm's single owner. A second copy is a bug.
4. **`new` = configuration, `compute` = data.** A constructor takes only the
   parameters that configure the analysis. Everything being analysed is passed
   to `compute`: arrays, `SimBox`, `Neighbors`, upstream results. There is no
   data-wrapper type to build first. Compute never touches a `Frame` and never
   names a field; reading columns is the caller's step through the
   `Frame`/`Block` API (D2).
5. **No silent fallback.** Every out-of-contract input is a `ComputeError`.
6. **Thin bindings.** Defaults, enum names, validation and conversions live once
   in core. Bindings, the WASM catalog and molpy are derived mirrors.

## Domain basis

This chain introduces no new physics. Every estimator kept here already exists
in the tree. The redesign changes **who owns it** and **how it is reached**, not
what it computes. There are two exceptions:

- **The numbers that change on purpose.** These are listed exhaustively in
  "Behaviour changes", and each one names the golden test it re-pins.
- **The literature conventions that pick between two existing in-tree
  conventions when they disagree:**
  - Unbiased time-correlation estimator: C(k) = (1/(T−k)) Σ_τ a(τ)·b(τ+k). This
    is the existing `Vacf` convention, and it becomes the single default.
  - FFT evaluation of the windowed MSD: Kneller et al., *Comput. Phys. Commun.*
    91, 191 (1995). This is the existing `msd/mod.rs` S-recurrence.
  - Bond-orientational order: Steinhardt, Nelson & Ronchetti, *Phys. Rev. B* 28,
    784 (1983). This is the existing `core::math` Y_lm. The planar ψ_k is its 2D
    analogue.
  - Histogram bins are half-open, [lo, hi). This is the existing `Rdf` rule, and
    VanHove is aligned to it.

## Decisions (D2 and D4 ruled by the operator on 2026-10-14)

**D1. One trait.** `Fit` and `Check` are deleted. A fit is a `Compute` whose
input is an upstream output, such as `&TimeCorrelation`. A check is a `Compute`
whose output implements `Verdict`. The "raw curve, then fit" doctrine stays as
naming and documentation, not as a second trait.
*Why:* one verb is what makes outputs chain generically into the next input, and
what lets every binding share one shape.

**D2. `new` takes configuration; `compute` takes the data. Compute knows no frames.**
(Operator rulings, 2026-10-14: no `from_frame` / `from_frames`; "new 只放配置参数，compute 才传入被计算的数据".)

**Constructors take configuration only.**
- A compute type's `new` / `with_*` accept only what *configures* the
  analysis: bin axes, cutoffs, `max_lag`, kernels, windows, tolerances, seeds,
  and the unit system.
- They never accept the data being analysed: positions, masses, charges,
  orientations, atom indices, pair lists, keys or reference structures.
- Current violations move into `compute`:
  - `SpatialDistribution::with_orientation(pairs)`
  - the `masses: Option<Vec<F>>` fields of `CenterOfMass`, `InertiaTensor`,
    `RadiusOfGyration` and `ClusterProperties`
  - the atom-index lists stored in `SpatialDistribution`
  - the `keys` of `Cluster::compute_keyed`
  - every WASM constructor that captured `dt`
- Link 01 audits every `pub fn new|with_*` in `compute` and lists the rest.

**`compute` takes everything being analysed.**
- That means arrays (`ArrayView`s / slices), existing core value types
  (`SimBox`, `Neighbors`, `PairSource`, `RegularGrid` when it describes where
  input values live), and upstream results.
- A sample interval `dt` describes the data, so it is `compute` data. So are a
  state's volume, temperature and Ewald boundary.
- No wrapper has to be built with `new(data)` before calling `compute`, so
  there is no `TimeSeries`, `Snapshot`, `Weights`, `SeriesPair`,
  `Orientations`, `Labels` or `ThermoState`.

**`Input` is named arguments, not an object.**
- In Rust, `Compute::Input<'a>` is a per-type struct of borrowed data, for
  example `MsdInput { positions, dt, simbox }`.
- It is written as a struct literal at the call site and has no `new`; it is
  Rust's spelling of named arguments.
- Python and WASM bind its fields as the keyword arguments of `compute`:
  `Msd(max_lag=500).compute(positions, dt=10.0)`.

**Compute knows no frames and no field names.**
- No type in `compute` has a constructor or method taking a `Frame`,
  `FrameAccess` or a field key.
- Reading velocity columns, masses or quaternions out of a frame is the
  caller's step, through the existing `Frame`/`Block` API.
- `FieldSpec` is not introduced, and compute contains no field-name literal.

**Composition goes through traits, not wrappers.**
- `Partition` is implemented by label arrays (`[u32]`, `Array1<u32>`) and by
  `ClusterResult`, `KmeansResult`, `EnvironmentMatchResult` and the H-bond
  component labels. So a user's `mol_id` column and an upstream result both
  fit.
- `FeatureMatrix` is implemented by `Array2<F>` and by the descriptor results.

**Series operations are computes.** Each takes its data in `compute`:
- `Derivative::new(stencil).compute(series, dt)`
- `Unwrap::new().compute(positions, images, boxes)`
- `DipoleMoment::new().compute(positions, charges, partition)`
- `PolarizabilityParts::new().compute(alpha)` (isotropic / anisotropic)

*Why:* an analysis object is a reusable configured instrument, and data flows
through `compute`. This removes the dummy `frames` argument, every hardcoded
field name, the per-binding frame-reading code and the data-holding parameter
bags, without adding a second data-container layer next to `Frame`.

**D3. One snapshot's data in, one result out; the caller iterates.**
- A per-snapshot kernel's `compute` takes one snapshot's data.
- A trajectory kernel is either:
  - a `Stream` (an accumulator folded over per-frame data), or
  - a kernel taking a T × N × D array plus `dt`.
- `Batch<C>` is the one generic parallel map for "the same kernel over many
  snapshots". It replaces the frame-loop and rayon boilerplate in 24 files.

**D4. `Correlator` is the single estimator; chemistry-named surfaces sit on top.**
(Operator ruling, 2026-10-14.)

`Correlator` is public and generic:
- Every time-correlation computation in molrs runs inside
  `Correlator { kernel: LagKernel, max_lag, origins, mean, norm, reduce }`.
- That covers FFT and streaming evaluation, the lag grid, normalisation and
  reduction.
- It is the escape hatch for any correlation that has no named surface.

A **surface** is a named, re-exported type for a quantity chemists know by name.
It has a fixed shape:
- It holds one `Correlator` with the configuration that *defines* the quantity
  fixed.
- It exposes only the parameters a user legitimately tunes.
- Its `compute` is a single delegating line, and it contains no algorithm code.
- Its streaming form is the inner `Correlator`'s accumulator.

Two surface rules:
- **One name per configuration.** Two surfaces with the same fixed
  configuration are one surface. For example, the Einstein–Helfand conductivity
  curve is `Msd` applied to the collective-dipole series.
- **Same numbers as today.** A surface's fixed configuration is pinned in link
  02 by reproducing the current golden numbers of the types it replaces.

| Surface | Fixed `Correlator` configuration | Replaces |
|---|---|---|
| `Msd` | `SquaredDisplacement`, auto; `origins` tunable (`All` = old Window, `First` = old Direct); `reduce` tunable (mean / per-particle) | `Msd`, `MsdAccumulator`, `MsdMode`, `MsdResult`, `MsdTimeSeries`, `EinsteinDiffusion`, `EinsteinConductivity` (raw) |
| `Vacf` | `Product`, auto, mean-subtracted, unbiased, mean over degrees of freedom | `Vacf`, `VacfAccumulator`, `GreenKuboDiffusion`, `Acf`, `autocorrelation` |
| `OnsagerCorrelation` | `SquaredDisplacement`, cross | `OnsagerCorrelation` |
| `DipoleAcf` | `Product`, auto, mean-subtracted, summed over Cartesian components | `DebyeRelaxation` (raw) |
| `CurrentAcf` | `Product`, auto, summed over Cartesian components | `GreenKuboConductivity` (raw), `IrFlux` (⟨Ṁ·Ṁ⟩ is the current ACF) |
| `ReorientationAcf` | `Legendre { l }`, auto | `LegendreReorientation` |
| `RotationalAutocorrelation` | `WignerD { l }`, auto, `Origins::First` (freud semantics) | `RotationalAutocorrelation` |
| `Survival` | `Persistence` (continuous) or `Product` on presence (intermittent) | `pair_survival_tcf`, `hbond_lifetimes` |

What remains a direct `Correlator` composition, with no surface, and is shown in
the spectroscopy docs:
- the VCD (μ̇·ṁ), ROA (α̇·Ġ′) and Raman iso/aniso correlators;
- the Ṁ·M rate cross term.

The operator may promote any of these to a surface later; each would be one
delegating type.

The physics outside the estimator moves to where it belongs:
- series construction, which is itself a compute: `DipoleMoment`,
  `Derivative`, `PolarizabilityParts`;
- the fit or transform that carries the physical prefactor.

*Why:* the algorithm exists once (`Correlator`), while users find the quantity
under the name they know.

**D5. A neighbour table knows its geometry.**
- `core::Neighbors` records the box it was searched in:
  `Neighbors::simbox() -> Option<&SimBox>`.
  - It is `Some` for every table a search produced (`NeighborList::neighbors`,
    `NeighborQuery::query` / `query_self` / `query_columns`, `VerletSkin`), and
    it survives `repack`, `filter_rad` and `filter_sann`.
  - It is `None` for tables nothing searched in a box: hand-built
    `from_pairs` tables (`with_simbox` attaches one), ghost-image tables, and
    force-field topology pair lists.
  - A kernel that needs a volume rejects `None`; it never invents a box.
- A new `core::PairSource` trait is implemented by the materialised
  `Neighbors` table and by a built `NeighborList`. Its methods are `mode`,
  `simbox`, `columns`, and
  `visit_pairs(&mut dyn FnMut(i, j, Option<dist_sq>, Option<disp>))`.
  An absent column arrives as `None`, never as zero.
- Pair kernels that only stream pairs take `&dyn PairSource`. Kernels that need
  columns or random access take `&Neighbors`.
- So `Rdf` has one entry and keeps the zero-materialisation path that
  `compute_self` exists for today.
- `RdfMode` is derived from the table's `QueryMode`, which becomes the single
  source of truth for half-shell vs directed.

**D6. `signal` folds into `compute`.**
- Its only consumer is `compute`; `ff` uses `rustfft` directly.
- The `signal` feature and module are deleted. `compute` takes `dep:rustfft`.
- `molrs.signal` and `mp.signal` are deleted.
- Windows, FFT planning and frequency grids become types: `Window`, `Spectrum`,
  `FrequencyGrid`.

**D7. A closed set of lag kernels is an enum, not a generic.**
`Correlator { kernel: LagKernel, … }` is one concrete type. That keeps it
bindable: one Python/WASM class with a kernel argument. The set of kernels is
the set molrs ships, so it is closed.

**D8. Defaults live only in core.**
- Every tunable has its default in the core type: `Default`, or `new` plus
  `with_*` setters.
- Bindings take `Option<T>` and call the setter only when given. Python
  signatures say `None`, never a literal.
- WASM `catalog.rs` is generated from core `ComputeSpec` descriptors. UI labels,
  ranges and categories move to molvis (`stage/src/analysis/registry.ts`).

**D9. A wrapped trajectory is rejected, not silently analysed.**
- There is no type-state wrapper (D2). Instead, the displacement computes take
  the per-frame boxes as optional `compute` data: `Msd`, `OnsagerCorrelation`
  and `Correlator { kernel: SquaredDisplacement }`.
- When boxes are given, a frame-to-frame step longer than half the box's
  shortest perpendicular height is the `Wrapped` error.
- Unwrapped positions come from `Unwrap` (image flags, or continuity).
- This closes the unchecked "MSD needs unwrapped coordinates" precondition
  (`msd/mod.rs:32-39`, the VanHove self part, Onsager, `dipole_moment`).

## Design

### 1. The contract — `compute/contract.rs` (replaces `analysis_contract.rs`)

```rust
/// One analysis = one type = one verb. `&self` is an immutable parameter bag;
/// identical (self, input) → identical output. Outputs are always final.
pub trait Compute {
    type Input<'a>;
    type Output: Clone + Send + Sync + 'static;
    fn compute(&self, input: Self::Input<'_>) -> Result<Self::Output, ComputeError>;
}

/// A check's output.
pub trait Verdict { fn passed(&self) -> bool; }

/// Bounded-memory, frame-by-frame form of a `Stream` compute.
pub trait Accumulate {
    type Item<'a>;
    type Output;
    fn push(&mut self, item: Self::Item<'_>) -> Result<(), ComputeError>;
    fn finish(&self) -> Result<Self::Output, ComputeError>; // non-consuming: peek mid-run
    fn fold<'a, I: IntoIterator<Item = Self::Item<'a>>>(mut self, items: I)
        -> Result<Self::Output, ComputeError> where Self: Sized { /* provided */ }
}

/// A compute that also streams. Its `compute` body is exactly
/// `self.accumulator().fold(input)` — the algorithm lives once, in the accumulator.
pub trait Stream: Compute {
    type Accumulator: Accumulate<Output = Self::Output>;
    fn accumulator(&self) -> Self::Accumulator;
}

/// The one generic "same kernel over many snapshots" (rayon when enabled).
pub struct Batch<C>(pub C);
impl<C: Compute + Sync> Batch<C> { pub fn compute<'a>(&self, inputs: &[C::Input<'a>]) -> Result<Vec<C::Output>, ComputeError> where C::Input<'a>: Sync + Copy; }
```

The following are deleted:
- `ComputeResult` and its `finalize`. A result is final when returned, which
  removes the `finalized` flags and the WASM `finalized = false; finalize()`
  patch.
- `DescriptorRow`, replaced by `FeatureMatrix` (§3).
- `Fit` and `Check` (D1).

Until link 09, the old trait is renamed `LegacyCompute` so the tree compiles
while families migrate one link at a time.

**Named enums.** Each core enum that is a user choice, such as `LagKernel` or
`Window`, uses one in-tree `named_enum!` macro. The macro generates `FromStr`
(canonical names only, no aliases), `as_str`, `ALL` and `Display`.
- **Deleted:** `SurvivalMethod::parse` (aliases `cr`/`rf`/`imm`),
  `EwaldBoundary::from_name`, `InternalCoordinate::from_kind`, and the private
  `parse_window_type`.
- **Typed instead of `String`:** `window_type: String` and the
  `RouteAgreement` `(String, Array1)` labels.

**Errors.** `ComputeError` stays as the single error type and absorbs
`SignalError`. It gains `Wrapped` (D9) and `NotConverged { what, iterations }`.
The `MissingBlock` / `MissingColumn` / `MissingSimBox` variants are deleted,
because compute no longer reads frames.

### 2. Data at the `compute` boundary — no wrapper types

Every row below is passed to `compute`, never to `new` (D2). No row is a type
the caller must construct.

| Data | Rust form inside `Input` | Notes |
|---|---|---|
| per-item vectors (positions, velocities, dipoles) | `ArrayView2<F>` (N×3) | |
| trajectories | `ArrayView3<F>` (T×N×D) + `dt: F` | plus optional per-frame `&[SimBox]` for displacement computes (D9) |
| per-item scalars (mass, charge, radius) | `ArrayView1<F>` | required wherever the physics needs it; never an implied 1.0 |
| orientations | quaternions `ArrayView2<F>` (N×4), directors (N×3) or planar angles (N) | each kernel takes the representation its math is written in, and validates it in `compute` (zero norm → error) |
| groups | `&dyn Partition` | `n_items()`, `group_of(i) -> Option<u64>` (`None` = in no group). Keys need not be contiguous; a consumer orders groups by ascending key. Implemented by `&[T]`, `Vec<T>` and `Array1<T>` for unsigned `T`, and by partition-producing results |
| features | `&dyn FeatureMatrix` | `Array2<F>` and descriptor results implement it |
| geometry | `&SimBox`, `&Neighbors`, `&dyn PairSource`, `&RegularGrid` | core value types that already exist |
| thermodynamic state | `volume: F`, `temperature: F`, `boundary: EwaldBoundary` | |
| upstream results | `&TimeCorrelation`, `&QlmTable`, `&DielectricSpectrumResult`, … | |

**Configuration value types**, which go into `new` and validate there:
- `Axis` (validated `(min, max, n)`, half-open bins);
- `RegularGrid` when it is the output grid;
- `Window`, `LagKernel`, `Origins`, `MeanPolicy`, `Normalization`, `Reduction`;
- `Stencil`, `UnitPreset`.

**Output types** are results; the caller never builds them:
- `TimeCorrelation` (`lag_times: Array1`, `values: Array2`, lags × channels),
  the output of `Correlator` and the input of every fit and spectrum;
- `Histogram<N>` (`density()`, `shell_volumes(dim)`);
- `QlmTable`;
- `Presence` (T × n_pairs, from `HBondsResult::presence()` or `PairPresence`);
- every `*Result`.

### 3. One owner per algorithm — the single-source table

Each row is the **only** implementation after the chain. Every "replaces"
location is deleted.

| Algorithm | Owner | Replaces (current copies) |
|---|---|---|
| Lagged product correlation (auto/cross, FFT + streaming ring) | `Correlator { kernel: Product }` | `dynamics/acf.rs:90`, `transport/correlation.rs:82` (+`apply_unbiased_norm`, `component_means`), `spectroscopy/mod.rs:197-290`, `msd/mod.rs:114`, `einstein_conductivity.rs:76`, `hbond/lifetime.rs:174`, `signal::acf_fft*`/`xcorr_fft*`, `VacfAccumulator` |
| Lagged squared displacement (FFT S-recurrence + streaming ring) | `Correlator { kernel: SquaredDisplacement }` | `msd/mod.rs:148-221`, `msd/accumulator.rs`, `einstein_conductivity.rs:66-103`, `onsager.rs:131-144` |
| Lagged Legendre / Wigner-D of orientations | `Correlator { kernel: Legendre{l} / WignerD{l} }` | `order/reorientation_legendre.rs`, `order/rotational_autocorrelation.rs` |
| Continuous persistence of a presence series | `Correlator { kernel: Persistence }` | `pair_survival.rs:223-236`, `hbond/lifetime.rs:158-171` |
| Lag grid / origins / mean policy / normalisation / reduction | `Correlator` parameters (`max_lag`, `Origins`, `MeanPolicy`, `Normalization`, `Reduction`) | `lag_times` + 5 inline copies, `MsdMode`, `AcfArgs` |
| Finite-difference time derivative | `Derivative { stencil }` | `spectroscopy/mod.rs:177`, `correlation.rs:137`, `dielectric/mod.rs:108-118`, `dielectric_spectrum.rs:186-191` |
| Trapezoid integration | `CumulativeTrapezoid` | `fitting/mod.rs:110`, `hbond/lifetime.rs:27`, `dielectric_spectrum.rs:128-131, 800-803` |
| Window + zero-pad + one-sided FFT + frequency grid | `Spectrum { window: Window, pad }` | `PowerSpectrum`, `IrSpectrum`, `VcdSpectrum` (identical), `signal::apply_window`/`forward_fft_onesided`/`frequency_grid`, `cosine_sq_window` (`mod.rs:144`), inline taper (`dielectric_spectrum.rs:175-180`), hardcoded 4× pad |
| Dielectric / conductivity prefactors | one `pub(crate)` prefactor implementation in `dielectric/`, called with compute-time volume/temperature and the configured `UnitPreset` | `dielectric/mod.rs:232, 316`, `dielectric_spectrum.rs:314, 413, 493, 582` |
| Collective / per-group dipole | `DipoleMoment` (`compute(positions, charges, partition?)`) | `dipole_moment`, `decompose_current`, docs' hand-rolled dipoles |
| Static dielectric constant | `StaticDielectric` | `static_dielectric_constant`, `static_dielectric_constant_components` |
| 1-D … N-D histogram, edges, centers, density, radial shell volumes | `Axis`, `Histogram<N>` | `rdf/mod.rs:63-95,130-142`, `rdf/result.rs:110-115`, `van_hove.rs:104-106,151-153,178-187`, `correlation_function.rs:38-68`, `histogram1d.rs`, `combined.rs:59-98`, `direct.rs:159`, `bond_orientational_order.rs:68-94`, PMFT push closures, `renormalize_density` |
| Grid spreading with a kernel | `RegularGrid` + `trait SpreadKernel { Gaussian, Indicator }` | `gaussian_density.rs:110-185`, `sphere_voxelization.rs:80-130`, `diffraction_pattern.rs:113-150`, `GridSpec`, `voronoi DensityGrid` |
| Per-particle bond-harmonic average (q_lm / ψ_k) | `BondAverage<B: BondBasis>` → `QlmTable` (`pub(crate)` engine; `QlmTable` public) | `steinhardt.rs:150-221` (+`:264`, `:307`), `hexatic.rs:96-145`, `solid_liquid.rs:87-115`, `continuous_coordination.rs:81-105`, bond→(θ,φ) at `steinhardt.rs:192`, `local_descriptors.rs:70`, `bond_orientational_order.rs:115` |
| PMFT binning + PMF | `Pmft<C: PairCoordinate>` (`R12`, `Xy`, `Xyt`, `Xyz`) | `pmft/{r12,xy,xyt,xyz}.rs` (box, symmetric test, push, ρ_ref, −ln, edges, frame loop) |
| Group reductions (anchor + MIC unwrap, weighted moments) | `GroupReduce` (`pub(crate)`) behind `CenterOfMass`, `GyrationTensor`, `InertiaTensor` | `center_of_mass.rs:72-97`, `cluster_centers.rs:52-76`, `cluster/properties.rs:113-178`, `gyration_tensor.rs:73`, `voronoi/integrate.rs:392` |
| Connected components | `core::Topology::connected_components` | `cluster/mod.rs:57-113` (CSR + BFS), `environment_match.rs:252` (`Dsu`) |
| Symmetric eigendecomposition | `op::linalg::eigh_sym` / `eigh_sym_3x3` | `pca.rs:196-258` (power iteration) |
| SO(3) sampling / axis-angle | `op::so3`, `op::rigid` | `cubatic.rs:196-237` |
| Cell-list neighbour search | `core::NeighborList` / `NeighborQuery` | `voronoi/radical.rs:251 CellGrid`, `voronoi/integrate.rs:452 CellList`, `pair_survival.rs:199-207` (O(n²) orthorhombic), `hbond/detect.rs:139-144` (per-frame internal query) |
| Frame field reading | **none in compute.** The caller uses the `Frame`/`Block` API (D2) | `positions.rs` (deleted), `cluster/mod.rs:43,154`, `rdf`, `msd`, `einstein_diffusion`, `shape/*`, `pmft/orientation.rs`, `voronoi/integrate.rs:54,66,90`, Python `order.rs:29,130-147,312`, `density.rs:211-221`, `distribution.rs:174-182,303-305`, WASM `voronoi.rs:12-33`, `pmft.rs:13-26` |
| Input guards (`disp`, `dist_sq`, self-query) | `require.rs` (kept) — called by every kernel, including PMFT, `LocalDensity`, `CorrelationFunction`, BOO, `EnvironmentMatch`, which inline `matches!` today | inline `symmetric = matches!(…)` copies |

### 4. Public surface — every current symbol and where it goes

Legend:
- **K**: kept, migrated to the new contract.
- **→ X**: deleted; the capability is reached through X.
- **M**: becomes a method on the named owner.
- **S**: a chemistry-named surface over `Correlator` (D4): one delegating `compute`, no algorithm code.

**Contract.**

| Symbol | Fate |
|---|---|
| `Compute` | K (redefined) |
| `Fit`, `Check` | → `Compute` |
| `Verdict` | K |
| `ComputeResult` | deleted |
| `DescriptorRow` | → `FeatureMatrix` |
| `ComputeError` | K |

**Time correlation** (`signal`, `dynamics`, `transport`, `msd`).

| Symbol | Fate |
|---|---|
| `Acf`, `AcfArgs`, `AcfResult`, `autocorrelation` | → `Vacf` surface or `Correlator` / `TimeCorrelation` |
| `Vacf` | S (surface over `Correlator{Product}`) |
| `VacfArgs`, `VacfResult`, `VacfAccumulator`, `GreenKuboDiffusion` | → `Vacf` (its stream is the `Correlator` accumulator) |
| `GreenKuboConductivity*` | → `CurrentAcf` on `Derivative` of the `DipoleMoment` series |
| `Msd` | S (surface over `Correlator{SquaredDisplacement}`) |
| `EinsteinDiffusion*`, `MsdAccumulator`, `MsdMode`, `MsdResult`, `MsdTimeSeries` | → `Msd` (`Origins::All` = old Window, `Origins::First` = old Direct, per-particle via `reduce`) |
| `EinsteinConductivity*` | → `Msd` on the collective-dipole series |
| `OnsagerCorrelation` | S (surface, cross `SquaredDisplacement`) |
| `OnsagerCorrelationArgs`, `OnsagerCorrelationResult` | → `compute(a, b, dt, boxes?)` data / `TimeCorrelation` |
| `DebyeRelaxation*` | → `DipoleAcf` surface; ⟨M²⟩ is `TimeCorrelation` at lag 0; V, T and boundary are `compute` data of the spectrum routes |
| `DipoleRateCross*` | → `Correlator{Product}` cross on (Ṁ, M) |
| `lag_times`, `unbiased_cartesian_xcorr`, `apply_unbiased_norm`, `component_means`, `gradient_axis0_order2` | → `Correlator` / `Derivative` internals |
| `signal::{acf_fft*, xcorr_fft*, forward_fft_onesided, frequency_grid, apply_window, WindowType, SignalError}` | → `Correlator`, `Spectrum`, `FrequencyGrid`, `Window`, `ComputeError` |

**Fits and spectra.**

| Symbol | Fate |
|---|---|
| `LinearFit`, `CumulativeTrapezoid`, `Plateau`, `DebyeFit` | K. Input is `&TimeCorrelation` plus a channel; window violations error. `DebyeFit` moves to `fitting/`. |
| `PowerSpectrum`, `IrSpectrum`, `VcdSpectrum`, `SpectrumResult` | → `Spectrum` |
| `IrFlux*` | → `DipoleMoment` → `Derivative` → `CurrentAcf` → `Spectrum` |
| `VcdCrossFlux*` | → `Correlator` cross of μ̇ and ṁ series |
| `RamanTensor*`, `ResonanceRamanTensor*` | → `PolarizabilityParts` + `Correlator` |
| `RoaCrossTensor*` | → `Derivative` + `Correlator` cross |
| `RamanSpectrum`, `RamanSpectrumResult` | K |
| `RoaSpectrum`, `ResonanceRamanSpectrum` | → `RamanSpectrum` parameters |
| `EinsteinHelfandSpectrum`, `GreenKuboSpectrum`, `DipoleAutocorrelationSpectrum`, `DipoleRateCrossSpectrum`, `DielectricSpectrumResult` | K. These are four distinct routes: `new(window, units)`, and `compute(&TimeCorrelation, volume, temperature, boundary)`; all share `Spectrum`. |
| `KramersKronig`, `ConductivitySumRule`, `RouteAgreement` (+`*Check` outputs) | K as `Compute` with `Verdict` outputs. Tolerances become fields, and the input is `&DielectricSpectrumResult`. |
| `EwaldBoundary` | K, `compute` data of the spectrum routes |

**Dielectric and kinetic.**

| Symbol | Fate |
|---|---|
| `dipole_moment` | → `DipoleMoment` |
| `decompose_current` | → `DipoleMoment` with a `Partition` |
| `current_density` | → `Derivative` |
| `static_dielectric_constant*`, `StaticDielectricResult` | → `StaticDielectric` |
| `kinetic_energy` | → `KineticEnergy` |
| `kinetic_temperature` | M `KineticEnergyResult::temperature(n_dof)` (units from `KineticEnergy::new(units)`) |
| `center_of_mass_velocity` | → `CenterOfMass` on velocities with no box |

**Dynamics and H-bonds.**

| Symbol | Fate |
|---|---|
| `pair_survival_tcf`, `PairSurvivalResult`, `SurvivalMethod` | → `PairPresence` → `Survival` surface |
| `VanHove` | K. It uses `RadialAxis` / `Histogram` and `Origins`, and holds a `core::NeighborQuery`: its pair set is per (origin, lag), so a caller cannot supply it. |
| `HBonds`, `HBond`, `HBondCriterion`, `HBondDistanceKind` | K. The input is per-frame cross-query `&Neighbors` plus the donor/acceptor selection; `+1.05` Å becomes a criterion field. |
| `presence_from_hbonds` | M `HBondsResult::presence` |
| `hbond_components`, `HBondNetworkResult` | M `HBondsResult::components(frame)`, returning a label array that implements `Partition` |
| `hbond_lifetimes`, `HBondLifetimeResult` | → `Survival` surface on `HBondsResult::presence()` + `CumulativeTrapezoid` |

**Histograms, distributions, density, diffraction.**

| Symbol | Fate |
|---|---|
| `Rdf` | K, as one `Stream`. Input item `&dyn PairSource`; `compute_self` / `compute_cross` deleted. |
| `RdfAccumulator` | K, obtained only through `Rdf::accumulator` |
| `RdfMode` | → derived from `QueryMode` |
| `RdfResult` | K |
| `Histogram1d`, `AxisSpec`, `renormalize_density` | → `Axis` / `Histogram<N>` |
| `DistributionFunction`, `CombinedDistribution`, `Observable` (+3 impls), `InternalCoordinate` | K, on `Histogram<N>`; `compute` takes the positions, box and the atom-index tuple array (N × arity) as data |
| `AtomGroups` | deleted (a data wrapper built with `new`); the index-tuple array is passed to `compute` |
| `CorrelationFunction*`, `LocalDensity*` | K (`PairSource`) |
| `GaussianDensity*`, `SphereVoxelization*`, `SpatialDistribution*` | K (`RegularGrid` in `new`). `SpatialDistribution`'s atom indices and orientation pairs move from the constructor to `compute`. |
| `GridSpec` | → `RegularGrid` |
| `StaticStructureFactorDebye*` | K. All-pairs non-periodic by definition, stated in its doc; `compute` takes positions only (no box); `1e-9` becomes a field. |
| `StaticStructureFactorDirect*`, `DiffractionPattern*` | K (`Axis` / `RegularGrid`) |

**Order, environment, PMFT.**

| Symbol | Fate |
|---|---|
| `Steinhardt`, `SteinhardtResult` | K. The result exposes `QlmTable` and implements `FeatureMatrix`. |
| `steinhardt_qlm` | → `SteinhardtResult::qlm` |
| `SolidLiquid`, `ContinuousCoordination` | K. Input is `(&QlmTable, &Neighbors)`, so the q_lm are not recomputed. |
| `Hexatic` | K (`BondAverage<Planar>`) |
| `Nematic`, `Cubatic` | K. `compute` takes one snapshot's orientation array, giving one result instead of N identical ones. |
| `LegendreReorientation*` | → `ReorientationAcf` surface (`Legendre { l }`) on an orientation-series array |
| `RotationalAutocorrelation` | S (surface, `WignerD { l }`, `Origins::First`); `RotationalAutocorrelationArgs` / `Result` → `compute` data / `TimeCorrelation` |
| `BondOrientationalOrder`, `LocalDescriptors` (`FeatureMatrix`), `LocalBondProjection` (an optional quaternion array as `compute` data; the flag is deleted), `EnvironmentMatch` (registration limit is an explicit field; exceeding it errors) | K |
| `AngularSeparation*` | K (quaternion arrays as `compute` data) |
| `PmftR12/Xy/Xyt/Xyz` (+`Args`, `Result`) | → `Pmft<R12 / Xy / Xyt / Xyz>`, `PmftResult<C>` |
| `orientation_quaternions`, `planar_orientation_angles` | deleted. Callers pass the quaternion or angle arrays they read from the frame. |

**Groups and learning.**

| Symbol | Fate |
|---|---|
| `Cluster` | K. Input is `{ pairs: &Neighbors, merge: Option<&dyn Partition> }`; `compute_keyed` is deleted (keys are the `merge` partition). |
| `ClusterResult` | K (`Partition`) |
| `ClusterProperties*` | → `CenterOfMass` + `GyrationTensor` + `Partition::sizes` |
| `CenterOfMass`, `InertiaTensor` | K. `compute(positions, simbox?, partition, masses)`. The `masses` config field is deleted, and there is no default 1.0. |
| `GyrationTensor` | K, with a weights array as `compute` data |
| `ClusterCenters*` | → `CenterOfMass` with a uniform weights array |
| `RadiusOfGyration*` | M `GyrationTensorResult::radius_of_gyration` |
| `Pca` | K. Input `&dyn FeatureMatrix`, `n_components` field, `eigh_sym`. |
| `Kmeans` | K. Input `&dyn FeatureMatrix`; `n_dims` comes from the matrix. |
| `PcaResult`, `KmeansResult` | `FeatureMatrix` / `Partition` |

**Voronoi.**

| Symbol | Fate |
|---|---|
| `RadicalVoronoi` | K as `Compute`: `compute(positions, simbox, radii)` → `VoronoiCells`, on `core::NeighborList` |
| `VoronoiIntegration` | K as `Compute`: named `Input` struct instead of 7 positional arguments; `LengthUnit` in `new`, with no Bohr fallback |
| `VoronoiDomainAnalysis`, `VoronoiVoidAnalysis` | K as `Compute`; `void` already uses `connected_components` |
| `DensityGrid` | → `RegularGrid` + values array as `compute` data |
| `MolecularMoments`, `VoronoiCells`, `VoronoiFace`, `VORONOI_BOUNDARY` | K |
| `polarizability_finite_field` | → `FiniteFieldPolarizability`; the unused `moments_zero` argument is deleted |

After the chain, `compute` exports no `pub fn` at module level and no `pub type`
alias of another public type.

### 5. Silent fallbacks removed — each becomes an error or an explicit parameter

This is the exhaustive list from the review. Each item is fixed in the link
that owns the file.

- **Lag windows:**
  - `max_lag.min(n-1)` everywhere becomes `OutOfRange`, and the
    `clamps_not_panics` test (`vacf.rs:155`) is rewritten to expect the error.
  - VanHove: dropped lags, `stride.max(1)` and `n_orig.max(1)` become errors.
    With no `SimBox` it errors instead of returning zeros plus a flag.
- **Fit windows:** `LinearFit` widening and `Plateau` collapsing become errors.
- **Spectroscopy:**
  - `bose_factor` returns 1.0 for T ≤ 0; that becomes an error.
  - The Raman 0.0 "skip" sentinel becomes `Option`.
  - The dielectric auto-plateau magic `0.1` becomes an explicit `Plateau`
    input.
- **Dielectric and kinetic:**
  - `current_density`'s NaN row 0 disappears, because the derivative stencil
    defines its edges.
  - `center_of_mass_velocity` with zero total mass becomes an error.
- **Masses and weights:** the shape kernels' `map_or(1.0, …)` disappears;
  masses are a required `compute` argument.
- **Orientations:**
  - Zero quaternions get three different answers today; all three become
    errors.
  - `planar_orientation_angles`' unchecked z-axis assumption becomes an error
    for non-planar input.
- **Voronoi:**
  - The `.unwrap_or(false)` Bohr assumption becomes an explicit `LengthUnit`.
  - `dims.max(1)` and the out-of-cell clamps become errors.
- **EnvironmentMatch:**
  - The silent fallback above 8 neighbours becomes an explicit-limit error.
  - `Err(_) => INFINITY` becomes error propagation.
- **PCA:**
  - An unconverged result becomes `NotConverged`; this disappears entirely once
    PCA uses `eigh_sym`.
  - Negative eigenvalues are no longer clamped with `max(0.0)`.
- **Histograms:**
  - `debug_assert`-only validation becomes `Axis::new` returning `Result`.
  - A NaN sample becomes `NonFinite`.
  - An empty `density()` becomes an error rather than zeros.
- **Pair survival and H-bonds:**
  - An `exclude_self` that is silently ignored becomes an error.
  - An empty donor/acceptor selection becomes `EmptyInput`.
  - Lifetimes that are all zero because no bond ever formed become explicit
    `Option`s.
- **Cubatic:** `with_n_chains(n.max(1))` becomes an error.
- **Panics:** `Rdf::with_dimensionality` `assert!`, `DiffractionPattern::with_axis`
  `assert!` and the `SpatialDistribution` unchecked index panic all become
  errors. The silent 3-D fallback at `rdf/result.rs:114` is removed.
- **C++ API:** every swallowed error becomes a `Result` (a C++ exception).

### 6. Bindings (link 10) — derived, thin, identical

- **Defaults only in core (D8).** Python signatures take `None`; WASM takes
  `Option`. A parity test builds every compute through each binding with no
  optional argument and compares its parameters with the core `Default`.
- **`new` = configuration, `compute` = data, in every binding.**
  - Python `compute(...)` takes keyword data arguments that mirror the fields
    of the core `Input` struct, and WASM does the same.
  - No binding constructor takes data. The 32 WASM classes that capture `dt`
    in their constructor move it to `compute`.
  - No binding adds a frame-reading helper (D2), and no binding accepts a
    `Frame`.
  - Surfaces (D4) bind as classes under their core names.
- **Enums.** Bindings accept `str` and parse it through the core `FromStr`
  (`named_enum!`). They have no `match` of their own.
- **Results.**
  - Core outputs get `#[cfg_attr(feature = "serde", derive(Serialize))]`.
  - WASM serialises them with `serde-wasm-bindgen`, which deletes the 31 `Out`
    structs and 52 `js_value` copies.
  - Python wraps each output in a `#[pyclass]` generated by one macro from its
    field list. No `PyDict` results remain, which deletes the 85 hand-typed keys.
- **Errors.** There is one `From<ComputeError>` per binding, which deletes the
  113 WASM `format!` sites.
- **Generic types.** Each `Pmft<C>` instantiation binds as `PmftR12`, `PmftXy`,
  `PmftXyt` and `PmftXyz`. `Correlator` binds as one class taking a `kernel`.
- **Name parity.** A test asserts that the set of `Compute` types in core equals
  the set bound in Python and the set bound in WASM, by name. No
  binding-exclusive analyses remain.
- **Python layer.**
  - `_native.pyi` defaults read `None`.
  - `_protocol.py`'s `Compute` Protocol is `compute(input)`, and its test
    asserts that **every** exported compute class satisfies it, not "at least
    one".
  - The empty `molrs/compute/{cluster,…}` directories are deleted.
- **C++ (cxxapi).**
  - The surface becomes the `Stream` accumulators of `Correlator` and `Rdf`,
    returning `Result`.
  - `MsdAccumulator::diffusion`, a façade combining MSD and fit, is deleted.
    Callers compose `Correlator` → `LinearFit`.
  - Atomiverse follows as a consumer change.

### 7. Catalog (link 11)

- **Core descriptors.** Each core compute type implements
  `const SPEC: ComputeSpec` via one `compute_spec!` macro. The spec holds:
  - the name;
  - the input kind, derived from `Input`;
  - the parameters, as (name, type, default taken from the type's `Default`).
- **Generated catalog.** `molrs-wasm` exports `computeSpecs()`, generated from
  these descriptors, and `catalog.rs` is deleted.
- **molvis.** UI labels, categories, units and slider ranges move into molvis
  `stage/src/analysis/registry.ts`, which validates against `computeSpecs()` at
  load.
- **Invalid literals.** The `volume = 0.0` and `"3.14159265"` literals cease to
  exist, because nothing types a default anymore.

### 8. molpy (link 12)

molpy mirrors by identity (`mp.compute is molrs.compute`), so no molpy code
changes. Its work is the following:

- **Deletions.** `mp.signal` (D6), and the ghost `__pycache__`-only packages
  (`potential/`, `typifier/`, `pack/`, `parser/`, `cli/`, `data/`,
  `builder/assembly/`, `optimize/`, `conformer/` dirs).
- **Docs.**
  - Rewrite `docs/compute/*.md`, `docs/api/compute.md` and
    `docs/developer/extending-compute.md` onto the new surface, showing each
    deleted recipe as its composition.
  - Delete the "module-level function" endorsements (`extending-compute.md:55`,
    `signal.md:9`) and the `CollectiveDipole` example.
- **Doc figures.** `scripts/docs_series/lj.py` is rebuilt on `mp.md` +
  `PairLjCut` + `NeighborList` + `KineticEnergy`, and stops importing `molrs`.
  `transport.py:171` uses `CumulativeTrapezoid`.
- **Tests.**
  - `tests/test_compute/test_*.py` science tests move to Rust unit tests
    (AGENTS.md: science depth lives in Rust).
  - molpy keeps one identity loop over `molrs.compute.__all__`, replacing the
    6-name spot check at `tests/test_public_paths.py:330-335` and
    `test_compute/test_init.py`.
- **Pin.** The pin moves to the molrs minor that ships this chain, following the
  release order: molrs first, tagged and published, then molpy.

## Link 01 audit — data held by analysis types (D2)

Found in 01 by scanning every `pub fn new|with_*` and every non-result struct
field under `molrs/src/compute`. Each row is moved to `compute` (or deleted) in
the named link.

| Type | Data it holds today | Moves to | Link |
|---|---|---|---|
| `CenterOfMass`, `InertiaTensor`, `RadiusOfGyration`, `ClusterProperties` | `with_masses(&[F])` → `masses: Option<Vec<F>>` | `compute(…, masses)`; `RadiusOfGyration` / `ClusterProperties` deleted | 07 |
| `SpatialDistribution` | `new(reference: Vec<usize>, template: Array2, target: Vec<usize>, grid)`; `with_orientation(pairs)` | reference/target indices, template coordinates and orientation pairs → `compute`; `grid` stays in `new` | 04 |
| `HBonds` | `new(donors: Vec<(u32, u32)>, acceptors: Vec<u32>, …)` | donor/acceptor selections → `compute` | 05 |
| `AtomGroups` | `new(arity, flat: Vec<Idx>)` (data wrapper) | deleted; the index-tuple array → `compute` of the distributions | 04 |
| `DensityGrid` | `new(origin, basis, dims, density: Vec<F>)` (data wrapper) | `RegularGrid` + values array → `VoronoiIntegration::compute` | 08 |
| `RdfAccumulator` | public `new(rdf: Rdf)` | private; reached only through `Rdf::accumulator()` | 04 |
| `Histogram1d` | `counts: Array1<F>` (a mutable accumulator posing as a type) | replaced by `Histogram<N>` output + accumulator | 04 |
| `MsdTimeSeries::new(per_frame)`, `ClusterCentersResult::new(centers)` | result constructors | deleted with their types | 02 / 07 |

Configuration that legitimately looks like data and stays in `new`:
- `Steinhardt` / `ContinuousCoordination` `l: &[u32]`
- `VanHove` `lags: Vec<usize>`
- `StaticStructureFactorDebye` `k_values`, `StaticStructureFactorDirect`
  `k_vecs` (where to evaluate)
- `CombinedDistribution` `observables` / `axes`
- `Rdf` / `CorrelationFunction` derived bin edges

## Behaviour changes (numbers that move on purpose)

Every edited golden is named in its link. Any other golden that moves is a
regression.

1. **Correlation normalisation becomes one default.**
   - The raw curves used by IR, VCD and Raman change from unnormalised sums to
     the unbiased estimator. The resulting spectra are in arbitrary units, so
     their scale changes and their shape does not.
   - VDOS and IR now use the same estimator.
   - Goldens re-pinned in 02/03: `ir_flux`, `vcd_cross_flux`, `raman_tensor`,
     `roa_cross_tensor`, `spectroscopy/*`.
2. **Gyration weighting.**
   - Before: `GyrationTensor` was unweighted about the geometric centre, while
     `RadiusOfGyration` and `ClusterProperties` were mass-weighted about the
     centre of mass.
   - After: both take an explicit weights array in `compute`, and Rg = √tr(G) of the same weighting.
     Re-pinned in 07.
3. **VanHove bin edges.** VanHove now uses half-open bins like Rdf, so d = r_max
   is no longer clamped into the last bin. Re-pinned in 05.
4. **Defaults.** Every binding and catalog default becomes the core default:
   - Cubatic 1.0 / 0.95 / 500 / 4
   - EnvironmentMatch `max_neighbors` 8
   - LocalDensity diameter 0
   - GaussianDensity r_max 3σ
   - `exclude_self`: none (it must be given)
   - GreenKubo window: the core `Window` default, chosen once in 03
   - Kmeans `max_iter` / `seed`: given core defaults, because core has none
     today

   Visible to molvis and Python users. Re-pinned in 10/11.
5. **Orientation pair order.** The order is unified to the core convention:
   (head, tail), with axis = head − tail under minimum image.
   - The WASM `setOrientationPairs` path and Python `Nematic`, which has no
     minimum image today, change.
   - Re-pinned in 06.
6. **Fallbacks become errors** (§5). Inputs that used to "work" with a silently
   wrong answer now fail.

## Chain

Every link runs `cargo test -p molcrafts-molrs --lib --features full,filesystem
-- compute` (plus `core::neighbors` for 01). Binder crates are allowed not to
compile from 02 until 10. Clippy, fmt, doctests and binder builds run once, at
the end of 10, as `prek run --all-files --hook-stage pre-push`. No release
happens inside the chain.

| Link | Scope | Depends |
|---|---|---|
| 01-contract (code-complete 2026-10-14) | New `Compute`/`Verdict`/`Accumulate`/`Stream`/`Batch` (`Batch<C>` is itself a `Compute`: `Input = Vec<C::Input>`, earliest error wins); the old trait renamed `LegacyCompute`. The per-type `Input` struct convention (named data arguments, no `new`). `Partition` / `FeatureMatrix` traits, implemented for label arrays / `Array2`. An audit table of every `new`/`with_*` that takes data (D2), each assigned to its migrating link. `Neighbors` gains `SimBox`; `PairSource` trait. `named_enum!`. `ComputeError` changes. | — |
| 02-time-series | `Correlator` (all five kernels, FFT + streaming ring, equal to FFT round-off), `Derivative`, `Unwrap`, the D4 surfaces with their fixed configurations pinned against today's goldens, `TimeCorrelation`, `Window`, `Spectrum`, `FrequencyGrid`. Delete `signal` (module + feature). Migrate or delete the time-correlation rows of §4. | 01 |
| 03-spectra-dielectric | Fits on `TimeCorrelation`, `Spectrum` unification, the Raman family with `PolarizabilityParts`, `DipoleMoment`, `StaticDielectric`, the four ε(ω) routes, the checks, the one prefactor implementation (fixes the fs/ps doc contradiction at `dielectric_spectrum.rs:59,281`), `KineticEnergy`. | 02 |
| 04-histogram-grid | `Axis`, `Histogram<N>`, `RegularGrid`, `SpreadKernel`; migrate Rdf (`Stream` over `PairSource`), CorrelationFunction, LocalDensity, distributions, S(k) ×2, DiffractionPattern, Gaussian/Sphere density, SpatialDistribution. | 01 |
| 05-dynamics | `Presence`/`PairPresence`, survival through `Correlator`, H-bond input/presence/components, VanHove on `Histogram` + `core::NeighborQuery`. | 02, 04 |
| 06-order-environment | `BondBasis`/`BondAverage`/`QlmTable`; Steinhardt, Hexatic, SolidLiquid, ContinuousCoordination, BOO, LocalDescriptors, LocalBondProjection, EnvironmentMatch, AngularSeparation, Nematic, Cubatic; `Pmft<C>`. | 01, 04 |
| 07-groups | `GroupReduce`; Cluster (`connected_components`, `merge`), CenterOfMass, GyrationTensor, InertiaTensor; delete ClusterCenters, RadiusOfGyration, ClusterProperties; Pca (`eigh_sym`, `n_components`), Kmeans (`FeatureMatrix`). | 01, 06 (FeatureMatrix producers) |
| 08-voronoi | `Compute` impls, core cell list, explicit units, `FiniteFieldPolarizability`, `GridField`. | 04, 07 |
| 09-retire-legacy | Delete `LegacyCompute`, `analysis_contract.rs`, `positions.rs`; rewrite the `compute/mod.rs` module docs (removes the stale `molpy.compute.Workflow` reference). Run the §4 closing greps. | 02–08 |
| 10-bindings | §6 in molrs-python, molrs-wasm, molrs-cxxapi; full pre-push gate; Atomiverse consumer note. | 09 |
| 11-catalog | §7; molvis `registry.ts` migration (cross-repo). | 10 |
| 12-molpy | §8, after the molrs release that carries 01–11. | molrs tag |

## Testing strategy

Unit tests live in `#[cfg(test)]` modules next to the code. There is no
third-party oracle (AGENTS.md § Testing Rules).

- **Primitives (new tests, each in its owner's module):**
  - `Correlator`:
    - FFT vs a direct O(T·L) loop, for every kernel on hand-written series.
    - Streaming ring vs batch FFT for every lag ≤ the cap. This test absorbs
      the existing VACF and MSD accumulator equality tests.
    - Each `Origins`, `MeanPolicy`, `Normalization` and `Reduction` variant on a
      3-sample series with hand-computed values.
    - `SquaredDisplacement` with boxes given on a wrapped trajectory is the
      `Wrapped` error.
  - `Axis`:
    - Half-open edge membership, and NaN → `NonFinite`.
    - Shell volumes in 2-D and 3-D against closed forms.
  - `RegularGrid::stencil`: periodic vs non-periodic wrap.
  - `GroupReduce`: a group straddling the box boundary.
  - `BondAverage`: a q_6 FCC/BCC golden carried over from Steinhardt.
  - `Pmft<C>`: a two-particle golden per coordinate system.
  - Every surface (D4): its `compute` equals the explicitly configured `Correlator` bit-for-bit on a hand-written series.
- **Invariance.** The existing golden numbers of every migrated kernel are kept.
  The test moves with the kernel, and its assertion literals do not change. The
  only exceptions are the re-pins listed in "Behaviour changes". Each link's
  diff shows any edited literal next to the re-pin item it implements.
- **Contract (acceptance greps, see the acceptance file):**
  - no module-level `pub fn`
  - no field-name literals
  - no `Fit`/`Check`/`ComputeResult`/`DescriptorRow`
  - no `unwrap_or` / `map_or` defaults on physical inputs
  - no `#[pyfunction]` and no free `#[wasm_bindgen]` fn in the compute bindings
- **Bindings (10):** the FFI seam only — construct, call, round-trip a type —
  plus the defaults-parity and name-parity tests.

## Out of scope

- **New analyses**, and any change to an estimator's mathematics beyond the
  listed re-pins.
- **Performance work** beyond keeping the existing paths. Rdf's
  zero-materialisation and the rayon parallelism are preserved through
  `PairSource` and `Batch`. A benchmark comparison runs once, in 10, against
  `.github/workflows/bench.yml`'s RDF/MSD benches: no regression above 10%.
- **`core::NeighborList` internals.** Only the `SimBox` field on `Neighbors` and
  the `PairSource` trait are in scope.
- **molvis UI redesign**, beyond moving labels and ranges out of molrs.
- **Atomiverse code**, which is a consumer follow-up after 10.
