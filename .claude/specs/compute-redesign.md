---
title: Compute redesign — one contract, one entry per quantity, one implementation per algorithm
slug: compute-redesign
status: approved
created: 2026-10-14
chain: compute-redesign (01-contract → 02-time-series → 03-spectra-dielectric → 04-histogram-grid → 05-dynamics → 06-order-environment → 07-groups → 08-voronoi → 09-retire-legacy → 10a-compute-spec → 10-bindings → 11-catalog → 12-molpy)
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

*As built (02):* `Correlator` also carries `with_dimension(d)` (the component
count every item must have), which `DipoleAcf`, `CurrentAcf` and
`OnsagerCorrelation` fix to 3 — it is how their `(T, 2)` rejection survives
without numeric code in the surface. `ReorientationAcf::new(l, max_lag)` is one
order per instance (the old type returned `C_1` and `C_2` together) and has no
origin stride (the old `with_stride` is dropped; all origins). `Survival` takes
`SurvivalKind { Continuous, Intermittent }`; `Ssp` stays with `pair_survival_tcf`
until `PairPresence` (05). Every surface also implements `Stream` by handing out
its correlator's accumulator.

*As built (05):* `Ssp` is gone: the stable-states picture is
`PairPresence::new(r0, exclude_self)?.with_r1(r1)?` (hysteresis: enter at
`d ≤ r0`, stay while `d ≤ r1`) followed by `Survival { Continuous }`.

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

*As built (02):* `Spectrum::new(window, pad)` returns `OneSidedSpectrum`
(`frequencies` on the `FrequencyGrid`, the **unscaled** real/imaginary bins,
`n_fft`); each caller applies its own scale (`1/n_fft`, `dt`). The dielectric
piecewise-linear transform pads a *derived* signal to a length of its own, so it
calls the crate-internal `Spectrum::onesided(signal, n_fft)` — the same single
FFT. `DiffractionPattern`'s 2-D FFT still imports `rustfft` directly; link 04
decides between a 2-D `Spectrum` form and amending ac-005.

*As built (04):* routed, not exempted. The 2-D power spectrum is the
crate-internal `Spectrum::power_2d_centered(image)` (row then column FFTs,
`|F|²`, FFT-shifted — the former `fft2_power_shifted`, same plans, same
numbers) in `time_series/spectrum.rs`; `DiffractionPattern` calls it, so
`rustfft` is imported only in `correlator.rs` and `spectrum.rs` and ac-005 holds
as written.

**D7. A closed set of lag kernels is an enum, not a generic.**
`Correlator { kernel: LagKernel, … }` is one concrete type. That keeps it
bindable: one Python/WASM class with a kernel argument. The set of kernels is
the set molrs ships, so it is closed.

*As built (02):* because `Legendre { l }` / `WignerD { l }` carry data,
`LagKernel` is a hand-written enum (`name()`, `NAMES`, `order()`, `Display`),
not a `named_enum!`, and has no `FromStr`; a binding parses the kind name plus
`l` (link 10). `Origins`, `MeanPolicy`, `Normalization`, `Reduction`, `Window`,
`Stencil` and `SurvivalKind` are `named_enum!`s.

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

*As built (02):* `Correlator` implements `Stream`, but its `compute` is the FFT
batch algorithm, not `self.accumulator().fold(..)` — the fold is `O(T · L)`.
The `Stream` doc names this as the one exception, and unit tests pin batch ==
fold at every lag for every kernel and option (ac-007). Because
`Stream::accumulator()` takes no data, the per-frame item
`CorrelatorFrame { a, b, dt, simbox }` carries `dt`; the first frame latches it
and a different `dt` later is `OutOfRange`.

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

*As built (09):* `analysis_contract.rs` is deleted with `LegacyCompute`, `Fit`,
`Check`, `ComputeResult` (and its `Vec<T>` impl) and `DescriptorRow` (its last
implementor was `RdfResult`). `ComputeError` keeps exactly the variants that
have a producer: `DimensionMismatch`, `BadShape`, `NonFinite`, `OutOfRange`,
`EmptyInput`, `UnknownName`, `Wrapped`. *Deviation:* besides the three
frame-reading variants, `NotConverged` and `MolRs` (with its
`From<MolRsError>`) are deleted too — `NotConverged`'s only intended producer
was PCA's power iteration, gone with `eigh_sym` in 07 (a full Jacobi
decomposition has no convergence flag), and no compute kernel calls a core API
returning `MolRsError` (verified by compiling without the conversion). Either
comes back with its first producer.

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

*As built (02):* `Derivative` has `Stencil::{Central, Gradient, Backward}` and
replaces `spectroscopy/mod.rs:177` and `correlation.rs:137`. `current_density`
(`dielectric/mod.rs:108-118`, link 03 removes its NaN row) and the even-function
stencil inside the Einstein–Helfand taper (`dielectric_spectrum.rs`) are not yet
routed through it — link 03. That taper's cos² now goes through
`Window::CosineSq`, so `Window` is the only cos² implementation. The
squared-displacement FFT centres each column first (a displacement does not see
an offset; this only removes cancellation) and sets lag 0 to exactly 0.

*As built (04):*
- **`Axis`** (`binning/axis.rs`) is the one bin rule: `Axis::new(min, max, n)
  -> Result`; `index(x)` is half-open and exact against `edges()` (a
  proportional guess corrected against `min + i·width`, the `.min(n − 1)`
  clamp lives only here); `index_closed(x)` also puts `x = max` in the last bin
  (polar/azimuthal angles in BOO, the closed `k ≤ k_max` reciprocal-lattice
  binning of `StaticStructureFactorDirect`); the crate-internal
  `linear(x)` is the cloud-in-cell split of the reference `CDF::AddToBin`
  (former `cic_split`, same formula, closed support `[min, max]` because its
  last half-bin includes `max`); `shell_volumes(dim)` (2 or 3, else
  `OutOfRange`; `min < 0` is `OutOfRange`). A non-finite sample is `NonFinite`
  (`index` 0 — a caller with a position re-labels it). **`Histogram<const N>`**
  (`binning/histogram.rs`) holds `ArrayD` counts indexed by bin tuple plus the
  binned weight; `add` (nearest, half-open), `add_linear` (tensor-product CIC),
  `merge`, `from_counts` are crate-internal; `density()` errors `EmptyInput`
  when empty.
- **`RegularGrid<const D = 3>`** (`grid/regular_grid.rs`): origin, voxel-edge
  basis, dims, pbc; `new`, `centered(extent, dims)`, `over_box(simbox, dims)`
  (3-D), `voxel_center`, `voxel_volume`, `index_of` (wraps periodic axes,
  `None` outside an open one, `NonFinite` on NaN — it used to land in voxel 0),
  `stencil(center, r, visit)` and `spread(kernel, center, visit)`. A general
  (sheared) basis indexes through `M⁻¹`; a diagonal one divides, so the ortho
  numbers are bit-identical to the old per-file loops. **`SpreadKernel`**
  (`cutoff`, `normalization(dim)`, `profile(r²)`) with **`Gaussian`**
  (`new(σ)`, `with_cutoff(r)`, default `3σ` = `DEFAULT_CUTOFF_SIGMAS`) and
  **`Indicator`** (`new(radius)`), in `grid/kernel.rs`.
- **Stencil reach unified.** One rule: the candidate range along each axis is
  `⌊t − reach − ½⌋ … ⌈t + reach − ½⌉` around the fractional coordinate `t`,
  a superset of every centre within `r`, and the inclusive `r² ≤ cutoff²` test
  alone decides membership. That is the old `ceil` reach (`SphereVoxelization`,
  `DiffractionPattern`: numbers unchanged); `GaussianDensity`'s freud-parity
  `floor` reach skipped voxels whose centres lie inside `r_max`, which it now
  includes — see Behaviour changes (04 note). Wrapping happens only on periodic
  axes; `DiffractionPattern`'s unconditional `rem_euclid` is gone.
- Replaced and deleted: `rdf/mod.rs` edges/centers/`accumulate_d2` and
  `rdf/result.rs` shell volumes; `correlation_function.rs` bins;
  `histogram1d.rs` (`Histogram1d`, `cic_split`, `renormalize_density`);
  `combined.rs` `AxisSpec`; the isotropic bin of `direct.rs`; the BOO θ/φ bins
  (BOO stays on `LegacyCompute` until 06 but bins through `Axis`);
  `density::wrap_index`; `GridSpec`. VanHove keeps its own closed-range
  nearest-bin closure until 05 (only its doc links to `Histogram1d` changed);
  PMFT needed no rewiring beyond `pmft/orientation.rs` reading its
  `orientations` endpoint columns inline (it used `AtomGroups::from_frame`;
  deleted with that file in 06).
- `require_dist_sq` now guards a `&dyn PairSource` (returns `()`; the column
  arrives through `visit_pairs`); `Rdf`, `CorrelationFunction` and
  `LocalDensity` call it.

*As built (03):* `current_density` is gone (`Derivative { Backward }` of the
`DipoleMoment` series, over the volume; no NaN row) and the Einstein–Helfand
taper derivative runs on `Derivative` (`Central` inside, `Backward` on the last
two lags, `C′(0) = 0` by evenness). The one trapezoid recurrence is the
crate-internal `CumulativeTrapezoid::running(y, step)` (uniform or per-interval
step); the dielectric DC bin, the conductivity sum rule and `hbond_lifetimes`
call it. Two `0.5 * (` hits remain that are not trapezoid sums and are exempt
from the ac-005 grep: the Kramers–Kronig principal-value quadrature weight
`½(ω_{j+1} − ω_{j−1})` and test-only hand-written references. The one
prefactor implementation is `dielectric::DielectricPrefactor` (`fluctuation`
= 4πκ/(3Vk_BT), `current_density` = V/(3k_BT), `total_current` =
1/(3Vk_BT), `inverse_permittivity` = 4πκ), built from the compute-time V, T and
the configured `UnitPreset`; `StaticDielectric`, the four routes and
`ConductivitySumRule` all use it. The fs/ps contradiction is resolved by
stating that no prefactor carries a time unit: ω is in radians per unit of the
lag spacing, any consistent time unit works.

*As built (06):*
- **`BondAverage<B: BondBasis>`** (`order/bond_average.rs`, crate-internal,
  itself a `Compute` over `&Neighbors`) is the one `q_ℓm` / `ψ_k`
  implementation: `BondBasis` (`order`, `n_components`, `planar`,
  `evaluate(disp, out) -> bool`, provided `parity`) with
  `SphericalHarmonic { l }` (core `ylm_all`; `SphericalHarmonic::angles(disp)`
  is the one bond→(θ, φ) conversion, also used by BOO) and
  `PlanarHarmonic { k }` (`e^{ikθ}` via `Complex::from_polar` — not `Y_lm`,
  which would need a normalisation and move ψ_k by ULPs). The point count is
  `require_self_query`'s `SelfQuery { n_points }` (the guard now returns it);
  a pair index past it is `OutOfRange` (was a panic). The Lechner–Dellago
  average is `QlmTable::neighbor_average` (crate-internal). Summation order
  is the old one, so every order golden holds bit-for-bit or within its
  window.
- **PMFT** binning goes through `Axis::index` into an integer `ArrayD<u64>`,
  not `Histogram<N>`: `Histogram`'s `N` is a const generic and a coordinate's
  `DIM` is an associated const (`Histogram<{ C::DIM }>` needs the unstable
  `generic_const_exprs`), and PMFT counts are integers (`raw_counts` goldens
  are `u64`). The bin rule is still `Axis`'s alone.
- `require.rs` also holds the orientation guards `require_quaternions`
  (`N × 4`, normalised), `require_directions` (`N × 3`, normalised) and
  `require_angles` (finite), and `is_half_shell(mode)` replacing the inline
  `matches!(…SelfQuery…)` copies in BOO, EnvironmentMatch and PMFT.
- `EnvironmentMatch`'s `Dsu` is gone: matches are edges of a
  `Topology::with_atoms(n)` (+ `add_bonds`), labelled by
  `connected_components` (same labels: components in order of their lowest
  particle). Memory is now O(matches) instead of O(n) — a dense bucket of
  identical environments stores its match edges.

*As built (07):*
- **`GroupReduce`** (`shape/group_reduce.rs`, `pub(crate) mod`, so link 08's
  Voronoi reference can reach it) is built once per snapshot from
  `(positions N × 3, Option<&SimBox>, &dyn Partition)`: groups in ascending
  key order through `Partition::sizes`, items with `group_of = None` skipped,
  each group's anchor its lowest member, each member stored as
  `SimBox::mic().apply(r − anchor)` (the core MIC). `centroids(weights,
  name)` → centres `anchor + Σ w d / Σ w`, totals; `second_moments(weights,
  centroids)` → unnormalised `S = Σ w s sᵀ` with `s = d − (centre − anchor)`:
  the moment is taken on the already-unwrapped offsets, not through a second
  minimum image about the centre as the old copies did (identical for a
  group within half a box of its anchor; round-off only). Gyration is `S / W`,
  inertia `tr(S) δ − S`. Link 08 reads the Voronoi Pass-A centre of nuclear
  charge as `centroids(Z)`; a per-item `unwrapped(item)` accessor is not
  added until a caller needs it.
- **Connected components:** `Cluster` builds a `Topology::with_atoms(n)` +
  `add_bonds` from the pairs (and from each `merge` group, joined as a star
  on its first member) and labels with `connected_components` — the same
  labels as the old BFS (components in order of their lowest particle).
- **Eigensolver:** `Pca` calls `op::linalg::eigh_sym` (already
  `pub(crate)`); `power_iteration`, `rayleigh_quotient`, `mat_vec`,
  `vec_norm` and the `pub(crate) fn fit_transform` are deleted.
- **`positions.rs` deleted here, not in 09:** after 07 nothing outside a test
  read it (the cluster/shape kernels were its last users), so it and the
  test-only `fixtures::nlist_from_frame` would be dead code under
  `clippy -D warnings`.

*As built (08):*
- **Cell-list neighbour search (Voronoi rows).** `radical.rs`'s `CellGrid` is
  gone: the periodic copies within the search radius come from
  `core::GhostSet::borders` (every image, not the minimum image — a small box
  needs several images of one generator, which `NeighborList` /
  `NeighborQuery` cannot report), and a `core::LinkCell` over the open box that
  holds `[generators | copies]` is walked per generator through the
  crate-internal `LinkCell::visit_neighbors_of_pt` (streaming; a materialised
  `NeighborQuery::query` table would hold every generator's ~250 candidates at
  once). Candidates sort by `(d², search-point index)`, a total order.
  `integrate.rs`'s `CellList` is gone: `PowerCellLocator` runs the same
  pruning guard + exhaustive fallback on a `LinkCell` in the caller's box, so
  the voxel owner stays bit-identical to brute force (the moved
  `cell_list_nearest_matches_brute_force`). A before/after capture of the old
  code on five tessellations (random polydisperse with generators outside the
  box, one atom, two atoms, an SC lattice, 400 random) is bit-identical; the
  integrated moments agree to 1e-14 relative (centroid through `GroupReduce`,
  voxel volume through `RegularGrid`'s determinant, rayon reassociation).
- **Grid.** `DensityGrid` → `RegularGrid<3>` + a `density: ArrayView3` shaped
  as `grid.dims()`. `RegularGrid` gains `voxel_corner(index)` — the lattice
  node point-sampled data (a Gaussian cube) sits on; `voxel_center` is the same
  private `lattice_point(index, t)` at `t = ½`, same arithmetic.
- **Group reduction.** `VoronoiIntegration`'s Pass-A centre of nuclear charge
  is `GroupReduce::centroids(Z)`; `GroupReduce` gains the crate-internal
  `group_of(item) -> Option<usize>` (the row of an item's group), which Pass B
  needs to credit a voxel to its owner's molecule.

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

*As built (02):* all of the above are deleted, along with `transport/` and
`msd/`. `EinsteinDiffusionResult::diffusion_coefficient` (a façade) went with
its type; `D = slope / 2d` is `Msd` → `LinearFit`. `DebyeFit` moved to
`fitting/` in 02 (not 03) because `transport/` is gone; `EwaldBoundary` moved to
`dielectric/` unchanged (its `from_name` → `named_enum!` is 03's).
`LegendreReorientation*` → `ReorientationAcf` and `RotationalAutocorrelation*`
→ the surface landed in 02. `hbond_lifetimes` now runs on `Survival`.

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

*As built (03):*
- **Fits** take `{ correlation: &TimeCorrelation, channel }`; `LinearFit::new(start, end)` /
  `Plateau::new(start, end)` validate `0 ≤ start < end ≤ 1`; `DebyeFit` (unit struct) reads
  the lag times off the curve. `CumulativeTrapezoid::new().with_n_lags(m)` returns a
  `TimeCorrelation` (the running integral on the input lag grid), so
  `CumulativeTrapezoid → Plateau` chains with no glue; `CumulativeTrapezoidResult` is deleted.
- **`Window` default** is `CosineSq` (behaviour change 4's "GreenKubo window, chosen once in 03").
- **`RamanSpectrum`**: `new(window, pad)` (`Default` = cos², 4×), `with_incident_frequency(ν̃₀)`
  (`Result`; replaces the 0.0 sentinel), `with_averaged(bool)`; `compute({ isotropic, anisotropic:
  &TimeCorrelation, temperature: Option<F> })`. *Deviation:* the brief put temperature in `new`;
  it is `compute` data per D2 (a state's temperature). `None` = no Bose factor; `Some(T ≤ 0)` is
  `OutOfRange`. `RamanSpectrumResult.n_frames` (actually the lag count) is dropped.
- **`PolarizabilityParts`**: `compute({ alpha: T × 6 Voigt })` → `{ isotropic: T × 1, anisotropic: T × 6 }`;
  the anisotropic columns carry `√½` / `√3`, so their squared norm is γ² and the Raman anisotropic
  correlation is a plain `Correlator { Product }` summed over components. `RoaSpectrum` /
  `ResonanceRamanSpectrum` are `RamanSpectrum` on cross / resonant correlations (documented).
- **ε(ω) routes**: `new(window, units: UnitPreset)` + `with_pad` (default 4) + `with_epsilon_inf`
  (default 1); `compute(DielectricSpectrumInput { correlation, volume, temperature, boundary })`
  on a single-channel correlation (no channel index); `dt` is read off the lag grid; the EH DC bin
  uses `C(0)` (the old `zero_lag_variance` field). `DielectricSpectrumResult` gains `epsilon_inf`.
  `DipoleAutocorrelationSpectrum::with_plateau(Plateau)` replaces `subtract_plateau: Option<bool>`
  and the auto `|C∞/C0| > 0.1` trigger; it subtracts the plateau's windowed mean (was: the last
  sample). No plateau = no subtraction.
- **`EwaldBoundary`** is a `named_enum!` (`tin_foil`, `vacuum`). *Deviation:* no route ever used the
  boundary numerically (all formulas are tin-foil); `Vacuum` is now `OutOfRange` in every
  permittivity compute instead of being silently treated as tin-foil. Implementing the Neumann
  vacuum correction is new physics, out of scope.
- **Checks**: `KramersKronig::new().with_tolerance(rel, abs)` (0.1 / 0.01) on
  `{ spectrum: &DielectricSpectrumResult }`; `RouteAgreement::new().with_part(PermittivityPart)
  .with_tolerance(t)` (`imaginary`, 0.10) on `{ spectra: &[&DielectricSpectrumResult] }`, pairwise
  `(i, j, rms)`, one frequency grid required (`BadShape`); `ConductivitySumRule::new(units)
  .with_tolerance(t)` (0.05) on `SumRuleInput { frequencies, conductivity, current_sq_mean, volume,
  temperature }` — σ(ω) is not part of `DielectricSpectrumResult`, so it is data, not the result.
- **`DipoleMoment`** → `DipoleMomentResult { groups: Vec<u64>, moments: T × G × 3 }` (`groups`
  empty without a partition) + `series(column) -> T × 3`; `moments` feeds `Msd` / `Correlator` directly.
- **`StaticDielectric::new(units).with_epsilon_inf(ε∞)`**, `compute({ dipoles, volume, temperature,
  boundary })` → `StaticDielectricResult { epsilon, epsilon_axes, dipole_mean, dipole_sq_mean,
  fluctuation, epsilon_inf, n_frames }` (`eps` → `epsilon_axes`, `eps_mean` dropped: `epsilon` is
  the isotropic value).
- **`KineticEnergy::new(units) -> Result`** applies the preset's `mass·velocity² → energy` factor
  (LAMMPS `mvv2e`, through `UnitRegistry`; exactly 1 in `si`) so `KineticEnergyResult { energy,
  boltzmann }::temperature(n_dof)` is in the preset's temperature unit.
- **`center_of_mass_velocity` is kept until 07** (`md::maxwell` needs it and `CenterOfMass` still
  reads frames); its zero-total-mass fallback is now `OutOfRange`. 07's `CenterOfMass` on velocities
  with no box replaces it and deletes the function.

**Dynamics and H-bonds.**

| Symbol | Fate |
|---|---|
| `pair_survival_tcf`, `PairSurvivalResult`, `SurvivalMethod` | → `PairPresence` → `Survival` surface |
| `VanHove` | K. It uses `RadialAxis` / `Histogram` and `Origins`, and holds a `core::NeighborQuery`: its pair set is per (origin, lag), so a caller cannot supply it. |
| `HBonds`, `HBond`, `HBondCriterion`, `HBondDistanceKind` | K. The input is per-frame cross-query `&Neighbors` plus the donor/acceptor selection; `+1.05` Å becomes a criterion field. |
| `presence_from_hbonds` | M `HBondsResult::presence` |
| `hbond_components`, `HBondNetworkResult` | M `HBondsResult::components(frame)`, returning a label array that implements `Partition` |
| `hbond_lifetimes`, `HBondLifetimeResult` | → `Survival` surface on `HBondsResult::presence()` + `CumulativeTrapezoid` |

*As built (05):*
- **`Presence { pairs: Vec<(u32, u32)>, values: T × n_pairs }`** (`dynamics/presence.rs`),
  columns in ascending pair order, never empty; `values.view()` is `SurvivalInput::presence`.
- **`PairPresence::new(r0, exclude_self) -> Result`**, `with_r1(r1) -> Result` (`r1 ≥ r0`, default
  `r0`); `compute(PairPresenceInput { frames: &[&(dyn PairSource + Sync)] })` → `Option<Presence>`
  (`None` = no pair ever within `r0`). One table per frame from core search (any box shape), with
  `dist_sq` and a cutoff ≥ `r1` (a pair the search did not report is absent — documented, not
  checkable); all frames one `QueryMode` (`BadShape`). `exclude_self` has no default (behaviour
  change 4) and is `OutOfRange` unless the tables are a cross-query with equal point counts.
  *Deviation:* the survival curve is now per pair (`Survival`'s mean over pairs), not per reference
  particle: `C(0) = ⟨h⟩` instead of the coordination number; and an SSP origin counts whenever the
  pair is in the bonded stable state (inside `r1` after entering `r0`), not only when `d ≤ r0` at
  that origin. Two `pair_survival_tcf` assertions moved with this (see Behaviour changes, 05 note).
- **`HBonds::new(criterion) -> Result`** (validates the criterion), `Default`;
  `compute(HBondsInput { positions: T × N × 3, donors: &[(D, H)], acceptors: &[A], pairs:
  &[&Neighbors] })` → `HBondsResult { per_frame, counts, n_atoms }`. Each table is the caller's
  cross-query of the donor sites (D or H per `dist_kind`, `donors` order) against the acceptors;
  the kernel gates on the table's own `dist_sq ≤ dist_cutoff²` (a wider search does not widen the
  criterion). *Deviation:* the box is not a separate argument — it is each table's
  `Neighbors::simbox()` (D5), so it cannot disagree with the search; a table without one is
  `BadShape`. The input is the whole trajectory (not per snapshot through `Batch`), because
  `presence()` spans frames.
- **`HBondCriterion`** gains `donor_acceptor_margin` (default 1.05 Å, the former `+ 1.05`) and
  `with_donor_acceptor_margin`.
- **`HBondsResult::presence() -> Option<Presence>`** keyed `(donor, acceptor)`, ascending (was
  first appearance); `None` when no bond formed in any frame — the §5 "explicit Option" lives here,
  upstream of `Survival` → `CumulativeTrapezoid`. Lifetimes are `∫ S / S(0)`.
- **`HBondsResult::components(frame) -> Result<Vec<u32>>`**: a label per atom (a `Partition`), on
  `Topology::connected_components`; each bond joins donor–hydrogen–acceptor. *Deviation:* the old
  function took caller-defined nodes/edges and returned sorted sizes; sizes are now
  `Partition`-derived.
- **`VanHove::new(axis: Axis, lags) -> Result`** (axis must start at `r ≥ 0`; no separate
  `RadialAxis` type — `Axis::shell_volumes(3)` is the radial part), `with_origins(Origins) ->
  Result`, `with_stride(n) -> Result` (`Origins::First` with a stride ≠ 1 is `OutOfRange`);
  `compute(VanHoveInput { positions: T × N × 3, boxes: &[SimBox] })` → `VanHoveResult` (no
  `has_distinct`). Bins through `Histogram<1>` (half-open), one core `NeighborQuery` per origin
  in that origin's box. *Additions:* a wrapped trajectory is `Wrapped` (D9, the shared guard
  `time_series::guards::first_wrapped_item`, now `pub(crate)`); a lag whose displacements all
  fall outside the axis is `EmptyInput` (`Histogram::density`, was a zero row).

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

*As built (04):*
- **`Rdf`**: `new(n_bins, r_max, r_min) -> Result` (builds the `Axis`;
  `r_min < 0` is `OutOfRange`), `with_dimensionality(2 | 3) -> Result`;
  `Compute { Input = &[&(dyn PairSource + Sync)] }` (one source per frame),
  body `self.accumulator().fold(..)`; `Stream { Accumulator = RdfAccumulator }`
  whose constructor is private. The accumulator latches the pairing from frame
  0 (`QueryMode`), requires each source's `simbox()` (`None` is `BadShape`),
  validates the volume once per push, bins into a fresh `Histogram<1>` and
  merges only on success. `RdfResult` keeps its fields minus `finalized` and
  `mode`, plus `self_query: bool` (what `RdfMode` recorded); it keeps
  `DescriptorRow` until 09 (PCA still reads it). *Deviation:* the
  `(n_bins, r_max, r_min)` form was kept over an `Axis` argument (freud order,
  binding-friendly); `Rdf::axis()` exposes the `Axis`.
- **`CorrelationFunction`**: `new(n_bins, r_max, r_min)`, `compute({ pairs:
  &(dyn PairSource + Sync), values_a, values_b: ArrayView1 })`; the fields must
  have one value per query / reference point (`DimensionMismatch`; it used to
  fail only when a pair indexed past them). **`LocalDensity`**: `new(r_max)`,
  `with_diameter(d) -> Result` (was `debug_assert`), `compute({ pairs })`; the
  per-point count is the source's `n_query_points` (was the frame's atom
  count). Both are per-snapshot; many snapshots go through `Batch`.
- **`DistributionFunction<O>`**: `new(observable, axis)`,
  `over_natural_range(observable, n_bins) -> Result`; a `Stream` with
  `Input = &[DistributionFrame { positions: N×3, simbox: Option<&SimBox>,
  tuples: n×arity usize }]` (`None` box = open boundaries, minimum image
  otherwise) and `DistributionAccumulator<O>`. `DistributionResult {
  histogram: Histogram<1>, density, density_sin_corrected, n_raw_samples,
  n_frames }`. **`CombinedDistribution<const N>`** (N ∈ {2, 3}, else
  `OutOfRange`): `new([InternalCoordinate; N], [Axis; N]) -> Result`,
  `with_sin_weight(axis) -> Result`; per frame `CombinedDistributionFrame {
  positions, simbox, tuples: [ArrayView2<usize>; N] }`; result `{ histogram:
  Histogram<N>, density: ArrayD, n_raw_samples, n_frames, sin_corrected }` with
  `marginal(axis) -> Result<DistributionResult>` and `free_energy(T)`.
  `Observable::sample_into(positions, &Mic, tuples, out)` (no frame, no
  generic — the trait is now dyn-compatible); every observable checks arity
  and indices (`OutOfRange` instead of an index panic).
  `InternalCoordinate` is a `named_enum!` (`distance`, `angle`, `dihedral`)
  implementing `Observable` by dispatch. *Deviation:* the three trajectory
  histograms (DF, CDF, SDF) are `Stream`s whose fold is serial — the per-frame
  rayon `try_fold` paths are gone (ac-007 fixes the body); the CIC sums were
  already order-tolerant, and integer counts are unchanged.
- **`StaticStructureFactorDebye`**: `new(&k)` / `linspace(..)`,
  `with_sinc_tolerance(t) -> Result` (the former `1e-9`,
  `DEFAULT_SINC_TOLERANCE`), `compute({ positions })` — all pairs, open
  system, stated in its doc. **`StaticStructureFactorDirect`**: `new(&k_vecs)`
  / `isotropic(k_max, n_bins)` (an `Axis`), `compute({ positions, simbox:
  Option<&SimBox> })`; the isotropic form without a box is `BadShape`.
- **`DiffractionPattern`**: `new(n_grid, σ)`, `with_axis(a) -> Result` (was
  `assert!`), `with_cutoff(r) -> Result` (the former `3σ`), `compute({
  positions, simbox })`; the image is a `RegularGrid<2>` over the box face.
- **`GaussianDensity`** `new(dims: [usize; 3], σ)`, `with_r_max(r) -> Result`
  (was `debug_assert`); **`SphereVoxelization`** `new(dims, r_max)`; both
  `compute({ positions, simbox })` and return the `grid: RegularGrid<3>` they
  used. *Deviation:* §4 says "`RegularGrid` in `new`" for these two; their grid
  *is* the box divided into `dims` voxels, and the box is per-snapshot data
  (D2), so `new` takes `dims` and `compute` builds
  `RegularGrid::over_box(simbox, dims)`. Triclinic boxes stay `OutOfRange`
  (unchanged; `RegularGrid` itself supports them).
- **`SpatialDistribution`**: `new(grid: RegularGrid<3>)` (`GridSpec { n,
  extent }` → `RegularGrid::centered(extent, n)`), `with_bulk_density(ρ) ->
  Result` (a non-positive ρ used to be ignored silently); a `Stream` over
  `SpatialDistributionFrame { positions, simbox, reference, template, target,
  orientation: Option<&[(head, tail)]> }` — the Link 01 audit row. Orientation
  pairs are `(head, tail)`, axis = head − tail (minimum image): the slot order
  of behaviour change 5, numbers unchanged. Orientation pairs must be present
  on every frame or none (`BadShape`). The result carries `grid` instead of
  `n` / `extent`.

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

*As built (06):* every type below is a per-snapshot `Compute` (many
snapshots: `Batch`); `Input`s are named-field structs; no frame is read.
- **`QlmTable`** (public): `order()`, `is_planar()`, `n_particles()`,
  `values()` (`N × n_components` `Complex`, column `m + ℓ`), `norms()`,
  `ql()` (`|ψ_k|` on a planar table), `wl(normalize) -> Result` (`OutOfRange`
  on a planar table), `bond_coherence(&Neighbors) -> Result<Array1>`
  (`Re Σ_m conj(q_i) q_j` per pair; `DimensionMismatch` for another point
  set).
- **`Steinhardt::new(&[l])`** (+ `with_average/with_wl/with_wl_normalize`),
  `compute(SteinhardtInput { neighbors })` → `SteinhardtResult { l, tables:
  Vec<QlmTable>, .. }` with `qlm(l) -> Option<&QlmTable>`, `ql()` /
  `wl()` (`N × n_l` views; was `Vec<Vec>` indexed `[l][particle]`) and
  `FeatureMatrix` (rows = particles, columns = `q_ℓ` then `w_ℓ`).
- **`SolidLiquid::new()`** (`Default`; *deviation:* `l` leaves the
  configuration — it is the table's), `with_q_threshold(t) -> Result` (NaN is
  `OutOfRange`), `with_n_threshold`, `with_normalize_q`;
  `compute(SolidLiquidInput { qlm, neighbors })`.
  **`ContinuousCoordination::new(power)`** — *deviation:* the `l: &[u32]` list
  (Link 01 audit: "stays in `new`") is gone, because the input is one
  `(&QlmTable, &Neighbors)`; several ℓ are several calls. Result `{ l, coord:
  Vec<F> }` (was `Vec<Vec<F>>` per ℓ).
- **`Hexatic::new(k)`**, `compute(HexaticInput { neighbors })` →
  `HexaticResult { k, psi }` (the planar `BondAverage`).
- **`Nematic`**, `compute(NematicInput { directors: N × 3 })`;
  **`Cubatic::new()`** with `with_seed`, and `with_initial_temp`,
  `with_cooling_rate` (`0 < r < 1`), `with_min_temp` (the former `1e-6`),
  `with_n_steps`, `with_n_chains` all `-> Result`; `compute(CubaticInput {
  directors })`. Its rotations are `op::so3::unit_vector_from_uniform` +
  `op::rigid::axis_angle` (same RNG draw order).
- **`BondOrientationalOrder::new(n_θ, n_φ)`** (holds its two `Axis`),
  `compute(BondOrientationalOrderInput { neighbors })`.
  **`LocalDescriptors::new(l_max)`**, `compute(LocalDescriptorsInput {
  neighbors })` → `LocalDescriptorsResult { l_max, n_sphs, .. }` with
  `descriptor(pair, l, m) -> Option<Complex>` and `FeatureMatrix` (one row per
  pair, `(Re, Im)` interleaved, `2·n_sphs` columns; the flat `Vec<Complex>`
  field is gone). **`LocalBondProjection`** (unit struct),
  `compute(LocalBondProjectionInput { neighbors, proj_vectors: n × 3,
  orientations: Option<N_q × 4> })`.
- **`EnvironmentMatch::new(rmsd)`**, `with_registration`,
  `with_max_neighbors_for_registration(n) -> Result` (default 8, 0 is
  `OutOfRange`), `compute(EnvironmentMatchInput { neighbors })`; the result
  is a `Partition`.
- **`AngularSeparationGlobal`** `compute({ query: N × 4, global: M × 4 })`,
  **`AngularSeparationNeighbor`** `compute({ neighbors, query_orientations,
  point_orientations })` (one per point of each set, `DimensionMismatch`
  otherwise). `angular_distance` is a private helper of the module.
- **`Pmft<C: PairCoordinate>`** with `R12`, `Xy`, `Xyt`, `Xyz`; one inherent
  `new` per instantiation taking the old scalars (`Pmft::<Xy>::new(x_max,
  y_max, n_x, n_y)`, …); `compute(PmftInput { pairs: &Neighbors,
  orientations: Option<C::Orientations> })` → `PmftResult<C> { density,
  raw_counts, pmf: Array<_, C::Dim>, edges: Vec<Array1> }`. `PairCoordinate`:
  `Orientation` (`F` / `Quat`), `Orientations<'a>` (`ArrayView1` angles /
  `ArrayView2` `N × 4`), `Dim`, `DIM`, `LAB_FRAME` (`Some` for `Xy`/`Xyz`,
  `None` → `BadShape` for `R12`/`Xyt`), `orientations`, `axes`, `coords(disp,
  oi, oj, out)`, `ref_measure(box)`, `bin_measure(center)` (default: product
  of widths). *Deviations:* (a) per snapshot, not per trajectory — the "frame
  loop" that exists once is `Batch`; (b) the box is the table's
  `Neighbors::simbox()` (`None` is `BadShape`; was the frame's); (c) one
  orientation array indexes both `i` and `j`, so it must have one entry per
  point of each set (`DimensionMismatch`); (d) half-open `Axis` bins now
  include `x = −x_max` (measure zero; no golden moved); a wrapped angle that
  rounds to `2π` is `0`; (e) **bug fixed:** `PmftR12`'s bin measure was
  `2π r_c dθ₁ dθ₂` (the `dr` factor missing, so an ideal gas had
  `PMF = −ln dr`); it is `2π r_c dr dθ₁ dθ₂`, pinned by
  `bin_measures_tile_the_disc`. No golden covered R12's density.

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

*As built (07):* every type below is a per-snapshot `Compute` (many
snapshots: `Batch`); `Input`s are named-field structs whose partition /
features are `&(dyn Partition + Sync)` / `&(dyn FeatureMatrix + Sync)` so
`Batch` can send them.
- **`Partition::sizes() -> BTreeMap<u64, usize>`** (provided method): group
  sizes keyed — and iterated — by ascending key; the "group sizes" of the
  former `ClusterProperties`.
- **`CenterOfMass`**, **`GyrationTensor`**, **`InertiaTensor`**: unit structs
  (`new()`, `Default`); `compute({ positions: N × 3, simbox: Option<&SimBox>,
  partition, masses | weights: ArrayView1 })` — `GyrationTensor` takes
  `weights`, the other two `masses`. Results: `CenterOfMassResult { groups:
  Vec<u64>, centers: G × 3, total_masses }`, `GyrationTensorResult { groups,
  tensors: G × 3 × 3 }` with `radius_of_gyration() -> Array1` (`√tr G`, no
  clamp: `G` is a non-negative-weighted sum of squares), `InertiaTensorResult
  { groups, tensors }`. *Deviation:* `GyrationTensor` / `InertiaTensor` no
  longer take an upstream centres result (`ClusterCentersResult` /
  `CenterOfMassResult`); each reduces about its own weighted centre through
  `GroupReduce`, which is what keeps the centre and the moment one weighting.
  The geometric centre is `CenterOfMass` with `Array1::ones(n)`; the
  centre-of-mass velocity is `CenterOfMass` on velocities, `simbox: None`,
  a one-group partition (`md::maxwell` does this).
- **`Cluster::new()`** (minimum size 1, `Default`) +
  **`with_min_cluster_size(n) -> Result`** (0 is `OutOfRange`). *Deviation:*
  the old `new(min_cluster_size)` positional argument became a setter so the
  default lives in core (D8). `compute(ClusterInput { pairs: &Neighbors,
  merge: Option<&(dyn Partition + Sync)> })`; the particle count is the
  table's `SelfQuery { n_points }` (`require_self_query`; a cross-query is
  `BadShape`, a pair past the count `OutOfRange` — was an index panic).
  *Deviation (freud semantics):* `merge` joins its groups **on top of** the
  pairs, like freud's `keys`; the old `compute_keyed` ignored geometry, which
  is now `merge` over a table with no pairs (the keyed tests do exactly
  that). `ClusterResult { cluster_idx: Array1<i64> (−1 = dropped), n_clusters,
  cluster_sizes, cluster_keys: Vec<Vec<u64>> }` (keys ascending and distinct
  per cluster, empty without `merge`), a `Partition` (`−1` → `None`).
- **`Pca::new(n_components) -> Result`** (0 is `OutOfRange`),
  `compute(PcaInput { features })` → `PcaResult { coords: n_rows ×
  n_components, variance: Array1 }`, a `FeatureMatrix`. Fewer columns than
  components is `OutOfRange`; no rows `EmptyInput`. Each axis's sign is fixed
  so its largest-magnitude loading is positive (the old power iteration's
  sign followed its start vector).
- **`Kmeans::new(k) -> Result`**, `with_max_iter(n) -> Result` (0 is
  `OutOfRange`), `with_seed(s)`; defaults `Kmeans::DEFAULT_MAX_ITER = 100`
  and `Kmeans::DEFAULT_SEED = 0` — the values Python and WASM already used,
  so behaviour change 4 moves no visible default. `compute(KmeansInput {
  features })` in the matrix's own `n_cols`; `KmeansResult { labels:
  Vec<u32>, centroids: k × n_cols }` (was `KmeansResult(Vec<i32>)`), a
  `Partition`. The k-means++ draw order is unchanged.
- `DescriptorRow` is no longer read by `Pca` / `Kmeans`; its last
  implementor is `RdfResult` (09 deletes the trait).

**Voronoi.**

| Symbol | Fate |
|---|---|
| `RadicalVoronoi` | K as `Compute`: `compute(positions, simbox, radii)` → `VoronoiCells`, on `core::NeighborList` |
| `VoronoiIntegration` | K as `Compute`: named `Input` struct instead of 7 positional arguments; `LengthUnit` in `new`, with no Bohr fallback |
| `VoronoiDomainAnalysis`, `VoronoiVoidAnalysis` | K as `Compute`; `void` already uses `connected_components` |
| `DensityGrid` | → `RegularGrid` + values array as `compute` data |
| `MolecularMoments`, `VoronoiCells`, `VoronoiFace`, `VORONOI_BOUNDARY` | K |
| `polarizability_finite_field` | → `FiniteFieldPolarizability`; the unused `moments_zero` argument is deleted |

*As built (08):* every type below is a per-snapshot `Compute` (many
snapshots: `Batch`) with a named `Input`; no frame is read.
- **`RadicalVoronoi::new()`** (unit struct, `Default`),
  `compute(RadicalVoronoiInput { positions: N × 3, simbox, radii: ArrayView1 })`
  → `VoronoiCells`.
- **`VoronoiIntegration::new(unit: LengthUnit)`** (no `Default`: the density
  unit must be stated), `compute(VoronoiIntegrationInput { positions, radii,
  nuclear_charges: ArrayView1<F>, partition: &(dyn Partition + Sync), simbox,
  grid: &RegularGrid<3>, density: ArrayView3 })` → `MolecularMoments`, which
  gains `groups: Vec<u64>` (rows are the partition's groups, ascending; was
  `0..n_mol`). **`LengthUnit`** is a `named_enum!` (`angstrom`, `bohr`): the
  density is electrons per cubic unit; positions and grid geometry are Å.
  *Deviations:* (a) the input carries no `VoronoiCells`: the integrator's
  cells are the power-distance argmin of `(positions, radii)`, exactly the
  cells `RadicalVoronoi` builds, so the tessellation would be unused data;
  (b) `atomic_numbers: &[i32]` → `nuclear_charges: ArrayView1<F>` (they are
  `GroupReduce` weights; a pseudopotential density's valence charges fit);
  (c) the minimum image is the box's own (`SimBox::mic`: periodicity and tilt
  honoured) — the old code forced `Mic::ortho(lengths)`, periodic on every
  axis even for a cube-read box (pbc off) and wrong for a tilted one; a
  periodic system read from a cube needs its box's pbc set by the caller;
  (d) `DensityGrid::from_cube_frame` is gone with the frame read: the caller
  builds `RegularGrid::over_box(&cube_box, dims)` (the cube reader's box is
  voxel axes × dims from the cube origin, so the voxel corners are the cube's
  sample points), takes the density column as a `dims`-shaped array, and
  picks the unit (`bohr` for a cube's volumetric block).
- **`VoronoiDomainAnalysis::new()`**, `compute(VoronoiDomainInput { cells,
  labels: &[i64] })`; **`VoronoiVoidAnalysis::new()`**,
  `compute(VoronoiVoidInput { cells, is_void: &[bool], box_volume })`.
- **`FiniteFieldPolarizability::new(field) -> Result`** (zero / non-finite is
  `OutOfRange`), `compute(FiniteFieldPolarizabilityInput { plus, minus:
  &MolecularMoments })` → `Array2<F>` (`n_mol × 3`, the column `∂μ/∂E_j`);
  runs over different molecules (`groups`) are `BadShape`.
- *Deviation (chain row):* the chain table named a `GridField` type; none was
  added — it would be a data wrapper the caller builds (D2). The grid is
  `RegularGrid` and the values are an `ArrayView3` in the `Input`.

*As built (09):*
- **`VoronoiDomainInput { cells, partition: &(dyn Partition + Sync) }`** (was
  `labels: &[i64]`), like every other grouping input: a label array or an
  upstream partition (`ClusterResult`, `KmeansResult`, …). A cell in no group
  (`group_of = None`) joins no domain and does not bridge its neighbours.
  `VoronoiDomainResult::domain_of` is `Vec<Option<usize>>` (contiguous ids in
  order of each domain's lowest cell), `largest_fraction` is over the grouped
  cells, no grouped cell is `EmptyInput`, and the result is itself a
  `Partition` (key = domain id). *Note:* the old signed `i64` labels are not a
  `Partition` (keys are unsigned); a caller with signed labels maps them.
- **`DipoleMomentInput::partition`** is `Option<&(dyn Partition + Sync)>` (was
  `Option<&dyn Partition>`), so it matches every other partition input and the
  input is `Send` for `Batch`.

After the chain, `compute` exports no `pub fn` at module level and no `pub type`
alias of another public type.

### 5. Silent fallbacks removed — each becomes an error or an explicit parameter

This is the exhaustive list from the review. Each item is fixed in the link
that owns the file.

- **Lag windows:**
  - `max_lag.min(n-1)` everywhere becomes `OutOfRange`, and the
    `clamps_not_panics` test (`vacf.rs:155`) is rewritten to expect the error.
    *As built (02):* done for `Correlator` and every surface (batch and
    stream). A series of fewer than two frames is `EmptyInput` (the old
    `signal::acf_fft` accepted one sample). The clamps left in the legacy
    `IrFlux` / `VcdCrossFlux` / `RamanTensor` / `RoaCrossTensor` (03),
    `hbond_lifetimes` and `pair_survival_tcf` (05) go with their owners.
  - VanHove: dropped lags, `stride.max(1)` and `n_orig.max(1)` become errors.
    With no `SimBox` it errors instead of returning zeros plus a flag.
- **Fit windows:** `LinearFit` widening and `Plateau` collapsing become errors.
- **Spectroscopy:**
  - `bose_factor` returns 1.0 for T ≤ 0; that becomes an error.
  - The Raman 0.0 "skip" sentinel becomes `Option`.
  - The dielectric auto-plateau magic `0.1` becomes an explicit `Plateau`
    input.
  - *As built (03):* all three done, plus the `max_lag` clamps of the deleted
    `IrFlux` / `VcdCrossFlux` / `RamanTensor` / `RoaCrossTensor`; `LinearFit`
    widening and `Plateau` collapsing are `OutOfRange` (`FractionalWindow`);
    `current_density`'s NaN row is gone with the function;
    `center_of_mass_velocity` with zero mass is `OutOfRange`; a vacuum
    `EwaldBoundary` is `OutOfRange` (was silently tin-foil).
- **Dielectric and kinetic:**
  - `current_density`'s NaN row 0 disappears, because the derivative stencil
    defines its edges.
  - `center_of_mass_velocity` with zero total mass becomes an error.
- **Masses and weights:** the shape kernels' `map_or(1.0, …)` disappears;
  masses are a required `compute` argument.
  *As built (07):* done in `GroupReduce`, with unit tests on each variant:
  weights not one per item → `DimensionMismatch`, non-finite → `NonFinite`,
  negative → `OutOfRange`, a group of total weight zero → `OutOfRange` (was a
  zero centre / zero tensor); the old zero-mass `OutOfRange` of
  `center_of_mass_velocity` is this.
- **Orientations:**
  - Zero quaternions get three different answers today; all three become
    errors.
  - `planar_orientation_angles`' unchecked z-axis assumption becomes an error
    for non-planar input.
- **Voronoi:**
  - The `.unwrap_or(false)` Bohr assumption becomes an explicit `LengthUnit`.
  - `dims.max(1)` and the out-of-cell clamps become errors.
  - *As built (08):* the unit is a required `VoronoiIntegration::new`
    argument (no `Default`). `dims.max(1)` went with the frame read; a zero
    dimension is `RegularGrid::new`'s `OutOfRange` and a density not shaped as
    the grid is `BadShape`. The out-of-cell clamps (`radical.rs:286,325`,
    `integrate.rs:427`) went with the private grids; their reachable trigger,
    a non-finite coordinate (NaN landed in cell 0), is `NonFinite` (positions,
    radii, nuclear charges, density values), each with a unit test. A point
    outside the box is wrapped (periodic axis) or binned to the edge cell
    (open axis) by the core `LinkCell` — complete, not a clamp. Also: a
    chargeless molecule (its reference fell back to its first atom) is
    `OutOfRange` (`GroupReduce`'s zero total); a generator in no molecule is
    `OutOfRange`; no generators is `EmptyInput` (was an index panic);
    `RadicalVoronoi` on a triclinic or not fully periodic box is `BadShape`
    (was tessellated as a periodic orthorhombic box of the lattice lengths);
    `VoronoiDomainAnalysis` on an empty tessellation is `EmptyInput`
    (`largest_fraction` was 0/0 → 0); a search radius needing more periodic
    copies than `ImageRange::MAX_IMAGES` is `OutOfRange`; the `leak()` label
    workaround is gone (static labels).
  - *As built (09) — the exhaustive fallback.* On inspection the 08 hand-off
    ("a cell still unbounded after three shells is reported as boundary
    faces") was not the real defect, so it was not made an error as worded:
    the initial polyhedron is `box_cell(L)` (half-widths `L`, twice the box)
    and a generator's own shell-1 images always cut it at `±L/2`, so with
    in-box generators no boundary face survives shell 1 and the three-shell
    loop was dead. It was reached only through a bug: the fallback offset the
    *raw* positions while its generator was *wrapped*, so a generator given
    outside the box missed its own shell-1 image (residual face → more
    shells). The real silent fallback was that the fallback returned at the
    first shell with no box face, **without proving** that no farther image
    could cut — unlike the fast path, which is accepted only once `r_cut ≥
    d_stop`. Fixed: the fast path now reports the termination bound `reach =
    d_stop(R_max)` of the cell it clipped (which contains the true cell); the
    fallback takes every image within `reach` of every **wrapped** generator
    from the core `ImageRange::new(cell, reach)` (half-width `⌈reach/L⌉` per
    axis, complete for wrapped points) and clips them through the same
    nearest-first loop as the fast path (`clip_nearest_first`, one
    implementation). It is complete by the same bound, so no cell can be
    unbounded; a reach needing more images than `ImageRange::MAX_IMAGES` is
    `OutOfRange { field: "RadicalVoronoi exhaustive reach" }` instead of a cell
    from a truncated image set (`MAX_SHELL`, `FACE_EPS` and the residual-face
    test are gone). Tests: `sparse_cells_are_exact_on_the_exhaustive_path`
    (4 × 4 × 40 box, two stacked generators: the fast path provably fails, the
    cells are the exact 320 Å³ columns with no boundary face, bit-identical when
    the generators are given whole boxes away) and
    `an_exhaustive_reach_beyond_the_image_limit_is_an_error` (2 × 2 × 200).
    Behaviour: a fallback cell now clips in distance order rather than
    generator order (round-off only; every Voronoi test passes unchanged), and
    a pathologically elongated sparse box (aspect ≳ 30:1 with a near-empty
    column) that the old code tessellated without proof is now `OutOfRange`.
- **EnvironmentMatch:**
  - The silent fallback above 8 neighbours becomes an explicit-limit error.
  - `Err(_) => INFINITY` becomes error propagation.
- **PCA:**
  - An unconverged result becomes `NotConverged`; this disappears entirely once
    PCA uses `eigh_sym`.
  - Negative eigenvalues are no longer clamped with `max(0.0)`.
  - *As built (07):* both done: `eigh_sym` is a full Jacobi
    decomposition with no convergence flag to drop, and `variance` is the
    eigenvalues as computed (`a_rank_one_matrix_has_one_axis` pins a zero
    eigenvalue reported unclamped).
- **Histograms:**
  - `debug_assert`-only validation becomes `Axis::new` returning `Result`.
  - A NaN sample becomes `NonFinite`.
  - An empty `density()` becomes an error rather than zeros.
- **Pair survival and H-bonds:**
  - An `exclude_self` that is silently ignored becomes an error.
  - An empty donor/acceptor selection becomes `EmptyInput`.
  - Lifetimes that are all zero because no bond ever formed become explicit
    `Option`s.
  - *As built (05):* all done, with VanHove's four items: `exclude_self` →
    `OutOfRange`; empty selection → `EmptyInput` (was an empty bond list);
    no bond ever → `HBondsResult::presence()` / `PairPresence` return `None`;
    the `max_lag` clamps went with `pair_survival_tcf` / `hbond_lifetimes`
    (`Survival` already errors). VanHove: a lag `≥ T` → `OutOfRange` (was
    dropped), `with_stride(0)` → `OutOfRange` (was `max(1)`), `n_orig.max(1)`
    is gone (every lag has origin 0), no box → the boxes are required data
    (`DimensionMismatch` when not one per frame; was zeros + a flag). Also: an
    `HBondCriterion` out of range is `OutOfRange` in `HBonds::new`; non-finite
    positions are `NonFinite` in `HBonds` / `VanHove`.
- **Cubatic:** `with_n_chains(n.max(1))` becomes an error.
- *As built (06):* every orientation item is done with a unit test on its
  variant: a zero quaternion is `OutOfRange` in `AngularSeparation*`
  (was distance 0), `LocalBondProjection` and `Pmft<Xyz>` (was the
  identity); a zero director is `OutOfRange` in `Nematic` / `Cubatic` (was
  skipped) and a zero reference direction in `LocalBondProjection` (was a
  zero projection). `planar_orientation_angles` is deleted, so its z-axis
  assumption went with it (callers pass angles). EnvironmentMatch: above the
  registration limit is `OutOfRange` (was a silent switch to magnitude-only
  matching); a superposition failure propagates as `OutOfRange` (was
  `INFINITY`); a non-finite bond is `NonFinite` (was a sort panic). Cubatic:
  `with_n_chains(0)` and every schedule constant out of range are
  `OutOfRange`. Also: `R12`/`Xyt` without orientations are `BadShape`; a pair
  index past a table's point count is `OutOfRange` in the bond average,
  EnvironmentMatch, AngularSeparationNeighbor and PMFT (were panics).
- **Panics:** `Rdf::with_dimensionality` `assert!`, `DiffractionPattern::with_axis`
  `assert!` and the `SpatialDistribution` unchecked index panic all become
  errors. The silent 3-D fallback at `rdf/result.rs:114` is removed.
- *As built (04):* every histogram and panic item above is done, each with a
  unit test on its variant (`Axis::new` / `index`, `Histogram::density`,
  `Rdf::with_dimensionality`, `DiffractionPattern::with_axis`, the SDF index).
  Also: a NaN distance in `Rdf` / `CorrelationFunction` / `LocalDensity` and a
  NaN sample in the distributions are `NonFinite` (they reached bin 0); an
  observable's out-of-range atom index is `OutOfRange` (was a panic);
  `LocalDensity::with_diameter` and `GaussianDensity::with_r_max` validate
  (were `debug_assert`); `SpatialDistribution::with_bulk_density(ρ ≤ 0)` errors
  (was ignored); an `Rdf` over sources with no points is `EmptyInput` (was an
  all-zero g(r)); the sin θ-corrected density with no positive integral is
  `OutOfRange` (was zeros); `RegularGrid::index_of` on NaN is `NonFinite` (was
  voxel 0). `StaticStructureFactorDebye`/`Direct` on zero particles still
  return `S = 0` (pinned by `empty_frame_*` tests; not a §5 item).
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

### 7. Catalog (links 10a, 11)

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

*As built (10a, core only):* `compute/spec/` holds the descriptor. The
generated catalog (`computeSpecs()`, deleting `catalog.rs`, molvis) is still
link 11.

- **Types** (re-exported from `molrs::compute`):
  - `ComputeSpec { name, params: Vec<ParamSpec>, inputs: &'static [InputSpec], form: InputForm }`,
    with `param(name)` / `input(name)`. `InputForm` is `Single` (one `Input`
    struct) or `Frames` (a slice of per-frame records: `Rdf`,
    `DistributionFunction`, `CombinedDistribution`, `SpatialDistribution`).
  - `ParamSpec { name, kind: ParamKind, default: ParamDefault }`, `is_required()`.
    `name` is the constructor argument or the `with_*` setter suffix.
  - `ParamKind`: `Float`, `Int`, `Bool`, `Choice(NAMES)`, `ChoiceList(NAMES)`,
    `FloatList`, `IntList`, `Vec3List`, `Axis`, `AxisList`, `Grid`, `Range`,
    `Units`. (*10b:* `LagKernel` removed — the kernel is a `Choice`.)
  - `ParamValue`: `Float`, `Int(u64)`, `Bool`, `Name(&'static str)`,
    `FloatList`, `IntList`, `Range(start, end)`, `Unset` (an option left off).
  - `ParamDefault`:
    - `Required`;
    - `RequiredIn(ctor)` for mutually exclusive constructor forms
      (`StaticStructureFactorDirect::new` / `::isotropic`,
      `DistributionFunction::new` / `::over_natural_range`);
    - `Value(v)`, with `ParamDefault::UNSET` = `Value(Unset)`;
    - `SameAs(param)` (`PairPresence.r1` = `r0`);
    - `Scaled { of, factor }` (`GaussianDensity.r_max` and
      `DiffractionPattern.cutoff` = `Gaussian::DEFAULT_CUTOFF_SIGMAS · sigma`).
  - `InputSpec { name, kind: InputKind, optional }`. `InputKind` names the
    data's role and shape: `Positions`, `Vectors`, `Trajectory`, `Series`,
    `Scalars`, `Scalar`, `Index`, `Indices`, `IndexPairs`, `Tuples`,
    `TuplesList`, `Mask`, `Images`, `Quaternions`, `Angles`, `SimBox`, `Boxes`,
    `Neighbors`, `NeighborFrames`, `PairSource`, `PairSourceFrames`,
    `Partition`, `FeatureMatrix`, `Grid`, `Field`, `Choice(NAMES)`,
    `Upstream(type)`, `UpstreamList(type)`.
  - All derive `Debug, Clone, PartialEq` and, under `serde`, `Serialize`.
- **Trait** `Described: Compute { fn spec() -> ComputeSpec; fn tunables(&self) -> Vec<(&'static str, ParamValue)>; }`.
  `tunables` reads the instance's value of every defaulted parameter, in spec
  order. On an instance built from the required arguments alone it equals the
  spec defaults. *Deviation from "`const SPEC: ComputeSpec`":* `spec()` is a
  function and `params` a `Vec`, because defaults are read from the type at
  run time (`Origins::default()` is not `const`).
- **Macro** `compute_spec!` (crate-internal, `spec/mod.rs`) is invoked next to
  each type, so readbacks see private fields. Its grammar:
  `name: Kind [= default [=> readback]];` per parameter,
  `name: Kind[, optional];` per input, `, Frames` for the frame form, and
  `@[const N: usize]` for a generic impl (`CombinedDistribution<N>`).
  `named_enum!` now also generates `From<Enum> for ParamValue / ParamDefault`
  (the canonical name).
- **Registry** `ComputeSpec::all() -> &'static [ComputeSpec]` (a `LazyLock` in
  `spec/registry.rs`) and `ComputeSpec::find(name)`.
  - 64 entries, 69 with `voronoi`.
  - `Pmft<C>` registers as `PmftR12` / `PmftXy` / `PmftXyt` / `PmftXyz`.
  - `DistributionFunction` is the `<InternalCoordinate>` instantiation.
    `CombinedDistribution` is one entry covering every `N`.
  - Not registered: `Batch` (its spec is the inner compute's) and the
    crate-internal `BondAverage`.
- **Where defaults come from.** A descriptor never holds a literal default. It
  reads one of:
  - the instance from a no-argument constructor (`Cubatic::new().seed`,
    `HBonds::default().criterion.*`, `RamanSpectrum::new()`, …);
  - an enum's `Default` (`Origins`, `MeanPolicy`, `Normalization`,
    `Reduction`, `Window`);
  - a `DEFAULT_*` constant the constructor itself uses. The new ones are
    `Spectrum::DEFAULT_PAD` (4), `StaticDielectric::DEFAULT_EPSILON_INF` (1),
    `ConductivitySumRule::DEFAULT_TOLERANCE`,
    `Steinhardt::DEFAULT_{AVERAGE,WL,WL_NORMALIZE}`,
    `EnvironmentMatch::DEFAULT_{REGISTRATION,MAX_NEIGHBORS_FOR_REGISTRATION}`,
    `LocalDensity::DEFAULT_DIAMETER`, `DiffractionPattern::DEFAULT_AXIS`,
    `Rdf::DEFAULT_DIMENSIONALITY` and `VanHove::DEFAULT_STRIDE`.

  `VanHove` now starts from `Origins::default()` (it was the literal
  `Origins::All`, the same value). Required parameters stay required:
  `max_lag`, bins, cutoffs, `start` / `end`, `units`, and `Spectrum`'s
  `window` and `pad`.
- **Core defaults that existed only in bindings (D8).**
  - The four ε(ω) routes are `new(units)` + `with_window(window)` (was
    `new(window, units)`), defaulting to `Window::default()` = `CosineSq`.
  - *Behaviour change for bindings:* when link 10 drops the binding literals,
    Python's `"none"` default for `DipoleRateCrossSpectrum` /
    `DipoleAutocorrelationSpectrum` and its `"hann"` for `GreenKuboSpectrum`
    become `cosine_sq`.
  - `RamanSpectrum` is `new()` (= `Default`) + `with_window` / `with_pad` (was
    `new(window, pad)`, which had no caller).
  - No golden literal moved. The route tests pass their windows through
    `with_window`.
- **Tests** (`spec/registry/tests.rs`):
  - `the_registry_names_every_compute_type` scans the non-test `compute`
    source for `impl … Compute for` lines. The registry must equal that set,
    minus the two exclusions and with `Pmft` expanded.
  - `every_default_is_what_the_constructor_produces` builds every registered
    type from its required arguments only and compares `tunables()` with every
    `Value` / `SameAs` / `Scaled` default. It was verified to fail when a
    constructor drifts from its constant.
  - `every_name_is_unique_and_found`, `every_default_fits_its_kind`,
    `tunables_read_the_instance`, `bindings_read_defaults_and_inputs`.
- **For link 10.**
  - Enumerate with `ComputeSpec::all()`. Every binding class name must be a
    `spec.name`.
  - Construct with only the required arguments, then assert `core.tunables()`
    equals the spec's defaults (`spec.params.iter().filter(|p| !p.is_required())`).
  - `compute` kwargs are the `spec.inputs` names.
- *As built (10b follow-ups):*
  - `TimeCorrelation::channel` returns `Result` (`OutOfRange` for a missing
    channel); the binding bounds checks are gone.
  - `LagKernelKind` (`named_enum!`: `product`, `squared_displacement`,
    `legendre`, `wigner_d`, `persistence`) and `LagKernel::new(kind, l)`
    (`l` required for `legendre` / `wigner_d`, refused otherwise) replace
    `LagKernel::NAMES` / `name()` and both bindings' kernel parsers.
    `Correlator`'s spec is `kernel: Choice(LagKernelKind::NAMES)`, `max_lag`,
    `l: Int = UNSET` (read back as `kernel.order()`); the conditional
    requirement of `l` is enforced by `LagKernel::new`, not described.
  - `AngularSeparationGlobalInput.global` → `reference` (a Python keyword).
  - `KineticEnergy::with_constants(mvv2e, boltzmann)` (both positive,
    finite); `new(units)` delegates to it. Spec: `units` `RequiredIn("new")`,
    `mvv2e` / `boltzmann` `RequiredIn("with_constants")`. Python
    `KineticEnergy(*, units= | mvv2e=, boltzmann=)`, WASM
    `KineticEnergy.withConstants`. `molrs.md.MdDriver`'s thermo uses it
    (`mvv2e = 1`, the caller's `kb`).
- **For link 11.** `computeSpecs()` serialises `ComputeSpec::all()` through
  serde, and molvis joins its labels and ranges on `(spec.name, param.name)`.

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
| `CenterOfMass`, `InertiaTensor`, `RadiusOfGyration`, `ClusterProperties` | `with_masses(&[F])` → `masses: Option<Vec<F>>` | `compute(…, masses)`; `RadiusOfGyration` / `ClusterProperties` deleted | 07 (done) |
| `SpatialDistribution` | `new(reference: Vec<usize>, template: Array2, target: Vec<usize>, grid)`; `with_orientation(pairs)` | reference/target indices, template coordinates and orientation pairs → `compute`; `grid` stays in `new` | 04 (done) |
| `HBonds` | `new(donors: Vec<(u32, u32)>, acceptors: Vec<u32>, …)` | donor/acceptor selections → `compute` | 05 (done) |
| `AtomGroups` | `new(arity, flat: Vec<Idx>)` (data wrapper) | deleted; the index-tuple array → `compute` of the distributions | 04 (done) |
| `DensityGrid` | `new(origin, basis, dims, density: Vec<F>)` (data wrapper) | `RegularGrid` + values array → `VoronoiIntegration::compute` | 08 |
| `RdfAccumulator` | public `new(rdf: Rdf)` | private; reached only through `Rdf::accumulator()` | 04 (done) |
| `Histogram1d` | `counts: Array1<F>` (a mutable accumulator posing as a type) | replaced by `Histogram<N>` output + accumulator | 04 (done) |
| `MsdTimeSeries::new(per_frame)`, `ClusterCentersResult::new(centers)` | result constructors | deleted with their types | 02 / 07 (done) |

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
   - *As built (02):* deferred to 03. The IR / VCD / Raman / ROA raw
     correlators run on `Correlator` with `Normalization::None`, so their
     numbers are unchanged in 02.
   - *As built (03):* applied. IR = `CurrentAcf` of the central `Derivative`
     of the dipole series; VCD, Raman and ROA = `Correlator { Product }` with
     the default `Normalization::Unbiased`. Re-pinned: the hand-written
     references of `irflux_plus_ir_transform_matches_manual_acf_path` and
     `ramantensor_plus_raman_transform_matches_manual_acf_path` now divide each
     lagged sum by `T − k` (`spectroscopy/mod.rs` tests); VCD and ROA gained
     the same pinned reference tests. No other literal moved.
2. **Gyration weighting.**
   - Before: `GyrationTensor` was unweighted about the geometric centre, while
     `RadiusOfGyration` and `ClusterProperties` were mass-weighted about the
     centre of mass.
   - After: both take an explicit weights array in `compute`, and Rg = √tr(G) of the same weighting.
     Re-pinned in 07.
   - *As built (07):* no golden literal moved. Each old test passes the
     weighting its type used to imply: the former unweighted
     `GyrationTensor` tests pass `ones`, the former `RadiusOfGyration` /
     `ClusterProperties` tests their masses (or `ones` where they defaulted),
     and every literal holds within its tolerance. The new
     `the_weighting_is_the_callers` pins the change (masses 3, 1 at x = 0, 4:
     `G_xx = 3` weighted, `4` number-averaged).
3. **VanHove bin edges.** VanHove now uses half-open bins like Rdf, so d = r_max
   is no longer clamped into the last bin. Re-pinned in 05.
   - *As built (05):* VanHove had no golden test, so no literal moved; the new
     `a_distance_of_exactly_r_max_is_outside_the_axis` pins the rule.
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

*Link 08 note (no golden literal moved):* the Voronoi goldens moved with
their kernels: `det3_and_point_indexing` → `RegularGrid`'s
`voxel_corners_are_the_lattice_nodes` (voxel volume 1, node `(1, 0, 1)`),
`cell_list_nearest_matches_brute_force` onto the `LinkCell` locator, the
polarizability `recovers_known_alpha_tensor` onto the type (same α, 1e-9);
`zero_field_is_error` also covers NaN / ∞. New: tessellation goldens (single
generator fills the box, SC unit cubes, the radical-plane split, Σ V = box on
a polydisperse packing with symmetric adjacency), integration goldens
(neutral diatomic, ±1 e transfer in both units, the −1 e·Å displaced
electron), domain / void aggregation, and the error variants.

*Link 07 note (no golden literal moved):* non-golden assertions that
changed: `Cluster`'s `empty_frames_is_error` and `keyed_empty_frames_is_error`
are gone (no frames; `Batch` of nothing is an empty `Vec`) and
`multi_frame_runs_per_frame` runs through `Batch`; `keyed_length_mismatch`
is a `merge` partition of the wrong length; the shape `masses_dimension_*`
tests assert the same `DimensionMismatch` on the `masses` / `weights` array;
the k-means / PCA tests build their inputs as `Array2` (labels compared as
`u32`, was `i32`). The `kinetic.rs` centre-of-mass-velocity tests moved to
`CenterOfMass` with their literals. New: `GroupReduce` boundary straddle
(ac-008), the weight-validation variants, `Cluster`'s cross-query / pair
index / `min_cluster_size = 0` / freud `merge` union, the PCA rank-one axis,
k-means in 3-D, and ac-010's `Cluster → CenterOfMass / GyrationTensor` and
`Steinhardt → Pca(3) → Kmeans`.

*Link 06 note (no golden literal moved):* behaviour change 5 has no core
site left in 06 — the only core place that turned `(head, tail)` pairs into
axes was `pmft/orientation.rs` (already head − tail), deleted here; callers
pass angles / quaternions / directors. Its named re-pins (Python `Nematic`
without minimum image, WASM `setOrientationPairs`) live in the bindings and
move to link 10. Non-golden assertions that changed: the per-type
`empty_frames_*` / `mismatched_nlist_count_is_error` tests are gone (there are
no frames; `Batch` of nothing is an empty `Vec`); the multi-frame /
`parallel_matches_serial` tests run through `Batch`; `ContinuousCoordination`'s
`empty_l_or_negative_power_errors` became `negative_power_errors`; Steinhardt
`wl[0].len() == 7` reads `wl().nrows() == 7` (same number, `N × n_l` layout).
New goldens: FCC/BCC/SC `q_4`/`q_6` on `BondAverage` (Steinhardt–Nelson–
Ronchetti Table I, 1e-5; SC analytic, 1e-12), and a two-particle PMF per
coordinate system (`−ln 50`, `−ln 200`, `−ln(640/π)`, `−ln 4000`).

*Link 05 note:* moving pair survival onto `PairPresence` → `Survival` changed
two hand-computed assertions of the deleted `pair_survival_tcf` tests (now
in `dynamics/presence.rs`): `coordination_number_at_lag_zero` `C(0) = 2.0` →
`every_pair_is_a_column` `1.0` (per pair, not per reference particle), and
`ssp_uses_inner_birth_outer_survival_cutoff` `C(0) = 0.5` →
`hysteresis_enters_at_r0_and_leaves_beyond_r1` `1.0` (frame 1 is in the bonded
stable state). Every other pair-survival and `hbond_lifetimes` literal is
unchanged; `a_bond_that_never_forms_gives_zero_curves` became
`no_bond_ever_is_no_presence` (§5).

*Link 04 note (no golden literal moved):* unifying the grid stencil reach on the
`ceil`-equivalent rule makes `GaussianDensity` include voxels whose centres
lie within `r_max` but which freud's `floor` reach skipped (a small,
uniform addition in the bulk; its tests are tolerance-based and pass
unchanged). `Axis::index` is exact against the reported edges, so a sample
lying exactly on a bin edge (to the last ULP) can move one bin relative to
the old per-file formulas — measure-zero, no test affected.

## Chain

Every link runs `cargo test -p molcrafts-molrs --lib --features full,filesystem
-- compute` (plus `core::neighbors` for 01). Binder crates are allowed not to
compile from 02 until 10. Clippy, fmt, doctests and binder builds run once, at
the end of 10, as `prek run --all-files --hook-stage pre-push`. No release
happens inside the chain.

| Link | Scope | Depends |
|---|---|---|
| 01-contract (code-complete 2026-10-14) | New `Compute`/`Verdict`/`Accumulate`/`Stream`/`Batch` (`Batch<C>` is itself a `Compute`: `Input = Vec<C::Input>`, earliest error wins); the old trait renamed `LegacyCompute`. The per-type `Input` struct convention (named data arguments, no `new`). `Partition` / `FeatureMatrix` traits, implemented for label arrays / `Array2`. An audit table of every `new`/`with_*` that takes data (D2), each assigned to its migrating link. `Neighbors` gains `SimBox`; `PairSource` trait. `named_enum!`. `ComputeError` changes. | — |
| 02-time-series (code-complete 2026-10-14) | As built: `compute/time_series/` holds `Correlator` (+ `CorrelatorAccumulator`), the eight D4 surfaces, `TimeCorrelation`, `Derivative`, `Unwrap`, `Window`/`Spectrum`/`FrequencyGrid`; `signal`, `transport/`, `msd/`, `dynamics/acf.rs` deleted; legacy spectra/dielectric/hbond rewired with unchanged numbers (deviations noted in D4, D6, D7, §1, §3, §4, §5, Behaviour change 1). Scope: `Correlator` (all five kernels, FFT + streaming ring, equal to FFT round-off), `Derivative`, `Unwrap`, the D4 surfaces with their fixed configurations pinned against today's goldens, `TimeCorrelation`, `Window`, `Spectrum`, `FrequencyGrid`. Delete `signal` (module + feature). Migrate or delete the time-correlation rows of §4. | 01 |
| 03-spectra-dielectric (code-complete 2026-10-14) | As built: fits on `{&TimeCorrelation, channel}`, `CumulativeTrapezoid` → `TimeCorrelation`; `Spectrum` is the only vibrational transform; `PolarizabilityParts` + `RamanSpectrum`; `DipoleMoment`, `StaticDielectric`, `KineticEnergy`; four routes `new(window, units)` / `compute(DielectricSpectrumInput)`; checks as `Compute` + `Verdict`; one `DielectricPrefactor`; behaviour change 1 applied; `center_of_mass_velocity` kept until 07 (deviations in §3, §4, §5, Behaviour change 1). Scope: Fits on `TimeCorrelation`, `Spectrum` unification, the Raman family with `PolarizabilityParts`, `DipoleMoment`, `StaticDielectric`, the four ε(ω) routes, the checks, the one prefactor implementation (fixes the fs/ps doc contradiction at `dielectric_spectrum.rs:59,281`), `KineticEnergy`. | 02 |
| 04-histogram-grid (code-complete 2026-10-14) | As built: `binning/` (`Axis`, `Histogram<N>`) and `grid/` (`RegularGrid<D>`, `SpreadKernel`, `Gaussian`, `Indicator`) are the one binning / spreading implementation; `Rdf` is one `Stream` over `&dyn PairSource` (`RdfMode`, `compute_self/cross/frame`, public `RdfAccumulator::new` deleted); CorrelationFunction, LocalDensity, S(k) ×2, DiffractionPattern, Gaussian/Sphere density per-snapshot `Compute`s; DF / CDF<N> / SDF `Stream`s with all data in `compute`; `Histogram1d`, `AxisSpec`, `renormalize_density`, `AtomGroups`, `GridSpec`, `InternalCoordinate::from_kind` deleted; the 2-D FFT routed through `Spectrum` (deviations in D6, §3, §4, §5, Behaviour changes' link 04 note). Scope: `Axis`, `Histogram<N>`, `RegularGrid`, `SpreadKernel`; migrate Rdf (`Stream` over `PairSource`), CorrelationFunction, LocalDensity, distributions, S(k) ×2, DiffractionPattern, Gaussian/Sphere density, SpatialDistribution. | 01 |
| 05-dynamics (code-complete 2026-10-14) | As built: `Presence` + `PairPresence` (hysteresis `r0`/`r1`, core neighbour tables as data, `Option` output) feed `Survival`; `HBonds::new(criterion)` with selections, trajectory and per-frame cross-query tables in `compute`, `HBondsResult::presence()` / `components(frame)`; `VanHove` on `Axis`/`Histogram` + `Origins`, boxes as data; `pair_survival_tcf`, `PairSurvivalResult`, `SurvivalMethod`, `hbond_lifetimes`, `HBondLifetimeResult`, `presence_from_hbonds`, `hbond_components`, `HBondNetworkResult` deleted (deviations in D4, §4, §5, Behaviour change 3 and the 05 note). Scope: `Presence`/`PairPresence`, survival through `Correlator`, H-bond input/presence/components, VanHove on `Histogram` + `core::NeighborQuery`. | 02, 04 |
| 06-order-environment (code-complete 2026-10-14) | As built: `order/bond_average.rs` (`BondBasis` + `SphericalHarmonic`/`PlanarHarmonic`, `BondAverage`, public `QlmTable`) is the one q_ℓm/ψ_k; every order/environment type is a per-snapshot `Compute`; SolidLiquid/ContinuousCoordination read `(&QlmTable, &Neighbors)`; `Pmft<C: PairCoordinate>` (`R12`/`Xy`/`Xyt`/`Xyz`) replaces the four PMFTs; `steinhardt_qlm`, `angular_distance`, `pmft/orientation.rs`, every `*Args`, `Dsu` deleted; zero orientations error everywhere; R12 bin-measure bug fixed (deviations in §3, §4, §5, Link 06 note). Scope: `BondBasis`/`BondAverage`/`QlmTable`; Steinhardt, Hexatic, SolidLiquid, ContinuousCoordination, BOO, LocalDescriptors, LocalBondProjection, EnvironmentMatch, AngularSeparation, Nematic, Cubatic; `Pmft<C>`. | 01, 04 |
| 07-groups (code-complete 2026-10-14) | As built: `shape/group_reduce.rs` (`GroupReduce`, crate-internal: anchor + core-MIC offsets, weighted centroid, weighted second moment) behind `CenterOfMass` / `GyrationTensor` / `InertiaTensor` (configuration-free; `compute({ positions, simbox, partition, masses/weights })`); `Cluster` on `Topology::connected_components` with `merge`; `ClusterResult` / `KmeansResult` are `Partition`s, `PcaResult` a `FeatureMatrix`; `Pca::new(n_components)` on `eigh_sym`; `Kmeans::new(k)` + `with_max_iter` / `with_seed` (core defaults 100 / 0) over any `FeatureMatrix`; `Partition::sizes`; `ClusterCenters`, `RadiusOfGyration`, `ClusterProperties`, `compute_keyed`, `center_of_mass_velocity`, the power iteration and (dead after 07) `positions.rs` deleted (deviations in §3, §4, §5, Behaviour change 2). Scope: `GroupReduce`; Cluster (`connected_components`, `merge`), CenterOfMass, GyrationTensor, InertiaTensor; delete ClusterCenters, RadiusOfGyration, ClusterProperties; Pca (`eigh_sym`, `n_components`), Kmeans (`FeatureMatrix`). | 01, 06 (FeatureMatrix producers) |
| 08-voronoi (code-complete 2026-10-14) | As built: `RadicalVoronoi`, `VoronoiIntegration` (`new(LengthUnit)`), `VoronoiDomainAnalysis`, `VoronoiVoidAnalysis`, `FiniteFieldPolarizability` (`new(field)`) are per-snapshot `Compute`s with named `Input`s; candidate search on core `LinkCell` (+ `GhostSet` copies for the tessellation's multi-image search); `DensityGrid` → `RegularGrid` (+ `voxel_corner`) and an `ArrayView3`; Pass A on `GroupReduce`; `CellGrid`, `CellList`, `DensityGrid`, `from_cube_frame`, `polarizability_finite_field`, `leak` deleted; no `GridField` (deviations in §3, §4, §5, Link 08 note). Scope: `Compute` impls, core cell list, explicit units, `FiniteFieldPolarizability`, `GridField`. | 04, 07 |
| 09-retire-legacy (code-complete 2026-10-14) | As built: `analysis_contract.rs` deleted (`LegacyCompute`, `Fit`, `Check`, `ComputeResult`, `DescriptorRow`; `RdfResult`'s impl); `ComputeError` cut to the seven variants with a producer (also `NotConverged`, `MolRs` — deviation in §1); `compute/mod.rs` docs rewritten (contract, config/data, no frames, `Batch`/`Stream`, `Partition`/`FeatureMatrix`, `PairSource`, surfaces, spectra, folders; five doctests); `VoronoiDomainAnalysis` on `&dyn Partition` (result is a `Partition`); the Voronoi exhaustive fallback proven complete or `OutOfRange` (§5 09 note); `DipoleMoment` partition `+ Sync`; ac-010's missing `DipoleMoment → Msd → LinearFit` test added; acceptance audit below ("Link 09 acceptance audit"). `positions.rs` was already gone (07). Scope: Delete `LegacyCompute`, `analysis_contract.rs`, `positions.rs`; rewrite the `compute/mod.rs` module docs (removes the stale `molpy.compute.Workflow` reference). Run the §4 closing greps. | 02–08 |
| 10a-compute-spec (code-complete 2026-10-14) | As built: `ComputeSpec` / `ParamSpec` / `ParamKind` / `ParamValue` / `ParamDefault` / `InputSpec` / `InputKind` / `InputForm`, trait `Described` (`spec()`, `tunables()`), crate-internal `compute_spec!` next to every type, registry `ComputeSpec::all()` / `find()` (64 entries; 69 with `voronoi`); defaults read from the type (no-arg instance, enum `Default`, or a `DEFAULT_*` the constructor uses — 12 new constants); ε(ω) routes `new(units)` + `with_window` (core `Window` default) and `RamanSpectrum::new()` + `with_window` / `with_pad`; completeness (source scan), unique-name and defaults-equal-constructor tests (deviations in §7). Scope: the core-only machine-readable descriptor of every public compute, the prerequisite of 10's parity tests and 11's generated catalog. | 09 |
| 10-bindings | §6 in molrs-python, molrs-wasm, molrs-cxxapi; full pre-push gate; Atomiverse consumer note. | 10a |
| 11-catalog | §7; molvis `registry.ts` migration (cross-repo). | 10 |
| 12-molpy | §8, after the molrs release that carries 01–11. | molrs tag |

## Link 09 acceptance audit

Run on 2026-10-14 against the molrs core (`molrs/src/compute`) after link 09.
Bindings parts of ac-003 / ac-016 and ac-011–015 belong to links 10–12.
"Non-test" means outside `#[cfg(test)]` items and the `tests.rs` /
`product_tests.rs` test modules.

| Criterion | Status | Evidence |
|---|---|---|
| ac-001 | pass | `pub trait` under compute: `Compute` is the one analysis trait (the others are `Verdict`, `Accumulate`, `Stream`, `Partition`, `FeatureMatrix`, `Observable`, `PairCoordinate`, `SpreadKernel`, crate-internal `BondBasis`); `grep -rnE 'trait Fit\|trait Check\|ComputeResult\|DescriptorRow\|LegacyCompute'` empty; all 71 `impl Compute for` blocks define only `compute`; `grep 'pub fn compute_'` empty. |
| ac-002 | pass | `grep -rnE '^pub fn ' molrs/src/compute` empty; `molrs/src/signal/` absent; `Cargo.toml`: `compute = ["dep:rustfft"]`, no `signal` feature. Also no `pub type` under compute. |
| ac-003 (core) | pass | Non-test code: no `FrameAccess`/`FrameView`/`core::Frame`/`core::schema`/`keys`, no `from_frame(s)`, zero field-name-literal hits; `positions.rs`, `pmft/orientation.rs` absent. Prose only: `contract.rs` / `mod.rs` say compute takes no `Frame`, and "Frame spacing" field docs. |
| ac-016 (core) | pass (exemptions) | No `TimeSeries`/`Snapshot`/`Weights`/`SeriesPair`/`Labels`/`ThermoState` type. Every public compute `new`/`with_*` takes scalars, `Axis`, `RegularGrid`, `Window`, `LagKernel`, `Origins`, `Reduction`, `MeanPolicy`, `Normalization`, `Stencil`, `UnitPreset`, named enums, or config values (`HBondCriterion`, `Plateau`, an `Observable`). Exempt: (a) `PairCoordinate::Orientations<'a>` is an associated type naming the `ArrayView` an orientation input *is* — not a wrapper anyone builds; (b) `Steinhardt::new(&[l])`, `VanHove::new(axis, lags)`, `StaticStructureFactorDebye::new(&k)`, `StaticStructureFactorDirect::new(&k_vecs)` take where-to-evaluate lists, the Link 01 audit's "configuration that legitimately looks like data"; (c) `FrequencyGrid::new(n_fft, dt)` and `RegularGrid::over_box(simbox, dims)` are value-type constructors, not compute types; `DielectricPrefactor`, `GroupReduce`, `BondAverage`, `Histogram::new`, `DistributionResult::new` are crate-internal. |
| ac-004 | pass | Every "→" symbol of §4 is absent from the re-exports and has no definition (remaining hits are "the former `X`" comments and freud URLs); all 66 K/S compute types checked (Correlator, the eight surfaces, fits, spectra, checks and every kept analysis) are re-exported with exactly one `impl Compute for`; M methods exist (`SteinhardtResult::qlm`, `HBondsResult::presence` / `components`, `GyrationTensorResult::radius_of_gyration`, `KineticEnergyResult::temperature`); the eight D4 surfaces hold `correlator: Correlator` and their `compute` is one `self.correlator.compute(CorrelatorInput { .. })` (axis insertion only). |
| ac-005 | pass (exemptions) | `rustfft` imported only in `time_series/correlator.rs` and `time_series/spectrum.rs`; no `CellGrid`/`CellList`/`Dsu`/`power_iteration`; `pmft/` = `mod.rs` + `r12`/`xy`/`xyt`/`xyz`; no non-test `.min(n − 1)` bin clamp. Non-test `0.5 * (` hits that are not trapezoid sums: the Kramers–Kronig quadrature weight (exempt since 03), the Hann/Blackman window formulas in `Window`, and the Voronoi radical-plane offset `½(d² + Rᵢ² − Rⱼ²)` (one site, `clip_nearest_first`). |
| ac-006 | pass | Non-test grep `map_or(1.0 \| unwrap_or(1.0 \| .max(1) \| min(n…-1)` empty (09 rewrote `DipoleMoment`'s structural `groups.len().max(1)` as an explicit one-series branch); no non-test `debug_assert!` / `assert!`; the remaining `.clamp(-1, 1)` are `acos` round-off guards and `LocalDensity`'s smoothing weight, not input fallbacks. Each §5 item's variant test is recorded in the 02–08 as-built notes; 09 adds `an_exhaustive_reach_beyond_the_image_limit_is_an_error` and `an_ungrouped_cell_is_in_no_domain`. |
| ac-007 (core) | pass | `Rdf`, `DistributionFunction`, `CombinedDistribution`, `SpatialDistribution` `compute` = `self.accumulator().fold(..)`; `Correlator` + surfaces are the documented FFT exception, pinned by `correlator::tests::streaming_equals_batch_at_every_lag` (every kernel incl. `Product` / `SquaredDisplacement`, every lag, 1e-12 relative), `vacf::streaming_matches_batch_fft_vacf`, msd streaming tests, and `rdf::streaming_accumulator_matches_batch_bitwise`. |
| ac-008 (core) | pass | `batch_matches_the_definition_for_every_kernel_and_option` (five kernels); `axis::{edges_and_centers, bins_are_half_open, non_finite_samples_are_errors, shell_volumes_match_closed_forms}`, `histogram::nan_sample_is_non_finite_and_counts_nothing`; `regular_grid::stencil_wraps_only_periodic_axes`; `group_reduce::a_group_straddling_the_boundary_is_unwrapped_about_its_anchor`; `bond_average::lattice_shells_give_the_tabulated_invariants`; `pmft::{r12,xy,xyt,xyz}::two_particle_pmf_golden`; eight `surface_is_its_correlator`; `a_wrapped_step_is_refused_and_an_unwrapped_one_is_not`. All pass. |
| ac-009 | pass | Every one of the 38 distinct fractional assertion literals in the pre-chain (`9db8de95`) compute/signal tests is present in the current tree; of 351 pre-chain test names, 80 are gone — renamed, merged into the new primitives' tests, or deleted with their types/fallbacks as recorded in the 02–08 notes (e.g. `test_hann_*` → `Window`, `coordination_number_at_lag_zero` → 05 re-pin). 09 changed no golden literal. |
| ac-010 (core) | pass | `vacf::velocities_to_diffusion_through_cumulative_trapezoid`; `dipole_moment::dipole_series_feeds_msd_then_linear_fit` (**added in 09** — no test covered it); `kmeans::steinhardt_pca_kmeans_composes`; `shape::cluster_feeds_the_shape_computes_as_a_partition`; `hbond::hbonds_presence_survival_lifetimes`; `dielectric_spectrum::tests::dipole_acf_to_kramers_kronig_pipeline`. |

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
