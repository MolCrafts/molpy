//! Analysis of molecular simulation data: one contract, one entry per
//! quantity, one implementation per algorithm.
//!
//! Every analysis in this module — a pair distribution, a correlation curve, a
//! fit of that curve, a spectrum, a check on the spectrum — is a type that
//! implements the one trait [`Compute`], whose one method is
//! [`compute`](Compute::compute). A fit is a `Compute` whose input is an
//! upstream result; a check is a `Compute` whose output implements
//! [`Verdict`]. Because there is only one verb, any output can be the next
//! analysis's input with no glue in between.
//!
//! # `new` is configuration, `compute` is data
//!
//! A constructor (`new`, and the `with_*` setters that refine it) takes only
//! what *configures* an analysis: bin axes, cutoffs, a maximum lag, kernels,
//! windows, tolerances, seeds and the unit system. It validates them and
//! returns a reusable instrument. Every default lives here, in the core type.
//!
//! [`compute`](Compute::compute) takes everything being analysed: arrays,
//! a [`SimBox`](molrs::core::SimBox), a neighbour table, a sample interval,
//! a volume or a temperature, an upstream result. These arrive as the
//! analysis's [`Input`](Compute::Input) — a plain struct of borrowed fields
//! written as a struct literal at the call site, which is Rust's spelling of
//! named arguments. An `Input` has no constructor: there is no data object to
//! build before calling `compute`, and the Python and WASM bindings expose its
//! fields as the keyword arguments of `compute`. `&self` never changes, so
//! identical configuration and data always give an identical result, and a
//! result is final when it is returned.
//!
//! ```
//! use molrs::compute::{Compute, CumulativeTrapezoid, CumulativeTrapezoidInput, Vacf, VacfInput};
//! use ndarray::Array2;
//!
//! // Configuration: lags 0 … 16. Data: the velocities and their frame spacing.
//! let velocities = Array2::from_shape_fn((64, 3), |(t, d)| ((t * 3 + d) as f64 * 0.7).sin());
//! let raw = Vacf::new(16).compute(VacfInput { velocities: velocities.view(), dt: 1.0 })?;
//!
//! // A fit is a `Compute` whose input is the upstream curve: no glue.
//! let running = CumulativeTrapezoid::new()
//!     .compute(CumulativeTrapezoidInput { correlation: &raw, channel: 0 })?; // D·d in MD units
//! assert_eq!(running.values.nrows(), raw.values.nrows());
//! # Ok::<(), molrs::compute::ComputeError>(())
//! ```
//!
//! # Compute reads no frames
//!
//! No type here takes a `Frame`, and no type here names a field. Reading the
//! velocity column, the masses or the quaternions out of a frame is the
//! caller's step, through the `Frame`/`Block` API; the analysis receives the
//! arrays. So one kernel serves a trajectory file, an in-memory MD run and a
//! NumPy array alike, and a binding adds no frame-reading code of its own.
//! Every out-of-contract input — a length that does not match, a NaN, a value
//! outside its physical range, nothing to analyse — is a [`ComputeError`];
//! nothing is silently clamped, defaulted or skipped.
//!
//! # Many snapshots, and long trajectories
//!
//! A per-snapshot analysis takes one snapshot's data. The same analysis over
//! many independent snapshots is [`Batch`], itself a `Compute`: it maps the
//! inner analysis over a `Vec` of inputs — in parallel when the `rayon`
//! feature is on — returns the results in input order, and reports the
//! earliest failing input's error, so the outcome never depends on scheduling.
//!
//! ```
//! use molrs::compute::{Batch, Compute, KineticEnergy, KineticEnergyInput};
//! use molrs::core::UnitPreset;
//! use ndarray::array;
//!
//! let masses = array![2.0, 1.0];
//! let snapshots = [
//!     array![[1.0, 0.0, 0.0], [0.0, 2.0, 2.0]],
//!     array![[0.0, 0.0, 0.0], [0.0, 0.0, 1.0]],
//! ];
//! let per_snapshot = Batch(KineticEnergy::new(UnitPreset::si())?).compute(
//!     snapshots
//!         .iter()
//!         .map(|v| KineticEnergyInput { masses: masses.view(), velocities: v.view() })
//!         .collect(),
//! )?;
//! assert_eq!(per_snapshot.iter().map(|k| k.energy).collect::<Vec<_>>(), vec![5.0, 0.5]);
//! # Ok::<(), molrs::compute::ComputeError>(())
//! ```
//!
//! An analysis that runs over a whole trajectory and can fold it in bounded
//! memory also implements [`Stream`]: [`Stream::accumulator`] hands out an
//! [`Accumulate`] that receives one frame at a time through
//! [`push`](Accumulate::push), and [`finish`](Accumulate::finish) reports the
//! result so far without consuming it, so a long run can be inspected and
//! continued. The algorithm lives once, in the accumulator: a `Stream`'s
//! `compute` is `self.accumulator().fold(input)`. The one exception is
//! [`Correlator`] (and so every correlation surface), whose `compute` is the
//! faster FFT evaluation; its tests pin the FFT and the fold equal at every
//! lag.
//!
//! ```
//! use molrs::compute::{Accumulate, Compute, CorrelatorFrame, Stream, Vacf, VacfInput};
//! use ndarray::{Array2, Axis};
//!
//! let velocities = Array2::from_shape_fn((40, 6), |(t, d)| ((t * 6 + d) as f64 * 0.37).sin());
//! let vacf = Vacf::new(8);
//! let batch = vacf.compute(VacfInput { velocities: velocities.view(), dt: 0.5 })?;
//!
//! // The same curve, one frame of `n_dof × 1` velocities at a time.
//! let mut running = vacf.accumulator();
//! for frame in velocities.outer_iter() {
//!     running.push(CorrelatorFrame {
//!         a: frame.insert_axis(Axis(1)),
//!         b: None,
//!         dt: 0.5,
//!         simbox: None,
//!     })?;
//! }
//! let streamed = running.finish()?;
//! for (s, b) in streamed.values.iter().zip(batch.values.iter()) {
//!     assert!((s - b).abs() <= 1e-10 * b.abs().max(1.0));
//! }
//! # Ok::<(), molrs::compute::ComputeError>(())
//! ```
//!
//! # Composition through traits, not wrappers
//!
//! Two traits let one analysis consume another's output — or the caller's own
//! arrays — without a conversion step:
//!
//! - [`Partition`] says which group each item belongs to. A label array
//!   (`&[u32]`, `Vec<u32>`, `Array1<u32>` — a `mol_id` column, say) is one, and
//!   so is every result that partitions items: [`ClusterResult`],
//!   [`KmeansResult`], [`EnvironmentMatchResult`], `VoronoiDomainResult` (feature `voronoi`)
//!   and the label array of [`HBondsResult::components`]. Group reductions
//!   ([`CenterOfMass`], [`GyrationTensor`], [`InertiaTensor`],
//!   [`DipoleMoment`], `VoronoiIntegration`, `VoronoiDomainAnalysis`) and
//!   [`Cluster`]'s `merge` accept any of them.
//! - [`FeatureMatrix`] is a table of samples, one row each. An `Array2` is
//!   one, and so are the descriptor results [`SteinhardtResult`],
//!   [`LocalDescriptorsResult`] and [`PcaResult`]; [`Pca`] and [`Kmeans`]
//!   accept any of them.
//!
//! ```
//! use molrs::compute::{
//!     CenterOfMass, CenterOfMassInput, Compute, Kmeans, KmeansInput, Pca, PcaInput,
//! };
//! use ndarray::{Array1, array};
//!
//! // Two well-separated groups of three particles.
//! let positions = array![
//!     [0.0, 0.1, 0.3], [0.2, 0.0, 0.1], [0.1, 0.3, 0.0],
//!     [9.0, 9.2, 9.1], [9.3, 9.0, 9.2], [9.1, 9.1, 9.0],
//! ];
//! // Positions are a feature matrix; the projection is one; the clustering
//! // is a partition the group reduction consumes.
//! let projected = Pca::new(2)?.compute(PcaInput { features: &positions })?;
//! let clusters = Kmeans::new(2)?.compute(KmeansInput { features: &projected })?;
//! let centres = CenterOfMass::new().compute(CenterOfMassInput {
//!     positions: positions.view(),
//!     simbox: None,
//!     partition: &clusters,
//!     masses: Array1::ones(6).view(),
//! })?;
//! assert_eq!(centres.total_masses.to_vec(), vec![3.0, 3.0]);
//! # Ok::<(), molrs::compute::ComputeError>(())
//! ```
//!
//! # Neighbour tables: pair sources, and which columns a kernel needs
//!
//! Most analyses here are *pair* analyses: they answer a question about every
//! pair of particles closer together than a cutoff. Finding those pairs is not
//! their job. The search lives in [`NeighborList`](molrs::core::NeighborList)
//! and [`NeighborQuery`](molrs::core::NeighborQuery), and its product is
//! passed to `compute` as data, one table per snapshot.
//!
//! A kernel that only streams pairs takes a
//! [`PairSource`](molrs::core::PairSource): either a materialised
//! [`Neighbors`](molrs::core::Neighbors) table or a built `NeighborList`,
//! which streams pairs straight out of its cell list without ever
//! materialising them — the bounded-memory path for large systems. A kernel
//! that needs columns or random access takes `&Neighbors`. A table knows the
//! box it was searched in, [`Neighbors::simbox`](molrs::core::Neighbors::simbox):
//! it is `Some` for every table a search produced and `None` for one built by
//! hand, and a kernel that needs a volume rejects `None` rather than invent a
//! box. [`HBonds`] and [`PairPresence`] take the caller's per-frame tables;
//! only [`VanHove`] searches for itself, one cross-query per time origin,
//! because its pair set depends on the origin and the lag.
//!
//! ```
//! use molrs::compute::{Compute, Rdf};
//! use molrs::core::{NeighborColumns, NeighborList, SimBox};
//! use ndarray::{Array2, array};
//!
//! let simbox = SimBox::cube(10.0, array![0.0, 0.0, 0.0], [true; 3]).unwrap();
//! let points = Array2::from_shape_fn((64, 3), |(i, d)| 2.5 * ((i >> (2 * d)) & 3) as f64 + 0.3);
//! let mut nl = NeighborList::new(4.0);
//! nl.build(points.view(), &simbox);
//!
//! let rdf = Rdf::new(20, 4.0, 0.0)?;
//! let streamed = rdf.compute(&[&nl])?; // pairs straight from the cell list
//! let table = nl.neighbors(NeighborColumns::DIST_SQ);
//! assert!(table.simbox().is_some()); // the table knows its box
//! let materialized = rdf.compute(&[&table])?;
//! assert_eq!(streamed.n_r, materialized.n_r);
//! # Ok::<(), molrs::compute::ComputeError>(())
//! ```
//!
//! ## Columns are opt-in, and a missing one is an error rather than a zero
//!
//! Every table stores each pair's two particle indices `(i, j)`. Its two
//! *physical* columns are stored only if the caller asked for them when the
//! table was materialised, by naming a
//! [`NeighborColumns`](molrs::core::NeighborColumns) policy:
//!
//! - `dist_sq` — the squared pair distance `|r_j − r_i|²` in Å², taken under
//!   the **minimum-image convention**: with periodic boundaries the box is
//!   tiled through space, so a pair has one separation per periodic copy
//!   ("image") of the box, and the search reports only the shortest of them.
//! - `disp` — the displacement `r_j − r_i` itself in Å, from that same
//!   minimum image, pointing from `i` to `j` and **not** normalised. Its length
//!   is the pair distance, so a caller who wants a direction divides by that
//!   length.
//!
//! A column the table never stored reads back as `None`, never as a fabricated
//! zero: a zero displacement is a physically meaningful value (two coincident
//! particles) and would be indistinguishable from real data. So a kernel that
//! needs a column rejects `None` instead of substituting anything, and every
//! kernel rejects it the same way — [`ComputeError::BadShape`], whose
//! `expected` text names the missing column and the number of pairs it was
//! needed for.
//!
//! Which column a kernel needs follows from what it measures:
//!
//! | Needs | Kernels | Materialise the table with |
//! |---|---|---|
//! | `disp` — bond *directions* | [`Steinhardt`], [`Hexatic`], [`Pmft`], [`BondOrientationalOrder`], [`LocalDescriptors`], [`LocalBondProjection`], [`EnvironmentMatch`] | `NeighborColumns::DISP` or `FULL` |
//! | `dist_sq` — distances | [`Rdf`], [`CorrelationFunction`], [`LocalDensity`], [`PairPresence`] (each takes any `PairSource`); [`HBonds`] | `NeighborColumns::DIST_SQ` or `FULL` |
//! | indices only — connectivity | [`Cluster`], [`AngularSeparationNeighbor`], and the pairs of a [`QlmTable`] in [`SolidLiquid`] and [`ContinuousCoordination`] | any policy, `INDICES_ONLY` included |
//!
//! `FULL` means *every column is present*. It never means a bidirectional pair
//! list — pair direction is an independent property, recorded by the table's
//! [`QueryMode`](molrs::core::QueryMode). A **self-query** searches one point
//! set against itself and is half-shell: each unordered pair appears exactly
//! once, with `i < j`. A **cross-query** searches query points against a
//! separate reference set and is directed, with both orderings present.
//! Radial-distribution normalisation is the one place that distinction
//! changes a number — a half-shell count is short by a factor of two — see
//! [`RdfResult`].
//!
//! # Binning and grids
//!
//! Every histogram bins through [`Axis`] (half-open `[lo, hi)` bins; a NaN
//! sample is [`ComputeError::NonFinite`]) into a [`Histogram`], and every
//! density grid spreads a [`SpreadKernel`] ([`Gaussian`], [`Indicator`]) over
//! a [`RegularGrid`], which wraps only its periodic axes.
//!
//! # Time correlation: one estimator, named surfaces
//!
//! Every lagged correlation is one configuration of [`Correlator`] — a
//! [`LagKernel`] (product, squared displacement, Legendre, Wigner-D,
//! persistence), a maximum lag, the [`Origins`], [`MeanPolicy`],
//! [`Normalization`] and [`Reduction`] — evaluated by FFT, or folded frame by
//! frame through its accumulator, into a [`TimeCorrelation`]. The quantities
//! chemists know by name are thin surfaces over it: each holds one
//! `Correlator` with the configuration that defines the quantity fixed,
//! exposes only what a user legitimately tunes, and delegates `compute` in one
//! line ([`Survival`] adds its two conventions — per-reference averaging and
//! inner-cutoff origins — around that call). A correlation with no named
//! surface is a `Correlator` configured directly.
//!
//! The raw curve is an observable; the step that turns it into a coefficient
//! is the analyst's explicit, parameterised choice of fit:
//!
//! | Surface | Raw curve | Then |
//! |--------|-----------|----------------|
//! | [`Vacf`] | velocity ACF | [`Spectrum`] (VDOS) / [`CumulativeTrapezoid`] (D) |
//! | [`Msd`] | self-MSD; on the [`DipoleMoment`] series, the Einstein–Helfand curve | [`LinearFit`] (D = slope / 2d; σ) |
//! | [`OnsagerCorrelation`] | Onsager L_ij displacement cross-correlation | [`LinearFit`] per pair |
//! | [`CurrentAcf`] | current ACF of the [`Derivative`] of the dipole series | [`CumulativeTrapezoid`] (σ) |
//! | [`DipoleAcf`] | dipole fluctuation ACF | [`DebyeFit`] (τ_D) / [`DipoleAutocorrelationSpectrum`] |
//! | [`ReorientationAcf`], [`RotationalAutocorrelation`] | orientational decay | [`DebyeFit`] |
//! | [`Survival`] | continuous / intermittent survival of a [`Presence`] ([`PairPresence`], [`HBondsResult::presence`]) | [`CumulativeTrapezoid`] (lifetime) |
//!
//! Units follow the caller's MD convention: time in the unit of `dt`,
//! velocities and dipoles as supplied; the fits document their prefactors.
//!
//! # Raw observable, then transform: spectra
//!
//! Each spectrum is an explicit composition — a series operation, a
//! correlation, a transform. No spectrum owns an estimator or an FFT:
//!
//! | Spectrum | Series → correlation | Transform |
//! |----------|----------------------|-----------|
//! | VDOS | velocities → [`Vacf`] | [`Spectrum`] |
//! | IR | [`DipoleMoment`] → [`Derivative`] → [`CurrentAcf`] | [`Spectrum`] |
//! | VCD | [`Derivative`] of μ and m → [`Correlator`] cross | [`Spectrum`] |
//! | Raman / resonance Raman | [`PolarizabilityParts`] → [`Derivative`] → [`Correlator`] | [`RamanSpectrum`] |
//! | ROA | parts of α̇ and Ġ′ → [`Correlator`] cross | [`RamanSpectrum`] |
//! | Dielectric ε(ω) | [`DipoleAcf`] / [`CurrentAcf`] / [`Correlator`] cross of (Ṁ, M) | [`EinsteinHelfandSpectrum`] / [`GreenKuboSpectrum`] / [`DipoleAutocorrelationSpectrum`] / [`DipoleRateCrossSpectrum`] |
//!
//! A permittivity spectrum is checked by a `Compute` whose output is a
//! [`Verdict`]: [`KramersKronig`] ([`KramersKronigCheck`]),
//! [`ConductivitySumRule`] ([`SumRuleCheck`]) and [`RouteAgreement`]
//! ([`RouteAgreementCheck`]). The static permittivity is [`StaticDielectric`].
//!
//! | quantity   | unit |
//! |------------|------|
//! | time / dt  | fs (vibrational spectra); any consistent unit (ε(ω)) |
//! | frequency  | cm⁻¹ ([`RamanSpectrum`]); rad per unit of dt ([`Spectrum`], ε(ω)) |
//! | intensity  | arb. |
//!
//! # Every compute describes itself
//!
//! Each type above implements [`Described`]: a [`ComputeSpec`] with its name,
//! its parameters (kind, and a default read from the type itself — never
//! retyped) and the fields of its `Input`. [`ComputeSpec::all`] lists every
//! one; bindings and catalogs read it instead of restating signatures, and
//! [`Described::tunables`] reports what a built instance holds.
//!
//! # Where things live
//!
//! One folder per kernel family; every public type is re-exported here.
//!
//! | Folder | Contents |
//! |----------|---------|
//! | `contract` | [`Compute`], [`Verdict`], [`Accumulate`], [`Stream`], [`Batch`], [`Partition`], [`FeatureMatrix`] |
//! | `spec` | [`ComputeSpec`], [`Described`] and the registry [`ComputeSpec::all`] |
//! | `binning` | [`Axis`], [`Histogram`] — the one binning implementation |
//! | `grid` | [`RegularGrid`], [`SpreadKernel`] ([`Gaussian`], [`Indicator`]) — the one grid-spreading implementation |
//! | `time_series` | [`Correlator`] and its surfaces, [`TimeCorrelation`], [`Derivative`], [`Unwrap`], [`Window`], [`Spectrum`], [`FrequencyGrid`] |
//! | `fitting` | fits of a [`TimeCorrelation`]: [`LinearFit`], [`CumulativeTrapezoid`], [`Plateau`], [`DebyeFit`] |
//! | `spectroscopy` | [`PolarizabilityParts`], [`RamanSpectrum`], the four ε(ω) routes and their checks |
//! | `dielectric` | [`DipoleMoment`], [`StaticDielectric`], [`EwaldBoundary`] |
//! | `kinetic` | [`KineticEnergy`] (and its temperature) of one state |
//! | `rdf` | [`Rdf`], the pair distribution g(r), and its [`RdfAccumulator`] |
//! | `density` | [`CorrelationFunction`], [`LocalDensity`], [`GaussianDensity`], [`SphereVoxelization`], [`SpatialDistribution`] |
//! | `distribution` | [`DistributionFunction`] and [`CombinedDistribution`] of distances, angles and dihedrals |
//! | `diffraction` | [`StaticStructureFactorDebye`], [`StaticStructureFactorDirect`], [`DiffractionPattern`] |
//! | `dynamics` | [`VanHove`] G(r, t), [`PairPresence`] and its [`Presence`] series |
//! | `hbond` | [`HBonds`]: detection, presence series and network components |
//! | `order` | the one bond-harmonic average ([`QlmTable`]) behind [`Steinhardt`], [`Hexatic`], [`SolidLiquid`], [`ContinuousCoordination`]; [`Nematic`], [`Cubatic`] |
//! | `environment` | [`BondOrientationalOrder`], [`LocalDescriptors`], [`LocalBondProjection`], [`EnvironmentMatch`], angular separation |
//! | `pmft` | one [`Pmft`] over a [`PairCoordinate`] ([`R12`], [`Xy`], [`Xyt`], [`Xyz`]) |
//! | `cluster` | [`Cluster`]: connected components of a neighbour table, optionally merged by a partition |
//! | `shape` | the one group reduction behind [`CenterOfMass`], [`GyrationTensor`], [`InertiaTensor`] |
//! | `decomposition` | [`Pca`] of any [`FeatureMatrix`] |
//! | `kmeans` | [`Kmeans`] over any [`FeatureMatrix`] |
//! | `voronoi` | `RadicalVoronoi`, `VoronoiDomainAnalysis`, `VoronoiVoidAnalysis`, `VoronoiIntegration`, `FiniteFieldPolarizability` (feature `voronoi`) |

mod binning;
mod cluster;
mod contract;
mod decomposition;
mod density;
mod dielectric;
mod diffraction;
mod distribution;
mod dynamics;
mod environment;
mod error;
mod fitting;
#[cfg(test)]
pub(crate) mod fixtures;
mod grid;
mod hbond;
mod kinetic;
mod kmeans;
mod named_enum;
mod order;
mod pmft;
mod rdf;
pub(crate) mod require;
mod shape;
mod spec;
mod spectroscopy;
mod time_series;
#[cfg(feature = "voronoi")]
mod voronoi;

// Re-exports
pub use binning::{Axis, Histogram, UpperEdge};
pub use cluster::{Cluster, ClusterInput, ClusterResult};
pub use contract::{Accumulate, Batch, Compute, FeatureMatrix, Partition, Stream, Verdict};
pub use decomposition::{Pca, PcaInput, PcaResult};
pub use density::{
    CorrelationFunction, CorrelationFunctionInput, CorrelationFunctionResult, GaussianDensity,
    GaussianDensityInput, GaussianDensityResult, LocalDensity, LocalDensityInput,
    LocalDensityResult, SpatialDistribution, SpatialDistributionAccumulator,
    SpatialDistributionFrame, SpatialDistributionResult, SphereVoxelization,
    SphereVoxelizationInput, SphereVoxelizationResult,
};
pub use dielectric::{
    DipoleMoment, DipoleMomentInput, DipoleMomentResult, EwaldBoundary, StaticDielectric,
    StaticDielectricInput, StaticDielectricResult,
};
pub use diffraction::{
    DiffractionPattern, DiffractionPatternInput, DiffractionPatternResult,
    StaticStructureFactorDebye, StaticStructureFactorDebyeInput, StaticStructureFactorDebyeResult,
    StaticStructureFactorDirect, StaticStructureFactorDirectInput,
    StaticStructureFactorDirectResult,
};
pub use distribution::{
    AngleObservable, CombinedDistribution, CombinedDistributionAccumulator,
    CombinedDistributionFrame, CombinedDistributionResult, DihedralObservable, DistanceObservable,
    DistributionAccumulator, DistributionFrame, DistributionFunction, DistributionResult,
    InternalCoordinate, Observable,
};
pub use dynamics::{
    PairPresence, PairPresenceInput, Presence, VanHove, VanHoveInput, VanHoveResult,
};
pub use environment::{
    AngularSeparationGlobal, AngularSeparationGlobalInput, AngularSeparationGlobalResult,
    AngularSeparationNeighbor, AngularSeparationNeighborInput, AngularSeparationNeighborResult,
    BondOrientationalOrder, BondOrientationalOrderInput, BondOrientationalOrderResult,
    EnvironmentMatch, EnvironmentMatchInput, EnvironmentMatchResult, LocalBondProjection,
    LocalBondProjectionInput, LocalBondProjectionResult, LocalDescriptors, LocalDescriptorsInput,
    LocalDescriptorsResult,
};
pub use error::ComputeError;
pub use fitting::{
    CumulativeTrapezoid, CumulativeTrapezoidInput, DebyeFit, DebyeFitInput, DebyeFitResult,
    LinearFit, LinearFitInput, LinearFitResult, Plateau, PlateauInput, PlateauResult,
};
pub use grid::{Gaussian, Indicator, RegularGrid, SpreadKernel, StencilReach};
pub use hbond::{HBond, HBondCriterion, HBondDistanceKind, HBonds, HBondsInput, HBondsResult};
pub use kinetic::{KineticEnergy, KineticEnergyInput, KineticEnergyResult};
pub use kmeans::{Kmeans, KmeansInput, KmeansResult};
pub use order::{
    ContinuousCoordination, ContinuousCoordinationInput, ContinuousCoordinationResult, Cubatic,
    CubaticInput, CubaticResult, Hexatic, HexaticInput, HexaticResult, Nematic, NematicInput,
    NematicResult, QlmTable, SolidLiquid, SolidLiquidInput, SolidLiquidResult, Steinhardt,
    SteinhardtInput, SteinhardtResult,
};
pub use pmft::{PairCoordinate, Pmft, PmftInput, PmftResult, R12, Xy, Xyt, Xyz};
pub use rdf::{Rdf, RdfAccumulator, RdfResult};
/// Crate-internal: the input guards every neighbor-consuming kernel calls
/// before it reads `disp` (Å) or `dist_sq` (Å²), or before it updates both
/// endpoints of a row and so depends on the table being half-shell.
/// Deliberately not public API — a caller outside the crate holds the table
/// itself and asks it directly with
/// [`Neighbors::disp()`](molrs::core::Neighbors::disp),
/// [`Neighbors::dist_sq()`](molrs::core::Neighbors::dist_sq) and
/// [`Neighbors::mode()`](molrs::core::Neighbors::mode), which
/// answer `Option` / [`QueryMode`](molrs::core::QueryMode) rather
/// than [`ComputeError`].
pub(crate) use require::{
    is_half_shell, require_angles, require_directions, require_disp, require_dist_sq,
    require_quaternions, require_self_query,
};
pub use shape::{
    CenterOfMass, CenterOfMassInput, CenterOfMassResult, GyrationTensor, GyrationTensorInput,
    GyrationTensorResult, InertiaTensor, InertiaTensorInput, InertiaTensorResult,
};
pub use spec::{
    ComputeSpec, Described, InputForm, InputKind, InputSpec, ParamDefault, ParamKind, ParamSpec,
    ParamValue,
};
pub use spectroscopy::{
    ConductivitySumRule, DielectricSpectrumInput, DielectricSpectrumResult,
    DipoleAutocorrelationSpectrum, DipoleRateCrossSpectrum, EinsteinHelfandSpectrum,
    GreenKuboSpectrum, KramersKronig, KramersKronigCheck, KramersKronigInput, PermittivityPart,
    PlateauValue, PolarizabilityParts, PolarizabilityPartsInput, PolarizabilityPartsResult,
    RamanSpectrum, RamanSpectrumInput, RamanSpectrumResult, RouteAgreement, RouteAgreementCheck,
    RouteAgreementInput, SumRuleCheck, SumRuleInput,
};
pub use time_series::{
    Correlator, CorrelatorAccumulator, CorrelatorFrame, CorrelatorInput, CurrentAcf,
    CurrentAcfInput, Derivative, DerivativeInput, DipoleAcf, DipoleAcfInput, FrequencyGrid,
    LagKernel, LagKernelKind, MeanPolicy, Msd, MsdInput, Normalization, OneSidedSpectrum,
    OnsagerCorrelation, OnsagerInput, Origins, Reduction, ReorientationAcf, ReorientationAcfInput,
    RotationalAutocorrelation, RotationalAutocorrelationInput, Spectrum, SpectrumInput, Stencil,
    Survival, SurvivalAccumulator, SurvivalFrame, SurvivalInput, SurvivalKind,
    SurvivalNormalization, SurvivalOrigin, TimeCorrelation, Unwrap, UnwrapInput, Vacf, VacfInput,
    Window,
};
#[cfg(feature = "voronoi")]
pub use voronoi::{
    FiniteFieldPolarizability, FiniteFieldPolarizabilityInput, LengthUnit, MolecularMoments,
    RadicalVoronoi, RadicalVoronoiInput, VORONOI_BOUNDARY, VoronoiCells, VoronoiDomainAnalysis,
    VoronoiDomainInput, VoronoiDomainResult, VoronoiFace, VoronoiIntegration,
    VoronoiIntegrationInput, VoronoiVoidAnalysis, VoronoiVoidInput, VoronoiVoidResult,
};

// ---------------------------------------------------------------------------
// Tests — spec neighborlist-03-compute, task 1: the column requirement helpers
// ---------------------------------------------------------------------------

/// `require_disp` / `require_dist_sq`: the one place a compute kernel turns
/// "this table never stored that column" into a [`ComputeError::BadShape`].
///
/// A [`Neighbors`](molrs::core::Neighbors) table reports an
/// absent column as `None`, never as a fabricated zero — so every kernel that
/// needs one has to reject `None` itself. These helpers are that rejection,
/// written once: a kernel calls one of them and gets either the column or an
/// error naming what was missing and how many pairs went unanswered.
///
/// The tests pin the **re-exported path** `crate::compute::require_*`; which
/// file inside `compute/` defines them is the implementer's choice.
///
/// The fixture is the two-pair table from the `neighbors` unit tests, so the
/// same hard-coded numbers describe the same pairs on both sides of the crate:
///
/// | k | i | j | `dist_sq` | `disp`        |
/// |---|---|---|-----------|---------------|
/// | 0 | 0 | 1 | 2.0       | (1.0,1.0,0.0) |
/// | 1 | 1 | 3 | 9.0       | (0.0,0.0,3.0) |
#[cfg(test)]
mod require_tests {
    use crate::compute::ComputeError;
    use crate::compute::{require_disp, require_dist_sq};
    use molrs::core::{NeighborColumns, NeighborPair, Neighbors, QueryMode};
    use molrs::op::F;

    /// Two hard-coded half-shell pairs (`i < j`), legal under
    /// `SelfQuery { n_points: 4 }`.
    fn two_pairs() -> [NeighborPair; 2] {
        [
            NeighborPair {
                i: 0,
                j: 1,
                dist_sq: 2.0,
                disp: [1.0, 1.0, 0.0],
            },
            NeighborPair {
                i: 1,
                j: 3,
                dist_sq: 9.0,
                disp: [0.0, 0.0, 3.0],
            },
        ]
    }

    fn table(columns: NeighborColumns) -> Neighbors {
        Neighbors::from_pairs(two_pairs(), columns, QueryMode::SelfQuery { n_points: 4 })
    }

    /// Basics: on a `FULL` table the displacement column comes back as an
    /// `n_pairs × 3` view holding exactly the pushed vectors (Å, exact — these
    /// are copies, not arithmetic).
    #[test]
    fn require_disp_returns_the_full_table_column() {
        let nb = table(NeighborColumns::FULL);
        let disp = require_disp(&nb).expect("FULL table has a disp column");

        assert_eq!(disp.nrows(), 2);
        assert_eq!(disp.ncols(), 3);
        let expected: [[F; 3]; 2] = [[1.0, 1.0, 0.0], [0.0, 0.0, 3.0]];
        for k in 0..2 {
            for c in 0..3 {
                assert!(
                    (disp[[k, c]] - expected[k][c]).abs() <= 1e-12,
                    "disp[{k}][{c}] = {} != {}",
                    disp[[k, c]],
                    expected[k][c]
                );
            }
        }
    }

    /// Basics: a `FULL` table passes, and its squared-distance column is the
    /// `n_pairs` values in table order (Å², exact copies).
    #[test]
    fn require_dist_sq_accepts_the_full_table() {
        let nb = table(NeighborColumns::FULL);
        require_dist_sq(&nb).expect("FULL table has a dist_sq column");

        let d2 = nb.dist_sq().unwrap();
        assert_eq!(d2.len(), nb.n_pairs());
        assert_eq!(d2, &[2.0_f64, 9.0_f64][..]);
    }

    /// Edge: an indices-only table that *does* have pairs is the case that
    /// would otherwise index an empty column — `BadShape`, and the `expected`
    /// text names the missing column and how many pairs it was needed for.
    #[test]
    fn require_disp_on_indices_only_is_bad_shape() {
        let nb = table(NeighborColumns::INDICES_ONLY);
        assert_eq!(
            nb.n_pairs(),
            2,
            "the guard must be tested on a non-empty table"
        );

        let err = require_disp(&nb).expect_err("indices-only table has no disp column");
        let ComputeError::BadShape { expected, got } = err else {
            panic!("expected BadShape, got {err:?}");
        };
        assert!(
            expected.contains("disp"),
            "BadShape.expected must name the missing column: {expected:?}"
        );
        assert!(
            expected.contains('2'),
            "BadShape.expected must name the pair count (2): {expected:?}"
        );
        assert!(!got.is_empty(), "BadShape.got must describe the table");
    }

    /// Edge: same for the radial column — a lean list with pairs but no `d²`.
    #[test]
    fn require_dist_sq_on_indices_only_is_bad_shape() {
        let nb = table(NeighborColumns::INDICES_ONLY);
        assert_eq!(
            nb.n_pairs(),
            2,
            "the guard must be tested on a non-empty table"
        );

        let err = require_dist_sq(&nb).expect_err("indices-only table has no dist_sq column");
        let ComputeError::BadShape { expected, got } = err else {
            panic!("expected BadShape, got {err:?}");
        };
        assert!(
            expected.contains("dist_sq"),
            "BadShape.expected must name the missing column: {expected:?}"
        );
        assert!(
            expected.contains('2'),
            "BadShape.expected must name the pair count (2): {expected:?}"
        );
        assert!(!got.is_empty(), "BadShape.got must describe the table");
    }
}
