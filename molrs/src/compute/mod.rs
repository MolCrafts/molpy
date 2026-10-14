//! Analysis compute modules for molrs molecular simulation.
//!
//! Trajectory analysis (RDF, MSD, transport, spectroscopy, clustering,
//! shape descriptors, PCA/k-means) built around a single unified [`Compute`]
//! trait. Every analysis is stateless — orchestrate from the caller.
//!
//! # Stateless `Compute` — orchestrate from the caller
//!
//! Each [`Compute`] impl is a pure function: `&self` is an immutable
//! parameter bag. Two `compute` calls with identical `frames` + `args`
//! always produce identical output. There is no hidden mutable state,
//! no DAG, no store — just the trait and per-category modules.
//!
//! For DAG orchestration (topological order, diamond reuse, external
//! input validation), use `molpy.compute.Workflow` on the Python side.
//! It composes `Compute` nodes via Python's stdlib `graphlib` and
//! calls each Rust kernel directly.
//!
//! # Unified trait
//!
//! Every analysis implements [`Compute`]. A single frame, a trajectory, and a
//! whole dataset are the same kind of input — a slice `&[&F]` — and each
//! impl decides how to interpret it:
//!
//! - **Whole-sequence accumulators** (RDF) iterate every frame.
//! - **Time-series analyses** (MSD) use `frames[0]` as a reference.
//! - **Per-frame analyses** (Cluster, COM) return one result per frame.
//! - **Matrix consumers** (PCA, k-means) take any [`FeatureMatrix`] — an
//!   `Array2` or a descriptor result — and their results feed each other.
//!
//! # Neighbor tables: what `Args` carries, and which columns a kernel needs
//!
//! Most analyses here are *pair* analyses: they answer a question about every
//! pair of particles lying closer together than a fixed **cutoff** distance.
//! Finding those pairs is not their job. The search lives in
//! [`molrs::core::NeighborList`], and what it
//! produces — a [`Neighbors`](molrs::core::Neighbors) table, a
//! column store holding one row per pair — is handed to the kernel as its
//! `Args`:
//! `&Neighbors` for a single frame, `&Vec<Neighbors>` for a trajectory, in
//! which case there is **one table per frame, index-aligned with `frames`**. A
//! length mismatch is [`ComputeError::DimensionMismatch`], never a silent
//! truncation.
//!
//! ## Columns are opt-in, and a missing one is an error rather than a zero
//!
//! Every table stores each pair's two particle indices `(i, j)`. Its two
//! *physical* columns are stored only if the caller asked for them when the
//! table was materialized, by naming a
//! [`NeighborColumns`](molrs::core::NeighborColumns) policy:
//!
//! - `dist_sq` — the squared pair distance `|r_j − r_i|²` in Å², taken under
//!   the **minimum-image convention**: with periodic boundaries the box is
//!   tiled through space, so a pair has one separation per periodic copy
//!   ("image") of the box, and the search reports only the shortest of them.
//! - `disp` — the displacement `r_j − r_i` itself in Å, from that same
//!   minimum image, pointing from `i` to `j` and **not** normalized. Its length
//!   is the pair distance, so a caller who wants a direction divides by that
//!   length.
//!
//! A column the table never stored reads back as `None`, never as a fabricated
//! zero: a zero displacement is a physically meaningful value (two coincident
//! particles) and would be indistinguishable from real data. So a kernel that
//! needs a column rejects `None` instead of substituting anything, and every
//! kernel rejects it the same way — [`ComputeError::BadShape`], whose
//! `expected` text names the missing column and the number of pairs it was
//! needed for. Nothing here indexes an absent column, which would be an
//! out-of-bounds panic on a native target and an `unreachable` trap under
//! WebAssembly.
//!
//! Which column a kernel needs follows from what it measures:
//!
//! | Needs | Kernels | Materialize the table with |
//! |---|---|---|
//! | `disp` — bond *directions* | [`Steinhardt`], [`Hexatic`] (and through Steinhardt's [`QlmTable`], [`SolidLiquid`] and [`ContinuousCoordination`]), [`Pmft`], [`BondOrientationalOrder`], [`LocalDescriptors`], [`LocalBondProjection`], [`EnvironmentMatch`] | `NeighborColumns::DISP` or `FULL` |
//! | `dist_sq` — distances only | [`Rdf`], [`CorrelationFunction`], [`LocalDensity`] (each takes any [`PairSource`](molrs::core::PairSource)) | `NeighborColumns::DIST_SQ` or `FULL` |
//! | indices only — connectivity | [`Cluster`], [`AngularSeparationNeighbor`] | any policy, `INDICES_ONLY` included |
//!
//! `FULL` means *every column is present*. It never means a bidirectional pair
//! list — pair direction is an independent property, recorded by the table's
//! [`QueryMode`](molrs::core::QueryMode). A **self-query**
//! searches one point set against itself and is half-shell: each unordered pair
//! appears exactly once, with `i < j`. A **cross-query** searches query points
//! against a separate reference set and is directed, with both orderings
//! present. Radial-distribution normalization is the one place that distinction
//! changes a number — a half-shell count is short by a factor of two — see
//! [`RdfResult`].
//!
//! A kernel that only streams pairs takes a
//! [`PairSource`](molrs::core::PairSource): a materialized table, or a built
//! [`NeighborList`](molrs::core::NeighborList), which streams pairs straight
//! out of its cell list without ever materializing them — the bounded-memory
//! path for large systems. [`HBonds`] and [`PairPresence`] take the caller's
//! per-frame tables; only [`VanHove`] builds its own cross-query, one per time
//! origin, because its pair set depends on the origin and the lag.
//!
//! # Binning and grids
//!
//! Every histogram bins through [`Axis`] (half-open `[lo, hi)` bins; a NaN
//! sample is [`ComputeError::NonFinite`]) into a [`Histogram`], and every
//! density grid spreads a [`SpreadKernel`] ([`Gaussian`], [`Indicator`]) over
//! a [`RegularGrid`], which wraps only its periodic axes.
//!
//! # Results
//!
//! Every Compute output implements [`ComputeResult`]. Accumulating outputs
//! (RDF) override [`finalize`](ComputeResult::finalize) to normalize; other
//! outputs use the default no-op.
//!
//! # Time correlation: one estimator, named surfaces
//!
//! Every lagged correlation is one configuration of [`Correlator`] (FFT batch
//! evaluation plus a bounded-memory streaming accumulator) and returns a
//! [`TimeCorrelation`]. The quantities chemists name are thin surfaces over it;
//! the step that turns a curve into a coefficient is the analyst's explicit,
//! parameterized choice:
//!
//! | Surface | Raw curve | Downstream fit |
//! |--------|-----------|----------------|
//! | [`Vacf`] | velocity ACF | [`Spectrum`] (VDOS) / [`CumulativeTrapezoid`] (D) |
//! | [`Msd`] | self-MSD; on the collective charge dipole, the Einstein–Helfand curve | [`LinearFit`] (D = slope/2d, σ) |
//! | [`CurrentAcf`] | current ACF | [`CumulativeTrapezoid`] (σ) |
//! | [`DipoleAcf`] | dipole fluctuation ACF | [`DebyeFit`] (τ_D) / [`DipoleAutocorrelationSpectrum`] |
//! | [`OnsagerCorrelation`] | Onsager L_ij displacement correlations | [`LinearFit`] per pair |
//! | [`ReorientationAcf`], [`RotationalAutocorrelation`] | orientational decay | [`DebyeFit`] |
//! | [`Survival`] | continuous / intermittent survival of a [`Presence`] ([`PairPresence`], [`HBondsResult::presence`]) | [`CumulativeTrapezoid`] (lifetime) |
//!
//! Units follow the MD convention of the caller (time in the `dt` unit,
//! velocities/dipoles as supplied); the fits document the MD→SI prefactors.
//!
//! ```
//! use molrs::compute::{Compute, CumulativeTrapezoid, CumulativeTrapezoidInput, Vacf, VacfInput};
//! use ndarray::Array2;
//!
//! let velocities = Array2::from_shape_fn((64, 3), |(t, d)| ((t * 3 + d) as f64 * 0.7).sin());
//! let raw = Vacf::new(16).compute(VacfInput { velocities: velocities.view(), dt: 1.0 })?;
//! let running = CumulativeTrapezoid::new()
//!     .compute(CumulativeTrapezoidInput { correlation: &raw, channel: 0 })?; // D·d in MD units
//! # Ok::<(), molrs::compute::ComputeError>(())
//! ```
//!
//! # Raw observable, then transform: spectra
//!
//! Each spectrum is an explicit composition — a series operation, a
//! correlation, a transform; no spectrum owns an estimator or an FFT:
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
//! Spectral units:
//!
//! | quantity   | unit |
//! |------------|------|
//! | time / dt  | fs (vibrational spectra); any consistent unit (ε(ω)) |
//! | frequency  | cm⁻¹ ([`RamanSpectrum`]); rad per unit of dt ([`Spectrum`], ε(ω)) |
//! | intensity  | arb. |
//!
//! # Implementation modules (`compute/<name>/`)
//!
//! One folder per kernel family. **UI/catalog categories** (see
//! `molrs-wasm` `molrsComputeCatalog`, catalog v3) follow freud's top-level
//! modules and are not always 1:1 with folder names:
//! - g(r) ([`Rdf`]) lives in `rdf/` but is catalogued under **density**
//!   (`freud.density.Rdf`)
//! - Voronoi kernels are catalogued under **locality**
//!   (`freud.locality.Voronoi`)
//! - van Hove / pair presence live under **transport** (with VACF/diffusion)
//! - static dielectric is under **spectroscopy**
//! - cluster radius-of-gyration ([`GyrationTensorResult::radius_of_gyration`]) is
//!   under **cluster** (freud.cluster.ClusterProperties)
//!
//! | Folder | Methods |
//! |----------|---------|
//! | `binning` | [`Axis`], [`Histogram`] — the one binning implementation |
//! | `grid` | [`RegularGrid`], [`SpreadKernel`] ([`Gaussian`], [`Indicator`]) — the one grid-spreading implementation |
//! | `rdf` | pair distribution g(r) (+ streaming [`RdfAccumulator`]) |
//! | `time_series` | [`Correlator`] and its surfaces (MSD, VACF, Onsager, dipole / current ACFs, reorientation, survival), [`Derivative`], [`Unwrap`], [`Spectrum`] |
//! | `spectroscopy` | [`PolarizabilityParts`], [`RamanSpectrum`], the four ε(ω) routes and their checks |
//! | `fitting` | fits on a [`TimeCorrelation`]: [`LinearFit`], [`CumulativeTrapezoid`], [`Plateau`], [`DebyeFit`] |
//! | `dynamics` | van Hove G(r, t), pair presence |
//! | `dielectric` | [`DipoleMoment`], [`StaticDielectric`], [`EwaldBoundary`] |
//! | `cluster` | cluster analysis (freud's `Cluster`): connected components of a neighbor table, optionally merged by keys; the result is a [`Partition`] |
//! | `shape` | per-group [`CenterOfMass`], [`GyrationTensor`] (and R_g), [`InertiaTensor`] over a [`Partition`] — the one group reduction |
//! | `decomposition` | [`Pca`] of any [`FeatureMatrix`] |
//! | `kmeans` | [`Kmeans`] over any [`FeatureMatrix`]; the result is a [`Partition`] |
//! | `density` | correlation function, Gaussian/local density, spatial distribution, voxelization |
//! | `order` | the one bond-harmonic average ([`QlmTable`]) behind Steinhardt, hexatic, solid-liquid, continuous coordination; nematic, cubatic |
//! | `environment` | bond order, local descriptors, environment matching, … |
//! | `diffraction` | S(k) (Debye & direct), diffraction pattern |
//! | `pmft` | potentials of mean force and torque: one [`Pmft`] over a [`PairCoordinate`] ([`R12`], [`Xy`], [`Xyt`], [`Xyz`]) |
//! | `distribution` | distance/angle/dihedral distribution functions |
//! | `hbond` | hydrogen-bond detection, presence series, network components |
//! | `kinetic` | [`KineticEnergy`] (and its temperature) of one state |
//! | `voronoi` | radical Voronoi cells, domains, voids, density integration, finite-field polarizability (feature `voronoi`) |

mod analysis_contract;
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
mod spectroscopy;
mod time_series;
#[cfg(feature = "voronoi")]
mod voronoi;

// Re-exports
pub use analysis_contract::{Check, ComputeResult, DescriptorRow, Fit, LegacyCompute};
pub use binning::{Axis, Histogram};
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
pub use grid::{Gaussian, Indicator, RegularGrid, SpreadKernel};
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
pub use spectroscopy::{
    ConductivitySumRule, DielectricSpectrumInput, DielectricSpectrumResult,
    DipoleAutocorrelationSpectrum, DipoleRateCrossSpectrum, EinsteinHelfandSpectrum,
    GreenKuboSpectrum, KramersKronig, KramersKronigCheck, KramersKronigInput, PermittivityPart,
    PolarizabilityParts, PolarizabilityPartsInput, PolarizabilityPartsResult, RamanSpectrum,
    RamanSpectrumInput, RamanSpectrumResult, RouteAgreement, RouteAgreementCheck,
    RouteAgreementInput, SumRuleCheck, SumRuleInput,
};
pub use time_series::{
    Correlator, CorrelatorAccumulator, CorrelatorFrame, CorrelatorInput, CurrentAcf,
    CurrentAcfInput, Derivative, DerivativeInput, DipoleAcf, DipoleAcfInput, FrequencyGrid,
    LagKernel, MeanPolicy, Msd, MsdInput, Normalization, OneSidedSpectrum, OnsagerCorrelation,
    OnsagerInput, Origins, Reduction, ReorientationAcf, ReorientationAcfInput,
    RotationalAutocorrelation, RotationalAutocorrelationInput, Spectrum, SpectrumInput, Stencil,
    Survival, SurvivalInput, SurvivalKind, TimeCorrelation, Unwrap, UnwrapInput, Vacf, VacfInput,
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
