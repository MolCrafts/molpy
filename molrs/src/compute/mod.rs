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
//! - **Matrix consumers** (PCA, k-means) take upstream per-frame outputs as
//!   rows (via [`DescriptorRow`]).
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
//! | `disp` — bond *directions* | [`Steinhardt`], [`Hexatic`], [`SolidLiquid`], [`ContinuousCoordination`] (the last three via [`steinhardt_qlm`]), every PMFT kernel, [`BondOrientationalOrder`], [`LocalDescriptors`], [`LocalBondProjection`], [`EnvironmentMatch`] | `NeighborColumns::DISP` or `FULL` |
//! | `dist_sq` — distances only | [`Rdf`] when fed a materialized table, [`CorrelationFunction`], [`LocalDensity`] | `NeighborColumns::DIST_SQ` or `FULL` |
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
//! [`RdfMode`].
//!
//! Some entry points take no table at all, because they run a search of their
//! own per call: [`Rdf::compute_self`], [`Rdf::compute_frame`] and
//! [`Rdf::compute_cross`] stream pairs straight out of the cell list without
//! ever materializing them, and [`HBonds`] and [`VanHove`] build a cross-query
//! internally, one per frame and one per time origin respectively.
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
//! | [`Survival`] | continuous / intermittent survival | [`CumulativeTrapezoid`] (lifetime) |
//!
//! Units follow the MD convention of the caller (time in the `dt` unit,
//! velocities/dipoles as supplied); the fits document the MD→SI prefactors.
//!
//! ```
//! use molrs::compute::{Compute, CumulativeTrapezoid, Fit, Vacf, VacfInput};
//! use ndarray::Array2;
//!
//! let velocities = Array2::from_shape_fn((64, 3), |(t, d)| ((t * 3 + d) as f64 * 0.7).sin());
//! let raw = Vacf::new(16).compute(VacfInput { velocities: velocities.view(), dt: 1.0 })?;
//! let d = CumulativeTrapezoid.fit((&raw.channel(0).to_owned(), 1.0, None))?; // D·d in MD units
//! # Ok::<(), molrs::compute::ComputeError>(())
//! ```
//!
//! # Raw observable, then fit: spectra
//!
//! Each spectrum is an explicit two-step composition — a raw compute produces
//! an (unwindowed) correlation function, a fit applies window + FFT (+
//! physical prefactors):
//!
//! | Spectrum | Raw compute | Fit transform |
//! |----------|-------------|---------------|
//! | VDOS | [`Vacf`] (velocity ACF) | [`PowerSpectrum`] |
//! | IR | [`IrFlux`] (dipole-flux ACF) | [`IrSpectrum`] |
//! | Raman | [`RamanTensor`] (polarizability iso/aniso ACFs) | [`RamanSpectrum`] |
//! | VCD | [`VcdCrossFlux`] (μ̇ × ṁ cross-correlation) | [`VcdSpectrum`] |
//! | ROA | [`RoaCrossTensor`] (α̇ × Ġ′ cross-correlations) | [`RoaSpectrum`] |
//! | Resonance Raman | [`ResonanceRamanTensor`] (resonant iso/aniso ACFs) | [`ResonanceRamanSpectrum`] |
//! | Dielectric ε(ω) | [`DipoleAcf`] / [`CurrentAcf`] / [`Correlator`] cross of (Ṁ, M) | [`EinsteinHelfandSpectrum`] / [`GreenKuboSpectrum`] / [`DipoleAutocorrelationSpectrum`] / [`DipoleRateCrossSpectrum`] |
//!
//! Spectral units:
//!
//! | quantity   | unit |
//! |------------|------|
//! | time / dt  | fs   |
//! | frequency  | cm⁻¹ |
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
//! - van Hove / pair survival live under **transport** (with VACF/diffusion)
//! - static dielectric is under **spectroscopy**
//! - cluster radius-of-gyration is under **cluster** (freud.cluster.ClusterProperties)
//!
//! | Folder | Methods |
//! |----------|---------|
//! | `rdf` | pair distribution g(r) (+ streaming [`RdfAccumulator`]) |
//! | `time_series` | [`Correlator`] and its surfaces (MSD, VACF, Onsager, dipole / current ACFs, reorientation, survival), [`Derivative`], [`Unwrap`], [`Spectrum`] |
//! | `spectroscopy` | IR / Raman / VCD / ROA / resonance-Raman raw correlators + spectral transforms, dielectric spectra |
//! | `fitting` | generic curve fits: [`LinearFit`], [`CumulativeTrapezoid`], [`Plateau`], [`DebyeFit`] |
//! | `dynamics` | van Hove G(r, t), pair survival |
//! | `dielectric` | static dielectric constant from dipole fluctuations |
//! | `cluster` | cluster analysis (freud's `Cluster`): connected components by distance + per-cluster properties |
//! | `shape` | center of mass, cluster centers, gyration/inertia tensors, Rg |
//! | `decomposition` | PCA projection |
//! | `kmeans` | k-means over a PCA projection |
//! | `density` | correlation function, Gaussian/local density, spatial distribution, voxelization |
//! | `order` | Steinhardt, hexatic, nematic, cubatic, solid-liquid, … |
//! | `environment` | bond order, local descriptors, environment matching, … |
//! | `diffraction` | S(k) (Debye & direct), diffraction pattern |
//! | `pmft` | potentials of mean force and torque (R12/XY/XYT/XYZ) |
//! | `distribution` | distance/angle/dihedral distribution functions |
//! | `hbond` | hydrogen-bond detection, lifetimes, network components |
//! | `kinetic` | kinetic energy, kinetic temperature, centre-of-mass velocity of one state |
//! | `voronoi` | radical Voronoi cells, domains, voids (feature `voronoi`) |

mod analysis_contract;
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
mod hbond;
mod kinetic;
mod kmeans;
mod named_enum;
mod order;
mod pmft;
pub(crate) mod positions;
mod rdf;
pub(crate) mod require;
mod shape;
mod spectroscopy;
mod time_series;
#[cfg(feature = "voronoi")]
mod voronoi;

// Re-exports
pub use analysis_contract::{Check, ComputeResult, DescriptorRow, Fit, LegacyCompute};
pub use cluster::{Cluster, ClusterProperties, ClusterPropertiesResult, ClusterResult};
pub use contract::{Accumulate, Batch, Compute, FeatureMatrix, Partition, Stream, Verdict};
pub use decomposition::{Pca, PcaResult};
pub use density::{
    CorrelationFunction, CorrelationFunctionArgs, CorrelationFunctionResult, GaussianDensity,
    GaussianDensityResult, GridSpec, LocalDensity, LocalDensityResult, SpatialDistribution,
    SpatialDistributionResult, SphereVoxelization, SphereVoxelizationResult,
};
pub use dielectric::{
    EwaldBoundary, StaticDielectricResult, current_density, decompose_current, dipole_moment,
    static_dielectric_constant, static_dielectric_constant_components,
};
pub use diffraction::{
    DiffractionPattern, DiffractionPatternResult, StaticStructureFactorDebye,
    StaticStructureFactorDebyeResult, StaticStructureFactorDirect,
    StaticStructureFactorDirectResult,
};
pub use distribution::{
    AngleObservable, AtomGroups, AxisSpec, CombinedDistribution, CombinedDistributionResult,
    DihedralObservable, DistanceObservable, DistributionFunction, DistributionResult, Histogram1d,
    InternalCoordinate, Observable, renormalize_density,
};
pub use dynamics::{PairSurvivalResult, SurvivalMethod, VanHove, VanHoveResult, pair_survival_tcf};
pub use environment::{
    AngularSeparationGlobal, AngularSeparationGlobalArgs, AngularSeparationGlobalResult,
    AngularSeparationNeighbor, AngularSeparationNeighborArgs, AngularSeparationNeighborResult,
    BondOrientationalOrder, BondOrientationalOrderResult, EnvironmentMatch, EnvironmentMatchResult,
    LocalBondProjection, LocalBondProjectionArgs, LocalBondProjectionResult, LocalDescriptors,
    LocalDescriptorsResult,
};
pub use error::ComputeError;
pub use fitting::{
    CumulativeTrapezoid, CumulativeTrapezoidResult, DebyeFit, DebyeFitResult, LinearFit,
    LinearFitResult, Plateau, PlateauResult,
};
pub use hbond::{
    HBond, HBondCriterion, HBondDistanceKind, HBondLifetimeResult, HBondNetworkResult, HBonds,
    HBondsResult, hbond_components, hbond_lifetimes, presence_from_hbonds,
};
pub use kinetic::{center_of_mass_velocity, kinetic_energy, kinetic_temperature};
pub use kmeans::{Kmeans, KmeansResult};
pub use order::{
    ContinuousCoordination, ContinuousCoordinationResult, Cubatic, CubaticResult, Hexatic,
    HexaticResult, Nematic, NematicResult, SolidLiquid, SolidLiquidResult, Steinhardt,
    SteinhardtResult, steinhardt_qlm,
};
pub use pmft::{
    PmftR12, PmftR12Args, PmftR12Result, PmftXy, PmftXyArgs, PmftXyResult, PmftXyt, PmftXytArgs,
    PmftXytResult, PmftXyz, PmftXyzArgs, PmftXyzResult, orientation_quaternions,
    planar_orientation_angles,
};
pub use rdf::{Rdf, RdfAccumulator, RdfMode, RdfResult};
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
pub(crate) use require::{require_disp, require_dist_sq, require_self_query};
pub use shape::{
    CenterOfMass, CenterOfMassResult, ClusterCenters, ClusterCentersResult, GyrationTensor,
    GyrationTensorResult, InertiaTensor, InertiaTensorResult, RadiusOfGyration,
    RadiusOfGyrationResult,
};
pub use spectroscopy::{
    ConductivitySumRule, DielectricSpectrumResult, DipoleAutocorrelationSpectrum,
    DipoleRateCrossSpectrum, EinsteinHelfandSpectrum, GreenKuboSpectrum, IrFlux, IrFluxArgs,
    IrFluxResult, IrSpectrum, KramersKronig, KramersKronigCheck, PowerSpectrum, RamanSpectrum,
    RamanSpectrumResult, RamanTensor, RamanTensorArgs, RamanTensorResult, ResonanceRamanSpectrum,
    ResonanceRamanTensor, ResonanceRamanTensorArgs, RoaCrossTensor, RoaCrossTensorArgs,
    RoaCrossTensorResult, RoaSpectrum, RouteAgreement, RouteAgreementCheck, SpectrumResult,
    SumRuleCheck, VcdCrossFlux, VcdCrossFluxArgs, VcdCrossFluxResult, VcdSpectrum,
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
    DensityGrid, MolecularMoments, RadicalVoronoi, VORONOI_BOUNDARY, VoronoiCells,
    VoronoiDomainAnalysis, VoronoiDomainResult, VoronoiFace, VoronoiIntegration,
    VoronoiVoidAnalysis, VoronoiVoidResult, polarizability_finite_field,
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

    /// Basics: on a `FULL` table the squared-distance column comes back as a
    /// slice of `n_pairs` values, in table order (Å², exact copies).
    #[test]
    fn require_dist_sq_returns_the_full_table_column() {
        let nb = table(NeighborColumns::FULL);
        let d2 = require_dist_sq(&nb).expect("FULL table has a dist_sq column");

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
