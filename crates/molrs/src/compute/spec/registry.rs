//! [`ComputeSpec::all`] — the one list of every public compute type.

use std::sync::LazyLock;

use crate::compute::{
    AngularSeparationGlobal, AngularSeparationNeighbor, BondOrientationalOrder, CenterOfMass,
    Cluster, CombinedDistribution, ComputeSpec, ConductivitySumRule, ContinuousCoordination,
    CorrelationFunction, Correlator, Cubatic, CumulativeTrapezoid, CurrentAcf, DebyeFit,
    Derivative, Described, DiffractionPattern, DipoleAcf, DipoleAutocorrelationSpectrum,
    DipoleMoment, DipoleRateCrossSpectrum, DistributionFunction, EinsteinHelfandSpectrum,
    EnvironmentMatch, GaussianDensity, GreenKuboSpectrum, GyrationTensor, HBonds, Hexatic,
    InertiaTensor, InternalCoordinate, KineticEnergy, Kmeans, KramersKronig, LinearFit,
    LocalBondProjection, LocalDensity, LocalDescriptors, Msd, Nematic, OnsagerCorrelation,
    PairPresence, Pca, Plateau, Pmft, PolarizabilityParts, R12, RamanSpectrum, Rdf,
    ReorientationAcf, RotationalAutocorrelation, RouteAgreement, SolidLiquid, SpatialDistribution,
    Spectrum, SphereVoxelization, StaticDielectric, StaticStructureFactorDebye,
    StaticStructureFactorDirect, Steinhardt, Survival, Unwrap, Vacf, VanHove, Xy, Xyt, Xyz,
};

static ALL: LazyLock<Vec<ComputeSpec>> = LazyLock::new(|| {
    [
        // time series: the estimator, its surfaces, the series operations
        Correlator::spec(),
        Msd::spec(),
        Vacf::spec(),
        OnsagerCorrelation::spec(),
        DipoleAcf::spec(),
        CurrentAcf::spec(),
        ReorientationAcf::spec(),
        RotationalAutocorrelation::spec(),
        Survival::spec(),
        Derivative::spec(),
        Unwrap::spec(),
        Spectrum::spec(),
        // fits
        CumulativeTrapezoid::spec(),
        DebyeFit::spec(),
        LinearFit::spec(),
        Plateau::spec(),
        // dielectric, spectroscopy and their checks
        KineticEnergy::spec(),
        DipoleMoment::spec(),
        StaticDielectric::spec(),
        EinsteinHelfandSpectrum::spec(),
        GreenKuboSpectrum::spec(),
        DipoleRateCrossSpectrum::spec(),
        DipoleAutocorrelationSpectrum::spec(),
        KramersKronig::spec(),
        ConductivitySumRule::spec(),
        RouteAgreement::spec(),
        PolarizabilityParts::spec(),
        RamanSpectrum::spec(),
        // binned and gridded structure
        Rdf::spec(),
        CorrelationFunction::spec(),
        LocalDensity::spec(),
        GaussianDensity::spec(),
        SphereVoxelization::spec(),
        SpatialDistribution::spec(),
        DistributionFunction::<InternalCoordinate>::spec(),
        CombinedDistribution::<2>::spec(),
        StaticStructureFactorDebye::spec(),
        StaticStructureFactorDirect::spec(),
        DiffractionPattern::spec(),
        // dynamics
        PairPresence::spec(),
        VanHove::spec(),
        HBonds::spec(),
        // order and environment
        Steinhardt::spec(),
        Hexatic::spec(),
        SolidLiquid::spec(),
        ContinuousCoordination::spec(),
        Nematic::spec(),
        Cubatic::spec(),
        BondOrientationalOrder::spec(),
        LocalDescriptors::spec(),
        LocalBondProjection::spec(),
        EnvironmentMatch::spec(),
        AngularSeparationGlobal::spec(),
        AngularSeparationNeighbor::spec(),
        Pmft::<R12>::spec(),
        Pmft::<Xy>::spec(),
        Pmft::<Xyt>::spec(),
        Pmft::<Xyz>::spec(),
        // groups and features
        Cluster::spec(),
        CenterOfMass::spec(),
        GyrationTensor::spec(),
        InertiaTensor::spec(),
        Pca::spec(),
        Kmeans::spec(),
    ]
    .into_iter()
    .chain(voronoi_specs())
    .collect()
});

#[cfg(feature = "voronoi")]
fn voronoi_specs() -> Vec<ComputeSpec> {
    use crate::compute::{
        FiniteFieldPolarizability, RadicalVoronoi, VoronoiDomainAnalysis, VoronoiIntegration,
        VoronoiVoidAnalysis,
    };
    vec![
        RadicalVoronoi::spec(),
        VoronoiIntegration::spec(),
        VoronoiDomainAnalysis::spec(),
        VoronoiVoidAnalysis::spec(),
        FiniteFieldPolarizability::spec(),
    ]
}

#[cfg(not(feature = "voronoi"))]
fn voronoi_specs() -> Vec<ComputeSpec> {
    Vec::new()
}

impl ComputeSpec {
    /// The descriptor of every public compute type compiled into this build,
    /// one per name (each `Pmft<C>` instantiation is its own entry).
    pub fn all() -> &'static [ComputeSpec] {
        &ALL
    }

    /// The descriptor named `name`.
    pub fn find(name: &str) -> Option<&'static ComputeSpec> {
        ALL.iter().find(|spec| spec.name == name)
    }
}

#[cfg(test)]
mod tests;
