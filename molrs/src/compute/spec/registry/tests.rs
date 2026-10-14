//! The registry is complete, its names are unique, and every default it
//! claims is what the type's own constructor produces.

use std::collections::{BTreeMap, BTreeSet};
use std::fs;
use std::path::Path;

use molrs::core::UnitPreset;
use molrs::op::F;

use super::*;
use crate::compute::{
    Axis, ComputeError, InputKind, LagKernel, ParamDefault, ParamKind, ParamValue, RegularGrid,
    Stencil, SurvivalKind, Window,
};

/// `impl Compute for` types that are deliberately not in the registry.
const NOT_REGISTERED: &[(&str, &str)] = &[
    (
        "Batch",
        "a combinator: `Batch<C>` maps `C` over snapshots, so its spec is `C`'s",
    ),
    (
        "BondAverage",
        "crate-internal: Steinhardt / Hexatic are its public faces",
    ),
];

/// A generic compute registered once per instantiation.
const INSTANTIATIONS: &[(&str, &[&str])] =
    &[("Pmft", &["PmftR12", "PmftXy", "PmftXyt", "PmftXyz"])];

/// Every type the non-test source of `compute` implements `Compute` for,
/// read off the `impl … Compute for Name` lines at the start of a line (test
/// modules are indented or live in `tests.rs` / `product_tests.rs`).
fn compute_impls() -> BTreeSet<String> {
    fn visit(dir: &Path, found: &mut BTreeSet<String>) {
        for entry in fs::read_dir(dir).expect("compute source is readable") {
            let path = entry.expect("directory entry").path();
            if path.is_dir() {
                if cfg!(feature = "voronoi") || !path.ends_with("voronoi") {
                    visit(&path, found);
                }
                continue;
            }
            let name = path.file_name().and_then(|n| n.to_str()).unwrap_or("");
            if !name.ends_with(".rs") || matches!(name, "tests.rs" | "product_tests.rs") {
                continue;
            }
            let text = fs::read_to_string(&path).expect("source file is readable");
            for line in text.lines().filter(|line| line.starts_with("impl")) {
                if let Some(rest) = line.split(" Compute for ").nth(1) {
                    found.insert(
                        rest.chars()
                            .take_while(|c| c.is_alphanumeric() || *c == '_')
                            .collect(),
                    );
                }
            }
        }
    }
    let mut found = BTreeSet::new();
    visit(
        &Path::new(env!("CARGO_MANIFEST_DIR")).join("src/compute"),
        &mut found,
    );
    found
}

/// Adding a public compute without a spec, or registering a name no compute
/// type answers to, fails here.
#[test]
fn the_registry_names_every_compute_type() {
    let instantiations: BTreeMap<&str, &[&str]> = INSTANTIATIONS.iter().copied().collect();
    let excluded: BTreeSet<&str> = NOT_REGISTERED.iter().map(|(name, _)| *name).collect();
    let impls = compute_impls();
    assert!(
        impls.contains("Rdf") && impls.contains("Pmft") && impls.contains("Batch"),
        "the source scan found {impls:?}"
    );
    for name in &excluded {
        assert!(impls.contains(*name), "{name} no longer implements Compute");
    }

    let expected: BTreeSet<&str> = impls
        .iter()
        .map(String::as_str)
        .filter(|name| !excluded.contains(name))
        .flat_map(|name| {
            instantiations
                .get(name)
                .map_or_else(|| vec![name], |names| names.to_vec())
        })
        .collect();
    let registered: BTreeSet<&str> = ComputeSpec::all().iter().map(|s| s.name).collect();
    assert_eq!(registered, expected);
}

#[test]
fn every_name_is_unique_and_found() {
    let all = ComputeSpec::all();
    let names: BTreeSet<&str> = all.iter().map(|s| s.name).collect();
    assert_eq!(names.len(), all.len(), "duplicate names in the registry");
    for spec in all {
        assert_eq!(ComputeSpec::find(spec.name), Some(spec));
    }
    assert_eq!(ComputeSpec::find("NoSuchCompute"), None);
}

/// Within one spec, parameter and input names are unique, and every default
/// is a value of its parameter's kind, or refers to a parameter that is.
#[test]
fn every_default_fits_its_kind() {
    for spec in ComputeSpec::all() {
        let params: BTreeSet<&str> = spec.params.iter().map(|p| p.name).collect();
        assert_eq!(params.len(), spec.params.len(), "{}", spec.name);
        let inputs: BTreeSet<&str> = spec.inputs.iter().map(|i| i.name).collect();
        assert_eq!(inputs.len(), spec.inputs.len(), "{}", spec.name);
        for input in spec.inputs {
            if let InputKind::Choice(names) = input.kind {
                assert!(!names.is_empty(), "{}.{}", spec.name, input.name);
            }
        }
        for param in &spec.params {
            let what = format!("{}.{}", spec.name, param.name);
            match &param.default {
                ParamDefault::Required | ParamDefault::RequiredIn(_) => {}
                ParamDefault::Value(value) => assert!(fits(param.kind, value), "{what}"),
                ParamDefault::SameAs(other) => {
                    let other = spec.param(other).expect("SameAs names a parameter");
                    assert_eq!(other.kind, param.kind, "{what}");
                }
                ParamDefault::Scaled { of, factor } => {
                    let of = spec.param(of).expect("Scaled names a parameter");
                    assert_eq!((of.kind, param.kind), (ParamKind::Float, ParamKind::Float));
                    assert!(factor.is_finite(), "{what}");
                }
            }
        }
    }
}

fn fits(kind: ParamKind, value: &ParamValue) -> bool {
    match (kind, value) {
        (_, ParamValue::Unset) => true,
        (ParamKind::Float, ParamValue::Float(v)) => v.is_finite(),
        (ParamKind::Choice(names), ParamValue::Name(name)) => names.contains(name),
        (ParamKind::Int, ParamValue::Int(_))
        | (ParamKind::Bool, ParamValue::Bool(_))
        | (ParamKind::IntList, ParamValue::IntList(_))
        | (ParamKind::FloatList, ParamValue::FloatList(_))
        | (ParamKind::Range, ParamValue::Range(..)) => true,
        _ => false,
    }
}

/// One registered type, built from its required arguments alone.
struct Case {
    spec: ComputeSpec,
    tunables: Vec<(&'static str, ParamValue)>,
    /// The required arguments a `SameAs` / `Scaled` default refers to.
    given: Vec<(&'static str, ParamValue)>,
}

fn case<T: Described>(instance: T) -> Case {
    Case {
        spec: T::spec(),
        tunables: instance.tunables(),
        given: Vec::new(),
    }
}

impl Case {
    fn given(self, name: &'static str, value: F) -> Self {
        let mut given = self.given;
        given.push((name, ParamValue::Float(value)));
        Self { given, ..self }
    }

    fn given_value(&self, name: &str) -> &ParamValue {
        let found = self.given.iter().find(|(n, _)| *n == name);
        &found
            .unwrap_or_else(|| panic!("{}: give `{name}` to the case", self.spec.name))
            .1
    }

    /// The instance holds exactly the spec's defaults, in spec order.
    fn check(&self) {
        let name = self.spec.name;
        let defaulted: Vec<_> = self
            .spec
            .params
            .iter()
            .filter(|p| !p.is_required())
            .collect();
        let read: Vec<&str> = self.tunables.iter().map(|(n, _)| *n).collect();
        let claimed: Vec<&str> = defaulted.iter().map(|p| p.name).collect();
        assert_eq!(read, claimed, "{name}: readback names");
        for (param, (_, held)) in defaulted.iter().zip(&self.tunables) {
            let what = format!("{name}.{}", param.name);
            assert!(fits(param.kind, held), "{what}: {held:?}");
            let expected = match &param.default {
                ParamDefault::Value(value) => value.clone(),
                ParamDefault::SameAs(other) => self.given_value(other).clone(),
                ParamDefault::Scaled { of, factor } => match self.given_value(of) {
                    ParamValue::Float(v) => ParamValue::Float(v * factor),
                    other => panic!("{what}: scaled by a non-float {other:?}"),
                },
                ParamDefault::Required | ParamDefault::RequiredIn(_) => unreachable!(),
            };
            assert_eq!(held, &expected, "{what}");
        }
    }
}

fn cases() -> Result<Vec<Case>, ComputeError> {
    let units = UnitPreset::real;
    let axis = || Axis::new(0.0, 5.0, 10);
    let grid = RegularGrid::new(
        [0.0; 3],
        [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]],
        [4; 3],
        [true; 3],
    )?;
    #[cfg_attr(not(feature = "voronoi"), allow(unused_mut))]
    let mut cases = vec![
        case(Correlator::new(LagKernel::Product, 8)),
        case(Msd::new(8)),
        case(Vacf::new(8)),
        case(OnsagerCorrelation::new(8)),
        case(DipoleAcf::new(8)),
        case(CurrentAcf::new(8)),
        case(ReorientationAcf::new(2, 8)),
        case(RotationalAutocorrelation::new(2, 8)),
        case(Survival::new(SurvivalKind::Continuous, 8)),
        case(Derivative::new(Stencil::Central)),
        case(Unwrap::new()),
        case(Spectrum::new(Window::Hann, 2)?),
        case(CumulativeTrapezoid::new()),
        case(DebyeFit),
        case(LinearFit::new(0.2, 0.8)?),
        case(Plateau::new(0.2, 0.8)?),
        case(KineticEnergy::new(units())?),
        case(DipoleMoment::new()),
        case(StaticDielectric::new(units())),
        case(EinsteinHelfandSpectrum::new(units())),
        case(GreenKuboSpectrum::new(units())),
        case(DipoleRateCrossSpectrum::new(units())),
        case(DipoleAutocorrelationSpectrum::new(units())),
        case(KramersKronig::new()),
        case(ConductivitySumRule::new(units())),
        case(RouteAgreement::new()),
        case(PolarizabilityParts::new()),
        case(RamanSpectrum::new()),
        case(Rdf::new(10, 5.0, 0.0)?),
        case(CorrelationFunction::new(10, 5.0, 0.0)?),
        case(LocalDensity::new(2.0)?),
        case(GaussianDensity::new([4; 3], 0.5)?).given("sigma", 0.5),
        case(SphereVoxelization::new([4; 3], 1.0)?),
        case(SpatialDistribution::new(grid)),
        case(DistributionFunction::new(
            InternalCoordinate::Distance,
            axis()?,
        )),
        case(CombinedDistribution::new(
            [InternalCoordinate::Distance, InternalCoordinate::Angle],
            [axis()?, Axis::new(0.0, 3.0, 6)?],
        )?),
        case(StaticStructureFactorDebye::new(&[1.0, 2.0])?),
        case(StaticStructureFactorDirect::isotropic(5.0, 10)?),
        case(DiffractionPattern::new(16, 0.5)?).given("sigma", 0.5),
        case(PairPresence::new(2.0, true)?).given("r0", 2.0),
        case(VanHove::new(axis()?, vec![1, 2])?),
        case(HBonds::default()),
        case(Steinhardt::new(&[4, 6])?),
        case(Hexatic::new(6)?),
        case(SolidLiquid::new()),
        case(ContinuousCoordination::new(2.0)?),
        case(Nematic::new()),
        case(Cubatic::new()),
        case(BondOrientationalOrder::new(4, 8)?),
        case(LocalDescriptors::new(4)),
        case(LocalBondProjection::new()),
        case(EnvironmentMatch::new(0.1)?),
        case(AngularSeparationGlobal::new()),
        case(AngularSeparationNeighbor::new()),
        case(Pmft::<R12>::new(3.0, 10, 8, 8)?),
        case(Pmft::<Xy>::new(3.0, 3.0, 10, 10)?),
        case(Pmft::<Xyt>::new(3.0, 3.0, 10, 10, 8)?),
        case(Pmft::<Xyz>::new(3.0, 3.0, 3.0, 10, 10, 10)?),
        case(Cluster::new()),
        case(CenterOfMass::new()),
        case(GyrationTensor::new()),
        case(InertiaTensor::new()),
        case(Pca::new(2)?),
        case(Kmeans::new(3)?),
    ];
    #[cfg(feature = "voronoi")]
    {
        use crate::compute::{
            FiniteFieldPolarizability, LengthUnit, RadicalVoronoi, VoronoiDomainAnalysis,
            VoronoiIntegration, VoronoiVoidAnalysis,
        };
        cases.extend([
            case(RadicalVoronoi::new()),
            case(VoronoiIntegration::new(LengthUnit::Angstrom)),
            case(VoronoiDomainAnalysis::new()),
            case(VoronoiVoidAnalysis::new()),
            case(FiniteFieldPolarizability::new(0.01)?),
        ]);
    }
    Ok(cases)
}

/// The guard against a retyped default: every registered type, built from
/// its required arguments only, holds exactly the defaults its spec claims.
#[test]
fn every_default_is_what_the_constructor_produces() {
    let cases = cases().expect("every case builds");
    let covered: BTreeSet<&str> = cases.iter().map(|c| c.spec.name).collect();
    let registered: BTreeSet<&str> = ComputeSpec::all().iter().map(|s| s.name).collect();
    assert_eq!(covered, registered, "one case per registered type");
    for case in &cases {
        assert_eq!(
            Some(&case.spec),
            ComputeSpec::find(case.spec.name),
            "the registry holds the type's own spec"
        );
        case.check();
    }
}

/// A setter moves the readback off the default — `tunables` reads the
/// instance, not the spec.
#[test]
fn tunables_read_the_instance() {
    let msd = Msd::new(8).with_origins(crate::compute::Origins::First);
    assert_eq!(msd.tunables()[0], ("origins", ParamValue::Name("first")));
    let correlator = Correlator::new(LagKernel::Product, 8).with_dimension(3);
    assert_eq!(
        correlator.tunables().last(),
        Some(&("dimension", ParamValue::Int(3)))
    );
    let kmeans = Kmeans::new(2).unwrap().with_seed(7);
    assert_eq!(kmeans.tunables()[1], ("seed", ParamValue::Int(7)));
}

/// What a binding reads: the defaults the brief pins (Kmeans from link 07,
/// the `Window` default from link 03) and the input kinds of a surface.
#[test]
fn bindings_read_defaults_and_inputs() {
    let kmeans = ComputeSpec::find("Kmeans").unwrap();
    assert!(kmeans.param("k").unwrap().is_required());
    assert_eq!(
        kmeans.param("max_iter").unwrap().default,
        ParamDefault::Value(ParamValue::Int(Kmeans::DEFAULT_MAX_ITER as u64))
    );
    let gk = ComputeSpec::find("GreenKuboSpectrum").unwrap();
    assert_eq!(
        gk.param("window").unwrap().default,
        ParamDefault::Value(ParamValue::Name("cosine_sq"))
    );
    assert!(gk.param("units").unwrap().is_required());
    let msd = ComputeSpec::find("Msd").unwrap();
    assert_eq!(msd.input("positions").unwrap().kind, InputKind::Trajectory);
    assert!(msd.input("boxes").unwrap().optional);
    assert_eq!(
        ComputeSpec::find("Rdf").unwrap().form,
        crate::compute::InputForm::Frames
    );
    let direct = ComputeSpec::find("StaticStructureFactorDirect").unwrap();
    assert_eq!(
        direct.param("k_max").unwrap().default,
        ParamDefault::RequiredIn("isotropic")
    );
}
