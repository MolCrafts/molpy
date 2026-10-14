//! The FFI seam of the compute bindings: name parity with the core registry,
//! defaults parity of every class built from its required arguments, and one
//! construct-and-compute round trip per family (handles in, handles out).
//!
//! The science is the core's and is tested there.

use std::collections::BTreeSet;

use js_sys::{Array, Float64Array, Object, Reflect, Uint32Array};
use molrs::compute::{ParamDefault, ParamValue};
use molrs::core::SimBox;
use molrs::op::F;
use ndarray::{Array2, array};
use wasm_bindgen::JsCast;
use wasm_bindgen::convert::TryFromJsValue;
use wasm_bindgen::prelude::*;
use wasm_bindgen_test::*;

use super::handle::Bound;
use super::*;
use crate::core::frame::Frame;
use crate::core::{Box as JsBox, NDArray, NeighborList, Neighbors};

// --------------------------------------------------------------------- data

fn object(fields: Vec<(&str, JsValue)>) -> JsValue {
    let object = Object::new();
    for (key, value) in fields {
        Reflect::set(&object, &JsValue::from_str(key), &value).unwrap();
    }
    object.into()
}

fn reals(values: &[F]) -> JsValue {
    Float64Array::from(values).into()
}

fn naturals(values: &[u32]) -> JsValue {
    Uint32Array::from(values).into()
}

fn shaped(data: Vec<F>, dim: &[usize]) -> JsValue {
    NDArray::from_vec(data, dim.into()).into()
}

fn list(values: Vec<JsValue>) -> JsValue {
    values.into_iter().collect::<Array>().into()
}

fn cube_box(len: F, periodic: bool) -> SimBox {
    SimBox::cube(len, array![0.0 as F, 0.0, 0.0], [periodic; 3]).unwrap()
}

fn js_box(len: F) -> JsValue {
    JsBox {
        inner: cube_box(len, true),
    }
    .into()
}

fn frame(positions: &[[F; 3]], len: F, periodic: bool) -> Frame {
    let coords = Array2::from_shape_fn((positions.len(), 3), |(i, k)| positions[i][k]);
    let mut rs = molrs::core::Frame::new();
    rs.set_coords(coords.view()).unwrap();
    rs.simbox = Some(cube_box(len, periodic));
    Frame::from_rs(rs).unwrap()
}

/// The half-shell table of `positions` within `cutoff`, all columns kept.
fn neighbors(positions: &[[F; 3]], len: F, cutoff: F) -> Neighbors {
    let mut list = NeighborList::new(cutoff).unwrap();
    list.build(&frame(positions, len, true)).unwrap();
    list.neighbors(None).unwrap()
}

/// A simple cubic lattice of `n³` sites, spacing 1.
fn lattice(n: usize) -> Vec<[F; 3]> {
    let mut sites = Vec::new();
    for i in 0..n {
        for j in 0..n {
            for k in 0..n {
                sites.push([i as F + 0.5, j as F + 0.5, k as F + 0.5]);
            }
        }
    }
    sites
}

fn ok<T>(result: Result<T, JsError>) -> T {
    result.unwrap_or_else(|err| {
        let message = JsValue::from(err)
            .dyn_into::<js_sys::Error>()
            .map(|e| String::from(e.message()))
            .unwrap_or_default();
        panic!("binding call failed: {message}")
    })
}

fn message<T>(result: Result<T, JsError>) -> String {
    match result {
        Ok(_) => panic!("expected an error"),
        Err(err) => String::from(
            JsValue::from(err)
                .dyn_into::<js_sys::Error>()
                .unwrap()
                .message(),
        ),
    }
}

fn get(value: &JsValue, key: &str) -> JsValue {
    Reflect::get(value, &JsValue::from_str(key)).unwrap()
}

// ------------------------------------------------------------ name parity

/// `(class name, core spec)` of every bound compute class. The class name is
/// the struct's identifier, which is its JS name (no class here renames
/// itself).
macro_rules! bound {
    ($($class:ident),+ $(,)?) => {
        vec![$((stringify!($class), <$class as Bound>::spec())),+]
    };
}

fn bound_specs() -> Vec<(&'static str, molrs::compute::ComputeSpec)> {
    let specs = bound![
        Correlator,
        Msd,
        Vacf,
        OnsagerCorrelation,
        DipoleAcf,
        CurrentAcf,
        ReorientationAcf,
        RotationalAutocorrelation,
        Survival,
        Derivative,
        Unwrap,
        Spectrum,
        CumulativeTrapezoid,
        DebyeFit,
        LinearFit,
        Plateau,
        KineticEnergy,
        DipoleMoment,
        StaticDielectric,
        EinsteinHelfandSpectrum,
        GreenKuboSpectrum,
        DipoleRateCrossSpectrum,
        DipoleAutocorrelationSpectrum,
        KramersKronig,
        ConductivitySumRule,
        RouteAgreement,
        PolarizabilityParts,
        RamanSpectrum,
        Rdf,
        CorrelationFunction,
        LocalDensity,
        GaussianDensity,
        SphereVoxelization,
        SpatialDistribution,
        DistributionFunction,
        CombinedDistribution,
        StaticStructureFactorDebye,
        StaticStructureFactorDirect,
        DiffractionPattern,
        PairPresence,
        VanHove,
        HBonds,
        Steinhardt,
        Hexatic,
        SolidLiquid,
        ContinuousCoordination,
        Nematic,
        Cubatic,
        BondOrientationalOrder,
        LocalDescriptors,
        LocalBondProjection,
        EnvironmentMatch,
        AngularSeparationGlobal,
        AngularSeparationNeighbor,
        PmftR12,
        PmftXy,
        PmftXyt,
        PmftXyz,
        Cluster,
        CenterOfMass,
        GyrationTensor,
        InertiaTensor,
        Pca,
        Kmeans,
    ];
    #[cfg(feature = "voronoi")]
    let specs = [
        specs,
        bound![
            RadicalVoronoi,
            VoronoiIntegration,
            VoronoiDomainAnalysis,
            VoronoiVoidAnalysis,
            FiniteFieldPolarizability,
        ],
    ]
    .concat();
    specs
}

#[wasm_bindgen_test]
fn every_core_compute_is_bound_under_its_name() {
    let bound = bound_specs();
    for (class, spec) in &bound {
        assert_eq!(
            *class, spec.name,
            "class {class} binds the spec {}",
            spec.name
        );
    }
    let bound: BTreeSet<&str> = bound.iter().map(|(class, _)| *class).collect();
    let core: BTreeSet<&str> = molrs::compute::ComputeSpec::all()
        .iter()
        .map(|spec| spec.name)
        .collect();
    assert_eq!(bound, core);
}

#[wasm_bindgen_test]
fn compute_spec_all_serialises_the_registry() {
    let all = Array::from(&ok(ComputeSpec::all()));
    assert_eq!(
        all.length() as usize,
        molrs::compute::ComputeSpec::all().len()
    );
    let msd = ok(ComputeSpec::find("Msd"));
    assert_eq!(get(&msd, "name").as_string().unwrap(), "Msd");
    let inputs = Array::from(&get(&msd, "inputs"));
    let names: Vec<String> = inputs
        .iter()
        .map(|input| get(&input, "name").as_string().unwrap())
        .collect();
    assert_eq!(names, ["positions", "dt", "boxes"]);
    assert!(ok(ComputeSpec::find("NoSuchCompute")).is_undefined());
    assert_eq!(
        get(&ok(Msd::spec()), "name").as_string().unwrap(),
        "Msd",
        "a class's static spec() is its registry entry"
    );
}

// --------------------------------------------------------- defaults parity

/// `class`, built from its required arguments `given` alone, holds exactly
/// the defaults its spec declares, in spec order.
fn holds_spec_defaults<B: Bound>(class: B, given: &[(&str, ParamValue)]) {
    let spec = B::spec();
    let value_of = |name: &str| {
        given
            .iter()
            .find(|(param, _)| *param == name)
            .map(|(_, value)| value.clone())
            .unwrap_or_else(|| panic!("{}: test gives no `{name}`", spec.name))
    };
    let expected: Vec<(&str, ParamValue)> = spec
        .params
        .iter()
        .filter(|param| !param.is_required())
        .map(|param| {
            let value = match &param.default {
                ParamDefault::Value(value) => value.clone(),
                ParamDefault::SameAs(other) => value_of(other),
                ParamDefault::Scaled { of, factor } => match value_of(of) {
                    ParamValue::Float(v) => ParamValue::Float(v * factor),
                    other => panic!("{}: `{of}` is {other:?}", spec.name),
                },
                ParamDefault::Required | ParamDefault::RequiredIn(_) => unreachable!(),
            };
            (param.name, value)
        })
        .collect();
    assert_eq!(class.tunables(), expected, "{}", spec.name);
}

#[wasm_bindgen_test]
fn every_class_built_from_required_arguments_holds_the_core_defaults() {
    use ParamValue::Float;
    holds_spec_defaults(
        ok(Correlator::new(
            "product", 4, None, None, None, None, None, None,
        )),
        &[],
    );
    holds_spec_defaults(ok(Msd::new(4, None, None)), &[]);
    holds_spec_defaults(Vacf::new(4), &[]);
    holds_spec_defaults(OnsagerCorrelation::new(4), &[]);
    holds_spec_defaults(DipoleAcf::new(4), &[]);
    holds_spec_defaults(CurrentAcf::new(4), &[]);
    holds_spec_defaults(ReorientationAcf::new(2, 4), &[]);
    holds_spec_defaults(ok(RotationalAutocorrelation::new(2, 4, None)), &[]);
    holds_spec_defaults(ok(Survival::new("continuous", 4)), &[]);
    holds_spec_defaults(ok(Derivative::new("central")), &[]);
    holds_spec_defaults(Unwrap::new(), &[]);
    holds_spec_defaults(ok(Spectrum::new("hann", 2)), &[]);

    holds_spec_defaults(ok(CumulativeTrapezoid::new(None)), &[]);
    holds_spec_defaults(DebyeFit::new(), &[]);
    holds_spec_defaults(ok(LinearFit::new(0.1, 0.9)), &[]);
    holds_spec_defaults(ok(Plateau::new(0.1, 0.9)), &[]);

    holds_spec_defaults(ok(KineticEnergy::new("real")), &[]);
    holds_spec_defaults(ok(KineticEnergy::with_constants(1.0, 0.5)), &[]);
    holds_spec_defaults(DipoleMoment::new(), &[]);
    holds_spec_defaults(ok(StaticDielectric::new("real", None)), &[]);

    holds_spec_defaults(
        ok(EinsteinHelfandSpectrum::new("real", None, None, None)),
        &[],
    );
    holds_spec_defaults(ok(GreenKuboSpectrum::new("real", None, None, None)), &[]);
    holds_spec_defaults(
        ok(DipoleRateCrossSpectrum::new("real", None, None, None)),
        &[],
    );
    holds_spec_defaults(
        ok(DipoleAutocorrelationSpectrum::new(
            "real", None, None, None, None,
        )),
        &[],
    );
    holds_spec_defaults(ok(KramersKronig::new(None, None)), &[]);
    holds_spec_defaults(ok(ConductivitySumRule::new("real", None)), &[]);
    holds_spec_defaults(ok(RouteAgreement::new(None, None)), &[]);
    holds_spec_defaults(PolarizabilityParts::new(), &[]);
    holds_spec_defaults(ok(RamanSpectrum::new(None, None, None, None)), &[]);

    holds_spec_defaults(ok(Rdf::new(10, 3.0, 0.0, None)), &[]);
    holds_spec_defaults(ok(CorrelationFunction::new(10, 3.0, 0.0)), &[]);
    holds_spec_defaults(ok(LocalDensity::new(3.0, None)), &[]);
    holds_spec_defaults(
        ok(GaussianDensity::new(vec![4, 4, 4], 0.5, None)),
        &[("sigma", Float(0.5))],
    );
    holds_spec_defaults(ok(SphereVoxelization::new(vec![4, 4, 4], 1.0)), &[]);
    let grid = ok(RegularGrid::new(
        vec![0.0; 3],
        vec![1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0],
        vec![4, 4, 4],
        vec![1, 1, 1],
    ));
    holds_spec_defaults(ok(SpatialDistribution::new(&grid, None)), &[]);
    let axis = ok(Axis::new(0.0, 4.0, 8));
    holds_spec_defaults(ok(DistributionFunction::new("distance", &axis)), &[]);
    holds_spec_defaults(
        ok(DistributionFunction::over_natural_range("angle", 18)),
        &[],
    );
    let axes = list(vec![
        axis.duplicate().into(),
        ok(Axis::new(0.0, 3.0, 6)).into(),
    ]);
    holds_spec_defaults(
        ok(CombinedDistribution::new(
            vec!["distance".into(), "angle".into()],
            axes.unchecked_into(),
            None,
        )),
        &[],
    );
    holds_spec_defaults(
        ok(StaticStructureFactorDebye::new(vec![1.0, 2.0], None)),
        &[],
    );
    holds_spec_defaults(
        ok(StaticStructureFactorDirect::new(vec![1.0, 0.0, 0.0])),
        &[],
    );
    holds_spec_defaults(ok(StaticStructureFactorDirect::isotropic(4.0, 8)), &[]);
    holds_spec_defaults(
        ok(DiffractionPattern::new(16, 0.5, None, None)),
        &[("sigma", Float(0.5))],
    );

    holds_spec_defaults(
        ok(PairPresence::new(1.5, false, None)),
        &[("r0", Float(1.5))],
    );
    holds_spec_defaults(ok(VanHove::new(&axis, vec![1, 2], None, None)), &[]);
    holds_spec_defaults(ok(HBonds::new(None, None, None, None)), &[]);

    holds_spec_defaults(ok(Steinhardt::new(vec![6], None, None, None)), &[]);
    holds_spec_defaults(ok(Hexatic::new(6)), &[]);
    holds_spec_defaults(ok(SolidLiquid::new(None, None, None)), &[]);
    holds_spec_defaults(ok(ContinuousCoordination::new(2.0)), &[]);
    holds_spec_defaults(Nematic::new(), &[]);
    holds_spec_defaults(ok(Cubatic::new(None, None, None, None, None, None)), &[]);

    holds_spec_defaults(ok(BondOrientationalOrder::new(8, 16)), &[]);
    holds_spec_defaults(LocalDescriptors::new(4), &[]);
    holds_spec_defaults(LocalBondProjection::new(), &[]);
    holds_spec_defaults(ok(EnvironmentMatch::new(0.3, None, None)), &[]);
    holds_spec_defaults(ok(AngularSeparationGlobal::new(None)), &[]);
    holds_spec_defaults(ok(AngularSeparationNeighbor::new(None)), &[]);

    holds_spec_defaults(ok(PmftR12::new(3.0, 4, 4, 4)), &[]);
    holds_spec_defaults(ok(PmftXy::new(3.0, 3.0, 4, 4)), &[]);
    holds_spec_defaults(ok(PmftXyt::new(3.0, 3.0, 4, 4, 4)), &[]);
    holds_spec_defaults(ok(PmftXyz::new(3.0, 3.0, 3.0, 4, 4, 4)), &[]);

    holds_spec_defaults(ok(Cluster::new(None)), &[]);
    holds_spec_defaults(CenterOfMass::new(), &[]);
    holds_spec_defaults(GyrationTensor::new(), &[]);
    holds_spec_defaults(InertiaTensor::new(), &[]);
    holds_spec_defaults(ok(Pca::new(2)), &[]);
    holds_spec_defaults(ok(Kmeans::new(2, None, None)), &[]);

    #[cfg(feature = "voronoi")]
    {
        holds_spec_defaults(RadicalVoronoi::new(), &[]);
        holds_spec_defaults(ok(VoronoiIntegration::new("angstrom")), &[]);
        holds_spec_defaults(VoronoiDomainAnalysis::new(), &[]);
        holds_spec_defaults(VoronoiVoidAnalysis::new(), &[]);
        holds_spec_defaults(ok(FiniteFieldPolarizability::new(0.001)), &[]);
    }
}

/// A given optional reaches its setter; the rest keep the core's value.
#[wasm_bindgen_test]
fn a_given_option_is_applied_and_only_it() {
    let msd = ok(Msd::new(4, None, Some("per_item".into())));
    let tunables = Bound::tunables(&msd);
    assert_eq!(tunables[1], ("reduction", ParamValue::Name("per_item")));
    assert_eq!(
        tunables[0],
        (
            "origins",
            ParamValue::from(molrs::compute::Origins::default())
        )
    );
    let kk = ok(KramersKronig::new(Some(0.5), None));
    let fresh = Bound::tunables(&ok(KramersKronig::new(None, None)));
    assert_eq!(
        Bound::tunables(&kk)[0],
        ("relative_tolerance", ParamValue::Float(0.5))
    );
    assert_eq!(Bound::tunables(&kk)[1], fresh[1]);
}

// ------------------------------------------------------- per-family seams

#[wasm_bindgen_test]
fn time_series_curve_feeds_a_fit_through_its_handle() {
    // Item 0 steps +1 along x every frame, item 1 stays put.
    let positions = shaped(
        vec![
            0.0, 0.0, 0.0, 5.0, 5.0, 5.0, //
            1.0, 0.0, 0.0, 5.0, 5.0, 5.0, //
            2.0, 0.0, 0.0, 5.0, 5.0, 5.0,
        ],
        &[3, 2, 3],
    );
    let msd = ok(Msd::new(2, None, None));
    let curve = ok(msd.compute(&object(vec![
        ("positions", positions),
        ("dt", JsValue::from(0.5)),
    ])));
    assert_eq!(curve.inner.lag_times.to_vec(), [0.0, 0.5, 1.0]);
    assert_eq!(curve.inner.values.column(0).to_vec(), [0.0, 0.5, 2.0]);

    let curve_js = JsValue::from(curve);
    let fit = ok(ok(LinearFit::new(0.0, 1.0)).compute(&object(vec![
        ("correlation", curve_js.clone()),
        ("channel", JsValue::from(0)),
    ])));
    assert!(fit.inner.slope > 0.0);
    // The handle was borrowed, not consumed.
    assert!(TimeCorrelation::try_from_js_value(curve_js).is_ok());

    let json = ok(fit.to_json());
    assert!(get(&json, "slope").as_f64().unwrap() > 0.0);

    let derivative = ok(ok(Derivative::new("central")).compute(&object(vec![
        ("series", shaped(vec![0.0, 1.0, 2.0, 3.0], &[4, 1])),
        ("dt", JsValue::from(1.0)),
    ])));
    // One channel in, one channel out (the stencil sets the row count).
    assert_eq!(derivative.shape()[1], 1);
}

#[wasm_bindgen_test]
fn dielectric_and_spectroscopy_take_partitions_and_flat_rows() {
    let moments = ok(DipoleMoment::new().compute(&object(vec![
        (
            "positions",
            shaped(vec![0.0, 0.0, 0.0, 1.0, 0.0, 0.0], &[1, 2, 3]),
        ),
        ("charges", reals(&[1.0, -1.0])),
        ("partition", naturals(&[0, 0])),
    ])));
    assert_eq!(moments.inner.moments.dim(), (1, 1, 3));
    assert_eq!(moments.inner.moments[[0, 0, 0]], -1.0);

    let energy = ok(ok(KineticEnergy::new("real")).compute(&object(vec![
        ("masses", reals(&[1.0, 1.0])),
        ("velocities", reals(&[1.0, 0.0, 0.0, 0.0, 1.0, 0.0])),
    ])));
    assert!(energy.inner.energy > 0.0);
    assert!(ok(energy.temperature(6)) > 0.0);

    let alpha: Vec<F> = (0..24).map(|i| i as F).collect();
    let parts = ok(PolarizabilityParts::new().compute(&object(vec![("alpha", reals(&alpha))])));
    assert_eq!(parts.inner.isotropic.nrows(), 4);
}

#[wasm_bindgen_test]
fn structure_and_distributions_fold_their_frames() {
    let sites = lattice(3);
    let tables: Vec<JsValue> = (0..2)
        .map(|_| object(vec![("pairs", neighbors(&sites, 3.0, 1.4).into())]))
        .collect();
    let rdf = ok(ok(Rdf::new(10, 1.4, 0.0, None)).compute(&list(tables)));
    assert_eq!(rdf.inner.n_frames, 2);
    assert_eq!(rdf.inner.rdf.len(), 10);

    let df = ok(DistributionFunction::new(
        "distance",
        &ok(Axis::new(0.0, 2.0, 4)),
    ));
    let result = ok(df.compute(&list(vec![object(vec![
        ("positions", reals(&[0.0, 0.0, 0.0, 1.0, 0.0, 0.0])),
        ("tuples", naturals(&[0, 1])),
    ])])));
    assert_eq!(result.inner.n_raw_samples, 1);

    let a_frame = |data: Vec<(&str, JsValue)>| list(vec![object(data)]);
    let err = message(ok(Rdf::new(10, 1.4, 0.0, None)).compute(&a_frame(vec![])));
    assert!(err.contains("pairs"), "{err}");
}

#[wasm_bindgen_test]
fn presence_round_trips_into_survival_through_to_json() {
    let near = [[1.0, 1.0, 1.0], [1.5, 1.0, 1.0]];
    let frames: Vec<JsValue> = (0..3).map(|_| neighbors(&near, 10.0, 2.0).into()).collect();
    let presence =
        ok(ok(PairPresence::new(1.0, false, None))
            .compute(&object(vec![("frames", list(frames))])))
        .expect("the pair is present");
    let values = get(&ok(presence.to_json()), "values");
    let tcf = ok(ok(Survival::new("continuous", 2)).compute(&object(vec![
        ("presence", values),
        ("dt", JsValue::from(1.0)),
    ])));
    assert_eq!(tcf.inner.values.column(0).to_vec(), [1.0, 1.0, 1.0]);
}

#[wasm_bindgen_test]
fn order_tables_flow_from_steinhardt_to_solid_liquid() {
    let sites = lattice(3);
    let table = neighbors(&sites, 3.0, 1.1);
    let table_js = JsValue::from(table);
    let q = ok(ok(Steinhardt::new(vec![6], None, None, None))
        .compute(&object(vec![("neighbors", table_js.clone())])));
    let qlm = q.qlm(6).expect("order 6 was computed");
    assert!(q.qlm(4).is_none());
    let solid = ok(ok(SolidLiquid::new(None, None, None)).compute(&object(vec![
        ("qlm", qlm.into()),
        ("neighbors", table_js.clone()),
    ])));
    assert_eq!(solid.inner.is_solid.len(), sites.len());
    // The neighbour table was borrowed twice and is still the caller's.
    assert_eq!(
        Neighbors::try_from_js_value(table_js).unwrap().n_pairs(),
        81
    );
}

#[wasm_bindgen_test]
fn groups_take_any_partition_and_feature_matrix() {
    let sites = [
        [1.0, 1.0, 1.0],
        [1.5, 1.0, 1.0],
        [8.0, 8.0, 8.0],
        [8.5, 8.0, 8.0],
    ];
    let clusters = ok(ok(Cluster::new(None)).compute(&object(vec![(
        "pairs",
        neighbors(&sites, 20.0, 1.0).into(),
    )])));
    assert_eq!(clusters.inner.n_clusters, 2);

    let flat: Vec<F> = sites.iter().flatten().copied().collect();
    let com = ok(CenterOfMass::new().compute(&object(vec![
        ("positions", reals(&flat)),
        ("partition", clusters.into()),
        ("masses", reals(&[1.0; 4])),
    ])));
    assert_eq!(com.inner.centers.nrows(), 2);

    let features = shaped(vec![0.0, 0.0, 0.1, 0.0, 5.0, 5.0, 5.1, 5.0], &[4, 2]);
    let pca = ok(ok(Pca::new(1)).compute(&object(vec![("features", features)])));
    let kmeans =
        ok(ok(Kmeans::new(2, None, None)).compute(&object(vec![("features", pca.into())])));
    let labels = &kmeans.inner.labels;
    assert_eq!(labels[0], labels[1]);
    assert_ne!(labels[0], labels[2]);
    let gyration = ok(GyrationTensor::new().compute(&object(vec![
        ("positions", reals(&flat)),
        ("simbox", js_box(20.0)),
        ("partition", kmeans.into()),
        ("weights", reals(&[1.0; 4])),
    ])));
    assert_eq!(gyration.radius_of_gyration().len(), 2);
}

#[wasm_bindgen_test]
fn environment_and_pmft_read_neighbour_handles() {
    let sites = lattice(3);
    let descriptors = ok(LocalDescriptors::new(2).compute(&object(vec![(
        "neighbors",
        neighbors(&sites, 3.0, 1.1).into(),
    )])));
    assert_eq!(descriptors.inner.l_max, 2);

    let plane = [[1.0, 1.0, 0.0], [1.5, 1.0, 0.0], [1.0, 1.5, 0.0]];
    let pmft = ok(ok(PmftXy::new(1.0, 1.0, 4, 4)).compute(&object(vec![
        ("pairs", neighbors(&plane, 10.0, 1.0).into()),
        ("orientations", reals(&[0.0, 0.0, 0.0])),
    ])));
    assert_eq!(pmft.inner.density.dim(), (4, 4));
}

#[cfg(feature = "voronoi")]
#[wasm_bindgen_test]
fn voronoi_cells_feed_the_void_analysis() {
    let sites: Vec<F> = lattice(2).into_iter().flatten().collect();
    let cells = ok(RadicalVoronoi::new().compute(&object(vec![
        ("positions", reals(&sites)),
        ("simbox", js_box(2.0)),
        ("radii", reals(&[0.5; 8])),
    ])));
    let total: F = cells.inner.volumes.iter().sum();
    assert!((total - 8.0).abs() < 1e-9, "cells tile the box: {total}");
    let voids = ok(VoronoiVoidAnalysis::new().compute(&object(vec![
        ("cells", cells.into()),
        (
            "is_void",
            list((0..8).map(|i| JsValue::from(i == 0)).collect()),
        ),
        ("box_volume", JsValue::from(8.0)),
    ])));
    assert!((voids.inner.void_fraction - 0.125).abs() < 1e-9);
}

// ------------------------------------------------------------------ errors

#[wasm_bindgen_test]
fn errors_are_the_core_compute_error() {
    // A missing field names the field.
    let err = message(ok(Msd::new(2, None, None)).compute(&object(vec![(
        "positions",
        shaped(vec![0.0; 6], &[1, 2, 3]),
    )])));
    assert!(err.contains("Msd input `dt`"), "{err}");
    // An unknown choice lists the canonical names, from the core parser.
    let err = message(Survival::new("forever", 2));
    assert!(err.contains("continuous, intermittent"), "{err}");
    // A kernel's order is required where the core needs one and refused elsewhere.
    let err = message(Correlator::new(
        "legendre", 2, None, None, None, None, None, None,
    ));
    assert!(err.contains("required"), "{err}");
    let err = message(Correlator::new(
        "product",
        2,
        Some(2),
        None,
        None,
        None,
        None,
        None,
    ));
    assert!(err.contains("only legendre"), "{err}");
    let err = message(Correlator::new(
        "cosine", 2, None, None, None, None, None, None,
    ));
    assert!(err.contains("unknown LagKernelKind"), "{err}");
    // The alternative constructor validates its constants.
    let err = message(KineticEnergy::with_constants(1.0, 0.0));
    assert!(err.contains("boltzmann"), "{err}");
    // A core refusal is passed through unchanged.
    let err = message(LinearFit::new(0.9, 0.1));
    assert!(err.contains("out of range"), "{err}");
    // A handle of the wrong class is refused, not reinterpreted.
    let err = message(ok(LinearFit::new(0.0, 1.0)).compute(&object(vec![
        ("correlation", js_box(1.0)),
        ("channel", JsValue::from(0)),
    ])));
    assert!(err.contains("TimeCorrelation"), "{err}");
}
