//! The registry's protocol: what it accepts, what it refuses and why, and
//! that the built-ins are registered through it.

use crate::ff::forcefield::Category;
use crate::ff::potential::fixtures::OneShot;
use std::collections::BTreeSet;
use std::sync::Arc;

use crate::ff::compile::PotentialCompiler;
use crate::ff::forcefield::{DefError, ForceField, StyleDefs};
use crate::ff::ir::Params;
use crate::ff::ir::{
    Arity, CategorySpec, ConformanceSample, Coordinate, EndpointOrder, IrError, ParamCombination,
    ParamDimension, ParamKind, ParamSpec, ParamValue, SpecialClass, StyleSpec, builtin_categories,
    builtin_styles,
};
use crate::ff::potential::CompileError;
use crate::ff::potential::bond::bond_harmonic_constructor;
use crate::ff::potential::form_kernel::{CompoundForm, ParamColumns, ScalarForm};
use crate::ff::style_registry::BuiltinKernels;
use crate::ff::style_registry::{
    ExpressionForm, ExpressionKernel, Kernel, Registry, RowSource, register_style,
};
#[cfg(feature = "zarr")]
use crate::io::mrec::ForceFieldSection;
use molrs::core::Block;
use molrs::core::Frame;
use molrs::op::{F, Idx};
use ndarray::Array1;

/// `scale · k (q − q0)²`, its derivative off by `wrong` (1 is right).
struct Harmonic {
    scale: F,
    wrong: F,
}

const RIGHT: Harmonic = Harmonic {
    scale: 1.0,
    wrong: 1.0,
};

impl ScalarForm for Harmonic {
    fn eval(&self, q: &[F], p: &ParamColumns<'_>, e: &mut [F], de: &mut [F]) {
        let (k, q0) = (p.get("k").unwrap(), p.get("r0").unwrap());
        for t in 0..q.len() {
            let d = q[t] - q0[t];
            e[t] = self.scale * k[t] * d * d;
            de[t] = self.wrong * 2.0 * self.scale * k[t] * d;
        }
    }
}

/// `k q1 q2 / r` — or, `asymmetric`, `k q1 / r`.
struct Charges {
    asymmetric: bool,
}

impl ScalarForm for Charges {
    fn eval(&self, r: &[F], p: &ParamColumns<'_>, e: &mut [F], de: &mut [F]) {
        let (k, q1, q2) = (
            p.get("k").unwrap(),
            p.get("q1").unwrap(),
            p.get("q2").unwrap(),
        );
        for t in 0..r.len() {
            let qq = if self.asymmetric {
                q1[t]
            } else {
                q1[t] * q2[t]
            };
            e[t] = k[t] * qq / r[t];
            de[t] = -e[t] / r[t];
        }
    }
}

fn harmonic_spec(category: &'static str, name: &'static str) -> StyleSpec {
    StyleSpec::new(category, name).params(vec![
        ParamSpec::new("k", "E/L^2".parse().unwrap()),
        ParamSpec::new("r0", ParamDimension::LENGTH),
    ])
}

fn sample() -> ConformanceSample {
    ConformanceSample {
        params: vec![
            ("k".into(), ParamValue::Num(300.0)),
            ("r0".into(), ParamValue::Num(1.5)),
        ],
        q: (0.9, 2.5),
    }
}

fn scalar(form: Harmonic) -> Option<Kernel> {
    Some(Kernel::Scalar(Arc::new(form)))
}

/// An expression-engine stand-in: what `ff::ir::expression` hands the registry,
/// with the variables and form chosen by the test.
struct FakeExpression {
    source: String,
    variables: Vec<String>,
    form: Arc<dyn ScalarForm>,
}

impl ExpressionKernel for FakeExpression {
    fn source(&self) -> &str {
        &self.source
    }
    fn variables(&self) -> Vec<String> {
        self.variables.clone()
    }
    fn form(&self) -> ExpressionForm {
        ExpressionForm::Scalar(self.form.clone())
    }
}

fn expression(variables: &[&str], form: Harmonic) -> Option<Kernel> {
    Some(Kernel::Expression(Arc::new(FakeExpression {
        source: "k*(r-r0)^2".into(),
        variables: variables.iter().map(|v| v.to_string()).collect(),
        form: Arc::new(form),
    })))
}

#[test]
fn every_builtin_kernel_has_its_spec() {
    let kernels: BTreeSet<(String, String)> = BuiltinKernels::builtin()
        .styles()
        .into_iter()
        .map(|(c, n)| (c.to_owned(), n.to_owned()))
        .collect();
    let specs: BTreeSet<(String, String)> = builtin_styles()
        .into_iter()
        .map(|s| (s.category.into_owned(), s.name.into_owned()))
        .collect();
    let without_kernel: BTreeSet<(String, String)> = specs.difference(&kernels).cloned().collect();
    assert!(
        kernels.is_subset(&specs),
        "{:?}",
        kernels.difference(&specs)
    );
    let expected: BTreeSet<(String, String)> = [
        ("atom", "full"),
        ("constraint", "fixed"),
        ("dihedral", "rb"),
        ("drude", "harmonic"),
        ("virtual_site", "average2"),
        ("virtual_site", "average3"),
        ("virtual_site", "outofplane3"),
    ]
    .iter()
    .map(|(c, n)| (c.to_string(), n.to_string()))
    .collect();
    assert_eq!(without_kernel, expected);
    let r = Registry::builtin();
    assert_eq!(r.styles(None).count(), specs.len());
    assert_eq!(r.categories().count(), builtin_categories().len());
    assert!(
        r.styles(None)
            .all(|(s, _)| r.is_sealed(&s.category, &s.name))
    );
    // `dihedral rb` is pure protocol: a built-in priced by its expression.
    let (rb, kernel) = r.style("dihedral", "rb").unwrap();
    assert!(kernel.is_none() && rb.expression.is_some());
}

/// The built-in table states each style's per-type parameters in the order
/// a LAMMPS `*_coeff` line lists them, with the protocol's dimensions.
#[test]
fn builtin_params_are_lammps_coeff_order_with_dims() {
    let r = Registry::builtin();
    let spec = |c: &str, s: &str| r.style(c, s).unwrap().0.clone();
    let order = |c: &str, s: &str| -> Vec<String> {
        spec(c, s)
            .params
            .iter()
            .map(|p| p.name.to_string())
            .collect()
    };
    assert_eq!(order("bond", "harmonic"), ["k", "r0"]);
    assert_eq!(order("bond", "class2"), ["r0", "k2", "k3", "k4"]);
    assert_eq!(order("angle", "charmm"), ["k", "theta0", "k_ub", "r_ub"]);
    assert_eq!(
        order("dihedral", "charmm"),
        ["k", "periodicity", "phase", "w"]
    );
    assert_eq!(order("pair", "buck"), ["a", "rho", "c"]);
    assert_eq!(
        order("pair", "lj/charmm"),
        ["epsilon", "sigma", "epsilon14", "sigma14"]
    );
    let angle = spec("angle", "harmonic");
    assert_eq!(
        angle.param("theta0").unwrap().dim,
        ParamDimension::ANGLE,
        "an angle value"
    );
    assert_eq!(
        angle.param("k").unwrap().dim.to_string(),
        "E/A^2",
        "per radian"
    );
    assert_eq!(
        spec("atom", "full").param("mass").unwrap().dim,
        ParamDimension::MASS
    );
    let periodic = spec("dihedral", "periodic");
    assert!(periodic.unindexed_one_term && periodic.params.iter().all(|p| p.indexed));
    assert!(!spec("pair", "coul/charmm").force_is_gradient);
    assert_eq!(
        spec("pair", "lj/cut").param("epsilon").unwrap().mix,
        ParamCombination::LjEpsilon {
            sigma: "sigma".into()
        }
    );
    assert_eq!(
        spec("angle", "harmonic").expression.as_deref(),
        Some("k*(theta-theta0*(pi/180))^2; pi=3.141592653589793")
    );
}

#[test]
fn a_builtin_is_sealed_and_an_identical_restatement_is_a_no_op() {
    let mut r = Registry::builtin();
    let (spec, kernel) = r.style("bond", "harmonic").unwrap();
    let (spec, kernel) = (spec.clone(), kernel.cloned());
    assert_eq!(r.register_style(spec, kernel), Ok(()));
    let err = r
        .register_style(
            StyleSpec::new("bond", "harmonic"),
            Some(Kernel::constructor(bond_harmonic_constructor)),
        )
        .unwrap_err();
    assert!(matches!(err, IrError::Sealed { .. }), "{err}");
    assert!(matches!(
        r.unregister_style("bond", "harmonic"),
        Err(IrError::Sealed { .. })
    ));
    // The global registry seals them too.
    assert!(matches!(
        register_style(
            StyleSpec::new("bond", "harmonic"),
            Some(Kernel::constructor(bond_harmonic_constructor))
        ),
        Err(IrError::Sealed { .. })
    ));
    // A built-in category likewise.
    let mut bond = r.category("bond").unwrap().clone();
    assert_eq!(r.register_category(bond.clone()), Ok(()));
    bond.order = EndpointOrder::Ordered;
    assert!(matches!(
        r.register_category(bond),
        Err(IrError::Sealed { .. })
    ));
}

#[test]
fn a_custom_style_conflicts_unless_restated_identically() {
    let mut r = Registry::builtin();
    let kernel = scalar(RIGHT);
    r.register_style(harmonic_spec("bond", "mine"), kernel.clone())
        .unwrap();
    assert_eq!(
        r.register_style(harmonic_spec("bond", "mine"), kernel),
        Ok(())
    );
    assert!(matches!(
        r.register_style(harmonic_spec("bond", "mine"), scalar(RIGHT)),
        Err(IrError::Conflict { .. })
    ));
    assert_eq!(r.unregister_style("bond", "mine"), Ok(()));
    assert!(matches!(
        r.unregister_style("bond", "mine"),
        Err(IrError::NoKernel { .. })
    ));
}

#[test]
fn custom_categories_follow_the_custom_rules() {
    let mut r = Registry::builtin();
    let ub = CategorySpec::custom(
        "urey_bradley",
        3,
        Coordinate::Compound,
        EndpointOrder::Reversible,
    );
    assert_eq!(ub.block, "urey_bradleys");
    assert_eq!(r.register_category(ub.clone()), Ok(()));
    assert_eq!(r.register_category(ub.clone()), Ok(()));
    let mut other = ub.clone();
    other.order = EndpointOrder::Ordered;
    assert!(matches!(
        r.register_category(other),
        Err(IrError::Conflict { .. })
    ));

    let bad = |c: CategorySpec| Registry::builtin().register_category(c).unwrap_err();
    assert!(matches!(
        bad(CategorySpec::custom(
            "Bad",
            2,
            Coordinate::Compound,
            EndpointOrder::Ordered
        )),
        IrError::BadName { .. }
    ));
    for arity in [1, 6] {
        assert!(matches!(
            bad(CategorySpec::custom(
                "x",
                arity,
                Coordinate::Compound,
                EndpointOrder::Ordered
            )),
            IrError::Arity { .. }
        ));
    }
    let mut elsewhere = CategorySpec::custom("x", 2, Coordinate::Distance, EndpointOrder::Ordered);
    elsewhere.block = "things".into();
    assert!(matches!(bad(elsewhere), IrError::BlockName { .. }));
    assert!(matches!(
        bad(CategorySpec::custom(
            "x",
            2,
            Coordinate::Angle,
            EndpointOrder::Ordered
        )),
        IrError::CoordinateMismatch { .. }
    ));
    assert!(matches!(
        bad(CategorySpec::new(
            "x",
            Arity::SelfOrPair,
            "xs",
            Coordinate::Distance,
            EndpointOrder::Unordered
        )),
        IrError::Arity { .. }
    ));
}

#[test]
fn nonconforming_styles_are_refused_by_name() {
    let mut r = Registry::builtin();

    let err = r
        .register_style(harmonic_spec("bend", "x"), scalar(RIGHT))
        .unwrap_err();
    assert!(matches!(err, IrError::UnknownCategory { .. }), "{err}");

    for reserved in [
        "name", "type", "style", "itom", "atomj", "desc", "theta", "chi", "p3",
    ] {
        let spec = StyleSpec::new("bond", "x")
            .params(vec![ParamSpec::new(reserved, ParamDimension::NONE)]);
        assert_eq!(
            r.register_style(spec, Some(Kernel::constructor(bond_harmonic_constructor))),
            Err(IrError::ReservedParam {
                style: "x".into(),
                param: reserved.into()
            }),
            "{reserved}"
        );
    }
    // A pair binds `q1`, `q2` and `<param>1`, `<param>2` itself.
    for reserved in ["q", "q2", "k1"] {
        let spec = StyleSpec::new("pair", "x").params(vec![
            ParamSpec::new("k", ParamDimension::ENERGY),
            ParamSpec::new(reserved, ParamDimension::NONE),
        ]);
        assert!(
            matches!(
                r.register_style(spec, Some(Kernel::constructor(bond_harmonic_constructor))),
                Err(IrError::ReservedParam { .. })
            ),
            "{reserved}"
        );
    }
    // A function name is no variable: a call always has its parenthesis.
    for function in ["exp", "delta", "select"] {
        let spec = StyleSpec::new("bond", function)
            .params(vec![ParamSpec::new(function, ParamDimension::NONE)]);
        r.register_style(spec, Some(Kernel::constructor(bond_harmonic_constructor)))
            .unwrap();
    }
    assert!(matches!(
        r.register_style(
            StyleSpec::new("bond", "x").params(vec![ParamSpec::new("2k", ParamDimension::NONE)]),
            Some(Kernel::constructor(bond_harmonic_constructor))
        ),
        Err(IrError::BadName { .. })
    ));

    let twice = StyleSpec::new("bond", "x")
        .params(vec![ParamSpec::new("k", ParamDimension::ENERGY)])
        .style_params(vec![ParamSpec::new("k", ParamDimension::ENERGY)]);
    assert!(matches!(
        r.register_style(twice, Some(Kernel::constructor(bond_harmonic_constructor))),
        Err(IrError::DuplicateParam { .. })
    ));

    let per_degree =
        StyleSpec::new("bond", "x").params(vec![ParamSpec::new("k", "E*A".parse().unwrap())]);
    assert!(matches!(
        r.register_style(
            per_degree,
            Some(Kernel::constructor(bond_harmonic_constructor))
        ),
        Err(IrError::Dimension { .. })
    ));

    // A scalar form on a compound category; a compound form on a pair is
    // the mirror case.
    let err = r
        .register_style(harmonic_spec("cmap", "x"), scalar(RIGHT))
        .unwrap_err();
    assert!(matches!(err, IrError::CoordinateMismatch { .. }), "{err}");

    // Any kernel on a category that prices nothing.
    let err = r
        .register_style(harmonic_spec("constraint", "x"), scalar(RIGHT))
        .unwrap_err();
    assert!(matches!(err, IrError::CoordinateMismatch { .. }), "{err}");

    // Neither a kernel nor an expression.
    assert!(matches!(
        r.register_style(harmonic_spec("bond", "x"), None),
        Err(IrError::NoKernel { .. })
    ));

    // `cutoff` keeps its reserved meaning.
    let cutoff = harmonic_spec("pair", "x")
        .style_params(vec![ParamSpec::new("cutoff", ParamDimension::ENERGY)]);
    assert!(matches!(
        r.register_style(cutoff, scalar(RIGHT)),
        Err(IrError::ReservedParam { .. })
    ));

    // A neighbour-driven form on a bonded category.
    let typed = Kernel::Constructor {
        compiled: bond_harmonic_constructor,
        typed: Some((bond_harmonic_constructor, SpecialClass::Vdw)),
        rows: RowSource::CategoryBlock,
    };
    assert!(matches!(
        r.register_style(StyleSpec::new("bond", "x"), Some(typed)),
        Err(IrError::CoordinateMismatch { .. })
    ));

    // A mixing rule on a bond.
    let mixing = StyleSpec::new("bond", "x").params(vec![
        ParamSpec::new("k", ParamDimension::ENERGY).mix(ParamCombination::Geometric),
    ]);
    assert!(matches!(
        r.register_style(mixing, Some(Kernel::constructor(bond_harmonic_constructor))),
        Err(IrError::Malformed { .. })
    ));

    // Nothing is left behind by a refusal.
    assert!(r.style("bond", "x").is_none());
}

#[test]
fn a_form_is_checked_on_its_samples_at_registration() {
    let mut r = Registry::builtin();
    let half = Harmonic {
        scale: 1.0,
        wrong: 0.5,
    };
    let err = r
        .register_style(harmonic_spec("bond", "x").sample(sample()), scalar(half))
        .unwrap_err();
    assert!(matches!(err, IrError::Derivative { .. }), "{err}");
    r.register_style(harmonic_spec("bond", "x").sample(sample()), scalar(RIGHT))
        .unwrap();

    // A pair energy must not change when its two atoms are exchanged.
    let coulomb = |name: &'static str| {
        StyleSpec::new("pair", name)
            .params(vec![
                ParamSpec::new("k", "E*L/Q^2".parse().unwrap()).mix(ParamCombination::Geometric),
            ])
            .sample(ConformanceSample {
                params: vec![("k".into(), ParamValue::Num(332.0))],
                q: (1.0, 5.0),
            })
    };
    let err = r
        .register_style(
            coulomb("lopsided"),
            Some(Kernel::Scalar(Arc::new(Charges { asymmetric: true }))),
        )
        .unwrap_err();
    assert!(matches!(err, IrError::Asymmetric { .. }), "{err}");
    r.register_style(
        coulomb("coulomb"),
        Some(Kernel::Scalar(Arc::new(Charges { asymmetric: false }))),
    )
    .unwrap();
}

/// A bond of two atoms 1.6 apart, typed `A-A`, and a force field defining
/// `bond <name>` over it.
fn one_bond(name: &str) -> (ForceField, Frame) {
    let mut ff = ForceField::new("t");
    ff.def_style("bond", name, Params::new())
        .unwrap()
        .def_type(
            "A-A",
            &["A", "A"],
            Params::from_iter([("k", 2.0), ("r0", 1.0)]),
        )
        .unwrap();
    let mut atoms = Block::new();
    for (key, v) in [("x", [0.0, 1.6]), ("y", [0.0, 0.0]), ("z", [0.0, 0.0])] {
        atoms
            .insert(key, Array1::from_vec(v.to_vec()).into_dyn(), None)
            .unwrap();
    }
    let mut bonds = Block::new();
    bonds
        .insert("atomi", Array1::from_vec(vec![0 as Idx]).into_dyn(), None)
        .unwrap();
    bonds
        .insert("atomj", Array1::from_vec(vec![1 as Idx]).into_dyn(), None)
        .unwrap();
    bonds
        .insert(
            "type",
            Array1::from_vec(vec!["A-A".to_string()]).into_dyn(),
            None,
        )
        .unwrap();
    let mut frame = Frame::new();
    frame.insert("atoms", atoms);
    frame.insert("bonds", bonds);
    (ff, frame)
}

/// Without samples the check runs at the first compile, on real terms, and
/// its verdict stands for every later compile.
#[test]
fn a_form_without_samples_is_checked_at_its_first_compile() {
    let mut r = Registry::builtin();
    let half = Harmonic {
        scale: 1.0,
        wrong: 0.5,
    };
    r.register_style(harmonic_spec("bond", "unchecked"), scalar(half))
        .unwrap();
    let (ff, frame) = one_bond("unchecked");
    for _ in 0..2 {
        let err = PotentialCompiler::new(&ff.clone().with_registry(Arc::new(r.clone())))
            .compile(&frame)
            .unwrap_err();
        assert!(err.to_string().contains("central difference"), "{err}");
    }
}

#[test]
fn an_expression_reads_only_what_its_style_declares() {
    let mut r = Registry::builtin();
    r.register_style(
        harmonic_spec("bond", "expr"),
        expression(&["r", "k", "r0"], RIGHT),
    )
    .unwrap();
    assert_eq!(
        r.register_style(
            harmonic_spec("bond", "x"),
            expression(&["theta", "k"], RIGHT)
        ),
        Err(IrError::UnboundVariable {
            style: "x".into(),
            name: "theta".into()
        })
    );
    // A pair binds the per-atom inputs as well; an improper `phi` and `chi`.
    r.register_style(
        harmonic_spec("pair", "expr"),
        expression(&["r", "k1", "r02", "q1"], RIGHT),
    )
    .unwrap();
    r.register_style(
        harmonic_spec("improper", "expr"),
        expression(&["phi", "chi", "k"], RIGHT),
    )
    .unwrap();
    // An indexed family by any member.
    let indexed = StyleSpec::new("dihedral", "series").params(vec![
        ParamSpec::new("k", ParamDimension::ENERGY).indexed(),
        ParamSpec::new("r0", ParamDimension::NONE),
    ]);
    r.register_style(indexed, expression(&["phi", "k1", "k7"], RIGHT))
        .unwrap();

    // The compiled expression is the spec's, byte for byte.
    let spec = harmonic_spec("bond", "y").expression("k*(r - r0)^2");
    assert!(matches!(
        r.register_style(spec, expression(&["r"], RIGHT)),
        Err(IrError::Malformed { .. })
    ));
}

/// The expression engine is reached through [`ExpressionCompiler`]: an
/// expression declared beside a native form must price what it prices, and
/// an expression-only style is priced by it.
///
/// [`ExpressionCompiler`]: crate::ff::style_registry::ExpressionCompiler
#[test]
fn an_expression_beside_a_native_form_must_agree_with_it() {
    fn agreeing(_: &CategorySpec, s: &StyleSpec) -> Result<Arc<dyn ExpressionKernel>, IrError> {
        Ok(Arc::new(FakeExpression {
            source: s.expression.clone().unwrap(),
            variables: vec!["r".into(), "k".into(), "r0".into()],
            form: Arc::new(RIGHT),
        }))
    }
    fn off_by_1e9(_: &CategorySpec, s: &StyleSpec) -> Result<Arc<dyn ExpressionKernel>, IrError> {
        Ok(Arc::new(FakeExpression {
            source: s.expression.clone().unwrap(),
            variables: vec!["r".into()],
            form: Arc::new(Harmonic {
                scale: 1.0 + 1e-9,
                wrong: 1.0,
            }),
        }))
    }
    let spec = || {
        harmonic_spec("bond", "native")
            .expression("k*(r-r0)^2")
            .sample(sample())
    };
    let mut r = Registry::builtin();
    r.set_expression_compiler(Some(agreeing));
    r.register_style(spec(), scalar(RIGHT)).unwrap();

    // Expression only: priced by the engine.
    r.register_style(
        harmonic_spec("bond", "by/expression").expression("k*(r-r0)^2"),
        None,
    )
    .unwrap();
    let (ff, frame) = one_bond("by/expression");
    let pots = PotentialCompiler::new(&ff.clone().with_registry(Arc::new(r.clone())))
        .compile(&frame)
        .unwrap();
    assert!((pots.energy_at(&[0.0, 0.0, 0.0, 1.6, 0.0, 0.0]) - 2.0 * 0.36).abs() < 1e-12);

    let mut r = Registry::builtin();
    r.set_expression_compiler(Some(off_by_1e9));
    let err = r.register_style(spec(), scalar(RIGHT)).unwrap_err();
    assert!(matches!(err, IrError::Disagree { .. }), "{err}");

    // Without an engine nothing about the expression can be checked, and an
    // expression-only style is priced by nothing.
    let mut r = Registry::builtin();
    r.set_expression_compiler(None);
    r.register_style(spec(), scalar(RIGHT)).unwrap();
    r.register_style(
        harmonic_spec("bond", "by/expression").expression("k*(r-r0)^2"),
        None,
    )
    .unwrap();
    let err = PotentialCompiler::new(&ff.clone().with_registry(Arc::new(r.clone())))
        .compile(&frame)
        .unwrap_err();
    assert!(
        err.to_string()
            .contains("no kernel for bond `by/expression`"),
        "{err}"
    );
}

/// A compile against a registry without the category names it.
#[test]
fn compiling_an_unregistered_category_is_an_error() {
    let (ff, _) = one_bond("harmonic");
    let empty = Registry::new();
    let err = PotentialCompiler::new(&ff.clone().with_registry(Arc::new(empty.clone())))
        .compile(&Frame::new())
        .unwrap_err();
    assert_eq!(
        err,
        CompileError::Ir(IrError::UnknownCategory {
            category: "bond".into()
        })
    );
}

/// A style registered in one registry is seen by a compile against it and
/// by nothing else.
#[test]
fn a_registry_of_ones_own_is_seen_by_its_compile_only() {
    let (ff, frame) = one_bond("only/here");
    let mut r = Registry::builtin();
    r.register_style(harmonic_spec("bond", "only/here"), scalar(RIGHT))
        .unwrap();
    let pots = PotentialCompiler::new(&ff.clone().with_registry(Arc::new(r.clone())))
        .compile(&frame)
        .unwrap();
    // k (r − r0)² = 2 · 0.6²
    assert!((pots.energy_at(&[0.0, 0.0, 0.0, 1.6, 0.0, 0.0]) - 0.72).abs() < 1e-12);
    let err = PotentialCompiler::new(&ff).compile(&frame).unwrap_err();
    assert!(err.to_string().contains("no kernel"), "{err}");
}

/// A text style parameter is checked to be text, and an array parameter is
/// accepted by a bonded form (it reads it through `ParamColumns::array`).
#[test]
fn text_and_array_params_are_declared_by_kind() {
    let mut r = Registry::builtin();
    let mixing = harmonic_spec("pair", "x")
        .style_params(vec![ParamSpec::new("mixing", ParamDimension::NONE)]);
    assert!(matches!(
        r.register_style(mixing, scalar(RIGHT)),
        Err(IrError::ReservedParam { .. })
    ));
    let table = harmonic_spec("bond", "tabled").params(vec![
        ParamSpec::new("k", "E/L^2".parse().unwrap()),
        ParamSpec::new("r0", ParamDimension::LENGTH),
        ParamSpec::new("table", ParamDimension::ENERGY).kind(ParamKind::Array { rank: 1 }),
    ]);
    r.register_style(table, scalar(RIGHT)).unwrap();
}

// ---------------------------------------------------------------------------
// The expression engine through the registry
// ---------------------------------------------------------------------------

/// Four atoms of type `A`, non-planar; one term of `category` over the
/// first `arity` of them, typed `t`; and a `pairs` list of the three pairs
/// that are not neighbours in the chain.
fn chain(category: &str, arity: usize) -> Frame {
    let mut atoms = Block::new();
    let xyz = [
        [0.0, 0.0, 0.0],
        [1.52, 0.1, 0.05],
        [2.1, 1.45, -0.1],
        [3.55, 1.6, 0.6],
    ];
    for (a, key) in ["x", "y", "z"].iter().enumerate() {
        let v: Vec<F> = xyz.iter().map(|p| p[a]).collect();
        atoms
            .insert(*key, Array1::from_vec(v).into_dyn(), None)
            .unwrap();
    }
    atoms
        .insert(
            "type",
            Array1::from_vec(vec!["A".to_string(); 4]).into_dyn(),
            None,
        )
        .unwrap();
    atoms
        .insert(
            "charge",
            Array1::from_vec(vec![0.4, -0.3, 0.2, -0.25]).into_dyn(),
            None,
        )
        .unwrap();
    let mut frame = Frame::new();
    frame.insert("atoms", atoms);
    if category != "pair" {
        let mut block = Block::new();
        for (key, atom) in ["atomi", "atomj", "atomk", "atoml"]
            .into_iter()
            .zip(0..arity)
        {
            block
                .insert(key, Array1::from_vec(vec![atom as Idx]).into_dyn(), None)
                .unwrap();
        }
        block
            .insert(
                "type",
                Array1::from_vec(vec!["t".to_string()]).into_dyn(),
                None,
            )
            .unwrap();
        frame.insert(format!("{category}s"), block);
    }
    let mut pairs = Block::new();
    pairs
        .insert(
            "atomi",
            Array1::from_vec(vec![0 as Idx, 0, 1]).into_dyn(),
            None,
        )
        .unwrap();
    pairs
        .insert(
            "atomj",
            Array1::from_vec(vec![2 as Idx, 3, 3]).into_dyn(),
            None,
        )
        .unwrap();
    frame.insert("pairs", pairs);
    frame
}

const COORDS: [F; 12] = [
    0.0, 0.0, 0.0, 1.52, 0.1, 0.05, 2.1, 1.45, -0.1, 3.55, 1.6, 0.6,
];

/// An unregistered style that carries an `expression` is priced by it, in a
/// fresh process with nothing registered (the compile fallback).
#[test]
fn an_unregistered_style_with_an_expression_is_priced_by_it() {
    let mut style = Params::new();
    style.set("expression", "-0.5*k*r0^2*log(1-(r/r0)^2)");
    let mut ff = ForceField::new("t");
    ff.def_style("bond", "fene/unregistered", style)
        .unwrap()
        .def_type(
            "t",
            &["A", "A"],
            Params::from_iter([("k", 30.0), ("r0", 2.0)]),
        )
        .unwrap();
    let frame = chain("bond", 2);
    let pots = PotentialCompiler::new(&ff).compile(&frame).unwrap();
    let r = 1.52_f64.hypot(0.1).hypot(0.05);
    let want = -0.5 * 30.0 * 2.0 * 2.0 * (1.0 - (r / 2.0).powi(2)).ln();
    let got = pots.energy_at(&COORDS);
    assert!((got - want).abs() <= 1e-12 * want.abs(), "{got} vs {want}");
}

/// A pair expression reads the two atoms' self rows as `x1`, `x2` and their
/// charges as `q1`, `q2`; one that is not symmetric under exchanging them is
/// refused on its samples.
#[test]
fn a_pair_expression_binds_self_rows_and_must_be_symmetric() {
    let spec = |name: &'static str, expression: &str| {
        StyleSpec::new("pair", name)
            .params(vec![
                ParamSpec::new("epsilon", ParamDimension::ENERGY),
                ParamSpec::new("sigma", ParamDimension::LENGTH),
            ])
            .style_params(vec![ParamSpec::new("cutoff", ParamDimension::LENGTH)])
            .expression(expression)
    };
    let mut r = Registry::builtin();
    let lb = "4*sqrt(epsilon1*epsilon2)*(((sigma1+sigma2)/2/r)^12-((sigma1+sigma2)/2/r)^6)+q1*q2/r";
    r.register_style(spec("lb", lb), None).unwrap();
    let lopsided = spec("lopsided", "epsilon1*sigma2/r").sample(ConformanceSample {
        params: vec![
            ("epsilon".into(), ParamValue::Num(0.2)),
            ("sigma".into(), ParamValue::Num(3.0)),
            ("cutoff".into(), ParamValue::Num(10.0)),
        ],
        q: (2.5, 6.0),
    });
    assert!(matches!(
        r.register_style(lopsided, None),
        Err(IrError::Asymmetric { .. })
    ));

    // Two types: the self rows of each atom, not a mixed value.
    let mut ff = ForceField::new("t");
    ff.def_style("pair", "lb", Params::from_iter([("cutoff", 10.0)]))
        .unwrap()
        .def_type(
            "A",
            &["A"],
            Params::from_iter([("epsilon", 0.2), ("sigma", 3.1)]),
        )
        .unwrap()
        .def_type(
            "B",
            &["B"],
            Params::from_iter([("epsilon", 0.1), ("sigma", 2.5)]),
        )
        .unwrap()
        // A cross row the expression does not read (it reads the self rows).
        .def_type(
            "A-B",
            &["A", "B"],
            Params::from_iter([("epsilon", 9.0), ("sigma", 9.0)]),
        )
        .unwrap();
    let mut frame = chain("pair", 2);
    let types = vec!["A".to_string(), "B".into(), "A".into(), "B".into()];
    frame
        .get_mut("atoms")
        .unwrap()
        .insert("type", Array1::from_vec(types).into_dyn(), None)
        .unwrap();
    let pots = PotentialCompiler::new(&ff.clone().with_registry(Arc::new(r.clone())))
        .compile(&frame)
        .unwrap();
    let rows = [
        (0.2, 3.1, 0.4),
        (0.1, 2.5, -0.3),
        (0.2, 3.1, 0.2),
        (0.1, 2.5, -0.25),
    ];
    let mut want = 0.0;
    for (i, j) in [(0, 2), (0, 3), (1, 3)] {
        let r = (0..3)
            .map(|c| (COORDS[j * 3 + c] - COORDS[i * 3 + c]).powi(2))
            .sum::<F>()
            .sqrt();
        let (a, b): ((F, F, F), (F, F, F)) = (rows[i], rows[j]);
        let s = (a.1 + b.1) / 2.0 / r;
        want += 4.0 * (a.0 * b.0).sqrt() * (s.powi(12) - s.powi(6)) + a.2 * b.2 / r;
    }
    let got = pots.energy_at(&COORDS);
    assert!((got - want).abs() <= 1e-12 * want.abs(), "{got} vs {want}");
}

// ---------------------------------------------------------------------------
// A custom category in a force field (protocol §6)
// ---------------------------------------------------------------------------

/// The Urey–Bradley 1-3 spring of LAMMPS `angle charmm` as an N-body form:
/// `k_ub (r₁₃ − r_ub)²`.
struct UreyBradley;

impl CompoundForm for UreyBradley {
    fn eval(
        &self,
        x: &[[F; 3]],
        arity: usize,
        p: &ParamColumns<'_>,
        e: &mut [F],
        grad: &mut [[F; 3]],
    ) {
        let (k, r0) = (p.get("k_ub").unwrap(), p.get("r_ub").unwrap());
        for t in 0..e.len() {
            let (a, c) = (x[t * arity], x[t * arity + 2]);
            let d = [c[0] - a[0], c[1] - a[1], c[2] - a[2]];
            let r = (d[0] * d[0] + d[1] * d[1] + d[2] * d[2]).sqrt();
            let dr = r - r0[t];
            e[t] = k[t] * dr * dr;
            let g = d.map(|di| 2.0 * k[t] * dr * di / r);
            grad[t * arity] = g.map(|gi| -gi);
            grad[t * arity + 1] = [0.0; 3];
            grad[t * arity + 2] = g;
        }
    }

    fn inputs(&self) -> Vec<String> {
        vec!["k_ub".into(), "r_ub".into()]
    }
}

const UB_EXPRESSION: &str = "k_ub*(distance(p1,p3)-r_ub)^2";

/// `urey_bradley`: arity 3, compound, block `urey_bradleys`.
fn urey_bradley_category() -> CategorySpec {
    CategorySpec::custom(
        "urey_bradley",
        3,
        Coordinate::Compound,
        EndpointOrder::Reversible,
    )
}

fn ub_params() -> Vec<ParamSpec> {
    vec![
        ParamSpec::new("k_ub", "E/L^2".parse().unwrap()),
        ParamSpec::new("r_ub", ParamDimension::LENGTH),
    ]
}

/// A builtin registry with `urey_bradley` and two styles of it: `harmonic`
/// (the native form) and `expr` (priced by its expression alone).
fn ub_registry() -> Registry {
    let mut r = Registry::builtin();
    r.register_category(urey_bradley_category()).unwrap();
    r.register_style(
        StyleSpec::new("urey_bradley", "harmonic").params(ub_params()),
        Some(Kernel::Compound(Arc::new(UreyBradley))),
    )
    .unwrap();
    r.register_style(
        StyleSpec::new("urey_bradley", "expr")
            .params(ub_params())
            .expression(UB_EXPRESSION),
        None,
    )
    .unwrap();
    r
}

/// The two type rows every case prices: `(name, k_ub, r_ub)`.
const UB_TYPES: [(&str, F, F); 2] = [("t", 20.0, 2.45), ("u", 11.0, 2.2)];

/// [`chain`]'s four atoms with two terms of `category` (3 atoms each,
/// typed `t` and `u`) in its block `<category>s`.
fn two_terms(category: &str) -> Frame {
    let mut frame = chain(category, 3);
    let mut block = Block::new();
    for (key, atoms) in [
        ("atomi", [0 as Idx, 1]),
        ("atomj", [1, 2]),
        ("atomk", [2, 3]),
    ] {
        block
            .insert(key, Array1::from_vec(atoms.to_vec()).into_dyn(), None)
            .unwrap();
    }
    block
        .insert(
            "type",
            Array1::from_vec(vec!["t".to_string(), "u".to_string()]).into_dyn(),
            None,
        )
        .unwrap();
    frame.insert(format!("{category}s"), block);
    frame
}

/// The reference: LAMMPS `angle charmm` with K = 0, a pure 1-3 spring.
fn charmm_reference() -> (F, Vec<F>) {
    let mut ff = ForceField::new("ref");
    let style = ff.def_style("angle", "charmm", Params::new()).unwrap();
    for (name, k_ub, r_ub) in UB_TYPES {
        style
            .def_type(
                name,
                &["A", "A", "A"],
                Params::from_iter([
                    ("k", 0.0),
                    ("theta0", 109.5),
                    ("k_ub", k_ub),
                    ("r_ub", r_ub),
                ]),
            )
            .unwrap();
    }
    PotentialCompiler::new(&ff)
        .compile(&two_terms("angle"))
        .unwrap()
        .energy_forces_at(&COORDS)
}

/// `ff` (empty) with one `urey_bradley` style `style` (style params
/// `params`) over [`UB_TYPES`], defined through `def`.
fn ub_ff(
    mut ff: ForceField,
    style: &str,
    params: Params,
    def: impl FnOnce(&mut ForceField, Params) -> Result<&mut crate::ff::forcefield::Style, DefError>,
) -> ForceField {
    let s = def(&mut ff, params).unwrap();
    assert_eq!(s.category(), "urey_bradley");
    assert_eq!(s.name(), style);
    for (name, k_ub, r_ub) in UB_TYPES {
        s.def_type(
            name,
            &["A", "A", "A"],
            Params::from_iter([("k_ub", k_ub), ("r_ub", r_ub)]),
        )
        .unwrap();
    }
    ff
}

fn assert_same_energy_forces(label: &str, got: (F, Vec<F>), want: &(F, Vec<F>)) {
    let scale = want.0.abs().max(1.0);
    assert!(
        (got.0 - want.0).abs() <= 1e-12 * scale,
        "{label}: energy {} vs {}",
        got.0,
        want.0
    );
    let fmax = want.1.iter().fold(1.0_f64, |m, f| m.max(f.abs()));
    for (i, (a, b)) in got.1.iter().zip(&want.1).enumerate() {
        assert!(
            (a - b).abs() <= 1e-12 * fmax,
            "{label}: force[{i}] {a} vs {b}"
        );
    }
}

/// A custom category registered in a registry is a style category of a
/// force field (`def_style` against a force field carrying the registry, its arity from the registration), and its
/// block `urey_bradleys` is priced by a native compound form and by an
/// expression alike: both equal LAMMPS `angle charmm` with K = 0, at both
/// compile doors.
#[test]
fn a_custom_category_prices_its_block_like_angle_charmm_without_k() {
    let r = ub_registry();
    let want = charmm_reference();
    assert!(want.0 > 0.0, "a non-trivial reference: {}", want.0);
    let frame = two_terms("urey_bradley");
    for style in ["harmonic", "expr"] {
        let seed = ForceField::new("ub").with_registry(Arc::new(r.clone()));
        let ff = ub_ff(seed, style, Params::new(), |ff, p| {
            ff.def_style("urey_bradley", style, p)
        });
        let StyleDefs::Relation { arity, types, .. } = ff.styles()[0].defs() else {
            panic!("a custom category is a relation");
        };
        assert_eq!((*arity, types.len()), (3, 2));
        let compiler = PotentialCompiler::new(&ff);
        let pots = compiler.compile(&frame).unwrap();
        assert_eq!(pots.members().len(), 1, "{style}");
        assert_same_energy_forces(style, pots.energy_forces_at(&COORDS), &want);
        let typed = compiler.compile_typed(&frame).unwrap();
        assert_eq!(typed.len(), 1, "{style}");
        assert!(
            typed[0].1.is_none(),
            "a relation takes no special-bonds weight"
        );
        // No block, nothing to price.
        assert!(
            compiler
                .compile(&chain("bond", 2))
                .unwrap()
                .members()
                .is_empty()
        );
    }
}

/// A type of a custom category names exactly its arity of endpoints; the
/// category is unknown to a registry that does not declare it.
#[test]
fn a_custom_category_checks_its_arity_and_needs_its_registration() {
    let r = ub_registry();
    let mut ff = ForceField::new("t").with_registry(Arc::new(r));
    let style = ff
        .def_style("urey_bradley", "harmonic", Params::new())
        .unwrap();
    let err = style.def_type("x", &["A", "A"], Params::new()).unwrap_err();
    assert!(
        matches!(&err, DefError::Arity { category, got: 2, .. } if category == "urey_bradley"),
        "{err:?}"
    );
    assert!(err.to_string().contains("expected 3 endpoints"), "{err}");

    let mut fresh = ForceField::new("t").with_registry(Arc::new(Registry::builtin()));
    assert!(matches!(
        fresh.def_style("urey_bradley", "harmonic", Params::new()),
        Err(DefError::UnknownCategory(c)) if c == "urey_bradley"
    ));
    // Held by the force field already, the category takes the arity it has,
    // against a registry that does not declare it.
    let mut ff = ff.with_registry(Arc::new(Registry::builtin()));
    ff.def_style("urey_bradley", "other", Params::new())
        .unwrap()
        .def_type("y", &["A", "B", "C"], Params::new())
        .unwrap();
    assert!(matches!(
        ff.def_style(Category::new("urey_bradley").arity(2), "two", Params::new()),
        Err(DefError::CategoryArity {
            expected: 3,
            got: 2,
            ..
        })
    ));
}

/// A custom category round-trips through the record's section (its arity
/// from the endpoint columns) and prices identically afterwards.
#[cfg(feature = "zarr")]
#[test]
fn a_custom_category_round_trips_through_its_section() {
    let r = ub_registry();
    let seed = ForceField::new("ub").with_registry(Arc::new(r.clone()));
    let ff = ub_ff(seed, "harmonic", Params::new(), |ff, p| {
        ff.def_style("urey_bradley", "harmonic", p)
    });
    let section = ForceFieldSection::from_forcefield(&ff).unwrap();
    let table = section.table("urey_bradley", "harmonic").unwrap();
    assert!(table.contains_key("ktom") && !table.contains_key("ltom"));
    let back = section.to_forcefield().unwrap();
    assert_eq!(back.styles().len(), 1);
    assert_eq!(back.styles()[0].category(), "urey_bradley");
    assert_eq!(back.styles()[0].arity(), 3);
    assert_eq!(back.styles()[0].type_rows(), ff.styles()[0].type_rows());
    assert_eq!(
        ForceFieldSection::from_forcefield(&back).unwrap().document,
        section.document
    );
    let frame = two_terms("urey_bradley");
    let price = |ff: &ForceField| {
        PotentialCompiler::new(&ff.clone().with_registry(Arc::new(r.clone())))
            .compile(&frame)
            .unwrap()
            .energy_forces_at(&COORDS)
    };
    let (e0, f0) = price(&ff);
    let (e1, f1) = price(&back);
    assert_eq!(e0.to_bits(), e1.to_bits());
    assert_eq!(f0, f1);
}

/// A category nothing declares is kept from a record with the arity of its
/// endpoint columns; with rows and no expression, compiling it is refused
/// naming the style, and with an `expression` it is priced by it alone.
#[cfg(feature = "zarr")]
#[test]
fn an_unregistered_category_round_trips_and_is_priced_only_by_an_expression() {
    let bare = ub_ff(ForceField::new("ub"), "spring", Params::new(), |ff, p| {
        ff.def_style(Category::new("urey_bradley").arity(3), "spring", p)
    });
    let back = ForceFieldSection::from_forcefield(&bare)
        .unwrap()
        .to_forcefield()
        .unwrap();
    assert_eq!(back.styles()[0].arity(), 3);
    assert_eq!(back.styles()[0].type_rows(), bare.styles()[0].type_rows());
    let compiler = PotentialCompiler::new(&back);
    let err = compiler.compile(&two_terms("urey_bradley")).unwrap_err();
    assert!(
        err.to_string()
            .contains("no kernel for urey_bradley `spring`"),
        "{err}"
    );
    let err = compiler
        .compile_typed(&two_terms("urey_bradley"))
        .unwrap_err();
    assert!(err.to_string().contains("urey_bradley `spring`"), "{err}");
    // Without its block there is nothing to price, and nothing refused.
    assert!(
        compiler
            .compile(&chain("bond", 2))
            .unwrap()
            .members()
            .is_empty()
    );

    let mut style = Params::new();
    style.set("expression", UB_EXPRESSION);
    let priced = ub_ff(ForceField::new("ub"), "spring", style, |ff, p| {
        ff.def_style(Category::new("urey_bradley").arity(3), "spring", p)
    });
    let section = ForceFieldSection::from_forcefield(&priced).unwrap();
    assert_eq!(
        section.document["styles"][0]["expression"],
        serde_json::json!(UB_EXPRESSION)
    );
    let back = section.to_forcefield().unwrap();
    let got = PotentialCompiler::new(&back)
        .compile(&two_terms("urey_bradley"))
        .unwrap()
        .energy_forces_at(&COORDS);
    assert_same_energy_forces("unregistered expression", got, &charmm_reference());
}

// ---------------------------------------------------------------------------
// Persistence (protocol §7, D9, D16)
// ---------------------------------------------------------------------------

/// LAMMPS `bond_style fene` (with its WCA term), as an expression.
#[cfg(feature = "zarr")]
const FENE: &str = "-0.5*k*r0^2*log(1-(r/r0)^2) + step(2^(1/6)*sigma-r)*(4*epsilon*((sigma/r)^12-(sigma/r)^6)+epsilon)";

#[cfg(feature = "zarr")]
fn fene_spec() -> StyleSpec {
    StyleSpec::new("bond", "fene")
        .params(vec![
            ParamSpec::new("k", "E/L^2".parse().unwrap()),
            ParamSpec::new("r0", ParamDimension::LENGTH),
            ParamSpec::new("epsilon", ParamDimension::ENERGY),
            ParamSpec::new("sigma", ParamDimension::LENGTH),
        ])
        .expression(FENE)
}

/// A tabulated torsion `E(φ)`: `table` holds `N` energies on the grid
/// `φᵢ = −π + 2πi/N`, interpolated linearly and periodically (a Tier-2
/// form reading an array parameter).
#[cfg(feature = "zarr")]
struct TableLinear;

#[cfg(feature = "zarr")]
impl TableLinear {
    fn at(table: &[F], phi: F) -> (F, F) {
        let n = table.len();
        let h = 2.0 * std::f64::consts::PI / n as F;
        let s = (phi + std::f64::consts::PI) / h;
        let i = s.floor();
        let w = s - i;
        let i = (i as usize) % n;
        let j = (i + 1) % n;
        (
            table[i] * (1.0 - w) + table[j] * w,
            (table[j] - table[i]) / h,
        )
    }
}

#[cfg(feature = "zarr")]
impl ScalarForm for TableLinear {
    fn eval(&self, phi: &[F], p: &ParamColumns<'_>, e: &mut [F], de: &mut [F]) {
        let table = p.array("table").unwrap();
        for t in 0..phi.len() {
            let row: Vec<F> = table
                .index_axis(ndarray::Axis(0), t)
                .iter()
                .copied()
                .collect();
            (e[t], de[t]) = Self::at(&row, phi[t]);
        }
    }
}

#[cfg(feature = "zarr")]
fn table_spec() -> StyleSpec {
    StyleSpec::new("dihedral", "table/linear").params(vec![
        ParamSpec::new("table", ParamDimension::ENERGY).kind(ParamKind::Array { rank: 1 }),
    ])
}

/// The table of type `t`: twelve energies, none of them round.
#[cfg(feature = "zarr")]
fn torsion_table() -> ndarray::ArrayD<F> {
    ndarray::ArrayD::from_shape_fn(vec![12], |ix| {
        let i = ix[0] as F;
        1.25 + (0.7 * i).sin() / 3.0 - 0.01 * i * i
    })
}

/// A registry extended the way a third party would: `bond fene` by its
/// expression, `urey_bradley` with an expression style and a native-only
/// one, and the native-only `dihedral table/linear` with an array param.
#[cfg(feature = "zarr")]
fn persist_registry() -> Registry {
    let mut r = ub_registry();
    r.register_style(fene_spec(), None).unwrap();
    r.register_style(table_spec(), Some(Kernel::Scalar(Arc::new(TableLinear))))
        .unwrap();
    r
}

/// The four records of the round trip, each a force field of one custom
/// style defined against `r` with **no** expression of its own; the table
/// record also holds a category nothing ever registers.
#[cfg(feature = "zarr")]
fn persist_cases(r: &Registry) -> Vec<(&'static str, ForceField, Frame)> {
    let registry = Arc::new(r.clone());
    let mut fene = ForceField::new("fene").with_registry(registry.clone());
    fene.def_style("bond", "fene", Params::new())
        .unwrap()
        .def_type(
            "t",
            &["A", "A"],
            Params::from_iter([("k", 30.0), ("r0", 2.25), ("epsilon", 1.1), ("sigma", 1.4)]),
        )
        .unwrap();
    let mut table = ForceField::new("table").with_registry(registry.clone());
    let mut row = Params::new();
    row.set("table", torsion_table());
    table
        .def_style("dihedral", "table/linear", Params::new())
        .unwrap()
        .def_type("t", &["A", "A", "A", "A"], row)
        .unwrap();
    table
        .def_style(
            Category::new("bespoke").arity(2),
            "x",
            Params::from_iter([("w", 0.5)]),
        )
        .unwrap()
        .def_type("q", &["A", "B"], Params::from_iter([("k", 1.5)]))
        .unwrap();
    vec![
        ("fene", fene, chain("bond", 2)),
        (
            "ub_expr",
            ub_ff(
                ForceField::new("ub").with_registry(registry.clone()),
                "expr",
                Params::new(),
                |ff, p| ff.def_style("urey_bradley", "expr", p),
            ),
            two_terms("urey_bradley"),
        ),
        (
            "ub_native",
            ub_ff(
                ForceField::new("ub").with_registry(registry.clone()),
                "harmonic",
                Params::new(),
                |ff, p| ff.def_style("urey_bradley", "harmonic", p),
            ),
            two_terms("urey_bradley"),
        ),
        ("table", table, chain("dihedral", 4)),
    ]
}

#[cfg(feature = "zarr")]
fn bits(e: F, f: &[F]) -> Vec<u64> {
    std::iter::once(e)
        .chain(f.iter().copied())
        .map(F::to_bits)
        .collect()
}

/// D16: a registered custom style with no expression of its own is written
/// with the registry's; a built-in style, and a native-only custom one, with
/// none; an instance's own expression byte for byte.
#[cfg(feature = "zarr")]
#[test]
fn a_section_writes_a_custom_styles_registry_expression() {
    let r = persist_registry();
    let expression = |ff: &ForceField| {
        ForceFieldSection::from_forcefield(&ff.clone().with_registry(Arc::new(r.clone())))
            .unwrap()
            .document["styles"][0]
            .get("expression")
            .cloned()
    };
    let cases = persist_cases(&r);
    assert_eq!(expression(&cases[0].1), Some(serde_json::json!(FENE)));
    assert_eq!(
        expression(&cases[1].1),
        Some(serde_json::json!(UB_EXPRESSION))
    );
    assert_eq!(expression(&cases[2].1), None, "native only");
    assert_eq!(expression(&cases[3].1), None, "native only");
    // The process-wide registry does not hold them: nothing to write.
    let mut global = cases[0].1.clone();
    global.registry = crate::ff::style_registry::RegistryRef::Global;
    assert!(
        ForceFieldSection::from_forcefield(&global)
            .unwrap()
            .document["styles"][0]
            .get("expression")
            .is_none()
    );

    // A built-in writes none, though the registry knows its expression.
    let mut harmonic = ForceField::new("h");
    harmonic
        .def_style("bond", "harmonic", Params::new())
        .unwrap();
    assert!(r.style("bond", "harmonic").unwrap().0.expression.is_some());
    assert_eq!(expression(&harmonic), None);

    // An instance's own expression wins, byte for byte (spacing kept).
    let own = "-0.5*k*r0^2*log(1 - (r/r0)^2)   + step(2^(1/6)*sigma-r)*(4*epsilon*((sigma/r)^12-(sigma/r)^6)+epsilon)";
    let mut params = Params::new();
    params.set("expression", own);
    let mut ff = ForceField::new("own").with_registry(Arc::new(r.clone()));
    ff.def_style("bond", "fene", params).unwrap();
    let section = ForceFieldSection::from_forcefield(&ff).unwrap();
    assert_eq!(
        section.document["styles"][0]["expression"],
        serde_json::json!(own)
    );
    let back = section.to_forcefield().unwrap();
    assert_eq!(
        back.styles()[0]
            .params()
            .get("expression")
            .and_then(|v| v.as_text()),
        Some(own)
    );
}

/// D16: an instance expression that differs from the registry's is checked
/// against the registered kernel at first compile — a native form or the
/// registry's own expression — and the registered kernel prices it.
#[test]
fn an_instance_expression_must_agree_with_the_registered_kernel() {
    let mut r = Registry::builtin();
    r.register_style(
        harmonic_spec("bond", "spring").expression("k*(r-r0)^2"),
        scalar(RIGHT),
    )
    .unwrap();
    r.register_style(
        harmonic_spec("bond", "spring/x").expression("k*(r-r0)^2"),
        None,
    )
    .unwrap();
    let with = |name: &str, expression: &str| {
        let (mut ff, frame) = one_bond(name);
        ff.get_style_mut("bond", name)
            .unwrap()
            .set_param("expression", expression);
        PotentialCompiler::new(&ff.clone().with_registry(Arc::new(r.clone()))).compile(&frame)
    };
    for name in ["spring", "spring/x"] {
        // Spelled otherwise, the same energy: priced by the registered kernel.
        let e = with(name, "k*(r-r0)*(r-r0)")
            .unwrap()
            .energy_at(&[0.0, 0.0, 0.0, 1.6, 0.0, 0.0]);
        assert!((e - 0.72).abs() < 1e-12, "{name}: {e}");
        // Another energy: refused, naming the style, at every compile.
        for _ in 0..2 {
            let err = with(name, "k*(r-r0)^2 + 0.001").unwrap_err();
            assert!(
                err.to_string().contains(name) && err.to_string().contains("disagree"),
                "{name}: {err}"
            );
        }
        // An expression the style cannot bind is refused by name too.
        let err = with(name, "k*(theta-r0)^2").unwrap_err();
        assert!(err.to_string().contains("theta"), "{name}: {err}");
    }
}

/// The env var naming the directory a parent test hands a fresh process.
#[cfg(feature = "filesystem")]
const FRESH_DIR: &str = "MOLRS_IR_PERSIST_DIR";

/// Custom styles persist in `.mrec`: a fresh process with nothing
/// registered reads every record back, prices the expression styles bit for
/// bit as the registering process did, keeps the native-only styles and the
/// array table, and refuses to price those by name (protocol §7, P-Rust
/// `mrec_round_trip`).
#[cfg(feature = "filesystem")]
#[test]
fn custom_styles_persist_to_a_fresh_process() {
    let r = persist_registry();
    let dir = tempfile::tempdir().unwrap();
    let mut expected = serde_json::Map::new();
    for (name, ff, frame) in persist_cases(&r) {
        let section = ForceFieldSection::from_forcefield(&ff).unwrap();
        let mut record = molrs::io::mrec::MolRec::new();
        record.forcefield = Some(section);
        molrs::io::write_mrec(&dir.path().join(format!("{name}.mrec")), &record).unwrap();
        let (e, f) = PotentialCompiler::new(&ff)
            .compile(&frame)
            .unwrap()
            .energy_forces_at(&COORDS);
        assert!(e.abs() > 1e-3, "{name}: a non-trivial energy {e}");
        expected.insert(name.into(), serde_json::json!(bits(e, &f)));
    }
    std::fs::write(
        dir.path().join("expected.json"),
        serde_json::Value::Object(expected).to_string(),
    )
    .unwrap();

    let test = "ff::ir::tests::fresh_process_reads_custom_styles";
    let out = std::process::Command::new(std::env::current_exe().unwrap())
        .args([test, "--exact", "--nocapture", "--test-threads=1"])
        .env(FRESH_DIR, dir.path())
        .output()
        .unwrap();
    let stdout = String::from_utf8_lossy(&out.stdout);
    let stderr = String::from_utf8_lossy(&out.stderr);
    assert!(
        out.status.success() && stdout.contains("1 passed"),
        "the fresh process failed:\n{stdout}\n{stderr}"
    );
}

/// The fresh-process half of [`custom_styles_persist_to_a_fresh_process`]:
/// a no-op unless that test runs it, alone, in a process of its own.
#[cfg(feature = "filesystem")]
#[test]
fn fresh_process_reads_custom_styles() {
    let Some(dir) = std::env::var_os(FRESH_DIR) else {
        return;
    };
    let dir = std::path::PathBuf::from(dir);
    let expected: serde_json::Value =
        serde_json::from_str(&std::fs::read_to_string(dir.join("expected.json")).unwrap()).unwrap();
    // Nothing is registered here: the process-wide registry is the builtin one.
    assert!(crate::ff::style_registry::with_global_registry(|g| g
        .category("urey_bradley")
        .is_none()
        && g.style("bond", "fene").is_none()
        && g.style("dihedral", "table/linear").is_none()));
    let read = |name: &str| {
        let section = molrs::io::mrec::MrecFile::open(&dir.join(format!("{name}.mrec")))
            .unwrap()
            .forcefield()
            .unwrap()
            .unwrap();
        section.to_forcefield().unwrap()
    };
    let want = |name: &str| -> Vec<u64> { serde_json::from_value(expected[name].clone()).unwrap() };

    // Priced by the expression the record carries, bit for bit.
    for (name, frame) in [
        ("fene", chain("bond", 2)),
        ("ub_expr", two_terms("urey_bradley")),
    ] {
        let ff = read(name);
        let (e, f) = PotentialCompiler::new(&ff)
            .compile(&frame)
            .unwrap_or_else(|err| panic!("{name}: {err}"))
            .energy_forces_at(&COORDS);
        assert_eq!(bits(e, &f), want(name), "{name}");
    }
    let fene = read("fene");
    assert_eq!(
        fene.styles()[0]
            .params()
            .get("expression")
            .and_then(|v| v.as_text()),
        Some(FENE)
    );

    // Native-only: read whole, refused to price by name.
    let ub = read("ub_native");
    assert_eq!(ub.styles()[0].arity(), 3);
    assert_eq!(ub.styles()[0].type_rows().len(), 2);
    let err = PotentialCompiler::new(&ub)
        .compile(&two_terms("urey_bradley"))
        .unwrap_err();
    assert!(
        err.to_string().contains("no kernel for urey_bradley `harmonic`: register it (molpy.ff.style_registry.register_style) or give it an expression"),
        "{err}"
    );
    let table = read("table");
    let rows = table.styles()[0].type_rows();
    assert_eq!(
        rows[0].2.get("table").and_then(|v| v.as_array()),
        Some(&torsion_table()),
        "bit for bit"
    );
    let err = PotentialCompiler::new(&table)
        .compile(&chain("dihedral", 4))
        .unwrap_err();
    assert!(
        err.to_string()
            .contains("no kernel for dihedral `table/linear`"),
        "{err}"
    );
    // The category nothing registers is kept, rows and style params.
    let bespoke = table.get_style("bespoke", "x").unwrap();
    assert_eq!(
        (
            bespoke.arity(),
            bespoke.params().get("w").and_then(|v| v.as_num())
        ),
        (2, Some(0.5))
    );
    assert_eq!(table.get_relationtypes("bespoke")[0].endpoints.len(), 2);
}

/// The array form against a hand interpolation, and the same energy after
/// the trip through a section with its kernel registered (D9).
#[cfg(feature = "zarr")]
#[test]
fn an_array_param_style_prices_its_table_and_round_trips() {
    let r = persist_registry();
    let (_, ff, frame) = persist_cases(&r).pop().unwrap();
    let price = |ff: &ForceField| {
        PotentialCompiler::new(&ff.clone().with_registry(Arc::new(r.clone())))
            .compile(&frame)
            .unwrap()
            .energy_forces_at(&COORDS)
    };
    let (e, f) = price(&ff);
    let phi = crate::ff::potential::flat_coords::compute_dihedral(&COORDS, 0, 1, 2, 3);
    let table: Vec<F> = torsion_table().iter().copied().collect();
    let (hand, _) = TableLinear::at(&table, phi);
    assert!((e - hand).abs() <= 1e-12 * hand.abs(), "{e} vs {hand}");
    let back = ForceFieldSection::from_forcefield(&ff)
        .unwrap()
        .to_forcefield()
        .unwrap();
    assert_eq!(bits(e, &f), {
        let (e, f) = price(&back);
        bits(e, &f)
    });
}
