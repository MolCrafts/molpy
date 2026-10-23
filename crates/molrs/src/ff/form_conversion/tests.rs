//! P3 acceptance (`ff-ir-01`): random parameters × random configurations
//! price the same before and after every exact conversion (1e-12); an
//! out-of-image conversion raises, naming the type and the condition; a
//! fit's residual is reported and monotone in its metric; a third party's
//! expression style joins a family by registering a codec.

use crate::ff::potential::fixtures::OneShot;
use std::f64::consts::PI;

use rand::rngs::StdRng;
use rand::{RngExt, SeedableRng};

use crate::ff::compile::PotentialCompiler;
use crate::ff::forcefield::ForceField;
use crate::ff::form_conversion::FitMetric;
use crate::ff::ir::torsion::{CosineTermForm, OplsForm, PeriodicForm};
use crate::ff::ir::{FormCodec, FormRefusal, TypeParams};
use crate::ff::ir::{IrError, ParamDimension, ParamSpec, StyleSpec};
use crate::ff::ir::{Params, pair_key};
use crate::ff::style_registry::Registry;
use molrs::core::Block;
use molrs::core::Frame;
use molrs::op::{F, Idx};
use ndarray::Array1;

const SEED: u64 = 0x0070_1510_0009;
const ENDS: [&str; 4] = ["atomi", "atomj", "atomk", "atoml"];

// ── frames and energies ─────────────────────────────────────────────────────

/// A frame whose `block` holds one row per `(atoms, type)`.
fn frame(block: &str, rows: &[(&[usize], &str)]) -> Frame {
    let arity = rows[0].0.len();
    let mut b = Block::new();
    for (e, key) in ENDS[..arity].iter().enumerate() {
        let col: Vec<Idx> = rows.iter().map(|(a, _)| a[e] as Idx).collect();
        b.insert(*key, Array1::from_vec(col).into_dyn(), None)
            .unwrap();
    }
    let types: Vec<String> = rows.iter().map(|(_, t)| (*t).to_owned()).collect();
    b.insert("type", Array1::from_vec(types).into_dyn(), None)
        .unwrap();
    let mut f = Frame::new();
    f.insert(block, b);
    f
}

fn energy(ff: &ForceField, registry: &Registry, frame: &Frame, coords: &[F]) -> F {
    PotentialCompiler::new(
        &ff.clone()
            .with_registry(std::sync::Arc::new(registry.clone())),
    )
    .compile(frame)
    .unwrap_or_else(|e| panic!("{e}\n{ff:#?}"))
    .energy_at(coords)
}

fn coords(rng: &mut StdRng, atoms: usize) -> Vec<F> {
    (0..3 * atoms)
        .map(|_| rng.random_range(-1.5..1.5))
        .collect()
}

/// `a` and `b` agree to 1e-12 of their magnitude (and of 1).
fn assert_same(a: F, b: F, what: &str) {
    assert!(
        (a - b).abs() <= 1e-12 * a.abs().max(b.abs()).max(1.0),
        "{what}: {a} vs {b} (Δ {:.2e})",
        a - b
    );
}

/// A force field of one `category style` row `t` on atom types a, b, c, d.
fn one_row(category: &str, style: &str, row: Params) -> ForceField {
    let arity = block_of(category).1;
    let mut ff = ForceField::new("p3");
    ff.def_style(category, style, Params::new())
        .unwrap()
        .def_type("t", &["a", "b", "c", "d"][..arity], row)
        .unwrap();
    ff
}

fn block_of(category: &str) -> (&'static str, usize) {
    match category {
        "bond" => ("bonds", 2),
        "angle" => ("angles", 3),
        "dihedral" => ("dihedrals", 4),
        "improper" => ("impropers", 4),
        other => panic!("{other}"),
    }
}

/// `ff` and `other` price one `category` term of type `t` alike at 8 random
/// configurations.
fn assert_same_energy(
    rng: &mut StdRng,
    registry: &Registry,
    category: &str,
    ff: &ForceField,
    other: &ForceField,
    what: &str,
) {
    let (block, arity) = block_of(category);
    let atoms: Vec<usize> = (0..arity).collect();
    let f = frame(block, &[(&atoms, "t")]);
    for _ in 0..8 {
        let x = coords(rng, arity);
        assert_same(
            energy(ff, registry, &f, &x),
            energy(other, registry, &f, &x),
            what,
        );
    }
}

// ── random rows of every built-in member ────────────────────────────────────

fn pairs(p: &[(&str, F)]) -> Params {
    Params::from_iter(p.iter().copied())
}

fn c(rng: &mut StdRng) -> F {
    rng.random_range(-5.0..5.0)
}

fn n(rng: &mut StdRng) -> F {
    rng.random_range(0..=6) as F
}

fn sign(rng: &mut StdRng) -> F {
    if rng.random_range(0..2) == 0 {
        1.0
    } else {
        -1.0
    }
}

/// A phase in degrees: a quarter of the time on the axes, else anywhere.
fn phase(rng: &mut StdRng) -> F {
    match rng.random_range(0..8) {
        0 => 0.0,
        1 => 180.0,
        _ => rng.random_range(-540.0..540.0),
    }
}

/// A random row of every built-in style with a form codec, in its own domain
/// (and, for the members with a condition on embedding, in the canonical
/// style's image: charmm `w = 0`, class2 `k3 = k4 = 0`, charmm angle
/// `k_ub = 0`).
fn random_rows(rng: &mut StdRng) -> Vec<(&'static str, &'static str, Params)> {
    let mut periodic = Params::new();
    for m in 1..=rng.random_range(1..=4) {
        periodic.set(&format!("k{m}"), c(rng));
        periodic.set(&format!("periodicity{m}"), n(rng));
        periodic.set(&format!("phase{m}"), phase(rng));
    }
    let mut nharmonic = Params::new();
    for i in 1..=rng.random_range(1..=7) {
        nharmonic.set(&format!("a{i}"), c(rng));
    }
    vec![
        ("dihedral", "periodic", periodic),
        (
            "dihedral",
            "charmm",
            pairs(&[
                ("k", c(rng)),
                ("periodicity", n(rng)),
                ("phase", phase(rng)),
                ("w", 0.0),
            ]),
        ),
        (
            "dihedral",
            "opls",
            pairs(&[
                ("k1", c(rng)),
                ("k2", c(rng)),
                ("k3", c(rng)),
                ("k4", c(rng)),
            ]),
        ),
        (
            "dihedral",
            "multi/harmonic",
            pairs(&[
                ("a1", c(rng)),
                ("a2", c(rng)),
                ("a3", c(rng)),
                ("a4", c(rng)),
                ("a5", c(rng)),
            ]),
        ),
        ("dihedral", "nharmonic", nharmonic),
        (
            "dihedral",
            "harmonic",
            pairs(&[("k", c(rng)), ("sign", sign(rng)), ("periodicity", n(rng))]),
        ),
        (
            "dihedral",
            "class2",
            pairs(&[
                ("k1", c(rng)),
                ("phi1", phase(rng)),
                ("k2", c(rng)),
                ("phi2", phase(rng)),
                ("k3", c(rng)),
                ("phi3", phase(rng)),
            ]),
        ),
        (
            "dihedral",
            "rb",
            pairs(&[
                ("c0", c(rng)),
                ("c1", c(rng)),
                ("c2", c(rng)),
                ("c3", c(rng)),
                ("c4", c(rng)),
                ("c5", c(rng)),
            ]),
        ),
        (
            "improper",
            "cvff",
            pairs(&[("k", c(rng)), ("sign", sign(rng)), ("periodicity", n(rng))]),
        ),
        (
            "improper",
            "periodic",
            pairs(&[
                ("k", c(rng)),
                ("periodicity", n(rng)),
                ("phase", phase(rng)),
            ]),
        ),
        (
            "bond",
            "harmonic",
            pairs(&[
                ("k", rng.random_range(10.0..500.0)),
                ("r0", rng.random_range(0.8..2.0)),
            ]),
        ),
        (
            "bond",
            "class2",
            pairs(&[
                ("r0", rng.random_range(0.8..2.0)),
                ("k2", rng.random_range(10.0..500.0)),
                ("k3", 0.0),
                ("k4", 0.0),
            ]),
        ),
        (
            "angle",
            "harmonic",
            pairs(&[
                ("k", rng.random_range(10.0..100.0)),
                ("theta0", rng.random_range(60.0..180.0)),
            ]),
        ),
        (
            "angle",
            "class2",
            pairs(&[
                ("theta0", rng.random_range(60.0..180.0)),
                ("k2", rng.random_range(10.0..100.0)),
                ("k3", 0.0),
                ("k4", 0.0),
            ]),
        ),
        (
            "angle",
            "charmm",
            pairs(&[
                ("k", rng.random_range(10.0..100.0)),
                ("theta0", rng.random_range(60.0..180.0)),
                ("k_ub", 0.0),
                ("r_ub", rng.random_range(1.5..2.5)),
            ]),
        ),
    ]
}

/// Every style of `category` with a codec of `family`.
fn members(registry: &Registry, category: &str, family: &str) -> Vec<String> {
    registry
        .forms()
        .filter(|(c, _, f)| *c == category && f.family == family)
        .map(|(_, s, _)| s.to_owned())
        .collect()
}

// ── the registry ────────────────────────────────────────────────────────────

#[test]
fn every_builtin_family_has_one_canonical_style() {
    let r = Registry::builtin();
    for (family, canonical) in [
        ("torsion", ("dihedral", "periodic")),
        ("bond", ("bond", "harmonic")),
        ("angle", ("angle", "harmonic")),
        ("lj", ("pair", "lj/cut")),
    ] {
        assert_eq!(r.canonical_form(family), Ok(canonical));
    }
    assert!(
        r.form("improper", "harmonic").is_none(),
        "not a Fourier series"
    );
    assert!(r.form("bond", "morse").is_none());
    assert!(r.form("pair", "lj/charmm").is_none(), "it always switches");
}

#[test]
fn register_form_refuses_what_does_not_conform() {
    let mut r = Registry::builtin();
    let identity = |tp: &TypeParams| -> Result<TypeParams, FormRefusal> { Ok(tp.clone()) };
    // A style that is not registered.
    assert_eq!(
        r.register_form(
            "dihedral",
            "nope",
            FormCodec::new("torsion", identity, identity)
        ),
        Err(IrError::NoKernel {
            category: "dihedral".into(),
            style: "nope".into()
        })
    );
    // A built-in's codec is sealed.
    assert!(matches!(
        r.register_form(
            "dihedral",
            "opls",
            FormCodec::new("torsion", identity, identity)
        ),
        Err(IrError::Sealed { .. })
    ));
    // A second canonical style of one family.
    r.register_style(
        StyleSpec::new("dihedral", "cos3")
            .params(vec![ParamSpec::new("k", ParamDimension::ENERGY)])
            .expression("k*(1+cos(3*phi))"),
        None,
    )
    .unwrap();
    let refused = r
        .register_form(
            "dihedral",
            "cos3",
            FormCodec::new("torsion", identity, identity).as_canonical(),
        )
        .unwrap_err();
    assert!(
        matches!(&refused, IrError::FormConflict { family, reason }
            if family == "torsion" && reason.contains("dihedral `periodic`")),
        "{refused}"
    );
    // The same codec twice is a no-op; another is a conflict.
    let codec = FormCodec::new("torsion", identity, identity);
    r.register_form("dihedral", "cos3", codec.clone()).unwrap();
    r.register_form("dihedral", "cos3", codec).unwrap();
    assert!(matches!(
        r.register_form(
            "dihedral",
            "cos3",
            FormCodec::new("torsion", identity, identity)
        ),
        Err(IrError::Conflict { .. })
    ));
    // A family without a canonical style.
    r.register_form(
        "dihedral",
        "mmff_torsion",
        FormCodec::new("nowhere", identity, identity),
    )
    .unwrap();
    assert!(matches!(
        r.canonical_form("nowhere"),
        Err(IrError::FormConflict { .. })
    ));
    let mut ff = ForceField::new("x");
    ff.def_style("dihedral", "mmff_torsion", Params::new())
        .unwrap();
    assert!(matches!(
        r.canonical(&ff),
        Err(IrError::FormConflict { family, .. }) if family == "nowhere"
    ));
    // Unregistering the style drops its codec.
    r.unregister_style("dihedral", "cos3").unwrap();
    assert!(r.form("dihedral", "cos3").is_none());
}

// ── exact conversions keep the energy ───────────────────────────────────────

/// P3: random parameters × random configurations, `E = E` to 1e-12 through
/// `canonical()` (idempotent), through the round trip back to the source
/// style, and through every conversion to another member of the family that
/// does not refuse — and a refusal is always `OutOfImage`.
#[test]
fn exact_conversions_keep_the_energy() {
    let r = Registry::builtin();
    let mut rng = StdRng::seed_from_u64(SEED);
    let (mut converted, mut refused) = (0, 0);
    for _ in 0..60 {
        for (category, style, row) in random_rows(&mut rng) {
            let ff = one_row(category, style, row);
            let what = format!("{category} {style}");
            let family = r.form(category, style).unwrap().family.to_string();
            let (cc, cs) = r.canonical_form(&family).unwrap();

            let canonical = r.canonical(&ff).unwrap_or_else(|e| panic!("{what}: {e}"));
            if category == cc {
                assert!(canonical.get_style(cc, cs).is_some(), "{what}");
                assert_eq!(canonical.get_styles(category).len(), 1, "{what}");
            }
            assert_same_energy(&mut rng, &r, category, &ff, &canonical, &what);
            let twice = r.canonical(&canonical).unwrap();
            let rows = |f: &ForceField| -> Vec<(String, Params)> {
                f.styles()
                    .iter()
                    .flat_map(|s| s.defs().collect_type_params())
                    .collect()
            };
            assert_eq!(
                rows(&twice),
                rows(&canonical),
                "{what}: canonical() is idempotent"
            );

            // Back to the source style: its own rows are in its image.
            let back = r
                .to_form(&canonical, category, style)
                .unwrap_or_else(|e| panic!("{what} round trip: {e}"));
            assert_same_energy(&mut rng, &r, category, &ff, &back, &what);

            for target in members(&r, category, &family) {
                match r.to_form(&ff, category, &target) {
                    Ok(out) => {
                        converted += 1;
                        assert!(out.get_style(category, &target).is_some());
                        assert_same_energy(
                            &mut rng,
                            &r,
                            category,
                            &ff,
                            &out,
                            &format!("{what} → {target}"),
                        );
                    }
                    Err(IrError::OutOfImage { from, type_, .. }) => {
                        refused += 1;
                        assert_eq!((from.as_str(), type_.as_str()), (what.as_str(), "t"));
                    }
                    Err(e) => panic!("{what} → {target}: {e}"),
                }
            }
        }
    }
    assert!(
        converted > 1000 && refused > 1000,
        "{converted} / {refused}"
    );
}

/// Several styles of one category become the canonical one, every row
/// priced as before on one frame; names that collide refuse, identical rows
/// merge; annotations travel.
#[test]
fn canonical_merges_the_styles_of_a_category() {
    let r = Registry::builtin();
    let mut rng = StdRng::seed_from_u64(SEED + 1);
    let mut ff = ForceField::new("mixed");
    let mut opls = pairs(&[("k1", 1.3), ("k2", -0.4), ("k3", 0.2), ("k4", 0.0)]);
    opls.set("desc", "from OPLS");
    ff.def_style("dihedral", "opls", Params::new())
        .unwrap()
        .def_type("x", &["a", "b", "c", "d"], opls)
        .unwrap();
    ff.def_style("dihedral", "charmm", Params::new())
        .unwrap()
        .def_type(
            "y",
            &["a", "b", "b", "a"],
            pairs(&[
                ("k", 0.7),
                ("periodicity", 2.0),
                ("phase", 37.0),
                ("w", 0.0),
            ]),
        )
        .unwrap();
    ff.def_style("bond", "morse", Params::new())
        .unwrap()
        .def_type(
            "a-b",
            &["a", "b"],
            pairs(&[("d0", 80.0), ("alpha", 2.0), ("r0", 1.2)]),
        )
        .unwrap();
    let out = r.canonical(&ff).unwrap();
    // A style of the category in no family stays, in its place.
    let mut with_mmff = ff.clone();
    with_mmff
        .def_style("dihedral", "mmff_torsion", Params::new())
        .unwrap();
    let mmff = r.canonical(&with_mmff).unwrap();
    let names: Vec<(&str, &str)> = mmff
        .styles()
        .iter()
        .map(|s| (s.category(), s.name()))
        .collect();
    assert_eq!(
        names,
        [
            ("dihedral", "periodic"),
            ("bond", "morse"),
            ("dihedral", "mmff_torsion")
        ],
        "the canonical style takes the first member's place; others stay"
    );
    let periodic = out.get_style("dihedral", "periodic").unwrap();
    let x = &periodic.type_rows()[0];
    assert_eq!(
        (x.0, x.2.get("desc").and_then(|v| v.as_text())),
        ("x", Some("from OPLS"))
    );
    let f = frame("dihedrals", &[(&[0, 1, 2, 3], "x"), (&[1, 2, 3, 4], "y")]);
    for _ in 0..8 {
        let c = coords(&mut rng, 5);
        assert_same(energy(&ff, &r, &f, &c), energy(&out, &r, &f, &c), "merged");
    }

    // One name in two styles: identical rows merge, different ones refuse.
    let mut twin = ForceField::new("twin");
    let row = pairs(&[("k", 0.5), ("periodicity", 3.0), ("phase", 0.0), ("w", 0.0)]);
    let ends = ["a", "b", "c", "d"];
    twin.def_style("dihedral", "charmm", Params::new())
        .unwrap()
        .def_type("x", &ends, row)
        .unwrap();
    twin.def_style("dihedral", "harmonic", Params::new())
        .unwrap()
        .def_type(
            "x",
            &ends,
            pairs(&[("k", 0.5), ("sign", 1.0), ("periodicity", 3.0)]),
        )
        .unwrap();
    let merged = r.canonical(&twin).unwrap();
    assert_eq!(
        merged
            .get_style("dihedral", "periodic")
            .unwrap()
            .type_rows()
            .len(),
        1
    );
    twin.def_style("dihedral", "opls", Params::new())
        .unwrap()
        .def_type("x", &ends, pairs(&[("k1", 1.0)]))
        .unwrap();
    let refused = r.canonical(&twin).unwrap_err();
    assert!(
        matches!(&refused, IrError::FormConflict { family, reason }
            if family == "torsion" && reason.contains("'x'")),
        "{refused}"
    );
}

/// `pair lj/class2` is `lj/cut` at n = 9, m = 6, σ' = (2/3)^⅓ σ, on every
/// row (self and cross).
#[test]
fn lj_class2_is_the_nine_six_lj_cut() {
    let r = Registry::builtin();
    let mut rng = StdRng::seed_from_u64(SEED + 2);
    for _ in 0..20 {
        let mut ff = ForceField::new("c2");
        ff.set_special_bonds(crate::ff::ir::SpecialBonds::EXCLUDE_12_13);
        let style = ff
            .def_style("pair", "lj/class2", Params::from_iter([("cutoff", 12.0)]))
            .unwrap();
        for (name, i, j) in [("A", "A", "A"), ("B", "B", "B"), ("AB", "A", "B")] {
            let row = pairs(&[
                ("epsilon", rng.random_range(0.05..0.5)),
                ("sigma", rng.random_range(3.0..4.0)),
            ]);
            if i == j {
                style.def_type(name, &[i], row).unwrap();
            } else {
                style.def_type(name, &[i, j], row).unwrap();
            }
        }
        let mut atoms = Block::new();
        atoms
            .insert(
                "type",
                Array1::from_vec(vec!["A".to_owned(), "B".to_owned(), "A".to_owned()]).into_dyn(),
                None,
            )
            .unwrap();
        let types = ["A", "B", "A"];
        let ij = [(0usize, 1usize), (0, 2), (1, 2)];
        let mut pairs_block = Block::new();
        pairs_block
            .insert(
                "atomi",
                Array1::from_vec(ij.iter().map(|p| p.0 as Idx).collect()).into_dyn(),
                None,
            )
            .unwrap();
        pairs_block
            .insert(
                "atomj",
                Array1::from_vec(ij.iter().map(|p| p.1 as Idx).collect()).into_dyn(),
                None,
            )
            .unwrap();
        pairs_block
            .insert(
                "type",
                Array1::from_vec(
                    ij.iter()
                        .map(|&(i, j)| pair_key(types[i], types[j]).unwrap())
                        .collect(),
                )
                .into_dyn(),
                None,
            )
            .unwrap();
        let mut f = Frame::new();
        f.insert("atoms", atoms);
        f.insert("pairs", pairs_block);

        let lj = r.canonical(&ff).unwrap();
        let s = lj.get_style("pair", "lj/cut").expect("lj/cut");
        assert_eq!(
            (
                s.params().get("n").and_then(|v| v.as_num()),
                s.params().get("m").and_then(|v| v.as_num())
            ),
            (Some(9.0), Some(6.0))
        );
        let back = r.to_form(&lj, "pair", "lj/class2").unwrap();
        for _ in 0..8 {
            // Atoms 3–7 apart: inside the cutoff, off the repulsive wall.
            let x = vec![
                0.0,
                0.0,
                0.0,
                rng.random_range(3.0..7.0),
                0.0,
                0.0,
                0.0,
                rng.random_range(3.0..7.0),
                0.0,
            ];
            let e = energy(&ff, &r, &f, &x);
            assert_same(e, energy(&lj, &r, &f, &x), "lj/class2 → lj/cut");
            assert_same(e, energy(&back, &r, &f, &x), "→ lj/class2");
        }
    }
    // The 12-6 has no lj/class2 form.
    let mut ff = ForceField::new("lj");
    ff.set_special_bonds(crate::ff::ir::SpecialBonds::EXCLUDE_12_13);
    ff.def_style("pair", "lj/cut", Params::from_iter([("cutoff", 10.0)]))
        .unwrap()
        .def_type("A", &["A"], pairs(&[("epsilon", 0.1), ("sigma", 3.0)]))
        .unwrap();
    let refused = r.to_form(&ff, "pair", "lj/class2").unwrap_err();
    assert!(
        matches!(&refused, IrError::OutOfImage { reason, .. } if reason.contains("n = 12")),
        "{refused}"
    );
}

// ── out of the image ────────────────────────────────────────────────────────

fn refusal(result: Result<ForceField, IrError>) -> (String, String, String, String) {
    match result {
        Err(IrError::OutOfImage {
            from,
            to,
            type_,
            reason,
        }) => (from, to, type_, reason),
        other => panic!("expected OutOfImage, got {other:?}"),
    }
}

#[test]
fn out_of_image_conversions_raise_naming_the_condition() {
    let r = Registry::builtin();
    let periodic = |k: F, n: F, phase: F| {
        one_row(
            "dihedral",
            "periodic",
            pairs(&[("k1", k), ("periodicity1", n), ("phase1", phase)]),
        )
    };
    // A Fourier series with bₙ ≠ 0 has no RB (nor OPLS) form.
    for target in ["rb", "opls", "multi/harmonic", "nharmonic"] {
        let (from, to, type_, reason) =
            refusal(r.to_form(&periodic(1.0, 2.0, 30.0), "dihedral", target));
        assert_eq!(
            (from.as_str(), to, type_.as_str()),
            ("dihedral periodic", format!("dihedral {target}"), "t")
        );
        assert!(reason.contains("sin(2φ)"), "{target}: {reason}");
    }
    // A constant the target's terms do not fix.
    let (.., reason) = refusal(r.to_form(&periodic(1.0, 2.0, 0.0), "dihedral", "opls"));
    assert!(reason.contains("constant term"), "{reason}");
    // Several orders into one term.
    let opls = one_row(
        "dihedral",
        "opls",
        pairs(&[("k1", 1.0), ("k2", 0.5), ("k3", 0.0), ("k4", 0.0)]),
    );
    let (.., reason) = refusal(r.to_form(&opls, "dihedral", "charmm"));
    assert!(reason.contains("[1, 2]"), "{reason}");
    // Rows the canonical style cannot hold.
    for (category, style, row, says) in [
        (
            "dihedral",
            "charmm",
            pairs(&[("k", 1.0), ("periodicity", 3.0), ("phase", 0.0), ("w", 0.5)]),
            "1-4 weight w = 0.5",
        ),
        (
            "bond",
            "class2",
            pairs(&[("r0", 1.5), ("k2", 300.0), ("k3", -10.0), ("k4", 0.0)]),
            "k3 = -10",
        ),
        (
            "angle",
            "charmm",
            pairs(&[("k", 50.0), ("theta0", 109.5), ("k_ub", 5.0), ("r_ub", 2.4)]),
            "Urey–Bradley",
        ),
    ] {
        let (from, _, type_, reason) = refusal(r.canonical(&one_row(category, style, row)));
        assert_eq!((from, type_), (format!("{category} {style}"), "t".into()));
        assert!(reason.contains(says), "{reason}");
    }
    // A style with no codec.
    let ff = one_row("improper", "harmonic", pairs(&[("k", 10.0), ("chi0", 0.0)]));
    assert_eq!(
        r.to_form(&ff, "improper", "harmonic").unwrap_err(),
        IrError::NoForm {
            category: "improper".into(),
            style: "harmonic".into()
        }
    );
    // `to_form` within one category: impropers stay out of `canonical()`.
    let cvff = one_row(
        "improper",
        "cvff",
        pairs(&[("k", 1.0), ("sign", -1.0), ("periodicity", 2.0)]),
    );
    let same = r.canonical(&cvff).unwrap();
    assert!(same.get_style("improper", "cvff").is_some());
    let periodic = r.to_form(&cvff, "improper", "periodic").unwrap();
    let p = &periodic
        .get_style("improper", "periodic")
        .unwrap()
        .type_rows()[0]
        .2;
    assert_eq!(
        (
            p.get("k").and_then(|v| v.as_num()),
            p.get("periodicity").and_then(|v| v.as_num()),
            p.get("phase").and_then(|v| v.as_num())
        ),
        (Some(1.0), Some(2.0), Some(180.0))
    );
}

// ── fits ────────────────────────────────────────────────────────────────────

/// A periodic row outside OPLS's image: sines at n = 2 (amplitude `t`) and
/// an order-5 term.
fn off_image(t: F) -> ForceField {
    one_row(
        "dihedral",
        "periodic",
        pairs(&[
            ("k1", 1.0),
            ("periodicity1", 1.0),
            ("phase1", 0.0),
            ("k2", t),
            ("periodicity2", 2.0),
            ("phase2", 90.0),
            ("k3", 0.3),
            ("periodicity3", 5.0),
            ("phase3", 180.0),
        ]),
    )
}

#[test]
fn a_row_in_the_image_fits_exactly() {
    let r = Registry::builtin();
    let opls = one_row(
        "dihedral",
        "opls",
        pairs(&[("k1", 1.3), ("k2", -0.4), ("k3", 0.2), ("k4", 0.1)]),
    );
    let (out, res) = r
        .fit_form(
            &opls,
            "dihedral",
            "multi/harmonic",
            &FitMetric::grid(-PI, PI, 36),
        )
        .unwrap();
    assert_eq!(res.types.len(), 1);
    assert!(res.types[0].exact && res.types[0].style == "opls" && res.types[0].type_ == "t");
    assert!(res.sum_sq() < 1e-24 && res.max_abs() < 1e-12, "{res:?}");
    let rows = |f: &ForceField| {
        f.get_style("dihedral", "multi/harmonic")
            .unwrap()
            .defs()
            .collect_type_params()
    };
    assert_eq!(
        rows(&out),
        rows(&r.to_form(&opls, "dihedral", "multi/harmonic").unwrap())
    );
}

/// A `dihedral charmm` row with `w ≠ 0` prices its 1-4 pair too, which no
/// torsion of the family holds: fitting it over E(φ) would silently drop that
/// term, so the fit refuses the row instead.
#[test]
fn a_row_its_family_cannot_embed_is_refused_not_fitted() {
    let r = Registry::builtin();
    let charmm = one_row(
        "dihedral",
        "charmm",
        pairs(&[
            ("k", 0.7),
            ("periodicity", 2.0),
            ("phase", 37.0),
            ("w", 0.5),
        ]),
    );
    let err = r
        .fit_form(
            &charmm,
            "dihedral",
            "periodic",
            &FitMetric::grid(-PI, PI, 36),
        )
        .unwrap_err();
    assert!(
        matches!(&err, IrError::OutOfImage { reason, .. } if reason.contains("outside its family")),
        "{err}"
    );
}

/// The fit's residual is reported and monotone in the metric: a superset of
/// points (the grid of 2n holds the grid of n) or pointwise larger weights
/// never lower the minimised `Σ w r²`; and it grows with the distance from
/// the image.
#[test]
fn the_residual_is_monotone_in_the_metric() {
    let r = Registry::builtin();
    let ff = off_image(0.5);
    let fit = |m: &FitMetric| r.fit_form(&ff, "dihedral", "opls", m).unwrap().1;

    let mut previous = 0.0;
    for n in [9, 18, 36, 72, 144] {
        let res = fit(&FitMetric::grid(-PI, PI, n));
        assert!(!res.types[0].exact);
        let s = res.sum_sq();
        assert!(
            s > 0.0 && s >= previous * (1.0 - 1e-9),
            "n = {n}: {s} < {previous}"
        );
        previous = s;
        // With a free offset, on a uniform full-period grid the fit is the
        // L2 projection: the cosines up to n = 4 (orthogonality).
        let (out, _) = r
            .fit_form(
                &ff,
                "dihedral",
                "opls",
                &FitMetric::grid(-PI, PI, n).free_offset(),
            )
            .unwrap();
        if n >= 18 {
            let row = &out.get_style("dihedral", "opls").unwrap().type_rows()[0].2;
            let series = PeriodicForm::from_params(
                &(r.form("dihedral", "periodic").unwrap().embed)(&TypeParams::row(
                    ff.get_style("dihedral", "periodic").unwrap().type_rows()[0]
                        .2
                        .clone(),
                ))
                .unwrap()
                .row,
            )
            .unwrap()
            .to_series()
            .unwrap();
            let want = OplsForm::nearest(&series);
            for (i, k) in want.k.iter().enumerate() {
                let got = row
                    .get(&format!("k{}", i + 1))
                    .and_then(|v| v.as_num())
                    .unwrap();
                assert!((got - k).abs() < 1e-7, "n = {n}: k{} = {got} vs {k}", i + 1);
            }
        }
    }

    let mut rng = StdRng::seed_from_u64(SEED + 3);
    let base = FitMetric::grid(-PI, PI, 48);
    let low = fit(&base).sum_sq();
    for _ in 0..5 {
        let heavier: Vec<F> = base
            .w
            .iter()
            .map(|w| w * (1.0 + rng.random_range(0.0..2.0)))
            .collect();
        let high = fit(&base.clone().weights(heavier)).sum_sq();
        assert!(high >= low * (1.0 - 1e-9), "{high} < {low}");
    }

    let mut previous = 0.0;
    for t in [0.1, 0.2, 0.4, 0.8, 1.6] {
        let rms = r
            .fit_form(&off_image(t), "dihedral", "opls", &base)
            .unwrap()
            .1
            .rms();
        assert!(rms > previous, "t = {t}: {rms} ≤ {previous}");
        previous = rms;
    }
}

/// A constant is no part of the forces: with a free offset the fit is exact
/// where the projection refused, and reports the offset.
#[test]
fn a_free_offset_absorbs_the_constant() {
    let r = Registry::builtin();
    let ff = one_row(
        "dihedral",
        "periodic",
        pairs(&[("k1", 1.0), ("periodicity1", 2.0), ("phase1", 0.0)]),
    );
    assert!(r.to_form(&ff, "dihedral", "opls").is_err());
    let grid = FitMetric::grid(-PI, PI, 24);
    let fixed = r.fit_form(&ff, "dihedral", "opls", &grid).unwrap().1;
    assert!(fixed.rms() > 0.2, "{fixed:?}");
    let (out, free) = r
        .fit_form(&ff, "dihedral", "opls", &grid.free_offset())
        .unwrap();
    // 1 + cos 2φ = −1·½[1 − cos 2φ]·2 + 2: k2 = −2, offset 2.
    assert!(free.rms() < 1e-10, "{free:?}");
    assert!((free.types[0].offset - 2.0).abs() < 1e-10, "{free:?}");
    let row = &out.get_style("dihedral", "opls").unwrap().type_rows()[0].2;
    assert!(
        (row.get("k2").and_then(|v| v.as_num()).unwrap() + 2.0).abs() < 1e-10,
        "{row:?}"
    );
}

/// No codec is needed: a harmonic improper fitted to a periodic one under a
/// Boltzmann metric about the minimum lands on the second-order relation
/// `K = n²k/2`, and a metric outside the coordinate's domain is refused.
#[test]
fn a_style_without_a_codec_fits_through_its_energy() {
    let r = Registry::builtin();
    let ff = one_row(
        "improper",
        "periodic",
        pairs(&[("k", 2.0), ("periodicity", 2.0), ("phase", 180.0)]),
    );
    let metric = FitMetric::grid(-0.5, 0.5, 101)
        .boltzmann(0.02)
        .free_offset();
    let (out, res) = r.fit_form(&ff, "improper", "harmonic", &metric).unwrap();
    let row = &out.get_style("improper", "harmonic").unwrap().type_rows()[0].2;
    let (k, chi0) = (
        row.get("k").and_then(|v| v.as_num()).unwrap(),
        row.get("chi0").and_then(|v| v.as_num()).unwrap(),
    );
    let second_order = CosineTermForm {
        k: 2.0,
        periodicity: 2.0,
        phase: 180.0,
    }
    .second_order_harmonic()
    .unwrap();
    assert!(
        (k - second_order.k).abs() < 0.05 * second_order.k,
        "K = {k}"
    );
    assert!(chi0.abs() < 1.0, "chi0 = {chi0}");
    assert!(!res.types[0].exact && res.types[0].rms().is_finite());

    let bad = r.fit_form(&ff, "improper", "harmonic", &FitMetric::new(vec![]));
    assert!(matches!(bad, Err(IrError::Malformed { .. })), "{bad:?}");
    let bond = one_row(
        "bond",
        "morse",
        pairs(&[("d0", 80.0), ("alpha", 2.0), ("r0", 1.2)]),
    );
    let bad = r.fit_form(&bond, "bond", "harmonic", &FitMetric::new(vec![-1.0, 1.0]));
    assert!(matches!(bad, Err(IrError::Malformed { .. })), "{bad:?}");
    let (_, res) = r
        .fit_form(&bond, "bond", "harmonic", &FitMetric::grid(1.0, 1.4, 41))
        .unwrap();
    assert!(
        res.rms() < 2.0,
        "a morse well near r0 is nearly harmonic: {res:?}"
    );
    let lj = ForceField::new("x");
    assert!(matches!(
        r.fit_form(&lj, "pair", "lj/cut", &FitMetric::grid(3.0, 6.0, 10)),
        Err(IrError::OutOfImage { .. })
    ));
}

// ── a third party's style joins a family ────────────────────────────────────

/// `k[1 + cos 3φ]`, priced by its expression alone, with a codec in the
/// `torsion` family: exact maps to and from the canonical `dihedral
/// periodic` row, refusing anything but one order-3 cosine whose constant is
/// its own.
fn cos3(r: &mut Registry) {
    r.register_style(
        StyleSpec::new("dihedral", "cos3")
            .params(vec![ParamSpec::new("k", ParamDimension::ENERGY)])
            .expression("k*(1+cos(3*phi))"),
        None,
    )
    .unwrap();
    let embed = |tp: &TypeParams| -> Result<TypeParams, FormRefusal> {
        let k = tp
            .row
            .get("k")
            .and_then(|v| v.as_num())
            .ok_or_else(|| FormRefusal::new("missing `k`"))?;
        Ok(TypeParams::row(Params::from_iter([
            ("k1", k),
            ("periodicity1", 3.0),
            ("phase1", 0.0),
        ])))
    };
    let project = |tp: &TypeParams| -> Result<TypeParams, FormRefusal> {
        let s = PeriodicForm::from_params(&tp.row)?.to_series()?;
        let t = CosineTermForm::from_series(&s)?;
        if t.k != 0.0 && (t.periodicity, t.phase) != (3.0, 0.0) {
            return Err(FormRefusal::new(format!(
                "periodicity {} phase {}: cos3 is k[1 + cos 3φ]",
                t.periodicity, t.phase
            )));
        }
        Ok(TypeParams::row(Params::from_iter([("k", t.k)])))
    };
    r.register_form(
        "dihedral",
        "cos3",
        FormCodec::new("torsion", embed, project),
    )
    .unwrap();
}

#[test]
fn a_registered_expression_style_takes_part_in_its_family() {
    let mut r = Registry::builtin();
    cos3(&mut r);
    let mut rng = StdRng::seed_from_u64(SEED + 4);
    for _ in 0..20 {
        let k = c(&mut rng);
        let ff = one_row("dihedral", "cos3", pairs(&[("k", k)]));
        let canonical = r.canonical(&ff).unwrap();
        assert!(canonical.get_style("dihedral", "periodic").is_some());
        assert_same_energy(&mut rng, &r, "dihedral", &ff, &canonical, "cos3 → periodic");

        // OPLS ½k₃[1 + cos 3φ] is cos3 with k = k₃/2, exactly.
        let opls = one_row(
            "dihedral",
            "opls",
            pairs(&[("k1", 0.0), ("k2", 0.0), ("k3", 2.0 * k), ("k4", 0.0)]),
        );
        let out = r.to_form(&opls, "dihedral", "cos3").unwrap();
        let row = &out.get_style("dihedral", "cos3").unwrap().type_rows()[0].2;
        assert_same(row.get("k").and_then(|v| v.as_num()).unwrap(), k, "k");
        assert_same_energy(&mut rng, &r, "dihedral", &opls, &out, "opls → cos3");
        let back = r.to_form(&out, "dihedral", "multi/harmonic").unwrap();
        assert_same_energy(&mut rng, &r, "dihedral", &opls, &back, "cos3 → multi");
    }
    // Out of its image: refused with the codec's own reason.
    let opls = one_row(
        "dihedral",
        "opls",
        pairs(&[("k1", 1.0), ("k2", 0.0), ("k3", 0.0), ("k4", 0.0)]),
    );
    let (from, to, _, reason) = refusal(r.to_form(&opls, "dihedral", "cos3"));
    assert_eq!(
        (from.as_str(), to.as_str()),
        ("dihedral opls", "dihedral cos3")
    );
    assert!(reason.contains("cos3 is k[1 + cos 3φ]"), "{reason}");
    // And it is a fit target, through its expression.
    let mixed = one_row(
        "dihedral",
        "opls",
        pairs(&[("k1", 0.3), ("k2", 0.0), ("k3", 2.0), ("k4", 0.0)]),
    );
    let (out, res) = r
        .fit_form(
            &mixed,
            "dihedral",
            "cos3",
            &FitMetric::grid(-PI, PI, 36).free_offset(),
        )
        .unwrap();
    let k = out.get_style("dihedral", "cos3").unwrap().type_rows()[0]
        .2
        .get("k")
        .and_then(|v| v.as_num())
        .unwrap();
    assert!((k - 1.0).abs() < 1e-8, "the cos 3φ coefficient: k = {k}");
    assert!(!res.types[0].exact && res.rms() > 0.05, "{res:?}");
}
