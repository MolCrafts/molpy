//! Every form kernel against a built-in re-expressed as a form.
//!
//! Each built-in below is written as the [`ScalarForm`] / [`CompoundForm`] a
//! third party would register, compiled through the same
//! [`PotentialCompiler`] door as the built-in, and compared on one molecule:
//! energies and forces to 1e-12 relative, at both compile doors, through
//! rebound index tables, and — for the pair — over a neighbour table with
//! per-pair weights, a cutoff and the virial.

use std::sync::Arc;

use ndarray::Array1;

use crate::ff::compile::PotentialCompiler;
use crate::ff::forcefield::ForceField;
use crate::ff::ir::{
    ParamCombination, ParamDimension, ParamSource, ParamSpec, SpecialClass, StyleSpec,
};
use crate::ff::ir::{Params, SpecialBonds};
use crate::ff::potential::fixtures::OneShot;
use crate::ff::potential::flat_coords::{accumulate_angle_forces, compute_angle};
use crate::ff::potential::form_kernel::{CompoundForm, ParamColumns, ScalarForm, ScalarPair};
use crate::ff::potential::pair::fixtures::{assert_virial_matches_forces, table_over};
use crate::ff::potential::{Accumulator, ForceTerm, PairDriven, Potentials};
use crate::ff::style_registry::{Kernel, Registry};
use molrs::core::Block;
use molrs::core::Frame;
use molrs::op::{F, Idx};

// ---------------------------------------------------------------------------
// The forms
// ---------------------------------------------------------------------------

/// `k (q − q0·unit)²`: LAMMPS `bond harmonic` (`q0` = `r0`, unit 1) and
/// `angle harmonic` (`q0` = `theta0` in degrees, unit π/180).
struct Harmonic {
    q0: &'static str,
    unit: F,
}

impl ScalarForm for Harmonic {
    fn eval(&self, q: &[F], p: &ParamColumns<'_>, e: &mut [F], de: &mut [F]) {
        let (k, q0) = (p.get("k").unwrap(), p.get(self.q0).unwrap());
        for t in 0..q.len() {
            let d = q[t] - q0[t] * self.unit;
            e[t] = k[t] * d * d;
            de[t] = 2.0 * k[t] * d;
        }
    }
}

/// LAMMPS `dihedral fourier` with two terms written out:
/// `Σₘ kₘ [1 + cos(nₘ φ − γₘ)]`, γ in degrees.
struct Periodic2;

impl ScalarForm for Periodic2 {
    fn eval(&self, phi: &[F], p: &ParamColumns<'_>, e: &mut [F], de: &mut [F]) {
        let col = |name: String| p.get(&name).unwrap();
        for t in 0..phi.len() {
            e[t] = 0.0;
            de[t] = 0.0;
            for m in 1..=2 {
                let (k, n, g) = (
                    col(format!("k{m}"))[t],
                    col(format!("periodicity{m}"))[t],
                    col(format!("phase{m}"))[t],
                );
                let arg = n * phi[t] - g.to_radians();
                e[t] += k * (1.0 + arg.cos());
                de[t] -= k * n * arg.sin();
            }
        }
    }
}

/// LAMMPS `pair lj/cut` 12-6: `4ε[(σ/r)¹² − (σ/r)⁶]`.
struct Lj;

impl ScalarForm for Lj {
    fn eval(&self, r: &[F], p: &ParamColumns<'_>, e: &mut [F], de: &mut [F]) {
        let (eps, sig) = (p.get("epsilon").unwrap(), p.get("sigma").unwrap());
        for t in 0..r.len() {
            let sr6 = (sig[t] / r[t]).powi(6);
            e[t] = 4.0 * eps[t] * (sr6 * sr6 - sr6);
            de[t] = -24.0 * eps[t] * (2.0 * sr6 * sr6 - sr6) / r[t];
        }
    }
}

/// `coulomb · q1 q2 / r`: a pair form reading the atoms' charges and a
/// numeric style parameter.
struct Coulomb;

impl ScalarForm for Coulomb {
    fn eval(&self, r: &[F], p: &ParamColumns<'_>, e: &mut [F], de: &mut [F]) {
        let (c, q1, q2) = (
            p.get("coulomb").unwrap(),
            p.get("q1").unwrap(),
            p.get("q2").unwrap(),
        );
        for t in 0..r.len() {
            e[t] = c[t] * q1[t] * q2[t] / r[t];
            de[t] = -e[t] / r[t];
        }
    }
}

/// LAMMPS `angle charmm` as an N-body form: `k (θ − θ0)² + k_ub (r₁₃ − r_ub)²`.
struct CharmmAngle;

impl CompoundForm for CharmmAngle {
    fn eval(
        &self,
        x: &[[F; 3]],
        arity: usize,
        p: &ParamColumns<'_>,
        e: &mut [F],
        grad: &mut [[F; 3]],
    ) {
        assert_eq!(arity, 3);
        let get = |name: &str| p.get(name).unwrap();
        let (k, theta0, k_ub, r_ub) = (get("k"), get("theta0"), get("k_ub"), get("r_ub"));
        for t in 0..e.len() {
            let pts = &x[t * 3..t * 3 + 3];
            let flat: Vec<F> = pts.iter().flatten().copied().collect();
            let theta = compute_angle(&flat, 0, 1, 2);
            let dth = theta - theta0[t].to_radians();
            let mut forces = vec![0.0; 9];
            accumulate_angle_forces(&flat, 0, 1, 2, 2.0 * k[t] * dth, &mut forces);
            let d: Vec<F> = (0..3).map(|c| pts[2][c] - pts[0][c]).collect();
            let r13 = (d[0] * d[0] + d[1] * d[1] + d[2] * d[2]).sqrt();
            let dr = r13 - r_ub[t];
            e[t] = k[t] * dth * dth + k_ub[t] * dr * dr;
            for c in 0..3 {
                let g_ub = 2.0 * k_ub[t] * dr * d[c] / r13;
                // `accumulate_angle_forces` adds forces, the form writes a
                // gradient.
                grad[t * 3][c] = -forces[c] - g_ub;
                grad[t * 3 + 1][c] = -forces[3 + c];
                grad[t * 3 + 2][c] = -forces[6 + c] + g_ub;
            }
        }
    }
}

// ---------------------------------------------------------------------------
// The molecule
// ---------------------------------------------------------------------------

fn idx(v: &[usize]) -> ndarray::ArrayD<Idx> {
    Array1::from_vec(v.iter().map(|&a| a as Idx).collect()).into_dyn()
}

fn strings(v: &[&str]) -> ndarray::ArrayD<String> {
    Array1::from_vec(v.iter().map(|s| s.to_string()).collect()).into_dyn()
}

fn relation(cols: &[&[usize]], types: &[&str]) -> Block {
    let mut b = Block::new();
    for (key, col) in ["atomi", "atomj", "atomk", "atoml"].iter().zip(cols) {
        b.insert(*key, idx(col), None).unwrap();
    }
    b.insert("type", strings(types), None).unwrap();
    b
}

const COORDS: [F; 15] = [
    0.0, 0.0, 0.0, //
    1.52, 0.1, 0.05, //
    2.1, 1.45, -0.1, //
    3.55, 1.6, 0.4, //
    4.1, 2.9, 0.9,
];

/// Five atoms, non-planar, carrying a term of every bonded arity and a
/// `pairs` list with two 1-4 rows.
fn molecule() -> Frame {
    let mut atoms = Block::new();
    for (a, key) in ["x", "y", "z"].iter().enumerate() {
        let v: Vec<F> = (0..5).map(|i| COORDS[i * 3 + a]).collect();
        atoms
            .insert(*key, Array1::from_vec(v).into_dyn(), None)
            .unwrap();
    }
    atoms
        .insert("type", strings(&["C", "C", "O", "C", "H"]), None)
        .unwrap();
    atoms
        .insert(
            "charge",
            Array1::from_vec(vec![0.3, -0.2, -0.5, 0.25, 0.15]).into_dyn(),
            None,
        )
        .unwrap();
    let mut frame = Frame::new();
    frame.insert("atoms", atoms);
    frame.insert(
        "bonds",
        relation(
            &[&[0, 1, 2, 3], &[1, 2, 3, 4]],
            &["C-C", "C-O", "C-O", "C-H"],
        ),
    );
    frame.insert(
        "angles",
        relation(&[&[0, 1, 2], &[1, 2, 3], &[2, 3, 4]], &["a1", "a2", "a3"]),
    );
    frame.insert(
        "dihedrals",
        relation(&[&[0, 1], &[1, 2], &[2, 3], &[3, 4]], &["d1", "d2"]),
    );
    let pairs = molrs::core::Topology::from_frame(&frame)
        .map(|t| t.pairs_block())
        .unwrap();
    frame.insert("pairs", pairs);
    frame
}

fn p(name: &'static str, dim: &str) -> ParamSpec {
    ParamSpec::new(name, dim.parse::<ParamDimension>().unwrap())
}

/// A registry holding `spec` with `kernel`, beside the built-ins.
fn registry_with(spec: StyleSpec, kernel: Kernel) -> Registry {
    let mut r = Registry::builtin();
    r.register_style(spec, Some(kernel)).unwrap();
    r
}

/// A type: its name, endpoints and parameters.
type TypeRow<'a> = (&'a str, &'a [&'a str], &'a [(&'a str, F)]);

/// `(category, style)` of `ff` defined twice: once under the built-in name,
/// once under the generic one, with the same rows.
fn twin(
    category: &str,
    builtin: &str,
    generic: &str,
    style: Params,
    rows: &[TypeRow<'_>],
) -> (ForceField, ForceField) {
    let make = |name: &str| {
        let mut ff = ForceField::new("t");
        ff.set_special_bonds(SpecialBonds {
            lj: [0.0, 0.0, 0.5],
            coul: [0.0, 0.0, 0.8333],
        });
        let s = ff.def_style(category, name, style.clone()).unwrap();
        for (label, ends, params) in rows {
            s.def_type(label, ends, Params::from_iter(params.iter().copied()))
                .unwrap();
        }
        ff
    };
    (make(builtin), make(generic))
}

/// Energies and forces equal to `rtol`, relative to the largest magnitude.
fn assert_close(label: &str, a: (F, Vec<F>), b: (F, Vec<F>), rtol: F) {
    let scale = a.1.iter().fold(a.0.abs(), |m, f| m.max(f.abs()));
    assert!(scale > 1e-6, "{label}: nothing to compare ({scale})");
    assert!(
        (a.0 - b.0).abs() <= rtol * scale,
        "{label}: energy {} vs {}",
        a.0,
        b.0
    );
    assert_eq!(a.1.len(), b.1.len());
    for (c, (x, y)) in a.1.iter().zip(&b.1).enumerate() {
        assert!(
            (x - y).abs() <= rtol * scale,
            "{label}: force {c}: {x} vs {y}"
        );
    }
}

/// The one member a compile produced, and its rows under a rebound table
/// (the same table here: rebinding must not change the answer).
fn one_member(pots: Potentials) -> ForceTerm {
    let mut members = pots.into_members();
    assert_eq!(members.len(), 1);
    members.pop().unwrap()
}

/// Compile `builtin` against the global registry and `generic` against
/// `reg`, and compare: plain, through rebound terms, and at the typed door.
fn compare_bonded(label: &str, builtin: &ForceField, generic: &ForceField, reg: &Registry) {
    let frame = molecule();
    let a = one_member(PotentialCompiler::new(builtin).compile(&frame).unwrap());
    let b = one_member(
        PotentialCompiler::new(
            &generic
                .clone()
                .with_registry(std::sync::Arc::new(reg.clone())),
        )
        .compile(&frame)
        .unwrap(),
    );
    assert_close(
        label,
        a.energy_forces_at(&COORDS),
        b.energy_forces_at(&COORDS),
        1e-12,
    );
    let (ForceTerm::Indexed(ia), ForceTerm::Indexed(ib)) = (&a, &b) else {
        panic!("{label}: a bonded member is indexed");
    };
    assert_eq!(ia.terms(), ib.terms(), "{label}: the same rows");
    let table = ia.terms();
    assert_close(
        &format!("{label} (rebound)"),
        ia.over_terms(&COORDS, table.view()),
        ib.over_terms(&COORDS, table.view()),
        1e-12,
    );
    let typed = PotentialCompiler::new(
        &generic
            .clone()
            .with_registry(std::sync::Arc::new(reg.clone())),
    )
    .compile_typed(&frame)
    .unwrap();
    assert_eq!(typed.len(), 1);
    assert!(
        typed[0].1.is_empty(),
        "{label}: a bonded member takes no weights"
    );
    assert_close(
        &format!("{label} (typed door)"),
        a.energy_forces_at(&COORDS),
        typed[0].0.energy_forces_at(&COORDS),
        1e-12,
    );
}

#[test]
fn scalar_bonded_equals_bond_harmonic() {
    let spec = StyleSpec::new("bond", "harmonic/form").params(vec![p("k", "E/L^2"), p("r0", "L")]);
    let form = Harmonic {
        q0: "r0",
        unit: 1.0,
    };
    let reg = registry_with(spec, Kernel::Scalar(Arc::new(form)));
    let (builtin, generic) = twin(
        "bond",
        "harmonic",
        "harmonic/form",
        Params::new(),
        &[
            ("C-C", &["C", "C"], &[("k", 310.0), ("r0", 1.526)]),
            ("C-O", &["C", "O"], &[("k", 320.0), ("r0", 1.41)]),
            ("C-H", &["C", "H"], &[("k", 340.0), ("r0", 1.09)]),
        ],
    );
    compare_bonded("bond harmonic", &builtin, &generic, &reg);
}

#[test]
fn scalar_bonded_equals_angle_harmonic() {
    let spec =
        StyleSpec::new("angle", "harmonic/form").params(vec![p("k", "E/A^2"), p("theta0", "A")]);
    let form = Harmonic {
        q0: "theta0",
        // one degree in radians
        unit: 1.0_f64.to_radians(),
    };
    let reg = registry_with(spec, Kernel::Scalar(Arc::new(form)));
    let (builtin, generic) = twin(
        "angle",
        "harmonic",
        "harmonic/form",
        Params::new(),
        &[
            ("a1", &["C", "C", "O"], &[("k", 50.0), ("theta0", 109.5)]),
            ("a2", &["C", "O", "C"], &[("k", 60.0), ("theta0", 112.0)]),
            ("a3", &["O", "C", "H"], &[("k", 35.0), ("theta0", 107.8)]),
        ],
    );
    compare_bonded("angle harmonic", &builtin, &generic, &reg);
}

#[test]
fn scalar_bonded_equals_dihedral_periodic() {
    let spec = StyleSpec::new("dihedral", "periodic/form").params(vec![
        p("k1", "E"),
        p("periodicity1", "1"),
        p("phase1", "A"),
        p("k2", "E"),
        p("periodicity2", "1"),
        p("phase2", "A"),
    ]);
    let reg = registry_with(spec, Kernel::Scalar(Arc::new(Periodic2)));
    let terms: &[(&str, F)] = &[
        ("k1", 1.4),
        ("periodicity1", 1.0),
        ("phase1", 0.0),
        ("k2", 0.25),
        ("periodicity2", 3.0),
        ("phase2", 180.0),
    ];
    let terms2: &[(&str, F)] = &[
        ("k1", 0.8),
        ("periodicity1", 2.0),
        ("phase1", 180.0),
        ("k2", 0.15),
        ("periodicity2", 3.0),
        ("phase2", 25.0),
    ];
    let (builtin, generic) = twin(
        "dihedral",
        "periodic",
        "periodic/form",
        Params::new(),
        &[
            ("d1", &["C", "C", "O", "C"], terms),
            ("d2", &["C", "O", "C", "H"], terms2),
        ],
    );
    compare_bonded("dihedral periodic", &builtin, &generic, &reg);
}

#[test]
fn compound_terms_equal_angle_charmm() {
    let spec = StyleSpec::new("angle", "charmm/form").params(vec![
        p("k", "E/A^2"),
        p("theta0", "A"),
        p("k_ub", "E/L^2"),
        p("r_ub", "L"),
    ]);
    let reg = registry_with(spec, Kernel::Compound(Arc::new(CharmmAngle)));
    let row =
        |k: F, t0: F, kub: F, rub: F| [("k", k), ("theta0", t0), ("k_ub", kub), ("r_ub", rub)];
    let (r1, r2, r3) = (
        row(50.0, 109.5, 20.0, 2.45),
        row(60.0, 112.0, 0.0, 0.0),
        row(35.0, 107.8, 11.0, 2.2),
    );
    let (builtin, generic) = twin(
        "angle",
        "charmm",
        "charmm/form",
        Params::new(),
        &[
            ("a1", &["C", "C", "O"], &r1),
            ("a2", &["C", "O", "C"], &r2),
            ("a3", &["O", "C", "H"], &r3),
        ],
    );
    compare_bonded("angle charmm", &builtin, &generic, &reg);
}

/// `pair lj/cut` as a [`ScalarForm`]: mixed rows, a cross row, the 1-4
/// weight at the compiled door; a neighbour table with weights, a cutoff
/// and the virial at the typed door.
#[test]
fn scalar_pair_equals_lj_cut_at_both_doors() {
    let spec = StyleSpec::new("pair", "lj/form")
        .params(vec![
            p("epsilon", "E").mix(ParamCombination::LjEpsilon {
                sigma: "sigma".into(),
            }),
            p("sigma", "L").mix(ParamCombination::LjSigma {
                epsilon: "epsilon".into(),
            }),
        ])
        .style_params(vec![p("cutoff", "L")])
        .special(SpecialClass::Vdw);
    let reg = registry_with(spec, Kernel::Scalar(Arc::new(Lj)));
    let mut style = Params::from_iter([("cutoff", 4.0)]);
    style.set("mixing", "geometric");
    let (builtin, generic) = twin(
        "pair",
        "lj/cut",
        "lj/form",
        style,
        &[
            ("C", &["C"], &[("epsilon", 0.11), ("sigma", 3.4)]),
            ("O", &["O"], &[("epsilon", 0.21), ("sigma", 2.96)]),
            ("H", &["H"], &[("epsilon", 0.015), ("sigma", 2.5)]),
            ("C-H", &["C", "H"], &[("epsilon", 0.05), ("sigma", 2.9)]),
        ],
    );
    let frame = molecule();

    // Compiled: every pair, the special pairs at the field's weights.
    let a = PotentialCompiler::new(&builtin).compile(&frame).unwrap();
    let b = PotentialCompiler::new(
        &generic
            .clone()
            .with_registry(std::sync::Arc::new(reg.clone())),
    )
    .compile(&frame)
    .unwrap();
    assert_close(
        "lj compiled",
        a.energy_forces_at(&COORDS),
        b.energy_forces_at(&COORDS),
        1e-12,
    );
    assert!(b.members()[0].binds_a_fixed_pair_list());

    // Typed: every pair of the molecule, weighted, through the cutoff.
    let ta = PotentialCompiler::new(&builtin)
        .compile_typed(&frame)
        .unwrap();
    let tb = PotentialCompiler::new(
        &generic
            .clone()
            .with_registry(std::sync::Arc::new(reg.clone())),
    )
    .compile_typed(&frame)
    .unwrap();
    assert_eq!(ta[0].1, tb[0].1, "the same special-bonds weights");
    let (ForceTerm::Pair(pa), ForceTerm::Pair(pb)) = (&ta[0].0, &tb[0].0) else {
        panic!("a pair member");
    };
    assert!(!pb.binds_a_fixed_pair_list());
    let links: Vec<(usize, usize)> = (0..5)
        .flat_map(|i| ((i + 1)..5).map(move |j| (i, j)))
        .collect();
    let table = table_over(&COORDS, &links);
    // Bonded pairs off, a 1-4 at ½, the rest whole; the 0–4 pair (5.3 Å)
    // lies beyond the 4 Å cutoff.
    let factor: Vec<F> = links
        .iter()
        .map(|&(i, j)| match j - i {
            1 | 2 => 0.0,
            3 => 0.5,
            _ => 1.0,
        })
        .collect();
    let weighted = |p: &dyn PairDriven| {
        let mut acc = Accumulator::new(15).with_virial();
        p.evaluate_pairs(&COORDS, &table, &factor, &mut acc);
        let (e, w) = (acc.energy(), acc.virial());
        (e, acc.into_forces(), w)
    };
    let (ea, fa, wa) = weighted(&**pa);
    let (eb, fb, wb) = weighted(&**pb);
    assert_close("lj typed", (ea, fa), (eb, fb), 1e-12);
    let (wa, wb) = (wa.unwrap(), wb.unwrap());
    for c in 0..6 {
        assert!(
            (wa.components[c] - wb.components[c]).abs()
                <= 1e-12 * wa.components.iter().fold(0.0_f64, |m, v| m.max(v.abs())),
            "virial {c}: {} vs {}",
            wa.components[c],
            wb.components[c]
        );
    }
    assert_virial_matches_forces("lj typed", &COORDS, pb.over_pairs_virial(&COORDS, &table));
}

/// A pair form reads the atoms' charges as `q1`, `q2`, and prices exactly
/// `coul/cut` over a neighbour table.
#[test]
fn scalar_pair_binds_charges() {
    let spec = StyleSpec::new("pair", "coul/form")
        .style_params(vec![p("coulomb", "E*L/Q^2"), p("cutoff", "L")])
        .source(ParamSource::PerInstance)
        .special(SpecialClass::Coulomb);
    let reg = registry_with(spec, Kernel::Scalar(Arc::new(Coulomb)));
    let style = Params::from_iter([("cutoff", 10.0), ("coulomb", 332.0716), ("dielectric", 1.0)]);
    let (builtin, generic) = twin("pair", "coul/cut", "coul/form", style, &[]);
    let frame = molecule();
    let ta = PotentialCompiler::new(&builtin)
        .compile_typed(&frame)
        .unwrap();
    let tb = PotentialCompiler::new(
        &generic
            .clone()
            .with_registry(std::sync::Arc::new(reg.clone())),
    )
    .compile_typed(&frame)
    .unwrap();
    assert_eq!(ta[0].1, tb[0].1);
    let coords = COORDS.to_vec();
    let (ForceTerm::Pair(pa), ForceTerm::Pair(pb)) = (&ta[0].0, &tb[0].0) else {
        panic!("a pair member");
    };
    let table = table_over(&coords, &[(0, 3), (1, 4), (2, 4), (0, 2)]);
    assert_close(
        "coulomb typed",
        pa.over_pairs(&coords, &table),
        pb.over_pairs(&coords, &table),
        1e-12,
    );
}

/// Every built-in pair style but PME (a scalar form) prices a fixed pair
/// list and a neighbour table over the same pairs alike, **bit for bit**,
/// and the neighbour door's virial is `Σ f ⊗ x` on a free boundary.
///
/// This is the whole claim of the two doors: nothing about the physics
/// differs, only *when* a parameter is found — against a list at
/// construction, or from the atoms when a pair turns up. Two types, unlike
/// each other, so a table indexed the wrong way round would answer
/// differently; every pair of five atoms, a cutoff where the style declares
/// one, and a weight of one.
#[test]
fn every_pair_form_prices_a_pair_list_and_a_table_alike() {
    use crate::ff::potential::pair::fixtures::assert_same;
    let row = |v: &[(&'static str, F)]| Params::from_iter(v.iter().copied());
    type Rows = Vec<(&'static str, Vec<&'static str>, Params)>;
    let cases: Vec<(&str, Params, Rows)> = vec![
        (
            "lj/cut",
            row(&[("cutoff", 4.0), ("shift", 1.0)]),
            vec![
                ("C", vec!["C"], row(&[("epsilon", 0.11), ("sigma", 3.4)])),
                ("O", vec!["O"], row(&[("epsilon", 0.21), ("sigma", 2.96)])),
            ],
        ),
        (
            "lj/class2",
            row(&[("cutoff", 4.0)]),
            vec![
                ("C", vec!["C"], row(&[("epsilon", 0.11), ("sigma", 3.4)])),
                ("O", vec!["O"], row(&[("epsilon", 0.21), ("sigma", 2.96)])),
            ],
        ),
        (
            "buck",
            row(&[("cutoff", 4.0)]),
            vec![
                (
                    "C",
                    vec!["C"],
                    row(&[("a", 12000.0), ("rho", 0.31), ("c", 280.0)]),
                ),
                (
                    "O",
                    vec!["O"],
                    row(&[("a", 7000.0), ("rho", 0.27), ("c", 140.0)]),
                ),
                (
                    "C-O",
                    vec!["C", "O"],
                    row(&[("a", 9000.0), ("rho", 0.29), ("c", 190.0)]),
                ),
            ],
        ),
        (
            "morse",
            row(&[("cutoff", 4.0)]),
            vec![
                (
                    "C",
                    vec!["C"],
                    row(&[("d0", 0.25), ("alpha", 1.9), ("r0", 3.4)]),
                ),
                (
                    "O",
                    vec!["O"],
                    row(&[("d0", 0.31), ("alpha", 1.7), ("r0", 2.9)]),
                ),
                (
                    "C-O",
                    vec!["C", "O"],
                    row(&[("d0", 0.18), ("alpha", 2.1), ("r0", 3.1)]),
                ),
            ],
        ),
        (
            "lj/charmm",
            row(&[("inner", 2.5), ("cutoff", 4.0)]),
            vec![
                ("C", vec!["C"], row(&[("epsilon", 0.11), ("sigma", 3.4)])),
                ("O", vec!["O"], row(&[("epsilon", 0.21), ("sigma", 2.96)])),
            ],
        ),
        (
            "coul/cut",
            row(&[
                ("coulomb", 332.0716),
                ("dielectric", 1.0),
                ("delta", 0.05),
                ("cutoff", 4.0),
            ]),
            vec![],
        ),
        (
            "coul/charmm",
            row(&[
                ("coulomb", 332.0716),
                ("dielectric", 1.0),
                ("inner", 2.5),
                ("cutoff", 4.0),
            ]),
            vec![],
        ),
        (
            "thole",
            row(&[("coulomb", 332.0716), ("dielectric", 1.0)]),
            vec![
                (
                    "C",
                    vec!["C"],
                    row(&[("charge", 0.4), ("alpha", 1.1), ("damp", 2.6)]),
                ),
                (
                    "O",
                    vec!["O"],
                    row(&[("charge", -0.7), ("alpha", 0.8), ("damp", 2.1)]),
                ),
            ],
        ),
        (
            "coul/tt",
            row(&[("coulomb", 332.0716), ("dielectric", 1.0)]),
            vec![
                ("C", vec!["C"], row(&[("charge", 0.4)])),
                ("O", vec!["O"], row(&[("charge", -0.7)])),
            ],
        ),
        (
            "mmff_vdw",
            Params::new(),
            vec![
                (
                    "C",
                    vec!["C"],
                    row(&[
                        ("alpha", 1.05),
                        ("n_eff", 2.49),
                        ("a_i", 3.89),
                        ("g_i", 1.282),
                    ]),
                ),
                (
                    "O",
                    vec!["O"],
                    row(&[
                        ("alpha", 0.7),
                        ("n_eff", 3.15),
                        ("a_i", 3.89),
                        ("g_i", 1.282),
                        ("da", 2.0),
                    ]),
                ),
            ],
        ),
        ("uff_lj", Params::new(), vec![]),
    ];
    let builtin = Registry::builtin();
    let pair_styles: Vec<String> = builtin
        .styles(Some("pair"))
        // `coul/long/pme` is a constructor: the reciprocal sum. Its real-space
        // term is the pair form, tested in `kspace::pme` and `pair::coul_long`.
        .filter(|(spec, _)| spec.name != "coul/long/pme")
        .map(|(spec, _)| spec.name.to_string())
        .collect();
    let covered: Vec<&str> = cases.iter().map(|c| c.0).collect();
    for name in &pair_styles {
        assert!(
            covered.contains(&name.as_str()),
            "pair `{name}` is not in the test"
        );
    }
    // Two types, C and O, alternating; per-atom `x1`, `D1` for `uff_lj`.
    let mut frame = molecule();
    let atoms = frame.get_mut("atoms").unwrap();
    atoms
        .insert("type", strings(&["C", "O", "C", "O", "C"]), None)
        .unwrap();
    for (key, v) in [
        ("x1", [3.5, 3.1, 3.8, 2.9, 3.3]),
        ("D1", [0.06, 0.09, 0.05, 0.12, 0.07]),
    ] {
        atoms
            .insert(key, Array1::from_vec(v.to_vec()).into_dyn(), None)
            .unwrap();
    }
    let links: Vec<(usize, usize)> = (0..5)
        .flat_map(|i| ((i + 1)..5).map(move |j| (i, j)))
        .collect();
    // No bonds and no special pair: every pair of `links` at full strength.
    frame.remove("pairs");
    frame.remove("bonds");
    frame.remove("angles");
    frame.remove("dihedrals");
    let table = table_over(&COORDS, &links);
    for (name, style, rows) in cases {
        let mut ff = ForceField::new("t");
        ff.set_special_bonds(SpecialBonds {
            lj: [1.0; 3],
            coul: [1.0; 3],
        });
        let s = ff.def_style("pair", name, style).unwrap();
        for (label, ends, params) in rows {
            s.def_type(label, &ends, params).unwrap();
        }
        let compiler = PotentialCompiler::new(&ff);
        let compiled = compiler
            .compile(&frame)
            .unwrap_or_else(|e| panic!("{name}: {e}"));
        let typed = compiler
            .compile_typed(&frame)
            .unwrap_or_else(|e| panic!("{name} (typed): {e}"));
        let ForceTerm::Pair(kernel) = &typed[0].0 else {
            panic!("{name}: a pair member");
        };
        assert_same(
            name,
            compiled.energy_forces_at(&COORDS),
            kernel.over_pairs(&COORDS, &table),
        );
        assert_virial_matches_forces(name, &COORDS, kernel.over_pairs_virial(&COORDS, &table));
        // The built-in prices pair by pair through its form's law,
        // monomorphized; its registered `dyn` form, a batch at a time through
        // `eval`. The two are one force law.
        let (spec, Some(Kernel::Scalar(form))) = builtin.style("pair", name).unwrap() else {
            panic!("{name}: a scalar form");
        };
        let style = ff.get_style("pair", name).unwrap();
        let rows = style.defs().kernel_type_params().unwrap();
        let refs: Vec<(&str, &Params)> = rows.iter().map(|(l, p)| (l.as_str(), p)).collect();
        let (params, gathered) = spec.gather(style.params(), &refs).unwrap();
        let tp: Vec<(&str, &Params)> = gathered.iter().map(|(l, p)| (l.as_str(), p)).collect();
        let batched = ScalarPair::typed(form.clone(), spec, &params, &tp, &frame).unwrap();
        assert_close(
            &format!("{name}: fused vs batched"),
            kernel.over_pairs(&COORDS, &table),
            batched.over_pairs(&COORDS, &table),
            1e-12,
        );
    }
}
