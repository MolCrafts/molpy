//! The built-in gate (`ff-ir-02-protocol` §5, G5): molrs's own styles are
//! registrations of the same form a third party's are.
//!
//! * [`every_appendix_a_expression_agrees_with_its_kernel`] — every built-in
//!   with an Appendix-A expression prices identically by its native kernel
//!   and by the expression (registered as a style of its own and built into
//!   the form kernels), on 64 seeded configurations × parameter sets per
//!   style: energy to 1e-10 relative, forces too unless the style's force is
//!   not the gradient (`coul/charmm`); a pair style at both compile doors,
//!   its cutoff straddling the pairs, and `compile` = `compile_typed` to
//!   1e-12 (both truncate at the cutoff, as LAMMPS).
//!   The table-generated expressions (`dihedral periodic` per term count,
//!   `nharmonic` per order) are generated here; `dihedral rb`, priced by its
//!   expression alone, is held to `multi/harmonic`'s native kernel.
//! * [`every_param_source_is_what_its_kernel_reads`] — the missing
//!   `ParamSource` gate: a `TypeRows` kernel reads its rows (another row,
//!   another energy; no rows, an error), a `PerInstance` one reads its
//!   per-instance columns and ignores the rows (the same energy, bit for bit,
//!   with or without a row).
//! * [`positional_codecs_match_golden_includes`] — every built-in written
//!   by a [`LammpsForm::Positional`](crate::ff::ir::LammpsForm) codec
//!   writes, byte for byte, the golden include for the LAMMPS-read hand
//!   molecule and every P4 source's LAMMPS form; the golden files are
//!   `ff/testdata/builtin_conformance/*.lmp`.

use crate::core::Cell;
use crate::ff::potential::fixtures::OneShot;
use crate::io::writer::WrittenText;
use std::collections::BTreeSet;
use std::path::Path;
use std::sync::Arc;

use ndarray::Array1;

use crate::ff::compile::PotentialCompiler;
use crate::ff::forcefield::ForceField;
use crate::ff::ir::Params;
use crate::ff::ir::{LammpsForm, ParamKind, ParamSource, StyleSpec};
use crate::ff::potential::ForceTerm;
use crate::ff::style_registry::conformance::{Rng, SEED};
use crate::ff::style_registry::{Registry, RowSource};
use crate::io::lammps::LammpsForcefieldWriter;
use crate::io::lammps::forcefield_reader::LammpsForcefieldReader;
use crate::io::reader::ForceFieldReader;
use molrs::core::Block;
use molrs::core::Frame;
use molrs::core::SimBox;
use molrs::core::TypeLabels;
use molrs::core::{NeighborColumns, NeighborPair, Neighbors, QueryMode};
use molrs::op::{F, Idx};

/// Configurations × parameter sets per style.
const CONFIGS: usize = 64;

/// Four atoms of type `A`, a non-planar chain (`super::tests::chain`'s,
/// moved by up to ±0.2 Å per coordinate), with charges; one term of
/// `category` over the first `arity` atoms typed `t`, and the `pairs` block
/// under which the pair styles price the three pairs that are no chain
/// neighbours.
fn chain(category: &str, arity: usize, rng: &mut Rng) -> (Frame, Vec<F>) {
    const XYZ: [F; 12] = [
        0.0, 0.0, 0.0, 1.52, 0.1, 0.05, 2.1, 1.45, -0.1, 3.55, 1.6, 0.6,
    ];
    let x: Vec<F> = XYZ.iter().map(|v| v + rng.uniform(-0.2, 0.2)).collect();
    let mut atoms = Block::new();
    for (a, key) in ["x", "y", "z"].iter().enumerate() {
        let col: Vec<F> = x.iter().skip(a).step_by(3).copied().collect();
        atoms
            .insert(*key, Array1::from_vec(col).into_dyn(), None)
            .unwrap();
    }
    let strings = |v: Vec<&str>| {
        Array1::from_vec(v.into_iter().map(str::to_owned).collect::<Vec<_>>()).into_dyn()
    };
    atoms.insert("type", strings(vec!["A"; 4]), None).unwrap();
    let charges: Vec<F> = (0..4).map(|_| rng.uniform(-0.6, 0.6)).collect();
    atoms
        .insert("charge", Array1::from_vec(charges).into_dyn(), None)
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
        block.insert("type", strings(vec!["t"]), None).unwrap();
        frame.insert(format!("{category}s"), block);
    }
    frame.insert(
        "pairs",
        crate::ff::potential::pair::fixtures::pricing_only(4, &[(0, 2), (0, 3), (1, 3)], &[]),
    );
    (frame, x)
}

/// The three pairs of [`chain`]'s `pairs` list as a neighbour table at `x`.
fn table(x: &[F]) -> Neighbors {
    Neighbors::from_pairs(
        [(0usize, 2usize), (0, 3), (1, 3)].iter().map(|&(i, j)| {
            let d = [0, 1, 2].map(|a| x[3 * j + a] - x[3 * i + a]);
            NeighborPair {
                i: i as u32,
                j: j as u32,
                dist_sq: d.iter().map(|v| v * v).sum(),
                disp: d,
            }
        }),
        NeighborColumns::FULL,
        QueryMode::SelfQuery { n_points: 4 },
    )
}

/// Energy and forces of `ff` over `frame` at `x`, through `door`.
fn price(ff: &ForceField, frame: &Frame, x: &[F], typed: bool) -> (F, Vec<F>) {
    let compiler = PotentialCompiler::new(ff);
    if !typed {
        return compiler
            .compile(frame)
            .unwrap_or_else(|e| panic!("{}: {e}", ff.name))
            .energy_forces_at(x);
    }
    let mut out = vec![0.0; x.len()];
    let mut acc = crate::ff::potential::Accumulator::adding_into(&mut out);
    for (member, _) in compiler
        .compile_typed(frame)
        .unwrap_or_else(|e| panic!("{} (typed): {e}", ff.name))
    {
        match &member {
            ForceTerm::Pair(p) => p.evaluate_pairs(x, &table(x), &[], &mut acc),
            other => other.as_potential().evaluate(x, &mut acc),
        }
    }
    let e = acc.energy();
    (e, out)
}

/// One gate case: a built-in, the parameters it is drawn around, and the
/// style its energy is compared with (its expression twin, or for `rb`,
/// `multi/harmonic`).
struct Case {
    category: &'static str,
    name: &'static str,
    row: Vec<(&'static str, F)>,
    style: Vec<(&'static str, F)>,
}

fn case(
    category: &'static str,
    name: &'static str,
    row: &[(&'static str, F)],
    style: &[(&'static str, F)],
) -> Case {
    Case {
        category,
        name,
        row: row.to_vec(),
        style: style.to_vec(),
    }
}

/// The base parameters of every built-in with a native kernel and an
/// expression. Pair cutoffs are short enough that the three pairs straddle
/// them (and CHARMM's switch), so both doors' `r < cutoff` and the switch
/// are exercised.
fn cases() -> Vec<Case> {
    let cut = [("cutoff", 3.6)];
    vec![
        case("bond", "harmonic", &[("k", 300.0), ("r0", 1.4)], &[]),
        case(
            "bond",
            "morse",
            &[("d0", 4.0), ("alpha", 1.5), ("r0", 1.4)],
            &[],
        ),
        case(
            "bond",
            "class2",
            &[("r0", 1.4), ("k2", 300.0), ("k3", -400.0), ("k4", 500.0)],
            &[],
        ),
        case("angle", "harmonic", &[("k", 55.0), ("theta0", 104.5)], &[]),
        case(
            "angle",
            "charmm",
            &[
                ("k", 50.0),
                ("theta0", 109.0),
                ("k_ub", 20.0),
                ("r_ub", 2.4),
            ],
            &[],
        ),
        case(
            "angle",
            "class2",
            &[("theta0", 110.0), ("k2", 50.0), ("k3", -10.0), ("k4", 5.0)],
            &[],
        ),
        case(
            "dihedral",
            "charmm",
            &[
                ("k", 1.2),
                ("periodicity", 3.0),
                ("phase", 30.0),
                ("w", 0.0),
            ],
            &[],
        ),
        case(
            "dihedral",
            "opls",
            &[("k1", 1.3), ("k2", -0.2), ("k3", 0.4), ("k4", 0.1)],
            &[],
        ),
        case(
            "dihedral",
            "multi/harmonic",
            &[
                ("a1", 0.3),
                ("a2", -0.5),
                ("a3", 0.2),
                ("a4", 0.7),
                ("a5", -0.1),
            ],
            &[],
        ),
        case(
            "dihedral",
            "harmonic",
            &[("k", 1.1), ("sign", -1.0), ("periodicity", 2.0)],
            &[],
        ),
        case(
            "dihedral",
            "class2",
            &[
                ("k1", 0.5),
                ("phi1", 10.0),
                ("k2", 0.2),
                ("phi2", 20.0),
                ("k3", 0.1),
                ("phi3", 30.0),
            ],
            &[],
        ),
        case("improper", "harmonic", &[("k", 20.0), ("chi0", 5.0)], &[]),
        case(
            "improper",
            "cvff",
            &[("k", 2.0), ("sign", -1.0), ("periodicity", 2.0)],
            &[],
        ),
        case(
            "improper",
            "periodic",
            &[("k", 1.5), ("periodicity", 2.0), ("phase", 180.0)],
            &[],
        ),
        case("pair", "lj/cut", &[("epsilon", 0.2), ("sigma", 3.1)], &cut),
        case(
            "pair",
            "lj/class2",
            &[("epsilon", 0.2), ("sigma", 3.1)],
            &cut,
        ),
        case(
            "pair",
            "buck",
            &[("a", 1000.0), ("rho", 0.3), ("c", 50.0)],
            &cut,
        ),
        case(
            "pair",
            "morse",
            &[("d0", 0.5), ("alpha", 1.2), ("r0", 3.0)],
            &cut,
        ),
        case(
            "pair",
            "lj/charmm",
            &[("epsilon", 0.2), ("sigma", 3.1)],
            &[("inner", 2.6), ("cutoff", 4.2)],
        ),
        case(
            "pair",
            "coul/cut",
            &[],
            &[("coulomb", 332.06371), ("dielectric", 1.0), ("cutoff", 3.6)],
        ),
        case(
            "pair",
            "coul/charmm",
            &[],
            &[
                ("coulomb", 332.06371),
                ("dielectric", 1.0),
                ("inner", 2.6),
                ("cutoff", 4.2),
            ],
        ),
        // Table-generated (the per-term families are drawn per seed below).
        case("dihedral", "periodic", &[], &[]),
        case("dihedral", "nharmonic", &[], &[]),
        case(
            "dihedral",
            "rb",
            &[
                ("c0", 0.2),
                ("c1", -0.6),
                ("c2", 0.4),
                ("c3", 0.9),
                ("c4", -0.3),
                // multi/harmonic stops at cos⁴ φ.
                ("c5", 0.0),
            ],
            &[],
        ),
    ]
}

/// Parameters drawn about `base`: each ±20 %, the integers (periodicity,
/// sign, `w`) kept.
fn drawn(base: &[(&'static str, F)], rng: &mut Rng) -> Vec<(String, F)> {
    base.iter()
        .map(|&(name, v)| {
            let fixed = name.starts_with("periodicity") || name == "sign" || name == "w";
            let v = if fixed { v } else { v * rng.uniform(0.8, 1.2) };
            (name.to_owned(), v)
        })
        .collect()
}

/// A force field of one style `name` of `category`, one type, in `r`.
fn one_style(
    r: &Arc<Registry>,
    category: &str,
    name: &str,
    row: &[(String, F)],
    style: &[(String, F)],
) -> ForceField {
    let mut ff = ForceField::new(name).with_registry(r.clone());
    ff.set_special_bonds(crate::ff::ir::SpecialBonds::EXCLUDE_12_13);
    let pairs = |v: &[(String, F)]| {
        let mut p = Params::new();
        for (k, x) in v {
            p.set(k, *x);
        }
        p
    };
    let s = ff
        .def_style(category, name, pairs(style))
        .unwrap_or_else(|e| panic!("{category} {name}: {e}"));
    if !row.is_empty() {
        let arity = r.category(category).unwrap().arity.endpoints();
        let (label, ends): (&str, Vec<&str>) = if category == "pair" {
            ("A", vec!["A"])
        } else {
            ("t", vec!["A"; arity])
        };
        s.def_type(label, &ends, pairs(row))
            .unwrap_or_else(|e| panic!("{category} {name}: {e}"));
    }
    ff
}

/// The relative difference of `(e, f)` and `(e2, f2)`: energy against
/// `max(|e|, max |f|)`, each force component likewise.
fn rel(a: &(F, Vec<F>), b: &(F, Vec<F>)) -> (F, F) {
    let scale = a.1.iter().fold(a.0.abs(), |m, v| m.max(v.abs()));
    let fr =
        a.1.iter()
            .zip(&b.1)
            .fold(0.0_f64, |m, (x, y)| m.max((x - y).abs() / scale));
    ((a.0 - b.0).abs() / scale, fr)
}

#[test]
fn every_appendix_a_expression_agrees_with_its_kernel() {
    let builtin = Registry::builtin();
    let mut checked = BTreeSet::new();
    let mut report = Vec::new();
    for (n, c) in cases().into_iter().enumerate() {
        let (spec, _) = builtin.style(c.category, c.name).unwrap();
        let arity = builtin.category(c.category).unwrap().arity.endpoints();
        let mut rng = Rng(SEED.wrapping_add(n as u64));
        let mut worst = (0.0_f64, 0.0_f64);
        for config in 0..CONFIGS {
            // The parameters, and the reference style and its row.
            let (mut row, style) = (drawn(&c.row, &mut rng), drawn(&c.style, &mut rng));
            let mut r = builtin.clone();
            let (reference, twin_name): (&str, String) = match (c.category, c.name) {
                // Σₘ k_m [1 + cos(n_m φ − d_m)], M = 1 … 3 terms.
                ("dihedral", "periodic") => {
                    let m = 1 + config % 3;
                    let mut terms = Vec::new();
                    for t in 1..=m {
                        row.push((format!("k{t}"), rng.uniform(-2.0, 2.0)));
                        row.push((format!("periodicity{t}"), (1 + (config + t) % 6) as F));
                        row.push((format!("phase{t}"), rng.uniform(-180.0, 180.0)));
                        terms.push(format!(
                            "k{t}*(1+cos(periodicity{t}*phi-phase{t}*(pi/180)))"
                        ));
                    }
                    let expression = format!("{}; pi=3.141592653589793", terms.join("+"));
                    (c.name, twin(&mut r, spec, expression, m))
                }
                // Σᵢ aᵢ cos^(i−1) φ, N = 2 … 5.
                ("dihedral", "nharmonic") => {
                    let order = 2 + config % 4;
                    let mut terms = Vec::new();
                    for i in 1..=order {
                        row.push((format!("a{i}"), rng.uniform(-1.0, 1.0)));
                        terms.push(format!("a{i}*c^{}", i - 1));
                    }
                    let expression = format!("{}; c=cos(phi)", terms.join("+"));
                    (c.name, twin(&mut r, spec, expression, order))
                }
                // rb has no native kernel: its expression against
                // multi/harmonic's, aᵢ₊₁ = (−1)ⁱ cᵢ.
                ("dihedral", "rb") => (c.name, String::new()),
                _ => {
                    let expression = spec.expression.clone().unwrap();
                    (c.name, twin(&mut r, spec, expression, 0))
                }
            };
            let (frame, x) = chain(c.category, arity, &mut rng);
            // Both doors truncate at the style's `cutoff` (`r < cutoff`, as
            // LAMMPS), and the cutoff straddles the three pairs: the
            // `pairs`-list door prices the same pairs the neighbour-driven
            // one does.
            let snapshot = Arc::new(r.clone());
            let r = &snapshot;
            let fields = || {
                if c.name == "rb" {
                    let multi: Vec<(String, F)> = row
                        .iter()
                        .filter(|(k, _)| k != "c5")
                        .map(|(k, v)| {
                            let i: usize = k[1..].parse().unwrap();
                            (
                                format!("a{}", i + 1),
                                if i.is_multiple_of(2) { *v } else { -v },
                            )
                        })
                        .collect();
                    (
                        one_style(r, "dihedral", "multi/harmonic", &multi, &style),
                        one_style(r, "dihedral", "rb", &row, &style),
                    )
                } else {
                    (
                        one_style(r, c.category, reference, &row, &style),
                        one_style(r, c.category, &twin_name, &row, &style),
                    )
                }
            };
            let doors: &[bool] = if c.category == "pair" {
                &[false, true]
            } else {
                &[false]
            };
            let (native_ff, twin_ff) = fields();
            let mut by_door = Vec::new();
            for &typed in doors {
                let a = price(&native_ff, &frame, &x, typed);
                let b = price(&twin_ff, &frame, &x, typed);
                let scale = a.1.iter().fold(a.0.abs(), |m, v| m.max(v.abs()));
                assert!(
                    scale > 1e-8,
                    "{} {} config {config}: the term must contribute",
                    c.category,
                    c.name
                );
                by_door.push(a.clone());
                let (re, rf) = rel(&a, &b);
                let rf = if spec.force_is_gradient { rf } else { 0.0 };
                assert!(
                    re <= 1e-10 && rf <= 1e-10,
                    "{} {} config {config} ({}): native ({:?}, {:?}), expression ({:?}, {:?}): \
                     rel energy {re:.1e}, force {rf:.1e}",
                    c.category,
                    c.name,
                    if typed { "compile_typed" } else { "compile" },
                    a.0,
                    a.1,
                    b.0,
                    b.1
                );
                worst = (worst.0.max(re), worst.1.max(rf));
            }
            // The `pairs`-list door is the neighbour-driven one over the same
            // pairs: the same cutoff, the same switch.
            if let [compiled, typed] = &by_door[..] {
                let (re, rf) = rel(compiled, typed);
                assert!(
                    re <= 1e-12 && rf <= 1e-12,
                    "{} {} config {config}: compile ({:?}, {:?}), compile_typed ({:?}, {:?}): \
                     rel energy {re:.1e}, force {rf:.1e}",
                    c.category,
                    c.name,
                    compiled.0,
                    compiled.1,
                    typed.0,
                    typed.1
                );
            }
        }
        report.push(format!(
            "{} {}: worst rel energy {:.1e}, force {:.1e}",
            c.category, c.name, worst.0, worst.1
        ));
        checked.insert((c.category, c.name));
    }
    println!("{}", report.join("\n"));
    // Every built-in with an expression, or priced by its table-generated
    // one, is in the gate (`drude harmonic`, a spec with no kernel, aside).
    for (spec, kernel) in builtin.styles(None) {
        let generated =
            spec.category == "dihedral" && ["periodic", "nharmonic"].contains(&&*spec.name);
        if (spec.expression.is_some() || generated) && (kernel.is_some() || spec.name == "rb") {
            assert!(
                checked.contains(&(spec.category.as_ref(), spec.name.as_ref())),
                "{} {} is not in the gate",
                spec.category,
                spec.name
            );
        }
    }
}

/// Register `spec` again as `<name>/expression[/m]`, priced by
/// `expression` alone; its name.
fn twin(r: &mut Registry, spec: &StyleSpec, expression: String, m: usize) -> String {
    let mut twin = spec.clone();
    twin.name = if m == 0 {
        format!("{}/expression", spec.name)
    } else {
        format!("{}/expression/{m}", spec.name)
    }
    .into();
    twin.expression = Some(expression);
    twin.lammps = LammpsForm::None;
    twin.samples.clear();
    let name = twin.name.to_string();
    r.register_style(twin, None)
        .unwrap_or_else(|e| panic!("{} {}: {e}", spec.category, name));
    name
}

// ---------------------------------------------------------------------------
// ParamSource
// ---------------------------------------------------------------------------

/// A plausible value of a parameter named `name` (any built-in's).
fn plausible(name: &str) -> F {
    match name {
        "theta0" | "chi0" => 109.0,
        "phase" | "phi1" | "phi2" | "phi3" => 15.0,
        "periodicity" | "order" => 2.0,
        "sign" => -1.0,
        "r0" | "r0_ij" | "r0_kj" | "r_ub" => 1.45,
        "sigma" | "sigma14" | "x1" => 3.1,
        "epsilon" | "epsilon14" | "D1" => 0.2,
        "charge" => 0.4,
        "linear" | "cosTerm" | "da" | "w" => 0.0,
        "n_eff" | "a_i" | "g_i" => 1.5,
        "c0" => 0.5,
        "alpha" => 1.1,
        "damp" => 2.6,
        _ => 0.7,
    }
}

/// The style params a built-in needs to build.
fn style_params(spec: &StyleSpec) -> Params {
    let mut p = Params::new();
    for decl in &spec.style_params {
        if decl.kind != ParamKind::Scalar {
            continue;
        }
        let v = match decl.name.as_ref() {
            "cutoff" => 9.0,
            "inner" => 8.0,
            "coulomb" => 332.06371,
            "dielectric" => 1.0,
            "alpha" => 0.3,
            "order" => 4.0,
            "grid_x" | "grid_y" | "grid_z" => 8.0,
            other => match decl.default.as_ref().and_then(|d| d.as_num()) {
                Some(v) => v,
                None => plausible(other),
            },
        };
        p.set(&decl.name, v);
    }
    p
}

/// `spec`'s scalar per-type parameters at their plausible values, times
/// `scale` (the integers kept).
fn row(spec: &StyleSpec, scale: F) -> Params {
    let mut p = Params::new();
    for decl in spec.params.iter().filter(|d| d.kind == ParamKind::Scalar) {
        let v = plausible(&decl.name);
        let integral = [
            "periodicity",
            "order",
            "sign",
            "linear",
            "cosTerm",
            "da",
            "w",
        ]
        .contains(&decl.name.as_ref());
        let v = if integral { v } else { v * scale };
        if decl.indexed {
            // Two terms, `k1 k2` (periodicity 2 and 3).
            for m in 1..=2 {
                let v = if integral {
                    v + (m - 1) as F
                } else {
                    v / m as F
                };
                p.set(&format!("{}{m}", decl.name), v);
            }
        } else {
            p.set(&decl.name, v);
        }
    }
    p
}

/// The invariant `potential::registry` states and nothing checked: a
/// built-in kernel — a form or a constructor — ignores its type rows if and
/// only if its style is registered `PerInstance`.
#[test]
fn every_param_source_is_what_its_kernel_reads() {
    let builtin = Arc::new(Registry::builtin());
    let mut rng = Rng(SEED);
    let mut checked = Vec::new();
    for (spec, kernel) in builtin.styles(None) {
        if kernel.is_none() {
            // Priced by its expression alone (`dihedral rb`), or nothing.
            continue;
        }
        if spec.params.iter().any(|d| d.kind != ParamKind::Scalar) {
            // `cmap charmm`: a grid per row, table-driven by construction.
            assert_eq!(spec.source, ParamSource::TypeRows);
            continue;
        }
        let category = builtin.category(&spec.category).unwrap();
        let arity = category.arity.endpoints();
        let (mut frame, x) = chain(&spec.category, arity, &mut rng);
        // The frame's own periodic box — frame data, which a kernel that
        // needs it (PME's Ewald sums) reads off the frame, never off the
        // style: every style builds from its declared parameters alone. A
        // pair style compile prices from its candidate pairs is priced in
        // free space (compile refuses it a periodic box).
        let candidate_priced = category.is_pair_driven()
            && builtin.row_source(&spec.category, &spec.name) != Some(RowSource::Atoms);
        if !candidate_priced {
            frame.simbox =
                Some(SimBox::new(Cell::Lengths([20.0; 3]), [0.0, 0.0, 0.0], [true; 3]).unwrap());
        }
        // A per-instance style's every declared parameter as a per-instance
        // column: the block's (one row), or the atoms' for a pair. A
        // table-driven style is handed none — a bonded form reads a column
        // its block carries before a row (molrec linking rule 4), which is
        // the row source of a per-instance style, not of a table.
        let block = if category.is_pair_driven() {
            "atoms".to_owned()
        } else {
            category.block.to_string()
        };
        let n = frame.get(&block).unwrap().n_rows().unwrap();
        let b = frame.get_mut(&block).unwrap();
        let columns = if spec.source == ParamSource::PerInstance {
            &spec.params[..]
        } else {
            &[]
        };
        for decl in columns {
            if b.get(&decl.name).is_some() {
                continue;
            }
            let v = plausible(&decl.name);
            if decl.name == "linear" {
                // MMFF's linear-centre flag is an integer column.
                let flags: Vec<molrs::op::I> = vec![v as molrs::op::I; n];
                b.insert("linear", Array1::from_vec(flags).into_dyn(), None)
                    .unwrap();
            } else {
                b.insert(
                    decl.name.as_ref(),
                    Array1::from_vec(vec![v; n]).into_dyn(),
                    None,
                )
                .unwrap();
            }
        }
        let field = |rows: Option<Params>| {
            let mut ff = ForceField::new("gate").with_registry(builtin.clone());
            ff.set_special_bonds(crate::ff::ir::SpecialBonds::EXCLUDE_12_13);
            // 1-4 pairs priced by no pair style: a `dihedral charmm` `w`
            // may price them.
            ff.set_special_bonds(crate::ff::ir::SpecialBonds {
                lj: [0.0; 3],
                coul: [0.0; 3],
            });
            let s = ff
                .def_style(spec.category.as_ref(), &spec.name, style_params(spec))
                .unwrap();
            if let Some(p) = rows {
                let (label, ends) = if category.is_pair_driven() {
                    ("A", vec!["A"])
                } else {
                    ("t", vec!["A"; arity])
                };
                s.def_type(label, &ends, p).unwrap();
            }
            ff
        };
        let energy = |ff: &ForceField| {
            PotentialCompiler::new(ff)
                .compile(&frame)
                .map(|p| p.energy_at(&x))
        };
        let what = format!("{} {}", spec.category, spec.name);
        match spec.source {
            ParamSource::TypeRows => {
                assert!(energy(&field(None)).is_err(), "{what}: no rows, no energy");
                let a =
                    energy(&field(Some(row(spec, 1.0)))).unwrap_or_else(|e| panic!("{what}: {e}"));
                let b =
                    energy(&field(Some(row(spec, 1.37)))).unwrap_or_else(|e| panic!("{what}: {e}"));
                assert!(
                    (a - b).abs() > 1e-9 * a.abs().max(b.abs()),
                    "{what}: declared TypeRows, but another row prices the same {a}"
                );
            }
            ParamSource::PerInstance => {
                let a = energy(&field(None)).unwrap_or_else(|e| panic!("{what}: {e}"));
                let b =
                    energy(&field(Some(row(spec, 1.37)))).unwrap_or_else(|e| panic!("{what}: {e}"));
                assert_eq!(
                    a.to_bits(),
                    b.to_bits(),
                    "{what}: declared PerInstance, but a row changes its energy"
                );
            }
        }
        checked.push(format!("{what} ({:?})", spec.source));
    }
    println!("{} kernels: {checked:?}", checked.len());
    assert!(checked.len() >= 30, "{checked:?}");
}

// ---------------------------------------------------------------------------
// Positional codecs: the golden includes
// ---------------------------------------------------------------------------

/// One field per positional built-in (and per units, `real` and `metal`:
/// the conversion by dimension), each type of it on a four-atom chain
/// `A-B-A-B`, written with 17 digits.
fn style_cases() -> Vec<(String, String)> {
    type Row = &'static [(&'static str, F)];
    let styles: [(&str, &str, Row, Row); 11] = [
        ("bond", "harmonic", &[("k", 312.5), ("r0", 1.4321)], &[]),
        (
            "bond",
            "morse",
            &[("d0", 4.123), ("alpha", 1.987), ("r0", 1.4321)],
            &[],
        ),
        (
            "bond",
            "class2",
            &[("r0", 1.4321), ("k2", 301.7), ("k3", -402.3), ("k4", 503.9)],
            &[],
        ),
        (
            "angle",
            "harmonic",
            &[("k", 55.25), ("theta0", 104.52)],
            &[],
        ),
        (
            "angle",
            "charmm",
            &[
                ("k", 50.5),
                ("theta0", 109.47),
                ("k_ub", 21.3),
                ("r_ub", 2.41),
            ],
            &[],
        ),
        (
            "dihedral",
            "opls",
            &[("k1", 1.3), ("k2", -0.2), ("k3", 0.4), ("k4", 0.1)],
            &[],
        ),
        (
            "dihedral",
            "multi/harmonic",
            &[
                ("a1", 0.3),
                ("a2", -0.5),
                ("a3", 0.2),
                ("a4", 0.7),
                ("a5", -0.1),
            ],
            &[],
        ),
        ("improper", "harmonic", &[("k", 20.25), ("chi0", 5.5)], &[]),
        (
            "pair",
            "buck",
            &[("a", 1000.5), ("rho", 0.3125), ("c", 50.75)],
            &[("cutoff", 9.5)],
        ),
        (
            "pair",
            "morse",
            &[("d0", 0.5125), ("alpha", 1.25), ("r0", 3.125)],
            &[("cutoff", 9.5)],
        ),
        (
            "pair",
            "lj/cut",
            &[("epsilon", 0.1953), ("sigma", 3.1234)],
            &[("cutoff", 9.5)],
        ),
    ];
    let strings = |v: &[&str]| {
        Array1::from_vec(v.iter().map(|s| (*s).to_owned()).collect::<Vec<_>>()).into_dyn()
    };
    let mut out = Vec::new();
    for units in ["real", "metal"] {
        for (category, name, row, style) in styles {
            let mut ff = ForceField::new("style");
            ff.set_special_bonds(crate::ff::ir::SpecialBonds::EXCLUDE_12_13);
            ff.set_units("real");
            let atoms = ff.def_style("atom", "full", Params::new()).unwrap();
            for (t, m) in [("A", 12.011), ("B", 14.007)] {
                atoms
                    .def_type(t, &[], Params::from_iter([("mass", m), ("charge", 0.0)]))
                    .unwrap();
            }
            let s = ff
                .def_style(category, name, Params::from_iter(style.iter().copied()))
                .unwrap();
            let arity = match category {
                "bond" => 2,
                "angle" => 3,
                _ => 4,
            };
            let mut frame = Frame::new();
            let mut a = Block::new();
            a.insert("type", strings(&["A", "B", "A", "B"]), None)
                .unwrap();
            frame.insert("atoms", a);
            if category == "pair" {
                for (t, ends) in [("A", vec!["A"]), ("B", vec!["B"]), ("A-B", vec!["A", "B"])] {
                    s.def_type(t, &ends, Params::from_iter(row.iter().copied()))
                        .unwrap();
                }
            } else {
                let ends = &["A", "B", "A", "B"][..arity];
                s.def_type("t", ends, Params::from_iter(row.iter().copied()))
                    .unwrap();
                let mut b = Block::new();
                for (k, key) in ["atomi", "atomj", "atomk", "atoml"][..arity]
                    .iter()
                    .enumerate()
                {
                    b.insert(*key, Array1::from_vec(vec![k as Idx]).into_dyn(), None)
                        .unwrap();
                }
                b.insert("type", strings(&["t"]), None).unwrap();
                frame.insert(format!("{category}s"), b);
            }
            let labels = TypeLabels::from_frame(&frame).unwrap();
            let text = LammpsForcefieldWriter::new(&labels)
                .with_precision(17)
                .with_units(units)
                .written(&ff)
                .unwrap_or_else(|e| panic!("{category} {name} in {units}: {e}"));
            let case = format!("style_{category}_{}_{units}", name.replace('/', "_"));
            out.push((case, text.to_string()));
        }
    }
    out
}

/// `(case, include)` of every field the positional codecs are held on: the
/// LAMMPS-read hand molecule, every P4 source's LAMMPS form, and one field
/// per positional built-in in `real` and `metal`, written with 17 digits.
fn positional_cases() -> Vec<(String, String)> {
    use crate::ff::equivalence_check::{engine_form, lammps_include, sources};
    use crate::ff::ir_invariance::{LAMMPS_FF, hand_frame};
    let mut out = Vec::new();
    let hand = [(
        "hand_lammps",
        LammpsForcefieldReader::new()
            .read(LAMMPS_FF.as_bytes())
            .unwrap(),
    )];
    for (name, ff) in hand {
        let frame = hand_frame(&ff);
        let labels = TypeLabels::from_frame(&frame).unwrap();
        let text = LammpsForcefieldWriter::new(&labels)
            .with_precision(17)
            .written(&ff)
            .unwrap_or_else(|e| panic!("{name}: {e}"));
        out.push((name.to_owned(), text.to_string()));
    }
    for source in sources() {
        let (ff, frame) = engine_form(&source.load(), "lammps");
        let (pre, post, _) = lammps_include(&ff, &frame);
        out.push((source.name.to_owned(), pre + &post));
    }
    out.extend(style_cases());
    out
}

/// The styles a field writes through a positional codec.
fn positional_styles(ff_text: &str) -> BTreeSet<String> {
    let r = Registry::builtin();
    ff_text
        .lines()
        .filter_map(|l| {
            let mut w = l.split_whitespace();
            let (cmd, name) = (w.next()?, w.next()?);
            let category = cmd.strip_suffix("_style")?;
            let (spec, _) = r.lammps_style(category, name)?;
            matches!(spec.lammps, LammpsForm::Positional { .. })
                .then(|| format!("{category} {}", spec.name))
        })
        .collect()
}

/// `text` against the golden `want`: the same lines and tokens, a number
/// that differs differing by rounding alone (≤ 4ε relative: two ulps). Some
/// golden files convert each value through LAMMPS's `lj` units by a unit
/// expression (even `real` → `real`: `bond morse`'s `alpha` is one ulp off
/// the stored 1.987); the positional codec multiplies by one exact factor
/// per dimension, the identity exactly. Returns the tokens that differ,
/// `"<case>: <golden> → <written>"`.
fn same_but_last_digit(case: &str, text: &str, want: &str) -> Vec<String> {
    let (written, golden): (Vec<&str>, Vec<&str>) =
        (text.lines().collect(), want.lines().collect());
    assert_eq!(
        written.len(),
        golden.len(),
        "{case}: lines\n{text}\n---\n{want}"
    );
    let mut out = Vec::new();
    for (a, b) in written.iter().zip(&golden) {
        let (ta, tb): (Vec<&str>, Vec<&str>) = (
            a.split_whitespace().collect(),
            b.split_whitespace().collect(),
        );
        assert_eq!(ta.len(), tb.len(), "{case}: {a:?} vs {b:?}");
        for (x, y) in ta.iter().zip(&tb) {
            if x == y {
                continue;
            }
            let (Ok(u), Ok(v)) = (x.parse::<F>(), y.parse::<F>()) else {
                panic!("{case}: {a:?} vs {b:?}");
            };
            assert!(
                (u - v).abs() <= 4.0 * F::EPSILON * u.abs().max(v.abs()),
                "{case}: golden {y}, written {x}: beyond rounding (4ε relative)"
            );
            out.push(format!("{case}: {y} → {x}"));
        }
    }
    out
}

/// The positional built-ins no golden include holds (`bond class2`,
/// `pair buck`, `pair morse` are held by read → write → read in
/// `ff::engine_codec_check` instead).
const WITHOUT_GOLDEN: [&str; 3] = ["bond class2", "pair buck", "pair morse"];

#[test]
fn positional_codecs_match_golden_includes() {
    let dir = Path::new(env!("CARGO_MANIFEST_DIR")).join("src/ff/testdata/builtin_conformance");
    let mut held = BTreeSet::new();
    let mut unpinned = BTreeSet::new();
    let (mut identical, mut ulps) = (Vec::new(), Vec::new());
    for (name, text) in positional_cases() {
        let Ok(want) = std::fs::read_to_string(dir.join(format!("{name}.lmp"))) else {
            unpinned.extend(positional_styles(&text));
            continue;
        };
        if text == want {
            identical.push(name.clone());
        } else {
            ulps.extend(same_but_last_digit(&name, &text, &want));
        }
        held.extend(positional_styles(&text));
    }
    println!("byte for byte: {identical:?}");
    println!("by rounding alone: {ulps:?}");
    println!("positional built-ins held: {held:?}");
    // The P4 sources and the hand molecule: byte for byte.
    for case in [
        "hand_lammps",
        "ff14sb",
        "gaff2",
        "chamber",
        "charmm36",
        "oplsaa",
    ] {
        assert!(identical.iter().any(|c| c == case), "{case}");
    }
    // Every positional built-in is held, or is one without a golden include.
    let r = Registry::builtin();
    for (spec, _) in r.styles(None) {
        if !matches!(spec.lammps, LammpsForm::Positional { .. }) {
            continue;
        }
        let name = format!("{} {}", spec.category, spec.name);
        if WITHOUT_GOLDEN.contains(&name.as_str()) {
            assert!(unpinned.contains(&name), "{name}: written by a case");
        } else {
            assert!(
                held.contains(&name),
                "{name}: not held byte for byte ({held:?})"
            );
        }
    }
}
