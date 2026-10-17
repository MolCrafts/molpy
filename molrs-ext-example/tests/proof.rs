//! The proof that the force-field IR is a protocol (`ff-ir-02-protocol`,
//! P-Rust): everything this crate adds — through molrs's `pub` API alone —
//! prices as LAMMPS prices it, persists, and is refused by name where it
//! does not conform.
//!
//! | Test | Criterion |
//! |---|---|
//! | [`pair_style_matches_lammps`] | E and every force component = pinned LAMMPS `lj/smooth/linear`, its cutoff straddling the pairs, rel ≤ 1e-10; `compile_typed` = `compile`, rel ≤ 1e-12 |
//! | [`new_category_matches_lammps`] | the `urey_bradley` category = pinned LAMMPS `angle_style charmm` with K = 0, rel ≤ 1e-10 |
//! | [`fene_matches_lammps`] | `bond fene` by expression = pinned LAMMPS `bond_style fene`, rel ≤ 1e-10, (r/R0)² < 0.9 |
//! | [`mrec_round_trip`] | energy bit for bit after `.mrec` write/read; with `Registry::builtin()` only, the expression styles still price (bit for bit), the native-only ones are `NoKernel` naming them |
//! | [`nonconforming_refused`] | every `IrError` variant of the protocol's §5 reachable from Rust registration, compile and export, asserted by variant and named item |
//! | [`bond_angle_cross_term`] | class2's bond-angle term as a new category: hand value, central differences, expression = native form, rel ≤ 1e-12 |
//!
//! The LAMMPS numbers are pinned in `tests/lammps.tsv`, which
//! `scripts/ff_ir_extension_lammps_check.sh --pin` writes: with
//! `MOLRS_FFEXT_DIR` set, [`write_lammps_inputs`] writes each case's deck
//! through molrs's LAMMPS writer (the positional codecs the specs derive),
//! the script runs `lmp` `run 0` on it (`thermo_modify format float %.17g`,
//! a `%.17g` forces dump) and prints LAMMPS's numbers beside molrs's.
//!
//! Relative error is `|got − want| / max(|want|, s)`, `s` the largest
//! |force component| of the configuration (an energy's own magnitude for
//! the energy): a near-zero component is held to the configuration's force
//! scale, not to itself.

use std::collections::BTreeMap;
use std::path::Path;
use std::sync::Arc;

use molrs::core::TypeLabels;
use molrs::core::{Block, Frame};
use molrs::core::{Cell, SimBox};
use molrs::core::{NeighborColumns, NeighborPair, Neighbors, QueryMode};
use molrs::ff::compile::PotentialCompiler;
use molrs::ff::forcefield::{DefError, ForceField};
use molrs::ff::ir::{
    Arity, CategorySpec, ConformanceSample, Coordinate, EndpointOrder, Engine, FormCodec,
    FormRefusal, IrError, LammpsForm, ParamDimension, ParamSpec, ParamValue, StyleSpec,
};
use molrs::ff::ir::{Params, SpecialBonds};
use molrs::ff::potential::form_kernel::{ParamColumns, ScalarForm};
use molrs::ff::potential::{Accumulator, CompileError, ForceTerm, Potential, intramolecular_pairs};
use molrs::ff::style_registry::{Kernel, Registry};
use molrs::io::gromacs::GromacsTopForcefieldWriter;
use molrs::io::lammps::LammpsForcefieldWriter;
use molrs::io::mrec::{ForceFieldSection, MolRec, MrecFile};
use molrs::io::writer::ForceFieldWriter;
use molrs::io::{write_lammps_data, write_mrec};
use molrs_ext_example as ext;
use ndarray::Array1;

// ---------------------------------------------------------------------------
// Systems
// ---------------------------------------------------------------------------

/// One LAMMPS case: what molrs prices (`ff` over `frame`, in the example's
/// registry), the deck LAMMPS prices (`deck` over `deck_frame`: the same
/// field, or its LAMMPS spelling), and the configurations (Å, on the
/// 0.01 Å grid a data file holds exactly).
struct Case {
    name: &'static str,
    ff: ForceField,
    frame: Frame,
    deck: ForceField,
    deck_frame: Frame,
    configs: Vec<Vec<f64>>,
    /// Extra lines after the include (a class2 `ba` line).
    extra: &'static str,
}

const MASSES: [(&str, f64); 2] = [("A", 12.011), ("B", 14.007)];

fn strings(v: &[&str]) -> ndarray::ArrayD<String> {
    Array1::from_vec(v.iter().map(|s| (*s).to_owned()).collect()).into_dyn()
}

/// A relation block: its name and its `(atoms, type)` rows.
type Relation<'a> = (&'a str, &'a [(&'a [u32], &'a str)]);

/// Atoms of `types` (no charge) and the `relations` blocks.
fn frame(types: &[&str], relations: &[Relation<'_>]) -> Frame {
    let n = types.len();
    let mut atoms = Block::new();
    atoms.insert("type", strings(types), None).unwrap();
    let mass: Vec<f64> = types
        .iter()
        .map(|t| MASSES.iter().find(|(k, _)| k == t).unwrap().1)
        .collect();
    atoms
        .insert("mass", Array1::from_vec(mass).into_dyn(), None)
        .unwrap();
    atoms
        .insert("charge", Array1::from_vec(vec![0.0; n]).into_dyn(), None)
        .unwrap();
    atoms
        .insert("mol_id", Array1::from_vec(vec![1u32; n]).into_dyn(), None)
        .unwrap();
    for key in ["x", "y", "z"] {
        atoms
            .insert(key, Array1::from_vec(vec![0.0; n]).into_dyn(), None)
            .unwrap();
    }
    let mut f = Frame::new();
    f.insert("atoms", atoms);
    for (block, rows) in relations {
        let mut b = Block::new();
        let arity = rows[0].0.len();
        for (k, key) in ["atomi", "atomj", "atomk"][..arity].iter().enumerate() {
            let col: Vec<u32> = rows.iter().map(|r| r.0[k]).collect();
            b.insert(*key, Array1::from_vec(col).into_dyn(), None)
                .unwrap();
        }
        let types: Vec<&str> = rows.iter().map(|r| r.1).collect();
        b.insert("type", strings(&types), None).unwrap();
        f.insert(*block, b);
    }
    f
}

/// `frame` with its `pairs` list under `ff`'s special-bonds weights.
fn with_pairs(ff: &ForceField, mut frame: Frame) -> Frame {
    let pairs = intramolecular_pairs(&frame, ff.special_bonds(), None).unwrap();
    frame.insert("pairs", pairs);
    frame
}

/// `ff`, read against `reg`, with an `atom full` style of the masses.
fn field(name: &str, special: [f64; 3], reg: &Registry) -> ForceField {
    let mut ff = ForceField::new(name).with_registry(Arc::new(reg.clone()));
    ff.set_units("real");
    ff.set_special_bonds(SpecialBonds {
        lj: special,
        coul: special,
    });
    let atoms = ff.def_style("atom", "full", Params::new()).unwrap();
    for (t, m) in MASSES {
        atoms
            .def_type(t, &[], Params::from_iter([("mass", m), ("charge", 0.0)]))
            .unwrap();
    }
    ff
}

/// `x` with atom `i` moved by `d` Å times `(i mod 3 − 1, (i + 1) mod 2,
/// 2 (i mod 2) − 1)`: a second configuration on the grid, every distance
/// and angle changed.
fn shifted(x: &[f64], d: f64) -> Vec<f64> {
    x.chunks(3)
        .enumerate()
        .flat_map(|(i, p)| {
            let step = [
                (i % 3) as f64 - 1.0,
                ((i + 1) % 2) as f64,
                2.0 * (i % 2) as f64 - 1.0,
            ];
            [p[0] + d * step[0], p[1] + d * step[1], p[2] + d * step[2]]
        })
        .collect()
}

/// `pair lj/smooth/linear`, six unbonded atoms of two types mixed by
/// Lorentz–Berthelot, the 5 Å cutoff straddling their pairs (ten inside,
/// five beyond, in each configuration; the nearest 0.19 Å from it): both
/// compile doors price a pair only at `r < cutoff`, as LAMMPS does.
fn smooth(reg: &Registry) -> Case {
    let mut ff = field("smooth", [0.0; 3], reg);
    let mut style = Params::from_iter([("cutoff", SMOOTH_CUTOFF)]);
    style.set("mixing", "arithmetic");
    let lj = ff.def_style("pair", "lj/smooth/linear", style).unwrap();
    for (t, eps, sigma) in [("A", 0.2, 3.1), ("B", 0.15, 3.6)] {
        lj.def_type(
            t,
            &[t],
            Params::from_iter([("epsilon", eps), ("sigma", sigma)]),
        )
        .unwrap();
    }
    let types = ["A", "A", "B", "B", "B", "A"];
    let frame = with_pairs(&ff, frame(&types, &[]));
    let x = vec![
        0.0, 0.0, 0.0, 3.71, 0.12, 0.0, 0.05, 3.93, 0.21, 3.62, 3.84, 0.43, 1.83, 1.91, 3.52, 6.53,
        0.24, 0.11,
    ];
    Case {
        name: "smooth",
        deck: ff.clone(),
        deck_frame: frame.clone(),
        ff,
        frame,
        configs: vec![x.clone(), shifted(&x, 0.13)],
        extra: "",
    }
}

/// The `lj/smooth/linear` cutoff of [`smooth`], Å.
const SMOOTH_CUTOFF: f64 = 5.0;

/// `bond fene` on a five-atom chain, three bond types, two bonds inside
/// the WCA range `2^(1/6) σ`.
fn fene(reg: &Registry) -> Case {
    let mut ff = field("fene", [0.0, 1.0, 1.0], reg);
    let style = ff.def_style("bond", "fene", Params::new()).unwrap();
    for (name, ends, k, r0, eps, sigma) in [
        ("A-A", ["A", "A"], 30.0, 1.5, 1.0, 1.0),
        ("A-B", ["A", "B"], 25.0, 1.75, 0.8, 1.1),
        ("B-B", ["B", "B"], 40.0, 1.6, 1.2, 0.95),
    ] {
        style
            .def_type(
                name,
                &ends,
                Params::from_iter([("k", k), ("r0", r0), ("epsilon", eps), ("sigma", sigma)]),
            )
            .unwrap();
    }
    let bonds: &[(&[u32], &str)] = &[
        (&[0, 1], "A-A"),
        (&[1, 2], "A-B"),
        (&[2, 3], "B-B"),
        (&[3, 4], "A-B"),
    ];
    let frame = frame(&["A", "A", "B", "B", "A"], &[("bonds", bonds)]);
    // 1.05 (inside the A-A WCA range 1.12), 1.30, 1.02 (inside B-B's
    // 1.07), 1.27: (r/R0)² ≤ 0.59.
    let x = vec![
        0.0, 0.0, 0.0, 1.05, 0.0, 0.0, 1.05, 1.3, 0.0, 1.05, 1.3, 1.02, 1.95, 2.2, 1.02,
    ];
    Case {
        name: "fene",
        deck: ff.clone(),
        deck_frame: frame.clone(),
        ff,
        frame,
        configs: vec![x.clone(), shifted(&x, 0.03)],
        extra: "",
    }
}

/// The 1-3 terms of a four-atom chain: `(atoms, type, k_ub, r_ub)`.
const UB_ROWS: [([u32; 3], &str, f64, f64); 2] = [
    ([0, 1, 2], "A-B-B", 22.5, 2.45),
    ([1, 2, 3], "B-B-A", 18.0, 2.62),
];
const CHAIN: [f64; 12] = [
    0.0, 0.0, 0.0, 1.42, 0.31, 0.0, 2.05, 1.62, 0.12, 3.47, 1.81, 0.4,
];

/// The `urey_bradley` category on a four-atom chain, and the deck LAMMPS
/// prices it as: `angle_style charmm`, `K = 0`, on the same rows.
fn urey_bradley(reg: &Registry) -> Case {
    let mut ff = field("ub", [0.0, 1.0, 1.0], reg);
    let style = ff
        .def_style("urey_bradley", "harmonic", Params::new())
        .unwrap();
    let mut deck = field("ub", [0.0, 1.0, 1.0], reg);
    let angle = deck.def_style("angle", "charmm", Params::new()).unwrap();
    for (_, name, k_ub, r_ub) in UB_ROWS {
        let ends: Vec<&str> = name.split('-').collect();
        style
            .def_type(
                name,
                &ends,
                Params::from_iter([("k_ub", k_ub), ("r_ub", r_ub)]),
            )
            .unwrap();
        // θ0 is any angle: K = 0 prices none of it.
        angle
            .def_type(
                name,
                &ends,
                Params::from_iter([
                    ("k", 0.0),
                    ("theta0", 109.5),
                    ("k_ub", k_ub),
                    ("r_ub", r_ub),
                ]),
            )
            .unwrap();
    }
    let rows: Vec<(&[u32], &str)> = UB_ROWS.iter().map(|r| (&r.0[..], r.1)).collect();
    let types = ["A", "B", "B", "A"];
    Case {
        name: "urey_bradley",
        frame: frame(&types, &[("urey_bradleys", &rows)]),
        deck_frame: frame(&types, &[("angles", &rows)]),
        ff,
        deck,
        configs: vec![CHAIN.to_vec(), shifted(&CHAIN, 0.09)],
        extra: "",
    }
}

/// `(n1, n2, r1, r2, theta0)` of the bond-angle rows of the chain.
const BA_ROWS: [([u32; 3], &str, [f64; 5]); 2] = [
    ([0, 1, 2], "A-B-B", [10.0, 8.0, 1.5, 1.45, 105.0]),
    ([1, 2, 3], "B-B-A", [-6.0, 12.0, 1.4, 1.5, 112.0]),
];

/// The `bond_angle` category on the chain, and its LAMMPS spelling:
/// `angle_style class2` with `K2 = K3 = K4 = 0` and every cross term but
/// `ba` zero — which needs a LAMMPS built with CLASS2
/// (`MOLRS_LMP_CLASS2`).
fn bond_angle(reg: &Registry) -> Case {
    let mut ff = field("ba", [0.0, 1.0, 1.0], reg);
    let style = ff.def_style("bond_angle", "class2", Params::new()).unwrap();
    let mut deck = field("ba", [0.0, 1.0, 1.0], reg);
    let angle = deck.def_style("angle", "class2", Params::new()).unwrap();
    for (_, name, [n1, n2, r1, r2, theta0]) in BA_ROWS {
        let ends: Vec<&str> = name.split('-').collect();
        style
            .def_type(
                name,
                &ends,
                Params::from_iter([
                    ("n1", n1),
                    ("n2", n2),
                    ("r1", r1),
                    ("r2", r2),
                    ("theta0", theta0),
                ]),
            )
            .unwrap();
        angle
            .def_type(
                name,
                &ends,
                Params::from_iter([("theta0", theta0), ("k2", 0.0), ("k3", 0.0), ("k4", 0.0)]),
            )
            .unwrap();
    }
    let rows: Vec<(&[u32], &str)> = BA_ROWS.iter().map(|r| (&r.0[..], r.1)).collect();
    let types = ["A", "B", "B", "A"];
    Case {
        name: "bond_angle",
        frame: frame(&types, &[("bond_angles", &rows)]),
        deck_frame: frame(&types, &[("angles", &rows)]),
        ff,
        deck,
        configs: vec![CHAIN.to_vec(), shifted(&CHAIN, 0.09)],
        extra: "angle_coeff A-B-B ba 10 8 1.5 1.45\nangle_coeff B-B-A ba -6 12 1.4 1.5\n",
    }
}

fn cases(reg: &Registry) -> Vec<Case> {
    vec![smooth(reg), fene(reg), urey_bradley(reg), bond_angle(reg)]
}

/// Energy and forces of `ff` over `frame` at `x`, compiled in `reg`.
fn price(ff: &ForceField, reg: &Registry, frame: &Frame, x: &[f64]) -> (f64, Vec<f64>) {
    let pots = PotentialCompiler::new(&against(ff, reg))
        .compile(frame)
        .unwrap_or_else(|e| panic!("{}: {e}", ff.name));
    let mut acc = Accumulator::new(x.len());
    pots.evaluate(x, &mut acc);
    (acc.energy(), acc.into_forces())
}

/// `ff`, read against `reg` instead of the registry it was built in.
fn against(ff: &ForceField, reg: &Registry) -> ForceField {
    ff.clone().with_registry(Arc::new(reg.clone()))
}

// ---------------------------------------------------------------------------
// LAMMPS: the decks, and the pinned numbers
// ---------------------------------------------------------------------------

/// `frame` at `x`, in a box past every atom (LAMMPS `boundary f f f`).
fn placed(frame: &Frame, x: &[f64]) -> Frame {
    let mut out = frame.clone();
    let atoms = out.get_mut("atoms").unwrap();
    for (d, key) in ["x", "y", "z"].iter().enumerate() {
        let col: Vec<f64> = x.iter().skip(d).step_by(3).copied().collect();
        atoms
            .insert(*key, Array1::from_vec(col).into_dyn(), None)
            .unwrap();
    }
    out.remove("pairs");
    out.simbox = Some(SimBox::new(Cell::Lengths([100.0; 3]), [-50.0; 3], [false; 3]).unwrap());
    out
}

/// The case's LAMMPS deck, through molrs's LAMMPS writer (`units` line
/// apart: it goes before `read_data`).
fn deck(case: &Case, reg: &Arc<Registry>) -> (String, String) {
    let labels = TypeLabels::from_frame(&case.deck_frame).unwrap();
    let mut buffer = Vec::new();
    LammpsForcefieldWriter::new(&labels)
        .with_precision(17)
        .write(&mut buffer, &against(&case.deck, reg))
        .unwrap_or_else(|e| panic!("{}: {e}", case.name));
    let text = String::from_utf8(buffer).unwrap();
    let (pre, post): (Vec<&str>, Vec<&str>) = text.lines().partition(|l| l.starts_with("units"));
    (pre.join("\n") + "\n", post.join("\n") + "\n" + case.extra)
}

/// With `MOLRS_FFEXT_DIR` set, write every case's LAMMPS inputs to
/// `$MOLRS_FFEXT_DIR/rust/<case>/` (`pre.lmp`, `system.ff`, `data_<k>.lmp`)
/// and molrs's numbers to `molrs.tsv` there; without it, nothing.
#[test]
fn write_lammps_inputs() {
    let Some(dir) = std::env::var_os("MOLRS_FFEXT_DIR") else {
        return;
    };
    let reg = Arc::new(ext::registry());
    for case in cases(&reg) {
        let root = Path::new(&dir).join("rust").join(case.name);
        std::fs::create_dir_all(&root).unwrap();
        let (pre, post) = deck(&case, &reg);
        std::fs::write(root.join("pre.lmp"), pre).unwrap();
        std::fs::write(root.join("system.ff"), post).unwrap();
        let mut tsv = String::new();
        for (k, x) in case.configs.iter().enumerate() {
            write_lammps_data(
                &root.join(format!("data_{k}.lmp")),
                &placed(&case.deck_frame, x),
            )
            .unwrap();
            let (e, f) = price(&case.ff, &reg, &case.frame, x);
            tsv.push_str(&format!("{}\t{k}\tpe\t{e:?}\n", case.name));
            for (atom, f) in f.chunks(3).enumerate() {
                tsv.push_str(&format!(
                    "{}\t{k}\tf\t{atom}\t{:?}\t{:?}\t{:?}\n",
                    case.name, f[0], f[1], f[2]
                ));
            }
        }
        std::fs::write(root.join("molrs.tsv"), tsv).unwrap();
    }
}

/// LAMMPS's energy and forces of one configuration.
#[derive(Default)]
struct Pinned {
    pe: f64,
    forces: Vec<[f64; 3]>,
}

/// `(case, config)` → LAMMPS's numbers, from `tests/lammps.tsv`.
fn pinned() -> BTreeMap<(String, usize), Pinned> {
    let mut out: BTreeMap<(String, usize), Pinned> = BTreeMap::new();
    for line in include_str!("lammps.tsv").lines() {
        if line.starts_with('#') || line.trim().is_empty() {
            continue;
        }
        let c: Vec<&str> = line.split('\t').collect();
        let entry = out
            .entry((c[0].to_owned(), c[1].parse().unwrap()))
            .or_default();
        match c[2] {
            "pe" => entry.pe = c[3].parse().unwrap(),
            "f" => {
                let atom: usize = c[3].parse().unwrap();
                assert_eq!(atom, entry.forces.len(), "{line}");
                entry
                    .forces
                    .push([c[4], c[5], c[6]].map(|v| v.parse().unwrap()));
            }
            other => panic!("pinned line of kind {other:?}: {line}"),
        }
    }
    out
}

/// The largest relative error of molrs's energy and forces against the
/// pinned LAMMPS numbers of `case`, every configuration.
fn worst_against_lammps(case: &Case, reg: &Registry) -> f64 {
    let table = pinned();
    let mut worst: f64 = 0.0;
    for (k, x) in case.configs.iter().enumerate() {
        let lmp = table
            .get(&(case.name.to_owned(), k))
            .unwrap_or_else(|| panic!("{} config {k}: no pinned LAMMPS numbers", case.name));
        let (e, f) = price(&case.ff, reg, &case.frame, x);
        let scale = lmp
            .forces
            .iter()
            .flatten()
            .fold(0.0_f64, |m, v| m.max(v.abs()));
        let rel_e = (e - lmp.pe).abs() / lmp.pe.abs().max(f64::MIN_POSITIVE);
        assert!(
            rel_e <= 1e-10,
            "{} config {k}: energy molrs {e:.17e}, LAMMPS {:.17e}, rel {rel_e:.1e}",
            case.name,
            lmp.pe
        );
        worst = worst.max(rel_e);
        assert_eq!(f.len(), 3 * lmp.forces.len(), "{}: atoms", case.name);
        for (c, (got, want)) in f.iter().zip(lmp.forces.iter().flatten()).enumerate() {
            let rel = (got - want).abs() / want.abs().max(scale);
            assert!(
                rel <= 1e-10,
                "{} config {k}: force {c} molrs {got:.17e}, LAMMPS {want:.17e}, rel {rel:.1e}",
                case.name
            );
            worst = worst.max(rel);
        }
    }
    println!(
        "{}: worst relative error against LAMMPS {worst:.1e}",
        case.name
    );
    worst
}

fn case(name: &str, reg: &Registry) -> Case {
    cases(reg).into_iter().find(|c| c.name == name).unwrap()
}

// ---------------------------------------------------------------------------
// The proof
// ---------------------------------------------------------------------------

/// A pair style written by a third party as a `ScalarForm` prices as
/// LAMMPS's `lj/smooth/linear`, at both compile doors.
#[test]
fn pair_style_matches_lammps() {
    let reg = ext::registry();
    let case = case("smooth", &reg);
    worst_against_lammps(&case, &reg);

    // The neighbour-driven door, over a table of every pair, each weighted
    // as MD weights it, prices what the pair-list door prices.
    let topo = molrs::core::Topology::from_frame(&case.frame).unwrap();
    let mut worst: f64 = 0.0;
    for x in &case.configs {
        let (e, f) = price(&case.ff, &reg, &case.frame, x);
        let n = x.len() / 3;
        let links: Vec<(usize, usize)> = (0..n)
            .flat_map(|i| (i + 1..n).map(move |j| (i, j)))
            .collect();
        let r: Vec<f64> = links
            .iter()
            .map(|&(i, j)| {
                (0..3)
                    .map(|a| (x[3 * j + a] - x[3 * i + a]).powi(2))
                    .sum::<f64>()
                    .sqrt()
            })
            .collect();
        assert!(
            r.iter().any(|&r| r < SMOOTH_CUTOFF) && r.iter().any(|&r| r >= SMOOTH_CUTOFF),
            "the cutoff must straddle the pairs: {r:?}"
        );
        let table = Neighbors::from_pairs(
            links.iter().map(|&(i, j)| {
                let d = [0, 1, 2].map(|a| x[3 * j + a] - x[3 * i + a]);
                NeighborPair {
                    i: i as u32,
                    j: j as u32,
                    dist_sq: d.iter().map(|v| v * v).sum(),
                    disp: d,
                }
            }),
            NeighborColumns::FULL,
            QueryMode::SelfQuery { n_points: n },
        );
        let mut ft = vec![0.0; x.len()];
        let mut acc = Accumulator::adding_into(&mut ft);
        let members = PotentialCompiler::new(&against(&case.ff, &reg))
            .compile_typed(&case.frame)
            .unwrap();
        assert_eq!(members.len(), 1, "one pair member");
        for (member, weights) in members {
            let (ForceTerm::Pair(p), Some(w)) = (&member, weights) else {
                panic!("a weighted pair member");
            };
            let special = w.special_weights(&topo);
            let factor: Vec<f64> = links.iter().map(|&(i, j)| special.weight(i, j)).collect();
            p.evaluate_pairs(x, &table, &factor, &mut acc);
        }
        let et = acc.energy();
        let scale = f.iter().fold(0.0_f64, |m, v| m.max(v.abs()));
        assert!(
            (et - e).abs() <= 1e-12 * e.abs(),
            "typed {et}, compiled {e}"
        );
        worst = worst.max((et - e).abs() / e.abs());
        for (a, b) in ft.iter().zip(&f) {
            assert!((a - b).abs() <= 1e-12 * scale, "typed {a}, compiled {b}");
            worst = worst.max((a - b).abs() / scale);
        }
    }
    println!("smooth: compile_typed vs compile {worst:.1e}");
}

/// A category nothing in molrs declares, priced by a native compound
/// form, is LAMMPS's Urey–Bradley term (`angle_style charmm`, K = 0).
#[test]
fn new_category_matches_lammps() {
    let reg = ext::registry();
    let spec = reg.category("urey_bradley").unwrap();
    assert_eq!(
        (spec.arity, spec.block.as_ref()),
        (Arity::Exact(3), "urey_bradleys")
    );
    worst_against_lammps(&case("urey_bradley", &reg), &reg);
}

/// `bond fene`, by its expression alone, is LAMMPS's `bond_style fene` —
/// and the deck that says so was written from its spec, positionally.
#[test]
fn fene_matches_lammps() {
    let reg = ext::registry();
    let case = case("fene", &reg);
    for x in &case.configs {
        for b in [[0, 1], [1, 2], [2, 3], [3, 4]] {
            let r2: f64 = (0..3)
                .map(|a| (x[3 * b[1] + a] - x[3 * b[0] + a]).powi(2))
                .sum();
            assert!(r2 / 1.5_f64.powi(2) < 0.9, "(r/R0)² < 0.9");
        }
    }
    let (_, post) = deck(&case, &Arc::new(ext::registry()));
    assert!(post.contains("bond_style fene\n"), "{post}");
    // `K R0 epsilon sigma`, LAMMPS's order, from the spec's `params`.
    let coeffs: Vec<f64> = post
        .lines()
        .find_map(|l| l.strip_prefix("bond_coeff A-A "))
        .unwrap_or_else(|| panic!("{post}"))
        .split_whitespace()
        .map(|v| v.parse().unwrap())
        .collect();
    assert_eq!(coeffs, [30.0, 1.5, 1.0, 1.0]);
    worst_against_lammps(&case, &reg);
}

/// What a style persists as, and what a registry that never saw this
/// crate makes of it.
#[test]
fn mrec_round_trip() {
    let reg = ext::registry();
    let mut ff = field("everything", [0.0, 1.0, 1.0], &reg);
    for case in [
        smooth(&reg),
        fene(&reg),
        urey_bradley(&reg),
        bond_angle(&reg),
    ] {
        for style in case.ff.styles().iter().filter(|s| s.category() != "atom") {
            let s = ff
                .def_style(style.category(), style.name(), style.params().clone())
                .unwrap();
            for (name, ends, p) in style.type_rows() {
                s.def_type(name, &ends, p.clone()).unwrap();
            }
        }
    }
    // A chain A-B-B-A: its bonds fene, its 1-3 terms both new categories,
    // its 1-3 and 1-4 pairs lj/smooth/linear.
    let bonds: &[(&[u32], &str)] = &[(&[0, 1], "A-B"), (&[1, 2], "B-B"), (&[2, 3], "A-B")];
    let ub: Vec<(&[u32], &str)> = UB_ROWS.iter().map(|r| (&r.0[..], r.1)).collect();
    let ba: Vec<(&[u32], &str)> = BA_ROWS.iter().map(|r| (&r.0[..], r.1)).collect();
    let blocks = |names: &[&str]| -> Frame {
        let all: [Relation<'_>; 3] = [
            ("bonds", bonds),
            ("urey_bradleys", &ub),
            ("bond_angles", &ba),
        ];
        let kept: Vec<Relation<'_>> = all.into_iter().filter(|(b, _)| names.contains(b)).collect();
        let f = frame(&["A", "B", "B", "A"], &kept);
        if names.contains(&"pairs") {
            with_pairs(&ff, f)
        } else {
            f
        }
    };
    let whole = blocks(&["bonds", "urey_bradleys", "bond_angles", "pairs"]);
    let x = CHAIN.to_vec();
    let before = price(&ff, &reg, &whole, &x);
    assert!(before.0.abs() > 1e-3);

    let dir = std::env::temp_dir().join(format!("molrs-ext-{}", std::process::id()));
    std::fs::create_dir_all(&dir).unwrap();
    let path = dir.join("everything.mrec");
    let mut record = MolRec::new();
    record.forcefield = Some(ForceFieldSection::from_forcefield(&ff).unwrap());
    write_mrec(&path, &record).unwrap();
    let read = MrecFile::open(&path)
        .unwrap()
        .forcefield()
        .unwrap()
        .unwrap();
    std::fs::remove_dir_all(&dir).unwrap();

    // The expressions, byte for byte: written from the registry, since the
    // instances state none.
    let expression = |category: &str, style: &str| -> Option<String> {
        read.document["styles"]
            .as_array()
            .unwrap()
            .iter()
            .find(|s| s["category"] == category && s["style"] == style)
            .unwrap_or_else(|| panic!("{category} {style} in the record"))
            .get("expression")
            .and_then(|e| e.as_str())
            .map(str::to_owned)
    };
    assert_eq!(expression("bond", "fene").as_deref(), Some(ext::FENE));
    assert_eq!(
        expression("bond_angle", "class2").as_deref(),
        Some(ext::BOND_ANGLE)
    );
    assert_eq!(expression("urey_bradley", "harmonic"), None);
    assert_eq!(expression("pair", "lj/smooth/linear"), None);

    // Read back into this crate's registry: the same bits.
    let back = read.to_forcefield().unwrap();
    let after = price(&back, &reg, &whole, &x);
    assert_eq!(before.0.to_bits(), after.0.to_bits(), "energy");
    assert!(
        before
            .1
            .iter()
            .zip(&after.1)
            .all(|(a, b)| a.to_bits() == b.to_bits()),
        "forces"
    );

    // A registry that never saw this crate. A style is priced only where
    // its block has rows, so each frame below holds one style's.
    let builtin = Registry::builtin();
    let fene_only = blocks(&["bonds"]);
    let here = price(&ff, &reg, &fene_only, &x);
    let there = price(&back, &builtin, &fene_only, &x);
    assert_eq!(here.0.to_bits(), there.0.to_bits(), "fene: energy");
    assert_eq!(
        here.1.iter().map(|v| v.to_bits()).collect::<Vec<_>>(),
        there.1.iter().map(|v| v.to_bits()).collect::<Vec<_>>(),
        "fene: forces"
    );
    // `bond_angle` was priced by its native form here; there by its
    // expression, which the registry held to agree with it.
    let ba_only = blocks(&["bond_angles"]);
    let (e_native, _) = price(&ff, &reg, &ba_only, &x);
    let (e_expr, _) = price(&back, &builtin, &ba_only, &x);
    assert!((e_native - e_expr).abs() <= 1e-10 * e_native.abs());
    // The native-only styles: kept, and refused by name.
    for (only, category, style) in [
        (blocks(&["urey_bradleys"]), "urey_bradley", "harmonic"),
        (blocks(&["pairs"]), "pair", "lj/smooth/linear"),
    ] {
        assert!(back.get_style(category, style).is_some());
        let err = PotentialCompiler::new(&against(&back, &builtin))
            .compile(&only)
            .unwrap_err();
        assert_eq!(
            err,
            CompileError::Ir(IrError::NoKernel {
                category: category.into(),
                style: style.into()
            })
        );
        assert!(
            err.to_string().contains(
                "register it (molrs.ff.style_registry.register_style) or give it an expression"
            ),
            "{err}"
        );
    }
}

/// A scalar form whose derivative is off by `wrong` and energy by `scale`.
struct Bent {
    scale: f64,
    wrong: f64,
}

impl ScalarForm for Bent {
    fn eval(&self, r: &[f64], p: &ParamColumns<'_>, e: &mut [f64], de: &mut [f64]) {
        let (k, r0) = (p.get("k").unwrap(), p.get("r0").unwrap());
        for t in 0..r.len() {
            e[t] = self.scale * k[t] * (r[t] - r0[t]).powi(2);
            de[t] = self.wrong * 2.0 * k[t] * (r[t] - r0[t]);
        }
    }
}

fn quadratic(name: &'static str) -> StyleSpec {
    StyleSpec::new("bond", name)
        .params(vec![
            ParamSpec::new("k", "E/L^2".parse().unwrap()),
            ParamSpec::new("r0", ParamDimension::LENGTH),
        ])
        .sample(ConformanceSample {
            params: vec![
                ("k".into(), ParamValue::Num(300.0)),
                ("r0".into(), ParamValue::Num(1.5)),
            ],
            q: (1.2, 1.8),
        })
}

/// Every refusal of the protocol a Rust caller can meet, each by its
/// variant and the item it names. (`KernelShape` is the Python kernels':
/// a Rust form writes into slices the kernel sized, so a wrong shape
/// cannot be returned — `molrs-python/tests/test_ff_ir_extension.py`
/// asserts it.)
#[test]
fn nonconforming_refused() {
    let mut r = ext::registry();
    let s = |name: &'static str| StyleSpec::new("bond", name);
    let p = |name: &'static str| ParamSpec::new(name, ParamDimension::ENERGY);
    let expr = |name: &'static str, e: &str| s(name).params(vec![p("k")]).expression(e);

    // ---- categories
    let err = r
        .register_style(StyleSpec::new("ghost", "x").expression("1"), None)
        .unwrap_err();
    assert_eq!(
        err,
        IrError::UnknownCategory {
            category: "ghost".into()
        }
    );
    let err = r
        .register_category(CategorySpec::custom(
            "Bad-Name",
            3,
            Coordinate::Compound,
            EndpointOrder::Reversible,
        ))
        .unwrap_err();
    assert!(
        matches!(&err, IrError::BadName { name, .. } if name == "Bad-Name"),
        "{err:?}"
    );
    let err = r
        .register_category(CategorySpec::custom(
            "sextet",
            6,
            Coordinate::Compound,
            EndpointOrder::Reversible,
        ))
        .unwrap_err();
    assert_eq!(
        err,
        IrError::Arity {
            category: "sextet".into(),
            arity: 6
        }
    );
    let err = r
        .register_category(CategorySpec::new(
            "spring",
            Arity::Exact(2),
            "spring_rows",
            Coordinate::Distance,
            EndpointOrder::Reversible,
        ))
        .unwrap_err();
    assert_eq!(
        err,
        IrError::BlockName {
            category: "spring".into(),
            block: "spring_rows".into()
        }
    );
    // A type with the wrong number of endpoints for its category.
    let mut ff = ForceField::new("t").with_registry(Arc::new(r.clone()));
    let err: DefError = ff
        .def_style("urey_bradley", "harmonic", Params::new())
        .unwrap()
        .def_type("A-B", &["A", "B"], Params::new())
        .unwrap_err();
    assert!(
        matches!(err.ir(), Some(IrError::Arity { category, arity: 2 }) if category == "urey_bradley"),
        "{err:?}"
    );

    // ---- parameters
    let err = r
        .register_style(expr("x", "theta").params(vec![p("theta")]), None)
        .unwrap_err();
    assert_eq!(
        err,
        IrError::ReservedParam {
            style: "x".into(),
            param: "theta".into()
        }
    );
    let err = r
        .register_style(s("x").params(vec![p("k"), p("k")]).expression("k*r"), None)
        .unwrap_err();
    assert_eq!(
        err,
        IrError::DuplicateParam {
            style: "x".into(),
            param: "k".into()
        }
    );
    let err = r
        .register_style(
            s("x")
                .params(vec![ParamSpec::new("k", "E*A".parse().unwrap())])
                .expression("k*r"),
            None,
        )
        .unwrap_err();
    assert!(
        matches!(&err, IrError::Dimension { param, .. } if param == "k"),
        "{err:?}"
    );

    // ---- expressions
    let err = r.register_style(expr("x", "k*(r-"), None).unwrap_err();
    assert!(
        matches!(&err, IrError::Parse { expression, .. } if expression == "k*(r-"),
        "{err:?}"
    );
    let err = r.register_style(expr("x", "k*theta"), None).unwrap_err();
    assert_eq!(
        err,
        IrError::UnboundVariable {
            style: "x".into(),
            name: "theta".into()
        }
    );
    let err = r.register_style(expr("x", "k*sinh(r)"), None).unwrap_err();
    assert_eq!(
        err,
        IrError::UnknownFunction {
            name: "sinh".into()
        }
    );
    let err = r.register_style(expr("x", "k*min(r)"), None).unwrap_err();
    assert_eq!(
        err,
        IrError::FunctionArity {
            name: "min".into(),
            given: 1,
            expected: 2
        }
    );
    let err = r
        .register_style(expr("x", "k*distance(p1,p3)"), None)
        .unwrap_err();
    assert_eq!(
        err,
        IrError::Point {
            style: "x".into(),
            point: "p3".into(),
            arity: 2
        }
    );

    // ---- kernels
    let err = r
        .register_style(
            StyleSpec::new("urey_bradley", "scalar").params(vec![p("k")]),
            Some(Kernel::Scalar(Arc::new(ext::LjSmoothLinear))),
        )
        .unwrap_err();
    assert!(
        matches!(&err, IrError::CoordinateMismatch { category, .. } if category == "urey_bradley"),
        "{err:?}"
    );
    let bent = |scale, wrong| Some(Kernel::Scalar(Arc::new(Bent { scale, wrong })));
    let err = r
        .register_style(quadratic("half"), bent(1.0, 0.5))
        .unwrap_err();
    assert!(
        matches!(&err, IrError::Derivative { style, .. } if style == "half"),
        "{err:?}"
    );
    let err = r
        .register_style(
            quadratic("off").expression("k*(r-r0)^2"),
            bent(1.0 + 1e-8, 1.0 + 1e-8),
        )
        .unwrap_err();
    assert!(
        matches!(&err, IrError::Disagree { style, .. } if style == "off"),
        "{err:?}"
    );
    let lopsided = StyleSpec::new("pair", "lopsided")
        .params(vec![p("a")])
        .style_params(vec![ParamSpec::new("cutoff", ParamDimension::LENGTH)])
        .expression("a1*exp(-r)")
        .sample(ConformanceSample {
            params: vec![
                ("a".into(), ParamValue::Num(1.0)),
                ("cutoff".into(), ParamValue::Num(9.0)),
            ],
            q: (1.0, 5.0),
        });
    let err = r.register_style(lopsided, None).unwrap_err();
    assert_eq!(
        err,
        IrError::Asymmetric {
            style: "lopsided".into()
        }
    );
    let err = r
        .register_style(s("bare").params(vec![p("k")]), None)
        .unwrap_err();
    assert_eq!(
        err,
        IrError::NoKernel {
            category: "bond".into(),
            style: "bare".into()
        }
    );

    // ---- the registry's own rules
    let mut harmonic = r.style("bond", "harmonic").unwrap().0.clone();
    harmonic.expression = Some("2*k*(r-r0)^2".into());
    assert_eq!(
        r.register_style(harmonic, None).unwrap_err(),
        IrError::Sealed {
            category: "bond".into(),
            style: "harmonic".into()
        }
    );
    assert_eq!(
        r.unregister_style("bond", "harmonic").unwrap_err(),
        IrError::Sealed {
            category: "bond".into(),
            style: "harmonic".into()
        }
    );
    let mut other = ext::fene();
    other.expression = Some("k*(r-r0)^2".into());
    assert_eq!(
        r.register_style(other, None).unwrap_err(),
        IrError::Conflict {
            category: "bond".into(),
            style: "fene".into()
        }
    );
    r.register_style(ext::fene(), None)
        .expect("an identical re-registration is a no-op");

    // ---- compile
    r.register_style(
        StyleSpec::new("pair", "soft")
            .params(vec![p("a")])
            .style_params(vec![ParamSpec::new("cutoff", ParamDimension::LENGTH)])
            .expression("a*exp(-r)"),
        None,
    )
    .unwrap();
    let compile = |ff: &ForceField, frame: &Frame| {
        PotentialCompiler::new(&against(ff, &r))
            .compile(frame)
            .map(|_| ())
            .unwrap_err()
    };
    let one_bond = frame(&["A", "B"], &[("bonds", &[(&[0, 1], "A-B")])]);
    let mut ff = field("t", [0.0, 1.0, 1.0], &r);
    ff.def_style("bond", "mystery", Params::new())
        .unwrap()
        .def_type("A-B", &["A", "B"], Params::from_iter([("k", 1.0)]))
        .unwrap();
    assert_eq!(
        compile(&ff, &one_bond),
        CompileError::Ir(IrError::NoKernel {
            category: "bond".into(),
            style: "mystery".into()
        })
    );
    let mut ff = field("t", [0.0, 1.0, 1.0], &r);
    ff.def_style("bond", "fene", Params::new())
        .unwrap()
        .def_type(
            "A-B",
            &["A", "B"],
            Params::from_iter([("k", 30.0), ("r0", 1.5), ("epsilon", 1.0)]),
        )
        .unwrap();
    assert_eq!(
        compile(&ff, &one_bond),
        CompileError::Ir(IrError::MissingParam {
            style: "fene".into(),
            type_: "A-B".into(),
            param: "sigma".into()
        })
    );
    // A parameter that does not mix, an unlike pair, no cross row.
    let mut ff = field("t", [0.0; 3], &r);
    let soft = ff
        .def_style("pair", "soft", Params::from_iter([("cutoff", 9.0)]))
        .unwrap();
    for t in ["A", "B"] {
        soft.def_type(t, &[t], Params::from_iter([("a", 1.0)]))
            .unwrap();
    }
    let unbonded = with_pairs(&ff, frame(&["A", "B"], &[]));
    assert!(matches!(
        compile(&ff, &unbonded),
        CompileError::Ir(IrError::NoMixing { style, param, .. }) if style == "soft" && param == "a"
    ));
    // A text value outside its declared choices.
    let mut ff = field("t", [0.0; 3], &r);
    let mut style = Params::from_iter([("cutoff", 8.0)]);
    style.set("mixing", "median");
    let lj = ff.def_style("pair", "lj/smooth/linear", style).unwrap();
    for t in ["A", "B"] {
        lj.def_type(
            t,
            &[t],
            Params::from_iter([("epsilon", 0.2), ("sigma", 3.1)]),
        )
        .unwrap();
    }
    let unbonded = with_pairs(&ff, frame(&["A", "B"], &[]));
    let err = compile(&ff, &unbonded);
    assert!(
        matches!(&err, CompileError::Ir(IrError::BadValue { style, param, .. })
            if style == "lj/smooth/linear" && param == "mixing"),
        "{err:?}"
    );

    // ---- engines
    let err = r
        .register_engine_form(Engine::Gromacs, "bond", "fene", LammpsForm::positional())
        .unwrap_err();
    assert!(
        matches!(&err, IrError::NoEngineForm { engine, style, .. } if engine == "GROMACS" && style == "fene"),
        "{err:?}"
    );
    let fene = case("fene", &r);
    let err = GromacsTopForcefieldWriter::new()
        .write(&mut Vec::new(), &fene.ff)
        .unwrap_err();
    assert!(
        matches!(err.ir(), Some(IrError::NoEngineForm { engine, category, style, .. })
            if engine == "GROMACS" && category == "bond" && style == "fene"),
        "{err}"
    );
    let ub = case("urey_bradley", &r);
    let labels = TypeLabels::from_frame(&ub.frame).unwrap();
    let err = LammpsForcefieldWriter::new(&labels)
        .write(&mut Vec::new(), &ub.ff)
        .unwrap_err();
    assert!(
        matches!(err.ir(), Some(IrError::NoEngineForm { engine, category, .. })
            if engine == "LAMMPS" && category == "urey_bradley"),
        "{err}"
    );

    // ---- forms
    let same = |tp: &molrs::ff::ir::TypeParams| -> Result<_, FormRefusal> { Ok(tp.clone()) };
    let err = r
        .register_form(
            "bond",
            "fene",
            FormCodec::new("torsion", same, same).as_canonical(),
        )
        .unwrap_err();
    assert!(
        matches!(&err, IrError::FormConflict { family, .. } if family == "torsion"),
        "{err:?}"
    );
    let err = fene.ff.to_form("bond", "fene").unwrap_err();
    assert_eq!(
        err,
        IrError::NoForm {
            category: "bond".into(),
            style: "fene".into()
        }
    );

    // Nothing was left behind by a refusal.
    for (category, style) in [
        ("bond", "x"),
        ("bond", "half"),
        ("bond", "off"),
        ("pair", "lopsided"),
    ] {
        assert!(r.style(category, style).is_none(), "{category} {style}");
    }
}

/// LAMMPS class2's bond-angle cross term as a new category: its
/// expression and the native form registered beside it are one energy —
/// the hand value, each other, and the central difference of the energy.
#[test]
fn bond_angle_cross_term() {
    let reg = ext::registry();
    let by_expression = {
        let mut r = Registry::builtin();
        r.register_category(ext::bond_angle_category()).unwrap();
        r.register_style(ext::bond_angle_class2(), None).unwrap();
        r
    };
    // The hand value: the middle atom at the origin, the arms 1.6 and 1.4
    // Å at 100°, (n1, n2, r1, r2, θ0) = (10, 8, 1.5, 1.45, 105°):
    // E = (10·0.1 − 8·0.05)·(−5°) = −π/60.
    let theta = 100.0_f64.to_radians();
    let hand = [
        1.6,
        0.0,
        0.0,
        0.0,
        0.0,
        0.0,
        1.4 * theta.cos(),
        1.4 * theta.sin(),
        0.0,
    ];
    let one = |x: &[f64], reg: &Registry| {
        let mut ff = field("ba", [0.0, 1.0, 1.0], reg);
        ff.def_style("bond_angle", "class2", Params::new())
            .unwrap()
            .def_type(
                "A-B-B",
                &["A", "B", "B"],
                Params::from_iter([
                    ("n1", 10.0),
                    ("n2", 8.0),
                    ("r1", 1.5),
                    ("r2", 1.45),
                    ("theta0", 105.0),
                ]),
            )
            .unwrap();
        let rows: &[(&[u32], &str)] = &[(&[0, 1, 2], "A-B-B")];
        price(
            &ff,
            reg,
            &frame(&["A", "B", "B"], &[("bond_angles", rows)]),
            x,
        )
    };
    let want = -std::f64::consts::PI / 60.0;
    for r in [&reg, &by_expression] {
        let (e, _) = one(&hand, r);
        assert!((e - want).abs() <= 1e-12 * want.abs(), "{e} vs {want}");
    }
    // Expression = native form, energy and forces, on the chain's rows.
    let case = case("bond_angle", &reg);
    let mut worst: f64 = 0.0;
    for x in &case.configs {
        let (e_n, f_n) = price(&case.ff, &reg, &case.frame, x);
        let (e_x, f_x) = price(&case.ff, &by_expression, &case.frame, x);
        let scale = f_n.iter().fold(e_n.abs(), |m, v| m.max(v.abs()));
        worst = worst.max((e_n - e_x).abs() / e_n.abs());
        for (a, b) in f_n.iter().zip(&f_x) {
            worst = worst.max((a - b).abs() / scale);
        }
        // The force is minus the central difference of the energy.
        let h = 1e-6;
        for k in 0..x.len() {
            let (mut up, mut down) = (x.clone(), x.clone());
            up[k] += h;
            down[k] -= h;
            let fd = -(price(&case.ff, &reg, &case.frame, &up).0
                - price(&case.ff, &reg, &case.frame, &down).0)
                / (2.0 * h);
            assert!(
                (fd - f_n[k]).abs() <= 1e-7 * scale,
                "component {k}: {fd} vs {}",
                f_n[k]
            );
        }
    }
    assert!(worst <= 1e-12, "expression vs native form: {worst:.1e}");
    println!("bond_angle: expression vs native form {worst:.1e}");
}
