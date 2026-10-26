//! 1-4 interactions, end to end: `lj/charmm` / `coul/charmm`, `dihedral
//! charmm` `w` and per-pair overrides on `pairs`, held against each other.
//!
//! The molecule is a seven-atom alcohol (`CT3 CT2 CT2 CT2 OH1 H`, a `CT3`
//! branch on the third atom) with charges; the force field reads with
//! `lj/charmm/coul/charmm 3.5 4.2 3.0 5.0` so that pairs fall inside, across
//! and beyond both switches; two more cases cut short of the 1-4 pairs
//! (`lj/charmm/coul/charmm 2.0 2.4` with `w` = 1, `lj/cut/coul/cut 3.0`
//! shifted with `special_bonds` ½ / ⅚): a `special_bonds` 1-4 pair is
//! truncated by its pair style, a `w` pair at no distance.

use crate::ff::potential::fixtures::OneShot;
use crate::io::writer::WrittenText;
use ndarray::Array1;

use crate::ff::compile::PotentialCompiler;
use crate::ff::compile::exceptions;
use crate::ff::forcefield::ForceField;
use crate::ff::special_pairs::SpecialPairs;
use crate::io::lammps::forcefield_reader::LammpsForcefieldReader;
use crate::io::reader::ForceFieldReader;
use molrs::core::Block;
use molrs::core::Frame;
use molrs::op::{F, Idx};

const TYPES: [&str; 7] = ["CT3", "CT2", "CT2", "CT2", "OH1", "H", "CT3"];
const CHARGES: [F; 7] = [-0.09, 0.03, -0.12, 0.05, -0.66, 0.43, 0.36];
const XYZ: [[F; 3]; 7] = [
    [0.0, 0.008415, 0.009093],
    [1.531411, -0.007568, -0.009589],
    [2.100354, 1.425161, 0.009894],
    [3.634062, 1.391516, 0.190091],
    [4.217611, 0.717029, -0.896879],
    [3.999616, -0.215936, -0.80915],
    [1.74947, 2.124338, -1.304302],
];
const BONDS: [[usize; 2]; 6] = [[0, 1], [1, 2], [2, 3], [3, 4], [4, 5], [6, 2]];
const ANGLES: [[usize; 3]; 6] = [
    [0, 1, 2],
    [1, 2, 3],
    [1, 2, 6],
    [3, 2, 6],
    [2, 3, 4],
    [3, 4, 5],
];
const DIHEDRALS: [[usize; 4]; 5] = [
    [0, 1, 2, 3],
    [0, 1, 2, 6],
    [1, 2, 3, 4],
    [6, 2, 3, 4],
    [2, 3, 4, 5],
];
const CHARMM: &str = "special_bonds charmm";
const AMBER_LIKE: &str = "special_bonds lj 0.0 0.0 0.5 coul 0.0 0.0 0.8333333333333334";

/// The LAMMPS include of the test, `special` and every dihedral's `w`.
fn ff_text(special: &str, w: &str) -> String {
    format!(
        "{special}
pair_style lj/charmm/coul/charmm 3.5 4.2 3.0 5.0
pair_modify mix arithmetic
pair_coeff CT3 CT3 0.078 3.6705 0.01 3.385
pair_coeff CT2 CT2 0.056 3.5814 0.01 3.385
pair_coeff OH1 OH1 0.1521 3.1508
pair_coeff H H 0.046 0.4
pair_coeff CT3 OH1 0.12 3.3 0.08 3.2
bond_style harmonic
bond_coeff CT3-CT2 222.5 1.528
bond_coeff CT2-CT2 222.5 1.53
bond_coeff CT2-OH1 428.0 1.42
bond_coeff OH1-H 545.0 0.96
angle_style harmonic
angle_coeff CT3-CT2-CT2 58.35 113.6
angle_coeff CT2-CT2-CT2 58.35 113.6
angle_coeff CT2-CT2-CT3 58.35 113.6
angle_coeff CT2-CT2-OH1 75.7 110.1
angle_coeff CT2-OH1-H 57.5 106.0
dihedral_style charmm
dihedral_coeff CT3-CT2-CT2-CT2 0.15 3 0 {w}
dihedral_coeff CT3-CT2-CT2-CT3 0.2 3 0 {w}
dihedral_coeff CT2-CT2-CT2-OH1 0.195 3 0 {w}
dihedral_coeff CT3-CT2-CT2-OH1 0.3 1 180 {w}
dihedral_coeff CT2-CT2-OH1-H 0.14 3 0 {w}
"
    )
}

fn read(special: &str, w: &str) -> ForceField {
    LammpsForcefieldReader::new()
        .read(ff_text(special, w).as_bytes())
        .unwrap()
}

fn label(rows: &[usize]) -> String {
    rows.iter().map(|&i| TYPES[i]).collect::<Vec<_>>().join("-")
}

fn relation(rows: &[Vec<usize>]) -> Block {
    let mut block = Block::new();
    for (k, key) in ["atomi", "atomj", "atomk", "atoml"]
        .iter()
        .take(rows[0].len())
        .enumerate()
    {
        let col: Vec<Idx> = rows.iter().map(|r| r[k] as Idx).collect();
        block
            .insert(*key, Array1::from_vec(col).into_dyn(), None)
            .unwrap();
    }
    let types: Vec<String> = rows.iter().map(|r| label(r)).collect();
    block
        .insert("type", Array1::from_vec(types).into_dyn(), None)
        .unwrap();
    block
}

/// The molecule's frame with its special pairs; `dihedrals` lists the third
/// dihedral twice when `dup`.
fn frame(dup: bool) -> Frame {
    let mut atoms = Block::new();
    for (d, key) in ["x", "y", "z"].iter().enumerate() {
        let col: Vec<F> = XYZ.iter().map(|p| p[d]).collect();
        atoms
            .insert(*key, Array1::from_vec(col).into_dyn(), None)
            .unwrap();
    }
    let types: Vec<String> = TYPES.iter().map(|t| (*t).to_owned()).collect();
    atoms
        .insert("type", Array1::from_vec(types).into_dyn(), None)
        .unwrap();
    atoms
        .insert(
            "charge",
            Array1::from_vec(CHARGES.to_vec()).into_dyn(),
            None,
        )
        .unwrap();
    let mut frame = Frame::new();
    frame.insert("atoms", atoms);
    frame.insert(
        "bonds",
        relation(&BONDS.iter().map(|r| r.to_vec()).collect::<Vec<_>>()),
    );
    frame.insert(
        "angles",
        relation(&ANGLES.iter().map(|r| r.to_vec()).collect::<Vec<_>>()),
    );
    let mut dihedrals: Vec<Vec<usize>> = DIHEDRALS.iter().map(|r| r.to_vec()).collect();
    if dup {
        dihedrals.push(DIHEDRALS[2].to_vec());
    }
    frame.insert("dihedrals", relation(&dihedrals));
    let pairs = molrs::core::Topology::from_frame(&frame)
        .unwrap()
        .pairs_block();
    frame.insert("pairs", pairs);
    frame
}

/// The exceptions plan of `ff` on `frame`'s special pairs.
fn plan_of(ff: &ForceField, frame: &Frame) -> Result<exceptions::Exceptions, String> {
    let pairs = SpecialPairs::of(frame)?.unwrap_or_default();
    exceptions::plan(ff, frame, &pairs, ff.required_special_bonds()?)
}

fn coords() -> Vec<F> {
    XYZ.iter().flatten().copied().collect()
}

/// `ff` with only its `(category, name)` style; a `dihedral charmm` copy
/// has its `w` zeroed (the torsion alone).
fn only(ff: &ForceField, category: &str, name: &str) -> ForceField {
    let mut one = ff.empty_like();
    let style = ff.get_style(category, name).unwrap();
    let s = one
        .def_style(category, name, style.params().clone())
        .unwrap();
    for (type_name, ends, params) in style.type_rows() {
        let mut params = params.clone();
        if params.get("w").and_then(|v| v.as_num()).is_some() {
            params.set("w", 0.0);
        }
        s.def_type(type_name, &ends, params).unwrap();
    }
    one
}

fn energy(ff: &ForceField, frame: &Frame) -> F {
    PotentialCompiler::new(ff)
        .compile(frame)
        .unwrap()
        .energy_at(&coords())
}

fn close(label: &str, got: F, want: F, rel: F) {
    let err = (got - want).abs() / want.abs().max(1e-300);
    assert!(
        err <= rel,
        "{label}: molrs {got:?} vs {want:?} (rel {err:e})"
    );
}

/// The include of [`ff_text`] under `lj/cut/coul/cut 3.0`, shifted, with
/// `special_bonds` ½ / ⅚ and `w` = 0: the cutoff straddles the 1-4 pairs
/// (2.64, 2.87, 2.92 Å inside; 3.04, 3.89 Å beyond) and the rest (a 1-5
/// pair at 2.60 Å inside; 3.28–4.37 Å beyond).
fn cut_text() -> String {
    let charmm = ff_text(AMBER_LIKE, "0.0");
    let bonded = &charmm[charmm.find("bond_style").unwrap()..];
    format!(
        "{AMBER_LIKE}
pair_style lj/cut/coul/cut 3.0
pair_modify mix arithmetic shift yes
pair_coeff CT3 CT3 0.078 3.6705
pair_coeff CT2 CT2 0.056 3.5814
pair_coeff OH1 OH1 0.1521 3.1508
pair_coeff H H 0.046 0.4
pair_coeff CT3 OH1 0.12 3.3
{bonded}"
    )
}

/// LAMMPS refuses `w > 0` beside non-zero 1-4 weights, and so does molrs.
#[test]
fn a_weight_beside_global_1_4_weights_is_refused() {
    let ff = read(AMBER_LIKE, "1.0");
    let err = PotentialCompiler::new(&ff)
        .compile(&frame(false))
        .unwrap_err();
    assert!(err.to_string().contains("special_bonds charmm"), "{err}");
    let err = PotentialCompiler::new(&ff)
        .compile_typed(&frame(false))
        .unwrap_err();
    assert!(err.to_string().contains("special_bonds charmm"), "{err}");
}

/// `w = 1` prices exactly `Σ 4ε₁₄[(σ₁₄/r)¹² − (σ₁₄/r)⁶] + C qᵢqⱼ/r` over the
/// dihedrals' end pairs, `ε₁₄` / `σ₁₄` mixed arithmetically (the `CT3-OH1`
/// pair from its explicit cross row's 1-4 numbers).
#[test]
fn a_weight_of_one_is_the_1_4_parameters_priced_directly() {
    let ff = read(CHARMM, "1.0");
    let frame = frame(false);
    let k = plan_of(&ff, &frame).unwrap().kernel.unwrap();
    let one_four = |t: &str| match t {
        "CT3" | "CT2" => (0.01, 3.385),
        "OH1" => (0.1521, 3.1508),
        _ => (0.046, 0.4),
    };
    let (mut lj, mut coul) = (0.0, 0.0);
    for d in DIHEDRALS {
        let (i, j) = (d[0], d[3]);
        let (a, b) = (TYPES[i], TYPES[j]);
        let (eps, sigma) = if (a, b) == ("CT3", "OH1") || (a, b) == ("OH1", "CT3") {
            (0.08, 3.2)
        } else {
            let ((ea, sa), (eb, sb)) = (one_four(a), one_four(b));
            ((ea * eb as F).sqrt(), 0.5 * (sa + sb))
        };
        let r = (0..3)
            .map(|k| (XYZ[i][k] - XYZ[j][k]).powi(2))
            .sum::<F>()
            .sqrt();
        let x = (sigma / r).powi(6);
        lj += 4.0 * eps * (x * x - x);
        coul += molrs::core::constants::COULOMB_REAL * CHARGES[i] * CHARGES[j] / r;
    }
    let (got_lj, got_coul) = k.energy_terms(&coords());
    close("1-4 LJ", got_lj, lj, 1e-12);
    close("1-4 Coulomb", got_coul, coul, 1e-12);
}

/// A copy of `frame` whose 1-4 `pairs` rows (`n_bonds` 3) carry `cells`
/// (column, value per such row), every other row null.
/// One override column and its value for the pair `(i, j)`.
type Cell<'a> = (&'a str, &'a dyn Fn(usize, usize) -> F);

fn with_overrides(frame: &Frame, cells: &[Cell<'_>]) -> Frame {
    let mut out = frame.clone();
    let mut pairs = frame.get("pairs").unwrap().clone();
    let is_14: Vec<bool> = pairs
        .get("n_bonds")
        .unwrap()
        .get::<u64>()
        .unwrap()
        .iter()
        .map(|&d| d == 3)
        .collect();
    let ai: Vec<usize> = pairs
        .get("atomi")
        .unwrap()
        .get::<u64>()
        .unwrap()
        .iter()
        .map(|&v| v as usize)
        .collect();
    let aj: Vec<usize> = pairs
        .get("atomj")
        .unwrap()
        .get::<u64>()
        .unwrap()
        .iter()
        .map(|&v| v as usize)
        .collect();
    for (key, value) in cells {
        let col: Vec<F> = (0..ai.len())
            .map(|r| if is_14[r] { value(ai[r], aj[r]) } else { 0.0 })
            .collect();
        pairs
            .insert(*key, Array1::from_vec(col).into_dyn(), Some(is_14.clone()))
            .unwrap();
    }
    out.insert("pairs", pairs);
    out
}

/// An `lj/cut/coul/cut` field (no switch, so a 1-4 pair is the same number
/// in the pair kernels and in the exceptions kernel) under `special`.
fn plain(special: &str) -> ForceField {
    let text = ff_text(special, "0.0")
        .replace(
            "pair_style lj/charmm/coul/charmm 3.5 4.2 3.0 5.0",
            "pair_style lj/cut/coul/cut 30.0",
        )
        .replace(" 0.01 3.385\n", "\n")
        .replace("OH1 0.12 3.3 0.08 3.2", "OH1 0.12 3.3");
    LammpsForcefieldReader::new().read(text.as_bytes()).unwrap()
}

/// The three ways to say "half": `special_bonds` ½, per-pair scales ½ on
/// the 1-4 rows (beside global 0), and per-pair explicit parameters (`ε/2`,
/// `qᵢqⱼ/2`) at scale 1 — one energy, one set of forces.
#[test]
fn global_half_equals_per_pair_scales_equals_per_pair_parameters() {
    let half = plain("special_bonds lj 0.0 0.0 0.5 coul 0.0 0.0 0.5");
    let zero = plain(CHARMM);
    let base = frame(false);
    let pots = |ff: &ForceField, frame: &Frame| {
        PotentialCompiler::new(ff)
            .compile(frame)
            .unwrap()
            .energy_forces_at(&coords())
    };
    let want = pots(&half, &base);

    let scaled = with_overrides(
        &base,
        &[("lj_scale", &|_, _| 0.5), ("coul_scale", &|_, _| 0.5)],
    );
    let mixed = |i: usize, j: usize| {
        let p = |t: &str| match t {
            "CT3" => (0.078, 3.6705),
            "CT2" => (0.056, 3.5814),
            "OH1" => (0.1521, 3.1508),
            _ => (0.046, 0.4),
        };
        let (a, b) = (TYPES[i], TYPES[j]);
        if (a, b) == ("CT3", "OH1") || (a, b) == ("OH1", "CT3") {
            return (0.12, 3.3);
        }
        let ((ea, sa), (eb, sb)) = (p(a), p(b));
        ((ea * eb as F).sqrt(), 0.5 * (sa + sb))
    };
    let explicit = with_overrides(
        &base,
        &[
            ("epsilon", &|i, j| 0.5 * mixed(i, j).0),
            ("sigma", &|i, j| mixed(i, j).1),
            ("charge_product", &|i, j| 0.5 * CHARGES[i] * CHARGES[j]),
            ("lj_scale", &|_, _| 1.0),
            ("coul_scale", &|_, _| 1.0),
        ],
    );
    for (label, frame) in [("scales", &scaled), ("parameters", &explicit)] {
        let got = pots(&zero, frame);
        close(label, got.0, want.0, 1e-12);
        for (a, b) in got.1.iter().zip(&want.1) {
            assert!((a - b).abs() < 1e-10, "{label}: force {a} vs {b}");
        }
        // The neighbour-driven door agrees.
        close(
            &format!("{label} typed"),
            typed_energy(&zero, frame).0,
            want.0,
            1e-12,
        );
    }
}

/// `w = 1` under `special_bonds charmm` and per-pair rows stating the same
/// 1-4 parameters (`ε₁₄`, `σ₁₄`, scales 1) beside `w = 0` are one energy.
#[test]
fn a_dihedral_weight_equals_per_pair_rows_of_its_parameters() {
    let with_w = read(CHARMM, "1.0");
    let without = read(CHARMM, "0.0");
    let base = frame(false);
    let want = energy(&with_w, &base);
    let p14 = |i: usize, j: usize| {
        let lj = with_w.get_style("pair", "lj/charmm").unwrap();
        let rows = lj.defs().kernel_type_params().unwrap();
        let refs = rows.iter().map(|(k, v)| (k.as_str(), v)).collect();
        crate::ff::potential::pair::lj_rows::charmm_pair_params(
            &refs,
            crate::ff::ir::CombiningRule::Arithmetic,
            TYPES[i],
            TYPES[j],
        )
        .unwrap()
        .1
    };
    let stated = with_overrides(
        &base,
        &[
            ("epsilon", &|i, j| p14(i, j).0),
            ("sigma", &|i, j| p14(i, j).1),
            ("lj_scale", &|_, _| 1.0),
            ("coul_scale", &|_, _| 1.0),
        ],
    );
    close("rows of eps14", energy(&without, &stated), want, 1e-12);
}

/// Precedence on one pair: a per-pair cell beats the dihedral's `w`, and a
/// null cell keeps it.
#[test]
fn an_override_cell_beats_the_weight_and_a_null_cell_keeps_it() {
    let ff = read(CHARMM, "1.0");
    let base = frame(false);
    let (lj_w, coul_w) = plan_of(&ff, &base)
        .unwrap()
        .kernel
        .unwrap()
        .energy_terms(&coords());
    // The coul_scale cell only: the Coulomb part of every 1-4 pair is
    // quartered, its van der Waals is the dihedral's.
    let quarter = with_overrides(&base, &[("coul_scale", &|_, _| 0.25)]);
    let (lj, coul) = plan_of(&ff, &quarter)
        .unwrap()
        .kernel
        .unwrap()
        .energy_terms(&coords());
    close("LJ kept", lj, lj_w, 1e-14);
    close("Coulomb quartered", coul, 0.25 * coul_w, 1e-13);
    // And the regular kernels price none of those rows.
    let plan = plan_of(&ff, &quarter).unwrap();
    assert_eq!(plan.replaced.len(), 5);
}

/// An override cell is priced only by the style it belongs to: a field
/// without a Lennard-Jones style ignores `epsilon` / `sigma` / `lj_scale`,
/// one without a Coulomb style `charge_product` / `coul_scale`, and a
/// bonded-only field prices no pair of a materialized frame.
#[test]
fn an_override_cell_is_priced_only_by_its_own_style() {
    let ff = plain(AMBER_LIKE);
    let base = frame(false);
    let lj_cells = with_overrides(
        &base,
        &[
            ("epsilon", &|_, _| 0.3),
            ("sigma", &|_, _| 3.1),
            ("lj_scale", &|_, _| 1.0),
        ],
    );
    let coul_cells = with_overrides(
        &base,
        &[
            ("charge_product", &|_, _| -0.2),
            ("coul_scale", &|_, _| 1.0),
        ],
    );
    let all = with_overrides(
        &lj_cells,
        &[
            ("charge_product", &|_, _| -0.2),
            ("coul_scale", &|_, _| 1.0),
        ],
    );
    let bond = only(&ff, "bond", "harmonic");
    let lj = only(&ff, "pair", "lj/cut");
    let coul = only(&ff, "pair", "coul/cut");
    close(
        "bond-only",
        energy(&bond, &all),
        energy(&bond, &base),
        1e-14,
    );
    close(
        "LJ-only",
        energy(&lj, &coul_cells),
        energy(&lj, &base),
        1e-14,
    );
    close(
        "Coulomb-only",
        energy(&coul, &lj_cells),
        energy(&coul, &base),
        1e-14,
    );
    // The cells do change their own style's energy.
    assert!((energy(&lj, &lj_cells) - energy(&lj, &base)).abs() > 1e-6);
    assert!((energy(&coul, &coul_cells) - energy(&coul, &base)).abs() > 1e-6);
}

/// Finite differences of the whole compiled field inside every switch
/// region's start (the Coulomb switch's force is LAMMPS's, not a gradient),
/// with exceptions, at `w = 1`.
#[test]
fn forces_are_the_gradient_inside_the_switches() {
    let text = ff_text(CHARMM, "1.0").replace("3.5 4.2 3.0 5.0", "12.0 14.0");
    let ff = LammpsForcefieldReader::new().read(text.as_bytes()).unwrap();
    let pots = PotentialCompiler::new(&ff).compile(&frame(false)).unwrap();
    let x = coords();
    let (_, f) = pots.energy_forces_at(&x);
    let h = 1e-6;
    for k in 0..x.len() {
        let (mut p, mut m) = (x.clone(), x.clone());
        p[k] += h;
        m[k] -= h;
        let fd = -(pots.energy_at(&p) - pots.energy_at(&m)) / (2.0 * h);
        assert!((f[k] - fd).abs() < 1e-6, "component {k}: {} vs {fd}", f[k]);
    }
}

/// The neighbour-driven door over a table of every pair, each weighted as
/// MD weights it.
fn typed_energy(ff: &ForceField, frame: &Frame) -> (F, Vec<F>) {
    use crate::ff::potential::ForceTerm;
    use crate::ff::potential::pair::fixtures::table_over;
    let x = coords();
    let links: Vec<(usize, usize)> = (0..7)
        .flat_map(|i| ((i + 1)..7).map(move |j| (i, j)))
        .collect();
    let table = table_over(&x, &links);
    let mut out = vec![0.0; x.len()];
    let mut acc = crate::ff::potential::Accumulator::adding_into(&mut out);
    for (member, weights) in PotentialCompiler::new(ff).compile_typed(frame).unwrap() {
        match &member {
            ForceTerm::Pair(p) => {
                let factor: Vec<F> = links.iter().map(|&(i, j)| weights.weight(i, j)).collect();
                p.evaluate_pairs(&x, &table, &factor, &mut acc);
            }
            m => m.as_potential().evaluate(&x, &mut acc),
        }
    }
    let e = acc.energy();
    (e, out)
}

/// Both compile doors price every case the same.
#[test]
fn compile_equals_compile_typed() {
    let short = ff_text(CHARMM, "1.0").replace("3.5 4.2 3.0 5.0", "2.0 2.4");
    let cases = [
        (CHARMM, "1.0", ff_text(CHARMM, "1.0")),
        (CHARMM, "0.5", ff_text(CHARMM, "0.5")),
        (AMBER_LIKE, "0.0", ff_text(AMBER_LIKE, "0.0")),
        (CHARMM, "1.0, cut at 2.4", short),
        (AMBER_LIKE, "0.0, lj/cut/coul/cut 3.0", cut_text()),
    ];
    for (special, w, text) in cases {
        let ff = LammpsForcefieldReader::new().read(text.as_bytes()).unwrap();
        let frame = frame(false);
        let (e, f) = PotentialCompiler::new(&ff)
            .compile(&frame)
            .unwrap()
            .energy_forces_at(&coords());
        let (et, ft) = typed_energy(&ff, &frame);
        close(&format!("{special} w={w}"), et, e, 1e-12);
        for (a, b) in f.iter().zip(&ft) {
            assert!((a - b).abs() < 1e-10, "{special} w={w}: force {a} vs {b}");
        }
    }
}

/// Exceptions price LJ 12-6 + Coulomb; a field whose van der Waals is
/// something else cannot hand them a pair.
#[test]
fn an_override_beside_a_non_lj_style_is_refused() {
    let ff = LammpsForcefieldReader::new()
        .read(ff_text(CHARMM, "0.0").as_bytes())
        .unwrap();
    let mut buck = ff.empty_like();
    buck.def_style("pair", "buck", Default::default())
        .unwrap()
        .def_type(
            "CT3",
            &["CT3"],
            crate::ff::ir::Params::from_iter([("a", 1.0), ("rho", 0.3), ("c", 1.0)]),
        )
        .unwrap();
    let base = frame(false);
    let over = with_overrides(&base, &[("lj_scale", &|_, _| 0.5)]);
    let err = PotentialCompiler::new(&buck).compile(&over).unwrap_err();
    assert!(err.to_string().contains("buck"), "{err}");
}

/// The LAMMPS writers refuse a frame carrying per-pair overrides, by name;
/// the LAMMPS reader and writer are the identity on `lj/charmm/coul/charmm`.
#[test]
fn lammps_round_trip_and_override_refusal() {
    use crate::io::lammps::forcefield_writer::{
        LammpsForcefieldWriter, refuse_inexpressible_pairs,
    };
    use molrs::core::TypeLabels;

    let ff = read(CHARMM, "0.5");
    let frame = frame(false);
    let labels = TypeLabels::from_frame(&frame).unwrap();
    let text = LammpsForcefieldWriter::new(&labels).written(&ff).unwrap();
    assert!(
        text.contains("pair_style lj/charmm/coul/charmm 3.500000 4.200000 3.000000 5.000000"),
        "{text}"
    );
    assert!(
        text.contains("pair_coeff CT3 CT3 0.078000 3.670500 0.010000 3.385000"),
        "{text}"
    );
    assert!(
        text.contains("pair_coeff OH1 OH1 0.152100 3.150800 0.152100 3.150800"),
        "{text}"
    );
    assert!(
        // LAMMPS reads the charmm phase as an integer number of degrees.
        text.contains("dihedral_coeff CT3-CT2-CT2-OH1 0.300000 1 180 0.500000"),
        "{text}"
    );
    let back = LammpsForcefieldReader::new().read(text.as_bytes()).unwrap();
    for (category, name) in [
        ("pair", "lj/charmm"),
        ("pair", "coul/charmm"),
        ("dihedral", "charmm"),
    ] {
        let (a, b) = (
            ff.get_style(category, name).unwrap(),
            back.get_style(category, name).unwrap(),
        );
        assert_eq!(a.params(), b.params(), "{category}/{name}");
        let sorted = |s: &crate::ff::forcefield::Style| {
            let mut rows: Vec<_> = s
                .type_rows()
                .into_iter()
                .map(|(n, e, p)| (n.to_owned(), e.join(" "), p.clone()))
                .collect();
            rows.sort_by(|x, y| x.0.cmp(&y.0));
            rows
        };
        assert_eq!(sorted(a), sorted(b), "{category}/{name}");
    }

    refuse_inexpressible_pairs(&frame).unwrap();
    let over = with_overrides(&frame, &[("coul_scale", &|_, _| 0.5)]);
    let err = refuse_inexpressible_pairs(&over).unwrap_err();
    assert!(err.contains("coul_scale"), "{err}");

    let mut data = Vec::new();
    let mut over = over;
    let mut atoms = over.get("atoms").unwrap().clone();
    atoms
        .insert(
            "mol_id",
            Array1::from_vec(vec![1 as Idx; 7]).into_dyn(),
            None,
        )
        .unwrap();
    over.insert("atoms", atoms);
    let err = molrs::io::writer::FrameWriter::write(
        &mut molrs::io::lammps::data::LammpsDataWriter::new(&mut data),
        &over,
    )
    .unwrap_err();
    assert!(err.to_string().contains("coul_scale"), "{err}");
}
