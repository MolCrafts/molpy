//! GROMACS-read force fields against GROMACS and LAMMPS, term by term.
//!
//! Four systems read with [`GromacsTopForcefieldReader::read_system`] — an
//! ACE-ALA-ALA-NME dipeptide under GROMACS's charmm27 (Urey-Bradley, CMAP,
//! `[ pairtypes ]`, plus a `[ nonbond_params ]` row), amber99sb-ildn (funct-9
//! multi-term dihedrals, funct-4 impropers, gen-pairs 0.5 / 0.8333) and oplsaa
//! (funct-3 Ryckaert-Bellemans, funct-1 impropers from `#define` macros), and
//! the AMBER one with two `[ pairs ]` rows of their own (funct 1 with
//! parameters, funct 2) — priced by molrs, with every term held to GROMACS's
//! single point (GROMACS 2025.3, double precision, `mdrun -rerun`, energies
//! from the .edr) and, for the three LAMMPS can express, to LAMMPS's `run 0`
//! on the data file and include molrs's LAMMPS writer produces.
//!
//! `scripts/gromacs_engine_check.sh` builds the fixtures (`testdata/`, with
//! `--regen`), runs GROMACS and LAMMPS, and runs this test with
//! `MOLRS_GMX_CHECK_DIR` set, when it writes the LAMMPS inputs and prints
//! molrs's terms instead of asserting; it prints the three engines side by
//! side, and the numbers pinned below are its output.
//!
//! Settings: a plain cut-off at 2.5 nm (no shift, no reaction field) in a
//! 6 nm box — every intramolecular pair inside it, no image — so nonbonded
//! energies are the plain sums. The reader states GROMACS's own Coulomb
//! constant ([`crate::core::constants::GROMACS_ONE_4PI_EPS0`], CODATA 2018), LAMMPS `real`'s
//! 332.06371 × (1 + 9.9·10⁻⁹); LAMMPS prices at its own, so molrs's Coulomb
//! terms are held to LAMMPS's times the constants' ratio (exact: the energy is
//! linear in it).
//!
//! molrs prices each system as read, its 1-4 pairs written out by
//! [`ForceField::materialize_one_four`] (CHARMM's `[ pairtypes ]` pairs are
//! `lj/charmm` `one_four = "epsilon14"` pairs, which `compile` prices only
//! through per-pair rows or `w`). LAMMPS prices CHARMM's 1-4 pairs only
//! through `dihedral charmm` `w`: the LAMMPS form of the system
//! ([`lammps_form`]) gives each `[ pairs ]` row a zero-`K` `dihedral charmm`
//! row with `w` = 1 over a bonded path between its atoms, beside
//! `special_bonds` 0 — exactly GROMACS's fudge-1 pricing. The two forms price
//! every term alike to 1e-12.

// The engines' numbers are kept as they printed them.
#![allow(clippy::excessive_precision)]

use crate::ff::potential::fixtures::OneShot;
use std::io::Cursor;
use std::path::Path;

use super::GromacsTopForcefieldReader;
use crate::ff::compile::PotentialCompiler;
use crate::ff::compile::exceptions;
use crate::ff::forcefield::ForceField;
use crate::ff::ir::Params;
use crate::io::lammps::LammpsForcefieldWriter;
use crate::io::writer::WrittenText;
use molrs::core::Frame;
use molrs::core::TypeLabels;
use molrs::core::constants::COULOMB_REAL;
use molrs::io::gro::GroReader;
use molrs::io::lammps::data::write_lammps_data;
use molrs::io::reader::FrameReader as _;
use molrs::op::F;

/// The terms compared, in print order.
const TERMS: [&str; 10] = [
    "bond", "angle", "dihedral", "improper", "cmap", "lj14", "coul14", "ljsr", "coulsr", "total",
];

/// GROMACS's cut-off, 2.5 nm, in Å; CHARMM's switch starts past every pair.
const CUTOFF: F = 25.0;
const INNER: F = 24.0;

use crate::ff::equivalence_check::{PAIR14, one_four_as_dihedral_weights as lammps_form};

struct Fixture {
    name: &'static str,
    top: &'static str,
    gro: &'static str,
    /// GROMACS's terms, kcal/mol (kJ/mol ÷ 4.184), in [`TERMS`] order; `None`
    /// where GROMACS has no such term.
    gromacs: [Option<F>; 10],
    /// LAMMPS's, when LAMMPS can express the system.
    lammps: Option<[Option<F>; 10]>,
}

const FIXTURES: [Fixture; 4] = [
    Fixture {
        name: "charmm",
        top: include_str!("../testdata/charmm.top"),
        gro: include_str!("../testdata/charmm.gro"),
        gromacs: [
            Some(182.25712787159014),
            Some(78.943347292483878),
            Some(13.129118289659084),
            Some(18.926079978379999),
            Some(-2.9772498920135844),
            Some(6.3755187870688639),
            Some(135.24737449193697),
            Some(-1.8378516332532395),
            Some(-141.49871524251137),
            Some(288.56474994334076),
        ],
        lammps: Some([
            Some(182.25712787159026),
            Some(78.943347292483779),
            Some(13.12911828965902),
            Some(18.926079978379807),
            Some(-2.9772498920135861),
            Some(6.3755187973046983),
            Some(135.24737314823099),
            Some(-1.837851633253009),
            Some(-141.49871383666317),
            Some(288.5647500157188),
        ]),
    },
    Fixture {
        name: "amber",
        top: include_str!("../testdata/amber.top"),
        gro: include_str!("../testdata/amber.gro"),
        gromacs: [
            Some(173.28146320772248),
            Some(83.310606638709288),
            Some(22.164945658994554),
            Some(11.976411229792317),
            None,
            Some(10.45294650893398),
            Some(121.64745488591926),
            Some(-2.7440438675344145),
            Some(-153.56352094778617),
            Some(266.52626331475125),
        ],
        lammps: Some([
            Some(173.2814632077216),
            Some(83.310606638709146),
            Some(22.164945658994494),
            Some(11.976411229792131),
            None,
            Some(10.452946523324799),
            Some(121.6474536773421),
            Some(-2.7440438675342653),
            Some(-153.56351942206942),
            Some(266.5262636462806),
        ]),
    },
    Fixture {
        name: "opls",
        top: include_str!("../testdata/opls.top"),
        gro: include_str!("../testdata/opls.gro"),
        gromacs: [
            Some(169.56795782837412),
            Some(67.809689238884062),
            Some(22.629534813112112),
            None,
            None,
            Some(14.776807892543285),
            Some(65.04424862112711),
            Some(-1.9972496167390061),
            Some(-136.84986492721868),
            Some(200.98112385008304),
        ],
        lammps: Some([
            Some(169.56795782837327),
            Some(67.809689238884019),
            Some(22.629534813111853),
            None,
            None,
            Some(14.776807909779722),
            Some(65.044247974906241),
            Some(-1.9972496167387521),
            Some(-136.84986356755866),
            Some(200.98112458075772),
        ]),
    },
    Fixture {
        name: "amber_pairs",
        top: include_str!("../testdata/amber_pairs.top"),
        gro: include_str!("../testdata/amber_pairs.gro"),
        gromacs: [
            Some(173.28146320772248),
            Some(83.310606638709288),
            Some(22.164945658994554),
            Some(11.976411229792317),
            None,
            Some(14.754553113511502),
            Some(118.45036932837711),
            Some(-2.7440438675344145),
            Some(-153.56352094778617),
            Some(267.63078436178665),
        ],
        // Per-pair parameters: LAMMPS cannot express them, and molrs's
        // LAMMPS writer refuses them.
        lammps: None,
    },
];

/// Relative tolerance against GROMACS, per term. The bonded terms, CMAP and
/// the van-der-Waals pairs agree to rounding; GROMACS prices 1-4 pairs from
/// cubic-spline tables (≈10⁻⁹, Coulomb-14 included).
const GROMACS_TOL: [F; 10] = [
    1e-12, 1e-12, 1e-12, 1e-12, 1e-12, 5e-9, 5e-9, 1e-12, 1e-11, 5e-9,
];

/// The fixture's force field (with the comparison's cutoffs declared), its
/// frame with the .gro's coordinates, and the coordinates.
fn load(f: &Fixture) -> (ForceField, Frame, Vec<F>) {
    let (mut ff, mut frame) = GromacsTopForcefieldReader::new()
        .read_system(f.top.as_bytes())
        .unwrap_or_else(|e| panic!("{}: {e}", f.name));
    let gro = GroReader::new(Cursor::new(f.gro))
        .read()
        .unwrap()
        .expect("one frame");
    let g = gro.get("atoms").expect("gro atoms");
    let atoms = frame.get_mut("atoms").expect("atoms");
    for key in ["x", "y", "z"] {
        let col = g.get(key).and_then(|c| c.get::<f64>()).expect("coordinate");
        atoms.insert(key, col.to_owned(), None).unwrap();
    }
    frame.simbox = gro.simbox.clone();
    let names: Vec<String> = ff
        .get_styles("pair")
        .iter()
        .map(|s| s.name().to_owned())
        .collect();
    for name in names {
        let style = ff.get_style_mut("pair", &name).unwrap();
        style.set_param("cutoff", CUTOFF);
        if name.ends_with("charmm") {
            style.set_param("inner", INNER);
        }
    }
    let coords: Vec<F> = frame.coords().unwrap().into_iter().collect();
    (ff, frame, coords)
}

/// `ff` with only its `(category, name)` style, and `frame`'s relation block
/// of that category cut to that style's rows. A `dihedral charmm` copy has
/// its `w` zeroed: the torsion alone (its 1-4 pairs are nonbonded terms).
fn only(ff: &ForceField, frame: &Frame, category: &str, name: &str) -> (ForceField, Frame) {
    let style = ff.get_style(category, name).unwrap();
    let mut one = ff.empty_like();
    let s = one
        .def_style(category, name, style.params().clone())
        .unwrap();
    for (type_name, ends, p) in style.type_rows() {
        let mut p = p.clone();
        if p.get("w").and_then(|v| v.as_num()).is_some() {
            p.set("w", 0.0);
        }
        s.def_type(type_name, &ends, p).unwrap();
    }
    let mut frame = frame.clone();
    if category != "pair" {
        // A bonded term alone: no pair list, so no 1-4 exception either.
        frame.remove("pairs");
    }
    let block_name = match category {
        "bond" => "bonds",
        "angle" => "angles",
        "dihedral" => "dihedrals",
        "improper" => "impropers",
        "cmap" => "cmaps",
        _ => return (one, frame),
    };
    if let Some(block) = frame.get(block_name) {
        let names: Vec<&str> = style.type_rows().iter().map(|r| r.0).collect();
        let types = block.get("type").unwrap().get::<String>().unwrap();
        let keep: Vec<usize> = (0..types.len())
            .filter(|&i| names.contains(&types[i].as_str()))
            .collect();
        let kept = block.select_rows(&keep).unwrap();
        frame.insert(block_name, kept);
    }
    (one, frame)
}

fn energy(ff: &ForceField, frame: &Frame, coords: &[F]) -> F {
    PotentialCompiler::new(ff)
        .compile(frame)
        .unwrap()
        .energy_at(coords)
}

/// `frame` with only the `pairs` rows `keep` selects.
fn with_pairs(frame: &Frame, keep: impl Fn(usize) -> bool) -> Frame {
    let pairs = frame.get("pairs").unwrap();
    let rows: Vec<usize> = (0..pairs.n_rows().unwrap()).filter(|&r| keep(r)).collect();
    let mut out = frame.clone();
    out.insert("pairs", pairs.select_rows(&rows).unwrap());
    out
}

/// molrs's terms of `ff` on `frame`, in [`TERMS`] order (`None`: no such
/// style).
fn terms(ff: &ForceField, frame: &Frame, coords: &[F]) -> [Option<F>; 10] {
    let mut out = [None; 10];
    let mut add = |slot: usize, e: F| *out[slot].get_or_insert(0.0) += e;
    for style in ff.styles() {
        let slot = match style.category() {
            "bond" => 0,
            "angle" => 1,
            "dihedral" => 2,
            "improper" => 3,
            "cmap" => 4,
            _ => continue,
        };
        let (one, cut) = only(ff, frame, style.category(), style.name());
        add(slot, energy(&one, &cut, coords));
    }
    // Nonbonded: the regular kernels on the 1-4 rows without an override cell
    // and on the other rows; the exceptions kernel (overrides, dihedral w).
    let pairs = frame.get("pairs").unwrap();
    let is_14 = pairs
        .get("is_14")
        .and_then(|c| c.get::<bool>())
        .unwrap()
        .clone();
    let overridden = |r: usize| {
        molrs::core::schema::PAIR_OVERRIDE_COLUMNS
            .iter()
            .any(|k| pairs.get(k).is_some() && pairs.validity(k).is_none_or(|m| m[r]))
    };
    let regular14 = with_pairs(frame, |r| is_14[r] && !overridden(r));
    let sr = with_pairs(frame, |r| !is_14[r]);
    for style in ff.get_styles("pair") {
        let lj = style.name().starts_with("lj");
        let (one, _) = only(ff, frame, "pair", style.name());
        add(if lj { 5 } else { 6 }, energy(&one, &regular14, coords));
        add(if lj { 7 } else { 8 }, energy(&one, &sr, coords));
    }
    let (lj14, coul14) = exceptions::plan(ff, frame)
        .unwrap()
        .kernel
        .map(|k| k.energy_terms(coords))
        .unwrap_or((0.0, 0.0));
    add(5, lj14);
    add(6, coul14);
    add(9, energy(ff, frame, coords));
    out
}

/// The LAMMPS inputs of the system into `dir` (see the script).
fn write_lammps(dir: &Path, ff: &ForceField, frame: &Frame) {
    std::fs::create_dir_all(dir).unwrap();
    let mut lammps = frame.clone();
    for block in ["pairs", "exclusions", "constraints"] {
        lammps.remove(block);
    }
    write_lammps_data(&dir.join("data.lmp"), &lammps).unwrap();
    let labels = TypeLabels::from_frame(&lammps).unwrap();
    let has_cmap = lammps.get("cmaps").is_some();
    let writer = LammpsForcefieldWriter::new(&labels)
        .with_precision(16)
        .with_skip_units(true);
    let writer = if has_cmap {
        writer.with_cmap_file("charmm.cmap")
    } else {
        writer
    };
    let include = writer.written(ff).unwrap();
    let (pre, rest): (Vec<&str>, Vec<&str>) = include
        .lines()
        .partition(|l| l.starts_with("fix cmap") || l.starts_with("fix_modify cmap"));
    std::fs::write(dir.join("pre.lmp"), pre.join("\n") + "\n").unwrap();
    std::fs::write(dir.join("system.ff"), rest.join("\n") + "\n").unwrap();
    if has_cmap {
        writer
            .write_cmap(dir.join("charmm.cmap").as_path(), ff)
            .unwrap();
    }
    let extra = if has_cmap {
        "fix cmap crossterm CMAP"
    } else {
        ""
    };
    std::fs::write(dir.join("read_data_extra"), extra).unwrap();
    std::fs::write(
        dir.join("thermo_extra"),
        if has_cmap { "f_cmap" } else { "" },
    )
    .unwrap();
    let hybrid = ff.get_styles("dihedral").len() > 1;
    let no14 = if ff.get_style("dihedral", "charmm").is_some() {
        format!(
            "dihedral_coeff {PAIR14} {}0.0 1 0 0.0",
            if hybrid { "charmm " } else { "" }
        )
    } else {
        "special_bonds lj 0.0 0.0 0.0 coul 0.0 0.0 0.0".to_owned()
    };
    std::fs::write(dir.join("no14.lmp"), no14 + "\n").unwrap();
}

fn assert_close(what: &str, got: F, want: F, rel: F) {
    let scale = want.abs().max(1e-12);
    assert!(
        (got - want).abs() <= rel * scale,
        "{what}: molrs {got:.17e} vs {want:.17e} (rel {:e} > {rel:e})",
        (got - want).abs() / scale
    );
}

/// molrs's terms of a fixture: the field as read, its 1-4 pairs written out
/// as per-pair rows ([`ForceField::materialize_one_four`]) — what GROMACS
/// prices — and the LAMMPS form ([`lammps_form`]) — what LAMMPS prices —
/// with that form's force field and frame.
fn molrs_terms(f: &Fixture) -> ([Option<F>; 10], [Option<F>; 10], ForceField, Frame) {
    let (ff, frame, coords) = load(f);
    let mut materialized = frame.clone();
    ff.materialize_one_four(&mut materialized)
        .unwrap_or_else(|e| panic!("{}: {e}", f.name));
    let native = terms(&ff, &materialized, &coords);
    let (lff, lframe) = lammps_form(&ff, &frame);
    let lammps = terms(&lff, &lframe, &coords);
    for (k, term) in TERMS.iter().enumerate() {
        assert_eq!(
            native[k].is_some(),
            lammps[k].is_some(),
            "{} {term}",
            f.name
        );
        if let (Some(a), Some(b)) = (native[k], lammps[k]) {
            assert_close(
                &format!("{} {term}: IR vs its LAMMPS form", f.name),
                a,
                b,
                1e-12,
            );
        }
    }
    (native, lammps, lff, lframe)
}

#[test]
fn gromacs_read_systems_price_as_gromacs_and_lammps() {
    let check_dir = std::env::var_os("MOLRS_GMX_CHECK_DIR");
    for f in &FIXTURES {
        let (molrs, lammps_form_terms, lff, lframe) = molrs_terms(f);
        if let Some(dir) = &check_dir {
            for (term, v) in TERMS.iter().zip(molrs) {
                if let Some(v) = v {
                    println!("molrs {} {term} {v:.17e}", f.name);
                }
            }
            if f.lammps.is_some() {
                write_lammps(
                    &Path::new(dir).join(format!("lmp-{}", f.name)),
                    &lff,
                    &lframe,
                );
            }
            continue;
        }
        for (k, term) in TERMS.iter().enumerate() {
            let what = format!("{} {term}", f.name);
            assert_eq!(
                molrs[k].is_some(),
                f.gromacs[k].is_some(),
                "{what}: present"
            );
            if let (Some(got), Some(want)) = (molrs[k], f.gromacs[k]) {
                assert_close(&format!("{what} vs GROMACS"), got, want, GROMACS_TOL[k]);
            }
            if let Some(lammps) = f.lammps
                && let (Some(got), Some(want)) = (lammps_form_terms[k], lammps[k])
            {
                // LAMMPS prices at its own Coulomb constant.
                let ratio = COULOMB_REAL / crate::core::constants::gromacs_coulomb_real();
                let coul = |t: &[Option<F>; 10]| t[6].unwrap_or(0.0) + t[8].unwrap_or(0.0);
                let got = match *term {
                    "coul14" | "coulsr" => got * ratio,
                    "total" => got + coul(&lammps_form_terms) * (ratio - 1.0),
                    _ => got,
                };
                assert_close(&format!("{what} vs LAMMPS"), got, want, 1e-13);
            }
        }
    }
}

/// The per-pair parameters of `amber_pairs` are what LAMMPS cannot hold: its
/// writer refuses them, by name.
#[test]
fn per_pair_parameters_are_refused_by_the_lammps_writer() {
    let (_, frame, _) = load(&FIXTURES[3]);
    let err =
        crate::io::lammps::forcefield_writer::refuse_pair_overrides(&frame).expect_err("overrides");
    assert!(err.contains("epsilon"), "{err}");
}

/// The fixtures' directives — pairtypes, NBFIX, UB, funct 9, RB, cmap —
/// written back by the GROMACS writer read as the same force field: every
/// style, type name, endpoint and parameter (to 1e-12, the decimal print).
#[test]
fn fixture_directives_survive_write_then_read() {
    use crate::io::gromacs::top_writer::GromacsTopForcefieldWriter;
    use crate::io::reader::ForceFieldReader;
    let reader = [
        "constrainttypes",
        "moleculetype",
        "atoms",
        "bonds",
        "pairs",
        "angles",
        "dihedrals",
        "cmap",
        "system",
        "molecules",
    ]
    .iter()
    .fold(GromacsTopForcefieldReader::new(), |r, s| {
        r.with_skipped_directive(s)
    });
    let close = |a: f64, b: f64| (a - b).abs() <= 1e-12 * a.abs().max(b.abs()).max(1e-3);
    for f in &FIXTURES[..3] {
        let ff = reader.read(f.top.as_bytes()).unwrap();
        let text = GromacsTopForcefieldWriter::new()
            .with_precision(17)
            .written(&ff)
            .unwrap_or_else(|e| panic!("{}: {e}", f.name));
        let back = reader
            .read(text.as_bytes())
            .unwrap_or_else(|e| panic!("{}: {e}\n{text}", f.name));
        assert_eq!(back.special_bonds(), ff.special_bonds(), "{}", f.name);
        let key = |ff: &ForceField| {
            let mut k: Vec<(String, String)> = ff
                .styles()
                .iter()
                .map(|s| (s.category().to_owned(), s.name().to_owned()))
                .collect();
            k.sort();
            k
        };
        assert_eq!(key(&back), key(&ff), "{}: styles", f.name);
        for style in ff.styles() {
            let other = back.get_style(style.category(), style.name()).unwrap();
            let what = format!("{} {}/{}", f.name, style.category(), style.name());
            assert_eq!(
                other
                    .params()
                    .iter()
                    .filter_map(|(k, v)| v.as_text().map(|s| (k, s)))
                    .collect::<Vec<_>>()
                    .len(),
                style
                    .params()
                    .iter()
                    .filter_map(|(k, v)| v.as_text().map(|s| (k, s)))
                    .count(),
                "{what}: string params"
            );
            for (k, v) in style
                .params()
                .iter()
                .filter_map(|(k, v)| v.as_text().map(|s| (k, s)))
            {
                assert_eq!(
                    other.params().get(k).and_then(|v| v.as_text()),
                    Some(v),
                    "{what}: {k}"
                );
            }
            let mut want = style.type_rows();
            let mut got = other.type_rows();
            want.sort_by(|a, b| a.0.cmp(b.0));
            got.sort_by(|a, b| a.0.cmp(b.0));
            let names = |rows: &[(&str, Vec<&str>, &Params)]| -> Vec<String> {
                rows.iter().map(|r| r.0.to_owned()).collect()
            };
            assert_eq!(names(&got), names(&want), "{what}: type names");
            for ((name, ends, p), (_, got_ends, q)) in want.iter().zip(&got) {
                assert_eq!(got_ends, ends, "{what} {name}: endpoints");
                let mut keys: Vec<&str> = p
                    .iter()
                    .filter_map(|(k, v)| v.as_num().map(|x| (k, x)))
                    .map(|(k, _)| k)
                    .collect();
                let mut got_keys: Vec<&str> = q
                    .iter()
                    .filter_map(|(k, v)| v.as_num().map(|x| (k, x)))
                    .map(|(k, _)| k)
                    .collect();
                keys.sort_unstable();
                got_keys.sort_unstable();
                assert_eq!(got_keys, keys, "{what} {name}: keys");
                for (k, v) in p.iter().filter_map(|(k, v)| v.as_num().map(|x| (k, x))) {
                    let g = q.get(k).and_then(|v| v.as_num()).unwrap();
                    assert!(close(g, v), "{what} {name}.{k}: {g} vs {v}");
                }
                for (k, a) in p.iter().filter_map(|(k, v)| v.as_array().map(|a| (k, a))) {
                    let b = q.get(k).and_then(|v| v.as_array()).expect("array param");
                    assert_eq!(a.shape(), b.shape(), "{what} {name}.{k}");
                    assert!(
                        a.iter().zip(b).all(|(x, y)| close(*x, *y)),
                        "{what} {name}.{k}"
                    );
                }
            }
        }
    }
}
