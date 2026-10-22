//! OpenMM force-field XML, end to end: OpenMM's own energies, LAMMPS's, and
//! molrs's, term by term, on three molecules.
//!
//! The fixtures (`testdata/openmm/<case>.{xml,json}`) come from
//! `scripts/openmm_xml_check.py`: each XML is the rows of a real force field
//! a molecule uses, plus its residue template, and each JSON holds the
//! molecule (types, charges, positions in Å, the topology OpenMM priced) and
//! OpenMM's per-term energies (OpenMM 8.6.1, `Reference` platform — double
//! precision —, `NoCutoff`, every force in its own group), in kcal/mol:
//!
//! - `charmm`: ACE-ALA-NME, CHARMM36 (`charmm36.xml`): Urey–Bradley,
//!   harmonic `CustomTorsionForce` impropers, CMAP, `LennardJonesForce` with
//!   `sigma14`/`epsilon14` (so `one_four = "epsilon14"`, priced through
//!   [`ForceField::materialize_one_four`]) and an NBFIX row;
//! - `amber`: ACE-ALA-NME, AMBER ff14SB: multi-term propers, `ordering="amber"`
//!   wildcard impropers, 1-4 scales ½ / ⅚;
//! - `opls`: 1-propanol, OPLS-AA (molpy's `oplsaa.xml`): RB torsions, foyer's
//!   geometric `combining_rule`.
//!
//! The molrs force field is the XML read by [`OpenmmXmlReader`]; the frame lists
//! the bonds, angles and propers of the bond graph and the impropers and
//! crossterms with the atoms OpenMM priced, each row typed by the rule OpenMM
//! matches with (the first row without a wildcard, else the first with one;
//! propers, bonds and angles either way round). Nonbonded pairs are priced
//! with a cutoff beyond every pair (`lj/charmm`: `inner` 900 Å, `cutoff`
//! 1000 Å), which is `NoCutoff`.
//!
//! The LAMMPS numbers are `run 0` of LAMMPS (`~/.local/bin/lmp`, `boundary f
//! f f`) on the data file and include molrs writes from the same force field
//! (`MOLRS_OPENMM_CHECK_DIR=<dir> cargo mrs-test -- ff::openmm_check
//! --nocapture`, then `scripts/openmm_xml_check.sh`): `evdwl ecoul ebond
//! eangle edihed eimp f_cmap pe` at `%.17g`. CHARMM's per-type 1-4
//! parameters have no LAMMPS per-pair form, so the LAMMPS deck of `charmm` is
//! its `dihedral charmm` form: `special_bonds` 0, and one `dihedral charmm`
//! row (k = 0, `w` = 1) per 1-4 pair beside the torsions — which molrs also
//! prices, to the override form's energy.

// The pinned numbers are kept as their engines printed them.
#![allow(clippy::excessive_precision)]

use crate::core::Cell;
use crate::io::writer::WrittenText;
use std::collections::{BTreeMap, HashMap};
use std::path::Path;

use ndarray::Array1;
use serde_json::Value;

use crate::ff::compile::PotentialCompiler;
use crate::ff::forcefield::ForceField;
use crate::ff::ir::Params;
use crate::ff::potential::intramolecular_pairs;
use crate::io::lammps::LammpsForcefieldWriter;
use crate::io::openmm_xml::reader::OpenmmXmlReader;
use crate::io::openmm_xml::writer::OpenmmXmlWriter;
use crate::io::reader::ForceFieldReader;
use molrs::core::Block;
use molrs::core::Frame;
use molrs::core::SimBox;
use molrs::core::TypeLabels;
use molrs::io::lammps::data::write_lammps_data;
use molrs::op::{F, Idx};

use crate::ff::equivalence_check::{self, TERMS};

pub(crate) struct Case {
    name: &'static str,
    xml: &'static str,
    json: &'static str,
}

pub(crate) const CASES: [Case; 3] = [
    Case {
        name: "charmm",
        xml: include_str!("testdata/openmm/charmm.xml"),
        json: include_str!("testdata/openmm/charmm.json"),
    },
    Case {
        name: "amber",
        xml: include_str!("testdata/openmm/amber.xml"),
        json: include_str!("testdata/openmm/amber.json"),
    },
    Case {
        name: "opls",
        xml: include_str!("testdata/openmm/opls.xml"),
        json: include_str!("testdata/openmm/opls.json"),
    },
];

fn json(c: &Case) -> Value {
    serde_json::from_str(c.json).unwrap()
}

fn rows(v: &Value, key: &str) -> Vec<Vec<usize>> {
    v[key]
        .as_array()
        .unwrap()
        .iter()
        .map(|r| {
            r.as_array()
                .unwrap()
                .iter()
                .map(|i| i.as_u64().unwrap() as usize)
                .collect()
        })
        .collect()
}

fn floats(v: &Value, key: &str) -> Vec<F> {
    v[key]
        .as_array()
        .unwrap()
        .iter()
        .map(|x| x.as_f64().unwrap())
        .collect()
}

fn strings(v: &Value, key: &str) -> Vec<String> {
    v[key]
        .as_array()
        .unwrap()
        .iter()
        .map(|x| x.as_str().unwrap().to_owned())
        .collect()
}

/// The XML read with OpenMM's `NoCutoff` stated as a cutoff beyond every
/// pair.
pub(crate) fn read(c: &Case) -> ForceField {
    let mut ff = OpenmmXmlReader::new().read(c.xml.as_bytes()).unwrap();
    for name in ["lj/charmm", "coul/charmm", "lj/cut", "coul/cut"] {
        if let Some(style) = ff.get_style_mut("pair", name) {
            style.set_param("cutoff", 1000.0);
            if name.ends_with("charmm") {
                style.set_param("inner", 900.0);
            }
        }
    }
    ff
}

/// The name of the first type of `category` whose endpoints match the atom
/// `types` (wildcard `""`, a type name or its class), preferring a row
/// without a wildcard, as OpenMM's generators do; `reversible` also tries the
/// endpoints backwards.
fn match_type(ff: &ForceField, category: &str, types: &[&str], reversible: bool) -> Option<String> {
    let class_of: HashMap<String, String> = ff
        .get_atomtypes()
        .into_iter()
        .filter_map(|t| {
            t.params
                .get("class")
                .and_then(|v| v.as_text())
                .map(|c| (t.name.clone(), c.to_owned()))
        })
        .collect();
    let hit = |label: &str, ty: &str| {
        label.is_empty() || label == ty || class_of.get(ty).is_some_and(|c| c == label)
    };
    let mut wildcard = None;
    for style in ff.get_styles(category) {
        for (name, ends, _) in style.type_rows() {
            let forward = ends.iter().zip(types).all(|(l, t)| hit(l, t));
            let backward = reversible && ends.iter().rev().zip(types).all(|(l, t)| hit(l, t));
            if !(forward || backward) {
                continue;
            }
            if ends.iter().all(|l| !l.is_empty()) {
                return Some(name.to_owned());
            }
            wildcard.get_or_insert_with(|| name.to_owned());
        }
    }
    wildcard
}

fn relation(rows: &[(Vec<usize>, String)]) -> Block {
    let mut block = Block::new();
    for (k, key) in ["atomi", "atomj", "atomk", "atoml", "atomm"]
        .iter()
        .take(rows[0].0.len())
        .enumerate()
    {
        let col: Vec<Idx> = rows.iter().map(|r| r.0[k] as Idx).collect();
        block
            .insert(*key, Array1::from_vec(col).into_dyn(), None)
            .unwrap();
    }
    let labels: Vec<String> = rows.iter().map(|r| r.1.clone()).collect();
    block
        .insert("type", Array1::from_vec(labels).into_dyn(), None)
        .unwrap();
    block
}

/// The frame OpenMM priced, typed against `ff`, with its `pairs` (1-4
/// flagged from every proper of the bond graph).
pub(crate) fn frame(c: &Case, ff: &ForceField) -> Frame {
    let v = json(c);
    let types = strings(&v, "types");
    let x = rows_f(&v);
    let mut atoms = Block::new();
    for (d, key) in ["x", "y", "z"].iter().enumerate() {
        let col: Vec<F> = x.iter().map(|p| p[d]).collect();
        atoms
            .insert(*key, Array1::from_vec(col).into_dyn(), None)
            .unwrap();
    }
    atoms
        .insert("type", Array1::from_vec(types.clone()).into_dyn(), None)
        .unwrap();
    atoms
        .insert(
            "charge",
            Array1::from_vec(floats(&v, "charges")).into_dyn(),
            None,
        )
        .unwrap();
    atoms
        .insert(
            "mass",
            Array1::from_vec(floats(&v, "masses")).into_dyn(),
            None,
        )
        .unwrap();
    atoms
        .insert(
            "mol_id",
            Array1::from_vec(vec![1 as Idx; types.len()]).into_dyn(),
            None,
        )
        .unwrap();
    let mut out = Frame::new();
    out.insert("atoms", atoms);
    let typed = |key: &str, category: &str, reversible: bool, required: bool| {
        let mut typed = Vec::new();
        for r in rows(&v, key) {
            let t: Vec<&str> = r.iter().map(|&i| types[i].as_str()).collect();
            match match_type(ff, category, &t, reversible) {
                Some(name) => typed.push((r, name)),
                None => assert!(!required, "{}: no {category} type for {t:?}", c.name),
            }
        }
        typed
    };
    out.insert("bonds", relation(&typed("bonds", "bond", true, true)));
    out.insert("angles", relation(&typed("angles", "angle", true, false)));
    // Every proper flags its 1-4 pair; only the typed ones are priced.
    let all: Vec<(Vec<usize>, String)> = rows(&v, "dihedrals")
        .into_iter()
        .map(|r| (r, String::new()))
        .collect();
    out.insert("dihedrals", relation(&all));
    let pairs = intramolecular_pairs(&out, ff.special_bonds(), None).unwrap();
    out.insert("pairs", pairs);
    out.insert(
        "dihedrals",
        relation(&typed("dihedrals", "dihedral", true, false)),
    );
    let impropers = typed("impropers", "improper", false, true);
    if !impropers.is_empty() {
        out.insert("impropers", relation(&impropers));
    }
    let cmaps = typed("cmaps", "cmap", false, true);
    if !cmaps.is_empty() {
        out.insert("cmaps", relation(&cmaps));
    }
    out.simbox =
        Some(SimBox::new(Cell::Lengths([150.0; 3]), [-60.0, -60.0, -60.0], [false; 3]).unwrap());
    out
}

fn rows_f(v: &Value) -> Vec<[F; 3]> {
    v["positions"]
        .as_array()
        .unwrap()
        .iter()
        .map(|p| {
            let p = p.as_array().unwrap();
            [0, 1, 2].map(|d| p[d].as_f64().unwrap())
        })
        .collect()
}

fn coords(frame: &Frame) -> Vec<F> {
    frame.coords().unwrap().into_iter().collect()
}

/// molrs's energy per term family (kcal/mol), at the field's own Coulomb
/// constant ([`equivalence_check::molrs_terms`]).
fn molrs_terms(ff: &ForceField, frame: &Frame) -> BTreeMap<&'static str, F> {
    let x = coords(frame);
    equivalence_check::molrs_terms(ff, frame, &x, equivalence_check::coulomb(ff), false)
}

/// The field and frame molrs prices `c` with: `charmm`'s 1-4 pairs as
/// per-pair rows ([`ForceField::materialize_one_four`]).
fn system(c: &Case) -> (ForceField, Frame) {
    let ff = read(c);
    let mut frame = frame(c, &ff);
    if c.name == "charmm" {
        assert!(ff.materialize_one_four(&mut frame).unwrap() > 0);
    }
    (ff, frame)
}

/// `charmm` in LAMMPS's form: `special_bonds` 0 and a `w` = 1 zero-`K`
/// `dihedral charmm` row per 1-4 pair
/// ([`equivalence_check::one_four_as_dihedral_weights`]); the others as
/// read.
fn lammps_form(c: &Case) -> (ForceField, Frame) {
    let ff = read(c);
    let frame = frame(c, &ff);
    equivalence_check::one_four_as_dihedral_weights(&ff, &frame)
}

/// Write `c`'s LAMMPS inputs into `dir`: `<case>.data`, `<case>.pre` (lines
/// that go before `read_data`: `units`, `fix cmap`), `<case>.ff` (the rest),
/// and `<case>.cmap`.
fn stage_lammps_inputs(dir: &Path, c: &Case) {
    let (ff, frame) = lammps_form(c);
    let mut data = frame.clone();
    data.remove("pairs");
    write_lammps_data(&dir.join(format!("{}.data", c.name)), &data).unwrap();
    let labels = TypeLabels::from_frame(&data).unwrap();
    let has_cmap = data.get("cmaps").is_some();
    let writer = match has_cmap.then(|| format!("{}.cmap", c.name)) {
        Some(file) => LammpsForcefieldWriter::new(&labels)
            .with_precision(17)
            .with_cmap_file(file),
        None => LammpsForcefieldWriter::new(&labels).with_precision(17),
    };
    if has_cmap {
        std::fs::write(
            dir.join(format!("{}.cmap", c.name)),
            crate::io::writer::written_with(|b| writer.write_cmap(b, &ff)).unwrap(),
        )
        .unwrap();
    }
    let text = writer.written(&ff).unwrap();
    let (pre, post): (Vec<&str>, Vec<&str>) = text.lines().partition(|l| {
        l.starts_with("units") || l.starts_with("fix ") || l.starts_with("fix_modify")
    });
    std::fs::write(dir.join(format!("{}.pre", c.name)), pre.join("\n") + "\n").unwrap();
    std::fs::write(dir.join(format!("{}.ff", c.name)), post.join("\n") + "\n").unwrap();
}

/// OpenMM's energies of `c`, kcal/mol.
fn openmm(c: &Case) -> BTreeMap<String, F> {
    json(c)["openmm_kcal"]
        .as_object()
        .unwrap()
        .iter()
        .map(|(k, v)| (k.clone(), v.as_f64().unwrap()))
        .collect()
}

fn rel(got: F, want: F) -> F {
    if got == want {
        0.0
    } else {
        (got - want).abs() / want.abs().max(1e-300)
    }
}

/// LAMMPS `run 0` on the inputs [`write_lammps`] writes (`charmm` in its
/// `dihedral charmm` form), kcal/mol: `evdwl ecoul ebond eangle edihed eimp
/// f_cmap pe`, mapped onto the term families.
fn lammps(name: &str) -> BTreeMap<&'static str, F> {
    let v: [F; 8] = match name {
        "charmm" => [
            3.69450478587081506e+01,
            1.74494238714566272e+01,
            6.19239592482836354e+00,
            2.71583446309682452e+00,
            4.32705663378741057e-01,
            4.40626691723333908e-01,
            -2.46455770478659808e+01,
            3.95304574253260625e+01,
        ],
        "amber" => [
            3.22759317635579990e+01,
            1.56616641063401882e+01,
            1.30968291682471456e+01,
            1.03069092943995999e+00,
            0.0,
            2.99553028143828115e+00,
            -3.64569381398514025e+01,
            2.86037081091721674e+01,
        ],
        "opls" => [
            2.50680192347599586e+01,
            3.65165436206925076e+00,
            -2.60852980307393234e-01,
            0.0,
            0.0,
            1.01917383938231659e-01,
            9.41561171672068653e-01,
            2.95022991721321155e+01,
        ],
        _ => unreachable!(),
    };
    TERMS.iter().copied().zip(v).collect()
}

/// molrs prices every term as OpenMM does: within 3·10⁻¹² relative here (the
/// CMAP; its interpolation is LAMMPS's, whose node slopes come from a natural
/// spline over the doubled map where OpenMM's are periodic splines), the
/// others within 3·10⁻¹⁴.
#[test]
fn every_term_is_openmm_s() {
    for c in &CASES {
        let (ff, frame) = system(c);
        let m = molrs_terms(&ff, &frame);
        let o = openmm(c);
        for t in TERMS {
            let want = o.get(t).copied().unwrap_or(0.0);
            let tol = if t == "cmap" || t == "total" {
                1e-10
            } else {
                1e-12
            };
            assert!(
                rel(m[t], want) <= tol,
                "{} {t}: molrs {} vs OpenMM {want} (rel {:e})",
                c.name,
                m[t],
                rel(m[t], want)
            );
        }
    }
}

/// molrs prices every term as LAMMPS does on the files molrs writes, but for
/// the Coulomb constant: an OpenMM-read field states OpenMM's
/// (332.06371329919216), LAMMPS `real` fixes its own (332.06371), so molrs's
/// Coulomb is LAMMPS's times their ratio. LAMMPS is also OpenMM to 10⁻⁸ per
/// term.
#[test]
fn every_term_is_lammps_s() {
    let ratio = crate::core::constants::openmm_coulomb_real() / 332.06371;
    for c in &CASES {
        let (ff, frame) = system(c);
        let m = molrs_terms(&ff, &frame);
        let l = lammps(c.name);
        let o = openmm(c);
        for t in TERMS {
            let got = if t == "coul" { m[t] / ratio } else { m[t] };
            let want = if t == "total" {
                l["total"] + l["coul"] * (ratio - 1.0)
            } else {
                l[t]
            };
            assert!(
                rel(got, want) <= 1e-12,
                "{} {t}: molrs {got} vs LAMMPS {want} (rel {:e})",
                c.name,
                rel(got, want)
            );
            // Per term, LAMMPS is OpenMM to 1e-8 (the Coulomb constants
            // differ by 9.9e-9); the sum can cancel below that.
            if t == "total" {
                continue;
            }
            let omm = o.get(t).copied().unwrap_or(0.0);
            assert!(
                rel(l[t], omm) <= 1e-8,
                "{} {t}: LAMMPS {} vs OpenMM {omm}",
                c.name,
                l[t]
            );
        }
    }
}

/// `charmm`'s LAMMPS form (`special_bonds` 0, a `w` = 1 `dihedral charmm`
/// row per 1-4 pair) prices as its per-pair override form.
#[test]
fn the_lammps_form_of_charmm_prices_as_its_override_form() {
    let c = &CASES[0];
    let (ff, frame) = system(c);
    let (lff, lframe) = lammps_form(c);
    let (a, b) = (molrs_terms(&ff, &frame), molrs_terms(&lff, &lframe));
    for t in TERMS {
        assert!(rel(a[t], b[t]) <= 1e-13, "{t}: {} vs {}", a[t], b[t]);
    }
}

/// A `one_four = "epsilon14"` field compiles only for a frame whose 1-4
/// pairs carry their per-pair rows; the refusal names the operation.
#[test]
fn an_unmaterialized_epsilon14_frame_is_refused() {
    let c = &CASES[0];
    let ff = read(c);
    let bare = frame(c, &ff);
    let err = PotentialCompiler::new(&ff).compile(&bare).err().unwrap();
    assert!(err.to_string().contains("materialize_one_four"), "{err}");
    let err = PotentialCompiler::new(&ff)
        .compile_typed(&bare)
        .err()
        .unwrap();
    assert!(err.to_string().contains("materialize_one_four"), "{err}");
}

/// `materialize_one_four` fills each 1-4 pair once (OpenMM's 1-4 list) and
/// nothing else, keeps a cell already set, and is idempotent.
#[test]
fn materialize_one_four_fills_each_one_four_pair_once() {
    let c = &CASES[0];
    let ff = read(c);
    let mut f = frame(c, &ff);
    let n14 = {
        let p = f.get("pairs").unwrap();
        p.get("is_14")
            .unwrap()
            .get::<bool>()
            .unwrap()
            .iter()
            .filter(|b| **b)
            .count()
    };
    assert_eq!(ff.materialize_one_four(&mut f).unwrap(), n14);
    let p = f.get("pairs").unwrap();
    let flags: Vec<bool> = p
        .get("is_14")
        .unwrap()
        .get::<bool>()
        .unwrap()
        .iter()
        .copied()
        .collect();
    for key in ["epsilon", "sigma", "lj_scale", "coul_scale"] {
        let mask = p.validity(key).unwrap();
        assert_eq!(mask, &flags[..], "{key}");
    }
    assert!(p.get("charge_product").is_none());
    // A set cell is final; a second call fills nothing.
    let mut again = f.clone();
    let first = (0..flags.len()).find(|&r| flags[r]).unwrap();
    {
        let pairs = again.get_mut("pairs").unwrap();
        let mut eps = pairs
            .get("epsilon")
            .unwrap()
            .get::<f64>()
            .unwrap()
            .to_owned();
        eps[[first]] = 0.5;
        pairs.insert("epsilon", eps, Some(flags.clone())).unwrap();
    }
    assert_eq!(ff.materialize_one_four(&mut again).unwrap(), 0);
    let eps = again
        .get("pairs")
        .unwrap()
        .get("epsilon")
        .unwrap()
        .get::<f64>()
        .unwrap()[[first]];
    assert_eq!(eps, 0.5);
}

/// The two `(name, endpoints, params)` lists of `a` and `b`, style by style.
fn assert_same_field(a: &ForceField, b: &ForceField, what: &str) {
    let key = |ff: &ForceField| {
        let mut v: Vec<String> = ff
            .styles()
            .iter()
            .map(|s| format!("{}/{}", s.category(), s.name()))
            .collect();
        v.sort();
        v
    };
    assert_eq!(key(a), key(b), "{what}: styles");
    for sa in a.styles() {
        let sb = b.get_style(sa.category(), sa.name()).unwrap();
        for (k, v) in sa
            .params()
            .iter()
            .filter_map(|(k, v)| v.as_num().map(|x| (k, x)))
        {
            assert!(
                rel(sb.params().get(k).and_then(|v| v.as_num()).unwrap(), v) <= 1e-14,
                "{what} {}: {k}",
                sa.name()
            );
        }
        let rows_b: HashMap<&str, (Vec<&str>, &Params)> = sb
            .type_rows()
            .into_iter()
            .map(|(n, e, p)| (n, (e, p)))
            .collect();
        assert_eq!(sa.type_rows().len(), rows_b.len(), "{what} {}", sa.name());
        for (name, ends, p) in sa.type_rows() {
            let (eb, pb) = &rows_b[name];
            assert_eq!(&ends, eb, "{what} {name}");
            for (k, v) in p.iter().filter_map(|(k, v)| v.as_num().map(|x| (k, x))) {
                let got = pb
                    .get(k)
                    .and_then(|v| v.as_num())
                    .unwrap_or_else(|| panic!("{what} {name}: lost {k}"));
                assert!(rel(got, v) <= 1e-14, "{what} {name} {k}: {got} vs {v}");
            }
            assert_eq!(
                p.iter()
                    .filter_map(|(k, v)| v.as_num().map(|x| (k, x)))
                    .count(),
                pb.iter()
                    .filter_map(|(k, v)| v.as_num().map(|x| (k, x)))
                    .count(),
                "{what} {name}"
            );
            for (k, v) in p.iter().filter_map(|(k, v)| v.as_text().map(|s| (k, s))) {
                assert_eq!(
                    pb.get(k).and_then(|v| v.as_text()),
                    Some(v),
                    "{what} {name} {k}"
                );
            }
            for (k, v) in p.iter().filter_map(|(k, v)| v.as_array().map(|a| (k, a))) {
                let w = pb.get(k).and_then(|v| v.as_array()).unwrap();
                assert!(
                    v.iter().zip(w.iter()).all(|(x, y)| rel(*y, *x) <= 1e-14),
                    "{what} {name} {k}"
                );
            }
        }
    }
    assert_eq!(
        a.special_bonds(),
        b.special_bonds(),
        "{what}: special_bonds"
    );
}

/// read → write → read is the identity on every fixture (to the last bits
/// a kJ ↔ kcal round trip moves), and prices the same.
#[test]
fn read_write_read_is_the_identity() {
    for c in &CASES {
        let ff = OpenmmXmlReader::new().read(c.xml.as_bytes()).unwrap();
        let xml = OpenmmXmlWriter::new().written(&ff).unwrap();
        let back = OpenmmXmlReader::new().read(xml.as_bytes()).unwrap();
        assert_same_field(&ff, &back, c.name);
        let again = OpenmmXmlWriter::new().written(&back).unwrap();
        let third = OpenmmXmlReader::new().read(again.as_bytes()).unwrap();
        assert_same_field(&back, &third, c.name);
    }
}

/// Prints the per-term table (molrs, OpenMM, LAMMPS) and, with
/// `MOLRS_OPENMM_CHECK_DIR` set, writes the LAMMPS inputs there.
#[test]
fn report() {
    let dir = std::env::var_os("MOLRS_OPENMM_CHECK_DIR");
    for c in &CASES {
        let (ff, frame) = system(c);
        let m = molrs_terms(&ff, &frame);
        let o = openmm(c);
        for t in TERMS {
            let want = o.get(t).copied().unwrap_or(0.0);
            println!(
                "{} {t:9} molrs {:.17e} openmm {want:.17e} rel {:.2e}",
                c.name,
                m[t],
                rel(m[t], want)
            );
        }
        if let Some(dir) = &dir {
            stage_lammps_inputs(Path::new(dir), c);
            // The molrs-written XML, for scripts/openmm_xml_check.py --written.
            let xml = OpenmmXmlWriter::new().written(&read(c)).unwrap();
            std::fs::write(Path::new(dir).join(format!("{}.written.xml", c.name)), xml).unwrap();
            let (lff, lframe) = lammps_form(c);
            let l = molrs_terms(&lff, &lframe);
            for t in TERMS {
                println!("{} {t:9} molrs-lammps-form {:.17e}", c.name, l[t]);
            }
        }
    }
}
