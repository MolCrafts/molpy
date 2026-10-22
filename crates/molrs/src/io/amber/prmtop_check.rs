//! The AMBER prmtop readers against sander and LAMMPS, term by term.
//!
//! Each case is a prmtop + restart that `scripts/prmtop_fixtures.py` built
//! with AmberTools 26.1 (tleap, antechamber/parmchk2, ParmEd's chamber) and
//! perturbed off its built geometry; the pinned `SANDER` numbers are
//! pysander's energy decomposition at those coordinates (`igb = 0`,
//! `ntb = 0`, `cut = 999`). molrs reads the frame and the force field,
//! builds the pair list with `intramolecular_pairs` (which keeps the frame
//! reader's per-pair 1-4 scales), and prices each term with only its style.
//! sander's `DIHED` holds AMBER impropers too, its `IMP` the CHARMM ones.
//!
//! `scripts/prmtop_check.sh` runs [`each_term_matches_sander_and_lammps`] with
//! `MOLRS_PRMTOP_LAMMPS_DIR` set: molrs then writes each case's LAMMPS data
//! file and include, LAMMPS runs `run 0` on them, and the script prints
//! LAMMPS's terms, pinned below as `LAMMPS`.

// The engines' numbers are kept as they printed them.
#![allow(clippy::excessive_precision)]

use crate::core::Cell;
use crate::ff::potential::fixtures::OneShot;
use crate::io::writer::WrittenText;
use std::path::Path;

use crate::ff::compile::PotentialCompiler;
use crate::ff::forcefield::ForceField;
use crate::ff::ir::Params;
use crate::ff::potential::intramolecular_pairs;
use crate::io::amber::prmtop_forcefield::AmberPrmtopForcefieldReader;
use crate::io::lammps::LammpsForcefieldWriter;
use molrs::core::Frame;
use molrs::core::SimBox;
use molrs::core::TypeLabels;
use molrs::io::lammps::data::write_lammps_data;
use molrs::io::read_inpcrd;
use molrs::op::F;

/// One energy decomposition, kcal/mol, in sander's terms.
#[derive(Debug, Clone, Copy, Default)]
struct Terms {
    bond: F,
    angle: F,
    angle_ub: F,
    dihedral: F,
    imp: F,
    cmap: F,
    vdw_14: F,
    elec_14: F,
    vdw: F,
    elec: F,
}

impl Terms {
    fn named(&self) -> [(&'static str, F); 10] {
        [
            ("bond", self.bond),
            ("angle", self.angle),
            ("angle_ub", self.angle_ub),
            ("dihedral", self.dihedral),
            ("imp", self.imp),
            ("cmap", self.cmap),
            ("vdw_14", self.vdw_14),
            ("elec_14", self.elec_14),
            ("vdw", self.vdw),
            ("elec", self.elec),
        ]
    }
}

struct Case {
    name: &'static str,
    parm: &'static str,
    rst: &'static str,
    sander: Terms,
    lammps: Terms,
}

macro_rules! fixture {
    ($name:literal) => {
        (
            include_str!(concat!("testdata/prmtop/", $name, ".parm7")),
            include_str!(concat!("testdata/prmtop/", $name, ".rst7")),
        )
    };
}

fn cases() -> Vec<Case> {
    let case = |name, (parm, rst): (&'static str, &'static str), sander, lammps| Case {
        name,
        parm,
        rst,
        sander,
        lammps,
    };
    vec![
        case("ff14sb", fixture!("ff14sb"), SANDER_FF14SB, LAMMPS_FF14SB),
        case("gaff2", fixture!("gaff2"), SANDER_GAFF2, LAMMPS_GAFF2),
        case(
            "gaff2_multi",
            fixture!("gaff2_multi"),
            SANDER_GAFF2_MULTI,
            LAMMPS_GAFF2_MULTI,
        ),
        case("glycam", fixture!("glycam"), SANDER_GLYCAM, LAMMPS_GLYCAM),
        case(
            "chamber",
            fixture!("chamber"),
            SANDER_CHAMBER,
            LAMMPS_CHAMBER,
        ),
        case("ff19sb", fixture!("ff19sb"), SANDER_FF19SB, LAMMPS_FF19SB),
    ]
}

// pysander, AmberTools 26.1 (`scripts/prmtop_fixtures.py`).
const SANDER_FF14SB: Terms = Terms {
    bond: 93.2978231153444,
    angle: 62.814326305362684,
    angle_ub: 0.0,
    dihedral: 24.927680026315617,
    imp: 0.0,
    cmap: 0.0,
    vdw_14: 14.82205711501749,
    elec_14: 45.91585188756373,
    vdw: 2603.9592048281297,
    elec: -79.80600251152761,
};
const SANDER_GAFF2: Terms = Terms {
    bond: 117.0577981504619,
    angle: 47.47024214207903,
    angle_ub: 0.0,
    dihedral: 24.288713569806802,
    imp: 0.0,
    cmap: 0.0,
    vdw_14: 6.721895284314586,
    elec_14: -394.97568321248525,
    vdw: -0.23843882018212811,
    elec: 144.73989274461692,
};
const SANDER_GAFF2_MULTI: Terms = Terms {
    bond: 117.0577981504619,
    angle: 47.47024214207903,
    angle_ub: 0.0,
    dihedral: 24.924762548996824,
    imp: 0.0,
    cmap: 0.0,
    vdw_14: 6.721895284314586,
    elec_14: -394.97568321248525,
    vdw: -0.23843882018212811,
    elec: 144.73989274461692,
};
const SANDER_GLYCAM: Terms = Terms {
    bond: 171.17712247984358,
    angle: 85.5319850750555,
    angle_ub: 0.0,
    dihedral: 22.159930988198603,
    imp: 0.0,
    cmap: 0.0,
    vdw_14: 18.2263946339782,
    elec_14: 295.2912756262633,
    vdw: 2.1250142784899535,
    elec: -214.20799587256528,
};
const SANDER_CHAMBER: Terms = Terms {
    bond: 205.89800531515192,
    angle: 59.30625820683123,
    angle_ub: 8.944012056753,
    dihedral: 32.219204827155565,
    imp: 1.0718540909516483,
    cmap: 0.2918309789686523,
    vdw_14: 8.88251947738584,
    elec_14: 228.71924198894848,
    vdw: 0.9048243974523592,
    elec: -174.78535680449875,
};
const SANDER_FF19SB: Terms = Terms {
    bond: 174.05366902482228,
    angle: 51.08419003689579,
    angle_ub: 0.0,
    dihedral: 4.938822301906974,
    imp: 0.0,
    cmap: -0.8896915180043046,
    vdw_14: 5.7556677548360975,
    elec_14: 44.86027661827858,
    vdw: 2.548125126241162,
    elec: -77.70909732408734,
};
// LAMMPS 30 Mar 2026 develop, `run 0` (`scripts/prmtop_check.sh`), the
// Coulomb terms rescaled from LAMMPS's `qqr2e` to the field's constant.
const LAMMPS_FF14SB: Terms = Terms {
    bond: 93.29782311534443,
    angle: 62.814326305362705,
    angle_ub: 0.0,
    dihedral: 24.927680026315624,
    imp: 0.0,
    cmap: 0.0,
    vdw_14: 14.822057120452428,
    elec_14: 45.91585188756363,
    vdw: 2603.959200896045,
    elec: -79.8060025115276,
};
const LAMMPS_GAFF2: Terms = Terms {
    bond: 117.05779815046192,
    angle: 47.47024214207903,
    angle_ub: 0.0,
    dihedral: 24.288713569806845,
    imp: 0.0,
    cmap: 0.0,
    vdw_14: 6.721895282447976,
    elec_14: -394.9756832124852,
    vdw: -0.23843881871852013,
    elec: 144.73989274461684,
};
const LAMMPS_GAFF2_MULTI: Terms = Terms {
    bond: 117.05779815046192,
    angle: 47.47024214207903,
    angle_ub: 0.0,
    dihedral: 24.924762548996874,
    imp: 0.0,
    cmap: 0.0,
    vdw_14: 6.721895282447976,
    elec_14: -394.9756832124852,
    vdw: -0.23843881871852013,
    elec: 144.73989274461684,
};
const LAMMPS_GLYCAM: Terms = Terms {
    bond: 171.1771224798436,
    angle: 85.53198507505556,
    angle_ub: 0.0,
    dihedral: 22.15993098819862,
    imp: 0.0,
    cmap: 0.0,
    vdw_14: 18.22639462598311,
    elec_14: 295.2912756262632,
    vdw: 2.125014279590698,
    elec: -214.20799587256522,
};
const LAMMPS_CHAMBER: Terms = Terms {
    bond: 205.89800531515192,
    angle: 59.3062582068312,
    angle_ub: 8.94401205675301,
    dihedral: 32.21920482715558,
    imp: 1.0718540909517282,
    cmap: 0.2918309789686183,
    vdw_14: 8.88251947738583,
    elec_14: 228.7192419889485,
    vdw: 0.9048243974523584,
    elec: -174.78535680449863,
};
const LAMMPS_FF19SB: Terms = Terms {
    bond: 174.05366902482228,
    angle: 51.084190036895805,
    angle_ub: 0.0,
    dihedral: 4.9388223019069875,
    imp: 0.0,
    cmap: -0.8896915180043052,
    vdw_14: 5.755667759004686,
    elec_14: 44.8602766182786,
    vdw: 2.5481251293409106,
    elec: -77.70909732408734,
};

/// The switch of a CHARMM pair style, beyond every pair of the cases.
const INNER: F = 50.0;
const CUTOFF: F = 60.0;

/// `case`'s frame (with its full pair list), force field and coordinates.
fn system(case: &Case) -> (Frame, ForceField, Vec<F>) {
    let (mut ff, mut frame) = AmberPrmtopForcefieldReader::new()
        .read_system(case.parm.as_bytes())
        .unwrap_or_else(|e| panic!("{}: system: {e}", case.name));
    // A prmtop states no cutoff; the CHARMM styles need theirs declared.
    for name in ["lj/charmm", "coul/charmm"] {
        if let Some(style) = ff.get_style_mut("pair", name) {
            style.set_param("inner", INNER);
            style.set_param("cutoff", CUTOFF);
        }
    }
    // The full pair list (keeping the system reader's per-pair 1-4 scales),
    // then the field's 1-4 pricing on it: a chamber file's `one_four =
    // "epsilon14"` 1-4 pairs need their ε₁₄/σ₁₄ as override cells.
    let pairs = intramolecular_pairs(&frame, ff.special_bonds(), None)
        .unwrap_or_else(|e| panic!("{}: pairs: {e}", case.name));
    frame.insert("pairs", pairs);
    ff.materialize_one_four(&mut frame)
        .unwrap_or_else(|e| panic!("{}: materialize_one_four: {e}", case.name));
    let xyz = read_inpcrd(case.rst.as_bytes()).unwrap();
    let atoms = xyz.get("atoms").unwrap();
    let col = |k: &str| atoms.get(k).unwrap().get::<f64>().unwrap().to_owned();
    let (x, y, z) = (col("x"), col("y"), col("z"));
    let coords = (0..x.len())
        .flat_map(|i| [x[[i]], y[[i]], z[[i]]])
        .collect();
    (frame, ff, coords)
}

/// `ff` with only its `(category, name)` style, each type's params mapped
/// by `edit`.
fn only(ff: &ForceField, category: &str, name: &str, edit: impl Fn(&mut Params)) -> ForceField {
    let mut one = ff.empty_like();
    let style = ff.get_style(category, name).unwrap();
    let s = one
        .def_style(category, name, style.params().clone())
        .unwrap();
    for (type_name, ends, params) in style.type_rows() {
        let mut params = params.clone();
        edit(&mut params);
        s.def_type(type_name, &ends, params).unwrap();
    }
    one
}

/// `frame` with only the `pairs` rows whose `is_14` is `want`.
fn pairs_where(frame: &Frame, want: bool) -> Frame {
    let pairs = frame.get("pairs").unwrap();
    let flags = pairs.get("is_14").unwrap().get::<bool>().unwrap();
    let keep: Vec<usize> = (0..flags.len()).filter(|&r| flags[[r]] == want).collect();
    let mut out = frame.clone();
    out.insert("pairs", pairs.select_rows(&keep).unwrap());
    out
}

fn energy(ff: &ForceField, frame: &Frame, coords: &[F]) -> F {
    PotentialCompiler::new(ff)
        .compile(frame)
        .unwrap()
        .energy_at(coords)
}

/// The energy of `(category, name)` alone, 0 when `ff` has no such style.
fn term(
    ff: &ForceField,
    frame: &Frame,
    coords: &[F],
    style: (&str, &str),
    edit: impl Fn(&mut Params),
) -> F {
    if ff.get_style(style.0, style.1).is_none() {
        return 0.0;
    }
    energy(&only(ff, style.0, style.1, edit), frame, coords)
}

fn molrs_terms(frame: &Frame, ff: &ForceField, coords: &[F]) -> Terms {
    let keep = |_: &mut Params| {};
    let charmm = ff.get_style("angle", "charmm").is_some();
    let (lj, coul) = if ff.get_style("pair", "lj/charmm").is_some() {
        (("pair", "lj/charmm"), ("pair", "coul/charmm"))
    } else {
        (("pair", "lj/cut"), ("pair", "coul/cut"))
    };
    let (near, far) = (pairs_where(frame, true), pairs_where(frame, false));
    Terms {
        bond: term(ff, frame, coords, ("bond", "harmonic"), keep),
        angle: if charmm {
            term(ff, frame, coords, ("angle", "charmm"), |p| {
                p.set("k_ub", 0.0)
            })
        } else {
            term(ff, frame, coords, ("angle", "harmonic"), keep)
        },
        angle_ub: term(ff, frame, coords, ("angle", "charmm"), |p| p.set("k", 0.0)),
        dihedral: term(ff, frame, coords, ("dihedral", "periodic"), keep)
            + term(ff, frame, coords, ("improper", "periodic"), keep),
        imp: term(ff, frame, coords, ("improper", "harmonic"), keep),
        cmap: term(ff, frame, coords, ("cmap", "charmm"), keep),
        vdw_14: term(ff, &near, coords, lj, keep),
        elec_14: term(ff, &near, coords, coul, keep),
        vdw: term(ff, &far, coords, lj, keep),
        elec: term(ff, &far, coords, coul, keep),
    }
}

fn close(case: &str, label: &str, got: F, want: F, rel: F) {
    let err = (got - want).abs() / want.abs().max(1e-300);
    assert!(
        err <= rel || (got - want).abs() <= 1e-12,
        "{case} {label}: molrs {got:?} vs {want:?} (rel {err:e})"
    );
}

#[test]
fn each_term_matches_sander_and_lammps() {
    let dir = std::env::var_os("MOLRS_PRMTOP_LAMMPS_DIR");
    for case in cases() {
        let (frame, ff, coords) = system(&case);
        let got = molrs_terms(&frame, &ff, &coords);
        if let Some(dir) = &dir {
            let dir = Path::new(dir).join(case.name);
            std::fs::create_dir_all(&dir).unwrap();
            for (label, value) in got.named() {
                println!("molrs {} {label} {value:.17e}", case.name);
            }
            write_lammps(&dir, &frame, &ff, &coords);
            continue;
        }
        for (((label, g), (_, sander)), (_, lammps)) in got
            .named()
            .into_iter()
            .zip(case.sander.named())
            .zip(case.lammps.named())
        {
            close(case.name, &format!("{label} (sander)"), g, sander, 1e-6);
            close(case.name, &format!("{label} (LAMMPS)"), g, lammps, 1e-10);
        }
    }
}

/// Write `frame` and `ff` as a LAMMPS data file and include in `dir`. LAMMPS
/// has no per-pair 1-4 scale and no `one_four = "epsilon14"`: the data file
/// carries no override columns and the include no flag, and the script
/// prices the 1-4 pairs those would change itself (see its comments).
fn write_lammps(dir: &Path, frame: &Frame, ff: &ForceField, coords: &[F]) {
    let mut lammps = frame.clone();
    lammps.remove("pairs");
    let mut atoms = lammps.get("atoms").unwrap().clone();
    for (axis, key) in ["x", "y", "z"].into_iter().enumerate() {
        let col: Vec<F> = coords.iter().skip(axis).step_by(3).copied().collect();
        atoms
            .insert(key, ndarray::Array1::from_vec(col).into_dyn(), None)
            .unwrap();
    }
    if atoms.get("mol_id").is_none() {
        let mol = molecules(frame);
        atoms
            .insert("mol_id", ndarray::Array1::from_vec(mol).into_dyn(), None)
            .unwrap();
    }
    lammps.insert("atoms", atoms);
    lammps.simbox = Some(
        SimBox::new(
            Cell::Lengths([400.0; 3]),
            [-200.0, -200.0, -200.0],
            [false; 3],
        )
        .unwrap(),
    );
    let mut ff = without_style_param(ff, "one_four");
    for style in ["lj/cut", "coul/cut"] {
        if let Some(s) = ff.get_style_mut("pair", style) {
            s.set_param("cutoff", CUTOFF);
        }
    }
    write_lammps_data(dir.join("data.lmp").as_path(), &lammps).unwrap();
    let labels = TypeLabels::from_frame(&lammps).unwrap();
    let writer = match lammps.get("cmaps").map(|_| "charmm.cmap".to_owned()) {
        Some(file) => LammpsForcefieldWriter::new(&labels)
            .with_precision(17)
            .with_cmap_file(file),
        None => LammpsForcefieldWriter::new(&labels).with_precision(17),
    };
    if lammps.get("cmaps").is_some() {
        std::fs::write(
            dir.join("charmm.cmap"),
            crate::io::writer::written_with(|b| writer.write_cmap(b, &ff)).unwrap(),
        )
        .unwrap();
    }
    std::fs::write(dir.join("system.ff"), writer.written(&ff).unwrap()).unwrap();
    let coul = ["coul/cut", "coul/charmm"]
        .iter()
        .find_map(|s| ff.get_style("pair", s))
        .unwrap();
    std::fs::write(
        dir.join("coulomb.txt"),
        format!(
            "{:?}\n",
            coul.params()
                .get("coulomb")
                .and_then(|v| v.as_num())
                .unwrap()
        ),
    )
    .unwrap();
    // Per-molecule 1-4 weights, when some differ from `special_bonds` (a
    // `materialize_one_four` cell restating them changes nothing).
    if let Some(weights) = molecule_weights(frame, &ff) {
        std::fs::write(dir.join("weights.txt"), weights).unwrap();
    }
}

/// `weights <molecule> <coul> <lj>` per molecule: the 1-4 weights of its
/// pairs (the `pairs` override cell, else `special_bonds`), which must be
/// one per molecule for the script to apply them; `None` when every one is
/// the `special_bonds` weight.
fn molecule_weights(frame: &Frame, ff: &ForceField) -> Option<String> {
    let mol = molecules(frame);
    let pairs = frame.get("pairs")?;
    let col = |k: &str| pairs.get(k).unwrap().get::<u64>().unwrap().to_owned();
    let (ai, flags) = (
        col("atomi"),
        pairs.get("is_14").unwrap().get::<bool>().unwrap(),
    );
    let cell = |k: &str, r: usize, default: F| -> F {
        match (pairs.get(k), pairs.validity(k)) {
            (Some(c), valid) if valid.is_none_or(|m| m[r]) => c.get::<f64>().unwrap()[[r]],
            _ => default,
        }
    };
    let sb = ff.special_bonds();
    let mut of: std::collections::BTreeMap<molrs::op::Idx, (F, F)> = Default::default();
    for r in (0..flags.len()).filter(|&r| flags[[r]]) {
        let w = (
            cell("coul_scale", r, sb.coul[2]),
            cell("lj_scale", r, sb.lj[2]),
        );
        let m = mol[ai[[r]] as usize];
        let prev = *of.entry(m).or_insert(w);
        assert!(
            (prev.0 - w.0).abs() < 1e-12 && (prev.1 - w.1).abs() < 1e-12,
            "molecule {m} has two 1-4 weights"
        );
    }
    if of.values().all(|&(c, l)| c == sb.coul[2] && l == sb.lj[2]) {
        return None;
    }
    Some(
        of.iter()
            .map(|(m, (c, l))| format!("weights {m} {c:?} {l:?}\n"))
            .collect(),
    )
}

/// A copy of `ff` whose styles carry no string or number parameter `key`.
fn without_style_param(ff: &ForceField, key: &str) -> ForceField {
    let mut out = ff.empty_like();
    for style in ff.styles() {
        let mut params = Params::new();
        for (k, v) in style
            .params()
            .iter()
            .filter_map(|(k, v)| v.as_num().map(|x| (k, x)))
            .filter(|(k, _)| *k != key)
        {
            params.set(k, v);
        }
        for (k, v) in style
            .params()
            .iter()
            .filter_map(|(k, v)| v.as_text().map(|s| (k, s)))
            .filter(|(k, _)| *k != key)
        {
            params.set(k, v);
        }
        for (k, v) in style
            .params()
            .iter()
            .filter_map(|(k, v)| v.as_array().map(|a| (k, a)))
        {
            params.set(k, v.clone());
        }
        let s = out
            .def_style(style.category(), style.name(), params)
            .unwrap();
        for (name, ends, p) in style.type_rows() {
            s.def_type(name, &ends, p.clone()).unwrap();
        }
    }
    out
}

/// The 1-based molecule of each atom: the bond graph's connected components,
/// numbered in order of their first atom.
fn molecules(frame: &Frame) -> Vec<molrs::op::Idx> {
    let n = frame.get("atoms").unwrap().n_rows().unwrap();
    let mut root: Vec<usize> = (0..n).collect();
    fn find(root: &mut [usize], a: usize) -> usize {
        let mut a = a;
        while root[a] != a {
            root[a] = root[root[a]];
            a = root[a];
        }
        a
    }
    let bonds = frame.get("bonds").unwrap();
    let (i, j) = (
        bonds.get("atomi").unwrap().get::<u64>().unwrap(),
        bonds.get("atomj").unwrap().get::<u64>().unwrap(),
    );
    for (&a, &b) in i.iter().zip(j.iter()) {
        let (ra, rb) = (find(&mut root, a as usize), find(&mut root, b as usize));
        root[ra.max(rb)] = ra.min(rb);
    }
    let mut id_of = std::collections::HashMap::new();
    (0..n)
        .map(|a| {
            let r = find(&mut root, a);
            let next = id_of.len() as molrs::op::Idx + 1;
            *id_of.entry(r).or_insert(next)
        })
        .collect()
}

/// molrs's own GAFF2 typing of the `gaff2` case prices it as sander prices
/// tleap's prmtop, term by term: the atoms keep the prmtop's GAFF2 types
/// (antechamber's) and charges, and `GaffTypifier` assigns every bonded term
/// — `gaff2.dat` rows, parmchk2's estimates and tleap's impropers — from the
/// bond graph alone. The bonds are taken in the prmtop's order, which is not
/// the mol2 order parmchk2 and tleap saw; this molecule's terms do not depend
/// on it.
#[test]
fn gaff2_typing_prices_the_gaff2_case_as_sander() {
    use crate::ff::typifier::Typing;
    use crate::ff::typifier::{GaffParameterSet, GaffTypifier};
    use molrs::core::keys;

    let case = cases().into_iter().find(|c| c.name == "gaff2").unwrap();
    let (prmtop, _, coords) = system(&case);
    let atoms = prmtop.get("atoms").unwrap();
    let z = atoms
        .get(keys::ATOMIC_NUMBER)
        .unwrap()
        .get::<u64>()
        .unwrap();
    let types = atoms.get("type").unwrap().get::<String>().unwrap();
    let mut mol = molrs::core::Atomistic::new();
    let ids: Vec<_> = (0..z.len())
        .map(|i| {
            mol.add_atom(crate::props! {
                keys::ATOMIC_NUMBER => u8::try_from(z[[i]]).unwrap(),
                keys::X => coords[3 * i],
                keys::Y => coords[3 * i + 1],
                keys::Z => coords[3 * i + 2],
                keys::TYPE => types[[i]].as_str(),
            })
            .unwrap()
        })
        .collect();
    let bonds = prmtop.get("bonds").unwrap();
    let (bi, bj) = (
        bonds.get("atomi").unwrap().get::<u64>().unwrap(),
        bonds.get("atomj").unwrap().get::<u64>().unwrap(),
    );
    for (&i, &j) in bi.iter().zip(bj.iter()) {
        mol.add_bond(ids[i as usize], ids[j as usize], crate::props! {})
            .unwrap();
    }

    let mut gaff = Typing::new(GaffTypifier::new(GaffParameterSet::Gaff2));
    let typed = gaff.typify(&mol).unwrap();
    let ff = gaff.forcefield();
    let mut frame = typed.to_frame().unwrap();
    let mut typed_atoms = frame.get("atoms").unwrap().clone();
    typed_atoms
        .insert(
            "charge",
            atoms
                .get("charge")
                .unwrap()
                .get::<f64>()
                .unwrap()
                .to_owned(),
            None,
        )
        .unwrap();
    frame.insert("atoms", typed_atoms);
    let pairs = intramolecular_pairs(&frame, ff.special_bonds(), None).unwrap();
    frame.insert("pairs", pairs);

    // tleap converts θ₀ to radians with π = 3.141594 (as it writes a phase of
    // π), so the prmtop's θ₀ is 4.3e-7 relative above `gaff2.dat`'s degrees,
    // which molrs prices as written: ~1e-6 of this angle energy.
    let got = molrs_terms(&frame, ff, &coords);
    for ((label, g), (_, sander)) in got.named().into_iter().zip(case.sander.named()) {
        let rel = if label == "angle" { 1e-5 } else { 1e-6 };
        close("gaff2 typing", &format!("{label} (sander)"), g, sander, rel);
    }
}
