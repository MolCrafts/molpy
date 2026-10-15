//! The force-field IR across engines: one molecule per force-field family,
//! read from its native format, written to every engine format that can hold
//! it, and priced by each engine, term by term, at several configurations.
//!
//! | Source | Family | Native format (engine) |
//! |---|---|---|
//! | `ff14sb` | AMBER ff14SB, ACE-PHE-NME | prmtop (sander) |
//! | `gaff2` | GAFF2 | prmtop (sander) |
//! | `chamber` | CHARMM36 (Urey–Bradley, CHARMM impropers, CMAP, 1-4 table), ALAD + glucose | chamber prmtop (sander) |
//! | `charmm36` | CHARMM36 (Urey–Bradley, CMAP, `sigma14`/`epsilon14`, an NBFIX row), ACE-ALA-NME | OpenMM XML (OpenMM) |
//! | `oplsaa` | OPLS-AA (RB, geometric mixing, funct-1 impropers), ACE-ALA-ALA-NME | GROMACS `.top` (GROMACS) |
//!
//! Each source is read into the IR (a [`ForceField`] and a typed [`Frame`]);
//! the IR is written as a LAMMPS data file and include ([`LammpsForcefieldWriter`]),
//! an OpenMM `<ForceField>` XML ([`OpenmmXmlWriter`], with residue
//! templates `scripts/ff_equivalence_check.py` builds from the frame) and a
//! GROMACS topology ([`GromacsTopForcefieldWriter::write_system`]); every file,
//! and the source itself in its native engine, is priced at [`CONFIGS`]
//! configurations — the source's coordinates plus a seeded 0.05 Å Gaussian,
//! on the 0.01 Å grid a `.gro` file holds exactly.
//!
//! Where an engine's form of a field is not the IR as read, the form written
//! is an **exact** rewrite of it, priced by molrs to the same energy (held by
//! [`each_engine_form_prices_as_the_source`]):
//!
//! - LAMMPS: a `lj/charmm` field with `one_four = "epsilon14"` (CHARMM's 1-4
//!   table) as `special_bonds` 0 and one zero-`K` `dihedral charmm` row with
//!   `w` = the 1-4 weight per 1-4 pair ([`one_four_as_dihedral_weights`]);
//! - OpenMM: per-atom charges in the residue templates (no type charge); a
//!   `dihedral periodic` row on atoms that are no bonded chain (OPLS-AA's
//!   impropers, GROMACS funct 1) as the `improper periodic` it is
//!   ([`chainless_dihedrals_as_impropers`]): OpenMM generates `<Proper>`
//!   torsions along bonds only.
//!
//! The engines' numbers are pinned in `testdata/equivalence/engines.tsv`,
//! which `scripts/ff_equivalence_check.sh --pin` writes on a compute node
//! (LAMMPS `run 0`; OpenMM 8.6.1 Reference, `NoCutoff`, one force group per
//! term; GROMACS 2025.3 double precision, `mdrun -rerun`, plain cut-off past
//! every pair; pysander, AmberTools 26.1, `cut = 999`). Each pinned value is
//! held to molrs's energy of the IR form that engine read, its Coulomb term
//! at that engine's own Coulomb constant ([`coulomb_of`]): the energy is
//! linear in it, and the constants differ (LAMMPS `real` 332.06371; OpenMM
//! and GROMACS CODATA 2018, 9.9·10⁻⁹ above; AMBER 332.0522173, 3.5·10⁻⁵
//! below). Forces enter as two numbers per configuration, ΣF·v (v a seeded
//! Gaussian vector) and Σ|F|².
//!
//! The terms are `bond`, `angle` (with Urey–Bradley), `dihedral`,
//! `improper`, `cmap`, `vdw` and `coul` (each with its 1-4 pairs) and
//! `total`; sander counts AMBER impropers in its `DIHED`, so its `dihedral`
//! is molrs's `dihedral` plus `improper periodic`.

// The engines' numbers are kept as they printed them.
#![allow(clippy::excessive_precision)]

use crate::core::Cell;
use crate::ff::potential::fixtures::OneShot;
use crate::io::writer::WrittenText;
use std::collections::{BTreeMap, HashMap, HashSet};
use std::fmt::Write as _;
use std::io::Cursor;
use std::path::Path;

use ndarray::Array1;
use serde_json::{Value, json};

use crate::ff::compile::PotentialCompiler;
use crate::ff::compile::exceptions;
use crate::ff::forcefield::{ForceField, Style};
use crate::ff::ir::{Params, SpecialBonds};
use crate::ff::potential::intramolecular_pairs;
use crate::io::amber::prmtop_forcefield::AmberPrmtopForcefieldReader;
use crate::io::gromacs::top_reader::GromacsTopForcefieldReader;
use crate::io::gromacs::top_writer::GromacsTopForcefieldWriter;
#[cfg(feature = "zarr")]
use crate::io::mrec::ForceFieldSection;
use crate::io::openmm_xml::reader::OpenmmXmlReader;
use crate::io::openmm_xml::writer::OpenmmXmlWriter;
use crate::io::reader::ForceFieldReader;
use crate::io::{lammps::LammpsForcefieldReader, lammps::LammpsForcefieldWriter};
use molrs::core::Block;
use molrs::core::Frame;
use molrs::core::SimBox;
use molrs::core::TypeLabels;
use molrs::core::constants::COULOMB_REAL;
use molrs::core::schema::PAIR_OVERRIDE_COLUMNS;
use molrs::io::gro::GroReader;
use molrs::io::lammps::data::{read_lammps_data, write_lammps_data};
use molrs::io::read_inpcrd;
use molrs::io::reader::FrameReader as _;
use molrs::op::{F, Idx};

/// The terms compared, in print order.
pub(crate) const TERMS: [&str; 8] = [
    "bond", "angle", "dihedral", "improper", "cmap", "vdw", "coul", "total",
];

/// Configurations per source.
pub(crate) const CONFIGS: usize = 3;

/// The engines, by the name the pinned table uses: the three writers' and
/// the source's own.
pub(crate) const ENGINES: [&str; 4] = ["native", "lammps", "openmm", "gromacs"];

/// A cutoff past every pair (Å), and CHARMM's switch past them too: every
/// engine's nonbonded energy is the plain sum.
const NO_CUTOFF: F = 1000.0;
const NO_SWITCH: F = 900.0;

/// The native engine of a source.
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub(crate) enum Native {
    Sander,
    OpenMm,
    Gromacs,
}

impl Native {
    fn name(self) -> &'static str {
        match self {
            Native::Sander => "sander",
            Native::OpenMm => "openmm",
            Native::Gromacs => "gromacs",
        }
    }
}

pub(crate) struct Source {
    pub name: &'static str,
    pub native: Native,
    /// The native file, relative to the repository root (for the script).
    pub file: &'static str,
    load: fn() -> System,
}

/// A source read into the IR: the force field (cutoffs past every pair), its
/// frame (with `pairs`; 1-4 pairs not materialized) and the source's
/// coordinates.
#[derive(Clone)]
pub(crate) struct System {
    pub ff: ForceField,
    pub frame: Frame,
    pub coords: Vec<F>,
}

pub(crate) fn sources() -> Vec<Source> {
    vec![
        Source {
            name: "ff14sb",
            native: Native::Sander,
            file: "molrs/src/io/amber/testdata/prmtop/ff14sb.parm7",
            load: || {
                prmtop(
                    include_str!("../io/amber/testdata/prmtop/ff14sb.parm7"),
                    include_str!("../io/amber/testdata/prmtop/ff14sb.rst7"),
                )
            },
        },
        Source {
            name: "gaff2",
            native: Native::Sander,
            file: "molrs/src/io/amber/testdata/prmtop/gaff2.parm7",
            load: || {
                prmtop(
                    include_str!("../io/amber/testdata/prmtop/gaff2.parm7"),
                    include_str!("../io/amber/testdata/prmtop/gaff2.rst7"),
                )
            },
        },
        Source {
            name: "chamber",
            native: Native::Sander,
            file: "molrs/src/io/amber/testdata/prmtop/chamber.parm7",
            load: || {
                prmtop(
                    include_str!("../io/amber/testdata/prmtop/chamber.parm7"),
                    include_str!("../io/amber/testdata/prmtop/chamber.rst7"),
                )
            },
        },
        Source {
            name: "charmm36",
            native: Native::OpenMm,
            file: "molrs/src/ff/testdata/openmm/charmm.xml",
            load: openmm_charmm36,
        },
        Source {
            name: "oplsaa",
            native: Native::Gromacs,
            file: "molrs/src/io/gromacs/testdata/opls.top",
            load: || {
                gromacs(
                    include_str!("../io/gromacs/testdata/opls.top"),
                    include_str!("../io/gromacs/testdata/opls.gro"),
                )
            },
        },
    ]
}

impl Source {
    pub(crate) fn load(&self) -> System {
        (self.load)()
    }
}

/// Every pair style's cutoff past every pair, CHARMM's switch past them too.
fn no_cutoff(ff: &mut ForceField) {
    for name in ["lj/cut", "coul/cut", "lj/charmm", "coul/charmm"] {
        if let Some(style) = ff.get_style_mut("pair", name) {
            style.set_param("cutoff", NO_CUTOFF);
            if name.ends_with("charmm") {
                style.set_param("inner", NO_SWITCH);
            }
        }
    }
}

fn prmtop(parm: &str, rst: &str) -> System {
    let (mut ff, mut frame) = AmberPrmtopForcefieldReader::new()
        .read_system(parm.as_bytes())
        .unwrap();
    no_cutoff(&mut ff);
    let pairs = intramolecular_pairs(&frame, ff.special_bonds(), None).unwrap();
    frame.insert("pairs", pairs);
    let xyz = read_inpcrd(rst.as_bytes()).unwrap();
    let atoms = xyz.get("atoms").unwrap();
    let col = |k: &str| atoms.get(k).unwrap().get::<f64>().unwrap().to_owned();
    let (x, y, z) = (col("x"), col("y"), col("z"));
    let coords = (0..x.len())
        .flat_map(|i| [x[[i]], y[[i]], z[[i]]])
        .collect();
    System { ff, frame, coords }
}

fn openmm_charmm36() -> System {
    let case = &crate::ff::openmm_check::CASES[0];
    let ff = crate::ff::openmm_check::read(case);
    let frame = crate::ff::openmm_check::frame(case, &ff);
    let coords = frame.coords().unwrap().into_iter().collect();
    System { ff, frame, coords }
}

fn gromacs(top: &str, gro: &str) -> System {
    let (mut ff, mut frame) = GromacsTopForcefieldReader::new()
        .read_system(top.as_bytes())
        .unwrap();
    no_cutoff(&mut ff);
    let gro = GroReader::new(Cursor::new(gro)).read().unwrap().unwrap();
    let g = gro.get("atoms").unwrap();
    let atoms = frame.get_mut("atoms").unwrap();
    for key in ["x", "y", "z"] {
        atoms
            .insert(
                key,
                g.get(key).unwrap().get::<f64>().unwrap().to_owned(),
                None,
            )
            .unwrap();
    }
    let coords = frame.coords().unwrap().into_iter().collect();
    System { ff, frame, coords }
}

// ── configurations ──────────────────────────────────────────────────────────

/// SplitMix64: a seeded stream both this test and its readers can name.
struct Rng(u64);

impl Rng {
    fn next(&mut self) -> u64 {
        self.0 = self.0.wrapping_add(0x9E37_79B9_7F4A_7C15);
        let mut z = self.0;
        z = (z ^ (z >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
        z = (z ^ (z >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
        z ^ (z >> 31)
    }

    fn uniform(&mut self) -> F {
        ((self.next() >> 11) as F + 0.5) / (1u64 << 53) as F
    }

    /// A standard normal deviate (Box–Muller).
    fn gauss(&mut self) -> F {
        let (u, v) = (self.uniform(), self.uniform());
        (-2.0 * u.ln()).sqrt() * (2.0 * std::f64::consts::PI * v).cos()
    }
}

fn seed(source: &str, config: usize, salt: u64) -> u64 {
    source
        .bytes()
        .fold(0xC0FF_EE00_u64 ^ salt, |h, b| {
            h.wrapping_mul(0x100_0000_01B3) ^ u64::from(b)
        })
        .wrapping_add(config as u64)
}

/// Configuration `k` of a source: its coordinates plus a 0.05 Å Gaussian,
/// rounded to 0.01 Å (a `.gro` file's 0.001 nm).
pub(crate) fn configuration(source: &str, base: &[F], k: usize) -> Vec<F> {
    let mut rng = Rng(seed(source, k, 1));
    base.iter()
        .map(|x| ((x + 0.05 * rng.gauss()) * 100.0).round() / 100.0)
        .collect()
}

/// The force fingerprint's vector of configuration `k`.
pub(crate) fn probe(source: &str, n: usize, k: usize) -> Vec<F> {
    let mut rng = Rng(seed(source, k, 2));
    (0..n).map(|_| rng.gauss()).collect()
}

// ── IR rewrites ─────────────────────────────────────────────────────────────

/// `p` without `key`, on every side.
fn without(p: &Params, key: &str) -> Params {
    let mut out = Params::new();
    for (k, v) in p
        .iter()
        .filter_map(|(k, v)| v.as_num().map(|x| (k, x)))
        .filter(|(k, _)| *k != key)
    {
        out.set(k, v);
    }
    for (k, v) in p
        .iter()
        .filter_map(|(k, v)| v.as_text().map(|s| (k, s)))
        .filter(|(k, _)| *k != key)
    {
        out.set(k, v);
    }
    for (k, v) in p
        .iter()
        .filter_map(|(k, v)| v.as_array().map(|a| (k, a)))
        .filter(|(k, _)| *k != key)
    {
        out.set(k, v.clone());
    }
    out
}

/// A copy of `ff` with each style's params and each type's params mapped.
fn rebuild(
    ff: &ForceField,
    style_params: impl Fn(&Style) -> Params,
    type_params: impl Fn(&Style, &Params) -> Params,
) -> ForceField {
    let mut out = ff.empty_like();
    for style in ff.styles() {
        let s = out
            .def_style(style.category(), style.name(), style_params(style))
            .unwrap();
        for (name, ends, p) in style.type_rows() {
            s.def_type(name, &ends, type_params(style, p)).unwrap();
        }
    }
    out
}

fn idx_col(frame: &Frame, block: &str, key: &str) -> Vec<usize> {
    frame
        .get(block)
        .and_then(|b| b.get(key))
        .and_then(|c| c.get::<u64>())
        .map(|c| c.iter().map(|&v| v as usize).collect())
        .unwrap_or_default()
}

fn adjacency(frame: &Frame) -> Vec<Vec<usize>> {
    let n = frame.get("atoms").unwrap().n_rows().unwrap();
    let mut adjacent = vec![Vec::new(); n];
    for (i, j) in idx_col(frame, "bonds", "atomi")
        .into_iter()
        .zip(idx_col(frame, "bonds", "atomj"))
    {
        adjacent[i].push(j);
        adjacent[j].push(i);
    }
    adjacent
}

/// The rows of a relation block: atoms and type.
fn relation_rows(frame: &Frame, block: &str, arity: usize) -> Vec<(Vec<usize>, String)> {
    let Some(b) = frame.get(block) else {
        return Vec::new();
    };
    let keys = ["atomi", "atomj", "atomk", "atoml", "atomm"];
    let cols: Vec<Vec<usize>> = keys[..arity]
        .iter()
        .map(|k| idx_col(frame, block, k))
        .collect();
    let types = b.get("type").unwrap().get::<String>().unwrap();
    (0..types.len())
        .map(|r| (cols.iter().map(|c| c[r]).collect(), types[[r]].clone()))
        .collect()
}

fn relation(rows: &[(Vec<usize>, String)], arity: usize) -> Block {
    let mut block = Block::new();
    for (k, key) in ["atomi", "atomj", "atomk", "atoml", "atomm"][..arity]
        .iter()
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

/// A path `i-a-b-j` of bonds between two atoms, if they are 1-4.
fn bonded_path(adjacent: &[Vec<usize>], i: usize, j: usize) -> Option<[usize; 4]> {
    for &a in &adjacent[i] {
        for &b in &adjacent[a] {
            if b != i && b != j && a != j && adjacent[b].contains(&j) {
                return Some([i, a, b, j]);
            }
        }
    }
    None
}

/// The `dihedral charmm` type of [`one_four_as_dihedral_weights`]'s rows.
pub(crate) const PAIR14: &str = "---@pairs";

/// A `one_four = "epsilon14"` field (CHARMM's 1-4 table) in the form LAMMPS
/// holds: `special_bonds` 0 and, per 1-4 pair of the frame, one zero-`K`
/// `dihedral charmm` row `w` = the 1-4 weight over a bonded path between its
/// atoms (LAMMPS prices `epsilon14`/`sigma14` only through `w`). The 1-4
/// weights of van der Waals and Coulomb must be one. Any other field is
/// itself.
pub(crate) fn one_four_as_dihedral_weights(ff: &ForceField, frame: &Frame) -> (ForceField, Frame) {
    let epsilon14 = ff
        .get_style("pair", "lj/charmm")
        .is_some_and(|s| s.params().get("one_four").and_then(|v| v.as_text()) == Some("epsilon14"));
    if !epsilon14 {
        return (ff.clone(), frame.clone());
    }
    let sb = *ff.special_bonds();
    assert_eq!(sb.lj_14(), sb.coul_14(), "one w prices both halves");
    let mut out = rebuild(
        ff,
        |s| {
            if s.name() == "lj/charmm" {
                without(s.params(), "one_four")
            } else {
                s.params().clone()
            }
        },
        |_, p| p.clone(),
    );
    out.set_special_bonds(SpecialBonds {
        lj: [0.0; 3],
        coul: [0.0; 3],
    });
    out.def_style("dihedral", "charmm", Params::new())
        .unwrap()
        .def_type(
            PAIR14,
            &["", "", "", ""],
            Params::from_iter([
                ("k", 0.0),
                ("periodicity", 1.0),
                ("phase", 0.0),
                ("w", sb.lj_14()),
            ]),
        )
        .unwrap();
    let adjacent = adjacency(frame);
    let pairs = frame.get("pairs").unwrap();
    let is_14 = pairs.get("is_14").unwrap().get::<bool>().unwrap();
    let (pi, pj) = (
        idx_col(frame, "pairs", "atomi"),
        idx_col(frame, "pairs", "atomj"),
    );
    let mut rows = relation_rows(frame, "dihedrals", 4);
    for r in 0..pi.len() {
        if is_14[[r]] {
            let path = bonded_path(&adjacent, pi[r], pj[r]).expect("a 1-4 pair is 1-4");
            rows.push((path.to_vec(), PAIR14.to_owned()));
        }
    }
    let mut out_frame = frame.clone();
    out_frame.insert("dihedrals", relation(&rows, 4));
    out_frame.insert(
        "pairs",
        intramolecular_pairs(&out_frame, out.special_bonds(), None).unwrap(),
    );
    (out, out_frame)
}

/// `dihedral periodic` rows on four atoms that are no bonded chain (OPLS-AA's
/// impropers, GROMACS funct 1 over an improper's atoms) as the same
/// `improper periodic` (one row and type `<type>@<m>` per term; the energy
/// k[1 + cos(nφ − γ)] of the stored atoms is the same function), so that
/// OpenMM's `<Improper>` generator, which prices the dihedral with the
/// centre — the third stored atom, bonded to the other three — third,
/// reaches them. A frame without such rows is itself.
pub(crate) fn chainless_dihedrals_as_impropers(
    ff: &ForceField,
    frame: &Frame,
) -> (ForceField, Frame) {
    let adjacent = adjacency(frame);
    let bonded = |a: usize, b: usize| adjacent[a].contains(&b);
    let rows = relation_rows(frame, "dihedrals", 4);
    let (chain, off): (Vec<_>, Vec<_>) = rows
        .into_iter()
        .partition(|(a, _)| bonded(a[0], a[1]) && bonded(a[1], a[2]) && bonded(a[2], a[3]));
    if off.is_empty() {
        return (ff.clone(), frame.clone());
    }
    let periodic = ff.get_style("dihedral", "periodic").unwrap();
    let mut out = ff.clone();
    let mut impropers = relation_rows(frame, "impropers", 4);
    let mut moved: BTreeMap<String, Vec<String>> = BTreeMap::new();
    for (atoms, name) in &off {
        let centre = atoms[2];
        assert!(
            [atoms[0], atoms[1], atoms[3]]
                .iter()
                .all(|&a| bonded(centre, a)),
            "{atoms:?}: the third atom is not the centre"
        );
        let names = moved.entry(name.clone()).or_insert_with(|| {
            let p = periodic.type_params(name).unwrap();
            let ends = periodic
                .type_rows()
                .into_iter()
                .find(|r| r.0 == name)
                .unwrap()
                .1
                .into_iter()
                .map(str::to_owned)
                .collect::<Vec<_>>();
            let ends: Vec<&str> = ends.iter().map(String::as_str).collect();
            let terms: Vec<Params> = if p.get("k1").and_then(|v| v.as_num()).is_none() {
                vec![p.clone()]
            } else {
                (1..)
                    .take_while(|m| p.get(&format!("k{m}")).and_then(|v| v.as_num()).is_some())
                    .map(|m| {
                        Params::from_iter([
                            (
                                "k",
                                p.get(&format!("k{m}")).and_then(|v| v.as_num()).unwrap(),
                            ),
                            (
                                "periodicity",
                                p.get(&format!("periodicity{m}"))
                                    .and_then(|v| v.as_num())
                                    .unwrap(),
                            ),
                            (
                                "phase",
                                p.get(&format!("phase{m}"))
                                    .and_then(|v| v.as_num())
                                    .unwrap_or(0.0),
                            ),
                        ])
                    })
                    .collect()
            };
            let style = out
                .def_style("improper", "periodic", Params::new())
                .unwrap();
            terms
                .into_iter()
                .enumerate()
                .map(|(m, t)| {
                    let n = format!("{name}@{}", m + 1);
                    style.def_type(&n, &ends, t).unwrap();
                    n
                })
                .collect()
        });
        for n in names.iter() {
            impropers.push((atoms.clone(), n.clone()));
        }
    }
    // The dihedral types only the moved rows used go with them.
    let still: HashSet<&str> = chain.iter().map(|r| r.1.as_str()).collect();
    let dihedral = out.get_style_mut("dihedral", "periodic").unwrap();
    for name in moved.keys() {
        if !still.contains(name.as_str()) {
            dihedral.del_type(name);
        }
    }
    // A style left without types is no style.
    if out
        .get_style("dihedral", "periodic")
        .is_some_and(|s| s.type_rows().is_empty())
    {
        let mut kept = out.empty_like();
        for style in out.styles() {
            if style.category() == "dihedral" && style.name() == "periodic" {
                continue;
            }
            let s = kept
                .def_style(style.category(), style.name(), style.params().clone())
                .unwrap();
            for (name, ends, p) in style.type_rows() {
                s.def_type(name, &ends, p.clone()).unwrap();
            }
        }
        out = kept;
    }
    let mut out_frame = frame.clone();
    out_frame.insert("dihedrals", relation(&chain, 4));
    out_frame.insert("impropers", relation(&impropers, 4));
    (out, out_frame)
}

/// The field without type charges (OpenMM's residue templates carry each
/// atom's own).
fn charges_per_atom(ff: &ForceField) -> ForceField {
    rebuild(
        ff,
        |s| s.params().clone(),
        |s, p| {
            if s.category() == "atom" {
                without(p, "charge")
            } else {
                p.clone()
            }
        },
    )
}

/// The IR form `engine` reads of `sys`: (field, frame).
pub(crate) fn engine_form(sys: &System, engine: &str) -> (ForceField, Frame) {
    match engine {
        "lammps" => one_four_as_dihedral_weights(&sys.ff, &sys.frame),
        "openmm" => {
            let (ff, frame) = chainless_dihedrals_as_impropers(&sys.ff, &sys.frame);
            (charges_per_atom(&ff), frame)
        }
        _ => (sys.ff.clone(), sys.frame.clone()),
    }
}

/// The Coulomb constant `engine` prices with (kcal·Å/(mol·e²)); the native
/// engine of a prmtop is AMBER's own, whose constant the IR states.
pub(crate) fn coulomb_of(source: &Source, ir: F, engine: &str) -> F {
    match (engine, source.native) {
        ("lammps", _) => COULOMB_REAL,
        ("openmm", _) | ("native", Native::OpenMm) => crate::core::constants::openmm_coulomb_real(),
        ("gromacs", _) | ("native", Native::Gromacs) => {
            crate::core::constants::gromacs_coulomb_real()
        }
        ("native", Native::Sander) => ir,
        _ => unreachable!("{engine}"),
    }
}

/// The Coulomb constant `ff` states.
pub(crate) fn coulomb(ff: &ForceField) -> F {
    ["coul/cut", "coul/charmm"]
        .iter()
        .find_map(|s| ff.get_style("pair", s))
        .and_then(|s| s.params().get("coulomb").and_then(|v| v.as_num()))
        .unwrap_or(COULOMB_REAL)
}

/// `ff` with its Coulomb style at constant `c`.
fn at_coulomb(ff: &ForceField, c: F) -> ForceField {
    let mut out = ff.clone();
    for name in ["coul/cut", "coul/charmm"] {
        if let Some(s) = out.get_style_mut("pair", name) {
            s.set_param("coulomb", c);
        }
    }
    out
}

// ── molrs's terms ───────────────────────────────────────────────────────────

/// `frame` without its per-pair override rows (the pair styles' own list).
fn without_overrides(frame: &Frame) -> Frame {
    let mut out = frame.clone();
    let Some(pairs) = frame.get("pairs") else {
        return out;
    };
    let n = pairs.n_rows().unwrap_or(0);
    let has = |r: usize| {
        PAIR_OVERRIDE_COLUMNS
            .iter()
            .any(|k| pairs.get(k).is_some() && pairs.validity(k).is_none_or(|m| m[r]))
    };
    let keep: Vec<usize> = (0..n).filter(|&r| !has(r)).collect();
    let mut kept = pairs.select_rows(&keep).unwrap();
    for k in PAIR_OVERRIDE_COLUMNS {
        kept.remove(k);
    }
    out.insert("pairs", kept);
    out
}

/// molrs's energy of `(ff, frame)` at `x` per term family, the Coulomb at
/// constant `c`; `sander` folds `improper periodic` into `dihedral`.
pub(crate) fn molrs_terms(
    ff: &ForceField,
    frame: &Frame,
    x: &[F],
    c: F,
    sander: bool,
) -> BTreeMap<&'static str, F> {
    let ff = &at_coulomb(ff, c);
    let mut frame = frame.clone();
    let one_four = ff
        .get_style("pair", "lj/charmm")
        .is_some_and(|s| s.params().get("one_four").and_then(|v| v.as_text()) == Some("epsilon14"));
    if one_four {
        ff.materialize_one_four(&mut frame).unwrap();
    }
    let plain = without_overrides(&frame);
    let mut out: BTreeMap<&'static str, F> = TERMS.iter().map(|t| (*t, 0.0)).collect();
    for style in ff.styles() {
        let term = match (style.category(), style.name()) {
            ("atom", _) => continue,
            ("improper", "periodic") if sander => "dihedral",
            ("bond", _) => "bond",
            ("angle", _) => "angle",
            ("dihedral", _) => "dihedral",
            ("improper", _) => "improper",
            ("cmap", _) => "cmap",
            ("pair", s) if s.starts_with("coul") => "coul",
            ("pair", _) => "vdw",
            (c, s) => unreachable!("{c}/{s}"),
        };
        // The style alone on the rows of its own types; its 1-4 semantics
        // are the exceptions kernel's, added below.
        let params = if style.name() == "lj/charmm" {
            let mut p = style.params().clone();
            p.set("one_four", "regular");
            p
        } else {
            style.params().clone()
        };
        let mut one = ff.empty_like();
        let s = one
            .def_style(style.category(), style.name(), params)
            .unwrap();
        for (name, ends, p) in style.type_rows() {
            let p = if p.get("w").and_then(|v| v.as_num()).is_some() {
                let mut p = p.clone();
                p.set("w", 0.0);
                p
            } else {
                p.clone()
            };
            s.def_type(name, &ends, p).unwrap();
        }
        let mut own = plain.clone();
        let block = match style.category() {
            "bond" => Some("bonds"),
            "angle" => Some("angles"),
            "dihedral" => Some("dihedrals"),
            "improper" => Some("impropers"),
            "cmap" => Some("cmaps"),
            _ => None,
        };
        if let Some(block) = block {
            let Some(b) = plain.get(block) else { continue };
            let names: HashSet<&str> = style.type_rows().iter().map(|r| r.0).collect();
            let labels = b.get("type").unwrap().get::<String>().unwrap();
            let keep: Vec<usize> = (0..labels.len())
                .filter(|&r| names.contains(labels[[r]].as_str()))
                .collect();
            if keep.is_empty() {
                continue;
            }
            own.insert(block, b.select_rows(&keep).unwrap());
        }
        *out.get_mut(term).unwrap() += PotentialCompiler::new(&one)
            .compile(&own)
            .unwrap()
            .energy_at(x);
    }
    let (lj14, coul14) = exceptions::plan(ff, &frame)
        .unwrap()
        .kernel
        .map(|k| k.energy_terms(x))
        .unwrap_or((0.0, 0.0));
    *out.get_mut("vdw").unwrap() += lj14;
    *out.get_mut("coul").unwrap() += coul14;
    out.insert(
        "total",
        PotentialCompiler::new(ff)
            .compile(&frame)
            .unwrap()
            .energy_at(x),
    );
    out
}

/// molrs's total forces of `(ff, frame)` at `x`, Coulomb at `c`.
pub(crate) fn molrs_forces(ff: &ForceField, frame: &Frame, x: &[F], c: F) -> Vec<F> {
    let ff = at_coulomb(ff, c);
    let mut frame = frame.clone();
    if ff
        .get_style("pair", "lj/charmm")
        .is_some_and(|s| s.params().get("one_four").and_then(|v| v.as_text()) == Some("epsilon14"))
    {
        ff.materialize_one_four(&mut frame).unwrap();
    }
    PotentialCompiler::new(&ff)
        .compile(&frame)
        .unwrap()
        .energy_forces_at(x)
        .1
}

/// The fingerprint of forces `f` on probe `v`: (ΣF·v, Σ|F|²).
pub(crate) fn fingerprint(f: &[F], v: &[F]) -> (F, F) {
    (
        f.iter().zip(v).map(|(a, b)| a * b).sum(),
        f.iter().map(|a| a * a).sum(),
    )
}

/// One engine's expected numbers, by molrs: the terms, then `fdotv`,
/// `fnorm2`.
pub(crate) fn expected(
    source: &Source,
    sys: &System,
    engine: &str,
    k: usize,
) -> BTreeMap<&'static str, F> {
    let (ff, frame) = engine_form(sys, engine);
    let c = coulomb_of(source, coulomb(&sys.ff), engine);
    let x = configuration(source.name, &sys.coords, k);
    let sander = engine == "native" && source.native == Native::Sander;
    let mut out = molrs_terms(&ff, &frame, &x, c, sander);
    let f = molrs_forces(&ff, &frame, &x, c);
    let (dot, norm2) = fingerprint(&f, &probe(source.name, x.len(), k));
    out.insert("fdotv", dot);
    out.insert("fnorm2", norm2);
    out
}

// ── the files each engine reads ─────────────────────────────────────────────

/// `frame` with coordinates `x` and a box around them (non-periodic).
fn placed(frame: &Frame, x: &[F]) -> Frame {
    let mut out = frame.clone();
    let atoms = out.get_mut("atoms").unwrap();
    for (d, key) in ["x", "y", "z"].iter().enumerate() {
        let col: Vec<F> = x.iter().skip(d).step_by(3).copied().collect();
        atoms
            .insert(*key, Array1::from_vec(col).into_dyn(), None)
            .unwrap();
    }
    let lo = x.iter().copied().fold(F::INFINITY, F::min) - 50.0;
    let hi = x.iter().copied().fold(F::NEG_INFINITY, F::max) + 50.0;
    out.simbox = Some(SimBox::new(Cell::Lengths([hi - lo; 3]), [lo, lo, lo], [false; 3]).unwrap());
    out
}

/// A data file a LAMMPS `atom_style full` reads: no pair list, exclusions
/// or constraints, a molecule id per atom.
fn lammps_frame(frame: &Frame) -> Frame {
    let mut out = frame.clone();
    for block in ["pairs", "exclusions", "constraints"] {
        out.remove(block);
    }
    let atoms = out.get_mut("atoms").unwrap();
    if atoms.get("mol_id").is_none() {
        let n = atoms.n_rows().unwrap();
        atoms
            .insert(
                "mol_id",
                Array1::from_vec(vec![1 as Idx; n]).into_dyn(),
                None,
            )
            .unwrap();
    }
    out
}

/// The LAMMPS include and cmap file of `(ff, frame)`: (`pre` lines that go
/// before `read_data`, the rest, the cmap file).
pub(crate) fn lammps_include(ff: &ForceField, frame: &Frame) -> (String, String, Option<String>) {
    let data = lammps_frame(frame);
    let labels = TypeLabels::from_frame(&data).unwrap();
    let has_cmap = data.get("cmaps").is_some();
    let writer = match has_cmap.then(|| "system.cmap".to_owned()) {
        Some(file) => LammpsForcefieldWriter::new(&labels)
            .with_precision(17)
            .with_cmap_file(file),
        None => LammpsForcefieldWriter::new(&labels).with_precision(17),
    };
    let text = writer.written(ff).unwrap();
    let (pre, post): (Vec<&str>, Vec<&str>) = text.lines().partition(|l| {
        l.starts_with("units") || l.starts_with("fix ") || l.starts_with("fix_modify")
    });
    (
        pre.join("\n") + "\n",
        post.join("\n") + "\n",
        has_cmap.then(|| crate::io::writer::written_with(|b| writer.write_cmap(b, ff)).unwrap()),
    )
}

/// A `.gro` of `frame` at `x` (Å; the 0.01 Å grid prints exactly) in a
/// 10 nm box.
fn gro(frame: &Frame, x: &[F]) -> String {
    let angstrom_to_nm = crate::core::unit_factors::ANGSTROM_TO_NM.get();
    let atoms = frame.get("atoms").unwrap();
    let n = atoms.n_rows().unwrap();
    let name = atoms.get("name").and_then(|c| c.get::<String>());
    let res = atoms.get("res_name").and_then(|c| c.get::<String>());
    let resid = atoms.get("res_id").and_then(|c| c.get::<u64>());
    let mut out = format!("molrs equivalence check\n{n}\n");
    for i in 0..n {
        let cut = |s: &str| s.chars().take(5).collect::<String>();
        writeln!(
            out,
            "{:>5}{:<5}{:>5}{:>5}{:8.3}{:8.3}{:8.3}",
            resid.map_or(1, |c| c[[i]]) % 100_000,
            cut(res.map_or("MOL", |c| c[[i]].as_str())),
            cut(&name.map_or_else(|| format!("A{}", i + 1), |c| c[[i]].clone())),
            (i + 1) % 100_000,
            x[3 * i] * angstrom_to_nm,
            x[3 * i + 1] * angstrom_to_nm,
            x[3 * i + 2] * angstrom_to_nm
        )
        .unwrap();
    }
    out.push_str("  10.00000  10.00000  10.00000\n");
    out
}

/// The OpenMM system description the script builds residue templates and a
/// topology from.
fn openmm_json(frame: &Frame, ff: &ForceField) -> Value {
    let atoms = frame.get("atoms").unwrap();
    let n = atoms.n_rows().unwrap();
    let types = atoms.get("type").unwrap().get::<String>().unwrap();
    let charges = atoms.get("charge").unwrap().get::<f64>().unwrap();
    let masses: Vec<F> = match atoms.get("mass").and_then(|c| c.get::<f64>()) {
        Some(m) => m.iter().copied().collect(),
        None => (0..n)
            .map(|i| {
                ff.get_atomtypes()
                    .into_iter()
                    .find(|t| t.name == types[[i]])
                    .and_then(|t| t.params.get("mass").and_then(|v| v.as_num()))
                    .unwrap()
            })
            .collect(),
    };
    let bonds: Vec<[usize; 2]> = idx_col(frame, "bonds", "atomi")
        .into_iter()
        .zip(idx_col(frame, "bonds", "atomj"))
        .map(|(i, j)| [i, j])
        .collect();
    let impropers: Vec<Vec<usize>> = relation_rows(frame, "impropers", 4)
        .into_iter()
        .map(|r| r.0)
        .collect();
    json!({
        "types": (0..n).map(|i| types[[i]].clone()).collect::<Vec<_>>(),
        "charges": charges.iter().copied().collect::<Vec<_>>(),
        "masses": masses,
        "bonds": bonds,
        "impropers": impropers,
    })
}

/// Write every engine's inputs of `source` into `dir/<source>/`, and molrs's
/// numbers for each into `molrs.tsv` lines.
fn stage_inputs(dir: &Path, source: &Source, tsv: &mut String) {
    let sys = source.load();
    let root = dir.join(source.name);
    for sub in ["lammps", "openmm", "gromacs"] {
        std::fs::create_dir_all(root.join(sub)).unwrap();
    }
    let n = sys.coords.len();
    let configs: Vec<Vec<F>> = (0..CONFIGS)
        .map(|k| configuration(source.name, &sys.coords, k))
        .collect();

    // LAMMPS.
    let (lff, lframe) = engine_form(&sys, "lammps");
    let (pre, include, cmap) = lammps_include(&lff, &lframe);
    std::fs::write(root.join("lammps/pre.lmp"), pre).unwrap();
    std::fs::write(root.join("lammps/system.ff"), include).unwrap();
    if let Some(cmap) = cmap {
        std::fs::write(root.join("lammps/system.cmap"), cmap).unwrap();
    }
    for (k, x) in configs.iter().enumerate() {
        write_lammps_data(
            &root.join(format!("lammps/data_{k}.lmp")),
            &lammps_frame(&placed(&lframe, x)),
        )
        .unwrap();
    }
    // OpenMM.
    let (off, oframe) = engine_form(&sys, "openmm");
    std::fs::write(
        root.join("openmm/ff.xml"),
        OpenmmXmlWriter::new().written(&off).unwrap(),
    )
    .unwrap();
    // GROMACS.
    let (gff, gframe) = engine_form(&sys, "gromacs");
    std::fs::write(
        root.join("gromacs/topol.top"),
        crate::io::writer::written_with(|b| {
            GromacsTopForcefieldWriter::new()
                .with_precision(17)
                .write_system(b, &gff, &gframe)
        })
        .unwrap(),
    )
    .unwrap();
    for (k, x) in configs.iter().enumerate() {
        std::fs::write(root.join(format!("gromacs/conf_{k}.gro")), gro(&gframe, x)).unwrap();
    }

    let mut system = openmm_json(&oframe, &off);
    system["source"] = json!(source.name);
    system["native"] = json!(source.native.name());
    system["native_file"] = json!(source.file);
    system["configs"] = json!(configs);
    system["probes"] = json!(
        (0..CONFIGS)
            .map(|k| probe(source.name, n, k))
            .collect::<Vec<_>>()
    );
    std::fs::write(
        root.join("system.json"),
        serde_json::to_string(&system).unwrap(),
    )
    .unwrap();

    for engine in ENGINES {
        for k in 0..CONFIGS {
            for (term, v) in expected(source, &sys, engine, k) {
                writeln!(tsv, "{}\t{k}\t{engine}\t{term}\t{v:?}", source.name).unwrap();
            }
        }
    }
}

// ── the pinned engine numbers ───────────────────────────────────────────────

/// `(source, config, engine) → term → value`, from the pinned table.
fn pinned() -> BTreeMap<(String, usize, String), BTreeMap<String, F>> {
    let mut out: BTreeMap<_, BTreeMap<String, F>> = BTreeMap::new();
    for line in include_str!("testdata/equivalence/engines.tsv").lines() {
        if line.starts_with('#') || line.trim().is_empty() {
            continue;
        }
        let c: Vec<&str> = line.split('\t').collect();
        let [source, k, engine, term, value] = c[..] else {
            panic!("bad line {line:?}");
        };
        out.entry((source.to_owned(), k.parse().unwrap(), engine.to_owned()))
            .or_default()
            .insert(term.to_owned(), value.parse().unwrap());
    }
    out
}

/// The error the check allows, per engine and term, relative to the term
/// (to 1 kcal/mol for a term below it, so an empty or cancelling term is
/// held to that absolute error), and for ΣF·v to |F||v|. The acceptance bar
/// is 1e-6; every exact path is held to 1e-9 and measures ≤ 10⁻¹² but where
/// an engine computes something else:
///
/// - GROMACS prices 1-4 pairs from cubic-spline tables (≈10⁻⁹ kcal/mol);
/// - sander reads the prmtop's eight-digit `LENNARD_JONES_ACOEF/BCOEF`, of
///   which molrs mixes the self terms (≈10⁻⁹);
/// - OpenMM interpolates CMAP with slopes from periodic splines where the IR
///   (LAMMPS `fix cmap`) splines the doubled map with natural ends (≤ 10⁻⁷
///   kcal/mol);
/// - LAMMPS's `improper_style harmonic` clamps sin χ at 0.001 in its force
///   (`improper_harmonic.cpp`, `SMALL`): its forces are not the gradient of
///   its energy for an improper within 0.057° of planar or of 180°, which a
///   configuration may hold ([`lammps_clamps`]).
fn tolerance(engine: &str, native: Native, term: &str, clamped: bool) -> F {
    let gromacs = engine == "gromacs" || (engine == "native" && native == Native::Gromacs);
    let sander = engine == "native" && native == Native::Sander;
    match term {
        "vdw" | "coul" | "total" | "fdotv" | "fnorm2" if gromacs || sander => 1e-7,
        "cmap" if engine == "openmm" => 1e-6,
        "fdotv" | "fnorm2" if engine == "lammps" && clamped => 1e-4,
        _ => 1e-9,
    }
}

/// Whether LAMMPS's `improper harmonic` force clamp ([`tolerance`]) acts at
/// `x`: some `improper harmonic` row with |sin χ| < 0.001.
fn lammps_clamps(ff: &ForceField, frame: &Frame, x: &[F]) -> bool {
    let Some(style) = ff.get_style("improper", "harmonic") else {
        return false;
    };
    let names: HashSet<&str> = style.type_rows().iter().map(|r| r.0).collect();
    relation_rows(frame, "impropers", 4)
        .iter()
        .filter(|(_, t)| names.contains(t.as_str()))
        .any(|(a, _)| {
            crate::ff::potential::flat_coords::compute_dihedral(x, a[0], a[1], a[2], a[3])
                .sin()
                .abs()
                < 0.001
        })
}

/// The relative error of `got` against `want`, on the scale `scale`.
fn rel(got: F, want: F, scale: F) -> F {
    if got == want {
        0.0
    } else {
        (got - want).abs() / scale.max(1e-300)
    }
}

/// Every pinned engine number is molrs's, term by term and in its force
/// fingerprint, for every source, configuration and engine.
#[test]
fn every_engine_prices_every_source_as_molrs() {
    let table = pinned();
    let mut worst: BTreeMap<String, F> = BTreeMap::new();
    for source in sources() {
        let sys = source.load();
        for engine in ENGINES {
            for k in 0..CONFIGS {
                let key = (source.name.to_owned(), k, engine.to_owned());
                let Some(engine_values) = table.get(&key) else {
                    panic!("{key:?}: not pinned");
                };
                let want = expected(&source, &sys, engine, k);
                let clamped = {
                    let (ff, frame) = engine_form(&sys, engine);
                    lammps_clamps(&ff, &frame, &configuration(source.name, &sys.coords, k))
                };
                let (_, fnorm2) = (want["fdotv"], want["fnorm2"]);
                let pnorm: F = probe(source.name, sys.coords.len(), k)
                    .iter()
                    .map(|v| v * v)
                    .sum::<F>()
                    .sqrt();
                for (term, &got) in engine_values {
                    let mine = want[term.as_str()];
                    let scale = match term.as_str() {
                        "fdotv" => fnorm2.sqrt() * pnorm,
                        _ => mine.abs().max(1.0),
                    };
                    let err = rel(got, mine, scale);
                    let tol = tolerance(engine, source.native, term, clamped);
                    let w = worst.entry(format!("{engine}/{term}")).or_insert(0.0);
                    *w = w.max(err);
                    assert!(
                        err <= tol,
                        "{} config {k} {engine} {term}: engine {got:?} vs molrs {mine:?} \
                         (rel {err:e} > {tol:e})",
                        source.name
                    );
                }
            }
        }
    }
    for (k, v) in worst {
        println!("worst {k}: {v:.1e}");
    }
}

/// Each engine's form of a field (CHARMM's 1-4 table as `dihedral charmm`
/// `w`, OPLS-AA's funct-1 impropers as `improper periodic`, charges per
/// atom) prices as the field as read, term for term: the rewrites are
/// exact.
#[test]
fn each_engine_form_prices_as_the_source() {
    for source in sources() {
        let sys = source.load();
        let c = coulomb(&sys.ff);
        let x = configuration(source.name, &sys.coords, 0);
        let base = molrs_terms(&sys.ff, &sys.frame, &x, c, false);
        let base_f = molrs_forces(&sys.ff, &sys.frame, &x, c);
        for engine in ["lammps", "openmm", "gromacs"] {
            let (ff, frame) = engine_form(&sys, engine);
            let got = molrs_terms(&ff, &frame, &x, c, false);
            for t in TERMS {
                // OPLS's moved rows change family, not energy.
                let (a, b) = if t == "dihedral" || t == "improper" {
                    (
                        got["dihedral"] + got["improper"],
                        base["dihedral"] + base["improper"],
                    )
                } else {
                    (got[t], base[t])
                };
                assert!(
                    rel(a, b, b.abs().max(1e-6)) <= 1e-12,
                    "{} {engine} {t}: {a} vs {b}",
                    source.name
                );
            }
            let f = molrs_forces(&ff, &frame, &x, c);
            let scale = base_f.iter().fold(0.0_f64, |m, v| m.max(v.abs()));
            for (a, b) in f.iter().zip(&base_f) {
                assert!((a - b).abs() <= 1e-11 * scale, "{} {engine}", source.name);
            }
        }
    }
}

// ── read → write → read ─────────────────────────────────────────────────────

/// What every writer writes reads back to the same IR: writing the read-back
/// field (and frame) again gives the same file, and molrs prices the
/// read-back system as the one written, term by term.
#[test]
fn every_written_file_reads_back_as_written() {
    let tmp = std::env::temp_dir().join(format!("molrs-equivalence-{}", std::process::id()));
    for source in sources() {
        let sys = source.load();
        let x = configuration(source.name, &sys.coords, 0);
        let dir = tmp.join(source.name);
        std::fs::create_dir_all(&dir).unwrap();

        // LAMMPS: include and data file.
        let (ff, frame) = engine_form(&sys, "lammps");
        let (pre, include, cmap) = lammps_include(&ff, &frame);
        std::fs::write(dir.join("system.ff"), format!("{pre}{include}")).unwrap();
        if let Some(cmap) = &cmap {
            std::fs::write(dir.join("system.cmap"), cmap).unwrap();
        }
        write_lammps_data(&dir.join("data.lmp"), &lammps_frame(&placed(&frame, &x))).unwrap();
        let back = LammpsForcefieldReader::new()
            .read(dir.join("system.ff").as_path())
            .unwrap_or_else(|e| panic!("{} LAMMPS include: {e}", source.name));
        let mut back_frame = read_lammps_data(&dir.join("data.lmp")).unwrap();
        named_types(&mut back_frame);
        back_frame.insert(
            "pairs",
            intramolecular_pairs(&back_frame, back.special_bonds(), None).unwrap(),
        );
        let (pre2, include2, cmap2) = lammps_include(&back, &back_frame);
        same_text(
            &format!("{pre}{include}"),
            &format!("{pre2}{include2}"),
            &format!("{}: LAMMPS include", source.name),
        );
        // The maps, not the comments naming them (the reader names map t "t").
        let maps = |text: &Option<String>| -> String {
            text.as_deref()
                .unwrap_or("")
                .lines()
                .filter(|l| !l.trim_start().starts_with('#'))
                .collect::<Vec<_>>()
                .join("\n")
        };
        same_text(
            &maps(&cmap),
            &maps(&cmap2),
            &format!("{}: LAMMPS cmap", source.name),
        );
        same_terms(
            source.name,
            "LAMMPS",
            (&ff, &frame),
            (&back, &back_frame),
            &x,
        );

        // OpenMM XML.
        let (ff, _) = engine_form(&sys, "openmm");
        let xml = OpenmmXmlWriter::new().written(&ff).unwrap();
        let back = OpenmmXmlReader::new()
            .read(xml.as_bytes())
            .unwrap_or_else(|e| panic!("{} OpenMM XML: {e}", source.name));
        let xml2 = OpenmmXmlWriter::new().written(&back).unwrap();
        same_text(&xml, &xml2, &format!("{}: OpenMM XML", source.name));

        // GROMACS topology.
        let (ff, frame) = engine_form(&sys, "gromacs");
        let writer = GromacsTopForcefieldWriter::new().with_precision(17);
        let top = crate::io::writer::written_with(|b| writer.write_system(b, &ff, &frame)).unwrap();
        let (back, mut back_frame) = GromacsTopForcefieldReader::new()
            .read_system(top.as_bytes())
            .unwrap_or_else(|e| panic!("{} GROMACS: {e}\n{top}", source.name));
        let top2 = crate::io::writer::written_with(|b| writer.write_system(b, &back, &back_frame))
            .unwrap();
        same_text(&top, &top2, &format!("{}: GROMACS topology", source.name));
        let mut back_ff = back;
        no_cutoff(&mut back_ff);
        let atoms = back_frame.get_mut("atoms").unwrap();
        for (d, key) in ["x", "y", "z"].iter().enumerate() {
            let col: Vec<F> = x.iter().skip(d).step_by(3).copied().collect();
            atoms
                .insert(*key, Array1::from_vec(col).into_dyn(), None)
                .unwrap();
        }
        same_terms(
            source.name,
            "GROMACS",
            (&ff, &frame),
            (&back_ff, &back_frame),
            &x,
        );
    }
    let _ = std::fs::remove_dir_all(&tmp);
}

/// Two written files are the same file but for the last bit of a number a
/// unit round trip (degrees ↔ radians, kcal ↔ kJ) moves: every token equal,
/// or both numbers within 10⁻¹⁴ relative.
fn same_text(a: &str, b: &str, what: &str) {
    let tokens = |s: &str| -> Vec<String> {
        s.split(|c: char| c.is_whitespace() || c == '"')
            .filter(|t| !t.is_empty())
            .map(str::to_owned)
            .collect()
    };
    let (ta, tb) = (tokens(a), tokens(b));
    assert_eq!(ta.len(), tb.len(), "{what}: token count");
    for (x, y) in ta.iter().zip(&tb) {
        if x == y {
            continue;
        }
        match (x.parse::<F>(), y.parse::<F>()) {
            (Ok(u), Ok(v)) => assert!(
                (u - v).abs() <= 1e-14 * u.abs().max(v.abs()),
                "{what}: {x} vs {y}"
            ),
            _ => panic!("{what}: {x} vs {y}"),
        }
    }
}

/// A LAMMPS-read frame with each block's `type` the name its id stands for
/// (the data file's Type Labels; a crossterm's map index, which the include
/// reader names "1" … "K").
fn named_types(frame: &mut Frame) {
    let labels = TypeLabels::from_frame(&*frame).unwrap();
    for block in [
        "atoms",
        "bonds",
        "angles",
        "dihedrals",
        "impropers",
        "cmaps",
    ] {
        let Some(types) = labels.block(block) else {
            continue;
        };
        let names: Vec<String> = types
            .type_ids()
            .iter()
            .map(|&id| match types.labels() {
                Some(l) => l[id as usize - 1].clone(),
                None => id.to_string(),
            })
            .collect();
        if let Some(b) = frame.get_mut(block)
            && !names.is_empty()
        {
            b.insert("type", Array1::from_vec(names).into_dyn(), None)
                .unwrap();
        }
    }
}

/// molrs prices `b` as `a`, term by term, to 1e-12 (the GROMACS reader
/// states LAMMPS's Coulomb constant: both at `a`'s).
fn same_terms(
    source: &str,
    what: &str,
    a: (&ForceField, &Frame),
    b: (&ForceField, &Frame),
    x: &[F],
) {
    let c = coulomb(a.0);
    let ta = molrs_terms(a.0, a.1, x, c, false);
    let tb = molrs_terms(b.0, b.1, x, c, false);
    for t in TERMS {
        assert!(
            rel(tb[t], ta[t], ta[t].abs().max(1e-6)) <= 1e-12,
            "{source} {what} read back, {t}: {} vs {}",
            tb[t],
            ta[t]
        );
    }
}

/// Every source's IR, and each engine's form of it, goes through a record
/// section (molrec v2) and back to the same field: molrs prices the read-back
/// field as the one written, term by term.
#[cfg(feature = "zarr")]
#[test]
fn every_source_persists_through_a_record() {
    for source in sources() {
        let sys = source.load();
        let x = configuration(source.name, &sys.coords, 0);
        for engine in ENGINES {
            let (ff, frame) = engine_form(&sys, engine);
            let back = ForceFieldSection::from_forcefield(&ff)
                .unwrap()
                .to_forcefield()
                .unwrap_or_else(|e| panic!("{} {engine}: {e}", source.name));
            same_terms(source.name, "record", (&ff, &frame), (&back, &frame), &x);
        }
    }
}

/// With `MOLRS_FF_EQUIV_DIR` set: write every engine's inputs and molrs's
/// numbers (`molrs.tsv`) there, for `scripts/ff_equivalence_check.sh`.
#[test]
fn write_engine_inputs() {
    let Some(dir) = std::env::var_os("MOLRS_FF_EQUIV_DIR") else {
        return;
    };
    let dir = Path::new(&dir);
    let mut tsv = String::new();
    for source in sources() {
        stage_inputs(dir, &source, &mut tsv);
    }
    std::fs::write(dir.join("molrs.tsv"), tsv).unwrap();
}

/// The engine of a pinned row is the source's or a writer's, and every
/// source is pinned for all of them.
#[test]
fn the_pinned_table_covers_every_source_engine_and_configuration() {
    let table = pinned();
    let mut seen = HashMap::new();
    for (source, k, engine) in table.keys() {
        *seen.entry(source.clone()).or_insert(0) += 1;
        assert!(*k < CONFIGS && ENGINES.contains(&engine.as_str()));
    }
    for source in sources() {
        assert_eq!(seen.get(source.name), Some(&(CONFIGS * ENGINES.len())));
    }
}
