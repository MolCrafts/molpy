//! Reference energies under the force-field IR (LAMMPS standard).
//!
//! Harmonic bonds and angles are `k(x − x0)²` (no hidden ½), every
//! angle-valued parameter is in degrees, and an improper is priced over the
//! dihedral of its stored atom order. Every reader, writer and typifier
//! follows the kernels, so one physical system has one energy. The tests hold
//! the energies to fixed reference values term by term at 1e-12 relative —
//! one engine family each: GAFF-, OPLS-AA-, MMFF94- and UFF-typed
//! acetanilide, and a GROMACS-, OpenMM- and LAMMPS-read field on a hand-built
//! molecule.
//!
//! UFF atoms are labelled from RDKit's own hybridization and conjugation, so
//! molrs prices them as RDKit does. A GAFF improper's atoms are ordered as
//! tleap orders them (`ff::typifier::gaff::improper`). An OpenMM
//! `<Improper>` is priced as OpenMM prices it, `(c2, c3, c1, c4)` from its
//! centre-first file order — the GROMACS-read value of the same improper, and
//! the hand value of OpenMM's formula. An OpenMM- or GROMACS-read Coulomb
//! style states that engine's own constant.

use crate::ff::potential::fixtures::OneShot;
use std::collections::BTreeMap;

use ndarray::Array1;

use crate::ff::compile::PotentialCompiler;
use crate::ff::forcefield::ForceField;
use crate::ff::potential::intramolecular_pairs;
use crate::io::gromacs::top_reader::GromacsTopForcefieldReader;
use crate::io::lammps::forcefield_reader::LammpsForcefieldReader;
use crate::io::openmm_xml::reader::OpenmmXmlReader;
use crate::io::reader::ForceFieldReader;
use molrs::core::Block;
use molrs::core::Frame;
use molrs::op::{F, Idx};

/// Per `(category/style)`, the energy of that style alone on `frame`.
fn per_style(ff: &ForceField, frame: &Frame) -> BTreeMap<String, f64> {
    let coords: Vec<F> = frame.coords().unwrap().into_iter().collect();
    let mut out = BTreeMap::new();
    for style in ff.styles() {
        if style.category() == "atom" {
            continue;
        }
        let mut one = ff.empty_like();
        let s = one
            .def_style(style.category(), style.name(), style.params().clone())
            .unwrap();
        for (name, ends, params) in style.type_rows() {
            s.def_type(name, &ends, params.clone()).unwrap();
        }
        // A relation block may name types of several styles of one category;
        // this style prices only its own rows.
        let mut frame = frame.clone();
        let block_name = match style.category() {
            "bond" => Some("bonds"),
            "angle" => Some("angles"),
            "dihedral" => Some("dihedrals"),
            "improper" => Some("impropers"),
            _ => None,
        };
        if let Some(block_name) = block_name
            && let Some(block) = frame.get(block_name)
        {
            let names: Vec<&str> = style.type_rows().iter().map(|r| r.0).collect();
            let types = block.get("type").unwrap().get::<String>().unwrap();
            let keep: Vec<usize> = (0..types.len())
                .filter(|&i| names.contains(&types[i].as_str()))
                .collect();
            let kept = block.select_rows(&keep).unwrap();
            frame.insert(block_name, kept);
        }
        let pots = PotentialCompiler::new(&one).compile(&frame).unwrap();
        out.insert(
            format!("{}/{}", style.category(), style.name()),
            pots.energy_at(&coords),
        );
    }
    out
}

/// `got` holds exactly `want`'s styles, each within 1e-12 relative or 1e-15
/// kcal/mol absolute. The absolute floor is for a term that is a small
/// difference of large ones: the GROMACS RB torsion here is 1.45e-6 kcal/mol
/// out of coefficients of 0.1 kcal/mol, and the same polynomial summed in
/// another order moves it by 2e-17.
fn assert_energies(label: &str, got: &BTreeMap<String, f64>, want: &[(&str, f64)]) {
    let names: Vec<&str> = got.keys().map(String::as_str).collect();
    let want_names: Vec<&str> = want.iter().map(|(n, _)| *n).collect();
    assert_eq!(names, want_names, "{label}: styles");
    for (style, w) in want {
        let g = got[*style];
        let rel = (g - w).abs() / w.abs().max(f64::MIN_POSITIVE);
        assert!(
            rel <= 1e-12 || (g - w).abs() <= 1e-15,
            "{label} {style}: {g:?} vs the reference {w:?} (rel {rel:e})"
        );
    }
}

// ---------------------------------------------------------------------------
// Hand-built 7-atom molecule for the file-read fields
// ---------------------------------------------------------------------------

/// Atom types of the hand molecule, in atom order.
const TYPES: [&str; 7] = ["N", "C", "O", "H", "CT", "HC", "HA"];
const XYZ: [[f64; 3]; 7] = [
    [0.00, 0.00, 0.00],
    [1.33, 0.05, 0.08],
    [1.98, 1.07, -0.12],
    [-0.52, -0.86, 0.11],
    [-0.71, 1.27, -0.21],
    [-1.40, 1.31, 0.63],
    [1.85, -0.91, 0.24],
];
const CHARGES: [f64; 7] = [-0.42, 0.51, -0.55, 0.29, 0.03, 0.06, 0.08];
const BONDS: [[usize; 2]; 6] = [[0, 1], [0, 3], [0, 4], [1, 2], [4, 5], [1, 6]];
const ANGLES: [[usize; 3]; 7] = [
    [1, 0, 3],
    [1, 0, 4],
    [3, 0, 4],
    [0, 1, 2],
    [0, 4, 5],
    [0, 1, 6],
    [2, 1, 6],
];
const DIHEDRALS: [[usize; 4]; 6] = [
    [2, 1, 0, 3],
    [2, 1, 0, 4],
    [1, 0, 4, 5],
    [3, 0, 4, 5],
    [6, 1, 0, 3],
    [6, 1, 0, 4],
];
/// The impropers in the ENGINE's own atom order: the N centre as AMBER /
/// GROMACS funct 4 list it (centre third), the C centre as CHARMM / GROMACS
/// funct 2 list it (centre first).
const IMPROPER_N: [usize; 4] = [1, 3, 0, 4];
const IMPROPER_C: [usize; 4] = [1, 0, 2, 6];

fn col_f(v: Vec<f64>) -> ndarray::ArrayD<F> {
    Array1::from_vec(v).into_dyn()
}

fn col_i(v: Vec<Idx>) -> ndarray::ArrayD<Idx> {
    Array1::from_vec(v).into_dyn()
}

fn col_s(v: Vec<String>) -> ndarray::ArrayD<String> {
    Array1::from_vec(v).into_dyn()
}

/// The type name of the `category` row whose endpoints match `atoms`' types,
/// and the atoms in the order of that row's endpoints.
fn resolve(ff: &ForceField, category: &str, atoms: &[usize]) -> Vec<(String, Vec<usize>)> {
    let mut out = Vec::new();
    for style in ff.get_styles(category) {
        for (name, ends, _) in style.type_rows() {
            let fits = |order: &[usize]| {
                order
                    .iter()
                    .zip(&ends)
                    .all(|(&a, e)| e.is_empty() || TYPES[a] == *e)
            };
            if category == "improper" {
                // Any permutation: the row's endpoint order says which atom
                // goes where.
                let mut found = None;
                for p in permutations4() {
                    let order: Vec<usize> = p.iter().map(|&i| atoms[i]).collect();
                    if fits(&order) {
                        found = Some(order);
                        break;
                    }
                }
                if let Some(order) = found {
                    out.push((name.to_owned(), order));
                }
            } else {
                let fwd = atoms.to_vec();
                let mut rev = fwd.clone();
                rev.reverse();
                if fits(&fwd) {
                    out.push((name.to_owned(), fwd));
                } else if fits(&rev) {
                    out.push((name.to_owned(), rev));
                }
            }
        }
    }
    out
}

fn permutations4() -> Vec<[usize; 4]> {
    let mut out = Vec::new();
    for a in 0..4 {
        for b in 0..4 {
            for c in 0..4 {
                for d in 0..4 {
                    let p = [a, b, c, d];
                    let mut seen = [false; 4];
                    if p.iter().all(|&i| !std::mem::replace(&mut seen[i], true)) {
                        out.push(p);
                    }
                }
            }
        }
    }
    out
}

fn relation_block(ff: &ForceField, category: &str, rows: &[Vec<usize>]) -> Option<Block> {
    let cols = ["atomi", "atomj", "atomk", "atoml"];
    let mut atoms: Vec<Vec<Idx>> = vec![Vec::new(); rows[0].len()];
    let mut names = Vec::new();
    for row in rows {
        for (name, order) in resolve(ff, category, row) {
            for (c, a) in order.iter().enumerate() {
                atoms[c].push(*a as Idx);
            }
            names.push(name);
        }
    }
    if names.is_empty() {
        return None;
    }
    let mut block = Block::new();
    for (c, col) in atoms.into_iter().enumerate() {
        block.insert(cols[c], col_i(col), None).unwrap();
    }
    block.insert("type", col_s(names), None).unwrap();
    Some(block)
}

pub(crate) fn hand_frame(ff: &ForceField) -> Frame {
    let mut frame = Frame::new();
    let mut atoms = Block::new();
    for (d, key) in ["x", "y", "z"].iter().enumerate() {
        atoms
            .insert(*key, col_f(XYZ.iter().map(|p| p[d]).collect()), None)
            .unwrap();
    }
    atoms
        .insert(
            "type",
            col_s(TYPES.iter().map(|t| (*t).to_owned()).collect()),
            None,
        )
        .unwrap();
    atoms
        .insert("charge", col_f(CHARGES.to_vec()), None)
        .unwrap();
    frame.insert("atoms", atoms);
    let bonds: Vec<Vec<usize>> = BONDS.iter().map(|r| r.to_vec()).collect();
    let angles: Vec<Vec<usize>> = ANGLES.iter().map(|r| r.to_vec()).collect();
    let dihedrals: Vec<Vec<usize>> = DIHEDRALS.iter().map(|r| r.to_vec()).collect();
    let impropers = vec![IMPROPER_N.to_vec(), IMPROPER_C.to_vec()];
    for (cat, block, rows) in [
        ("bond", "bonds", bonds),
        ("angle", "angles", angles),
        ("dihedral", "dihedrals", dihedrals),
        ("improper", "impropers", impropers),
    ] {
        if let Some(b) = relation_block(ff, cat, &rows) {
            frame.insert(block, b);
        }
    }
    let pairs = intramolecular_pairs(&frame, ff.special_bonds(), None).unwrap();
    frame.insert("pairs", pairs);
    frame
}

const GROMACS_FF: &str = "\
[ defaults ]
1 2 yes 0.5 0.8333333333

[ atomtypes ]
N   7 14.01 0.0 A 0.325 0.711
C   6 12.01 0.0 A 0.339967 0.359824
O   8 16.00 0.0 A 0.295992 0.87864
H   1 1.008 0.0 A 0.106908 0.0656888
CT  6 12.01 0.0 A 0.339967 0.4577296
HC  1 1.008 0.0 A 0.264953 0.0656888
HA  1 1.008 0.0 A 0.259964 0.06276

[ bondtypes ]
N  C  1 0.1335 410032.0
N  H  1 0.1010 363171.2
N  CT 1 0.1449 282001.6
C  O  1 0.1229 476976.0
CT HC 1 0.1090 284512.0
C  HA 1 0.1100 307105.6

[ angletypes ]
C  N  H  1 120.0  418.4
C  N  CT 1 121.9  418.4
H  N  CT 1 118.04 418.4
N  C  O  1 122.9  669.44
N  CT HC 1 109.5  418.4
N  C  HA 1 114.0  418.4
O  C  HA 1 123.0  418.4

[ dihedraltypes ]
O  C  N  H  1 180.0 10.46 2
O  C  N  CT 1 180.0 10.46 2
C  N  CT HC 1 0.0 0.0 3
H  N  CT HC 1 0.0 0.65 3
HA C  N  H  1 180.0 10.46 2
HA C  N  CT 3 1.0 -0.5 0.2 -0.7 0.0 0.0
C  H  N  CT 4 180.0 4.6024 2
C  N  O  HA 2 0.0 167.36
";

const OPENMM_FF: &str = r#"<ForceField name="hand">
 <AtomTypes>
  <Type name="N" class="N" element="N" mass="14.01"/>
  <Type name="C" class="C" element="C" mass="12.01"/>
  <Type name="O" class="O" element="O" mass="16.00"/>
  <Type name="H" class="H" element="H" mass="1.008"/>
  <Type name="CT" class="CT" element="C" mass="12.01"/>
  <Type name="HC" class="HC" element="H" mass="1.008"/>
  <Type name="HA" class="HA" element="H" mass="1.008"/>
 </AtomTypes>
 <HarmonicBondForce>
  <Bond class1="N" class2="C" length="0.1335" k="410032.0"/>
  <Bond class1="N" class2="H" length="0.1010" k="363171.2"/>
  <Bond class1="N" class2="CT" length="0.1449" k="282001.6"/>
  <Bond class1="C" class2="O" length="0.1229" k="476976.0"/>
  <Bond class1="CT" class2="HC" length="0.1090" k="284512.0"/>
  <Bond class1="C" class2="HA" length="0.1100" k="307105.6"/>
 </HarmonicBondForce>
 <HarmonicAngleForce>
  <Angle class1="C" class2="N" class3="H" angle="2.0943951023931953" k="418.4"/>
  <Angle class1="C" class2="N" class3="CT" angle="2.1275564400266706" k="418.4"/>
  <Angle class1="H" class2="N" class3="CT" angle="2.060188" k="418.4"/>
  <Angle class1="N" class2="C" class3="O" angle="2.1450096507010237" k="669.44"/>
  <Angle class1="N" class2="CT" class3="HC" angle="1.9111355" k="418.4"/>
  <Angle class1="N" class2="C" class3="HA" angle="1.9896753" k="418.4"/>
  <Angle class1="O" class2="C" class3="HA" angle="2.1467549799530254" k="418.4"/>
 </HarmonicAngleForce>
 <PeriodicTorsionForce>
  <Proper class1="O" class2="C" class3="N" class4="H" periodicity1="2" phase1="3.141592653589793" k1="10.46"/>
  <Proper class1="O" class2="C" class3="N" class4="CT" periodicity1="2" phase1="3.141592653589793" k1="10.46" periodicity2="1" phase2="0.0" k2="1.2"/>
  <Proper class1="C" class2="N" class3="CT" class4="HC" periodicity1="3" phase1="0.0" k1="0.3"/>
  <Proper class1="H" class2="N" class3="CT" class4="HC" periodicity1="3" phase1="0.0" k1="0.65"/>
  <Proper class1="HA" class2="C" class3="N" class4="H" periodicity1="2" phase1="3.141592653589793" k1="10.46"/>
  <Proper class1="HA" class2="C" class3="N" class4="CT" periodicity1="2" phase1="3.141592653589793" k1="8.0"/>
  <Improper class1="N" class2="C" class3="H" class4="CT" periodicity1="2" phase1="3.141592653589793" k1="4.6024"/>
 </PeriodicTorsionForce>
 <NonbondedForce coulomb14scale="0.8333333333333334" lj14scale="0.5">
  <Atom type="N" charge="-0.42" sigma="0.325" epsilon="0.711"/>
  <Atom type="C" charge="0.51" sigma="0.339967" epsilon="0.359824"/>
  <Atom type="O" charge="-0.55" sigma="0.295992" epsilon="0.87864"/>
  <Atom type="H" charge="0.29" sigma="0.106908" epsilon="0.0656888"/>
  <Atom type="CT" charge="0.03" sigma="0.339967" epsilon="0.4577296"/>
  <Atom type="HC" charge="0.06" sigma="0.264953" epsilon="0.0656888"/>
  <Atom type="HA" charge="0.08" sigma="0.259964" epsilon="0.06276"/>
 </NonbondedForce>
</ForceField>
"#;

pub(crate) const LAMMPS_FF: &str = "\
units real
special_bonds lj 0.0 0.0 0.5 coul 0.0 0.0 0.8333333333333334
pair_style lj/cut/coul/cut 10.0
pair_coeff N N 0.17 3.25
pair_coeff C C 0.086 3.39967
pair_coeff O O 0.21 2.95992
pair_coeff H H 0.0157 1.06908
pair_coeff CT CT 0.1094 3.39967
pair_coeff HC HC 0.0157 2.64953
pair_coeff HA HA 0.015 2.59964
bond_style harmonic
bond_coeff N-C 490.0 1.335
bond_coeff N-H 434.0 1.01
bond_coeff N-CT 337.0 1.449
bond_coeff C-O 570.0 1.229
bond_coeff CT-HC 340.0 1.09
bond_coeff C-HA 367.0 1.1
angle_style harmonic
angle_coeff C-N-H 50.0 120.0
angle_coeff C-N-CT 50.0 121.9
angle_coeff H-N-CT 50.0 118.04
angle_coeff N-C-O 80.0 122.9
angle_coeff N-CT-HC 50.0 109.5
angle_coeff N-C-HA 50.0 114.0
angle_coeff O-C-HA 50.0 123.0
dihedral_style fourier
dihedral_coeff O-C-N-H 1 2.5 2 180.0
dihedral_coeff O-C-N-CT 2 2.5 2 180.0 0.3 1 0.0
dihedral_coeff C-N-CT-HC 1 0.0 3 0.0
dihedral_coeff H-N-CT-HC 1 0.155 3 0.0
dihedral_coeff HA-C-N-H 1 2.5 2 180.0
dihedral_coeff HA-C-N-CT 1 1.9 2 180.0
improper_style harmonic
improper_coeff C-N-O-HA 20.0 0.0
improper_coeff N-C-H-CT 10.5 12.0
";

/// The per-style reference energies of the GROMACS-, OpenMM- and LAMMPS-read
/// fields on the hand molecule. A GROMACS funct-3 torsion reads as
/// `dihedral/multi/harmonic` (the polynomial, without the OPLS projection).
/// Each Coulomb reference is stated at LAMMPS real's 332.06371 and scaled to
/// the engine's own constant; the OpenMM improper is checked in the next test.
#[test]
fn file_read_fields_price_at_the_reference_energies() {
    let gmx = GromacsTopForcefieldReader::new()
        .read(GROMACS_FF.as_bytes())
        .unwrap();
    let mut gmx_energies = per_style(&gmx, &hand_frame(&gmx));
    // A GROMACS-read field is priced with GROMACS's own Coulomb constant
    // (its ONE_4PI_EPS0, CODATA 2018): the LAMMPS-real reference times their
    // ratio, exactly.
    let coul = gmx_energies.remove("pair/coul/cut").unwrap();
    let ratio = crate::core::constants::gromacs_coulomb_real() / 332.06371;
    let want = -10.706661989420029 * ratio;
    assert!(
        (coul - want).abs() <= 1e-12 * want.abs(),
        "GROMACS pair/coul/cut: {coul} vs {want}"
    );
    assert_energies(
        "GROMACS",
        &gmx_energies,
        &[
            ("angle/harmonic", 1.3595933975169494),
            ("bond/harmonic", 0.16275010462128736),
            ("dihedral/multi/harmonic", 1.4519861291139018e-6),
            ("dihedral/periodic", 0.09469745227704568),
            ("improper/harmonic", 0.07338136844630398),
            ("improper/periodic", 0.003384791688934619),
            ("pair/lj/cut", 1.264689101097666),
        ],
    );
    let omm = OpenmmXmlReader::new().read(OPENMM_FF.as_bytes()).unwrap();
    let mut omm_energies = per_style(&omm, &hand_frame(&omm));
    // 0.0013423609738289378 is this improper over the dihedral of OpenMM's
    // file order (centre first), which is not what OpenMM prices.
    let improper = omm_energies.remove("improper/periodic").unwrap();
    // An OpenMM-read field is priced with OpenMM's own Coulomb constant
    // (ONE_4PI_EPS0 = 332.06371329919216 kcal·Å/(mol·e²)): the LAMMPS-real
    // (332.06371) reference times their ratio, exactly.
    let coul = omm_energies.remove("pair/coul/cut").unwrap();
    let ratio = crate::core::constants::openmm_coulomb_real() / 332.06371;
    let want = -10.706661989738114 * ratio;
    assert!(
        (coul - want).abs() <= 1e-12 * want.abs(),
        "OpenMM pair/coul/cut: {coul} vs {want}"
    );
    assert_energies(
        "OpenMM",
        &omm_energies,
        &[
            ("angle/harmonic", 1.3595889901856875),
            ("bond/harmonic", 0.16275010462128736),
            ("dihedral/periodic", 0.8087245811237659),
            ("pair/lj/cut", 1.264689101097666),
        ],
    );
    assert!(improper > 0.0013423609738289378 * 2.0, "{improper}");
    let lmp = LammpsForcefieldReader::new()
        .read(LAMMPS_FF.as_bytes())
        .unwrap();
    assert_energies(
        "LAMMPS",
        &per_style(&lmp, &hand_frame(&lmp)),
        &[
            ("angle/harmonic", 1.3595933975169494),
            ("bond/harmonic", 0.1627501046212882),
            ("dihedral/periodic", 0.6929798914238418),
            ("improper/harmonic", 0.43171701286731823),
            ("pair/coul/cut", -10.706661989738114),
            ("pair/lj/cut", 1.220127959380373),
        ],
    );
}

/// Regression for a silent miscalculation: one AMBER improper — centre `N`,
/// `k` = 4.6024 kJ/mol, n = 2, phase 180° — read from OpenMM XML (centre
/// first, `<Improper class1="N" class2="C" class3="H" class4="CT">`) and from
/// GROMACS (`C H N CT 4`, centre third) prices identically, at OpenMM's own
/// value `k[1 + cos(2φ − π)]` with φ the dihedral C-H-N-CT. Pricing the
/// OpenMM row over its file order N-C-H-CT gives 0.40× the energy here.
#[test]
fn an_openmm_improper_prices_as_gromacs_and_openmm_do() {
    let gmx = GromacsTopForcefieldReader::new()
        .read(GROMACS_FF.as_bytes())
        .unwrap();
    let omm = OpenmmXmlReader::new().read(OPENMM_FF.as_bytes()).unwrap();
    let e_gmx = per_style(&gmx, &hand_frame(&gmx))["improper/periodic"];
    let e_omm = per_style(&omm, &hand_frame(&omm))["improper/periodic"];
    let coords: Vec<F> = XYZ.iter().flatten().copied().collect();
    let [c, h, n, ct] = IMPROPER_N;
    let phi = crate::ff::potential::flat_coords::compute_dihedral(&coords, c, h, n, ct);
    let k = 4.6024 / 4.184;
    let hand = k * (1.0 + (2.0 * phi - std::f64::consts::PI).cos());
    assert!(
        (e_gmx - hand).abs() <= 1e-12 * hand,
        "GROMACS {e_gmx} vs {hand}"
    );
    assert!(
        (e_omm - hand).abs() <= 1e-12 * hand,
        "OpenMM {e_omm} vs {hand}"
    );
}

/// Acetanilide, as `add_hydrogens` orders it, at an ETKDG conformer (seed 7),
/// hard-coded so the test does not depend on the embedder.
const ACETANILIDE_XYZ: [[f64; 3]; 19] = [
    [
        3.4074919590548594,
        0.11878851274428995,
        -0.33228306862388435,
    ],
    [2.0260536948054644, -0.4571683448888246, -0.5221033020386832],
    [1.8591505549081395, -1.4874143930549852, -1.1648489243326972],
    [1.0496149580076168, 0.32159354372938, 0.07741475328198548],
    [
        -0.31882609280612834,
        0.12080667078571715,
        0.11494091329012104,
    ],
    [-1.078500739147817, 1.03677439604918, 0.7634735759510434],
    [-2.4170960679390103, 0.9063511330365619, 0.8410607638045803],
    [-3.0236393830383315, -0.1481223386586445, 0.2687279923566584],
    [-2.2868319995043906, -1.0675005940797127, -0.378711280615214],
    [
        -0.9480847903681884,
        -0.9326392570783705,
        -0.4536146590406588,
    ],
    [3.534478681342836, 0.49421863131366467, 0.6872278404518046],
    [3.564026348919401, 0.9305721650227372, -1.0474467763997268],
    [4.154263207251315, -0.6622374790651407, -0.5020725557537553],
    [1.3799612977205442, 1.1550895176367948, 0.547287909253901],
    [-0.6234857025125715, 1.9016105044380223, 1.237907566325366],
    [-3.008021362742696, 1.6519877190160193, 1.3655211613678315],
    [-4.102778287010533, -0.2580764141442648, 0.3283076743199545],
    [
        -2.7690068589682153,
        -1.9228969434479155,
        -0.8439450291224427,
    ],
    [-0.398769417904483, -1.7017370182791989, -0.9868445483647516],
];

fn acetanilide() -> molrs::core::Atomistic {
    use crate::io::smiles::SmilesIr;
    use crate::perceive::add_hydrogens;
    let mut mol = add_hydrogens(
        &(SmilesIr::parse("CC(=O)Nc1ccccc1").unwrap())
            .to_atomistic()
            .unwrap(),
    )
    .unwrap();
    let ids: Vec<_> = mol.atoms().map(|(id, _)| id).collect();
    assert_eq!(ids.len(), ACETANILIDE_XYZ.len());
    for (id, xyz) in ids.into_iter().zip(ACETANILIDE_XYZ) {
        for (key, v) in ["x", "y", "z"].into_iter().zip(xyz) {
            mol.set_atom(id, key, v).unwrap();
        }
    }
    mol
}

/// The frame of a typed molecule with its intramolecular pair list, and, when
/// the typifier assigns none, the fixed test charges `0.1·(i mod 5 − 2)`.
fn typed_frame(typed: &molrs::core::Atomistic, ff: &ForceField) -> Frame {
    let mut frame = typed.to_frame().unwrap();
    let mut atoms = frame.get("atoms").unwrap().clone();
    if atoms.get("charge").is_none() {
        let n = atoms.n_rows().unwrap();
        let q: Vec<f64> = (0..n).map(|i| 0.1 * ((i % 5) as f64 - 2.0)).collect();
        atoms.insert("charge", col_f(q), None).unwrap();
        frame.insert("atoms", atoms);
    }
    let pairs = intramolecular_pairs(&frame, ff.special_bonds(), None).unwrap();
    frame.insert("pairs", pairs);
    frame
}

/// GAFF (ATD types, then the table), OPLS-AA, MMFF94 and UFF typings of
/// acetanilide price at the reference energies: GAFF and OPLS-AA through the
/// re-based table values (`K`, degrees), MMFF94 and UFF through their
/// per-instance columns (θ0 in degrees, the out-of-plane centre first).
#[test]
fn typed_molecules_price_at_the_reference_energies() {
    use crate::ff::typifier::mmff::Mmff94Typifier;
    use crate::ff::typifier::{AtdParameterSet, AtdTypifier};
    use crate::ff::typifier::{GaffParameterSet, GaffTypifier};
    use crate::ff::typifier::{OplsAaTypifier, Typing, UffTypifier};

    let mol = acetanilide();

    let labelled = Typing::new(AtdTypifier::new(AtdParameterSet::Gff))
        .typify(&mol)
        .unwrap();
    let mut gaff = Typing::new(GaffTypifier::new(GaffParameterSet::Gaff));
    let typed = gaff.typify(&labelled).unwrap();
    let ff = gaff.forcefield();
    assert_energies(
        "GAFF",
        &per_style(ff, &typed_frame(&typed, ff)),
        &[
            ("angle/harmonic", 2.2194097346204344),
            ("bond/harmonic", 7.3769495524832465),
            ("dihedral/periodic", 2.541103465113677),
            // GAFF impropers are built as tleap builds them: atom order from
            // the matched row, parmchk2's estimates
            // (`ff::typifier::gaff::improper`).
            ("improper/periodic", 0.010339596944304806),
            ("pair/coul/cut", -37.15184225195395),
            ("pair/lj/cut", 8.660474394181266),
        ],
    );

    let mut opls = Typing::new(OplsAaTypifier::oplsaa());
    let typed = opls.typify(&mol).unwrap();
    let ff = opls.forcefield();
    assert_energies(
        "OPLS-AA",
        &per_style(ff, &typed_frame(&typed, ff)),
        &[
            ("angle/harmonic", 2.828467617797212),
            ("bond/harmonic", 8.89620781373482),
            ("dihedral/opls", 0.006529522050728192),
            ("pair/coul/cut", -15.716061914696931),
            ("pair/lj/cut", 11.997170104249419),
        ],
    );

    let mut mmff = Typing::new(Mmff94Typifier::new());
    let typed = mmff.typify(&mol).unwrap();
    let ff = mmff.forcefield();
    assert_energies(
        "MMFF94",
        &per_style(ff, &typed_frame(&typed, ff)),
        &[
            ("angle/mmff_angle", 3.4673399764538027),
            ("angle/mmff_stbn", 0.289919226317351),
            ("bond/mmff_bond", 1.4495889299861888),
            ("dihedral/mmff_torsion", 1.0969837819211763),
            ("improper/mmff_oop", 0.00988136481655142),
            ("pair/coul/cut", -25.66342275717315),
            ("pair/mmff_vdw", 24.778280542971032),
        ],
    );

    let mut uff = Typing::new(UffTypifier::new());
    let typed = uff.typify(&mol).unwrap();
    let ff = uff.forcefield();
    assert_energies(
        "UFF",
        &per_style(ff, &typed_frame(&typed, ff)),
        &[
            // UFF's atom labels come from RDKit's hybridization and
            // conjugation (`perceive::perceive_hybridizations`): the amide N,
            // carbonyl C and O are `N_R` / `C_R` / `O_R`, and the amide C-N
            // is priced at order 1, not 1.41. The four bonded terms below sum
            // to RDKit 2026.03's UFF energy on the same geometry without vdW,
            // 16.88335231598728, to 1e-14.
            ("angle/uff_angle", 9.878623919580525),
            ("bond/uff_bond", 6.018563091754078),
            ("dihedral/uff_torsion", 0.9769947445298511),
            ("improper/uff_inversion", 0.009170560122818843),
            ("pair/uff_lj", 25.591590220999027),
        ],
    );
}
