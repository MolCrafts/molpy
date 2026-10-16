//! molrs's native GAFF route and AmberTools' prmtop of the same molecule
//! give one force field: [`ForceField::merge`] takes the two, either way
//! round, and the union holds no type the native route does not.
//!
//! Each case is a molecule `scripts/gaff_merge_fixtures.sh` ran through
//! AmberTools 26.0 (`testdata/gaff/`): antechamber typed `<mol>.pdb` and
//! wrote `<mol>_<set>.mol2`, parmchk2 estimated what the set lacks, and tleap
//! wrote `<mol>_<set>.parm7`. molrs types the mol2's atoms and bonds itself
//! (`AtdTypifier`, then `GaffTypifier`) and reads the prmtop
//! (`AmberPrmtopForcefieldReader`). Ethanol is covered by table rows alone;
//! TFSI needs parmchk2's estimates of a bond (GAFF), angles and torsions.

use std::collections::BTreeSet;

use crate::ff::forcefield::ForceField;
use crate::ff::typifier::Typing;
use crate::ff::typifier::{AtdParameterSet, AtdTypifier, GaffParameterSet, GaffTypifier};
use crate::io::amber::prmtop_forcefield::AmberPrmtopForcefieldReader;
use crate::io::reader::ForceFieldReader;
use molrs::core::Atomistic;
use molrs::core::keys;

struct Case {
    name: &'static str,
    set: GaffParameterSet,
    mol2: &'static str,
    parm: &'static str,
}

macro_rules! case {
    ($mol:literal, $set:literal, $param_set:expr) => {
        Case {
            name: concat!($mol, "_", $set),
            set: $param_set,
            mol2: include_str!(concat!("testdata/gaff/", $mol, "_", $set, ".mol2")),
            parm: include_str!(concat!("testdata/gaff/", $mol, "_", $set, ".parm7")),
        }
    };
}

fn cases() -> [Case; 4] {
    [
        case!("ethanol", "gaff", GaffParameterSet::Gaff),
        case!("ethanol", "gaff2", GaffParameterSet::Gaff2),
        case!("tfsi", "gaff", GaffParameterSet::Gaff),
        case!("tfsi", "gaff2", GaffParameterSet::Gaff2),
    ]
}

/// The case's molecule as the mol2 holds it — its atoms (elements from the
/// prmtop, which tleap wrote in the mol2's atom order) and its bonds in the
/// mol2's order, the order parmchk2 searched — typed by molrs: ATD, then
/// GAFF. Returns the native force field and the atom types ATD gave.
fn native(case: &Case) -> (ForceField, Vec<String>) {
    let structure = crate::io::read_prmtop_str(case.parm).unwrap();
    let elements = structure["atoms"]
        .get("element")
        .unwrap()
        .as_string()
        .unwrap()
        .to_owned();
    let frame = crate::io::read_mol2_str(case.mol2).unwrap();
    let atoms = &frame["atoms"];
    let xyz = ["x", "y", "z"].map(|c| atoms.get(c).unwrap().as_float().unwrap().to_owned());
    let mut mol = Atomistic::new();
    let ids: Vec<_> = (0..elements.len())
        .map(|i| mol.add_atom_xyz(&elements[[i]], xyz[0][[i]], xyz[1][[i]], xyz[2][[i]]))
        .collect();
    let bonds = &frame["bonds"];
    let (bi, bj) = (
        bonds.get("atomi").unwrap().as_uint().unwrap(),
        bonds.get("atomj").unwrap().as_uint().unwrap(),
    );
    for (&i, &j) in bi.iter().zip(bj.iter()) {
        mol.add_bond(ids[i as usize], ids[j as usize]).unwrap();
    }
    let atd = match case.set {
        GaffParameterSet::Gaff => AtdParameterSet::Gff,
        GaffParameterSet::Gaff2 => AtdParameterSet::Gff2,
    };
    let labelled = Typing::new(AtdTypifier::new(atd)).typify(&mol).unwrap();
    let types = labelled
        .atoms()
        .map(|(_, atom)| atom.get_str(keys::TYPE).unwrap().to_owned())
        .collect();
    let mut gaff = Typing::new(GaffTypifier::new(case.set));
    gaff.typify(&labelled).unwrap();
    (gaff.forcefield().clone(), types)
}

/// Every type of `ff` as `(category, style, name)`.
fn types(ff: &ForceField) -> BTreeSet<(String, String, String)> {
    ff.styles()
        .iter()
        .flat_map(|s| {
            s.type_rows().into_iter().map(|(name, _, _)| {
                (
                    s.category().to_owned(),
                    s.name().to_owned(),
                    name.to_owned(),
                )
            })
        })
        .collect()
}

#[test]
fn native_gaff_merges_with_tleaps_prmtop_of_the_same_molecule() {
    for case in cases() {
        let name = case.name;
        let (native, atom_types) = native(&case);
        let mol2_types: Vec<String> = crate::io::read_mol2_str(case.mol2).unwrap()["atoms"]
            .get(keys::TYPE)
            .unwrap()
            .as_string()
            .unwrap()
            .iter()
            .cloned()
            .collect();
        assert_eq!(atom_types, mol2_types, "{name}: ATD types as antechamber");
        let prmtop = AmberPrmtopForcefieldReader::new()
            .read_str(case.parm)
            .unwrap();

        // Both name every term alike: each prmtop type is a native one.
        let ours = types(&native);
        let theirs = types(&prmtop);
        let only_prmtop: Vec<_> = theirs.difference(&ours).collect();
        assert!(
            only_prmtop.is_empty(),
            "{name}: prmtop-only {only_prmtop:?}"
        );

        let mut merged = native.clone();
        merged
            .merge(&prmtop)
            .unwrap_or_else(|e| panic!("{name}: native.merge(prmtop): {e}"));
        assert_eq!(types(&merged), ours, "{name}: native.merge(prmtop)");

        let mut merged = prmtop.clone();
        merged
            .merge(&native)
            .unwrap_or_else(|e| panic!("{name}: prmtop.merge(native): {e}"));
        assert_eq!(types(&merged), ours, "{name}: prmtop.merge(native)");
    }
}

/// The native route names a torsion as the prmtop reader does — oriented
/// (`c3-c3-oh-ho`, where `gaff.dat` writes `ho-oh-c3-c3`), its terms by
/// periodicity — and an estimate by its types (`f-c3-sy`), the provenance in
/// its params at the frcmod's precision (`ne-s6`'s r₀ is parmchk2's printed
/// `1.672`, not its analog `ne-sy`'s 1.6723).
#[test]
fn native_names_and_numbers_are_the_prmtops() {
    let [ethanol, _, tfsi, _] = cases();
    let (ff, _) = native(&ethanol);
    let torsion = ff
        .get_style("dihedral", "periodic")
        .unwrap()
        .type_params("c3-c3-oh-ho")
        .expect("oriented name");
    assert_eq!(torsion.get("periodicity1"), Some(1.0));
    assert_eq!(torsion.get("k1"), Some(0.25));
    assert_eq!(torsion.get("periodicity2"), Some(3.0));
    assert_eq!(torsion.get("k2"), Some(0.16));
    let lj = ff.get_style("pair", "lj/cut").unwrap();
    assert_eq!(lj.params().get_str("mixing"), Some("arithmetic"));

    let (ff, _) = native(&tfsi);
    let angle = ff
        .get_style("angle", "harmonic")
        .unwrap()
        .type_params("f-c3-sy")
        .expect("an estimate is named by its types");
    assert_eq!(angle.get("estimated"), Some(1.0));
    assert_eq!(angle.get_str("estimate_analog"), Some("f-c3-s6"));
    let bond = ff
        .get_style("bond", "harmonic")
        .unwrap()
        .type_params("ne-s6")
        .unwrap();
    assert_eq!(bond.get("r0"), Some(1.672));
    assert_eq!(bond.get("k"), Some(317.1));
}
