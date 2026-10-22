//! Executable specification for the Aromatic Bond Representation Standard.
//!
//! This module is **tests only**. It asserts the standard's data invariants over
//! the structure matrix the standard names (§17), by reading canonical **key
//! names** rather than any Rust type — so it is a statement about the data
//! model, and cannot drift with the implementation that satisfies it.
//!
//! They are the standard's acceptance gate.
//!
//! Run them with:
//!
//! ```text
//! cargo test -p molcrafts-molrs --lib --features smiles aromatic_standard
//! ```
//!
//! # The invariants, in one place
//!
//! ```text
//! atom.is_aromatic   the atom belongs to an aromatic system
//! bond_type          0 unknown, 1 single, 2 double, 3 triple, 4 aromatic
//! bond_number        0 unknown, 1 single, 2 double, 3 triple, 4 quadruple
//! ```
//!
//! An aromatic bond is `bond_type = 4` **with** a `bond_number` of 1 or 2. There
//! is no fractional bond number, and no second localized-order field.

#![cfg(all(test, feature = "smiles"))]

use std::collections::{HashMap, HashSet};

use crate::core::Atomistic;
use crate::core::PropValue;
use crate::core::Props;
use crate::core::{NodeId, RelationId};
use crate::perceive::{add_hydrogens, assign_aromaticity, assign_kekule_bond_orders};

/// Bond prop: the chemical class (§2.1).
const BOND_TYPE: &str = "bond_type";
/// Bond prop: the localized Lewis/Kekulé integer (§2.2).
const BOND_NUMBER: &str = "bond_number";
/// Atom prop: aromatic system membership (§2.3).
use crate::core::keys::IS_AROMATIC;
/// Atom prop: the species, as an atomic number.
const ATOMIC_NUMBER: &str = "atomic_number";

/// Atomic numbers of the elements the clauses below single out.
const Z_C: u32 = 6;
const Z_N: u32 = 7;
const Z_O: u32 = 8;

/// `bond_type` code for an aromatic bond (§2.1). Not a quadruple bond.
const TYPE_AROMATIC: u32 = 4;
const TYPE_SINGLE: u32 = 1;
const TYPE_DOUBLE: u32 = 2;
const TYPE_TRIPLE: u32 = 3;

const NUMBER_UNKNOWN: u32 = 0;
const NUMBER_SINGLE: u32 = 1;
const NUMBER_DOUBLE: u32 = 2;

// ---------------------------------------------------------------------------
// The structure matrix (§17)
// ---------------------------------------------------------------------------

/// Every structure the standard requires coverage of, as `(name, SMILES)`.
///
/// Written as a directory of the matrix rather than one test per molecule, so a
/// clause is asserted over *all* of them and a structure cannot be quietly
/// dropped from a subset.
const MATRIX: &[(&str, &str)] = &[
    ("benzene", "c1ccccc1"),
    ("benzene_kekule", "C1=CC=CC=C1"),
    ("toluene", "Cc1ccccc1"),
    ("aspirin", "O=C(C)Oc1ccccc1C(=O)O"),
    ("pyridine", "c1ccncc1"),
    ("pyrrole", "c1cc[nH]c1"),
    ("furan", "c1ccoc1"),
    ("imidazole", "c1c[nH]cn1"),
    ("naphthalene", "c1ccc2ccccc2c1"),
    ("imidazolium", "c1c[nH+]c[nH]1"),
    ("tropylium", "c1ccc[cH+]cc1"),
];

/// A structure with no legal Kekulé assignment (§17, last row).
///
/// Five neutral aromatic CH carbons: each needs exactly one localized double,
/// and an odd number of such atoms has no perfect matching — there is always
/// one left over. (The *anion* `[cH-]1cccc1` is a different molecule and is
/// perfectly kekulizable as `C1=CC=C[CH-]1`; its carbanion carbon supplies the
/// lone pair instead of demanding a double.) Standardization must fail
/// *transactionally* (§10) rather than leave a half-assigned ring.
const UNKEKULIZABLE: &str = "c1cccc1";

// ---------------------------------------------------------------------------
// Helpers — all reading by key name, never by implementation type
// ---------------------------------------------------------------------------

fn parse(smiles: &str) -> Atomistic {
    use crate::io::smiles::SmilesIr;
    let ir = SmilesIr::parse(smiles).unwrap_or_else(|e| panic!("{smiles}: parse failed: {e}"));
    ir.to_atomistic()
        .unwrap_or_else(|e| panic!("{smiles}: to_atomistic failed: {e}"))
}

/// Parse and standardize.
///
/// §9 makes perception responsible for the whole answer — aromatic atoms,
/// aromatic bond type, *and* a legal localized integer — so `assign_aromaticity`
/// is the one entry point the standard's acceptance runs through.
///
/// Hydrogens are **not** added: perception reads implicit hydrogens off each
/// atom's valence, so `add_hydrogens` stays a separate, optional operation and
/// standardization never changes the structure behind the caller's back.
fn standardize(smiles: &str) -> Atomistic {
    let mut mol = parse(smiles);
    assign_aromaticity(&mut mol).expect("standardization writes its flags");
    mol
}

fn uint_prop(props: &Props, key: &str) -> Option<u32> {
    props.get(key).and_then(PropValue::as_f64).map(|v| {
        assert!(
            v.fract() == 0.0 && v >= 0.0,
            "{key} must be a non-negative integer, got {v}"
        );
        v as u32
    })
}

fn bond_type(mol: &Atomistic, bid: RelationId) -> u32 {
    let b = mol.get_bond(bid).expect("bond");
    uint_prop(&b.props, BOND_TYPE).unwrap_or_else(|| panic!("bond has no {BOND_TYPE}"))
}

fn bond_number(mol: &Atomistic, bid: RelationId) -> u32 {
    let b = mol.get_bond(bid).expect("bond");
    uint_prop(&b.props, BOND_NUMBER).unwrap_or_else(|| panic!("bond has no {BOND_NUMBER}"))
}

fn atomic_number(mol: &Atomistic, id: NodeId) -> u32 {
    let a = mol.get_atom(id).expect("atom");
    uint_prop(&a, ATOMIC_NUMBER).unwrap_or_else(|| panic!("atom has no {ATOMIC_NUMBER}"))
}

fn atom_is_aromatic(mol: &Atomistic, id: NodeId) -> bool {
    mol.get_atom(id)
        .ok()
        .is_some_and(|a| a.get(IS_AROMATIC) == Some(&PropValue::Bool(true)))
}

fn aromatic_bonds(mol: &Atomistic) -> Vec<RelationId> {
    mol.bonds()
        .filter(|(bid, _)| bond_type(mol, *bid) == TYPE_AROMATIC)
        .map(|(bid, _)| bid)
        .collect()
}

/// The molecule's `(bond_type, bond_number)` per bond, keyed by endpoint index
/// pair — a comparison that survives a different bond iteration order.
fn signature(mol: &Atomistic) -> HashMap<(usize, usize), (u32, u32)> {
    let index: HashMap<NodeId, usize> = mol
        .atoms()
        .enumerate()
        .map(|(i, (id, _))| (id, i))
        .collect();
    mol.bonds()
        .map(|(bid, _)| {
            let ends = mol.get_bond(bid).expect("endpoints").nodes;
            let (a, b) = (ends[0], ends[1]);
            let (i, j) = (index[&a], index[&b]);
            (
                (i.min(j), i.max(j)),
                (bond_type(mol, bid), bond_number(mol, bid)),
            )
        })
        .collect()
}

// ---------------------------------------------------------------------------
// §18.1 — no aromatic 1.5 anywhere in the core data
// ---------------------------------------------------------------------------

#[test]
fn no_bond_carries_a_fractional_order() {
    for (name, smiles) in MATRIX {
        let mol = standardize(smiles);
        for (bid, bond) in mol.bonds() {
            for (key, value) in bond.props.iter() {
                if let Some(v) = value.as_f64() {
                    assert!(
                        v.fract() == 0.0,
                        "{name}: bond {bid:?} prop {key} = {v} is fractional; \
                         aromaticity is a bond type, never a number"
                    );
                }
            }
        }
    }
}

// ---------------------------------------------------------------------------
// §18.2 — no separate localized-order field
// ---------------------------------------------------------------------------

#[test]
fn there_is_exactly_one_localized_order_field() {
    // Two fields that can disagree is the defect, not the fix.
    for (name, smiles) in MATRIX {
        let mol = standardize(smiles);
        for (_, bond) in mol.bonds() {
            for banned in ["kekule_order", "order"] {
                assert!(
                    !bond.props.contains_key(banned),
                    "{name}: bond still carries `{banned}`; \
                     the localized integer is `{BOND_NUMBER}` alone"
                );
            }
        }
    }
}

// ---------------------------------------------------------------------------
// §18.3 / §18.4 — where each fact lives
// ---------------------------------------------------------------------------

#[test]
fn aromatic_atoms_and_bonds_are_marked_in_their_own_places() {
    for (name, smiles) in MATRIX {
        let mol = standardize(smiles);
        let aromatic = aromatic_bonds(&mol);
        assert!(!aromatic.is_empty(), "{name}: no aromatic bond perceived");

        // A bond typed aromatic must join two aromatic atoms (§2.3: the two
        // facts are recorded separately, and must agree).
        for bid in aromatic {
            let ends = mol.get_bond(bid).expect("endpoints").nodes;
            let (a, b) = (ends[0], ends[1]);
            assert!(
                atom_is_aromatic(&mol, a) && atom_is_aromatic(&mol, b),
                "{name}: aromatic bond between atoms not marked aromatic"
            );
        }
    }
}

// ---------------------------------------------------------------------------
// §18.5 / §4.2 — every aromatic bond has a legal integer number
// ---------------------------------------------------------------------------

#[test]
fn every_aromatic_bond_has_a_localized_single_or_double() {
    for (name, smiles) in MATRIX {
        let mol = standardize(smiles);
        assert!(
            !aromatic_bonds(&mol).is_empty(),
            "{name}: nothing was perceived aromatic, so this clause asserts nothing"
        );
        for bid in aromatic_bonds(&mol) {
            let n = bond_number(&mol, bid);
            assert!(
                n == NUMBER_SINGLE || n == NUMBER_DOUBLE,
                "{name}: aromatic bond has {BOND_NUMBER} = {n}; \
                 a standardized aromatic bond is single or double, never unknown"
            );
        }
    }
}

// ---------------------------------------------------------------------------
// §4.1 — a plain bond's two facts agree
// ---------------------------------------------------------------------------

#[test]
fn a_plain_bond_type_matches_its_number() {
    for (name, smiles) in MATRIX {
        let mol = standardize(smiles);
        for (bid, _) in mol.bonds() {
            let t = bond_type(&mol, bid);
            if t == TYPE_AROMATIC {
                continue;
            }
            assert!(
                matches!(t, TYPE_SINGLE | TYPE_DOUBLE | TYPE_TRIPLE),
                "{name}: bond has an unknown {BOND_TYPE} = {t} after standardization"
            );
            assert_eq!(
                bond_number(&mol, bid),
                t,
                "{name}: a {BOND_TYPE} of {t} must carry the matching {BOND_NUMBER}"
            );
        }
    }
}

// ---------------------------------------------------------------------------
// §18.6 — the two spellings converge
// ---------------------------------------------------------------------------

#[test]
fn the_aromatic_and_kekule_spellings_agree_on_every_aromatic_fact() {
    for (aromatic_smi, kekule_smi) in [
        ("c1ccccc1", "C1=CC=CC=C1"),
        ("Cc1ccccc1", "CC1=CC=CC=C1"),
        ("O=C(C)Oc1ccccc1C(=O)O", "O=C(C)OC1=CC=CC=C1C(=O)O"),
        ("c1ccncc1", "C1=CC=NC=C1"),
        ("c1ccc2ccccc2c1", "C1=CC=C2C=CC=CC2=C1"),
    ] {
        let a = standardize(aromatic_smi);
        let k = standardize(kekule_smi);

        assert_eq!(
            a.n_atoms(),
            k.n_atoms(),
            "{aromatic_smi} vs {kekule_smi}: different atom counts"
        );
        assert!(
            !aromatic_bonds(&a).is_empty(),
            "{aromatic_smi}: nothing perceived aromatic; the comparison would be vacuous"
        );
        assert_eq!(
            aromatic_bonds(&a).len(),
            aromatic_bonds(&k).len(),
            "{aromatic_smi} vs {kekule_smi}: different aromatic bond counts"
        );

        let arom_atoms = |m: &Atomistic| -> usize {
            m.atoms().filter(|(id, _)| atom_is_aromatic(m, *id)).count()
        };
        assert_eq!(
            arom_atoms(&a),
            arom_atoms(&k),
            "{aromatic_smi} vs {kekule_smi}: different aromatic atom counts"
        );

        // §8.3 allows a different but equivalent Kekulé phase, so the *numbers*
        // need not match — only the aromatic semantics above.
    }
}

// ---------------------------------------------------------------------------
// §7 — π-electron demand differs by atom
// ---------------------------------------------------------------------------

#[test]
fn pyrrole_type_nitrogen_is_not_forced_into_a_double() {
    // The rule that would pass benzene and fail chemistry: "every aromatic atom
    // gets exactly one localized double".
    let pyrrole = standardize("c1cc[nH]c1");
    let n = pyrrole
        .atoms()
        .find(|(_, a)| uint_prop(a, ATOMIC_NUMBER) == Some(Z_N))
        .map(|(id, _)| id)
        .expect("pyrrole N");

    let doubles_at_n = pyrrole
        .bonds()
        .filter(|(bid, _)| bond_number(&pyrrole, *bid) == NUMBER_DOUBLE)
        .filter(|(bid, _)| {
            let ends = pyrrole.get_bond(*bid).expect("endpoints").nodes;
            let (a, b) = (ends[0], ends[1]);
            a == n || b == n
        })
        .count();
    assert_eq!(
        doubles_at_n, 0,
        "pyrrole's [nH] contributes its lone pair; it takes no localized double"
    );

    // Pyridine's N, by contrast, does take one.
    let pyridine = standardize("c1ccncc1");
    let pn = pyridine
        .atoms()
        .find(|(_, a)| uint_prop(a, ATOMIC_NUMBER) == Some(Z_N))
        .map(|(id, _)| id)
        .expect("pyridine N");
    let doubles_at_pn = pyridine
        .bonds()
        .filter(|(bid, _)| bond_number(&pyridine, *bid) == NUMBER_DOUBLE)
        .filter(|(bid, _)| {
            let ends = pyridine.get_bond(*bid).expect("endpoints").nodes;
            let (a, b) = (ends[0], ends[1]);
            a == pn || b == pn
        })
        .count();
    assert_eq!(doubles_at_pn, 1, "pyridine's N takes exactly one double");
}

// ---------------------------------------------------------------------------
// §9 — perception must not spill outside the ring system
// ---------------------------------------------------------------------------

#[test]
fn a_bond_from_a_ring_atom_to_a_substituent_is_not_aromatic() {
    // Aspirin's ester oxygen and its carboxyl carbon both hang off the ring.
    let aspirin = standardize("O=C(C)Oc1ccccc1C(=O)O");
    for bid in aromatic_bonds(&aspirin) {
        let ends = aspirin.get_bond(bid).expect("endpoints").nodes;
        let (a, b) = (ends[0], ends[1]);
        assert_eq!(
            (atomic_number(&aspirin, a), atomic_number(&aspirin, b)),
            (Z_C, Z_C),
            "aspirin: only the six ring C–C bonds are aromatic"
        );
    }
    assert_eq!(
        aromatic_bonds(&aspirin).len(),
        6,
        "aspirin has exactly six aromatic bonds"
    );

    // Toluene's methyl bond likewise.
    let toluene = standardize("Cc1ccccc1");
    assert_eq!(
        aromatic_bonds(&toluene).len(),
        6,
        "toluene: ring bonds only"
    );
}

// ---------------------------------------------------------------------------
// §18.8 — the data cannot be read as "all doubles"
// ---------------------------------------------------------------------------

#[test]
fn a_benzene_ring_is_never_six_localized_doubles() {
    // This is the reported defect, stated as a property of the data: whatever a
    // renderer does, the numbers it reads must not say "six doubles".
    for (name, smiles) in [
        ("benzene", "c1ccccc1"),
        ("aspirin", "O=C(C)Oc1ccccc1C(=O)O"),
    ] {
        let mol = standardize(smiles);
        let ring = aromatic_bonds(&mol);
        assert_eq!(
            ring.len(),
            6,
            "{name}: expected a six-membered aromatic ring"
        );
        let doubles = ring
            .iter()
            .filter(|bid| bond_number(&mol, **bid) == NUMBER_DOUBLE)
            .count();
        assert_eq!(
            doubles,
            ring.len() / 2,
            "{name}: a six-membered aromatic ring carries three localized doubles"
        );

        // And every ring atom is in exactly one of them — that, not a printed
        // `1,2,1,2,…` list, is what alternating means. The bond table's order is
        // not the ring's.
        let mut touched: HashMap<NodeId, usize> = HashMap::new();
        for bid in ring
            .iter()
            .filter(|bid| bond_number(&mol, **bid) == NUMBER_DOUBLE)
        {
            let ends = mol.get_bond(*bid).expect("endpoints").nodes;
            let (a, b) = (ends[0], ends[1]);
            *touched.entry(a).or_default() += 1;
            *touched.entry(b).or_default() += 1;
        }
        assert!(
            touched.values().all(|n| *n == 1),
            "{name}: an aromatic atom carries two localized doubles"
        );
    }
}

// ---------------------------------------------------------------------------
// §10 / §18.10 — the standardization operation's own properties
// ---------------------------------------------------------------------------

#[test]
fn standardizing_twice_changes_nothing() {
    for (name, smiles) in MATRIX {
        let once = standardize(smiles);
        let mut twice = once.clone();
        assign_aromaticity(&mut twice).expect("standardization writes its flags");
        assert_eq!(
            signature(&once),
            signature(&twice),
            "{name}: standardization is not idempotent"
        );
    }
}

#[test]
fn the_same_input_gives_the_same_assignment() {
    for (name, smiles) in MATRIX {
        assert_eq!(
            signature(&standardize(smiles)),
            signature(&standardize(smiles)),
            "{name}: standardization is not deterministic"
        );
    }
}

#[test]
fn standardization_does_not_touch_the_non_aromatic_region() {
    // Aspirin's two C=O and its ester/methyl single bonds are outside every
    // aromatic system and must come through untouched.
    let before = parse("O=C(C)Oc1ccccc1C(=O)O");
    let after = standardize("O=C(C)Oc1ccccc1C(=O)O");

    let carbonyls = |m: &Atomistic| {
        m.bonds()
            .filter(|(bid, _)| {
                let ends = m.get_bond(*bid).expect("endpoints").nodes;
                let (a, b) = (ends[0], ends[1]);
                let pair: HashSet<u32> = [atomic_number(m, a), atomic_number(m, b)]
                    .into_iter()
                    .collect();
                pair == HashSet::from([Z_C, Z_O])
            })
            .count()
    };
    assert_eq!(
        carbonyls(&before),
        carbonyls(&after),
        "aspirin: the C–O/C=O set changed during standardization"
    );

    for (bid, _) in after.bonds() {
        if bond_type(&after, bid) == TYPE_AROMATIC {
            continue;
        }
        assert_ne!(
            bond_number(&after, bid),
            NUMBER_UNKNOWN,
            "aspirin: a non-aromatic bond lost its number"
        );
    }
}

#[test]
fn a_legal_kekule_input_keeps_its_own_phase() {
    // §8.2: an input that already states a legal localized assignment has one;
    // there is no reason to renumber it, and doing so would make round-tripping
    // a file rewrite its bonds.
    let before = parse("C1=CC=CC=C1");
    let after = standardize("C1=CC=CC=C1");

    let numbers = |m: &Atomistic| -> HashMap<(usize, usize), u32> {
        signature(m).into_iter().map(|(k, (_, n))| (k, n)).collect()
    };
    // `before` is pre-standardization, so read its stated orders the same way
    // the standard says they arrive: as bond numbers.
    let stated: HashMap<(usize, usize), u32> = numbers(&before);
    assert_eq!(
        stated,
        numbers(&after),
        "the input's own Kekulé phase was rewritten"
    );
}

#[test]
fn a_system_with_no_legal_assignment_is_left_untouched() {
    // Five aromatic CH carbons: an odd number of atoms each demanding one
    // localized double cannot all be matched. Whatever the failure signal is,
    // the molecule must not come back half-assigned.
    let before = parse(UNKEKULIZABLE);
    let after = standardize(UNKEKULIZABLE);

    let half_assigned = after.bonds().any(|(bid, _)| {
        bond_type(&after, bid) == TYPE_AROMATIC && bond_number(&after, bid) == NUMBER_UNKNOWN
    }) && after.bonds().any(|(bid, _)| {
        bond_type(&after, bid) == TYPE_AROMATIC && bond_number(&after, bid) != NUMBER_UNKNOWN
    });

    assert!(
        !half_assigned,
        "an unkekulizable ring came back partly assigned; failure must be transactional"
    );
    assert_eq!(
        before.n_bonds(),
        after.n_bonds(),
        "a failed standardization must not change connectivity"
    );
}

// ---------------------------------------------------------------------------
// The two paths are different questions, and stay separate
// ---------------------------------------------------------------------------

#[test]
fn a_declared_aromatic_input_is_kekulized_not_re_perceived() {
    // The notation already answered "which bonds are aromatic". Standardization
    // owes it a phase and a valence check — not a second opinion.
    let declared = parse("c1ccccc1");
    assert_eq!(
        declared
            .bonds()
            .filter(|(bid, _)| bond_type(&declared, *bid) == TYPE_AROMATIC)
            .count(),
        6,
        "the parser is expected to declare the aromatic subgraph"
    );

    let out = standardize("c1ccccc1");
    assert_eq!(aromatic_bonds(&out).len(), 6, "the declaration is honoured");
}

#[test]
fn kekulization_alone_never_invents_aromaticity() {
    // `assign_kekule_bond_orders` decides a phase; it does not decide which bonds are
    // aromatic. A Kekulé benzene has no aromatic bonds to phase, so it comes
    // back untouched — that is the boundary that keeps the two from calling
    // each other.
    let kekule = parse("C1=CC=CC=C1");
    let mut out = kekule.clone();
    assert!(
        assign_kekule_bond_orders(&mut out).expect("kekulization writes its numbers"),
        "nothing to phase is a success"
    );

    assert_eq!(
        out.bonds()
            .filter(|(bid, _)| bond_type(&out, *bid) == TYPE_AROMATIC)
            .count(),
        0,
        "kekulization must not mark anything aromatic"
    );
    assert_eq!(
        signature(&kekule),
        signature(&out),
        "a molecule with no declared aromatic bond is left exactly as it was"
    );
}

#[test]
fn perception_alone_never_needs_hydrogens_added() {
    // Implicit hydrogens are read off valence, so `add_hydrogens` is optional
    // and changes no answer. If this ever fails, standardization has grown a
    // hidden structural modification.
    for (name, smiles) in MATRIX {
        let heavy = standardize(smiles);
        let repleted =
            add_hydrogens(&parse(smiles)).expect("repletion succeeds on a well-formed graph");
        let mut with_h = repleted;
        assign_aromaticity(&mut with_h).expect("standardization writes its flags");
        assert_eq!(
            aromatic_bonds(&heavy).len(),
            aromatic_bonds(&with_h).len(),
            "{name}: perception disagrees with and without explicit hydrogens"
        );
    }
}

#[test]
fn a_protonated_ring_nitrogen_never_ends_up_four_valent() {
    // §7, the case pyrrole and pyridine alone do not cover. Imidazole has two
    // candidate Kekulé structures, and they tie unless the penalty can see the
    // hydrogen the graph does not draw — otherwise the tie-break could put the
    // double on the [nH], giving a neutral nitrogen four bonds.
    for (name, smiles) in [
        ("imidazole", "c1c[nH]cn1"),
        ("tetrazole", "c1nnn[nH]1"),
        ("benzimidazole", "c1ccc2[nH]cnc2c1"),
        ("purine", "c1nc2[nH]cnc2cn1"),
    ] {
        let mol = standardize(smiles);
        let index: HashMap<NodeId, usize> = mol
            .atoms()
            .enumerate()
            .map(|(i, (id, _))| (id, i))
            .collect();

        let mut valence = vec![0u32; mol.n_atoms()];
        for (bid, _) in mol.bonds() {
            let ends = mol.get_bond(bid).expect("endpoints").nodes;
            let (a, b) = (ends[0], ends[1]);
            let n = bond_number(&mol, bid);
            valence[index[&a]] += n;
            valence[index[&b]] += n;
        }

        for (id, atom) in mol.atoms() {
            if uint_prop(&atom, ATOMIC_NUMBER) != Some(Z_N) {
                continue;
            }
            let total =
                valence[index[&id]] + crate::perceive::n_implicit_hydrogens(&mol, id).unwrap_or(0);
            assert!(
                total <= 3,
                "{name}: a neutral ring nitrogen reached valence {total}"
            );
        }
    }
}
