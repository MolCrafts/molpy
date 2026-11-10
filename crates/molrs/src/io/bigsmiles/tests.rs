//! Reading `BigSMILES` / generative `BigSMILES` and turning it into the
//! polymer model. Strings are the notation's own examples or written to
//! exercise one rule; expected values follow from the rules in `mod.rs`.

use super::*;
use crate::builder::{Amount, Distribution, Part};
use crate::core::{CoarseGrain, keys};
use crate::io::smiles::{Notation, SmilesErrorKind};

fn ir(text: &str) -> BigSmilesIr {
    BigSmilesIr::parse(text).unwrap_or_else(|e| panic!("parse({text:?}) failed: {e}"))
}

fn kind(text: &str) -> SmilesErrorKind {
    BigSmilesIr::parse(text)
        .expect_err("the string must be refused")
        .kind
}

fn object(spec: &MoleculeSpec, part: usize) -> &ObjectSpec {
    match &spec.parts[part] {
        PartSpec::Stochastic(o) => o,
        PartSpec::Smiles(_) => panic!("part {part} is SMILES"),
    }
}

fn bead_types(sites: &CoarseGrain) -> Vec<String> {
    sites
        .bead_ids()
        .map(|id| {
            sites
                .get_bead(id)
                .unwrap()
                .get_str(keys::BEAD_TYPE)
                .unwrap()
                .to_owned()
        })
        .collect()
}

// -- reading -------------------------------------------------------------------

#[test]
fn the_generative_bigsmiles_readme_example_reads_its_units_and_distribution() {
    let text = "{[][<]CC([>])c1ccccc1, [<]CC([>])C(=O)OC; CC(C)[>], CC(C)[>], [<][Br][]}\
                |schulz_zimm(700, 600)|";
    let ir = ir(text);
    assert_eq!(ir.molecules.len(), 1);
    let o = object(&ir.molecules[0], 0);
    assert_eq!(o.repeat_units.len(), 2);
    assert_eq!(o.end_groups.len(), 3);
    assert!(o.left.is_none() && o.right.is_none());
    assert_eq!(
        o.distribution,
        Some(Distribution::SchulzZimm {
            mw: 700.0,
            mn: 600.0
        })
    );
    // It samples: every open port finds an end group.
    let molecules = ir.to_molecules().unwrap();
    let types = bead_types(&molecules[0].sample(3).unwrap());
    assert!(types.iter().any(|t| t.starts_with("0.0.e")));
}

#[test]
fn descriptor_weights_are_read_per_port() {
    let ir = ir("{[][<|0.5|]CC[>|0 1 2|];[<]C[]}");
    let fragment = &object(&ir.molecules[0], 0).repeat_units[0];
    assert_eq!(
        fragment.weights,
        vec![Some(vec![0.5]), Some(vec![0.0, 1.0, 2.0])]
    );
    let cap = &object(&ir.molecules[0], 0).end_groups[0];
    assert_eq!(cap.weights, vec![None]);
}

#[test]
fn terminal_ids_and_weights_are_read() {
    let ir = ir("C{[>1|1 0 0|][<1]CC[>1];[<1]C[<1]}");
    let o = object(&ir.molecules[0], 1);
    let left = o.left.as_ref().unwrap();
    assert_eq!(left.kind, DescriptorKind::Right);
    assert_eq!(left.label, "1");
    assert_eq!(left.weights, Some(vec![1.0, 0.0, 0.0]));
}

#[test]
fn smiles_beside_an_object_carries_its_terminals() {
    let ir = ir("CC{[>][<]CC[>][<]}O");
    let parts = &ir.molecules[0].parts;
    assert_eq!(parts.len(), 3);
    let ports = |p: usize| match &parts[p] {
        PartSpec::Smiles(body) => body.ir.to_template().unwrap().n_ports(),
        PartSpec::Stochastic(_) => panic!("part {p} is an object"),
    };
    assert_eq!(ports(0), 1, "CC[>]");
    assert_eq!(ports(2), 1, "[<]O");
}

#[test]
fn a_bond_symbol_next_to_an_object_is_its_terminal_order() {
    let ir = ir("C={[>][<]=CC=[>][<]}=C");
    let o = object(&ir.molecules[0], 1);
    assert_eq!(o.left.as_ref().unwrap().order, Some(BondKind::Double));
    assert_eq!(o.right.as_ref().unwrap().order, Some(BondKind::Double));
    // The pieces agree on the double bond, so the molecule is built.
    assert!(ir.to_molecules().is_ok());
}

#[test]
fn molecules_and_amounts_are_split_at_dots() {
    let ir = ir("CCO.|1000|{[][<]CC[>];[<]C,[>]C[]}|gauss(500, 0)|.|20%|");
    assert_eq!(ir.molecules.len(), 2);
    assert_eq!(ir.molecules[0].amount, Some(Amount::Total(1000.0)));
    assert_eq!(ir.molecules[1].amount, Some(Amount::Percent(20.0)));
    let system = ir.to_system().unwrap();
    let targets = system.targets();
    assert!((targets[1] - 250.0).abs() < 1e-9);
}

#[test]
fn every_distribution_name_reads() {
    for (text, expected) in [
        ("flory_schulz(0.1)", Distribution::FlorySchulz { a: 0.1 }),
        ("flory_schulz(0.1,)", Distribution::FlorySchulz { a: 0.1 }),
        (
            "gauss(500, 20)",
            Distribution::Gauss {
                mu: 500.0,
                sigma: 20.0,
            },
        ),
        (
            "uniform(100, 300)",
            Distribution::Uniform {
                low: 100.0,
                high: 300.0,
            },
        ),
        (
            "log_normal(1000, 1.5)",
            Distribution::LogNormal { mn: 1000.0, d: 1.5 },
        ),
        ("poisson(40)", Distribution::Poisson { n: 40.0 }),
    ] {
        let ir = ir(&format!("{{[][<]CC[>];[<]C,[>]C[]}}|{text}|"));
        assert_eq!(
            object(&ir.molecules[0], 0).distribution,
            Some(expected),
            "{text}"
        );
    }
}

// -- the model -------------------------------------------------------------------

#[test]
fn a_system_samples_to_its_heavy_atom_mass() {
    let system = ir("{[][<]CC[>];[<][H],[>][H][]}|poisson(300)|.|3000|")
        .to_system()
        .unwrap();
    let sample = system.sample(7).unwrap();
    let total: f64 = sample.molecules.iter().map(|m| m.heavy_atom_mass).sum();
    assert!(total >= 3000.0);
    let last = sample.molecules.last().unwrap().heavy_atom_mass;
    assert!(total - last < 3000.0);
}

#[test]
fn units_are_named_by_position() {
    let molecules = ir("[H]{[>][<]CC[>];[<][H][]}|gauss(100, 0)|")
        .to_molecules()
        .unwrap();
    let Part::Fixed(prefix) = &molecules[0].parts()[0] else {
        panic!("the prefix is fixed");
    };
    assert_eq!(prefix.name(), "0.0");
    let types = bead_types(&molecules[0].sample(1).unwrap());
    assert_eq!(types[0], "0.0");
    assert!(
        types
            .iter()
            .all(|t| t == "0.0" || t == "0.1.r0" || t == "0.1.e0")
    );
}

#[test]
fn a_molecule_without_an_amount_is_not_a_system() {
    let err = ir("{[][<]CC[>];[<]C,[>]C[]}|poisson(100)|")
        .to_system()
        .unwrap_err();
    assert!(matches!(err.kind, SmilesErrorKind::BigBuild(_)));
}

// -- refusals ----------------------------------------------------------------------

#[test]
fn an_unknown_distribution_is_refused() {
    assert_eq!(
        kind("{[][<]CC[>][]}|zipf(2)|"),
        SmilesErrorKind::BigUnknownDistribution("zipf".to_owned())
    );
}

#[test]
fn a_distribution_with_the_wrong_arity_is_refused() {
    assert!(matches!(
        kind("{[][<]CC[>][]}|gauss(1)|"),
        SmilesErrorKind::BigDistributionArity {
            expected: 2,
            found: 1,
            ..
        }
    ));
}

#[test]
fn out_of_range_distribution_parameters_are_refused() {
    assert!(matches!(
        kind("{[][<]CC[>][]}|schulz_zimm(600, 700)|"),
        SmilesErrorKind::BigInvalidNumber(_)
    ));
}

#[test]
fn unsupported_notation_is_named() {
    for text in [
        "{[][<]C{[<]CC[>]}C[>][]}",
        "C({[>][<]CC[>][]})C",
        "C[#frag]C",
        "{[][<[<1]1]CC[>][]}",
        "{[][$:1]CC[$][]}",
    ] {
        assert!(
            matches!(kind(text), SmilesErrorKind::BigUnsupported(_)),
            "{text}: {:?}",
            kind(text)
        );
    }
}

#[test]
fn a_descriptor_id_is_an_integer() {
    assert!(matches!(
        kind("{[][$a]CC[$a][]}"),
        SmilesErrorKind::InvalidDescriptorLabel(_)
    ));
}

#[test]
fn the_squash_operator_is_not_bigsmiles() {
    assert_eq!(kind("{[][!]CC[!][]}"), SmilesErrorKind::UnexpectedChar('!'));
}

#[test]
fn smiles_joined_to_an_empty_terminal_is_refused() {
    assert!(matches!(
        kind("CC{[][<]CC[>][]}"),
        SmilesErrorKind::BigMalformedObject(_)
    ));
}

#[test]
fn malformed_objects_are_refused() {
    for text in [
        "{[][<]CC[>]",
        "{CC}",
        "{[][]}",
        "{[];[<]C[]}",
        "{[][<]CC[>],,[<]C[]}",
    ] {
        assert!(
            matches!(kind(text), SmilesErrorKind::BigMalformedObject(_)),
            "{text}: {:?}",
            kind(text)
        );
    }
}

#[test]
fn invalid_amounts_are_refused() {
    for text in ["CCO.|-5|", "CCO.|120%|", "CCO.|lots|"] {
        assert!(
            matches!(kind(text), SmilesErrorKind::BigInvalidAmount(_)),
            "{text}"
        );
    }
}

#[test]
fn a_unit_error_points_into_the_whole_string() {
    let text = "{[][<]CC[>],[<]CXC[>][]}";
    let err = BigSmilesIr::parse(text).unwrap_err();
    assert_eq!(err.notation, Notation::BigSmiles);
    assert_eq!(err.input, text);
    assert_eq!(&text[err.span.start..err.span.start + 1], "X");
}

// -- assembly ----------------------------------------------------------------------

/// Assemble one sample: hydrogens added to every template, coordinates that
/// are merely distinct points, one site per fragment on a line.
fn assembled(text: &str, seed: u64) -> (crate::core::Atomistic, Vec<String>) {
    use crate::builder::{Assembler, SitePlacer};
    use crate::perceive::add_hydrogens;
    let molecule = ir(text).to_molecules().unwrap().remove(0);
    let mut library = std::collections::HashMap::new();
    for (name, template) in molecule.library() {
        let mut template = add_hydrogens(&template).unwrap();
        for (row, id) in template
            .atom_ids()
            .collect::<Vec<_>>()
            .into_iter()
            .enumerate()
        {
            template.set_atom(id, keys::X, row as f64).unwrap();
            template.set_atom(id, keys::Y, (row % 2) as f64).unwrap();
            template.set_atom(id, keys::Z, 0.0).unwrap();
        }
        library.insert(name, template);
    }
    let mut sites = molecule.sample(seed).unwrap();
    let types = bead_types(&sites);
    for (row, id) in sites.bead_ids().collect::<Vec<_>>().into_iter().enumerate() {
        sites.set_bead(id, keys::X, 4.0 * row as f64).unwrap();
        sites.set_bead(id, keys::Y, 0.0).unwrap();
        sites.set_bead(id, keys::Z, 0.0).unwrap();
    }
    let world = Assembler::new(library, Box::new(SitePlacer::new()), None)
        .assemble(&sites)
        .unwrap();
    (world, types)
}

fn n_element(mol: &crate::core::Atomistic, z: u8) -> usize {
    mol.atom_ids()
        .filter(|&id| mol.get_atom(id).unwrap().atomic_number() == Some(z))
        .count()
}

#[test]
fn an_assembled_polyethylene_chain_is_an_alkane() {
    // CH3-(CH2-CH2)n-CH3: every hydrogen handle of a joined port leaves.
    let (world, types) = assembled("C{[>][<]CC[>];[<]C[]}|gauss(120, 0)|", 2);
    let n = types.iter().filter(|t| t.ends_with(".r0")).count();
    let carbons = n_element(&world, 6);
    assert_eq!(carbons, 2 * n + 2);
    assert_eq!(n_element(&world, 1), 2 * carbons + 2, "CnH2n+2");
    assert_eq!(world.n_ports(), 0, "every port joined");
}

// -- review regressions ---------------------------------------------------------------

#[test]
fn a_wide_gauss_with_hydrogen_caps_still_grows_every_chain() {
    // Negative draws clip to zero; a chain is never the H–H cap dimer.
    let system = ir("{[][<]CC[>];[<][H],[>][H][]}|gauss(100, 60)|.|1e4|")
        .to_system()
        .unwrap();
    let sample = system.sample(1).unwrap();
    assert!(sample.molecules.iter().all(|m| m.n_repeat_units >= 1));
}

#[test]
fn a_mean_no_sampler_represents_is_refused_when_read() {
    assert!(matches!(
        kind("{[][<]CC[>];[<][H],[>][H][]}|poisson(1e30)|"),
        SmilesErrorKind::BigInvalidNumber(_)
    ));
}

#[test]
fn an_object_without_end_groups_ends_in_hydrogen() {
    // The commonest BigSMILES homopolymer: no end groups written.
    let molecule = ir("{[][$]CC[$][]}|poisson(100)|")
        .to_molecules()
        .unwrap()
        .remove(0);
    assert!(molecule.library().contains_key("0.0.h0"));
    let (world, types) = assembled("{[][$]CC[$][]}|gauss(120, 0)|", 2);
    let n = types.iter().filter(|t| t.ends_with(".r0")).count();
    assert_eq!(types.iter().filter(|t| t.ends_with(".h0")).count(), 2);
    assert_eq!(n_element(&world, 6), 2 * n);
    assert_eq!(n_element(&world, 1), 4 * n + 2, "H-(CH2CH2)n-H");
    assert_eq!(world.n_ports(), 0);
}

#[test]
fn each_kind_of_port_gets_its_own_hydrogen() {
    let molecule = ir("{[][<]CC[>][]}|gauss(60, 0)|")
        .to_molecules()
        .unwrap()
        .remove(0);
    let library = molecule.library();
    assert!(library.contains_key("0.0.h0") && library.contains_key("0.0.h1"));
    assert_eq!(
        molecule.sample(0).unwrap().n_bonds(),
        bead_types(&molecule.sample(0).unwrap()).len() - 1
    );
}

#[test]
fn written_list_weights_keep_their_positions_beside_implicit_caps() {
    // Two written ports, so lists of two; the implicit caps are not counted.
    let molecule = ir("{[][<|1 0|]CC[>|1 0|][]}|gauss(60, 0)|").to_molecules();
    assert!(
        molecule.is_ok(),
        "{:?}",
        molecule.err().map(|e| e.to_string())
    );
}

#[test]
fn written_end_groups_are_not_supplemented() {
    // `<` ends have no cap among the written end groups: no hydrogen is added.
    let molecule = ir("{[][<]CC[>];[<]C[]}|gauss(60, 0)|")
        .to_molecules()
        .unwrap()
        .remove(0);
    assert!(molecule.library().keys().all(|k| !k.contains(".h")));
    assert!(molecule.sample(0).is_err());
}

#[test]
fn a_double_bond_port_has_no_hydrogen_cap() {
    // Hydrogen forms one bond; with no end group there is nothing to start from.
    let err = ir("{[][$]=CC=[$][]}|poisson(100)|")
        .to_molecules()
        .unwrap_err();
    assert!(matches!(err.kind, SmilesErrorKind::BigBuild(_)));
    // A refusal of the model is not a parse error.
    assert!(err.to_string().starts_with("BigSMILES error at position 0"));
}

#[test]
fn a_build_refusal_points_at_the_piece_that_was_refused() {
    // The second molecule's prefix has two ports for one neighbour.
    let text = "CCO.[<]CC{[>][<]CC[>];[<]C[]}";
    let err = ir(text).to_molecules().unwrap_err();
    assert!(matches!(err.kind, SmilesErrorKind::BigBuild(_)));
    assert_eq!(
        &text[err.span.start..err.span.end],
        "[<]CC{[>][<]CC[>];[<]C[]}"
    );

    // An end group with two ports: the object is refused, at the object.
    let text = "C{[>][<]CC[>];[<]C[>][]}";
    let err = ir(text).to_molecules().unwrap_err();
    assert_eq!(
        &text[err.span.start..err.span.end],
        "{[>][<]CC[>];[<]C[>][]}"
    );
}

#[test]
fn a_terminal_facing_nothing_is_refused() {
    let err = ir("C{[>][<]CC[>][<]}|gauss(60, 0)|")
        .to_molecules()
        .unwrap_err();
    assert!(matches!(err.kind, SmilesErrorKind::BigBuild(_)));
}
