//! PSMILES, the polymer SMILES of the Ramprasad group: one repeat unit written
//! as SMILES whose two `*` atoms mark where the unit joins its neighbours
//! (`[*]CC[*]` is polyethylene, `[*]CC([*])c1ccccc1` polystyrene).
//!
//! A `*` here is not an atom. It is a joining site with no head or tail, which
//! is exactly the `BigSMILES` / `CGsmiles` descriptor `[$]`, so the parser reads
//! it as one ([`Dialect::Psmiles`]): `[*]CC[*]` and `[$]CC[$]` give the same
//! IR. Everything downstream of the parse — [`FragmentIr`], ported templates,
//! the assembler — is therefore shared with the fragment dialect.
//!
//! What the notation adds on top of that reading is two rules, both checked
//! here rather than discovered later:
//!
//! * each `*` is bonded to exactly one atom — `C*C`, `**C`, `C(*C)` and a ring
//!   digit on a star are [`SmilesErrorKind::PsmilesStarNotTerminal`];
//! * a repeat unit has exactly two `*` — anything else is
//!   [`SmilesErrorKind::PsmilesStarCount`].
//!
//! [`FragmentIr`]: crate::io::smiles::FragmentIr
//!
//! References: Kuenneth, C. et al., *polyBERT: a chemical language model to
//! enable fully machine-driven ultrafast polymer informatics*, Nat. Commun.
//! **14**, 4099 (2023), DOI: 10.1038/s41467-023-39868-6 (the PSMILES
//! convention); Lin, T.-S. et al., *BigSMILES*, ACS Cent. Sci. **5**,
//! 1523–1531 (2019), DOI: 10.1021/acscentsci.9b00476 (the `$` descriptor).

use crate::line_notation::Dialect;
use crate::line_notation::ast::{BondingDescriptor, Chain, ChainElement, SmilesIr};
use crate::line_notation::error::{SmilesError, SmilesErrorKind};
use crate::line_notation::parser::parse_paired;

/// The number of `*` joining sites a PSMILES repeat unit carries.
const STARS_PER_UNIT: usize = 2;

/// Parse a PSMILES repeat unit into the shared IR, each `*` read as an
/// unlabelled `$` descriptor on the atom it is bonded to.
///
/// # Errors
///
/// Everything [`parse_fragment_smiles`](crate::line_notation::parser::parse_fragment_smiles)
/// refuses, stamped [`Notation::Psmiles`](crate::line_notation::error::Notation::Psmiles);
/// [`SmilesErrorKind::PsmilesStarNotTerminal`] for a `*` bonded to anything
/// but one atom; [`SmilesErrorKind::PsmilesStarCount`] when the unit does not
/// carry exactly two `*`; [`SmilesErrorKind::DescriptorInPlainSmiles`] for a
/// `BigSMILES` bracket (`[$]`, `[<]`) — PSMILES spells a joining site `*`.
pub(crate) fn parse_psmiles(input: &str) -> Result<SmilesIr, SmilesError> {
    let ir = parse_paired(input, Dialect::Psmiles)?;
    let n_stars: usize = ir.components.iter().map(count_descriptors).sum();
    if n_stars != STARS_PER_UNIT {
        return Err(SmilesError::new(
            SmilesErrorKind::PsmilesStarCount(n_stars),
            ir.span,
            input,
            Dialect::Psmiles.notation(),
        ));
    }
    Ok(ir)
}

fn count_descriptors(chain: &Chain) -> usize {
    let own: &[BondingDescriptor] = &chain.head.descriptors;
    own.len()
        + chain
            .tail
            .iter()
            .map(|elem| match elem {
                ChainElement::BondedAtom { atom, .. } => atom.descriptors.len(),
                ChainElement::Branch { chain, .. } => count_descriptors(chain),
                ChainElement::RingClosure { .. } => 0,
            })
            .sum::<usize>()
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::line_notation::ast::{BondKind, DescriptorKind};
    use crate::line_notation::error::Notation;
    use crate::line_notation::fixtures::{atom_nodes, descriptors};
    use crate::line_notation::parser::parse_fragment_smiles;

    fn psmiles(input: &str) -> SmilesIr {
        parse_psmiles(input).unwrap_or_else(|e| panic!("parse_psmiles({input:?}) failed: {e}"))
    }

    fn kinds_and_orders(ir: &SmilesIr) -> Vec<(DescriptorKind, String, Option<BondKind>)> {
        descriptors(ir)
            .into_iter()
            .map(|d| (d.kind, d.label, d.order))
            .collect()
    }

    fn err_kind(input: &str) -> SmilesErrorKind {
        parse_psmiles(input)
            .expect_err("the PSMILES string must be refused")
            .kind
    }

    #[test]
    fn test_stars_are_unlabelled_symmetric_descriptors_on_their_neighbours() {
        let ir = psmiles("*CC*");
        let nodes = atom_nodes(&ir);
        assert_eq!(nodes.len(), 2, "a star is not an atom");
        for node in nodes {
            assert_eq!(node.descriptors.len(), 1);
            assert_eq!(node.descriptors[0].kind, DescriptorKind::Symmetric);
            assert_eq!(node.descriptors[0].label, "");
            assert_eq!(node.descriptors[0].order, None);
        }
    }

    #[test]
    fn test_bracket_star_reads_as_the_bare_star() {
        assert_eq!(
            kinds_and_orders(&psmiles("[*]CC[*]")),
            kinds_and_orders(&psmiles("*CC*"))
        );
    }

    #[test]
    fn test_star_is_the_dollar_descriptor() {
        let star = psmiles("[*]CC([*])c1ccccc1");
        let dollar = parse_fragment_smiles("[$]CC([$])c1ccccc1").unwrap();
        assert_eq!(atom_nodes(&star).len(), atom_nodes(&dollar).len());
        assert_eq!(kinds_and_orders(&star), kinds_and_orders(&dollar));
        let nodes = atom_nodes(&star);
        assert_eq!(nodes[0].descriptors.len(), 1, "head star on the first C");
        assert_eq!(nodes[1].descriptors.len(), 1, "branch star on the second C");
    }

    #[test]
    fn test_both_stars_may_sit_on_one_atom() {
        // `[*]C[*]` is polymethylene: one carbon, two joining sites.
        let ir = psmiles("[*]C[*]");
        let nodes = atom_nodes(&ir);
        assert_eq!(nodes.len(), 1);
        assert_eq!(nodes[0].descriptors.len(), 2);
    }

    #[test]
    fn test_bond_written_to_a_star_is_its_order() {
        let ir = psmiles("*=CC=*");
        assert_eq!(
            kinds_and_orders(&ir),
            vec![
                (
                    DescriptorKind::Symmetric,
                    String::new(),
                    Some(BondKind::Double)
                ),
                (
                    DescriptorKind::Symmetric,
                    String::new(),
                    Some(BondKind::Double)
                ),
            ]
        );
    }

    #[test]
    fn test_star_between_two_atoms_is_refused() {
        assert_eq!(err_kind("*C*C"), SmilesErrorKind::PsmilesStarNotTerminal);
        assert_eq!(err_kind("*CC*C"), SmilesErrorKind::PsmilesStarNotTerminal);
    }

    #[test]
    fn test_star_bonded_to_a_star_is_refused() {
        assert_eq!(err_kind("**CC"), SmilesErrorKind::PsmilesStarNotTerminal);
        assert_eq!(err_kind("CC**"), SmilesErrorKind::PsmilesStarNotTerminal);
    }

    #[test]
    fn test_star_opening_a_branch_chain_is_refused() {
        assert_eq!(err_kind("*C(*C)C"), SmilesErrorKind::PsmilesStarNotTerminal);
    }

    #[test]
    fn test_ring_digit_on_a_star_is_refused() {
        assert_eq!(
            err_kind("*CC*1CC1"),
            SmilesErrorKind::PsmilesStarNotTerminal
        );
    }

    #[test]
    fn test_unit_without_exactly_two_stars_is_refused() {
        assert_eq!(err_kind("*CC"), SmilesErrorKind::PsmilesStarCount(1));
        assert_eq!(err_kind("CC"), SmilesErrorKind::PsmilesStarCount(0));
        assert_eq!(err_kind("*CC(*)C*"), SmilesErrorKind::PsmilesStarCount(3));
    }

    #[test]
    fn test_lone_star_is_dangling() {
        assert_eq!(err_kind("*"), SmilesErrorKind::DanglingDescriptor);
    }

    #[test]
    fn test_bracket_star_with_annotations_is_refused() {
        assert!(parse_psmiles("[*:1]CC[*:2]").is_err());
    }

    #[test]
    fn test_bigsmiles_descriptor_is_not_psmiles() {
        assert_eq!(
            err_kind("[$]CC[$]"),
            SmilesErrorKind::DescriptorInPlainSmiles
        );
    }

    #[test]
    fn test_errors_are_stamped_psmiles() {
        let err = parse_psmiles("*CC").unwrap_err();
        assert_eq!(err.notation, Notation::Psmiles);
        let err = parse_psmiles("*C*C").unwrap_err();
        assert_eq!(err.notation, Notation::Psmiles);
    }

    #[test]
    fn test_plain_smiles_star_is_still_a_wildcard_atom() {
        let ir = crate::line_notation::parser::parse_smiles("*CC*").unwrap();
        assert_eq!(atom_nodes(&ir).len(), 4);
        assert!(descriptors(&ir).is_empty());
    }
}
