//! Grammar-level validation helpers shared across the dialects.
//!
//! Validation that depends only on the shared AST vocabulary lives here — the
//! checks that read the same way whichever dialect wrote the string. Checks
//! specific to one dialect, such as element-symbol validity, live with that
//! dialect: for SMILES that is the sibling
//! SMILES element check of [`crate::io::smiles`].
//!
//! Two of these helpers run at different times, which is worth keeping
//! straight: [`validate_ring_closures`] is a post-parse pass over a finished
//! IR, while [`validate_descriptor`] runs during the parse, at the one site
//! that constructs a descriptor.

use std::collections::HashMap;

use crate::line_notation::ast::{BondKind, BondingDescriptor, Chain, ChainElement, SmilesIr, Span};
use crate::line_notation::error::{Notation, SmilesError, SmilesErrorKind};

/// Ensure every ring-closure digit is opened and closed exactly once.
///
/// A ring closure is the pair of matching digits that tells the notation two
/// non-adjacent atoms are bonded (`c1ccccc1` closes ring 1 between the first
/// and last carbon). The check applies equally to every dialect, because ring
/// closure is a construct of the shared grammar.
///
/// # Errors
///
/// Returns [`SmilesErrorKind::UnmatchedRingClosure`] carrying the ring number
/// of a digit that was opened and never closed; the span points at the
/// unmatched digit. Which unmatched digit is reported, when several are, is
/// unspecified. The error is stamped [`Notation::Smiles`]: ring closures are
/// validated on the SMILES-family post-parse path only.
pub(crate) fn validate_ring_closures(mol: &SmilesIr, input: &str) -> Result<(), SmilesError> {
    let mut open: HashMap<u16, Span> = HashMap::new();

    for component in &mol.components {
        collect_ring_closures(component, &mut open);
    }

    if let Some((&rnum, &span)) = open.iter().next() {
        return Err(SmilesError::new(
            SmilesErrorKind::UnmatchedRingClosure(rnum),
            span,
            input,
            Notation::Smiles,
        ));
    }

    Ok(())
}

/// Ensure a bonding descriptor is well formed, before it reaches the AST.
///
/// Two rules, both from the `CGsmiles` grammar:
///
/// * the label is ASCII alphanumeric or empty — `BigSMILES` allows positive
///   integers, `CGsmiles` widens that to alphanumerics and no further;
/// * the out-of-bracket bond order, when one was written, is one of
///   `Single | Double | Triple | Quadruple`. A descriptor annotates the order
///   of the bond its pairing will create, and aromatic (`c:[$]`), directional
///   (`/`, `\`), wildcard (`~`) and ring (`@`) are not orders a created bond
///   can take.
///
/// `span` covers the descriptor bracket and `input` is the whole parsed
/// string, so the returned error carries the usual caret context.
///
/// `notation` is the notation the error is stamped with. It is a parameter
/// because the check is shared and its callers are not: the SMILES-family
/// parser passes its dialect's own notation ([`Dialect::notation`]), the
/// `CGsmiles` parser passes [`Notation::CgSmiles`]. The notation is a fact
/// owned by the entry point, and passing it in is how this check learns it
/// without guessing.
///
/// [`Dialect::notation`]: crate::line_notation::Dialect::notation
///
/// # Errors
///
/// Returns [`SmilesErrorKind::InvalidDescriptorLabel`] for a non-alphanumeric
/// label, and [`SmilesErrorKind::InvalidDescriptorOrder`] for an order outside
/// the four allowed kinds.
pub(crate) fn validate_descriptor(
    desc: &BondingDescriptor,
    span: Span,
    input: &str,
    notation: Notation,
) -> Result<(), SmilesError> {
    if !desc.label.chars().all(|c| c.is_ascii_alphanumeric()) {
        return Err(SmilesError::new(
            SmilesErrorKind::InvalidDescriptorLabel(desc.label.clone()),
            span,
            input,
            notation,
        ));
    }

    if let Some(order) = desc.order
        && !matches!(
            order,
            BondKind::Single | BondKind::Double | BondKind::Triple | BondKind::Quadruple
        )
    {
        return Err(SmilesError::new(
            SmilesErrorKind::InvalidDescriptorOrder(order),
            span,
            input,
            notation,
        ));
    }

    Ok(())
}

fn collect_ring_closures(chain: &Chain, open: &mut HashMap<u16, Span>) {
    for elem in &chain.tail {
        match elem {
            ChainElement::RingClosure { rnum, span, .. } => {
                if open.remove(rnum).is_none() {
                    open.insert(*rnum, *span);
                }
            }
            ChainElement::Branch { chain, .. } => {
                collect_ring_closures(chain, open);
            }
            ChainElement::BondedAtom { .. } => {}
        }
    }
}

// ==========================================================================
// Tests
// ==========================================================================

#[cfg(test)]
mod tests {
    use super::*;
    use crate::line_notation::ast::{BondKind, BondingDescriptor, DescriptorKind};

    fn descriptor(label: &str, order: Option<BondKind>) -> BondingDescriptor {
        BondingDescriptor {
            kind: DescriptorKind::Symmetric,
            label: label.to_owned(),
            order,
        }
    }

    #[test]
    fn test_validate_descriptor_accepts_alphanumeric_label_with_double_order() {
        let desc = descriptor("a1", Some(BondKind::Double));
        assert!(validate_descriptor(&desc, Span::new(0, 3), "[$]", Notation::Smiles).is_ok());
    }

    #[test]
    fn test_validate_descriptor_rejects_non_alphanumeric_label() {
        let desc = descriptor("a-", None);
        let err = validate_descriptor(&desc, Span::new(0, 3), "[$]", Notation::Smiles).unwrap_err();
        assert!(matches!(
            err.kind,
            SmilesErrorKind::InvalidDescriptorLabel(_)
        ));
    }

    #[test]
    fn test_validate_descriptor_rejects_aromatic_order() {
        let desc = descriptor("", Some(BondKind::Aromatic));
        let err = validate_descriptor(&desc, Span::new(0, 3), "[$]", Notation::Smiles).unwrap_err();
        assert!(matches!(
            err.kind,
            SmilesErrorKind::InvalidDescriptorOrder(BondKind::Aromatic)
        ));
    }

    #[test]
    fn test_validate_descriptor_rejects_directional_order() {
        let desc = descriptor("", Some(BondKind::Up));
        let err = validate_descriptor(&desc, Span::new(0, 3), "[$]", Notation::Smiles).unwrap_err();
        assert!(matches!(
            err.kind,
            SmilesErrorKind::InvalidDescriptorOrder(BondKind::Up)
        ));
    }
}
