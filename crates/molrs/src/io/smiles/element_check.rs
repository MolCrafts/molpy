//! Post-parse semantic validation for SMILES ASTs.
//!
//! The parser enforces syntactic correctness (balanced brackets, valid grammar).
//! This module adds SMILES-specific semantic checks:
//!
//! * Ring closures must come in matched pairs (shared with SMARTS through the
//!   crate-private line-notation grammar).
//! * Element symbols must refer to real elements (SMILES-specific; SMARTS
//!   permits query primitives in their place).

use crate::io::smiles::{Notation, SmilesError, SmilesErrorKind};
use crate::line_notation::ast::*;
use crate::line_notation::is_element_symbol;
use crate::line_notation::validation::validate_ring_closures;

impl SmilesIr {
    /// Check this IR as plain SMILES: ring closures pair up, every symbol is
    /// an element, no node carries a bonding descriptor. `input` is the text
    /// the IR came from; it is quoted in an error.
    ///
    /// Returns `Ok(())` if valid, or the first validation error found.
    ///
    /// # Errors
    ///
    /// Returns [`SmilesErrorKind::UnmatchedRingClosure`] for an unpaired ring
    /// digit, [`SmilesErrorKind::InvalidElement`] for a symbol that is not an
    /// element, and [`SmilesErrorKind::DescriptorInPlainSmiles`] for a node
    /// carrying a bonding descriptor: this is the plain-SMILES validator, and a
    /// fragment body belongs to the fragment dialect.
    pub fn validate(&self, input: &str) -> Result<(), SmilesError> {
        validate_ring_closures(self, input)?;
        validate_elements(self, input)?;
        Ok(())
    }
}

// ---------------------------------------------------------------------------
// Element validation (SMILES-specific)
// ---------------------------------------------------------------------------

/// Validate that all element symbols refer to real elements.
///
/// [`SmilesIr::parse`] refuses an unknown
/// symbol itself, by this same lookup, so on a parsed IR this pass has
/// nothing left to find; it stands for the IRs nobody parsed — built by hand
/// or edited after parsing.
fn validate_elements(mol: &SmilesIr, input: &str) -> Result<(), SmilesError> {
    for component in &mol.components {
        validate_chain_elements(component, input)?;
    }
    Ok(())
}

fn validate_chain_elements(chain: &Chain, input: &str) -> Result<(), SmilesError> {
    validate_atom_element(&chain.head, input)?;
    for elem in &chain.tail {
        match elem {
            ChainElement::BondedAtom { atom, .. } => {
                validate_atom_element(atom, input)?;
            }
            ChainElement::Branch { chain, .. } => {
                validate_chain_elements(chain, input)?;
            }
            ChainElement::RingClosure { .. } => {}
        }
    }
    Ok(())
}

fn validate_atom_element(atom: &AtomNode, input: &str) -> Result<(), SmilesError> {
    // Plain SMILES has no bonding-descriptor notation, so a node carrying one
    // reached this validator through the fragment dialect (or a hand-built
    // IR). Refusing here keeps `SmilesIr::validate` symmetric with
    // `to_atomistic`: neither plain-path stage ever drops a descriptor.
    if !atom.descriptors.is_empty() {
        return Err(SmilesError::new(
            SmilesErrorKind::DescriptorInPlainSmiles,
            atom.span,
            input,
            Notation::Smiles,
        ));
    }

    match &atom.spec {
        AtomSpec::Organic { symbol, .. } => {
            validate_symbol(symbol, atom.span, input)?;
        }
        AtomSpec::Bracket { symbol, .. } => match symbol {
            BracketSymbol::Element { symbol, .. } => {
                validate_symbol(symbol, atom.span, input)?;
            }
            BracketSymbol::Any | BracketSymbol::Aliphatic | BracketSymbol::Aromatic => {}
        },
        AtomSpec::Wildcard => {}
        AtomSpec::Query(_) => {
            // SMARTS query atoms may contain primitives — skip deep validation
            // here as it is handled by the SMARTS-specific validator.
        }
    }
    Ok(())
}

fn validate_symbol(symbol: &str, span: Span, input: &str) -> Result<(), SmilesError> {
    if !is_element_symbol(symbol) {
        return Err(SmilesError::new(
            SmilesErrorKind::InvalidElement(symbol.to_owned()),
            span,
            input,
            Notation::Smiles,
        ));
    }
    Ok(())
}

// ==========================================================================
// Tests
// ==========================================================================

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_valid_smiles() {
        let mol = SmilesIr::parse("C1CCCCC1").unwrap();
        assert!(mol.validate("C1CCCCC1").is_ok());
    }

    /// Hand-built IR for `CCCC1` — a chain of four carbons whose last element
    /// is a ring digit that never closes.
    ///
    /// The parser refuses an unmatched ring closure itself, so this unit can
    /// only be reached with an IR built directly.
    fn unmatched_ring_ir() -> SmilesIr {
        fn carbon(start: usize) -> AtomNode {
            AtomNode {
                spec: AtomSpec::Organic {
                    symbol: "C".to_owned(),
                    aromatic: false,
                },
                span: Span::new(start, start + 1),
                descriptors: Vec::new(),
            }
        }

        SmilesIr {
            components: vec![Chain {
                head: carbon(0),
                tail: vec![
                    ChainElement::BondedAtom {
                        bond: None,
                        atom: carbon(1),
                    },
                    ChainElement::BondedAtom {
                        bond: None,
                        atom: carbon(2),
                    },
                    ChainElement::BondedAtom {
                        bond: None,
                        atom: carbon(3),
                    },
                    ChainElement::RingClosure {
                        bond: None,
                        rnum: 1,
                        span: Span::new(4, 5),
                    },
                ],
            }],
            span: Span::new(0, 5),
        }
    }

    #[test]
    fn test_unmatched_ring_closure() {
        let mol = unmatched_ring_ir();
        let err = mol.validate("CCCC1").unwrap_err();
        assert!(matches!(err.kind, SmilesErrorKind::UnmatchedRingClosure(1)));
    }

    /// Hand-built IR for `[Xx]` — a bracket atom whose symbol is not an
    /// element.
    ///
    /// `parse_smiles` refuses this string itself, so this unit is reached
    /// only with an IR built directly (as a hand-built IR or a future
    /// notation could still carry one).
    fn unknown_element_ir() -> SmilesIr {
        SmilesIr {
            components: vec![Chain {
                head: AtomNode {
                    spec: AtomSpec::Bracket {
                        isotope: None,
                        symbol: BracketSymbol::Element {
                            symbol: "Xx".to_owned(),
                            aromatic: false,
                        },
                        chirality: None,
                        hcount: None,
                        charge: None,
                        atom_class: None,
                    },
                    span: Span::new(0, 4),
                    descriptors: Vec::new(),
                },
                tail: Vec::new(),
            }],
            span: Span::new(0, 4),
        }
    }

    /// The payload is the symbol exactly as written, which is the same kind
    /// and payload `parse_smiles` reports for the same rule
    /// (`parser.rs::test_unknown_bracket_element_is_refused_by_parse_smiles`).
    #[test]
    fn test_unknown_bracket_element_is_an_invalid_element() {
        let mol = unknown_element_ir();
        let err = mol.validate("[Xx]").unwrap_err();
        match &err.kind {
            SmilesErrorKind::InvalidElement(symbol) => assert_eq!(symbol, "Xx"),
            other => panic!("expected InvalidElement, got {other:?}"),
        }
    }

    #[test]
    fn test_valid_elements() {
        let mol = SmilesIr::parse("[Fe+2]").unwrap();
        assert!(mol.validate("[Fe+2]").is_ok());
    }

    #[test]
    fn test_valid_aromatic_element() {
        let mol = SmilesIr::parse("c1ccccc1").unwrap();
        assert!(mol.validate("c1ccccc1").is_ok());
    }

    #[test]
    fn test_multiple_ring_closures() {
        let mol = SmilesIr::parse("c1ccc2ccccc2c1").unwrap();
        assert!(mol.validate("c1ccc2ccccc2c1").is_ok());
    }

    #[test]
    fn test_disconnected_valid() {
        let mol = SmilesIr::parse("[Na+].[Cl-]").unwrap();
        assert!(mol.validate("[Na+].[Cl-]").is_ok());
    }

    #[test]
    fn test_descriptor_bearing_ir_is_rejected_by_the_plain_validator() {
        let mut mol = SmilesIr::parse("CCO").unwrap();
        mol.components[0].head.descriptors.push(BondingDescriptor {
            kind: DescriptorKind::Symmetric,
            label: String::new(),
            order: None,
        });
        let err = mol.validate("CCO").unwrap_err();
        assert!(matches!(err.kind, SmilesErrorKind::DescriptorInPlainSmiles));
    }
}
