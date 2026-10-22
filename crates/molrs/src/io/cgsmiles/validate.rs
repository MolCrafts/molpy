//! Post-parse validation of a `CGsmiles` IR.
//!
//! One walk, one rule: the squash operator `[!]` — the only `CGsmiles` syntax
//! that places one atom in two beads (R5.2 of
//! `.claude/specs/cgsmiles-01c-fragments.md` § Domain basis) — is refused
//! wherever it was written (R5.5). Detection is **structural** rather than
//! textual: the walk looks for a [`DescriptorKind::Shared`] descriptor in the
//! parsed IR, so a base-graph node, a coarse fragment body and an atomistic
//! fragment body are all covered by the same rule, and no stage re-lexes the
//! input looking for the glyph.
//!
//! A [`BondingDescriptor`](crate::io::smiles::BondingDescriptor) carries no
//! span of its own, so a refusal is spanned at the node that carries it: a
//! [`CgNode::span`](crate::io::cgsmiles::CgNode::span) for a coarse node, and
//! the definition's [`CgFragmentDef::span`] for an atomistic body, whose own
//! atom spans index the body rather than the whole string.

use crate::io::cgsmiles::{CgFragmentDef, CgGraph, CgSmilesIr, FragmentBody};
use crate::io::smiles::{
    AtomNode, BondingDescriptor, Chain, ChainElement, DescriptorKind, SmilesIr, Span,
};
use crate::io::smiles::{Notation, SmilesError, SmilesErrorKind};

/// Refuse a parsed `CGsmiles` IR that carries notation this version does not
/// model.
///
/// Walks every level of `ir` and every body of every fragment table. `input`
/// is the whole `CGsmiles` string, so the caret of any refusal lands in the
/// text the caller passed in.
///
/// # Errors
///
/// [`SmilesErrorKind::CgSquashUnsupported`], spanned at the node carrying the
/// `[!]`.
pub(crate) fn validate_ir(ir: &CgSmilesIr, input: &str) -> Result<(), SmilesError> {
    for level in &ir.levels {
        check_graph(level, input)?;
    }
    for table in &ir.fragments {
        for def in table.values() {
            check_body(def, input)?;
        }
    }
    Ok(())
}

/// Refuse a squash descriptor on any node of one coarse graph.
fn check_graph(graph: &CgGraph, input: &str) -> Result<(), SmilesError> {
    for node in &graph.nodes {
        if node.descriptors.iter().any(is_squash) {
            return Err(squash_error(node.span, input));
        }
    }
    Ok(())
}

/// Refuse a squash descriptor in one fragment body, whichever shape it has.
fn check_body(def: &CgFragmentDef, input: &str) -> Result<(), SmilesError> {
    match &def.body {
        FragmentBody::Graph(graph) => check_graph(graph, input),
        FragmentBody::Smiles(body) if smiles_squashes(body.ir()) => {
            Err(squash_error(def.span, input))
        }
        FragmentBody::Smiles(_) => Ok(()),
    }
}

/// The refusal itself, so every site spells the same kind and notation.
fn squash_error(span: Span, input: &str) -> SmilesError {
    SmilesError::new(
        SmilesErrorKind::CgSquashUnsupported,
        span,
        input,
        Notation::CgSmiles,
    )
}

/// True for the squash operator, the one descriptor kind this version refuses.
fn is_squash(descriptor: &BondingDescriptor) -> bool {
    descriptor.kind == DescriptorKind::Shared
}

/// True when any atom of an atomistic fragment body carries a squash
/// descriptor.
///
/// A production walker rather than the `cfg(test)` traversal in
/// `line_notation::fixtures`: this one runs in a shipped build.
fn smiles_squashes(body: &SmilesIr) -> bool {
    body.components.iter().any(chain_squashes)
}

/// True when a chain, or anything branching off it, carries a squash
/// descriptor. A ring closure carries no atom and so carries no descriptor.
fn chain_squashes(chain: &Chain) -> bool {
    atom_squashes(&chain.head)
        || chain.tail.iter().any(|element| match element {
            ChainElement::BondedAtom { atom, .. } => atom_squashes(atom),
            ChainElement::Branch { chain, .. } => chain_squashes(chain),
            ChainElement::RingClosure { .. } => false,
        })
}

/// True when one atom carries a squash descriptor.
fn atom_squashes(atom: &AtomNode) -> bool {
    atom.descriptors.iter().any(is_squash)
}

#[cfg(test)]
mod tests {
    use super::*;

    use std::collections::BTreeMap;

    use crate::io::cgsmiles::fixtures::fragment;
    use crate::io::cgsmiles::{CgFragmentDef, CgGraph, CgNode, CgSmilesIr, FragmentBody};
    use crate::io::smiles::{BondingDescriptor, DescriptorKind, Notation, SmilesErrorKind, Span};

    // R5.2: `[!]` is the only syntax placing one atom in two beads, and R5.5
    // lists it among the features molrs refuses. The walk is structural — one
    // rule over `DescriptorKind::Shared` — so it must fire on a base-graph
    // node, on a `Graph` body and on a `Smiles` body alike.
    //
    // `BondingDescriptor` carries no span of its own (01a), so the span a
    // diagnostic can name is the span of the node the descriptor binds to;
    // the fixtures below set that span to the bytes of the descriptor's own
    // text where the caret is asserted.

    // -- hand-built fixtures ------------------------------------------------

    /// An unlabelled descriptor of `kind`, with no bond order written.
    fn descriptor(kind: DescriptorKind) -> BondingDescriptor {
        BondingDescriptor {
            kind,
            label: String::new(),
            order: None,
        }
    }

    /// A coarse node named `name`, carrying `kinds`, spanning `span`.
    fn node(name: &str, kinds: &[DescriptorKind], span: Span) -> CgNode {
        CgNode {
            name: name.to_owned(),
            charge: None,
            annotations: Vec::new(),
            descriptors: kinds.iter().copied().map(descriptor).collect(),
            parent: None,
            span,
        }
    }

    /// A one-node graph.
    fn one_node(name: &str, kinds: &[DescriptorKind], span: Span) -> CgGraph {
        CgGraph {
            nodes: vec![node(name, kinds, span)],
            edges: Vec::new(),
        }
    }

    /// One fragment table, `name` → `body`.
    fn table(name: &str, body: FragmentBody, span: Span) -> BTreeMap<String, CgFragmentDef> {
        BTreeMap::from([(
            name.to_owned(),
            CgFragmentDef {
                name: name.to_owned(),
                body,
                span,
            },
        )])
    }

    // -- a `Shared` descriptor in an intermediate (graph) body --------------

    /// `{[#A]}.{#A=[!][#X]}.{#X=[$]C[$]}`, with the `[!]` at bytes 11..14 and
    /// the node it binds to spanning `[!][#X]`, 11..18.
    #[test]
    fn test_validate_ir_refuses_a_shared_descriptor_in_a_graph_body() {
        let text = "{[#A]}.{#A=[!][#X]}.{#X=[$]C[$]}";
        let body = one_node("X", &[DescriptorKind::Shared], Span::new(11, 18));
        // One table, its level not yet built: all this walker reads.
        let ir = CgSmilesIr {
            levels: vec![one_node("A", &[], Span::new(1, 5))],
            fragments: vec![table("A", FragmentBody::Graph(body), Span::new(8, 18))],
            pairs: vec![Vec::new()],
            span: Span::new(0, text.len()),
        };
        let err = validate_ir(&ir, text).expect_err("a squash descriptor must be refused");
        assert!(
            matches!(err.kind, SmilesErrorKind::CgSquashUnsupported),
            "kind was {:?}",
            err.kind
        );
        assert_eq!(err.span.start, 11, "the caret must point at the '[!]'");
        assert_eq!(err.input, text);
        assert_eq!(err.notation, Notation::CgSmiles);
    }

    // -- a `Shared` descriptor in a last-block (atomistic) body -------------

    /// The atomistic half of F9's squash fixture: `[!]CC[!]` as the body of
    /// `#A`. The body's own spans index the body, so only the kind and the
    /// stamped notation are asserted here.
    #[test]
    fn test_validate_ir_refuses_a_shared_descriptor_in_a_smiles_body() {
        let text = "{[#A]}.{#A=[!]CC[!]}";
        let body = fragment("[!]CC[!]");
        let ir = CgSmilesIr {
            levels: vec![one_node("A", &[], Span::new(1, 5))],
            fragments: vec![table("A", FragmentBody::Smiles(body), Span::new(8, 19))],
            pairs: vec![Vec::new()],
            span: Span::new(0, text.len()),
        };
        let err = validate_ir(&ir, text).expect_err("a squash descriptor must be refused");
        assert!(
            matches!(err.kind, SmilesErrorKind::CgSquashUnsupported),
            "kind was {:?}",
            err.kind
        );
        assert_eq!(err.input, text);
        assert_eq!(err.notation, Notation::CgSmiles);
    }

    // -- a `Shared` descriptor on a base-graph node -------------------------

    /// `{[#A][!]}`: the same rule covers level 0. Built by hand rather than
    /// through `CgSmilesIr::parse`, which must itself refuse this string once the
    /// walk is wired in.
    #[test]
    fn test_validate_ir_refuses_a_shared_descriptor_in_the_base_graph() {
        let text = "{[#A][!]}";
        let ir = CgSmilesIr {
            levels: vec![one_node("A", &[DescriptorKind::Shared], Span::new(1, 5))],
            fragments: Vec::new(),
            pairs: vec![Vec::new()],
            span: Span::new(0, text.len()),
        };
        let err = validate_ir(&ir, text).expect_err("a squash descriptor must be refused");
        assert!(
            matches!(err.kind, SmilesErrorKind::CgSquashUnsupported),
            "kind was {:?}",
            err.kind
        );
        assert_eq!(
            err.span,
            Span::new(1, 5),
            "the span is the node the '[!]' binds to"
        );
    }

    // -- the other three descriptor kinds are not squash --------------------

    #[test]
    fn test_validate_ir_accepts_a_symmetric_descriptor() {
        let text = "{[#A][$]}";
        let ir = CgSmilesIr {
            levels: vec![one_node("A", &[DescriptorKind::Symmetric], Span::new(1, 5))],
            fragments: Vec::new(),
            pairs: vec![Vec::new()],
            span: Span::new(0, text.len()),
        };
        assert_eq!(validate_ir(&ir, text), Ok(()));
    }
}
