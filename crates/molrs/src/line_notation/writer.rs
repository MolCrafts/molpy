//! Write [`SmilesIr`] back to SMILES, SMARTS or fragment-body strings.
//!
//! One emitter serves all three dialects, branching on the crate-internal
//! `Dialect`: each entry point refuses the constructs its own dialect cannot
//! spell, instead of emitting text the matching parser would reject.
//!
//! Pure syntax: no chemical policy. Graph → IR lives in [`SmilesIr::from_atomistic`](crate::io::smiles::SmilesIr::from_atomistic).

use crate::line_notation::Dialect;
use crate::line_notation::ast::*;
use crate::line_notation::error::{SmilesError, SmilesErrorKind};

/// Write a plain SMILES string from the IR.
///
/// Strict: the constructs plain SMILES cannot spell — SMARTS query atoms,
/// SMARTS bond operators, and fragment-dialect bonding descriptors — are
/// refused rather than emitted as text
/// [`parse_smiles`](crate::line_notation::parser::parse_smiles) would then reject.
///
/// # Errors
///
/// Returns [`SmilesErrorKind::Emit`] for an IR with no components,
/// [`SmilesErrorKind::InvalidQueryPrimitive`] for a SMARTS query atom
/// ([`AtomSpec::Query`]) or a SMARTS bond query (`!`, `&`, `,`), and
/// [`SmilesErrorKind::DescriptorInPlainSmiles`] for a node carrying a bonding
/// descriptor — write that IR with [`fragment_smiles_text`] instead.
#[cfg(test)]
pub(crate) fn smiles_text(ir: &SmilesIr) -> Result<String, SmilesError> {
    emit_ir(ir, Dialect::Smiles)
}

/// Write a SMARTS string from the IR.
///
/// SMILES is a subset of SMARTS, so every IR [`smiles_text`] accepts is
/// writable here too, plus query atoms and the bond operators `!`, `&` and
/// `,`.
///
/// # Errors
///
/// Returns [`SmilesErrorKind::Emit`] for an IR with no components, and
/// [`SmilesErrorKind::DescriptorInPlainSmiles`] for a node carrying a bonding
/// descriptor: descriptors are fragment notation, and SMARTS has no spelling
/// for them either.
pub fn smarts_text(ir: &SmilesIr) -> Result<String, SmilesError> {
    emit_ir(ir, Dialect::Smarts)
}

/// Write a SMILES **fragment** body: SMILES plus `CGsmiles` / `BigSMILES`
/// bonding descriptors.
///
/// The fragment-dialect sibling of [`smiles_text`], which refuses a
/// descriptor-bearing IR rather than emit text its own parser rejects; this
/// function is where such an IR is meant to go.
///
/// Descriptors are written in the canonical **trailing** form: the atom, then
/// each of its descriptors preceded by the bond order it carries, as in
/// `C=[$]CC`. An explicit single order is written out (`CC-[$]`): unlike a
/// chain bond, where `-` is the omitted default, a descriptor with no bond
/// symbol next to it is the distinct `order: None` state, which re-reads
/// differently. `None` therefore emits no symbol at all, and the two states
/// survive a write-then-parse round trip.
///
/// The input *text* is not preserved — the IR is. The leading form `[$]=CCC`
/// annotates the head carbon with a double-bond order, so it is written as
/// `C=[$]CC`, which parses back to the same IR; a second parse-and-write pass
/// is then stable.
///
/// # Errors
///
/// Returns [`SmilesErrorKind::Emit`] for an empty IR,
/// [`SmilesErrorKind::InvalidDescriptorOrder`] for a descriptor order outside
/// `Single`, `Double`, `Triple` and `Quadruple` (no parser produces one, but a
/// hand-built IR can carry it), and
/// [`SmilesErrorKind::InvalidQueryPrimitive`] for SMARTS query atoms or bond
/// queries, which are not fragment notation.
///
/// It never returns [`SmilesErrorKind::DescriptorInPlainSmiles`]: that is the
/// plain writers' refusal of the input this one exists to accept.
pub(crate) fn fragment_smiles_text(ir: &SmilesIr) -> Result<String, SmilesError> {
    emit_ir(ir, Dialect::FragmentSmiles)
}

fn emit_ir(ir: &SmilesIr, dialect: Dialect) -> Result<String, SmilesError> {
    if ir.components.is_empty() {
        return Err(SmilesError::new(
            SmilesErrorKind::Emit("empty SmilesIr".into()),
            ir.span,
            "",
            dialect.notation(),
        ));
    }
    let mut out = String::new();
    for (i, chain) in ir.components.iter().enumerate() {
        if i > 0 {
            out.push('.');
        }
        emit_chain(&mut out, chain, dialect, /*in_branch*/ false)?;
    }
    Ok(out)
}

fn emit_chain(
    out: &mut String,
    chain: &Chain,
    dialect: Dialect,
    _in_branch: bool,
) -> Result<(), SmilesError> {
    emit_atom(out, &chain.head, dialect)?;
    for elem in &chain.tail {
        match elem {
            ChainElement::BondedAtom { bond, atom } => {
                emit_bond(
                    out,
                    bond.as_ref(),
                    dialect,
                    /*omit_default_single*/ true,
                )?;
                emit_atom(out, atom, dialect)?;
            }
            ChainElement::Branch { bond, chain, .. } => {
                out.push('(');
                emit_bond(out, bond.as_ref(), dialect, true)?;
                emit_chain(out, chain, dialect, true)?;
                out.push(')');
            }
            ChainElement::RingClosure { bond, rnum, .. } => {
                emit_bond(out, bond.as_ref(), dialect, true)?;
                emit_rnum(out, *rnum);
            }
        }
    }
    Ok(())
}

fn emit_rnum(out: &mut String, rnum: u16) {
    if rnum < 10 {
        out.push(char::from(b'0' + rnum as u8));
    } else {
        out.push('%');
        out.push_str(&rnum.to_string());
    }
}

fn emit_atom(out: &mut String, node: &AtomNode, dialect: Dialect) -> Result<(), SmilesError> {
    match &node.spec {
        AtomSpec::Organic { symbol, aromatic } => {
            if *aromatic {
                for c in symbol.chars() {
                    out.push(c.to_ascii_lowercase());
                }
            } else {
                out.push_str(symbol);
            }
            Ok(())
        }
        AtomSpec::Wildcard => {
            out.push('*');
            Ok(())
        }
        AtomSpec::Bracket {
            isotope,
            symbol,
            chirality,
            hcount,
            charge,
            atom_class,
        } => {
            out.push('[');
            if let Some(iso) = isotope {
                out.push_str(&iso.to_string());
            }
            emit_bracket_symbol(out, symbol);
            if let Some(ch) = chirality {
                match ch {
                    Chirality::CounterClockwise => out.push('@'),
                    Chirality::Clockwise => out.push_str("@@"),
                }
            }
            if let Some(h) = hcount {
                out.push('H');
                if *h != 1 {
                    out.push_str(&h.to_string());
                }
            }
            if let Some(c) = charge {
                emit_charge(out, *c);
            }
            if let Some(cls) = atom_class {
                out.push(':');
                out.push_str(&cls.to_string());
            }
            out.push(']');
            Ok(())
        }
        AtomSpec::Query(q) => {
            if dialect != Dialect::Smarts {
                return Err(SmilesError::new(
                    SmilesErrorKind::InvalidQueryPrimitive(format!(
                        "SMARTS query atoms cannot be written as {}",
                        dialect_name(dialect)
                    )),
                    node.span,
                    "",
                    dialect.notation(),
                ));
            }
            out.push('[');
            emit_atom_query(out, q)?;
            out.push(']');
            Ok(())
        }
    }?;

    emit_descriptors(out, node, dialect)
}

/// Write `node`'s bonding descriptors in the canonical trailing form.
///
/// `C=[$]`, never `[$]=C`: each descriptor follows the atom it is anchored on,
/// preceded by the bond order it carries. An order of `Single` is written `-`
/// rather than omitted — a descriptor with no bond symbol next to it parses
/// back as `order: None`, which is a different IR, not the same one spelled
/// shorter (the omitted default applies to chain bonds only).
///
/// A node with no descriptors writes nothing and is accepted in every dialect,
/// which is what keeps plain output byte-identical to what it was before the
/// field existed.
///
/// # Errors
///
/// Returns [`SmilesErrorKind::DescriptorInPlainSmiles`] in any dialect but
/// `FragmentSmiles`, and [`SmilesErrorKind::InvalidDescriptorOrder`] for an
/// order outside `Single`, `Double`, `Triple` and `Quadruple`.
fn emit_descriptors(
    out: &mut String,
    node: &AtomNode,
    dialect: Dialect,
) -> Result<(), SmilesError> {
    if node.descriptors.is_empty() {
        return Ok(());
    }
    if dialect != Dialect::FragmentSmiles {
        return Err(SmilesError::new(
            SmilesErrorKind::DescriptorInPlainSmiles,
            node.span,
            "",
            dialect.notation(),
        ));
    }
    for desc in &node.descriptors {
        match desc.order {
            // No bond symbol was written next to the bracket.
            None => {}
            Some(
                k @ (BondKind::Single | BondKind::Double | BondKind::Triple | BondKind::Quadruple),
            ) => emit_bond_kind(out, k, /*omit_default_single*/ false),
            // A hand-built IR can carry an order no parser produces; emitting
            // it would write text this dialect's own parser rejects.
            Some(k) => {
                return Err(SmilesError::new(
                    SmilesErrorKind::InvalidDescriptorOrder(k),
                    node.span,
                    "",
                    dialect.notation(),
                ));
            }
        }
        out.push('[');
        out.push(match desc.kind {
            DescriptorKind::Symmetric => '$',
            DescriptorKind::Left => '<',
            DescriptorKind::Right => '>',
            DescriptorKind::Shared => '!',
        });
        out.push_str(&desc.label);
        out.push(']');
    }
    Ok(())
}

/// Write the symbol slot of a bracket atom: `[<here>H2+]`.
///
/// Aromaticity is notation here, not a property: a declared-aromatic element
/// is written lowercase, which is the only thing that distinguishes `[cH]`
/// from `[CH]`.
fn emit_bracket_symbol(out: &mut String, symbol: &BracketSymbol) {
    match symbol {
        BracketSymbol::Element { symbol, aromatic } => {
            if *aromatic {
                for c in symbol.chars() {
                    out.push(c.to_ascii_lowercase());
                }
            } else {
                out.push_str(symbol);
            }
        }
        BracketSymbol::Any => out.push('*'),
        BracketSymbol::Aliphatic => out.push('A'),
        BracketSymbol::Aromatic => out.push('a'),
    }
}

fn emit_charge(out: &mut String, c: i8) {
    if c == 0 {
        return;
    }
    if c > 0 {
        out.push('+');
        if c != 1 {
            out.push_str(&c.to_string());
        }
    } else {
        out.push('-');
        let a = (-c) as u8;
        if a != 1 {
            out.push_str(&a.to_string());
        }
    }
}

fn emit_atom_query(out: &mut String, q: &AtomQuery) -> Result<(), SmilesError> {
    match q {
        AtomQuery::Primitive(p) => emit_primitive(out, p),
        AtomQuery::Not(inner) => {
            out.push('!');
            emit_atom_query(out, inner)
        }
        AtomQuery::And(parts) => {
            // `&` binds tighter than `,`, so an AND over an OR is spelled with
            // the low-precedence `;`. The separator is always written:
            // juxtaposing terms can fuse two into one token (`C` then `a`
            // reads back as calcium).
            let sep = if parts
                .iter()
                .any(|p| matches!(p, AtomQuery::Or(_) | AtomQuery::LowAnd(_)))
            {
                ';'
            } else {
                '&'
            };
            for (i, p) in parts.iter().enumerate() {
                if i > 0 {
                    out.push(sep);
                }
                emit_atom_query(out, p)?;
            }
            Ok(())
        }
        AtomQuery::Or(parts) => {
            for (i, p) in parts.iter().enumerate() {
                if i > 0 {
                    out.push(',');
                }
                emit_atom_query(out, p)?;
            }
            Ok(())
        }
        AtomQuery::LowAnd(parts) => {
            for (i, p) in parts.iter().enumerate() {
                if i > 0 {
                    out.push(';');
                }
                emit_atom_query(out, p)?;
            }
            Ok(())
        }
    }
}

fn emit_primitive(out: &mut String, p: &AtomPrimitive) -> Result<(), SmilesError> {
    match p {
        AtomPrimitive::Element { symbol, aromatic } => {
            if *aromatic {
                for c in symbol.chars() {
                    out.push(c.to_ascii_lowercase());
                }
            } else {
                out.push_str(symbol);
            }
            Ok(())
        }
        AtomPrimitive::AtomicNumber(z) => {
            out.push('#');
            out.push_str(&z.to_string());
            Ok(())
        }
        AtomPrimitive::Wildcard => {
            out.push('*');
            Ok(())
        }
        AtomPrimitive::Aliphatic => {
            out.push('A');
            Ok(())
        }
        AtomPrimitive::Aromatic => {
            out.push('a');
            Ok(())
        }
        AtomPrimitive::Degree(n) => {
            out.push('D');
            out.push_str(&n.to_string());
            Ok(())
        }
        AtomPrimitive::TotalConnections(n) => {
            out.push('X');
            out.push_str(&n.to_string());
            Ok(())
        }
        AtomPrimitive::HCount(n) => {
            out.push('H');
            if *n != 1 {
                out.push_str(&n.to_string());
            }
            Ok(())
        }
        AtomPrimitive::ImplicitH(n) => {
            out.push('h');
            if *n != 1 {
                out.push_str(&n.to_string());
            }
            Ok(())
        }
        AtomPrimitive::RingMembership(None) => {
            out.push('R');
            Ok(())
        }
        AtomPrimitive::RingMembership(Some(n)) => {
            out.push('R');
            out.push_str(&n.to_string());
            Ok(())
        }
        AtomPrimitive::RingSize(n) => {
            out.push('r');
            out.push_str(&n.to_string());
            Ok(())
        }
        AtomPrimitive::RingSizeRange { lo, hi } => {
            out.push_str("r{");
            if *lo > 0 {
                out.push_str(&lo.to_string());
            }
            out.push('-');
            if let Some(h) = hi {
                out.push_str(&h.to_string());
            }
            out.push('}');
            Ok(())
        }
        AtomPrimitive::RingBondCount(n) => {
            out.push('x');
            out.push_str(&n.to_string());
            Ok(())
        }
        AtomPrimitive::Hybridization(n) => {
            out.push('^');
            out.push_str(&n.to_string());
            Ok(())
        }
        AtomPrimitive::ContextLabel(label) => {
            out.push('%');
            out.push_str(label);
            Ok(())
        }
        AtomPrimitive::Valence(n) => {
            out.push('v');
            out.push_str(&n.to_string());
            Ok(())
        }
        AtomPrimitive::Charge(c) => {
            emit_charge(out, *c);
            Ok(())
        }
        AtomPrimitive::Isotope(iso) => {
            out.push_str(&iso.to_string());
            Ok(())
        }
        AtomPrimitive::AtomClass(cls) => {
            out.push(':');
            out.push_str(&cls.to_string());
            Ok(())
        }
        AtomPrimitive::Chirality(ch) => {
            match ch {
                Chirality::CounterClockwise => out.push('@'),
                Chirality::Clockwise => out.push_str("@@"),
            }
            Ok(())
        }
        AtomPrimitive::Recursive(ir) => {
            out.push_str("$(");
            // recursive body is a full SMILES/SMARTS molecule fragment
            let body = smarts_text(ir)?;
            out.push_str(&body);
            out.push(')');
            Ok(())
        }
    }
}

fn emit_bond(
    out: &mut String,
    bond: Option<&BondQuery>,
    dialect: Dialect,
    omit_default_single: bool,
) -> Result<(), SmilesError> {
    let Some(q) = bond else {
        return Ok(());
    };
    match q {
        BondQuery::Kind(k) => {
            emit_bond_kind(out, *k, omit_default_single);
            Ok(())
        }
        BondQuery::Not(inner) => {
            if dialect != Dialect::Smarts {
                return Err(bond_query_err(dialect));
            }
            out.push('!');
            emit_bond(out, Some(inner), dialect, false)
        }
        BondQuery::And(parts) | BondQuery::Or(parts) => {
            if dialect != Dialect::Smarts {
                return Err(bond_query_err(dialect));
            }
            // `&` binds tighter than `,`: an AND over an OR is spelled with
            // the low-precedence `;`.
            let sep = match q {
                BondQuery::And(parts) if parts.iter().any(|p| matches!(p, BondQuery::Or(_))) => ';',
                BondQuery::And(_) => '&',
                _ => ',',
            };
            for (i, p) in parts.iter().enumerate() {
                if i > 0 {
                    out.push(sep);
                }
                emit_bond(out, Some(p), dialect, false)?;
            }
            Ok(())
        }
    }
}

/// The error a SMARTS bond query earns in a dialect that cannot spell it.
fn bond_query_err(dialect: Dialect) -> SmilesError {
    SmilesError::new(
        SmilesErrorKind::InvalidQueryPrimitive(format!(
            "SMARTS bond query cannot be written as {}",
            dialect_name(dialect)
        )),
        Span::new(0, 0),
        "",
        dialect.notation(),
    )
}

/// How a dialect names itself in a rejection message.
fn dialect_name(dialect: Dialect) -> &'static str {
    match dialect {
        Dialect::Smiles => "SMILES",
        Dialect::Smarts => "SMARTS",
        Dialect::FragmentSmiles => "a SMILES fragment",
    }
}

fn emit_bond_kind(out: &mut String, k: BondKind, omit_default_single: bool) {
    match k {
        BondKind::Single if omit_default_single => {}
        BondKind::Single => out.push('-'),
        BondKind::Double => out.push('='),
        BondKind::Triple => out.push('#'),
        BondKind::Quadruple => out.push('$'),
        BondKind::Aromatic => out.push(':'),
        BondKind::Up => out.push('/'),
        BondKind::Down => out.push('\\'),
        BondKind::Any => out.push('~'),
        BondKind::Ring => out.push('@'),
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::line_notation::fixtures::descriptors;
    use crate::line_notation::parser::{parse_smarts, parse_smiles};

    #[test]
    fn smiles_text_ethanol_stable() {
        let ir = parse_smiles("CCO").unwrap();
        let s1 = smiles_text(&ir).unwrap();
        let ir2 = parse_smiles(&s1).unwrap();
        let s2 = smiles_text(&ir2).unwrap();
        assert_eq!(s1, s2);
        assert!(!s1.is_empty());
    }

    #[test]
    fn smiles_text_acetic_and_benzene() {
        for src in ["C(=O)O", "c1ccccc1", "[NH4+]", "CCO.O"] {
            let ir = parse_smiles(src).unwrap();
            let s = smiles_text(&ir).unwrap();
            let ir2 = parse_smiles(&s).unwrap();
            assert_eq!(smiles_text(&ir2).unwrap(), s, "src={src} wrote={s}");
        }
    }

    #[test]
    fn smiles_text_rejects_query() {
        // Explicit OR query cannot be concrete SMILES.
        let ir = parse_smarts("[C,N]").unwrap();
        assert!(smiles_text(&ir).is_err(), "wrote {:?}", smiles_text(&ir));
    }

    #[test]
    fn smarts_text_query_and_recursive() {
        for src in ["[#6;D3]", "[C;$(C=O)]"] {
            let ir = parse_smarts(src).unwrap();
            let s = smarts_text(&ir).unwrap();
            let ir2 = parse_smarts(&s).unwrap();
            let _ = smarts_text(&ir2).unwrap();
        }
    }

    // -- fragment dialect: bonding descriptors ------------------------------

    fn fragment(input: &str) -> SmilesIr {
        crate::line_notation::parser::parse_fragment_smiles(input)
            .unwrap_or_else(|e| panic!("parse_fragment_smiles({input:?}) failed: {e}"))
    }

    #[test]
    fn write_fragment_smiles_round_trips_descriptors() {
        // The text is not preserved, the IR is: re-parsing the written string
        // must give back the same descriptor kinds, labels and orders.
        let ir = fragment("[$]COC[$]");
        let written = fragment_smiles_text(&ir).unwrap();
        let reparsed = fragment(&written);
        assert_eq!(descriptors(&reparsed), descriptors(&ir));
    }

    #[test]
    fn write_fragment_smiles_is_idempotent() {
        let s1 = fragment_smiles_text(&fragment("[$]COC[$]")).unwrap();
        let s2 = fragment_smiles_text(&fragment(&s1)).unwrap();
        assert_eq!(s2, s1);
    }

    #[test]
    fn write_fragment_smiles_emits_the_canonical_trailing_form() {
        // `[$]=CCC` is a double-bond-annotated `$` on the first carbon over a
        // single-bonded C-C-C chain (R4.4); the canonical form writes the
        // descriptor after its atom, so the leading input form is not kept.
        assert_eq!(
            fragment_smiles_text(&fragment("[$]=CCC")).unwrap(),
            "C=[$]CC"
        );
    }

    #[test]
    fn write_fragment_smiles_trailing_form_is_stable() {
        let s1 = fragment_smiles_text(&fragment("[$]=CCC")).unwrap();
        let s2 = fragment_smiles_text(&fragment(&s1)).unwrap();
        assert_eq!(s2, s1);
    }

    #[test]
    fn write_fragment_smiles_keeps_input_already_in_trailing_form() {
        assert_eq!(
            fragment_smiles_text(&fragment("C[$]=CC")).unwrap(),
            "C[$]=CC"
        );
    }

    #[test]
    fn smiles_text_rejects_descriptors() {
        // Plain SMILES has no descriptor notation, so writing one would emit
        // text its own parser refuses.
        let err = smiles_text(&fragment("[$]COC[$]")).unwrap_err();
        assert!(matches!(err.kind, SmilesErrorKind::DescriptorInPlainSmiles));
    }

    #[test]
    fn smarts_text_rejects_descriptors() {
        let err = smarts_text(&fragment("[$]COC[$]")).unwrap_err();
        assert!(matches!(err.kind, SmilesErrorKind::DescriptorInPlainSmiles));
    }

    #[test]
    fn write_fragment_smiles_rejects_query_atoms() {
        // The fragment dialect is SMILES plus descriptors — not SMARTS: a
        // query atom must not leak into a fragment string.
        let ir = parse_smarts("[!C]").unwrap();
        assert!(
            fragment_smiles_text(&ir).is_err(),
            "wrote {:?}",
            fragment_smiles_text(&ir)
        );
    }

    #[test]
    fn write_fragment_smiles_rejects_bond_queries() {
        let ir = parse_smarts("C!=C").unwrap();
        assert!(
            fragment_smiles_text(&ir).is_err(),
            "wrote {:?}",
            fragment_smiles_text(&ir)
        );
    }

    #[test]
    fn write_fragment_smiles_rejects_an_illegal_descriptor_order() {
        // `BondingDescriptor` has public fields, so an aromatic order — which
        // no parser produces (R4.5) — can reach the writer; it must refuse
        // rather than emit `C:[$]`, text its own parser rejects.
        let mut ir = fragment("C");
        ir.components[0].head.descriptors.push(BondingDescriptor {
            kind: DescriptorKind::Symmetric,
            label: String::new(),
            order: Some(BondKind::Aromatic),
        });
        let err = fragment_smiles_text(&ir).unwrap_err();
        assert!(matches!(
            err.kind,
            SmilesErrorKind::InvalidDescriptorOrder(BondKind::Aromatic)
        ));
    }

    #[test]
    fn write_fragment_smiles_keeps_an_explicit_single_order() {
        // `-` before a descriptor is an *explicit* single order, which 01d's
        // pairing rule distinguishes from "no symbol at all"; dropping it on
        // write silently rewrites the IR into `order: None`.
        let ir = fragment("CC-[$]");
        assert_eq!(
            descriptors(&ir),
            vec![BondingDescriptor {
                kind: DescriptorKind::Symmetric,
                label: String::new(),
                order: Some(BondKind::Single),
            }],
        );
        let written = fragment_smiles_text(&ir).unwrap();
        assert_eq!(written, "CC-[$]");
        assert_eq!(descriptors(&fragment(&written)), descriptors(&ir));
    }

    #[test]
    fn write_fragment_smiles_emits_the_shared_glyph() {
        // `[!]` is the shared descriptor; like every other kind it is written
        // in the canonical trailing form, after its anchor atom.
        assert_eq!(fragment_smiles_text(&fragment("[!]C")).unwrap(), "C[!]");
    }
}
