//! The line notations' shared grammar: SMILES, SMARTS, the SMILES fragment
//! body and PSMILES (see [`Dialect`]) — one abstract syntax tree, one byte
//! scanner, one recursive-descent parser, one writer, one error type.
//!
//! Crate-private. The SMILES format's public face is [`crate::io::smiles`]
//! (the IR, its error, the text doors); SMARTS is [`crate::perceive::smarts`]
//! (parse, compile, match, generate); both build on this module, so neither
//! depends on the other.
//!
//! A fourth notation borrows from here without being a `Dialect`: the
//! `CGsmiles` coarse-graph parser ([`crate::io::cgsmiles`]) runs on the same
//! scanner, mints the same `Span`s and reuses the bonding-descriptor check,
//! but it has its own grammar and its own AST, because its token vocabulary
//! collides with SMARTS (`[#NAME]` against the atomic-number primitive `[#6]`,
//! `$` against both a bond order and a descriptor) in ways a shared,
//! mode-switching parser would silently mis-read.

use crate::core::Element;
use error::Notation;

pub(crate) mod ast;
pub(crate) mod error;
#[cfg(test)]
pub(crate) mod fixtures;
pub(crate) mod parser;
pub(crate) mod psmiles;
pub(crate) mod scanner;
pub(crate) mod validation;
pub(crate) mod writer;

/// Line-notation helpers over the schema keys
/// [`IS_AROMATIC`](crate::core::keys::IS_AROMATIC),
/// [`H_COUNT`](crate::core::keys::H_COUNT),
/// [`ISOTOPE`](crate::core::keys::ISOTOPE) and
/// [`STEREO`](crate::core::keys::STEREO), shared by the SMILES builder, the
/// SMILES writer, the SMARTS matcher and the CGsmiles expansion.
pub(crate) mod atom_keys {
    pub(crate) use crate::core::keys::{H_COUNT, IS_AROMATIC, ISOTOPE, STEREO};

    /// The atomic number of the wildcard atom `*`, which names no element.
    pub(crate) const WILDCARD_Z: u8 = 0;

    /// Whether `atom` is marked aromatic: its [`IS_AROMATIC`] is `true`.
    /// Absent means "not declared aromatic"; the schema declares the key
    /// `bool`, so no other value can be stored under it.
    pub(crate) fn is_aromatic(atom: &crate::core::Props) -> bool {
        matches!(
            atom.get(IS_AROMATIC),
            Some(crate::core::PropValue::Bool(true))
        )
    }
}

/// The element symbol a SMILES atom symbol denotes.
///
/// SMILES writes aromatic atoms in lowercase (`c`, `n`, `se`); that is
/// notation, not an element symbol. [`Element::by_symbol`] is looked up on
/// the canonical capitalisation, so both validation and graph construction
/// normalise through here.
pub(crate) fn canonical_element_symbol(symbol: &str) -> String {
    let mut chars = symbol.chars();
    match chars.next() {
        None => String::new(),
        Some(first) => first.to_ascii_uppercase().to_string() + chars.as_str(),
    }
}

/// Whether `symbol`, as written in a SMILES atom, names a real element.
///
/// The lookup is on the canonical capitalisation, so the aromatic lowercase
/// `se` is the element `Se` and `[Xx]` is nothing at all. Both stages that
/// decide this question ask here — the parser when it reads a bracket atom,
/// and the SMILES element check when it re-checks an IR it did not build — so
/// the two cannot drift into two answers about the same symbol.
pub(crate) fn is_element_symbol(symbol: &str) -> bool {
    Element::by_symbol(&canonical_element_symbol(symbol)).is_some()
}

/// Which notation the parser or writer speaks.
///
/// The dialects share one AST, one scanner and one recursive-descent
/// parser; this enum is what that shared machinery branches on, so "which
/// language is this" has a single home rather than one private copy per
/// direction.
///
/// `FragmentSmiles` is a SMILES fragment body extended with `CGsmiles` /
/// `BigSMILES` bonding descriptors ([`BondingDescriptor`](ast::BondingDescriptor)),
/// the bracketed joining-site markers `[$]`, `[<]`, `[>]` and `[!]`. It is
/// selected by [`parser::parse_fragment_smiles`] and
/// [`writer::fragment_smiles_text`] only — the plain-SMILES entry points stay
/// strict, so no `.smi` line can silently carry descriptors.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) enum Dialect {
    /// Standard SMILES — only concrete atoms and bonds.
    Smiles,
    /// SMARTS — adds query primitives, logical operators, wildcard/ring bonds.
    Smarts,
    /// SMILES fragment body with `CGsmiles` / `BigSMILES` bonding descriptors.
    FragmentSmiles,
    /// PSMILES repeat unit: SMILES whose `*` / `[*]` are joining sites, read
    /// as the unlabelled `$` descriptor ([`psmiles`]). Bracket descriptors
    /// are refused, as in plain SMILES.
    Psmiles,
    /// `BigSMILES` unit body: the fragment dialect with integer descriptor
    /// ids, no `[!]`, and generative-`BigSMILES` weights `[<|w|]`
    /// ([`parser::parse_bigsmiles_body`]).
    BigSmiles,
}

/// How a dialect reads the brackets `[$]`, `[<]`, `[>]`, `[!]`.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) enum Descriptors {
    /// The dialect has no descriptor notation and says so: plain SMILES.
    Refused,
    /// The glyphs are not descriptors here: in SMARTS `[$(C)]` is a
    /// recursive pattern and `[!C]` a negation, left to the query parser.
    QuerySyntax,
    /// The dialect reads descriptors, by these rules.
    Read(DescriptorRules),
}

/// The descriptor rules of one notation — everything the shared parser
/// needs to know to read that notation's joining sites. A notation is added
/// by writing its rules here, in one place, not by branching in the parser.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) struct DescriptorRules {
    /// The descriptor is spelled as a star, `*` or `[*]`, read as the
    /// unlabelled `$`; the bracket glyphs are then refused.
    pub star: bool,
    /// The squash operator `[!]` is part of the notation.
    pub squash: bool,
    /// What a label may be written with.
    pub labels: LabelRule,
    /// Generation weights `|w …|` may follow the label inside the bracket.
    pub weights: bool,
    /// The notation also has ladder and non-covalent descriptors
    /// (`[<[<1]1]`, `[$:1]`); they are named as unsupported rather than
    /// read as a label.
    pub extended: bool,
    /// A descriptor bonds exactly one atom: one per run, and nothing may
    /// bond on after one written behind an atom.
    pub terminal: bool,
    /// `CGsmiles` atom annotations (`[C;0.5]`) are part of the notation and
    /// are named as unsupported rather than reported as an unclosed bracket.
    pub atom_annotations: bool,
}

/// What a descriptor label may be written with.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) enum LabelRule {
    /// ASCII letters and digits (`CGsmiles`: `[$a1]`).
    Alphanumeric,
    /// ASCII digits (`BigSMILES`: `[$1]`).
    Digits,
}

impl LabelRule {
    /// Whether `label` (possibly empty) is one.
    pub(crate) fn admits(self, label: &str) -> bool {
        match self {
            LabelRule::Alphanumeric => label.chars().all(|c| c.is_ascii_alphanumeric()),
            LabelRule::Digits => label.chars().all(|c| c.is_ascii_digit()),
        }
    }
}

impl Dialect {
    /// How this dialect reads descriptors. The one table of per-notation
    /// descriptor rules.
    pub(crate) fn descriptors(self) -> Descriptors {
        let rules = DescriptorRules {
            star: false,
            squash: false,
            labels: LabelRule::Alphanumeric,
            weights: false,
            extended: false,
            terminal: false,
            atom_annotations: false,
        };
        match self {
            Dialect::Smiles => Descriptors::Refused,
            Dialect::Smarts => Descriptors::QuerySyntax,
            Dialect::FragmentSmiles => Descriptors::Read(DescriptorRules {
                squash: true,
                atom_annotations: true,
                ..rules
            }),
            Dialect::Psmiles => Descriptors::Read(DescriptorRules {
                star: true,
                terminal: true,
                ..rules
            }),
            Dialect::BigSmiles => Descriptors::Read(DescriptorRules {
                labels: LabelRule::Digits,
                weights: true,
                extended: true,
                ..rules
            }),
        }
    }

    /// The notation a diagnostic about this dialect names.
    ///
    /// The fragment body is SMILES widened with bonding descriptors, not a
    /// language of its own, so it reports as SMILES; PSMILES and `BigSMILES`
    /// are named notations and report as themselves. The mapping is total and lives here alone, so the parser and
    /// the writer cannot drift into two answers — and neither holds a second
    /// notation field beside its `dialect`.
    pub(crate) fn notation(self) -> Notation {
        match self {
            Dialect::Smiles | Dialect::FragmentSmiles => Notation::Smiles,
            Dialect::Smarts => Notation::Smarts,
            Dialect::Psmiles => Notation::Psmiles,
            Dialect::BigSmiles => Notation::BigSmiles,
        }
    }
}
