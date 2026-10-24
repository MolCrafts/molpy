//! SMILES, the Simplified Molecular Input Line Entry System: a molecule as one
//! line of text (`CCO` is ethanol, `c1ccccc1` benzene).
//!
//! This module holds the notation's classes:
//!
//! - [`SmilesIr`], the parsed text — the syntax tree ([`Chain`],
//!   [`AtomNode`], [`AtomSpec`], …): [`SmilesIr::parse`] reads SMILES text,
//!   [`SmilesIr::to_atomistic`] builds the molecule (connectivity only:
//!   implicit hydrogens are not added, no coordinates are generated),
//!   [`SmilesIr::from_atomistic`] goes the other way as
//!   [`SmilesEmitOptions`] say, and its `Display` writes the text;
//!   [`SmilesIr::validate`] checks it;
//! - [`FragmentIr`], a fragment body with `CGsmiles` / `BigSMILES` bonding
//!   descriptors `[$]`, `[<]`, `[>]`, `[!]` ([`SmilesIr::from_fragment`]
//!   parses one), whose [`to_atomistic`](FragmentIr::to_atomistic) returns
//!   the descriptor map beside the graph;
//! - [`SmilesError`], every refusal of the SMILES family of notations
//!   (SMILES, SMARTS, CGsmiles: its [`Notation`] says which);
//! - [`SmilesReader`], a `.smi` file read one molecule per line.
//!
//! The grammar is the crate's one line-notation grammar, shared with SMARTS
//! ([`crate::perceive::smarts`]): the syntax tree carries SMARTS query nodes
//! too ([`AtomQuery`], [`BondQuery`]), which a SMILES conversion refuses.
//!
//! # Pipeline
//!
//! ```text
//! SMILES text → SmilesIr::parse() → SmilesIr → SmilesIr::to_atomistic() → Atomistic
//! Atomistic → SmilesIr::from_atomistic() → SmilesIr → to_string() → SMILES text
//! fragment body → SmilesIr::from_fragment() → FragmentIr
//!     → FragmentIr::to_atomistic() → (Atomistic, descriptor map)
//! ```
//!
//! # Examples
//!
//! ```
//! use molrs::io::smiles::{SmilesEmitOptions, SmilesIr};
//!
//! let ir = SmilesIr::parse("CCO").unwrap();
//! let mol = ir.to_atomistic().unwrap();
//! assert_eq!(mol.n_atoms(), 3);
//! let text = SmilesIr::from_atomistic(&mol, &SmilesEmitOptions::default())
//!     .unwrap()
//!     .to_string();
//! assert_eq!(SmilesIr::parse(&text).unwrap().to_atomistic().unwrap().n_atoms(), 3);
//! ```

pub(crate) mod bond_stereo;
pub(crate) mod chirality;
pub(crate) mod element_check;
pub(crate) mod emit_options;
pub(crate) mod ir_from_atomistic;
pub(crate) mod ir_to_atomistic;
mod line_reader;

pub use crate::line_notation::ast::{
    AtomNode, AtomPrimitive, AtomQuery, AtomSpec, BondKind, BondQuery, BondingDescriptor,
    BracketSymbol, Chain, ChainElement, Chirality, DescriptorKind, SmilesIr, Span,
};
pub use crate::line_notation::error::{Notation, SmilesError, SmilesErrorKind};
pub use emit_options::{AromaticEmit, HydrogenEmit, MultiComponentEmit, SmilesEmitOptions};
pub use ir_to_atomistic::FragmentIr;
pub use line_reader::SmilesReader;

/// The IR written as text: SMILES, or SMARTS where it holds query atoms or
/// bond operators (a SMILES IR writes the same either way).
///
/// [`SmilesIr::from_atomistic`] then `to_string()` is how a molecule is
/// written as SMILES.
///
/// # Errors
///
/// A formatting error — `to_string` panics — for an IR no parser or
/// [`SmilesIr::from_atomistic`] produces: one with no components, or one
/// carrying bonding descriptors (a fragment body, [`FragmentIr`]'s).
///
/// ```
/// use molrs::io::smiles::{SmilesEmitOptions, SmilesIr};
///
/// let mol = SmilesIr::parse("OCC")?.to_atomistic()?;
/// let text = SmilesIr::from_atomistic(&mol, &SmilesEmitOptions::default())?.to_string();
/// assert_eq!(SmilesIr::parse(&text)?.to_atomistic()?.n_atoms(), 3);
/// # Ok::<(), molrs::io::smiles::SmilesError>(())
/// ```
impl std::fmt::Display for SmilesIr {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        let text = crate::line_notation::writer::smarts_text(self).map_err(|_| std::fmt::Error)?;
        f.write_str(&text)
    }
}

impl SmilesIr {
    /// Parse SMILES text into its IR.
    ///
    /// # Errors
    ///
    /// A [`SmilesError`] for text that is not SMILES: a syntax error, an
    /// unknown element, a bonding descriptor (a fragment body goes through
    /// [`from_fragment`](Self::from_fragment)).
    pub fn parse(smiles: &str) -> Result<SmilesIr, SmilesError> {
        crate::line_notation::parser::parse_smiles(smiles)
    }

    /// Parse a SMILES fragment body — SMILES plus `CGsmiles` / `BigSMILES`
    /// bonding descriptors (`[<]OCC[>]`, `[$]COC[$]`) — into its IR.
    ///
    /// # Errors
    ///
    /// A [`SmilesError`] for text that is not fragment notation.
    pub fn from_fragment(body: &str) -> Result<FragmentIr, SmilesError> {
        crate::line_notation::parser::parse_fragment_smiles(body).map(FragmentIr::new)
    }
}
