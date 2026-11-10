//! `BigSMILES` and generative `BigSMILES`: polymers as SMILES with
//! stochastic objects, read into the polymer model of
//! [`crate::builder`].
//!
//! `BigSMILES` (Lin et al., ACS Cent. Sci. 2019) writes a polymer as SMILES
//! in which a *stochastic object* `{[] … []}` stands for a chain of repeat
//! units of unspecified length: `{[][<]CC[>],[<]CC([>])C;[<][H],[>][H][]}`
//! lists two repeat units and two end groups between two empty terminals.
//! The bracketed `[<]`, `[>]`, `[$]` (with an optional integer id, `[$1]`) are
//! the bonding descriptors of the fragment dialect — the same ports CGsmiles
//! and PSMILES write. Generative `BigSMILES` (Schneider & de Pablo, Digital
//! Discovery 2024) adds what a generator needs: weights inside descriptors
//! (`[<|0.5|]`, `[$|0.2 0.8|]`), a size distribution after an object
//! (`|schulz_zimm(1500, 1400)|`) and a system size after a molecule
//! (`.|5e5|`, or `.|20%|` for a share of a mixture).
//!
//! [`BigSmilesIr::parse`] reads both into records of what was written;
//! [`BigSmilesIr::to_molecules`] and [`BigSmilesIr::to_system`] turn them
//! into [`Molecule`](crate::builder::Molecule)s and a
//! [`System`](crate::builder::System), whose samples an
//! [`Assembler`](crate::builder::Assembler) builds.
//!
//! # Grammar this reader accepts
//!
//! ```text
//! string    → molecule ( '.' amount? molecule? )*
//! molecule  → ( smiles | object )+
//! object    → '{' terminal unit (',' unit)* (';' unit (',' unit)*)? terminal '}' distribution?
//! terminal  → '[' ( ('$' | '<' | '>') digits? weights? )? ']'
//! unit      → fragment body, descriptors '[$1]' '[<|w|]' …
//! distribution → '|' name '(' number (',' number)* ')' '|'
//! amount    → '|' number '%'? '|'
//! ```
//!
//! Whitespace is allowed inside an object around its units and terminals,
//! and inside `|…|`. A piece of SMILES next to an object is joined to it
//! through the object's terminal: `CC{[>][<]CC[>][<]}O` bonds the second
//! carbon of `CC` to the object's first unit and the oxygen to its last, as
//! if `CC[>]` and `[<]O` had been written. A bond symbol between the two
//! (`C={[>]…}`) is the order of that bond.
//!
//! # Units and measures
//!
//! A size distribution counts the **heavy-atom mass** of its object in
//! g/mol, and a system amount the heavy-atom mass of all molecules of that
//! kind — the measure generative `BigSMILES` writes both in
//! ([`Measure::HeavyAtomMass`](crate::builder::Measure::HeavyAtomMass)).
//!
//! # Refusals of valid notation
//!
//! Each is
//! [`SmilesErrorKind::BigUnsupported`](crate::io::smiles::SmilesErrorKind::BigUnsupported): stochastic objects nested in
//! a unit, an object inside a branch, fragment declarations `[#name]` /
//! `.{#name=…}`, and ladder or non-covalent descriptors.
//!
//! A string that reads but spells a model the builder refuses is
//! [`SmilesErrorKind::BigBuild`](crate::io::smiles::SmilesErrorKind::BigBuild),
//! raised by the conversions and spanned at the piece refused: an object
//! with nothing before it and no end group to start from
//! (`{[][$]CC[$][]}`), a terminal facing no part, pieces whose ports do not
//! join. List weights count the ports of the object's repeat units and end
//! groups, in written order; the terminals are not positions.
//!
//! References: Lin, T.-S. et al., *BigSMILES: A Structurally-Based Line
//! Notation for Describing Macromolecules*, ACS Cent. Sci. **5**, 1523–1531
//! (2019), DOI: 10.1021/acscentsci.9b00476; Schneider, L.; de Pablo, J. J.
//! *Generative BigSMILES*, Digital Discovery **3**, 51–61 (2024),
//! DOI: 10.1039/D3DD00147D.

mod lower;
mod parser;
#[cfg(test)]
mod tests;

use crate::builder::{Amount, Distribution};
use crate::io::smiles::{BondKind, DescriptorKind, FragmentIr, SmilesError, Span};

/// A parsed `BigSMILES` / generative-`BigSMILES` string: its molecules, in
/// written order.
#[derive(Debug, Clone)]
pub struct BigSmilesIr {
    /// The molecules, separated by `.` in the string.
    pub molecules: Vec<MoleculeSpec>,
    /// The whole string.
    pub span: Span,
    /// The text that was parsed, for the diagnostics of the conversions.
    input: String,
}

/// One molecule as written: its pieces and, in a system, its amount.
#[derive(Debug, Clone)]
pub struct MoleculeSpec {
    /// The pieces, in written order.
    pub parts: Vec<PartSpec>,
    /// The amount `.|…|` after the molecule, if one is written.
    pub amount: Option<Amount>,
    /// The molecule's text, amount excluded.
    pub span: Span,
}

/// One piece of a molecule.
#[derive(Debug, Clone)]
pub enum PartSpec {
    /// SMILES between stochastic objects (or the whole of a small molecule),
    /// with the terminals of its neighbouring objects as its descriptors.
    Smiles(FragmentSpec),
    /// A stochastic object.
    Stochastic(ObjectSpec),
}

/// A unit's body and the generation weights of its descriptors.
#[derive(Debug, Clone)]
pub struct FragmentSpec {
    /// The body as a fragment.
    pub ir: FragmentIr,
    /// The weights written inside each descriptor, in port order; `None`
    /// where none is written.
    pub weights: Vec<Option<Vec<f64>>>,
    /// The body's text.
    pub span: Span,
}

/// A stochastic object as written.
#[derive(Debug, Clone)]
pub struct ObjectSpec {
    /// The left terminal; `None` for `[]`.
    pub left: Option<TerminalSpec>,
    /// The repeat units.
    pub repeat_units: Vec<FragmentSpec>,
    /// The end groups (after `;`).
    pub end_groups: Vec<FragmentSpec>,
    /// The right terminal; `None` for `[]`.
    pub right: Option<TerminalSpec>,
    /// The size distribution `|…|` after the object, if one is written.
    pub distribution: Option<Distribution>,
    /// The object's text, distribution included.
    pub span: Span,
}

/// A non-empty terminal descriptor.
#[derive(Debug, Clone, PartialEq)]
pub struct TerminalSpec {
    /// The descriptor's operator.
    pub kind: DescriptorKind,
    /// Its id, `""` when none is written.
    pub label: String,
    /// The bond symbol written between the terminal's object and the
    /// neighbouring SMILES, if any.
    pub order: Option<BondKind>,
    /// Weights written inside it, if any.
    pub weights: Option<Vec<f64>>,
}

impl BigSmilesIr {
    /// Parse a `BigSMILES` or generative-`BigSMILES` string.
    ///
    /// # Errors
    ///
    /// A [`SmilesError`] stamped
    /// [`Notation::BigSmiles`](crate::io::smiles::Notation::BigSmiles): what a
    /// unit body's fragment parse refuses, re-based into the whole string;
    /// [`SmilesErrorKind::BigMalformedObject`],
    /// [`SmilesErrorKind::BigUnsupported`],
    /// [`SmilesErrorKind::BigUnknownDistribution`],
    /// [`SmilesErrorKind::BigDistributionArity`],
    /// [`SmilesErrorKind::BigInvalidNumber`],
    /// [`SmilesErrorKind::BigInvalidWeights`] and
    /// [`SmilesErrorKind::BigInvalidAmount`] for the rules of the outer
    /// grammar.
    ///
    /// [`SmilesErrorKind::BigMalformedObject`]: crate::io::smiles::SmilesErrorKind::BigMalformedObject
    /// [`SmilesErrorKind::BigUnsupported`]: crate::io::smiles::SmilesErrorKind::BigUnsupported
    /// [`SmilesErrorKind::BigUnknownDistribution`]: crate::io::smiles::SmilesErrorKind::BigUnknownDistribution
    /// [`SmilesErrorKind::BigDistributionArity`]: crate::io::smiles::SmilesErrorKind::BigDistributionArity
    /// [`SmilesErrorKind::BigInvalidNumber`]: crate::io::smiles::SmilesErrorKind::BigInvalidNumber
    /// [`SmilesErrorKind::BigInvalidWeights`]: crate::io::smiles::SmilesErrorKind::BigInvalidWeights
    /// [`SmilesErrorKind::BigInvalidAmount`]: crate::io::smiles::SmilesErrorKind::BigInvalidAmount
    ///
    /// # Examples
    ///
    /// ```
    /// use molrs::io::bigsmiles::BigSmilesIr;
    ///
    /// let ir = BigSmilesIr::parse("{[][<]CC[>];[<][H],[>][H][]}|poisson(300)|.|3000|")?;
    /// let system = ir.to_system()?;
    /// let sample = system.sample(7)?;
    /// assert!(sample.molecules.iter().map(|m| m.heavy_atom_mass).sum::<f64>() >= 3000.0);
    /// # Ok::<(), Box<dyn std::error::Error>>(())
    /// ```
    pub fn parse(text: &str) -> Result<BigSmilesIr, SmilesError> {
        parser::parse(text)
    }
}
