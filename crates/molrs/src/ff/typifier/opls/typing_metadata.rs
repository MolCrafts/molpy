//! OPLS-AA typing metadata: per-type SMARTS definition, overrides, explicit
//! priority and overlay layer.

use std::collections::HashMap;

/// One `<Type>` row of an OPLS-AA `<AtomTypes>` section, holding the typing
/// metadata (not the potential parameters).
///
/// This is the **runtime** typing record. Its static, compile-time counterpart
/// is [`OplsRuleRow`](crate::ff::params::OplsRuleRow): the shipped OPLS-AA
/// typifier builds one `OplsTypeRow` from each `OplsRuleRow`, taking `class`
/// from the matching [`OplsAtomRow`](crate::ff::params::OplsAtomRow) and
/// `layer` 0, while
/// [`OpenmmXmlReader::read_opls_typifier`](crate::io::openmm_xml::OpenmmXmlReader::read_opls_typifier)
/// builds them from XML attributes.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct OplsTypeRow {
    /// Chemical class (the `class` attribute, e.g. `"CT"`). Bonded forces key on
    /// this class vocabulary, distinct from the `opls_NNN` type vocabulary.
    pub class: String,
    /// SMARTS definition (the `def` attribute), or `None` for rows that
    /// carry no `def` and therefore cannot be matched automatically.
    pub def: Option<String>,
    /// Type names this row overrides (parsed from a comma-separated `overrides`
    /// attribute); empty when absent.
    pub overrides: Vec<String>,
    /// Explicit `priority` attribute, if present (absent reads as 0). It orders
    /// only candidates that no other candidate dominates.
    pub priority: Option<i64>,
    /// Overlay layer (the `layer` attribute); `0` (base force field) when absent.
    pub layer: u32,
}

/// Parsed OPLS-AA typing metadata: [`OplsTypeRow`]s keyed by `opls_NNN` type
/// name.
///
/// Read from the same XML as the potential [`ForceField`](crate::ff::forcefield::ForceField)
/// but kept separate — this table drives SMARTS atom typing, the `ForceField`
/// drives energy evaluation.
///
/// This is the typing-metadata half of an OPLS-AA force field, kept separate
/// from the potential parameters (the
/// [`ForceField`](crate::ff::forcefield::ForceField)). The shipped table is
/// joined from the molrs-owned rules of
/// [`OPLSAA_TYPING`](crate::ff::params::OPLSAA_TYPING); for a caller's own OPLS / CL&P XML, the
/// potential reader
/// ([`OpenmmXmlReader`](crate::io::openmm_xml::OpenmmXmlReader))
/// drops the `def` / `overrides` / `priority` / `layer` attributes and
/// [`OpenmmXmlReader::read_opls_typifier`](crate::io::openmm_xml::OpenmmXmlReader::read_opls_typifier)
/// reads them into the [`OplsTypingMetadata`] table here.
///
/// # How the fields rank candidates
///
/// The table carries the inputs; the
/// layered typing engine ranks with
/// them. `layer` and `overrides` define a pairwise *dominance*: a type on a
/// higher layer, or on the same layer and overriding another (directly or
/// transitively), always wins over it. `priority` (absent = 0) only orders
/// candidates that nothing dominates. No field is folded into a single score.
///
/// # Scope
///
/// Only types carrying a SMARTS `def` participate in automatic SMARTS typing.
/// Rows with no `def` (the united-atom `opls_001`–`opls_134` block, for one)
/// can only be assigned by hand or read back from a LAMMPS data file.
#[derive(Debug, Clone, Default)]
pub struct OplsTypingMetadata {
    rows: HashMap<String, OplsTypeRow>,
}

impl OplsTypingMetadata {
    /// Create an empty metadata table.
    pub fn new() -> Self {
        Self::default()
    }

    /// Create from pre-parsed rows keyed by `opls_NNN`.
    pub fn from_rows(rows: HashMap<String, OplsTypeRow>) -> Self {
        Self { rows }
    }

    /// Insert or replace a row.
    pub fn insert(&mut self, name: impl Into<String>, row: OplsTypeRow) {
        self.rows.insert(name.into(), row);
    }

    /// Look up a row by `opls_NNN` name.
    pub fn get(&self, name: &str) -> Option<&OplsTypeRow> {
        self.rows.get(name)
    }

    /// Number of rows.
    pub fn len(&self) -> usize {
        self.rows.len()
    }

    /// Whether the table is empty.
    pub fn is_empty(&self) -> bool {
        self.rows.is_empty()
    }

    /// Iterate `(name, row)` pairs.
    pub fn iter(&self) -> impl Iterator<Item = (&String, &OplsTypeRow)> {
        self.rows.iter()
    }

    /// Check that every `overrides` names a type this table declares and that
    /// the overrides form no cycle — the invariant a typifier ranks by.
    ///
    /// # Errors
    ///
    /// A dangling override (naming both types) or a cycle (naming its
    /// members).
    pub fn validate(&self) -> Result<(), String> {
        super::layered::Dominance::new(self).map(|_| ())
    }
}
