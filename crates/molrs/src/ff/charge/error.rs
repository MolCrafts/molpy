//! The one error type every charge model fails with.

use std::fmt;

use crate::ff::typifier::species_label;

/// Why a [`ChargeModel`](super::ChargeModel) could not produce charges.
///
/// Every variant is a refusal, never a fallback: molrs does not answer with a
/// plausible-looking charge when it does not have one.
///
/// Typed, not a `String`: the C++ and Python bridges have to *discriminate*. A
/// molecule the parameter set cannot correct (boron: `BCCPARM.DAT` has no row) is
/// a permanent, structural refusal the caller must surface to a user, while a
/// charge-count mismatch is the caller's own marshalling bug. Prose could not
/// tell them apart.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum ChargeError {
    /// The model needs QM base charges and was handed `None`.
    ///
    /// AM1-BCC, ABCG2 and Mulliken all consume QM charges — molrs does not solve
    /// AM1 itself (that is Atomiverse's job). A model that needs them and has none
    /// fails here rather than inventing a base to correct.
    MissingQmCharges {
        /// The model that needed them, for the message.
        model: &'static str,
    },
    /// The supplied QM charges do not match the molecule's atom count.
    ///
    /// The push API's one new failure mode: the charges arrive as a slice, so
    /// they can disagree with the molecule they belong to.
    ChargeCountMismatch {
        /// Atoms in the molecule.
        expected: usize,
        /// Charges supplied.
        got: usize,
    },
    /// No `ATOMTYPE_*.DEF` rule matched an atom, so its BCC type is unknown.
    ///
    /// An atom the table cannot even name is an atom whose corrected charge is
    /// unknown; it is not defaulted to an element fallback.
    MissingAtomType {
        /// The atom-type table that was walked (e.g. `ATOMTYPE_BCC.DEF`).
        table: &'static str,
        /// The atom's 0-based index in graph atom order.
        atom: usize,
        /// The atom's atomic number (`0` for a wildcard).
        atomic_number: u8,
    },
    /// No `BCCPARM*.DAT` row covers a perceived bond.
    ///
    /// The bond is named by the two **BCC** atom types and the perceived
    /// antechamber bond type — the model's own internal labels, not the caller's
    /// force-field types.
    MissingCorrection {
        /// BCC atom type of one endpoint.
        left: String,
        /// BCC atom type of the other endpoint.
        right: String,
        /// The perceived antechamber bond type (1/2/3/6/7/8/9).
        bond_type: i32,
    },
    /// The molecule itself could not be read.
    ///
    /// A malformed graph (an atom with no valid atomic number, a bond whose
    /// endpoints are gone) — a defect in the input, not in the parameter set.
    Malformed {
        /// What the graph layer said.
        detail: String,
    },
}

impl fmt::Display for ChargeError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::MissingQmCharges { model } => write!(
                f,
                "the {model} charge model requires QM base charges; none were supplied"
            ),
            Self::ChargeCountMismatch { expected, got } => write!(
                f,
                "{got} QM charges were supplied for a molecule with {expected} atoms"
            ),
            Self::MissingAtomType {
                table,
                atom,
                atomic_number,
            } => write!(
                f,
                "missing {table} atom type for atom {atom} ({})",
                species_label(*atomic_number)
            ),
            Self::MissingCorrection {
                left,
                right,
                bond_type,
            } => write!(
                f,
                "missing BCC correction for bond {left}|{right}|{bond_type}"
            ),
            Self::Malformed { detail } => write!(f, "malformed molecule: {detail}"),
        }
    }
}

impl std::error::Error for ChargeError {}

#[cfg(test)]
mod tests {
    use super::*;

    /// The absent-QM message must not read like an absent-parameter one.
    ///
    /// The two are the failures a bridge has to tell apart: "you did not give me
    /// AM1 charges" is the caller's omission, "this molecule has no BCC row" is a
    /// permanent property of the parameter set.
    #[test]
    fn missing_qm_charges_does_not_read_as_a_missing_parameter() {
        let err = ChargeError::MissingQmCharges { model: "AM1-BCC" };
        let text = err.to_string();
        assert!(!text.contains("missing BCC correction"), "{text}");
        assert!(text.contains("QM base charges"), "{text}");
    }

    /// A missing row names the bond it could not correct.
    #[test]
    fn missing_correction_names_the_bond() {
        let err = ChargeError::MissingCorrection {
            left: "B".to_owned(),
            right: "11".to_owned(),
            bond_type: 1,
        };
        assert_eq!(
            err.to_string(),
            "missing BCC correction for bond B|11|1",
            "the two BCC atom types and the perceived bond type"
        );
    }
}
