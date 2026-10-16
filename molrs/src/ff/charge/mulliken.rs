//! [`MullikenModel`] — the QM charges, unchanged.

use molrs::core::Atomistic;

use super::error::ChargeError;
use super::model::{ChargeModel, check_count};

/// Mulliken populations, passed through.
///
/// The model with no topology correction at all: `antechamber -c mul` writes `sqm`'s
/// Mulliken populations straight out, with `-eq 0` (no equivalencing, because that is
/// per charge method and Mulliken's default is off).
///
/// # Examples
///
/// ```
/// use molrs::core::{Atomistic, Element, keys};
/// use molrs::props;
/// use molrs::ff::charge::{ChargeModel, MullikenModel};
///
/// # fn main() -> Result<(), Box<dyn std::error::Error>> {
/// let mut mol = Atomistic::new();
/// mol.add_atom(props! {
///     keys::ATOMIC_NUMBER => Element::O,
///     keys::X => 0.0,
///     keys::Y => 0.0,
///     keys::Z => 0.0,
/// })?;
/// mol.add_atom(props! {
///     keys::ATOMIC_NUMBER => Element::H,
///     keys::X => 0.96,
///     keys::Y => 0.0,
///     keys::Z => 0.0,
/// })?;
///
/// let q = MullikenModel.assign(&mol, Some(&[-0.4, 0.4]))?;
/// assert_eq!(q, vec![-0.4, 0.4]);
/// assert!(!MullikenModel.needs_equivalencing());
/// # Ok(())
/// # }
/// ```
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct MullikenModel;

impl ChargeModel for MullikenModel {
    fn needs_equivalencing(&self) -> bool {
        // antechamber's `-eq` defaults to 0 for every method but `bcc` / `abcg2` /
        // `resp`. Averaging here would be a correction, and this model has none.
        false
    }

    /// The charges it was given — the same bits.
    ///
    /// Not "within 1e-10": a pass-through that rounded, rescaled, or renormalized the
    /// total to the net charge would satisfy a tolerance and still be *doing
    /// something*, and the entire content of this model is that it does nothing.
    ///
    /// # Arguments
    ///
    /// * `mol` — used only for its atom count; nothing is read off it and nothing is
    ///   written to it.
    /// * `qm` — the QM charges, in graph atom order.
    ///
    /// # Returns
    ///
    /// `qm`, unchanged.
    ///
    /// # Errors
    ///
    /// [`ChargeError::MissingQmCharges`] when `qm` is `None` — Mulliken populations
    /// are a QM result, and molrs will not invent them;
    /// [`ChargeError::ChargeCountMismatch`] when they are not one per atom.
    fn assign(&self, mol: &Atomistic, qm: Option<&[f64]>) -> Result<Vec<f64>, ChargeError> {
        let qm = qm.ok_or(ChargeError::MissingQmCharges { model: "Mulliken" })?;
        check_count(mol, qm)?;
        Ok(qm.to_vec())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::props;
    use molrs::core::Element;
    use molrs::core::keys;

    fn water() -> Atomistic {
        let mut mol = Atomistic::new();
        let o = mol
            .add_atom(props! {
                keys::ATOMIC_NUMBER => Element::O,
                keys::X => 0.0,
                keys::Y => 0.0,
                keys::Z => 0.0,
            })
            .unwrap();
        let h1 = mol
            .add_atom(props! {
                keys::ATOMIC_NUMBER => Element::H,
                keys::X => 0.96,
                keys::Y => 0.0,
                keys::Z => 0.0,
            })
            .unwrap();
        let h2 = mol
            .add_atom(props! {
                keys::ATOMIC_NUMBER => Element::H,
                keys::X => -0.24,
                keys::Y => 0.93,
                keys::Z => 0.0,
            })
            .unwrap();
        mol.add_bond(o, h1, props! {}).expect("add O-H");
        mol.add_bond(o, h2, props! {}).expect("add O-H");
        mol
    }

    /// Pass-through means the same bits, not merely the same value to a tolerance.
    #[test]
    fn the_charges_come_back_bitwise() {
        let qm = [-0.8123456789012345, 0.4061728394506172, 0.4061728394506173];
        let got = MullikenModel.assign(&water(), Some(&qm)).expect("assign");
        for (q, w) in got.iter().zip(&qm) {
            assert_eq!(q.to_bits(), w.to_bits());
        }
    }

    /// Two topologically-equivalent hydrogens stay split: `needs_equivalencing` is
    /// `false`, and the declaration and the behaviour must be the same thing.
    #[test]
    fn equivalent_atoms_are_not_averaged() {
        let qm = [-0.8, 0.35, 0.45];
        let got = MullikenModel.assign(&water(), Some(&qm)).expect("assign");
        assert_ne!(got[1].to_bits(), got[2].to_bits());
    }
}
