//! The [`ChargeModel`] trait — one seam for every charge method.

use std::collections::HashMap;

use molrs::core::keys;
use molrs::core::{Atomistic, NodeId};

use molrs::perceive::{EquivalenceOptions, perceive_equivalence_classes};

use crate::ff::typifier::atd::{AtdError, DUMMY_TYPE};

use super::error::ChargeError;

/// A charge model: molecule (and optionally QM base charges) in, charges out.
///
/// Object-safe on purpose. The C++ and Python bridges choose a model at runtime, so
/// they must be able to hold one as `Box<dyn ChargeModel>`; and a trait whose users
/// have to downcast to find out what they are holding is a special case wearing a
/// trait's clothes.
///
/// Nothing is written back into the molecule. The charges are the return value, so
/// a caller keeps their force-field atom types and gets BCC charges — the standard
/// AM1-BCC workflow needs both at once.
pub trait ChargeModel {
    /// Does this model average its base charges over the topological-equivalence
    /// classes before correcting them?
    ///
    /// antechamber's `-eq`, whose default is **per charge method**: `1` for `bcc` /
    /// `abcg2` / `resp`, `0` for everything else. That makes it a property of the
    /// model, not of the graph and not of the caller's memory — hence a method here
    /// rather than a flag on either.
    ///
    /// # Returns
    ///
    /// `true` when [`assign`](Self::assign) applies the class-mean to the QM charges
    /// it is given.
    fn needs_equivalencing(&self) -> bool;

    /// The final charges of every atom, in graph atom order.
    ///
    /// This is the **whole** model: raw QM charges in (as an AM1 backend really
    /// hands them over — un-averaged, one conformer), final charges out. A model
    /// that declares [`needs_equivalencing`](Self::needs_equivalencing) applies the
    /// class-mean itself, because otherwise the declaration would be inert
    /// documentation that every caller had to re-implement.
    ///
    /// # Arguments
    ///
    /// * `mol` — the molecule; left untouched, including its `type` columns.
    /// * `qm` — QM base charges in graph atom order, or `None` when the caller has
    ///   none. A model that needs them fails with
    ///   [`ChargeError::MissingQmCharges`]; a model that does not (Gasteiger)
    ///   ignores the argument.
    ///
    /// # Returns
    ///
    /// One charge per atom, in graph atom order.
    ///
    /// # Errors
    ///
    /// [`ChargeError`] — missing QM input, a charge-count mismatch, an atom the
    /// parameter set cannot type, or a bond it cannot correct. Never a fallback
    /// value.
    fn assign(&self, mol: &Atomistic, qm: Option<&[f64]>) -> Result<Vec<f64>, ChargeError>;
}

/// The atoms of `mol`, in graph atom order — the order every charge slice is in.
pub(super) fn atom_ids(mol: &Atomistic) -> Vec<NodeId> {
    mol.atom_ids().collect()
}

/// Reject a charge slice that does not belong to this molecule.
///
/// # Errors
///
/// [`ChargeError::ChargeCountMismatch`], carrying both counts, when the slice is
/// not one charge per atom.
pub(super) fn check_count(mol: &Atomistic, charges: &[f64]) -> Result<(), ChargeError> {
    let expected = mol.n_atoms();
    if charges.len() != expected {
        return Err(ChargeError::ChargeCountMismatch {
            expected,
            got: charges.len(),
        });
    }
    Ok(())
}

/// Average `qm` over the molecule's topological-equivalence classes (`-eq 1`).
///
/// The step antechamber runs between AM1 and BCC (`charge.c::bccharge()`). A
/// semi-empirical calculation is done on one conformer, so its Mulliken charges
/// are not symmetric — methanol's three methyl hydrogens come out of `sqm` split
/// `0.053 / 0.098 / 0.053` purely because one of them eclipses the O–H — and the
/// class-mean removes that artefact before any bond-charge correction is applied.
///
/// Perception owns the classes ([`perceive_equivalence_classes`]); the mean is this
/// charge-model step. Each class's charges are summed in graph atom order, as
/// `bccharge()` sums them, and the mean is given to every member; a singleton
/// keeps its charge bit for bit.
///
/// # Precision
///
/// The mean is a rounded `f64`, so broadcasting it perturbs the molecule's total
/// charge by a few ULP (measured over the 37-molecule antechamber oracle: at most
/// `3.7e-16` e). That residual is inherent to an arithmetic mean — no assignment
/// that gives every class member *the same* `f64` can also reproduce the total
/// bit-for-bit — and antechamber carries the identical residual. Nothing is
/// renormalized away, because renormalizing would be a divergence from
/// antechamber, not a fix.
///
/// # Arguments
///
/// * `mol` — the molecule to partition; left untouched.
/// * `qm` — one base charge per atom, in graph atom order.
///
/// # Returns
///
/// The class-averaged charges, in the same order.
pub(super) fn equivalence_average(mol: &Atomistic, qm: &[f64]) -> Vec<f64> {
    let index: HashMap<NodeId, usize> = atom_ids(mol)
        .into_iter()
        .enumerate()
        .map(|(i, id)| (id, i))
        .collect();
    let mut out = qm.to_vec();
    for members in perceive_equivalence_classes(mol, EquivalenceOptions::bcc()).classes() {
        if members.len() < 2 {
            continue;
        }
        let rows: Vec<usize> = members.iter().map(|id| index[id]).collect();
        let mean = rows.iter().map(|&i| qm[i]).sum::<f64>() / rows.len() as f64;
        for i in rows {
            out[i] = mean;
        }
    }
    out
}

/// A copy of `mol` with both `type` columns removed.
///
/// What a model perceives from: the caller's atom types (GAFF `c3`, OPLS `opls_135`)
/// and bond types (LAMMPS bond-type ids) are *their* labels, and a model's answer
/// must be a function of the molecule alone — which is what
/// `bcc::correct_ignores_the_{atom,bond}_types_the_molecule_arrives_with` assert to
/// the bit.
///
/// The **atom** column is the load-bearing one: the ATD engine labels atoms *into*
/// [`keys::TYPE`], so a molecule arriving with an incompatible column there (LAMMPS
/// integer atom-type ids, say) would refuse the write. The **bond** column is
/// stripped as defence in depth only: perception keeps its perceived bond types
/// in their own [`BCC_BOND_TYPE`](molrs::core::keys::BCC_BOND_TYPE) prop and
/// neither reads nor writes a bond's `type`, so nothing downstream can be steered by
/// a caller's bond labels even when they are left in place.
///
/// Shared by every model that types a molecule for itself ([`BccModel`] against
/// `ATOMTYPE_BCC.DEF`, [`GasteigerModel`] against `ATOMTYPE_GAS.DEF`). One copy: a
/// second one would be a second chance to forget the bond column.
///
/// [`BccModel`]: super::BccModel
/// [`GasteigerModel`]: super::GasteigerModel
///
/// # Errors
///
/// [`ChargeError::Malformed`] when a column cannot be cleared.
pub(super) fn without_type_columns(mol: &Atomistic) -> Result<Atomistic, ChargeError> {
    let mut work = mol.clone();
    let malformed = |e: molrs::core::MolRsError| ChargeError::Malformed {
        detail: e.to_string(),
    };

    for aid in atom_ids(mol) {
        work.clear_atom(aid, keys::TYPE).map_err(malformed)?;
    }
    let bond_ids: Vec<_> = work.bond_ids().collect();
    for bid in bond_ids {
        work.clear_bond(bid, keys::TYPE).map_err(malformed)?;
    }
    Ok(work)
}

/// Refuse a molecule whose atoms the `.DEF` table could not name.
///
/// Every `ATOMTYPE_*.DEF` ends with an unconditional catch-all row that labels an
/// unmatched atom [`DUMMY_TYPE`], so the engine returns a type for *every* atom and
/// "no rule matched" arrives as a label rather than an error. For a charge model that
/// label is a refusal: no parameter table has a `DU` row.
///
/// A bonded `DU` atom would fail anyway, at the parameter lookup. A **bondless** one
/// would not — nothing looks its type up — so a bare sulfur (the one element whose
/// BCC rules all require at least one bond) would come back with its charge silently
/// left alone, which is exactly the plausible-looking answer molrs's
/// no-fallback-values rule exists to prevent.
///
/// # Arguments
///
/// * `perceived` — the molecule the types were read off, in graph atom order.
/// * `types` — one type per atom.
/// * `table` — the `.DEF` file that was walked, for the message.
///
/// # Errors
///
/// [`ChargeError::MissingAtomType`], naming the first untyped atom.
pub(super) fn reject_dummy_types(
    perceived: &Atomistic,
    types: &[&str],
    table: &'static str,
) -> Result<(), ChargeError> {
    let Some((atom, _)) = types.iter().enumerate().find(|(_, ty)| **ty == DUMMY_TYPE) else {
        return Ok(());
    };
    Err(ChargeError::MissingAtomType {
        table,
        atom,
        atomic_number: atomic_number_of(perceived, atom)?,
    })
}

/// The atomic number of the atom at index `atom` in graph atom order, read from
/// [`keys::ATOMIC_NUMBER`] — for an error that names the atom.
///
/// # Errors
///
/// [`ChargeError::Malformed`] when no atom sits at that index or it carries no
/// valid atomic number: the graph is then the defect to report.
pub(super) fn atomic_number_of(mol: &Atomistic, atom: usize) -> Result<u8, ChargeError> {
    atom_ids(mol)
        .get(atom)
        .and_then(|aid| mol.get_atom(*aid).ok())
        .and_then(|a| a.atomic_number())
        .ok_or_else(|| ChargeError::Malformed {
            detail: format!("atom {atom} has no valid '{}'", keys::ATOMIC_NUMBER),
        })
}

/// An atom-typing failure, as a charge failure.
pub(super) fn charge_error(err: AtdError) -> ChargeError {
    match err {
        AtdError::MissingAtomType {
            table,
            atom,
            atomic_number,
        } => ChargeError::MissingAtomType {
            table,
            atom,
            atomic_number,
        },
        AtdError::Malformed { detail } => ChargeError::Malformed { detail },
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::props;
    use molrs::core::Element;
    use molrs::core::keys;

    /// Methanol's three methyl hydrogens, which `sqm` splits and `-eq 1` merges.
    fn methanol() -> Atomistic {
        let mut mol = Atomistic::new();
        let c = mol
            .add_atom(props! {
                keys::ATOMIC_NUMBER => Element::C,
                keys::X => 0.0,
                keys::Y => 0.0,
                keys::Z => 0.0,
            })
            .unwrap();
        let o = mol
            .add_atom(props! {
                keys::ATOMIC_NUMBER => Element::O,
                keys::X => 1.43,
                keys::Y => 0.0,
                keys::Z => 0.0,
            })
            .unwrap();
        let h1 = mol
            .add_atom(props! {
                keys::ATOMIC_NUMBER => Element::H,
                keys::X => -0.63,
                keys::Y => 0.63,
                keys::Z => 0.63,
            })
            .unwrap();
        let h2 = mol
            .add_atom(props! {
                keys::ATOMIC_NUMBER => Element::H,
                keys::X => -0.63,
                keys::Y => -0.63,
                keys::Z => 0.63,
            })
            .unwrap();
        let h3 = mol
            .add_atom(props! {
                keys::ATOMIC_NUMBER => Element::H,
                keys::X => -0.63,
                keys::Y => 0.0,
                keys::Z => -0.89,
            })
            .unwrap();
        let ho = mol
            .add_atom(props! {
                keys::ATOMIC_NUMBER => Element::H,
                keys::X => 1.76,
                keys::Y => 0.89,
                keys::Z => 0.0,
            })
            .unwrap();
        for (a, b) in [(c, o), (c, h1), (c, h2), (c, h3), (o, ho)] {
            mol.add_bond(a, b, props! {}).expect("add bond");
        }
        mol
    }

    /// The class-mean lands on the three methyl hydrogens and nowhere else.
    ///
    /// The hydroxyl H is in a class of its own (it is bonded to O, not C), so it
    /// must come back bit-for-bit; the methyl three become their mean.
    #[test]
    fn equivalence_average_merges_the_methyl_hydrogens_only() {
        let raw = [-0.073, -0.326, 0.053, 0.098, 0.053, 0.195];
        let got = equivalence_average(&methanol(), &raw);

        let mean: f64 = (0.053 + 0.098 + 0.053) / 3.0;
        for (k, q) in got.iter().enumerate().take(5).skip(2) {
            assert_eq!(q.to_bits(), mean.to_bits(), "methyl H {k}");
        }
        assert_eq!(
            got[5].to_bits(),
            raw[5].to_bits(),
            "the hydroxyl H is alone"
        );
        assert_eq!(got[0].to_bits(), raw[0].to_bits(), "carbon is alone");
    }

    /// A charge slice that is not one-per-atom is refused with both counts.
    #[test]
    fn check_count_reports_both_counts() {
        let err = check_count(&methanol(), &[0.0; 3]).expect_err("6 atoms, 3 charges");
        assert_eq!(
            err,
            ChargeError::ChargeCountMismatch {
                expected: 6,
                got: 3
            }
        );
    }
}
