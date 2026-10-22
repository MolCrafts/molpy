//! [`BccModel`] — AM1-BCC and ABCG2, as one model over two parameter sets.

use std::collections::HashMap;

use molrs::core::{Atomistic, NodeId};

use crate::ff::params::{BccAlias, BccCorrectionRow};
use crate::ff::typifier::atd::antechamber_bond_type;
use crate::ff::typifier::{AtdParameterSet, AtdTypifier};

use super::error::ChargeError;
use super::model::{
    ChargeModel, charge_error, check_count, equivalence_average, reject_dummy_types,
    without_type_columns,
};

/// AM1-BCC correction-family selector.
///
/// The two variants are exactly the two `BCCPARM*.DAT` files that exist. This is
/// a **narrower** axis than [`AtdParameterSet`]: every correction family names an
/// atom-type table (via [`BccParameterSet::atd_set`]), but not every atom-type
/// table has a correction family — `ATOMTYPE_GAS.DEF` has no `BCCPARM_GAS.DAT`,
/// so GAS is reachable through [`AtdTypifier`] and cannot be a variant here.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum BccParameterSet {
    /// Original AM1-BCC corrections (`BCCPARM.DAT`, `-c bcc`).
    Bcc,
    /// ABCG2 corrections (`BCCPARM_ABCG2.DAT`, `-c abcg2`).
    Abcg2,
}

impl BccParameterSet {
    /// Every correction family. Two, not seven: `BCCPARM.DAT` and
    /// `BCCPARM_ABCG2.DAT` are the only ones that exist. `ATOMTYPE_GAS.DEF`
    /// is a set of atom *types* with no correction table, and `gaff` is an
    /// atom-type table too — naming either would be naming a table that
    /// cannot correct a bond.
    pub const ALL: [BccParameterSet; 2] = [Self::Bcc, Self::Abcg2];

    /// The antechamber `-c` flag naming this family: `"bcc"` or `"abcg2"`.
    pub fn name(self) -> &'static str {
        match self {
            Self::Bcc => "bcc",
            Self::Abcg2 => "abcg2",
        }
    }

    /// The family an antechamber `-c` flag names — the inverse of
    /// [`name`](Self::name).
    ///
    /// # Errors
    ///
    /// An unknown name, including the atom-type table names (`"gaff"`) a
    /// caller might reasonably confuse for one. Never a fallback: a correction
    /// row is keyed on atom types, so the wrong family silently looks up the
    /// wrong rows, which is indistinguishable from correct output.
    pub fn from_name(name: &str) -> Result<Self, String> {
        Self::ALL
            .into_iter()
            .find(|set| set.name() == name)
            .ok_or_else(|| {
                let known: Vec<&str> = Self::ALL.iter().map(|set| set.name()).collect();
                format!(
                    "unknown BCC correction family {name:?}; expected one of {} \
                 (BCCPARM.DAT, BCCPARM_ABCG2.DAT)",
                    known.join(", ")
                )
            })
    }

    /// The set's bond charge corrections, as compile-time table data.
    fn corrections(self) -> &'static [BccCorrectionRow] {
        match self {
            Self::Bcc => crate::ff::params::BCC_CORRECTIONS,
            Self::Abcg2 => crate::ff::params::ABCG2_CORRECTIONS,
        }
    }

    /// The set's `CORR` alias rows.
    fn aliases(self) -> &'static [BccAlias] {
        match self {
            Self::Bcc => crate::ff::params::BCC_ALIASES,
            Self::Abcg2 => crate::ff::params::ABCG2_ALIASES,
        }
    }

    /// The atom-type table this correction family is defined against.
    ///
    /// A correction row is keyed on atom types, so a correction family is only
    /// meaningful next to the table those types come from: `BCCPARM.DAT` rows
    /// speak `ATOMTYPE_BCC.DEF`, `BCCPARM_ABCG2.DAT` rows speak
    /// `ATOMTYPE_ABCG2.DEF`. Pairing them any other way silently looks up the
    /// wrong rows.
    ///
    /// # Returns
    ///
    /// The [`AtdParameterSet`] whose atom types this family's rows are written in.
    pub fn atd_set(self) -> AtdParameterSet {
        match self {
            Self::Bcc => AtdParameterSet::Bcc,
            Self::Abcg2 => AtdParameterSet::Abcg2,
        }
    }
}

/// The AM1-BCC family: QM base charges plus bond charge corrections.
///
/// One engine, two parameter sets — [`BccParameterSet::Bcc`] reads `BCCPARM.DAT`
/// against `ATOMTYPE_BCC.DEF`, [`BccParameterSet::Abcg2`] reads `BCCPARM_ABCG2.DAT`
/// against `ATOMTYPE_ABCG2.DEF` — and no special case for either.
///
/// ```text
/// QM charges ──▶ equivalence classes ──▶ class-mean ──▶ + BCCPARM increments ──▶ q
/// ```
///
/// # The `type` column is the caller's
///
/// The model perceives its BCC atom and bond types into local `Vec`s and never writes
/// them into `mol` — nor reads the ones `mol` arrives with. Both halves matter:
///
/// * **write** — the standard AM1-BCC workflow needs GAFF types *and* BCC charges at
///   the same time (GAFF for LJ and bonded terms, BCC for the electrostatics). A
///   charge pass that relabelled `c3` as `11` would destroy the force field it was
///   meant to complete;
/// * **read** — a molecule read from a LAMMPS data file carries integer *bond type
///   ids* in the very prop the BCC bond type lives in. Re-interpreting `1, 2, 3` as
///   BCC bond types would split acetate's two carboxylate oxygens by ~0.2 e without a
///   word. So the working copy is stripped of both `type` columns before perception.
///
/// # Examples
///
/// ```
/// use molrs::core::{Atomistic, Element, keys};
/// use molrs::props;
/// use molrs::ff::charge::{BccModel, BccParameterSet};
///
/// # fn main() -> Result<(), Box<dyn std::error::Error>> {
/// let mut mol = Atomistic::new();
/// let c = mol.add_atom(props! {
///     keys::ATOMIC_NUMBER => Element::C,
///     keys::X => 0.0,
///     keys::Y => 0.0,
///     keys::Z => 0.0,
/// })?;
/// for [x, y, z] in [
///     [0.63, 0.63, 0.63],
///     [-0.63, -0.63, 0.63],
///     [-0.63, 0.63, -0.63],
///     [0.63, -0.63, -0.63],
/// ] {
///     let h = mol.add_atom(props! {
///         keys::ATOMIC_NUMBER => Element::H,
///         keys::X => x,
///         keys::Y => y,
///         keys::Z => z,
///     })?;
///     mol.add_bond(c, h, props! {})?;
/// }
///
/// // The AM1 charges come from an AM1 solver; here, a zero base, so that what
/// // comes back is the BCC increments themselves: C–H is +0.0393 e per bond.
/// let q = BccModel::new(BccParameterSet::Bcc).correct(&mol, &[0.0; 5])?;
/// assert!((q[0] - 4.0 * 0.0393).abs() < 1e-12);
/// assert!((q[1] + 0.0393).abs() < 1e-12);
/// # Ok(())
/// # }
/// ```
#[derive(Debug, Clone)]
pub struct BccModel {
    set: BccParameterSet,
    table: CorrectionTable,
}

impl BccModel {
    /// The model for one correction family: its correction rows and `CORR`
    /// aliases, from the compiled tables.
    ///
    /// The parameter set is an argument because there is no default one: the two that
    /// exist are `BCCPARM.DAT` and `BCCPARM_ABCG2.DAT`.
    ///
    /// # Arguments
    ///
    /// * `set` — the correction family, which also names the atom-type table its rows
    ///   are written in (see [`BccParameterSet::atd_set`]).
    pub fn new(set: BccParameterSet) -> Self {
        Self {
            set,
            table: CorrectionTable::of(set),
        }
    }

    /// The correction family this model applies.
    ///
    /// # Returns
    ///
    /// The [`BccParameterSet`] it was built with.
    pub fn parameter_set(&self) -> BccParameterSet {
        self.set
    }

    /// Correct already-equivalenced base charges with the bond charge increments.
    ///
    /// The pure BCC stage, and the whole push API: molecule in behind a shared
    /// reference, base charges in as a slice, corrected charges out. Nothing is
    /// written back — not the charges, and above all not the BCC atom types, which
    /// the model perceives for itself and keeps to itself.
    ///
    /// The base charges are expected to be **already averaged over the equivalence
    /// classes** (antechamber's `-eq 1`), because that is what the BCC stage consumes;
    /// [`ChargeModel::assign`] is the door that does the averaging for you.
    ///
    /// # Arguments
    ///
    /// * `mol` — the molecule; left untouched, `type` columns included.
    /// * `am1` — one base charge per atom, in graph atom order.
    ///
    /// # Returns
    ///
    /// `am1[i] + Σ increments` for every atom, in graph atom order. The increments are
    /// pairwise antisymmetric, so the total charge is conserved — including the AM1
    /// rounding residual, which is **carried through, never normalized away**:
    /// `am1bcc.c` ends at the increment loop, and matching antechamber means carrying
    /// its residual rather than removing it.
    ///
    /// # Errors
    ///
    /// [`ChargeError::ChargeCountMismatch`] when `am1` is not one charge per atom;
    /// [`ChargeError::MissingAtomType`] when the table cannot type an atom;
    /// [`ChargeError::MissingCorrection`] when it has no row for a bond.
    pub fn correct(&self, mol: &Atomistic, am1: &[f64]) -> Result<Vec<f64>, ChargeError> {
        check_count(mol, am1)?;

        let mut perceived = without_type_columns(mol)?;
        let atd = self.set.atd_set();
        let typifier = AtdTypifier::new(atd);
        typifier
            .perceive_bond_types(&mut perceived)
            .map_err(|e| ChargeError::Malformed {
                detail: e.to_string(),
            })?;
        let types = typifier.types_of(&perceived).map_err(charge_error)?;
        reject_dummy_types(&perceived, &types, atd.table().name)?;

        let delta = bcc_increments(&self.table, &perceived, &types).map_err(charge_error_row)?;

        Ok(am1.iter().zip(delta).map(|(q, d)| q + d).collect())
    }
}

impl ChargeModel for BccModel {
    fn needs_equivalencing(&self) -> bool {
        // antechamber's `-eq 1`, the default for `-c bcc` and `-c abcg2`.
        true
    }

    fn assign(&self, mol: &Atomistic, qm: Option<&[f64]>) -> Result<Vec<f64>, ChargeError> {
        let qm = qm.ok_or(ChargeError::MissingQmCharges { model: "AM1-BCC" })?;
        check_count(mol, qm)?;

        let base = if self.needs_equivalencing() {
            equivalence_average(mol, qm)
        } else {
            qm.to_vec()
        };
        self.correct(mol, &base)
    }
}

/// A correction-lookup failure, as a charge failure.
fn charge_error_row(err: BccIncrementError) -> ChargeError {
    match err {
        BccIncrementError::MissingRow {
            left,
            right,
            bond_type,
        } => ChargeError::MissingCorrection {
            left,
            right,
            bond_type,
        },
        BccIncrementError::Malformed { detail } => ChargeError::Malformed { detail },
    }
}

/// Oriented BCC correction table.
///
/// A row `(left, right, bond_type, delta)` means a bond typed
/// `left|right|bond_type` adds `+delta` to the left atom and `-delta` to the
/// right atom. A reversed bond applies the same magnitude with reversed sign.
#[derive(Debug, Clone)]
struct CorrectionTable {
    corrections: HashMap<(String, String, i32), f64>,
    aliases: HashMap<String, String>,
}

impl CorrectionTable {
    /// The correction rows and `CORR` aliases of a parameter set.
    fn of(set: BccParameterSet) -> Self {
        Self {
            corrections: set
                .corrections()
                .iter()
                .map(|row| {
                    (
                        (row.left.to_owned(), row.right.to_owned(), row.bond_type),
                        row.delta,
                    )
                })
                .collect(),
            aliases: set
                .aliases()
                .iter()
                .map(|a| (a.atom_type.to_owned(), a.reference.to_owned()))
                .collect(),
        }
    }

    /// The increment for a typed bond, following `CORR` aliases.
    ///
    /// # Arguments
    ///
    /// * `left` / `right` — the endpoints' BCC atom types.
    /// * `bond_type` — the perceived antechamber bond type.
    ///
    /// # Returns
    ///
    /// The increment to add to `left` and subtract from `right`, or `None` when no
    /// row (and no aliased row) covers the bond.
    fn correction(&self, left: &str, right: &str, bond_type: i32) -> Option<f64> {
        if let Some(v) = self.direct_correction(left, right, bond_type) {
            return Some(v);
        }

        let left_alias = self.aliases.get(left).map(String::as_str);
        let right_alias = self.aliases.get(right).map(String::as_str);

        if let Some(alias) = left_alias
            && let Some(v) = self.direct_correction(alias, right, bond_type)
        {
            return Some(v);
        }
        if let Some(alias) = right_alias
            && let Some(v) = self.direct_correction(left, alias, bond_type)
        {
            return Some(v);
        }
        if let (Some(left_alias), Some(right_alias)) = (left_alias, right_alias)
            && let Some(v) = self.direct_correction(left_alias, right_alias, bond_type)
        {
            return Some(v);
        }
        None
    }

    fn direct_correction(&self, left: &str, right: &str, bond_type: i32) -> Option<f64> {
        self.corrections
            .get(&(left.to_owned(), right.to_owned(), bond_type))
            .copied()
            .or_else(|| {
                self.corrections
                    .get(&(right.to_owned(), left.to_owned(), bond_type))
                    .map(|v| -*v)
            })
    }
}

/// Why a correction pass could not be applied.
///
/// Kept typed rather than a message so that the charge model can hand the C++ and
/// Python bridges a `ChargeError` they can discriminate on: a bond with no row is a
/// permanent property of the parameter set, a malformed graph is the caller's.
#[derive(Debug)]
enum BccIncrementError {
    /// No row (and no aliased row) covers this bond.
    MissingRow {
        /// BCC atom type of one endpoint.
        left: String,
        /// BCC atom type of the other endpoint.
        right: String,
        /// The perceived antechamber bond type.
        bond_type: i32,
    },
    /// The graph could not be read.
    Malformed {
        /// What the graph layer said.
        detail: String,
    },
}

/// The per-atom BCC increment of every atom, in graph atom order.
///
/// The one place the corrections are applied. Every row is added to one endpoint and
/// subtracted from the other, so the increments sum to zero and the pass conserves
/// the molecule's total charge exactly.
///
/// The atom types are an **argument**, not something read off `mol`: the charge model
/// perceives its BCC types into a `Vec` and never writes them into the caller's
/// graph.
///
/// # Arguments
///
/// * `table` — the correction rows to apply.
/// * `mol` — the molecule, whose bonds carry perceived antechamber bond types.
/// * `types` — the BCC atom type of every atom, in graph atom order.
///
/// # Returns
///
/// The increment for every atom, in graph atom order.
///
/// # Errors
///
/// [`BccIncrementError::MissingRow`] for a bond the table does not cover;
/// [`BccIncrementError::Malformed`] for a bond with no usable `type`.
fn bcc_increments(
    table: &CorrectionTable,
    mol: &Atomistic,
    types: &[&str],
) -> Result<Vec<f64>, BccIncrementError> {
    let atom_ids: Vec<NodeId> = mol.atoms().map(|(id, _)| id).collect();
    let index: HashMap<NodeId, usize> = atom_ids
        .iter()
        .enumerate()
        .map(|(i, aid)| (*aid, i))
        .collect();

    let mut delta = vec![0.0; atom_ids.len()];
    for (bid, bond) in mol.bonds() {
        let (Some(&i), Some(&j)) = (index.get(&bond.nodes[0]), index.get(&bond.nodes[1])) else {
            return Err(BccIncrementError::Malformed {
                detail: format!("bond {bid:?} has an endpoint that is not an atom of the molecule"),
            });
        };
        let bond_type = antechamber_bond_type(&bond)
            .map_err(|detail| BccIncrementError::Malformed { detail })?;
        let Some(dq) = table.correction(types[i], types[j], bond_type) else {
            return Err(BccIncrementError::MissingRow {
                left: types[i].to_owned(),
                right: types[j].to_owned(),
                bond_type,
            });
        };
        delta[i] += dq;
        delta[j] -= dq;
    }
    Ok(delta)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::props;
    use molrs::core::Element;

    use molrs::core::keys;

    #[test]
    fn a_family_name_round_trips_and_an_atom_type_table_is_refused() {
        for set in BccParameterSet::ALL {
            assert_eq!(BccParameterSet::from_name(set.name()), Ok(set));
        }
        for wrong in ["gaff", "gas", "BCC", ""] {
            assert!(BccParameterSet::from_name(wrong).is_err(), "{wrong:?}");
        }
    }

    /// Methane, untyped — the molecule a user has.
    fn methane() -> Atomistic {
        let mut mol = Atomistic::new();
        let c = mol
            .add_atom(props! {
                keys::ATOMIC_NUMBER => Element::C,
                keys::X => 0.0,
                keys::Y => 0.0,
                keys::Z => 0.0,
            })
            .unwrap();
        for [x, y, z] in [
            [0.63, 0.63, 0.63],
            [-0.63, -0.63, 0.63],
            [-0.63, 0.63, -0.63],
            [0.63, -0.63, -0.63],
        ] {
            let h = mol
                .add_atom(props! {
                    keys::ATOMIC_NUMBER => Element::H,
                    keys::X => x,
                    keys::Y => y,
                    keys::Z => z,
                })
                .unwrap();
            mol.add_bond(c, h, props! {}).expect("add C-H");
        }
        mol
    }

    /// The molecule comes back untouched: `correct` takes `&Atomistic` and writes
    /// nothing, not even the charges it computed.
    #[test]
    fn correct_writes_nothing_into_the_molecule() {
        let mol = methane();
        let model = BccModel::new(BccParameterSet::Bcc);
        let q = model.correct(&mol, &[0.0; 5]).expect("correct methane");

        assert!((q[0] - 4.0 * 0.0393).abs() < 1e-12, "{}", q[0]);
        for (_, atom) in mol.atoms() {
            assert_eq!(atom.get_str(keys::TYPE), None, "no BCC code was written");
            assert_eq!(atom.get_f64(keys::CHARGE), None, "no charge was written");
        }
        for (_, bond) in mol.bonds() {
            assert!(
                !bond.props.contains_key(keys::TYPE),
                "no BCC bond type was written"
            );
        }
    }

    /// The increments cancel: what goes in as a total comes out as a total.
    #[test]
    fn the_increments_conserve_the_total_charge() {
        let am1 = [-0.266, 0.066, 0.066, 0.066, 0.066];
        let q = BccModel::new(BccParameterSet::Bcc)
            .correct(&methane(), &am1)
            .expect("correct methane");

        let before: f64 = am1.iter().sum();
        let after: f64 = q.iter().sum();
        assert!((after - before).abs() < 1e-14, "{before} -> {after}");
    }

    /// A molecule read from a frame that spells its species only as
    /// `atomic_number` — no `element` column anywhere — is typed and corrected
    /// exactly like the hand-built one: the typing path reads Z, never a symbol.
    #[test]
    fn a_frame_carrying_only_atomic_numbers_is_typed_and_corrected() {
        let frame = methane().to_frame().expect("methane converts");
        let atoms = frame.get("atoms").expect("an atoms block");
        assert!(atoms.contains_key(keys::ATOMIC_NUMBER));
        assert!(
            !atoms.contains_key(keys::ELEMENT),
            "the frame spells no symbols"
        );
        let mol = Atomistic::from_frame(&frame).expect("a Z-only frame promotes");

        let model = BccModel::new(BccParameterSet::Bcc);
        let am1 = [-0.266, 0.066, 0.066, 0.066, 0.066];
        let q = model
            .assign(&mol, Some(&am1))
            .expect("AM1-BCC types the molecule");

        assert_eq!(q, model.assign(&methane(), Some(&am1)).unwrap());
        assert!((q[0] - (-0.266 + 4.0 * 0.0393)).abs() < 1e-12, "{}", q[0]);
    }

    /// `assign(None)` refuses; it does not invent a base to correct.
    #[test]
    fn assign_without_qm_charges_is_an_error() {
        let model = BccModel::new(BccParameterSet::Bcc);
        assert_eq!(
            model.assign(&methane(), None),
            Err(ChargeError::MissingQmCharges { model: "AM1-BCC" })
        );
    }
}
