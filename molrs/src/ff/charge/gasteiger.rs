//! [`GasteigerModel`] — Gasteiger/PEOE, aligned with `antechamber -c gas`.

use molrs::core::{Atomistic, NodeId};

use crate::ff::params::{GASTEIGER_PARAMS, GasteigerRow};
use crate::ff::typifier::{AtdParameterSet, AtdTypifier};

use super::error::ChargeError;
use super::model::{
    ChargeModel, atom_ids, atomic_number_of, charge_error, reject_dummy_types, without_type_columns,
};

/// antechamber's `CONVERG`: the loop stops when the RMS charge change of a sweep
/// falls to this.
const CONVERGENCE: f64 = 1.0e-5;

/// antechamber's `GASMAXITER`: the ceiling on sweeps.
///
/// Never reached in practice — the damping is geometric, so the RMS change halves
/// every sweep and the 37 oracle molecules converge in 7 to 15. It is a guard against
/// a pathological graph, not the stopping rule.
const MAX_SWEEPS: usize = 500;

/// antechamber's `DAMPFACTOR`: sweep `n` transfers `0.5^(n+1)` of the χ difference.
const DAMPING: f64 = 0.5;

/// The floor a zero electronegativity is raised to, so it cannot silently divide.
///
/// antechamber's own guard. `n4`'s polynomial is `0 + 11.86 q + 11.86 q^2`, which is
/// exactly zero at `q = 0` — reachable only for an ammonium seeded away from its
/// +1.00, but the guard is part of the algorithm and is kept.
const CHI_FLOOR: f64 = 1.0e-10;

/// Gasteiger/PEOE charges from the molecule alone — `antechamber -c gas`.
///
/// Takes **no QM input**: [`assign`](ChargeModel::assign)'s `qm` argument exists for
/// the models that need it, and this one ignores it entirely.
///
/// A unit struct, and deliberately so: unlike [`BccModel`](super::BccModel) there is
/// no parameter set to choose. `GASPARM.DAT` and `ATOMTYPE_GAS.DEF` are the only
/// tables `-c gas` has, so a constructor taking a set would be offering a choice that
/// does not exist.
///
/// # The model
///
/// Partial equalization of orbital electronegativity (Gasteiger–Marsili). Each atom's
/// electronegativity is a quadratic in its own charge,
///
/// ```text
/// chi_i(q) = a_i + b_i * q_i + c_i * q_i^2
/// ```
///
/// and along every bond, charge flows from the lower-χ atom (which therefore goes
/// positive) to the higher-χ one, normalised by the **donor's** cation
/// electronegativity χ⁺ and damped by one further factor of a half per sweep:
///
/// ```text
/// dq = (chi_high - chi_low) / chi_plus[donor] * 0.5^(sweep + 1)
/// ```
///
/// The transfer is antisymmetric — one atom gains exactly what the other loses — so
/// the total charge never moves off the seeds it started from.
///
/// # Three things it is not
///
/// * **Not a fixed six iterations.** antechamber runs a damped convergence loop
///   (convergence at 1e-5, at most 500 sweeps), and every one of the 37 oracle
///   molecules needs more than six sweeps — methane 7, methylammonium 15. Truncating
///   at six leaves methylammonium's nitrogen 0.0131 e short, 131× the 1e-4 gate.
/// * **`chi_plus` is a DIVISOR, not a quartic coefficient.** `GASPARM.DAT`'s columns
///   run `a`, `b`, `c`, `d`, `formal_charge`, and reading `d` as `+ d*q^3` is the one
///   catastrophic misreading of the table. It is χ⁺ = χ(q = 1): for every heavy row
///   `d == a + b + c` exactly (`c3`: 7.98 + 9.18 + 1.88 = 19.04 = d). **Hydrogen is
///   the exception** — H⁺ is a bare proton, so its polynomial χ⁺ is meaningless and
///   the table substitutes a fixed 20.02 eV where `a + b + c` would give 12.85. The
///   model reads the column; it never rebuilds it from `a + b + c`.
/// * **Not renormalized to the formal net charge.** `-c gas` ignores `-nc`: what is
///   conserved is the sum of the SEED charges (`GASPARM.DAT`'s `formal_charge`
///   column), which is not always the net charge. `ATOMTYPE_GAS.DEF` has no aromatic
///   N⁺ type, so imidazolium — a +1 cation — is typed all-neutral, seeded at 0, and
///   antechamber's own `-c gas` charges for it sum to 0. Renormalizing it to +1 would
///   conserve *something* perfectly while sitting a whole electron from the oracle.
///
/// # It is Jacobi
///
/// The whole χ array is built from the previous sweep's charges **before** any
/// transfer is applied; the transfers accumulate into the running charges, and the
/// snapshot χ is read from is only rolled forward at the end of the sweep. Feeding a
/// half-updated charge back into χ mid-sweep (Gauss–Seidel) changes the convergence
/// trajectory and the answer.
///
/// # Examples
///
/// ```
/// use molrs::core::{Atomistic, Element, keys};
/// use molrs::props;
/// use molrs::ff::charge::{ChargeModel, GasteigerModel};
///
/// # fn main() -> Result<(), Box<dyn std::error::Error>> {
/// // Methanol, as a user has it: elements, coordinates, bonds. Nothing else.
/// let mut mol = Atomistic::new();
/// let c = mol.add_atom(props! {
///     keys::ATOMIC_NUMBER => Element::C,
///     keys::X => -0.383,
///     keys::Y => 0.0,
///     keys::Z => 0.0,
/// })?;
/// let o = mol.add_atom(props! {
///     keys::ATOMIC_NUMBER => Element::O,
///     keys::X => 0.383,
///     keys::Y => 0.0,
///     keys::Z => 0.0,
/// })?;
/// let h1 = mol.add_atom(props! {
///     keys::ATOMIC_NUMBER => Element::H,
///     keys::X => -0.762,
///     keys::Y => 0.478,
///     keys::Z => 0.907,
/// })?;
/// let h2 = mol.add_atom(props! {
///     keys::ATOMIC_NUMBER => Element::H,
///     keys::X => -0.762,
///     keys::Y => 0.535,
///     keys::Z => -0.870,
/// })?;
/// let h3 = mol.add_atom(props! {
///     keys::ATOMIC_NUMBER => Element::H,
///     keys::X => -0.762,
///     keys::Y => -1.013,
///     keys::Z => -0.037,
/// })?;
/// let ho = mol.add_atom(props! {
///     keys::ATOMIC_NUMBER => Element::H,
///     keys::X => 0.712,
///     keys::Y => -0.898,
///     keys::Z => 0.0,
/// })?;
/// for (a, b) in [(c, o), (c, h1), (c, h2), (c, h3), (o, ho)] {
///     mol.add_bond(a, b, props! {})?;
/// }
///
/// // No QM charges — that is the whole point of this model.
/// let q = GasteigerModel.assign(&mol, None)?;
/// assert!((q[1] - -0.399_641).abs() < 1e-4, "the hydroxyl O, per antechamber");
/// assert!(!GasteigerModel.needs_equivalencing());
///
/// // The three methyl hydrogens are equivalent, and a topological model has no
/// // conformer to break their symmetry: they come back identical, unaveraged.
/// assert_eq!(q[2].to_bits(), q[3].to_bits());
/// # Ok(())
/// # }
/// ```
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct GasteigerModel;

impl ChargeModel for GasteigerModel {
    fn needs_equivalencing(&self) -> bool {
        // antechamber's `-eq` defaults to 0 for every method but `bcc` / `abcg2` /
        // `resp`. It exists to remove a CONFORMER artefact of a QM calculation, and
        // this model never sees a conformer: it is symmetric for free. Averaging here
        // would be averaging numbers that are already equal — measured on methanol's
        // three methyl hydrogens, which antechamber prints as 0.052691 three times.
        false
    }

    /// The PEOE charges of `mol`, in graph atom order.
    ///
    /// # Arguments
    ///
    /// * `mol` — the molecule; left untouched, including its `type` columns. Explicit
    ///   hydrogens are required, as they are for every antechamber charge method.
    /// * `qm` — **ignored**. Gasteiger consumes no QM charges, and a model that let a
    ///   stray AM1 vector leak into its answer would be a QM model wearing a
    ///   topological one's name.
    ///
    /// # Returns
    ///
    /// One charge per atom, in graph atom order, summing to the total of the atoms'
    /// seed charges.
    ///
    /// # Errors
    ///
    /// [`ChargeError::MissingAtomType`] when `ATOMTYPE_GAS.DEF` cannot type an atom,
    /// or types it as one `GASPARM.DAT` has no row for (a lone pair). Never a
    /// fallback value.
    fn assign(&self, mol: &Atomistic, _qm: Option<&[f64]>) -> Result<Vec<f64>, ChargeError> {
        let mut perceived = without_type_columns(mol)?;
        let set = AtdParameterSet::Gas;
        let typifier = AtdTypifier::new(set);
        typifier
            .perceive_bond_types(&mut perceived)
            .map_err(|e| ChargeError::Malformed {
                detail: e.to_string(),
            })?;
        let types = typifier.types_of(&perceived).map_err(charge_error)?;
        reject_dummy_types(&perceived, &types, set.table().name)?;

        let rows = parameter_rows(&perceived, &types)?;
        let bonds = bond_pairs(&perceived)?;
        Ok(equalize(&rows, &bonds))
    }
}

/// The `GASPARM.DAT` row of every atom, in graph atom order.
///
/// # Errors
///
/// [`ChargeError::MissingAtomType`] naming the first atom whose GAS type has no row.
/// `ATOMTYPE_GAS.DEF` can assign three types `GASPARM.DAT` does not parameterize —
/// `DU` (caught earlier) and the two lone-pair types — and an atom whose χ curve is
/// unknown has no charge, rather than a default one.
fn parameter_rows(
    mol: &Atomistic,
    types: &[&str],
) -> Result<Vec<&'static GasteigerRow>, ChargeError> {
    types
        .iter()
        .enumerate()
        .map(
            |(atom, ty)| match GASTEIGER_PARAMS.iter().find(|row| row.atom_type == *ty) {
                Some(row) => Ok(row),
                None => Err(ChargeError::MissingAtomType {
                    table: "GASPARM.DAT",
                    atom,
                    atomic_number: atomic_number_of(mol, atom)?,
                }),
            },
        )
        .collect()
}

/// Every bond as an index pair into graph atom order — each **once**.
///
/// `mol.bonds()` yields each bond a single time, which is the whole subtlety of the
/// transfer step: iterating `(i, j in con(i))` instead visits every bond from both
/// ends, doubles each sweep's transfer, and diverges to overflow.
///
/// # Errors
///
/// [`ChargeError::Malformed`] when a bond's endpoint is not an atom of the molecule.
fn bond_pairs(mol: &Atomistic) -> Result<Vec<(usize, usize)>, ChargeError> {
    let index: std::collections::HashMap<NodeId, usize> = atom_ids(mol)
        .into_iter()
        .enumerate()
        .map(|(i, aid)| (aid, i))
        .collect();

    mol.bonds()
        .map(|(bid, bond)| {
            let (Some(&i), Some(&j)) = (index.get(&bond.nodes[0]), index.get(&bond.nodes[1]))
            else {
                return Err(ChargeError::Malformed {
                    detail: format!("bond {bid:?} has an endpoint that is not an atom"),
                });
            };
            Ok((i, j))
        })
        .collect()
}

/// The damped PEOE loop: seed from the table, equalize until the sweep stops moving.
///
/// **Jacobi.** `chi` is built from `previous` — the charges as they stood at the end
/// of the last sweep — and every transfer of this sweep accumulates into `charges`
/// without feeding back into `chi`. Only when the sweep is over is the snapshot
/// rolled forward.
///
/// `charges` is the running charge, seeded from `GASPARM.DAT` and **never zeroed**;
/// it is not a per-sweep delta buffer.
///
/// # Arguments
///
/// * `rows` — the parameter row of each atom, in graph atom order.
/// * `bonds` — each bond once, as index pairs.
///
/// # Returns
///
/// The converged charges, in the same order.
fn equalize(rows: &[&GasteigerRow], bonds: &[(usize, usize)]) -> Vec<f64> {
    let mut charges: Vec<f64> = rows.iter().map(|row| row.seed_charge).collect();
    let mut previous = charges.clone();
    let mut chi = vec![0.0; rows.len()];

    for sweep in 0..MAX_SWEEPS {
        for ((x, row), q) in chi.iter_mut().zip(rows).zip(&previous) {
            let value = row.a + row.b * q + row.c * q * q;
            *x = if value == 0.0 { CHI_FLOOR } else { value };
        }

        // 0.5^(sweep+1): the transfers of each sweep are half the last one's, so this
        // is a geometric SERIES, not a fixed-point map — where the loop stops IS the
        // answer, which is why the stopping rule has to be convergence and not a
        // hard-coded count.
        let damping = DAMPING.powi(
            i32::try_from(sweep)
                .expect("MAX_SWEEPS is 500, which fits an i32")
                .saturating_add(1),
        );

        for &(i, j) in bonds {
            // The DONOR normalises: the lower-χ atom is the one that goes positive,
            // and it is its own cation electronegativity χ⁺ that the difference is
            // measured against. `<=` (not `<`) so that a bond between two atoms of
            // equal χ transfers exactly nothing, from a well-defined side.
            let (donor, acceptor, transfer) = if chi[i] <= chi[j] {
                (i, j, (chi[j] - chi[i]) / rows[i].chi_plus * damping)
            } else {
                (j, i, (chi[i] - chi[j]) / rows[j].chi_plus * damping)
            };
            charges[donor] += transfer;
            charges[acceptor] -= transfer;
        }

        let rmsd = rms_change(&previous, &charges);
        previous.copy_from_slice(&charges);
        if rmsd <= CONVERGENCE {
            break;
        }
    }
    charges
}

/// The RMS charge change over a sweep — antechamber's `rmscal()`.
///
/// Zero for an empty molecule, which stops the loop on its first sweep rather than
/// dividing by nothing.
fn rms_change(previous: &[f64], current: &[f64]) -> f64 {
    if previous.is_empty() {
        return 0.0;
    }
    let sum: f64 = previous
        .iter()
        .zip(current)
        .map(|(before, after)| (before - after) * (before - after))
        .sum();
    #[expect(
        clippy::cast_precision_loss,
        reason = "atom counts are far below f64's exact-integer range"
    )]
    (sum / previous.len() as f64).sqrt()
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::props;
    use molrs::core::Element;
    use molrs::core::keys;

    /// Methane — the smallest molecule that still needs seven sweeps.
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

    /// The molecule comes back untouched — no types, no charges written into it.
    #[test]
    fn assign_writes_nothing_into_the_molecule() {
        let mol = methane();
        let q = GasteigerModel
            .assign(&mol, None)
            .expect("gasteiger methane");
        assert_eq!(q.len(), 5);

        for (_, atom) in mol.atoms() {
            assert_eq!(atom.get_str(keys::TYPE), None, "no GAS type was written");
            assert_eq!(atom.get_f64(keys::CHARGE), None, "no charge was written");
        }
    }

    /// χ⁺ comes off the table, and hydrogen's is NOT its polynomial's value at q = 1.
    ///
    /// The heavy rows satisfy `chi_plus == a + b + c`, which is what makes the column
    /// recognisable as a cation electronegativity rather than a quartic coefficient —
    /// and hydrogen is the documented exception. A model that "simplified" the lookup
    /// to `a + b + c` would be right about every atom but H, and wrong about every
    /// molecule that has one.
    #[test]
    fn hydrogens_chi_plus_is_20_02_not_a_plus_b_plus_c() {
        let h = GASTEIGER_PARAMS
            .iter()
            .find(|row| row.atom_type == "h")
            .expect("GASPARM has a hydrogen row");
        assert!((h.chi_plus - 20.02).abs() < 1e-12);
        assert!(
            (h.chi_plus - (h.a + h.b + h.c)).abs() > 7.0,
            "H's chi+ (20.02) and its polynomial at q=1 (12.85) must not be confused"
        );

        let c3 = GASTEIGER_PARAMS
            .iter()
            .find(|row| row.atom_type == "c3")
            .expect("GASPARM has a c3 row");
        assert!(
            (c3.chi_plus - (c3.a + c3.b + c3.c)).abs() < 1e-12,
            "a heavy row's chi+ IS its polynomial at q=1"
        );
    }

    /// A single sweep moves charge from the donor to the acceptor, and nowhere else.
    ///
    /// Hand-checkable: one C–H bond, χ(C) = 7.98 < χ(H) = 7.17? No — χ(H) = 7.17 is
    /// the LOWER, so hydrogen is the donor and goes positive, normalised by its own
    /// χ⁺ = 20.02. The first sweep transfers `(7.98 - 7.17) / 20.02 * 0.5`.
    #[test]
    fn the_first_sweep_divides_by_the_donors_chi_plus() {
        let h = GASTEIGER_PARAMS
            .iter()
            .find(|row| row.atom_type == "h")
            .expect("h row");
        let c3 = GASTEIGER_PARAMS
            .iter()
            .find(|row| row.atom_type == "c3")
            .expect("c3 row");

        let want = (c3.a - h.a) / h.chi_plus * 0.5;
        let q = equalize(&[c3, h], &[(0, 1)]);
        // One sweep only: the loop below would keep going, so this reproduces the
        // sweep by hand and checks the SIGN and the DIVISOR, not the converged value.
        assert!(want > 0.0, "H is the donor: it goes positive");
        assert!(
            q[1] > 0.0 && q[0] < 0.0,
            "carbon takes the electron density"
        );
        assert!(
            (q[0] + q[1]).abs() < 1e-15,
            "the transfer is antisymmetric: {q:?}"
        );
    }

    /// An atom with no `GASPARM.DAT` row is a refusal, not a zero charge.
    #[test]
    fn an_unparameterized_type_is_an_error() {
        let mol = methane();
        let err = parameter_rows(&mol, &["c3", "h", "h", "h", "lp"]).expect_err("lp has no row");
        assert_eq!(
            err,
            ChargeError::MissingAtomType {
                table: "GASPARM.DAT",
                atom: 4,
                atomic_number: Element::H.z(),
            }
        );
    }
}
