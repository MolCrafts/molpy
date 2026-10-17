//! MMFF94 bond stretching: E = (1/2)*143.9325*kb*dr^2*(1 + cs*dr + 7/12*cs^2*dr^2)

use crate::ff::potential::param_reads;
use molrs::core::schema::block_names::BONDS;
use ndarray::{Array2, ArrayView2};

use crate::ff::ir::Params;
use crate::ff::potential::flat_coords::{sub3, term_table, validate_coords};
use crate::ff::potential::{Accumulator, ForceTerm, IndexedTerms, Potential};
use crate::op::vec3::norm;
use molrs::core::Frame;
use molrs::op::F;

use crate::core::constants::MMFF_MDYNE_A_TO_KCAL_MOL;
/// Cubic stretch constant (A^-1).
const CS: f64 = -2.0;

pub struct BondMmff {
    atom_i: Vec<usize>,
    atom_j: Vec<usize>,
    kb: Vec<F>,
    r0: Vec<F>,
}

impl BondMmff {
    /// The physics, once. Which atoms a term names is the only thing
    /// that differs between the two entry points, so it is the only thing
    /// passed in — a second copy of the loop would be a second place for
    /// the force expression to drift.
    fn fold(
        &self,
        coords: &[F],
        out: &mut [F],
        n_terms: usize,
        atoms: impl Fn(usize) -> (usize, usize),
    ) -> F {
        let _n = validate_coords(coords);
        let mut energy: F = 0.0;
        let forces = out;
        let cs = CS as F;
        let conv = MMFF_MDYNE_A_TO_KCAL_MOL as F;

        for idx in 0..n_terms {
            let (i, j) = atoms(idx);
            let d = sub3(coords, j, coords, i);
            let r = norm(d);
            let dr = r - self.r0[idx];
            let cs2 = cs * cs;

            energy += 0.5
                * conv
                * self.kb[idx]
                * dr
                * dr
                * (1.0 + cs * dr + (7.0 / 12.0) * cs2 * dr * dr);

            if r < 1e-12 as F {
                continue;
            }
            let de_dr =
                conv * self.kb[idx] * dr * (1.0 + 1.5 * cs * dr + (7.0 / 6.0) * cs2 * dr * dr);
            let factor = -de_dr / r;
            for dim in 0..3 {
                forces[j * 3 + dim] += factor * d[dim];
                forces[i * 3 + dim] -= factor * d[dim];
            }
        }
        energy
    }
}

impl Potential for BondMmff {
    fn evaluate(&self, coords: &[F], acc: &mut Accumulator<'_>) {
        let energy = self.fold(
            coords,
            acc.forces_without_virial(),
            self.atom_i.len(),
            |t| (self.atom_i[t], self.atom_j[t]),
        );
        acc.add_energy(energy);
    }
}

impl IndexedTerms for BondMmff {
    fn terms(&self) -> Array2<u32> {
        term_table(&[&self.atom_i, &self.atom_j])
    }
    fn evaluate_terms(&self, coords: &[F], terms: ArrayView2<'_, u32>, acc: &mut Accumulator<'_>) {
        debug_assert_eq!(
            terms.nrows(),
            self.atom_i.len(),
            "the row set is the force field's; only the atoms a row names may be rebound"
        );
        let energy = self.fold(coords, acc.forces_without_virial(), terms.nrows(), |t| {
            (terms[[t, 0]] as usize, terms[[t, 1]] as usize)
        });
        acc.add_energy(energy);
    }
}

pub fn bond_mmff_constructor(
    _sp: &Params,
    _tp: &[(&str, &Params)],
    frame: &Frame,
) -> Result<ForceTerm, crate::ff::potential::CompileError> {
    // Per-instance parameters: the MMFF typifier baked `kb`/`r0` onto each bond
    // (table → equivalence fallback → empirical rules). This kernel only reads the
    // columns and evaluates — no force-field-specific resolution lives here.
    let block = frame
        .get(BONDS)
        .ok_or("mmff_bond: missing \"bonds\" block")?;
    let i_col = block
        .get("atomi")
        .and_then(|c| c.get::<u64>())
        .ok_or("mmff_bond: missing \"atomi\"")?;
    let j_col = block
        .get("atomj")
        .and_then(|c| c.get::<u64>())
        .ok_or("mmff_bond: missing \"atomj\"")?;
    let kb_col = param_reads::instance_col("mmff_bond", block, "kb")?;
    let r0_col = param_reads::instance_col("mmff_bond", block, "r0")?;

    let n = i_col.len();
    let (mut ai, mut aj, mut kb, mut r0) = (
        Vec::with_capacity(n),
        Vec::with_capacity(n),
        Vec::with_capacity(n),
        Vec::with_capacity(n),
    );
    for idx in 0..n {
        ai.push(i_col[idx] as usize);
        aj.push(j_col[idx] as usize);
        kb.push(kb_col[idx] as F);
        r0.push(r0_col[idx] as F);
    }
    Ok(ForceTerm::indexed(BondMmff {
        atom_i: ai,
        atom_j: aj,
        kb,
        r0,
    }))
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::ff::potential::fixtures::OneShot;

    #[test]
    fn test_mmff_bond_at_equilibrium() {
        let pot = BondMmff {
            atom_i: vec![0],
            atom_j: vec![1],
            kb: vec![4.258],
            r0: vec![1.508],
        };
        let coords: Vec<F> = vec![0.0, 0.0, 0.0, 1.508, 0.0, 0.0];
        let (e, _) = pot.energy_forces_at(&coords);
        assert!(
            e.abs() < 1e-6,
            "energy at equilibrium should be ~0, got {}",
            e
        );
    }

    #[test]
    fn test_mmff_bond_stretched() {
        let pot = BondMmff {
            atom_i: vec![0],
            atom_j: vec![1],
            kb: vec![4.258],
            r0: vec![1.508],
        };
        let coords: Vec<F> = vec![0.0, 0.0, 0.0, 1.608, 0.0, 0.0];
        let (e, forces) = pot.energy_forces_at(&coords);
        assert!(e > 0.0, "stretched bond should have positive energy");
        for dim in 0..3 {
            let sum = forces[dim] + forces[3 + dim];
            assert!(sum.abs() < 1e-4, "force sum should be ~0, got {}", sum);
        }
    }
}
