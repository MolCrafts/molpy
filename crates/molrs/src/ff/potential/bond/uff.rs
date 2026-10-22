//! UFF bond stretch: E = ½ · kb · (r − r0)² (RDKit `BondStretchContrib`).

use crate::ff::potential::param_reads;
use molrs::core::schema::block_names::BONDS;
use ndarray::{Array2, ArrayView2};

use crate::ff::ir::Params;
use crate::ff::potential::flat_coords::term_table;
use crate::ff::potential::flat_coords::validate_coords;
use crate::ff::potential::{Accumulator, ForceTerm, IndexedTerms, Potential};
use molrs::core::Frame;
use molrs::op::F;

/// Harmonic UFF bond stretch with per-instance `kb` / `r0`.
pub struct BondUff {
    atom_i: Vec<usize>,
    atom_j: Vec<usize>,
    kb: Vec<F>,
    r0: Vec<F>,
}

impl BondUff {
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
        let mut energy = 0.0 as F;
        let forces = out;
        for idx in 0..n_terms {
            let (i, j) = atoms(idx);
            let dx = coords[j * 3] - coords[i * 3];
            let dy = coords[j * 3 + 1] - coords[i * 3 + 1];
            let dz = coords[j * 3 + 2] - coords[i * 3 + 2];
            let r = (dx * dx + dy * dy + dz * dz).sqrt();
            let dr = r - self.r0[idx];
            energy += 0.5 * self.kb[idx] * dr * dr;
            if r < 1e-12 as F {
                continue;
            }
            // F = −∇E; ∇_j r = (j−i)/r, so F_j = −(kb·dr)·(j−i)/r
            let pref = -self.kb[idx] * dr / r;
            for dim in 0..3 {
                let d = [dx, dy, dz][dim];
                forces[j * 3 + dim] += pref * d;
                forces[i * 3 + dim] -= pref * d;
            }
        }
        energy
    }
}

impl Potential for BondUff {
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

impl IndexedTerms for BondUff {
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

pub fn bond_uff_constructor(
    _sp: &Params,
    _tp: &[(&str, &Params)],
    frame: &Frame,
) -> Result<ForceTerm, crate::ff::potential::CompileError> {
    let block = frame
        .get(BONDS)
        .ok_or("uff_bond: missing \"bonds\" block")?;
    let i = block
        .get("atomi")
        .and_then(|c| c.get::<u64>())
        .ok_or("uff_bond: missing atomi")?;
    let j = block
        .get("atomj")
        .and_then(|c| c.get::<u64>())
        .ok_or("uff_bond: missing atomj")?;
    let kb = param_reads::instance_col("uff_bond", block, "kb")?;
    let r0 = param_reads::instance_col("uff_bond", block, "r0")?;
    let n = i.len();
    Ok(ForceTerm::indexed(BondUff {
        atom_i: (0..n).map(|t| i[t] as usize).collect(),
        atom_j: (0..n).map(|t| j[t] as usize).collect(),
        kb: kb.iter().map(|&v| v as F).collect(),
        r0: r0.iter().map(|&v| v as F).collect(),
    }))
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::ff::potential::fixtures::OneShot;
    use crate::ff::potential::fixtures::assert_forces_are_negative_gradient;

    fn stretched(kb: F, r0: F, r: F) -> (BondUff, Vec<F>) {
        let pot = BondUff {
            atom_i: vec![0],
            atom_j: vec![1],
            kb: vec![kb],
            r0: vec![r0],
        };
        (pot, vec![0.0, 0.0, 0.0, r, 0.0, 0.0])
    }

    #[test]
    fn energy_is_half_kb_times_the_squared_stretch() {
        let (pot, coords) = stretched(700.0, 1.5, 1.6);
        let (e, f) = pot.energy_forces_at(&coords);
        assert!((e - 0.5 * 700.0 * 0.01).abs() < 1e-9);
        // The pull is along x, opposite on the two atoms.
        assert!((f[0] - 700.0 * 0.1).abs() < 1e-9);
        assert!((f[3] + 700.0 * 0.1).abs() < 1e-9);
    }

    #[test]
    fn energy_and_force_vanish_at_the_rest_length() {
        let (pot, coords) = stretched(700.0, 1.5, 1.5);
        let (e, f) = pot.energy_forces_at(&coords);
        assert_eq!(e, 0.0);
        assert!(f.iter().all(|x| x.abs() < 1e-12));
    }

    #[test]
    fn forces_are_the_negative_energy_gradient() {
        let pot = BondUff {
            atom_i: vec![0],
            atom_j: vec![1],
            kb: vec![700.0],
            r0: vec![1.5],
        };
        let coords = vec![0.1, -0.2, 0.3, 1.4, 0.5, -0.6];
        assert_forces_are_negative_gradient(&pot, &coords, 1e-5);
    }
}
