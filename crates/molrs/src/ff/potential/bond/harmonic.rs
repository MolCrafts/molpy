//! Harmonic bond (LAMMPS `bond_style harmonic`).

use crate::ff::potential::param_reads;
use molrs::core::schema::block_names::BONDS;
use std::collections::HashMap;

use ndarray::{Array2, ArrayView2};

use crate::ff::ir::Params;
use crate::ff::potential::flat_coords::{term_table, validate_coords};
use crate::ff::potential::{Accumulator, ForceTerm, IndexedTerms, Potential};
use molrs::core::Frame;
use molrs::op::F;

/// Harmonic bond potential with pre-resolved flat arrays.
///
/// LAMMPS `bond_style harmonic`: E = k·(r − r0)².
///
/// `k` is LAMMPS's `K`, energy/length², and carries the usual ½: there is no
/// hidden factor, so a `bond_coeff t K r0` line is `k = K` here.
pub struct BondHarmonic {
    atom_i: Vec<usize>,
    atom_j: Vec<usize>,
    k: Vec<F>,
    r0: Vec<F>,
}

impl BondHarmonic {
    pub fn new(atom_i: Vec<usize>, atom_j: Vec<usize>, k: Vec<F>, r0: Vec<F>) -> Self {
        assert_eq!(atom_i.len(), atom_j.len());
        assert_eq!(atom_i.len(), k.len());
        assert_eq!(atom_i.len(), r0.len());
        Self {
            atom_i,
            atom_j,
            k,
            r0,
        }
    }
}

impl BondHarmonic {
    /// The physics, once. Which atoms a term names is the only thing that
    /// differs between the two entry points, so it is the only thing passed
    /// in — a second copy of the loop would be a second place for the force
    /// expression to drift.
    fn fold(
        &self,
        coords: &[F],
        out: &mut [F],
        n_terms: usize,
        atoms: impl Fn(usize) -> (usize, usize),
    ) -> F {
        let n_atoms = validate_coords(coords);
        let mut energy: F = 0.0;
        let forces = out;

        for idx in 0..n_terms {
            let (i, j) = atoms(idx);
            debug_assert!(i < n_atoms && j < n_atoms);

            let k = self.k[idx];
            let r0 = self.r0[idx];

            let dx = coords[j * 3] - coords[i * 3];
            let dy = coords[j * 3 + 1] - coords[i * 3 + 1];
            let dz = coords[j * 3 + 2] - coords[i * 3 + 2];
            let r = (dx * dx + dy * dy + dz * dz).sqrt();
            let dr = r - r0;
            energy += k * dr * dr;

            if r < 1e-12 {
                continue;
            }

            // dE/dr = 2k(r − r0)
            let factor = -2.0 * k * dr / r;
            let fx = factor * dx;
            let fy = factor * dy;
            let fz = factor * dz;

            forces[j * 3] += fx;
            forces[j * 3 + 1] += fy;
            forces[j * 3 + 2] += fz;
            forces[i * 3] -= fx;
            forces[i * 3 + 1] -= fy;
            forces[i * 3 + 2] -= fz;
        }

        energy
    }
}

impl Potential for BondHarmonic {
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

impl IndexedTerms for BondHarmonic {
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

/// Construct a [`BondHarmonic`] from style params, type params, and Frame topology.
pub fn bond_harmonic_constructor(
    _style_params: &Params,
    type_params: &[(&str, &Params)],
    frame: &Frame,
) -> Result<ForceTerm, crate::ff::potential::CompileError> {
    let type_map: HashMap<&str, &Params> = type_params.iter().copied().collect();

    let block = frame
        .get(BONDS)
        .ok_or_else(|| "BondHarmonic: frame missing \"bonds\" block".to_string())?;
    let i_col = block
        .get("atomi")
        .and_then(|c| c.get::<u64>())
        .ok_or_else(|| "BondHarmonic: bonds block missing \"atomi\" column".to_string())?;
    let j_col = block
        .get("atomj")
        .and_then(|c| c.get::<u64>())
        .ok_or_else(|| "BondHarmonic: bonds block missing \"atomj\" column".to_string())?;
    let type_col = block
        .get("type")
        .and_then(|c| c.get::<String>())
        .ok_or_else(|| "BondHarmonic: bonds block missing \"type\" column".to_string())?;

    let mut atom_i = Vec::with_capacity(i_col.len());
    let mut atom_j = Vec::with_capacity(i_col.len());
    let mut k_vec = Vec::with_capacity(i_col.len());
    let mut r0_vec = Vec::with_capacity(i_col.len());

    for idx in 0..i_col.len() {
        let label = &type_col[idx];
        let params = type_map
            .get(label.as_str())
            .ok_or_else(|| format!("BondHarmonic: unknown bond type '{}'", label))?;
        // `k` is LAMMPS's `K` (= AMBER's `RK`): E = k(r − r0)², no ½.
        let k = param_reads::type_num("harmonic", label, params, "k")?;
        let r0 = param_reads::type_num("harmonic", label, params, "r0")?;

        atom_i.push(i_col[idx] as usize);
        atom_j.push(j_col[idx] as usize);
        k_vec.push(k);
        r0_vec.push(r0);
    }

    Ok(ForceTerm::indexed(BondHarmonic::new(
        atom_i, atom_j, k_vec, r0_vec,
    )))
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::ff::potential::fixtures::OneShot;

    #[test]
    fn test_bond_harmonic_energy_and_force() {
        // LAMMPS bond_style harmonic: E = K(r − r0)², F = −2K(r − r0).
        let pot = BondHarmonic::new(vec![0], vec![1], vec![150.0], vec![1.5]);
        let coords: Vec<F> = vec![0.0, 0.0, 0.0, 2.0, 0.0, 0.0];

        let (e, forces) = pot.energy_forces_at(&coords);
        assert_eq!(e, 150.0 * 0.5 * 0.5);
        assert!((forces[0] - 150.0).abs() < 1e-12);
        assert!((forces[3] + 150.0).abs() < 1e-12);
    }
}
