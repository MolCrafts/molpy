//! Morse bond (LAMMPS `bond_style morse`).

use crate::ff::potential::param_reads;
use molrs::core::schema::block_names::BONDS;
use std::collections::HashMap;

use ndarray::{Array2, ArrayView2};

use crate::ff::ir::Params;
use crate::ff::potential::flat_coords::term_table;
use crate::ff::potential::flat_coords::validate_coords;
use crate::ff::potential::{Accumulator, ForceTerm, IndexedTerms, Potential};
use molrs::core::Frame;
use molrs::op::F;

/// Morse bond potential with pre-resolved flat arrays.
///
/// LAMMPS `bond_style morse`: E = d0·(1 − e^{−alpha·(r − r0)})²
///
/// Anharmonic bond with a finite dissociation energy. Parameters per type, as
/// LAMMPS names them: `d0` (LAMMPS `D0`, well depth, energy), `alpha`
/// (steepness, 1/length), `r0` (equilibrium length).
pub struct BondMorse {
    atom_i: Vec<usize>,
    atom_j: Vec<usize>,
    d: Vec<F>,
    alpha: Vec<F>,
    r0: Vec<F>,
}

impl BondMorse {
    pub fn new(
        atom_i: Vec<usize>,
        atom_j: Vec<usize>,
        d: Vec<F>,
        alpha: Vec<F>,
        r0: Vec<F>,
    ) -> Self {
        let n = atom_i.len();
        assert_eq!(atom_j.len(), n);
        assert_eq!(d.len(), n);
        assert_eq!(alpha.len(), n);
        assert_eq!(r0.len(), n);
        Self {
            atom_i,
            atom_j,
            d,
            alpha,
            r0,
        }
    }
}

impl BondMorse {
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
        let n_atoms = validate_coords(coords);
        let mut energy: F = 0.0;
        let forces = out;

        for idx in 0..n_terms {
            let (i, j) = atoms(idx);
            debug_assert!(i < n_atoms && j < n_atoms);

            let (d, alpha, r0) = (self.d[idx], self.alpha[idx], self.r0[idx]);

            let dx = coords[j * 3] - coords[i * 3];
            let dy = coords[j * 3 + 1] - coords[i * 3 + 1];
            let dz = coords[j * 3 + 2] - coords[i * 3 + 2];
            let r = (dx * dx + dy * dy + dz * dz).sqrt();
            if r < 1e-12 {
                continue;
            }
            let y = (-alpha * (r - r0)).exp(); // exp(-alpha (r-r0))
            let one_my = 1.0 - y;
            energy += d * one_my * one_my;

            // dE/dr = 2 D alpha y (1 - y)
            let dedr = 2.0 * d * alpha * y * one_my;
            let factor = -dedr / r;
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

impl Potential for BondMorse {
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

impl IndexedTerms for BondMorse {
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

/// Construct a [`BondMorse`] from style params, type params, and Frame topology.
pub fn bond_morse_constructor(
    _style_params: &Params,
    type_params: &[(&str, &Params)],
    frame: &Frame,
) -> Result<ForceTerm, crate::ff::potential::CompileError> {
    let type_map: HashMap<&str, &Params> = type_params.iter().copied().collect();

    let block = frame
        .get(BONDS)
        .ok_or_else(|| "BondMorse: frame missing \"bonds\" block".to_string())?;
    let i_col = block
        .get("atomi")
        .and_then(|c| c.get::<u64>())
        .ok_or_else(|| "BondMorse: bonds block missing \"atomi\" column".to_string())?;
    let j_col = block
        .get("atomj")
        .and_then(|c| c.get::<u64>())
        .ok_or_else(|| "BondMorse: bonds block missing \"atomj\" column".to_string())?;
    let type_col = block
        .get("type")
        .and_then(|c| c.get::<String>())
        .ok_or_else(|| "BondMorse: bonds block missing \"type\" column".to_string())?;

    let (mut ai, mut aj) = (Vec::new(), Vec::new());
    let (mut dv, mut av, mut rv) = (Vec::new(), Vec::new(), Vec::new());
    let need = |p: &Params, key: &str, label: &str| param_reads::type_num("morse", label, p, key);
    for idx in 0..i_col.len() {
        let label = &type_col[idx];
        let p = type_map
            .get(label.as_str())
            .ok_or_else(|| format!("BondMorse: unknown bond type '{}'", label))?;
        ai.push(i_col[idx] as usize);
        aj.push(j_col[idx] as usize);
        dv.push(need(p, "d0", label)?);
        av.push(need(p, "alpha", label)?);
        rv.push(need(p, "r0", label)?);
    }

    Ok(ForceTerm::indexed(BondMorse::new(ai, aj, dv, av, rv)))
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::ff::potential::fixtures::OneShot;

    fn numerical_forces(pot: &BondMorse, coords: &[F]) -> Vec<F> {
        let h = 1e-6;
        let mut num = vec![0.0; coords.len()];
        for k in 0..coords.len() {
            let mut cp = coords.to_vec();
            let mut cm = coords.to_vec();
            cp[k] += h;
            cm[k] -= h;
            num[k] = -(pot.energy_at(&cp) - pot.energy_at(&cm)) / (2.0 * h);
        }
        num
    }

    #[test]
    fn energy_zero_and_force_zero_at_equilibrium() {
        let pot = BondMorse::new(vec![0], vec![1], vec![100.0], vec![2.0], vec![1.5]);
        let coords: Vec<F> = vec![0.0, 0.0, 0.0, 1.5, 0.0, 0.0];
        let (e, f) = pot.energy_forces_at(&coords);
        assert!(e.abs() < 1e-12, "energy {e}");
        for fi in f {
            assert!(fi.abs() < 1e-9);
        }
    }

    #[test]
    fn forces_match_finite_difference() {
        let pot = BondMorse::new(vec![0], vec![1], vec![100.0], vec![2.0], vec![1.5]);
        let coords: Vec<F> = vec![0.1, -0.2, 0.05, 1.3, 0.6, -0.3];
        let (_, analytical) = pot.energy_forces_at(&coords);
        let numerical = numerical_forces(&pot, &coords);
        for k in 0..coords.len() {
            assert!(
                (analytical[k] - numerical[k]).abs() < 1e-5,
                "k={k} a={} n={}",
                analytical[k],
                numerical[k]
            );
        }
    }
}
