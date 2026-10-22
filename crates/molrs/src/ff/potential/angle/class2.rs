//! Class2 (quartic) angle (LAMMPS `angle_style class2`).

use crate::ff::potential::param_reads;
use molrs::core::schema::block_names::ANGLES;
use std::collections::HashMap;

use ndarray::{Array2, ArrayView2};

use crate::ff::ir::Params;
use crate::ff::potential::flat_coords::{compute_angle, term_table, validate_coords};
use crate::ff::potential::{Accumulator, ForceTerm, IndexedTerms, Potential};
use molrs::core::Frame;
use molrs::op::F;

/// Class2 quartic angle potential. Its own `theta0` array is in radians.
///
/// Class2 (quartic) angle potential:
/// E = k2*(theta-theta0)^2 + k3*(theta-theta0)^3 + k4*(theta-theta0)^4
///
/// The COMPASS/class2 anharmonic angle core term (cross-terms bb/ba are
/// separate styles, not implemented here). Parameters per type, as LAMMPS
/// `angle_style class2` takes them: `theta0` in **degrees**, `k2`, `k3`, `k4`
/// in energy/radianⁿ. The kernel converts `theta0` to radians once.
pub struct AngleClass2 {
    atom_i: Vec<usize>,
    atom_j: Vec<usize>,
    atom_k: Vec<usize>,
    theta0: Vec<F>,
    k2: Vec<F>,
    k3: Vec<F>,
    k4: Vec<F>,
}

impl AngleClass2 {
    #[allow(clippy::too_many_arguments)]
    pub fn new(
        atom_i: Vec<usize>,
        atom_j: Vec<usize>,
        atom_k: Vec<usize>,
        theta0: Vec<F>,
        k2: Vec<F>,
        k3: Vec<F>,
        k4: Vec<F>,
    ) -> Self {
        let n = atom_i.len();
        assert_eq!(atom_j.len(), n);
        assert_eq!(atom_k.len(), n);
        assert_eq!(theta0.len(), n);
        assert_eq!(k2.len(), n);
        assert_eq!(k3.len(), n);
        assert_eq!(k4.len(), n);
        Self {
            atom_i,
            atom_j,
            atom_k,
            theta0,
            k2,
            k3,
            k4,
        }
    }
}

impl AngleClass2 {
    /// The physics, once. Which atoms a term names is the only thing
    /// that differs between the two entry points, so it is the only thing
    /// passed in — a second copy of the loop would be a second place for
    /// the force expression to drift.
    fn fold(
        &self,
        coords: &[F],
        out: &mut [F],
        n_terms: usize,
        atoms: impl Fn(usize) -> (usize, usize, usize),
    ) -> F {
        let _n_atoms = validate_coords(coords);
        let mut energy: F = 0.0;
        let forces = out;

        for idx in 0..n_terms {
            let (i, j, k) = atoms(idx);
            let (k2, k3, k4) = (self.k2[idx], self.k3[idx], self.k4[idx]);
            let theta0 = self.theta0[idx];

            let theta = compute_angle(coords, i, j, k);
            let dt = theta - theta0;
            let dt2 = dt * dt;
            energy += k2 * dt2 + k3 * dt2 * dt + k4 * dt2 * dt2;

            // dE/dtheta = 2 k2 dt + 3 k3 dt^2 + 4 k4 dt^3
            let de_dtheta = 2.0 * k2 * dt + 3.0 * k3 * dt2 + 4.0 * k4 * dt2 * dt;
            super::accumulate_angle_forces(coords, i, j, k, de_dtheta, forces);
        }

        energy
    }
}

impl Potential for AngleClass2 {
    fn evaluate(&self, coords: &[F], acc: &mut Accumulator<'_>) {
        let energy = self.fold(
            coords,
            acc.forces_without_virial(),
            self.atom_i.len(),
            |t| (self.atom_i[t], self.atom_j[t], self.atom_k[t]),
        );
        acc.add_energy(energy);
    }
}

impl IndexedTerms for AngleClass2 {
    fn terms(&self) -> Array2<u32> {
        term_table(&[&self.atom_i, &self.atom_j, &self.atom_k])
    }
    fn evaluate_terms(&self, coords: &[F], terms: ArrayView2<'_, u32>, acc: &mut Accumulator<'_>) {
        debug_assert_eq!(
            terms.nrows(),
            self.atom_i.len(),
            "the row set is the force field's; only the atoms a row names may be rebound"
        );
        let energy = self.fold(coords, acc.forces_without_virial(), terms.nrows(), |t| {
            (
                terms[[t, 0]] as usize,
                terms[[t, 1]] as usize,
                terms[[t, 2]] as usize,
            )
        });
        acc.add_energy(energy);
    }
}

/// Construct an [`AngleClass2`] from style params, type params, and Frame topology.
pub fn angle_class2_constructor(
    _style_params: &Params,
    type_params: &[(&str, &Params)],
    frame: &Frame,
) -> Result<ForceTerm, crate::ff::potential::CompileError> {
    let type_map: HashMap<&str, &Params> = type_params.iter().copied().collect();

    let block = frame
        .get(ANGLES)
        .ok_or_else(|| "AngleClass2: frame missing \"angles\" block".to_string())?;
    let i_col = block
        .get("atomi")
        .and_then(|c| c.get::<u64>())
        .ok_or_else(|| "AngleClass2: angles block missing \"atomi\" column".to_string())?;
    let j_col = block
        .get("atomj")
        .and_then(|c| c.get::<u64>())
        .ok_or_else(|| "AngleClass2: angles block missing \"atomj\" column".to_string())?;
    let k_col = block
        .get("atomk")
        .and_then(|c| c.get::<u64>())
        .ok_or_else(|| "AngleClass2: angles block missing \"atomk\" column".to_string())?;
    let type_col = block
        .get("type")
        .and_then(|c| c.get::<String>())
        .ok_or_else(|| "AngleClass2: angles block missing \"type\" column".to_string())?;

    let (mut ai, mut aj, mut ak) = (Vec::new(), Vec::new(), Vec::new());
    let (mut t0, mut k2, mut k3, mut k4) = (Vec::new(), Vec::new(), Vec::new(), Vec::new());
    let need = |p: &Params, key: &str, label: &str| param_reads::type_num("class2", label, p, key);
    for idx in 0..i_col.len() {
        let label = &type_col[idx];
        let p = type_map
            .get(label.as_str())
            .ok_or_else(|| format!("AngleClass2: unknown angle type '{}'", label))?;
        ai.push(i_col[idx] as usize);
        aj.push(j_col[idx] as usize);
        ak.push(k_col[idx] as usize);
        t0.push(need(p, "theta0", label)?.to_radians()); // degrees → radians
        k2.push(need(p, "k2", label)?);
        k3.push(need(p, "k3", label)?);
        k4.push(need(p, "k4", label)?);
    }

    Ok(ForceTerm::indexed(AngleClass2::new(
        ai, aj, ak, t0, k2, k3, k4,
    )))
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::ff::potential::fixtures::OneShot;

    fn numerical_forces(pot: &AngleClass2, coords: &[F]) -> Vec<F> {
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
    fn energy_zero_at_equilibrium() {
        let theta0: F = std::f64::consts::FRAC_PI_2;
        let pot = AngleClass2::new(
            vec![0],
            vec![1],
            vec![2],
            vec![theta0],
            vec![50.0],
            vec![10.0],
            vec![5.0],
        );
        let coords: Vec<F> = vec![1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0];
        assert!(pot.energy_at(&coords).abs() < 1e-9);
    }

    #[test]
    fn forces_match_finite_difference() {
        let theta0: F = 1.9;
        let pot = AngleClass2::new(
            vec![0],
            vec![1],
            vec![2],
            vec![theta0],
            vec![50.0],
            vec![10.0],
            vec![5.0],
        );
        // bent, off-axis geometry
        let coords: Vec<F> = vec![1.0, 0.2, 0.0, 0.0, 0.0, 0.1, -0.3, 1.0, 0.0];
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
