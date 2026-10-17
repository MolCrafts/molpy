//! UFF angle bend (RDKit `AngleBendContrib`).

use crate::ff::potential::param_reads;
use molrs::core::schema::block_names::ANGLES;
use ndarray::{Array2, ArrayView2};

use crate::ff::ir::Params;
use crate::ff::potential::angle::accumulate_angle_forces;
use crate::ff::potential::flat_coords::{sub3, term_table, validate_coords};
use crate::ff::potential::{Accumulator, ForceTerm, IndexedTerms, Potential};
use crate::op::vec3::{dot, norm};
use molrs::core::Frame;
use molrs::op::F;

/// Per-instance columns: `ka`, `order` (0–4), and for `order==0` the Fourier
/// coefficients `c0`/`c1`/`c2` derived from θ₀.
pub struct AngleUff {
    atom_i: Vec<usize>,
    atom_j: Vec<usize>,
    atom_k: Vec<usize>,
    ka: Vec<F>,
    order: Vec<u8>,
    c0: Vec<F>,
    c1: Vec<F>,
    c2: Vec<F>,
}

impl AngleUff {
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
        let _n = validate_coords(coords);
        let mut energy = 0.0 as F;
        let forces = out;

        for idx in 0..n_terms {
            let (i, j, k) = atoms(idx);
            let rji = sub3(coords, i, coords, j);
            let rjk = sub3(coords, k, coords, j);
            let d1 = norm(rji);
            let d2 = norm(rjk);
            if d1 < 1e-12 as F || d2 < 1e-12 as F {
                continue;
            }
            let cos_t = (dot(rji, rjk) / (d1 * d2)).clamp(-1.0, 1.0);
            let sin_sq = (1.0 - cos_t * cos_t).max(0.0);
            let sin_t = sin_sq.sqrt().max(1e-8 as F);
            let cos2 = cos_t * cos_t - sin_sq;

            let order = self.order[idx];
            let ka = self.ka[idx];
            let (term, d_e_d_theta) = if order == 0 {
                let term = self.c0[idx] + self.c1[idx] * cos_t + self.c2[idx] * cos2;
                let d_e = -ka * (self.c1[idx] * sin_t + 2.0 * self.c2[idx] * (2.0 * sin_t * cos_t));
                (term, d_e)
            } else {
                let n = order as F;
                let cos_n = match order {
                    1 => cos_t,
                    2 => cos2,
                    3 => cos_t * (cos_t * cos_t - 3.0 * sin_sq),
                    4 => cos_t.powi(4) - 6.0 * cos_t * cos_t * sin_sq + sin_sq * sin_sq,
                    _ => cos_t,
                };
                let term = (1.0 - cos_n) / (n * n);
                // dE/dθ = (ka/n) * sin(nθ)  with sign from d cos(nθ)/dθ = -n sin(nθ)
                // so d(1-cos(nθ))/dθ = n sin(nθ), and /n² gives sin(nθ)/n
                let sin_n = match order {
                    1 => sin_t,
                    2 => 2.0 * sin_t * cos_t,
                    3 => sin_t * (3.0 - 4.0 * sin_t * sin_t),
                    4 => cos_t * sin_t * (4.0 - 8.0 * sin_t * sin_t),
                    _ => sin_t,
                };
                let d_e = ka * sin_n / n;
                (term, d_e)
            };

            energy += ka * term;
            accumulate_angle_forces(coords, i, j, k, d_e_d_theta, forces);
        }
        energy
    }
}

impl Potential for AngleUff {
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

impl IndexedTerms for AngleUff {
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

pub fn angle_uff_constructor(
    _sp: &Params,
    _tp: &[(&str, &Params)],
    frame: &Frame,
) -> Result<ForceTerm, crate::ff::potential::CompileError> {
    let block = frame
        .get(ANGLES)
        .ok_or("uff_angle: missing \"angles\" block")?;
    let i = block
        .get("atomi")
        .and_then(|c| c.get::<u64>())
        .ok_or("uff_angle: missing atomi")?;
    let j = block
        .get("atomj")
        .and_then(|c| c.get::<u64>())
        .ok_or("uff_angle: missing atomj")?;
    let k = block
        .get("atomk")
        .and_then(|c| c.get::<u64>())
        .ok_or("uff_angle: missing atomk")?;
    let ka = param_reads::instance_col("uff_angle", block, "ka")?;
    let order = param_reads::instance_col("uff_angle", block, "order")?;
    let c0 = param_reads::instance_col("uff_angle", block, "c0")?;
    let c1 = param_reads::instance_col("uff_angle", block, "c1")?;
    let c2 = param_reads::instance_col("uff_angle", block, "c2")?;
    let n = i.len();
    Ok(ForceTerm::indexed(AngleUff {
        atom_i: (0..n).map(|t| i[t] as usize).collect(),
        atom_j: (0..n).map(|t| j[t] as usize).collect(),
        atom_k: (0..n).map(|t| k[t] as usize).collect(),
        ka: ka.iter().map(|&v| v as F).collect(),
        order: order.iter().map(|&v| v as u8).collect(),
        c0: c0.iter().map(|&v| v as F).collect(),
        c1: c1.iter().map(|&v| v as F).collect(),
        c2: c2.iter().map(|&v| v as F).collect(),
    }))
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::ff::potential::fixtures::OneShot;
    use crate::ff::potential::fixtures::assert_forces_are_negative_gradient;

    fn bent(order: u8, c: [F; 3]) -> AngleUff {
        AngleUff {
            atom_i: vec![0],
            atom_j: vec![1],
            atom_k: vec![2],
            ka: vec![100.0],
            order: vec![order],
            c0: vec![c[0]],
            c1: vec![c[1]],
            c2: vec![c[2]],
        }
    }

    /// A right angle at the centre atom.
    fn right_angle() -> Vec<F> {
        vec![1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0]
    }

    #[test]
    fn fourier_order_zero_is_ka_times_the_cosine_series() {
        // c0 + c1·cosθ + c2·cos2θ at θ = 90° is c0 − c2.
        let pot = bent(0, [0.75, 0.3, 0.25]);
        let (e, _) = pot.energy_forces_at(&right_angle());
        assert!((e - 100.0 * (0.75 - 0.25)).abs() < 1e-9);
    }

    #[test]
    fn forces_are_the_negative_energy_gradient_for_every_order() {
        let coords = vec![1.1, 0.2, -0.1, 0.0, 0.0, 0.0, -0.3, 1.2, 0.4];
        for order in 0..=4 {
            let pot = bent(order, [0.75, 0.3, 0.25]);
            assert_forces_are_negative_gradient(&pot, &coords, 1e-5);
        }
    }
}
