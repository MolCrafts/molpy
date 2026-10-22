//! UFF inversion (out-of-plane), RDKit `InversionContrib`.

use crate::ff::potential::param_reads;
use molrs::core::schema::block_names::IMPROPERS;
use ndarray::{Array2, ArrayView2};

use crate::ff::ir::Params;
use crate::ff::potential::flat_coords::{sub3, term_table, validate_coords};
use crate::ff::potential::{Accumulator, ForceTerm, IndexedTerms, Potential};
use crate::op::vec3::{cross, dot, norm};
use molrs::core::Frame;
use molrs::op::F;

/// UFF inversion (out-of-plane), RDKit `InversionContrib`.
///
/// Per-instance columns on `impropers`: `K`, `c0`, `c1`, `c2`.
///
/// No LAMMPS style has this name; its form is LAMMPS `improper_style fourier`
/// (`K[C0 + C1 cos ω + C2 cos 2ω]`, ω between the I-L axis and the I-J-K plane)
/// and so is its atom order: the centre is **first** (`atomi`). RDKit lists the
/// centre second; the UFF typifier writes it first.
pub struct ImproperUff {
    atom_i: Vec<usize>,
    atom_j: Vec<usize>,
    atom_k: Vec<usize>,
    atom_l: Vec<usize>,
    k: Vec<F>,
    c0: Vec<F>,
    c1: Vec<F>,
    c2: Vec<F>,
}

impl ImproperUff {
    /// The physics, once. Which atoms a term names is the only thing
    /// that differs between the two entry points, so it is the only thing
    /// passed in — a second copy of the loop would be a second place for
    /// the force expression to drift.
    fn fold(
        &self,
        coords: &[F],
        out: &mut [F],
        n_terms: usize,
        atoms: impl Fn(usize) -> (usize, usize, usize, usize),
    ) -> F {
        let _n = validate_coords(coords);
        let mut energy = 0.0 as F;
        let forces = out;

        for idx in 0..n_terms {
            // Stored centre first; the RDKit math below names the centre `j`.
            let (j, i, k, l) = atoms(idx);
            // cosY from RDKit Utils::calculateCosY
            let r_ji = sub3(coords, i, coords, j);
            let r_jk = sub3(coords, k, coords, j);
            let r_jl = sub3(coords, l, coords, j);
            let d_ji = norm(r_ji);
            let d_jk = norm(r_jk);
            let d_jl = norm(r_jl);
            if d_ji < 1e-12 || d_jk < 1e-12 || d_jl < 1e-12 {
                continue;
            }
            // The gradient block below (t1 = r̂_JL × r̂_JK, t2 = r̂_JI × r̂_JL,
            // t3 = r̂_JK × r̂_JI) is written for the normal (−r_JI) × r_JK.
            // E is even in cosY so the wrong orientation is invisible in the
            // energy; the force is odd in it and comes out inverted.
            let mut n = cross([-r_ji[0], -r_ji[1], -r_ji[2]], r_jk);
            let ln = norm(n);
            if ln < 1e-12 {
                continue;
            }
            n = [n[0] / ln, n[1] / ln, n[2] / ln];
            let r_jl_u = [r_jl[0] / d_jl, r_jl[1] / d_jl, r_jl[2] / d_jl];
            let cos_y = dot(n, r_jl_u).clamp(-1.0, 1.0);
            let sin_y_sq = (1.0 - cos_y * cos_y).max(0.0);
            let sin_y = sin_y_sq.sqrt();
            let cos2_w = 2.0 * sin_y * sin_y - 1.0;
            let kk = self.k[idx];
            energy += kk * (self.c0[idx] + self.c1[idx] * sin_y + self.c2[idx] * cos2_w);

            // Gradient (RDKit getGrad) — forces = −grad
            let r_ji_u = [r_ji[0] / d_ji, r_ji[1] / d_ji, r_ji[2] / d_ji];
            let r_jk_u = [r_jk[0] / d_jk, r_jk[1] / d_jk, r_jk[2] / d_jk];
            let cos_theta = dot(r_ji_u, r_jk_u).clamp(-1.0, 1.0);
            let sin_theta = (1.0 - cos_theta * cos_theta).max(0.0).sqrt().max(1e-8);
            let sin_y_s = sin_y.max(1e-8);
            // dE/dW = −K (C1 cosY + 4 C2 cosY sinY)
            let d_e_d_w = -kk * (self.c1[idx] * cos_y + 4.0 * self.c2[idx] * cos_y * sin_y);

            let t1 = cross(r_jl_u, r_jk_u);
            let t2 = cross(r_ji_u, r_jl_u);
            let t3 = cross(r_jk_u, r_ji_u);
            let term1 = sin_y_s * sin_theta;
            let term2 = cos_y / (sin_y_s * sin_theta * sin_theta);

            for dim in 0..3 {
                let tg1 =
                    (t1[dim] / term1 - (r_ji_u[dim] - r_jk_u[dim] * cos_theta) * term2) / d_ji;
                let tg3 =
                    (t2[dim] / term1 - (r_jk_u[dim] - r_ji_u[dim] * cos_theta) * term2) / d_jk;
                let tg4 = (t3[dim] / term1 - r_jl_u[dim] * cos_y / sin_y_s) / d_jl;
                // RDKit writes to grad; force = −grad
                forces[i * 3 + dim] -= d_e_d_w * tg1;
                forces[j * 3 + dim] -= -d_e_d_w * (tg1 + tg3 + tg4);
                forces[k * 3 + dim] -= d_e_d_w * tg3;
                forces[l * 3 + dim] -= d_e_d_w * tg4;
            }
        }
        energy
    }
}

impl Potential for ImproperUff {
    fn evaluate(&self, coords: &[F], acc: &mut Accumulator<'_>) {
        let energy = self.fold(
            coords,
            acc.forces_without_virial(),
            self.atom_i.len(),
            |t| {
                (
                    self.atom_i[t],
                    self.atom_j[t],
                    self.atom_k[t],
                    self.atom_l[t],
                )
            },
        );
        acc.add_energy(energy);
    }
}

impl IndexedTerms for ImproperUff {
    fn terms(&self) -> Array2<u32> {
        term_table(&[&self.atom_i, &self.atom_j, &self.atom_k, &self.atom_l])
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
                terms[[t, 3]] as usize,
            )
        });
        acc.add_energy(energy);
    }
}

pub fn improper_uff_constructor(
    _sp: &Params,
    _tp: &[(&str, &Params)],
    frame: &Frame,
) -> Result<ForceTerm, crate::ff::potential::CompileError> {
    let Some(block) = frame.get(IMPROPERS) else {
        return Ok(ForceTerm::indexed(ImproperUff {
            atom_i: vec![],
            atom_j: vec![],
            atom_k: vec![],
            atom_l: vec![],
            k: vec![],
            c0: vec![],
            c1: vec![],
            c2: vec![],
        }));
    };
    if block.n_rows().unwrap_or(0) == 0 {
        return Ok(ForceTerm::indexed(ImproperUff {
            atom_i: vec![],
            atom_j: vec![],
            atom_k: vec![],
            atom_l: vec![],
            k: vec![],
            c0: vec![],
            c1: vec![],
            c2: vec![],
        }));
    }
    let i = block
        .get("atomi")
        .and_then(|c| c.get::<u64>())
        .ok_or("uff_inversion: missing atomi")?;
    let j = block
        .get("atomj")
        .and_then(|c| c.get::<u64>())
        .ok_or("uff_inversion: missing atomj")?;
    let k = block
        .get("atomk")
        .and_then(|c| c.get::<u64>())
        .ok_or("uff_inversion: missing atomk")?;
    let l = block
        .get("atoml")
        .and_then(|c| c.get::<u64>())
        .ok_or("uff_inversion: missing atoml")?;
    let kk = param_reads::instance_col("uff_inversion", block, "K")?;
    let c0 = param_reads::instance_col("uff_inversion", block, "c0")?;
    let c1 = param_reads::instance_col("uff_inversion", block, "c1")?;
    let c2 = param_reads::instance_col("uff_inversion", block, "c2")?;
    let n = i.len();
    Ok(ForceTerm::indexed(ImproperUff {
        atom_i: (0..n).map(|t| i[t] as usize).collect(),
        atom_j: (0..n).map(|t| j[t] as usize).collect(),
        atom_k: (0..n).map(|t| k[t] as usize).collect(),
        atom_l: (0..n).map(|t| l[t] as usize).collect(),
        k: kk.iter().map(|&v| v as F).collect(),
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

    fn inversion(c: [F; 3]) -> ImproperUff {
        ImproperUff {
            atom_i: vec![1],
            atom_j: vec![0],
            atom_k: vec![2],
            atom_l: vec![3],
            k: vec![6.0],
            c0: vec![c[0]],
            c1: vec![c[1]],
            c2: vec![c[2]],
        }
    }

    /// The centre (atom 1, stored first) at the origin with three planar
    /// neighbours.
    fn planar() -> Vec<F> {
        vec![
            1.2, 0.0, 0.0, 0.0, 0.0, 0.0, -0.6, 1.0, 0.0, -0.6, -1.0, 0.0,
        ]
    }

    #[test]
    fn the_sp2_series_is_zero_in_the_plane_and_positive_out_of_it() {
        // E = K·(c0 + c1·sinY + c2·cos2W): in the plane the fourth bond is
        // perpendicular to the normal, sinY = 1, and (1, −1, 0) gives zero.
        let pot = inversion([1.0, -1.0, 0.0]);
        let (e_flat, _) = pot.energy_forces_at(&planar());
        assert!(e_flat.abs() < 1e-9);
        let mut lifted = planar();
        lifted[11] = 0.4;
        let (e_up, _) = pot.energy_forces_at(&lifted);
        assert!(e_up > 1e-3);
    }

    #[test]
    fn forces_are_the_negative_energy_gradient() {
        let mut coords = planar();
        coords[2] = 0.15;
        coords[8] = -0.25;
        coords[11] = 0.35;
        assert_forces_are_negative_gradient(&inversion([1.0, -1.0, 0.0]), &coords, 1e-5);
        assert_forces_are_negative_gradient(&inversion([0.5, 0.2, 0.3]), &coords, 1e-5);
    }
}
