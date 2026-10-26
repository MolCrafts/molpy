//! UFF inversion (out-of-plane), RDKit `InversionContrib`.

use crate::ff::potential::flat_coords::sub3;
use crate::ff::potential::form_kernel::{CompoundForm, ParamColumns};
use crate::op::vec3::{cross, dot, norm};
use molrs::op::F;

/// UFF inversion (out-of-plane), RDKit `InversionContrib`:
/// `E = K (c0 + c1 sin Y + c2 cos 2W)`, Y the angle between the centre's
/// fourth bond and the normal of the plane of the other two — an N-body form
/// of the four positions.
///
/// Per-instance columns on `impropers`: `K`, `c0`, `c1`, `c2`.
///
/// No LAMMPS style has this name; its form is LAMMPS `improper_style fourier`
/// (`K[C0 + C1 cos ω + C2 cos 2ω]`, ω between the I-L axis and the I-J-K plane)
/// and so is its atom order: the centre is **first** (`atomi`). RDKit lists the
/// centre second; the UFF typifier writes it first. A degenerate term (a
/// zero-length bond, collinear neighbours) prices nothing.
#[derive(Clone, Copy, Debug, Default)]
pub struct Uff;

impl CompoundForm for Uff {
    fn eval(
        &self,
        x: &[[F; 3]],
        arity: usize,
        p: &ParamColumns<'_>,
        e: &mut [F],
        grad: &mut [[F; 3]],
    ) {
        debug_assert_eq!(arity, 4);
        let (kk, c0, c1, c2) = (p.input("K"), p.input("c0"), p.input("c1"), p.input("c2"));
        for t in 0..e.len() {
            let flat: [F; 12] = std::array::from_fn(|c| x[t * 4 + c / 3][c % 3]);
            let (energy, f) = inversion(&flat, kk[t], c0[t], c1[t], c2[t]);
            e[t] = energy;
            for (a, g) in grad[t * 4..t * 4 + 4].iter_mut().enumerate() {
                *g = [-f[a * 3], -f[a * 3 + 1], -f[a * 3 + 2]];
            }
        }
    }

    fn inputs(&self) -> Vec<String> {
        ["K", "c0", "c1", "c2"].map(String::from).to_vec()
    }
}

/// One inversion term over its four points (centre first, flat): its energy
/// and the forces on the points.
fn inversion(coords: &[F; 12], kk: F, c0: F, c1: F, c2: F) -> (F, [F; 12]) {
    let mut forces = [0.0; 12];
    // Stored centre first; the RDKit math below names the centre `j`.
    let (j, i, k, l) = (0, 1, 2, 3);
    // cosY from RDKit Utils::calculateCosY
    let r_ji = sub3(coords, i, coords, j);
    let r_jk = sub3(coords, k, coords, j);
    let r_jl = sub3(coords, l, coords, j);
    let d_ji = norm(r_ji);
    let d_jk = norm(r_jk);
    let d_jl = norm(r_jl);
    if d_ji < 1e-12 || d_jk < 1e-12 || d_jl < 1e-12 {
        return (0.0, forces);
    }
    // The gradient block below (t1 = r̂_JL × r̂_JK, t2 = r̂_JI × r̂_JL,
    // t3 = r̂_JK × r̂_JI) is written for the normal (−r_JI) × r_JK.
    // E is even in cosY so the wrong orientation is invisible in the
    // energy; the force is odd in it and comes out inverted.
    let mut n = cross([-r_ji[0], -r_ji[1], -r_ji[2]], r_jk);
    let ln = norm(n);
    if ln < 1e-12 {
        return (0.0, forces);
    }
    n = [n[0] / ln, n[1] / ln, n[2] / ln];
    let r_jl_u = [r_jl[0] / d_jl, r_jl[1] / d_jl, r_jl[2] / d_jl];
    let cos_y = dot(n, r_jl_u).clamp(-1.0, 1.0);
    let sin_y_sq = (1.0 - cos_y * cos_y).max(0.0);
    let sin_y = sin_y_sq.sqrt();
    let cos2_w = 2.0 * sin_y * sin_y - 1.0;
    let energy = kk * (c0 + c1 * sin_y + c2 * cos2_w);

    // Gradient (RDKit getGrad) — forces = −grad
    let r_ji_u = [r_ji[0] / d_ji, r_ji[1] / d_ji, r_ji[2] / d_ji];
    let r_jk_u = [r_jk[0] / d_jk, r_jk[1] / d_jk, r_jk[2] / d_jk];
    let cos_theta = dot(r_ji_u, r_jk_u).clamp(-1.0, 1.0);
    let sin_theta = (1.0 - cos_theta * cos_theta).max(0.0).sqrt().max(1e-8);
    let sin_y_s = sin_y.max(1e-8);
    // dE/dW = −K (C1 cosY + 4 C2 cosY sinY)
    let d_e_d_w = -kk * (c1 * cos_y + 4.0 * c2 * cos_y * sin_y);

    let t1 = cross(r_jl_u, r_jk_u);
    let t2 = cross(r_ji_u, r_jl_u);
    let t3 = cross(r_jk_u, r_ji_u);
    let term1 = sin_y_s * sin_theta;
    let term2 = cos_y / (sin_y_s * sin_theta * sin_theta);

    for dim in 0..3 {
        let tg1 = (t1[dim] / term1 - (r_ji_u[dim] - r_jk_u[dim] * cos_theta) * term2) / d_ji;
        let tg3 = (t2[dim] / term1 - (r_jk_u[dim] - r_ji_u[dim] * cos_theta) * term2) / d_jk;
        let tg4 = (t3[dim] / term1 - r_jl_u[dim] * cos_y / sin_y_s) / d_jl;
        // RDKit writes to grad; force = −grad
        forces[i * 3 + dim] -= d_e_d_w * tg1;
        forces[j * 3 + dim] -= -d_e_d_w * (tg1 + tg3 + tg4);
        forces[k * 3 + dim] -= d_e_d_w * tg3;
        forces[l * 3 + dim] -= d_e_d_w * tg4;
    }
    (energy, forces)
}

#[cfg(test)]
mod tests {
    use crate::ff::compile::ExplicitTerms;
    use crate::ff::ir::Params;
    use crate::ff::potential::Potentials;
    use crate::ff::potential::fixtures::OneShot;
    use crate::ff::potential::fixtures::assert_forces_are_negative_gradient;
    use molrs::op::F;

    fn inversion(c: [F; 3]) -> Potentials {
        ExplicitTerms::new("improper", "uff_inversion")
            .term(
                &[1, 0, 2, 3],
                Params::from_iter([("K", 6.0), ("c0", c[0]), ("c1", c[1]), ("c2", c[2])]),
            )
            .compile()
            .unwrap()
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
