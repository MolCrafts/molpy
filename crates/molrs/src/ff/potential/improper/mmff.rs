//! MMFF94 out-of-plane bending (Wilson angle).

use crate::core::constants::MMFF_MDYNE_A_TO_KCAL_MOL;
use crate::ff::potential::flat_coords::sub3;
use crate::ff::potential::form_kernel::{CompoundForm, ParamColumns};
use crate::op::vec3::{cross, dot, norm};
use molrs::op::F;

/// MMFF94 out-of-plane bending: `E = ½ · 143.9325 · koop · χ²`, χ the Wilson
/// angle — an N-body form of the four positions, `koop` (md·Å/rad²) a
/// per-instance column the MMFF typifier bakes onto each improper.
///
/// No LAMMPS style has this name. Its atom order is that of LAMMPS's
/// out-of-plane styles (`improper_style fourier`, `umbrella`): the centre is
/// **first** (`atomi`), and χ is the angle between the bond centre→`atoml` and
/// the plane (centre, `atomj`, `atomk`) — MMFF's `I J K L` with `J` central,
/// read centre first. A degenerate term (collinear neighbours, a zero-length
/// bond) prices nothing.
#[derive(Clone, Copy, Debug, Default)]
pub struct Mmff;

impl CompoundForm for Mmff {
    fn eval(
        &self,
        x: &[[F; 3]],
        arity: usize,
        p: &ParamColumns<'_>,
        e: &mut [F],
        grad: &mut [[F; 3]],
    ) {
        debug_assert_eq!(arity, 4);
        let koop = p.input("koop");
        for t in 0..e.len() {
            let flat: [F; 12] = std::array::from_fn(|c| x[t * 4 + c / 3][c % 3]);
            let (energy, f) = wilson(&flat, koop[t]);
            e[t] = energy;
            for (a, g) in grad[t * 4..t * 4 + 4].iter_mut().enumerate() {
                *g = [-f[a * 3], -f[a * 3 + 1], -f[a * 3 + 2]];
            }
        }
    }

    fn inputs(&self) -> Vec<String> {
        vec!["koop".to_owned()]
    }
}

/// One out-of-plane term over its four points (centre first, flat): its
/// energy and the forces on the points.
fn wilson(coords: &[F; 12], koop: F) -> (F, [F; 12]) {
    let mut forces = [0.0; 12];
    let conv = MMFF_MDYNE_A_TO_KCAL_MOL as F;
    // Stored centre first; the Wilson math below names the centre `j`.
    let (j, i, k, l) = (0, 1, 2, 3);
    let a = sub3(coords, i, coords, j); // ji
    let b = sub3(coords, k, coords, j); // jk
    let c = sub3(coords, l, coords, j); // jl
    let n = cross(a, b);
    let n_mag = norm(n);
    let c_mag = norm(c);
    if n_mag < 1e-12 as F || c_mag < 1e-12 as F {
        return (0.0, forces);
    }

    let sin_chi = (dot(c, n) / (c_mag * n_mag)).clamp(-1.0, 1.0);
    let chi = sin_chi.asin();
    let cos_chi = chi.cos();
    let energy = 0.5 * conv * koop * chi * chi;

    if cos_chi.abs() < 1e-12 as F {
        return (energy, forces);
    }
    let prefactor = conv * koop * chi / cos_chi;

    // d(sin chi)/d(r_l)
    let inv_cn = 1.0 / (c_mag * n_mag);
    for dim in 0..3 {
        let dsdc = n[dim] * inv_cn - sin_chi * c[dim] / (c_mag * c_mag);
        forces[l * 3 + dim] += -prefactor * dsdc;
        forces[j * 3 + dim] -= -prefactor * dsdc;
    }
    // d(sin chi)/d(r_i)  (a = r_i - r_j, affects n = a x b)
    let bxc = cross(b, c);
    let bxn = cross(b, n);
    for dim in 0..3 {
        let dsda = bxc[dim] * inv_cn - sin_chi * bxn[dim] / (n_mag * n_mag);
        forces[i * 3 + dim] += -prefactor * dsda;
        forces[j * 3 + dim] -= -prefactor * dsda;
    }
    // d(sin chi)/d(r_k)  (b = r_k - r_j, affects n = a x b)
    let axc = cross(a, c);
    let axn = cross(a, n);
    for dim in 0..3 {
        let dsdb = -axc[dim] * inv_cn + sin_chi * axn[dim] / (n_mag * n_mag);
        forces[k * 3 + dim] += -prefactor * dsdb;
        forces[j * 3 + dim] -= -prefactor * dsdb;
    }
    (energy, forces)
}

#[cfg(test)]
mod tests {
    use crate::ff::compile::ExplicitTerms;
    use crate::ff::ir::Params;
    use crate::ff::potential::fixtures::{OneShot, assert_forces_are_negative_gradient};
    use molrs::op::F;

    /// The centre (atom 0, stored first) at the origin, its plane neighbours
    /// 1 and 2, and atom 3 lifted out of their plane by `h`.
    fn pyramid(h: F) -> Vec<F> {
        vec![0.0, 0.0, 0.0, 1.2, 0.0, 0.0, -0.6, 1.0, 0.0, -0.6, -1.0, h]
    }

    /// `½ · 143.9325 · koop · χ²`, χ the angle between the centre→3 bond and
    /// the plane of the centre, 1 and 2 (the xy plane here).
    #[test]
    fn energy_is_the_wilson_angle_formula_and_forces_its_gradient() {
        let koop = 0.05;
        let pot = ExplicitTerms::new("improper", "mmff_oop")
            .term(&[0, 1, 2, 3], Params::from_iter([("koop", koop)]))
            .compile()
            .unwrap();
        assert!(pot.energy_at(&pyramid(0.0)).abs() < 1e-12);
        let h = 0.4;
        let chi = (h / (0.36_f64 + 1.0 + h * h).sqrt()).asin();
        let want = 0.5 * 143.9325 * koop * chi * chi;
        let got = pot.energy_at(&pyramid(h));
        assert!((got - want).abs() < 1e-9 * want, "{got} vs {want}");
        let mut skew = pyramid(h);
        skew[2] = 0.1;
        skew[7] = 0.15;
        assert_forces_are_negative_gradient(&pot, &skew, 1e-5);
    }
}
