//! Harmonic improper (LAMMPS `improper_style harmonic`).

use crate::ff::potential::form_kernel::{ParamColumns, ScalarForm};
use molrs::op::F;

/// Harmonic improper (LAMMPS `improper_style harmonic`):
///
/// E(χ) = K · (χ − χ₀)²
///
/// Following LAMMPS, the improper angle χ is the **unsigned** dihedral of the
/// quadruple I-J-K-L: χ = |φ| ∈ [0, π], where φ is the signed dihedral the
/// kernel hands the form. Hence dχ/dφ = sign(φ), so dE/dφ =
/// 2K(χ − χ₀)·sign(φ), with sign(0) = 0 (stationary, planar). `chi0` is the
/// equilibrium angle in **degrees** (0 for a planar centre), as LAMMPS takes
/// it. `k` is LAMMPS's `K` (energy/rad², the ½ included).
///
/// # Atom order
///
/// LAMMPS's: the improper angle is the dihedral I-J-K-L of the stored order,
/// and the first atom I is the centre (LAMMPS's "atom of symmetry" for this
/// style; CHARMM writes its impropers this way).
#[derive(Clone, Copy, Debug, Default)]
pub struct Harmonic;

impl ScalarForm for Harmonic {
    fn eval(&self, phi: &[F], p: &ParamColumns<'_>, e: &mut [F], de: &mut [F]) {
        let (k, chi0) = (p.input("k"), p.input("chi0"));
        for t in 0..phi.len() {
            let dchi = phi[t].abs() - chi0[t].to_radians();
            e[t] = k[t] * dchi * dchi;
            let sign = if phi[t] > 0.0 {
                1.0
            } else if phi[t] < 0.0 {
                -1.0
            } else {
                0.0
            };
            de[t] = 2.0 * k[t] * dchi * sign;
        }
    }

    fn inputs(&self) -> Vec<String> {
        ["k", "chi0"].map(String::from).to_vec()
    }
}

#[cfg(test)]
mod tests {
    use crate::ff::compile::ExplicitTerms;
    use crate::ff::ir::Params;
    use crate::ff::potential::Potentials;
    use crate::ff::potential::fixtures::OneShot;
    use molrs::op::F;

    fn quad(phi: F) -> Vec<F> {
        let (s, c) = phi.sin_cos();
        vec![0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 1.0, c, s]
    }

    fn single(k: F, chi0_deg: F) -> Potentials {
        ExplicitTerms::new("improper", "harmonic")
            .term(
                &[0, 1, 2, 3],
                Params::from_iter([("k", k), ("chi0", chi0_deg)]),
            )
            .compile()
            .unwrap()
    }

    #[test]
    fn energy_minimum_at_chi0() {
        // χ₀ = 60°, geometry at φ = 60° → E = 0.
        let p = single(3.0, 60.0);
        let e = p.energy_forces_at(&quad(60.0_f64.to_radians())).0;
        assert!(e.abs() < 1e-9, "E at χ₀ got {e}");
        // φ = 90° → E = K(30°)² in radians.
        let e2 = p.energy_forces_at(&quad(90.0_f64.to_radians())).0;
        let want = 3.0 * (30.0_f64.to_radians()).powi(2);
        assert!((e2 - want).abs() < 1e-9, "E got {e2} want {want}");
    }

    #[test]
    fn numerical_gradient() {
        // χ₀ chosen so φ stays well away from 0 and ±π under perturbation.
        let pot = single(5.0, 20.0);
        let coords: Vec<F> = vec![0.1, 1.0, 0.2, 0.0, 0.0, 0.0, 1.0, 0.0, -0.1, 1.2, -0.8, 0.5];
        let (_, forces) = pot.energy_forces_at(&coords);
        let h = 1e-6;
        for d in 0..coords.len() {
            let mut cp = coords.clone();
            let mut cm = coords.clone();
            cp[d] += h;
            cm[d] -= h;
            let ep = pot.energy_forces_at(&cp).0;
            let em = pot.energy_forces_at(&cm).0;
            let fd = -(ep - em) / (2.0 * h);
            assert!(
                (forces[d] - fd).abs() < 1e-5,
                "comp {d}: analytic {} vs fd {fd}",
                forces[d]
            );
        }
    }

    #[test]
    fn newtons_third_law() {
        let pot = single(5.0, 20.0);
        let coords: Vec<F> = vec![0.1, 1.0, 0.2, 0.0, 0.0, 0.0, 1.0, 0.0, -0.1, 1.2, -0.8, 0.5];
        let (_, f) = pot.energy_forces_at(&coords);
        for dim in 0..3 {
            let s: F = (0..4).map(|a| f[a * 3 + dim]).sum();
            assert!(s.abs() < 1e-9, "dim {dim} force sum {s}");
        }
    }
}
