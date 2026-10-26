//! CVFF improper (LAMMPS `improper_style cvff`), and LAMMPS
//! `dihedral_style harmonic`, the same function of the dihedral.

use crate::ff::potential::form_kernel::{ParamColumns, ScalarForm};
use molrs::op::F;

/// CVFF improper (LAMMPS `improper_style cvff`):
///
/// E(χ) = k · [1 + s · cos(n·χ)]
///
/// `sign` is s = ±1 (a sign, **not** a phase — hence its own canonical name)
/// and `periodicity` is the integer multiplicity n. The improper angle χ is the
/// dihedral angle defined by the quadruple I-J-K-L of the stored order, I the
/// centre (LAMMPS's symmetry atom for this style).
///
/// The same function of the dihedral is LAMMPS's `dihedral_style harmonic`
/// (`K[1 + d cos(nφ)]`), which this form prices too.
#[derive(Clone, Copy, Debug, Default)]
pub struct Cvff;

impl ScalarForm for Cvff {
    fn eval(&self, chi: &[F], p: &ParamColumns<'_>, e: &mut [F], de: &mut [F]) {
        let (k, d, n) = (p.input("k"), p.input("sign"), p.input("periodicity"));
        for t in 0..chi.len() {
            e[t] = k[t] * (1.0 + d[t] * (n[t] * chi[t]).cos());
            // dE/dχ = −K·d·n·sin(n·χ)
            de[t] = -k[t] * d[t] * n[t] * (n[t] * chi[t]).sin();
        }
    }

    fn inputs(&self) -> Vec<String> {
        ["k", "sign", "periodicity"].map(String::from).to_vec()
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

    fn single(k: F, d: F, n: F) -> Potentials {
        ExplicitTerms::new("improper", "cvff")
            .term(
                &[0, 1, 2, 3],
                Params::from_iter([("k", k), ("sign", d), ("periodicity", n)]),
            )
            .compile()
            .unwrap()
    }

    #[test]
    fn energy_phase() {
        // K[1 + d cos(nχ)], k=2,d=1,n=1: E(0)=4, E(π)=0.
        let p = single(2.0, 1.0, 1.0);
        assert!((p.energy_forces_at(&quad(0.0)).0 - 4.0).abs() < 1e-9);
        assert!(p.energy_forces_at(&quad(std::f64::consts::PI)).0.abs() < 1e-9);
    }

    #[test]
    fn numerical_gradient() {
        let pot = single(2.3, -1.0, 2.0);
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
        let pot = single(1.0, 1.0, 3.0);
        let coords: Vec<F> = vec![0.1, 1.0, 0.2, 0.0, 0.0, 0.0, 1.0, 0.0, -0.1, 1.2, -0.8, 0.5];
        let (_, f) = pot.energy_forces_at(&coords);
        for dim in 0..3 {
            let s: F = (0..4).map(|a| f[a * 3 + dim]).sum();
            assert!(s.abs() < 1e-9, "dim {dim} force sum {s}");
        }
    }

    /// LAMMPS `dihedral_style harmonic` is the same function of the dihedral,
    /// `K[1 + d·cos(nφ)]`: K = 2, d = −1, n = 2 at φ = 30°.
    #[test]
    fn dihedral_harmonic_is_the_lammps_formula() {
        let pots = ExplicitTerms::new("dihedral", "harmonic")
            .term(
                &[0, 1, 2, 3],
                Params::from_iter([("k", 2.0), ("sign", -1.0), ("periodicity", 2.0)]),
            )
            .compile()
            .unwrap();
        let phi: F = 30.0_f64.to_radians();
        let want = 2.0 * (1.0 - (2.0 * phi).cos());
        let got = pots.energy_at(&quad(phi));
        assert!((got - want).abs() < 1e-12, "{got} vs {want}");
    }
}
