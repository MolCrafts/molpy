//! OPLS 4-cosine (Fourier) proper dihedral (LAMMPS `dihedral_style opls`).

use crate::ff::potential::form_kernel::{ParamColumns, ScalarForm};
use molrs::op::F;

/// OPLS 4-cosine (Fourier) proper dihedral:
///
/// E(φ) = ½[ K1(1+cos φ) + K2(1−cos 2φ) + K3(1+cos 3φ) + K4(1−cos 4φ) ]
///
/// The OPLS-AA torsion form (Jorgensen, Maxwell & Tirado-Rives,
/// J. Am. Chem. Soc. 1996, 118, 11225), LAMMPS's `K_n` (energies; the ½ is
/// the style's, not the coefficients').
#[derive(Clone, Copy, Debug, Default)]
pub struct Opls;

impl ScalarForm for Opls {
    fn eval(&self, phi: &[F], p: &ParamColumns<'_>, e: &mut [F], de: &mut [F]) {
        let (f1, f2, f3, f4) = (p.input("k1"), p.input("k2"), p.input("k3"), p.input("k4"));
        for t in 0..phi.len() {
            let (s1, c1) = phi[t].sin_cos();
            let (s2, c2) = (2.0 * phi[t]).sin_cos();
            let (s3, c3) = (3.0 * phi[t]).sin_cos();
            let (s4, c4) = (4.0 * phi[t]).sin_cos();
            e[t] = 0.5
                * (f1[t] * (1.0 + c1)
                    + f2[t] * (1.0 - c2)
                    + f3[t] * (1.0 + c3)
                    + f4[t] * (1.0 - c4));
            // dE/dφ = ½[ −K1 sinφ + 2K2 sin2φ − 3K3 sin3φ + 4K4 sin4φ ]
            de[t] = 0.5 * (-f1[t] * s1 + 2.0 * f2[t] * s2 - 3.0 * f3[t] * s3 + 4.0 * f4[t] * s4);
        }
    }

    fn inputs(&self) -> Vec<String> {
        ["k1", "k2", "k3", "k4"].map(String::from).to_vec()
    }
}

#[cfg(test)]
mod tests {
    use crate::ff::compile::ExplicitTerms;
    use crate::ff::ir::Params;
    use crate::ff::potential::Potentials;
    use crate::ff::potential::fixtures::OneShot;
    use crate::ff::potential::flat_coords::compute_dihedral;
    use molrs::op::F;

    /// Build a 4-atom geometry at a target dihedral φ in the xy/z layout used
    /// by the MMFF torsion test, then read it back.
    fn quad(phi: F) -> Vec<F> {
        // i=(0,1,0) j=(0,0,0) k=(1,0,0); l rotates around the j-k (x) axis.
        let (s, c) = phi.sin_cos();
        vec![0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 1.0, c, s]
    }

    fn single(f1: F, f2: F, f3: F, f4: F) -> Potentials {
        ExplicitTerms::new("dihedral", "opls")
            .term(
                &[0, 1, 2, 3],
                Params::from_iter([("k1", f1), ("k2", f2), ("k3", f3), ("k4", f4)]),
            )
            .compile()
            .unwrap()
    }

    #[test]
    fn per_term_phase() {
        // φ at the reference geometry: l=(1, cos φ, sin φ) gives dihedral = φ.
        let at = |c: &[F]| compute_dihedral(c, 0, 1, 2, 3);
        // F1 term: E(0)=F1, E(π)=0.
        let e0 = single(2.0, 0.0, 0.0, 0.0).energy_forces_at(&quad(0.0)).0;
        let epi = single(2.0, 0.0, 0.0, 0.0)
            .energy_forces_at(&quad(std::f64::consts::PI))
            .0;
        assert!((e0 - 2.0).abs() < 1e-9, "F1 E(0) got {e0}");
        assert!(epi.abs() < 1e-9, "F1 E(pi) got {epi}");
        // F2 term: E(0)=0, E(π/2)=F2.
        let c = quad(std::f64::consts::FRAC_PI_2);
        assert!((at(&c) - std::f64::consts::FRAC_PI_2).abs() < 1e-9);
        let e = single(0.0, 3.0, 0.0, 0.0).energy_forces_at(&c).0;
        assert!((e - 3.0).abs() < 1e-9, "F2 E(pi/2) got {e}");
    }

    #[test]
    fn numerical_gradient() {
        let pot = single(1.3, -0.7, 0.9, 0.4);
        // A generic, non-degenerate geometry.
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
            let fd = -(ep - em) / (2.0 * h); // force = -dE/dx
            assert!(
                (forces[d] - fd).abs() < 1e-5,
                "comp {d}: analytic {} vs fd {fd}",
                forces[d]
            );
        }
    }

    #[test]
    fn newtons_third_law() {
        let pot = single(1.0, 0.5, 0.3, 0.2);
        let coords: Vec<F> = vec![0.1, 1.0, 0.2, 0.0, 0.0, 0.0, 1.0, 0.0, -0.1, 1.2, -0.8, 0.5];
        let (_, f) = pot.energy_forces_at(&coords);
        for dim in 0..3 {
            let s: F = (0..4).map(|a| f[a * 3 + dim]).sum();
            assert!(s.abs() < 1e-9, "dim {dim} force sum {s}");
        }
    }

    /// A `dihedral periodic` term table on `dihedral opls` (the two share the
    /// spellings `k1 … k4`) is refused by name rather than priced as an OPLS
    /// quartet, as is any number the style does not declare.
    #[test]
    fn a_periodic_bag_on_opls_is_refused() {
        let err = ExplicitTerms::new("dihedral", "opls")
            .term(
                &[0, 1, 2, 3],
                Params::from_iter([("k1", 1.0), ("periodicity1", 3.0), ("phase1", 0.0)]),
            )
            .compile()
            .map(|_| ())
            .unwrap_err();
        assert!(
            matches!(
                err.ir(),
                Some(crate::ff::ir::IrError::BadValue { style, param, .. })
                    if style == "opls" && ["periodicity1", "phase1"].contains(&param.as_str())
            ),
            "{err}"
        );
    }
}
