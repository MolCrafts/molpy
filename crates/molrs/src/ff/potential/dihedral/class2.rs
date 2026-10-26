//! Class2 (COMPASS / PCFF) proper dihedral, core torsion term (LAMMPS
//! `dihedral_style class2`).

use crate::ff::potential::form_kernel::{ParamColumns, ScalarForm};
use molrs::op::F;

/// Class2 (COMPASS / PCFF) proper dihedral — core torsion term:
///
/// E(φ) = Σ_{n=1..3} K_n · [1 − cos(n·φ − φ_n)]
///
/// `K_n` are force constants (energy) and `φ_n` the per-term phases in
/// **degrees**, as LAMMPS `dihedral_style class2` takes them. The optional
/// class2 cross terms (mbt / ebt / at / aat / bb13), which LAMMPS writes as
/// separate coeff lines, are not part of this style's per-type params (its
/// codec writes them at zero and refuses them otherwise).
#[derive(Clone, Copy, Debug, Default)]
pub struct Class2;

impl ScalarForm for Class2 {
    fn eval(&self, phi: &[F], p: &ParamColumns<'_>, e: &mut [F], de: &mut [F]) {
        let terms = [
            (p.input("k1"), p.input("phi1")),
            (p.input("k2"), p.input("phi2")),
            (p.input("k3"), p.input("phi3")),
        ];
        for t in 0..phi.len() {
            let (mut energy, mut de_dphi): (F, F) = (0.0, 0.0);
            for (m, (kn, pn)) in terms.iter().enumerate() {
                let n = (m + 1) as F;
                let arg = n * phi[t] - pn[t].to_radians();
                energy += kn[t] * (1.0 - arg.cos());
                // dE/dφ = K_n·n·sin(n·φ − φ_n)
                de_dphi += kn[t] * n * arg.sin();
            }
            e[t] = energy;
            de[t] = de_dphi;
        }
    }

    fn inputs(&self) -> Vec<String> {
        ["k1", "phi1", "k2", "phi2", "k3", "phi3"]
            .map(String::from)
            .to_vec()
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

    /// Three `(K_n, φ_n)` terms, `φ_n` in radians (stored in degrees).
    fn single(terms: [(F, F); 3]) -> Potentials {
        let mut row = Params::new();
        for (m, (k, phase)) in terms.into_iter().enumerate() {
            row.set(&format!("k{}", m + 1), k);
            row.set(&format!("phi{}", m + 1), phase.to_degrees());
        }
        ExplicitTerms::new("dihedral", "class2")
            .term(&[0, 1, 2, 3], row)
            .compile()
            .unwrap()
    }

    #[test]
    fn energy_phase() {
        // Single term K1[1−cos(φ−φ1)], φ1=0: E(0)=0, E(π)=2K1.
        let t = [(1.5, 0.0), (0.0, 0.0), (0.0, 0.0)];
        assert!(single(t).energy_forces_at(&quad(0.0)).0.abs() < 1e-9);
        let epi = single(t).energy_forces_at(&quad(std::f64::consts::PI)).0;
        assert!((epi - 3.0).abs() < 1e-9, "E(pi) got {epi}");
    }

    #[test]
    fn numerical_gradient() {
        let pot = single([(1.3, 0.0), (-0.7, std::f64::consts::PI), (0.4, 0.0)]);
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
        let pot = single([(1.0, 0.0), (0.5, 1.0), (0.3, 0.5)]);
        let coords: Vec<F> = vec![0.1, 1.0, 0.2, 0.0, 0.0, 0.0, 1.0, 0.0, -0.1, 1.2, -0.8, 0.5];
        let (_, f) = pot.energy_forces_at(&coords);
        for dim in 0..3 {
            let s: F = (0..4).map(|a| f[a * 3 + dim]).sum();
            assert!(s.abs() < 1e-9, "dim {dim} force sum {s}");
        }
    }
}
