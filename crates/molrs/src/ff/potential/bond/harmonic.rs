//! Harmonic bond (LAMMPS `bond_style harmonic`).

use crate::ff::potential::form_kernel::{ParamColumns, ScalarForm};
use molrs::op::F;

/// LAMMPS `bond_style harmonic`: `E = k (r − r0)²`.
///
/// `k` is LAMMPS's `K`, energy/length², and carries the usual ½: there is no
/// hidden factor, so a `bond_coeff t K r0` line is `k = K` here.
#[derive(Clone, Copy, Debug, Default)]
pub struct Harmonic;

impl ScalarForm for Harmonic {
    fn eval(&self, r: &[F], p: &ParamColumns<'_>, e: &mut [F], de: &mut [F]) {
        let (k, r0) = (p.input("k"), p.input("r0"));
        for t in 0..r.len() {
            let dr = r[t] - r0[t];
            e[t] = k[t] * dr * dr;
            de[t] = 2.0 * k[t] * dr;
        }
    }

    fn inputs(&self) -> Vec<String> {
        ["k", "r0"].map(String::from).to_vec()
    }
}

#[cfg(test)]
mod tests {
    use crate::ff::compile::ExplicitTerms;
    use crate::ff::ir::Params;
    use crate::ff::potential::fixtures::OneShot;
    use molrs::op::F;

    #[test]
    fn test_bond_harmonic_energy_and_force() {
        // LAMMPS bond_style harmonic: E = K(r − r0)², F = −2K(r − r0).
        let pot = ExplicitTerms::new("bond", "harmonic")
            .term(&[0, 1], Params::from_iter([("k", 150.0), ("r0", 1.5)]))
            .compile()
            .unwrap();
        let coords: Vec<F> = vec![0.0, 0.0, 0.0, 2.0, 0.0, 0.0];

        let (e, forces) = pot.energy_forces_at(&coords);
        assert_eq!(e, 150.0 * 0.5 * 0.5);
        assert!((forces[0] - 150.0).abs() < 1e-12);
        assert!((forces[3] + 150.0).abs() < 1e-12);
    }
}
