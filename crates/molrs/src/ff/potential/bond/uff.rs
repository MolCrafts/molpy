//! UFF bond stretch: `E = ½ · kb · (r − r0)²` (RDKit `BondStretchContrib`).

use crate::ff::potential::form_kernel::{ParamColumns, ScalarForm};
use molrs::op::F;

/// Harmonic UFF bond stretch, `kb` and `r0` per-instance columns the UFF
/// typifier bakes onto each bond.
#[derive(Clone, Copy, Debug, Default)]
pub struct Uff;

impl ScalarForm for Uff {
    fn eval(&self, r: &[F], p: &ParamColumns<'_>, e: &mut [F], de: &mut [F]) {
        let (kb, r0) = (p.input("kb"), p.input("r0"));
        for t in 0..r.len() {
            let dr = r[t] - r0[t];
            e[t] = 0.5 * kb[t] * dr * dr;
            de[t] = kb[t] * dr;
        }
    }

    fn inputs(&self) -> Vec<String> {
        ["kb", "r0"].map(String::from).to_vec()
    }
}

#[cfg(test)]
mod tests {
    use crate::ff::compile::ExplicitTerms;
    use crate::ff::ir::Params;
    use crate::ff::potential::Potentials;
    use crate::ff::potential::fixtures::OneShot;
    use crate::ff::potential::fixtures::assert_forces_are_negative_gradient;
    use molrs::op::F;

    fn stretched(kb: F, r0: F, r: F) -> (Potentials, Vec<F>) {
        let pot = ExplicitTerms::new("bond", "uff_bond")
            .term(&[0, 1], Params::from_iter([("kb", kb), ("r0", r0)]))
            .compile()
            .unwrap();
        (pot, vec![0.0, 0.0, 0.0, r, 0.0, 0.0])
    }

    #[test]
    fn energy_is_half_kb_times_the_squared_stretch() {
        let (pot, coords) = stretched(700.0, 1.5, 1.6);
        let (e, f) = pot.energy_forces_at(&coords);
        assert!((e - 0.5 * 700.0 * 0.01).abs() < 1e-9);
        // The pull is along x, opposite on the two atoms.
        assert!((f[0] - 700.0 * 0.1).abs() < 1e-9);
        assert!((f[3] + 700.0 * 0.1).abs() < 1e-9);
    }

    #[test]
    fn energy_and_force_vanish_at_the_rest_length() {
        let (pot, coords) = stretched(700.0, 1.5, 1.5);
        let (e, f) = pot.energy_forces_at(&coords);
        assert_eq!(e, 0.0);
        assert!(f.iter().all(|x| x.abs() < 1e-12));
    }

    #[test]
    fn forces_are_the_negative_energy_gradient() {
        let (pot, _) = stretched(700.0, 1.5, 1.5);
        let coords = vec![0.1, -0.2, 0.3, 1.4, 0.5, -0.6];
        assert_forces_are_negative_gradient(&pot, &coords, 1e-5);
    }
}
