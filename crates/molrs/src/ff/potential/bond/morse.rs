//! Morse bond (LAMMPS `bond_style morse`).

use crate::ff::potential::form_kernel::{ParamColumns, ScalarForm};
use molrs::op::F;

/// LAMMPS `bond_style morse`: `E = D₀ [1 − e^{−α(r − r0)}]²`.
#[derive(Clone, Copy, Debug, Default)]
pub struct Morse;

impl ScalarForm for Morse {
    fn eval(&self, r: &[F], p: &ParamColumns<'_>, e: &mut [F], de: &mut [F]) {
        let (d0, alpha, r0) = (p.input("d0"), p.input("alpha"), p.input("r0"));
        for t in 0..r.len() {
            let y = (-alpha[t] * (r[t] - r0[t])).exp();
            let one_my = 1.0 - y;
            e[t] = d0[t] * one_my * one_my;
            // dE/dr = 2 D α y (1 − y)
            de[t] = 2.0 * d0[t] * alpha[t] * y * one_my;
        }
    }

    fn inputs(&self) -> Vec<String> {
        ["d0", "alpha", "r0"].map(String::from).to_vec()
    }
}

#[cfg(test)]
mod tests {
    use crate::ff::compile::ExplicitTerms;
    use crate::ff::ir::Params;
    use crate::ff::potential::Potentials;
    use crate::ff::potential::fixtures::OneShot;
    use molrs::op::F;

    fn morse() -> Potentials {
        ExplicitTerms::new("bond", "morse")
            .term(
                &[0, 1],
                Params::from_iter([("d0", 100.0), ("alpha", 2.0), ("r0", 1.5)]),
            )
            .compile()
            .unwrap()
    }

    fn numerical_forces(pot: &Potentials, coords: &[F]) -> Vec<F> {
        let h = 1e-6;
        let mut num = vec![0.0; coords.len()];
        for k in 0..coords.len() {
            let mut cp = coords.to_vec();
            let mut cm = coords.to_vec();
            cp[k] += h;
            cm[k] -= h;
            num[k] = -(pot.energy_at(&cp) - pot.energy_at(&cm)) / (2.0 * h);
        }
        num
    }

    #[test]
    fn energy_zero_and_force_zero_at_equilibrium() {
        let pot = morse();
        let coords: Vec<F> = vec![0.0, 0.0, 0.0, 1.5, 0.0, 0.0];
        let (e, f) = pot.energy_forces_at(&coords);
        assert!(e.abs() < 1e-12, "energy {e}");
        for fi in f {
            assert!(fi.abs() < 1e-9);
        }
    }

    #[test]
    fn forces_match_finite_difference() {
        let pot = morse();
        let coords: Vec<F> = vec![0.1, -0.2, 0.05, 1.3, 0.6, -0.3];
        let (_, analytical) = pot.energy_forces_at(&coords);
        let numerical = numerical_forces(&pot, &coords);
        for k in 0..coords.len() {
            assert!(
                (analytical[k] - numerical[k]).abs() < 1e-5,
                "k={k} a={} n={}",
                analytical[k],
                numerical[k]
            );
        }
    }
}
