//! Morse pair style (LAMMPS `pair_style morse`).

use crate::ff::potential::form_kernel::{ParamColumns, ScalarForm, eval_by_term};
use molrs::op::F;

/// LAMMPS `pair_style morse`: `E = D₀ [e^{−2α(r − r0)} − 2 e^{−α(r − r0)}]`,
/// written `D₀ [(1 − e^{−α(r − r0)})² − 1]`. Its parameters do not mix: an
/// unlike pair needs its own cross row.
#[derive(Clone, Copy, Debug, Default)]
pub struct Morse;

impl ScalarForm for Morse {
    fn eval(&self, r: &[F], p: &ParamColumns<'_>, e: &mut [F], de: &mut [F]) {
        eval_by_term(self, r, p, e, de);
    }

    fn pointwise(&self, p: &ParamColumns<'_>) -> Option<impl Fn(F, &[F]) -> (F, F) + '_> {
        let _ = p;
        Some(|r: F, x: &[F]| {
            let (d0, alpha, r0) = (x[0], x[1], x[2]);
            let y = (-alpha * (r - r0)).exp();
            let one_my = 1.0 - y;
            (d0 * (one_my * one_my - 1.0), 2.0 * d0 * alpha * y * one_my)
        })
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

    fn one() -> Potentials {
        ExplicitTerms::new("pair", "morse")
            .term(
                &[0, 1],
                Params::from_iter([("d0", 5.0), ("alpha", 1.5), ("r0", 3.0)]),
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
    fn well_minimum_is_minus_d0() {
        // At r = r0: E = D0 (0 - 1) = -D0, force zero.
        let pot = one();
        let coords: Vec<F> = vec![0.0, 0.0, 0.0, 3.0, 0.0, 0.0];
        let (e, f) = pot.energy_forces_at(&coords);
        assert!((e - (-5.0)).abs() < 1e-12, "energy {e}");
        for fi in f {
            assert!(fi.abs() < 1e-9);
        }
    }

    #[test]
    fn forces_match_finite_difference() {
        let pot = one();
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
