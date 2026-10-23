//! UFF torsion: `E = V/2 · (1 − cosTerm · cos(n·φ))` (RDKit `TorsionAngleContrib`).

use crate::ff::potential::form_kernel::{ParamColumns, ScalarForm};
use molrs::op::F;

/// UFF torsion, `V`, `order` (the periodicity n) and `cosTerm` (the sign of
/// the barrier, ±1) per-instance columns the UFF typifier bakes onto each
/// dihedral: `E = V/2 · (1 − cosTerm · cos nφ)`, `dE/dφ = V/2 · cosTerm · n ·
/// sin nφ`. The energy is even in φ, so the signed dihedral prices it as
/// RDKit's unsigned one does.
#[derive(Clone, Copy, Debug, Default)]
pub struct Uff;

impl ScalarForm for Uff {
    fn eval(&self, phi: &[F], p: &ParamColumns<'_>, e: &mut [F], de: &mut [F]) {
        let (v, order, cos_term) = (p.input("V"), p.input("order"), p.input("cosTerm"));
        for t in 0..phi.len() {
            let n = order[t];
            let (sin_n, cos_n) = (n * phi[t]).sin_cos();
            e[t] = v[t] / 2.0 * (1.0 - cos_term[t] * cos_n);
            de[t] = v[t] / 2.0 * cos_term[t] * n * sin_n;
        }
    }

    fn inputs(&self) -> Vec<String> {
        ["V", "order", "cosTerm"].map(String::from).to_vec()
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

    fn torsion(v: F, order: u8, cos_term: F) -> Potentials {
        ExplicitTerms::new("dihedral", "uff_torsion")
            .term(
                &[0, 1, 2, 3],
                Params::from_iter([("V", v), ("order", F::from(order)), ("cosTerm", cos_term)]),
            )
            .compile()
            .unwrap()
    }

    /// Planar anti arrangement: φ = 180°.
    fn anti() -> Vec<F> {
        vec![0.0, 0.0, 0.0, 1.5, 0.0, 0.0, 2.1, 1.4, 0.0, 3.6, 1.4, 0.0]
    }

    #[test]
    fn energy_is_half_v_times_one_minus_cos_term_cos_n_phi() {
        // cos(3·180°) = −1, so E = V/2·(1 + cosTerm).
        let (e_plus, _) = torsion(4.0, 3, 1.0).energy_forces_at(&anti());
        let (e_minus, _) = torsion(4.0, 3, -1.0).energy_forces_at(&anti());
        assert!((e_plus - 4.0).abs() < 1e-9);
        assert!(e_minus.abs() < 1e-9);
    }

    #[test]
    fn forces_are_the_negative_energy_gradient() {
        // A skew chain, no three atoms collinear, nothing planar.
        let coords = vec![0.0, 0.1, 0.2, 1.5, 0.0, 0.0, 2.1, 1.4, 0.3, 3.4, 1.7, 1.1];
        for (order, cos_term) in [(3, 1.0), (2, -1.0), (6, 1.0)] {
            assert_forces_are_negative_gradient(&torsion(4.0, order, cos_term), &coords, 1e-5);
        }
    }
}
