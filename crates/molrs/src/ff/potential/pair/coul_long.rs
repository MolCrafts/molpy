//! Real-space half of smooth particle-mesh Ewald (`pair coul/long/pme`).
//!
//! ```text
//! E(r) = (k / D) · qᵢqⱼ · erfc(α r) / r      r < r_cut,  else 0
//! ```
//!
//! `k` (`coulomb`), `D` (`dielectric`) and the Ewald splitting `α` (`alpha`)
//! are style parameters. `qᵢ`, `qⱼ` are the atoms' `charge`. The cutoff and
//! a 1-4 weight are the pair kernel's, as for every other pair form. The
//! reciprocal-space sum, the self energy and the exclusion correction stay
//! in [`PairCoulLongPme`](crate::ff::potential::kspace::PairCoulLongPme):
//! a form of one distance cannot carry them.

use crate::ff::potential::form_kernel::{ParamColumns, ScalarForm, eval_by_term};
use molrs::op::F;

/// Energy and `dE/dr` of one real-space Ewald pair at distance `r`.
///
/// `prefactor` is `coulomb / dielectric`. The pair kernel applies the cutoff
/// and any 1-4 weight; this is the bare term.
#[inline]
pub(crate) fn real_space_pair(prefactor: F, alpha: F, q1: F, q2: F, r: F) -> (F, F) {
    let alpha_r = alpha * r;
    let erfc = libm::erfc(alpha_r);
    let qq = prefactor * q1 * q2;
    let exp_term = 2.0 * alpha / std::f64::consts::PI.sqrt() * r * (-alpha_r * alpha_r).exp();
    let e = qq * erfc / r;
    // d/dr [erfc(αr)/r] = −(erfc(αr) + 2αr/√π · exp(−α²r²)) / r².
    let de = -qq * (erfc + exp_term) / (r * r);
    (e, de)
}

/// Real-space Ewald Coulomb, `pair coul/long/pme`'s pair form.
#[derive(Clone, Copy, Debug, Default)]
pub struct CoulLong;

impl ScalarForm for CoulLong {
    fn eval(&self, r: &[F], p: &ParamColumns<'_>, e: &mut [F], de: &mut [F]) {
        eval_by_term(self, r, p, e, de);
    }

    fn pointwise(&self, p: &ParamColumns<'_>) -> Option<impl Fn(F, &[F]) -> (F, F) + '_> {
        let prefactor = p.style_input("coulomb") / p.style_input("dielectric");
        let alpha = p.style_input("alpha");
        Some(move |r: F, x: &[F]| real_space_pair(prefactor, alpha, x[0], x[1], r))
    }

    fn inputs(&self) -> Vec<String> {
        ["q1", "q2", "coulomb", "dielectric", "alpha"]
            .map(String::from)
            .to_vec()
    }
}

#[cfg(test)]
mod tests {
    use std::sync::Arc;

    use super::*;
    use crate::ff::ir::Params;
    use crate::ff::potential::fixtures::OneShot;
    use crate::ff::potential::form_kernel::ScalarPair;
    use molrs::core::{Block, Frame};
    use ndarray::Array1;

    fn spec() -> crate::ff::ir::StyleSpec {
        crate::ff::ir::builtin_styles()
            .into_iter()
            .find(|spec| spec.category == "pair" && spec.name == "coul/long/pme")
            .expect("pair coul/long/pme")
    }

    /// Two charges, one candidate pair at weight `scale` when `is_14` (else
    /// 1). The caller places the atoms.
    fn kernel(q1: F, q2: F, alpha: F, cutoff: F, scale: F, is_14: bool) -> ScalarPair<CoulLong> {
        let spec = spec();
        let style = Params::from_iter([
            ("coulomb", 1.0),
            ("dielectric", 1.0),
            ("alpha", alpha),
            ("cutoff", cutoff),
            ("order", 4.0),
            ("grid_x", 4.0),
            ("grid_y", 4.0),
            ("grid_z", 4.0),
        ]);
        let (style, rows) = spec.gather(&style, &[]).unwrap();
        let refs: Vec<(&str, &Params)> = rows.iter().map(|(l, p)| (l.as_str(), p)).collect();
        let mut atoms = Block::new();
        atoms
            .insert("charge", Array1::from_vec(vec![q1, q2]).into_dyn(), None)
            .unwrap();
        let mut frame = Frame::new();
        frame.insert("atoms", atoms);
        let mut candidates = crate::ff::potential::pair::CandidatePairs::default();
        let w = if is_14 { scale } else { 1.0 };
        candidates.push(0, 1, w, w);
        ScalarPair::compiled(
            Arc::new(CoulLong),
            &spec,
            &style,
            &refs,
            &frame,
            &candidates,
        )
        .unwrap()
        .fused()
    }

    #[test]
    fn alpha_zero_is_the_bare_coulomb() {
        let pot = kernel(1.0, -1.0, 0.0, F::INFINITY, 1.0, false);
        let coords = [0.0, 0.0, 0.0, 2.0, 0.0, 0.0];
        let (e, forces) = pot.energy_forces_at(&coords);
        assert!((e - -0.5).abs() < 1e-12, "E got {e}");
        // Unlike charges attract: the force on the +x atom points toward −x.
        assert!((forces[3] - -0.25).abs() < 1e-12, "fxj {}", forces[3]);
    }

    #[test]
    fn past_the_cutoff_is_zero_and_a_1_4_row_takes_the_scale() {
        let far = kernel(1.0, 1.0, 0.0, 2.0, 1.0, false);
        let coords = [0.0, 0.0, 0.0, 3.0, 0.0, 0.0];
        assert_eq!(far.energy_at(&coords), 0.0);

        let full = kernel(1.0, 1.0, 0.25, 9.0, 1.0, true);
        let half = kernel(1.0, 1.0, 0.25, 9.0, 0.5, true);
        let coords = [0.0, 0.0, 0.0, 2.0, 0.0, 0.0];
        let (e_full, f_full) = full.energy_forces_at(&coords);
        let (e_half, f_half) = half.energy_forces_at(&coords);
        assert!(e_full.abs() > 1e-6, "the pair must interact");
        assert!(
            (e_half - 0.5 * e_full).abs() < 1e-12,
            "{e_half} vs {e_full}"
        );
        assert!((f_half[3] - 0.5 * f_full[3]).abs() < 1e-12);
    }

    #[test]
    fn numerical_gradient() {
        let pot = kernel(0.4, -0.7, 0.35, 9.0, 1.0, false);
        let coords = [0.2, -0.4, 0.1, 1.5, 0.6, -0.3];
        let (_, forces) = pot.energy_forces_at(&coords);
        let h = 1e-6;
        for dim in 0..coords.len() {
            let mut plus = coords;
            let mut minus = coords;
            plus[dim] += h;
            minus[dim] -= h;
            let fd = -(pot.energy_at(&plus) - pot.energy_at(&minus)) / (2.0 * h);
            assert!(
                (forces[dim] - fd).abs() < 1e-6,
                "comp {dim}: analytic {} vs fd {fd}",
                forces[dim]
            );
        }
    }
}
