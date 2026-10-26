//! Buckingham pair style (LAMMPS `pair_style buck`).

use crate::ff::potential::form_kernel::{ParamColumns, ScalarForm, eval_by_term};
use molrs::op::F;

/// LAMMPS `pair_style buck`: `E = A e^{−r/ρ} − C/r⁶`. Its parameters do not
/// mix: an unlike pair needs its own cross row.
#[derive(Clone, Copy, Debug, Default)]
pub struct Buck;

impl ScalarForm for Buck {
    fn eval(&self, r: &[F], p: &ParamColumns<'_>, e: &mut [F], de: &mut [F]) {
        eval_by_term(self, r, p, e, de);
    }

    fn pointwise(&self, p: &ParamColumns<'_>) -> Option<impl Fn(F, &[F]) -> (F, F) + '_> {
        let _ = p;
        Some(|r: F, x: &[F]| {
            let (a, rho, c) = (x[0], x[1], x[2]);
            let exp_term = a * (-r / rho).exp();
            let r2 = r * r;
            let r6 = r2 * r2 * r2;
            (exp_term - c / r6, -exp_term / rho + 6.0 * c / (r6 * r))
        })
    }

    fn inputs(&self) -> Vec<String> {
        ["a", "rho", "c"].map(String::from).to_vec()
    }
}

#[cfg(test)]
mod tests {

    /// A flagged 1-4 pair is scaled by `lj14scale`, and scaled in **both**
    /// terms.
    ///
    /// Buckingham is `A·exp(-r/rho) − C/r⁶`. The energy is linear in `A` and
    /// in `C` separately, so scaling the pair means scaling both; scaling only
    /// the repulsive `A` — the obvious half-fix — leaves the dispersion term
    /// at full strength and gives a different number at every separation.
    /// Halving the weight must halve the energy exactly, whatever `r` is.
    #[test]
    fn a_1_4_pair_is_scaled_in_both_buckingham_terms() {
        use molrs::core::Block;
        use molrs::core::Frame;
        use ndarray::Array1;

        let build = |scale: f64| {
            let mut frame = Frame::new();
            let mut atoms = Block::new();
            atoms
                .insert(
                    "type",
                    Array1::from(vec!["a".to_string(), "a".to_string()]).into_dyn(),
                    None,
                )
                .unwrap();
            frame.insert("atoms", atoms);
            frame.insert(
                "pairs",
                crate::ff::potential::pair::fixtures::pricing_only(2, &[(0, 1)], &[true]),
            );

            let mut ff = crate::ff::forcefield::ForceField::new("t");
            ff.set_special_bonds(crate::ff::ir::SpecialBonds {
                lj: [0.0, 0.0, scale],
                coul: [0.0, 0.0, scale],
            });
            let mut tp = Params::new();
            tp.set("a", 12000.0);
            tp.set("rho", 0.31);
            tp.set("c", 280.0);
            ff.def_style("pair", "buck", Params::from_iter([("cutoff", F::INFINITY)]))
                .unwrap()
                .def_type("a", &["a"], tp)
                .unwrap();
            crate::ff::compile::PotentialCompiler::new(&ff)
                .compile(&frame)
                .expect("buck kernel")
        };

        // Two separations: one where repulsion dominates, one where dispersion
        // does. A half-fix passes at neither, and passing at both is what makes
        // this a statement about the pair rather than about one term.
        for r in [1.5_f64, 4.5] {
            let coords = vec![0.0, 0.0, 0.0, r, 0.0, 0.0];
            let full = build(1.0).energy_forces_at(&coords);
            let half = build(0.5).energy_forces_at(&coords);
            assert!(
                full.0.abs() > 1e-9,
                "r = {r}: the pair must carry energy, or this proves nothing"
            );
            assert!(
                (half.0 - 0.5 * full.0).abs() < 1e-12 * full.0.abs().max(1.0),
                "r = {r}: scaled {} vs half of {}",
                half.0,
                full.0
            );
            for k in 0..full.1.len() {
                assert!(
                    (half.1[k] - 0.5 * full.1[k]).abs() < 1e-12 * full.1[k].abs().max(1.0),
                    "r = {r}: force component {k}: {} vs half of {}",
                    half.1[k],
                    full.1[k]
                );
            }
        }
    }

    use crate::ff::compile::ExplicitTerms;
    use crate::ff::ir::Params;
    use crate::ff::potential::Potentials;
    use crate::ff::potential::fixtures::OneShot;
    use molrs::op::F;

    fn one() -> Potentials {
        ExplicitTerms::new("pair", "buck")
            .style_params(Params::from_iter([("cutoff", F::INFINITY)]))
            .term(
                &[0, 1],
                Params::from_iter([("a", 1000.0), ("rho", 0.3), ("c", 100.0)]),
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
            let ep = pot.energy_at(&cp);
            let em = pot.energy_at(&cm);
            num[k] = -(ep - em) / (2.0 * h); // force = -dE/dx
        }
        num
    }

    #[test]
    fn energy_matches_closed_form() {
        // A=1000, rho=0.3, C=100 at r=2.0:
        // E = 1000*exp(-2/0.3) - 100/64
        let pot = one();
        let coords: Vec<F> = vec![0.0, 0.0, 0.0, 2.0, 0.0, 0.0];
        let e = pot.energy_at(&coords);
        let expected = 1000.0 * (-2.0f64 / 0.3).exp() - 100.0 / 64.0;
        assert!((e - expected).abs() < 1e-9, "energy {e} vs {expected}");
    }

    #[test]
    fn forces_match_finite_difference() {
        let pot = one();
        // off-axis geometry exercising all three components
        let coords: Vec<F> = vec![0.1, -0.2, 0.05, 1.3, 0.6, -0.3];
        let (_, analytical) = pot.energy_forces_at(&coords);
        let numerical = numerical_forces(&pot, &coords);
        for k in 0..coords.len() {
            assert!(
                (analytical[k] - numerical[k]).abs() < 1e-5,
                "k={k} analytical={} numerical={}",
                analytical[k],
                numerical[k]
            );
        }
    }

    #[test]
    fn newtons_third_law() {
        let pot = one();
        let coords: Vec<F> = vec![0.0, 0.0, 0.0, 1.2, 0.7, -0.4];
        let (_, f) = pot.energy_forces_at(&coords);
        for dim in 0..3 {
            assert!((f[dim] + f[3 + dim]).abs() < 1e-9, "dim {dim}");
        }
    }
}
