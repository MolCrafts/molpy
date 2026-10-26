//! Class2 (COMPASS) 9-6 Lennard-Jones pair style (LAMMPS `lj/class2`).

use crate::ff::potential::form_kernel::{ParamColumns, ScalarForm, eval_by_term};
use molrs::op::F;

/// LAMMPS `pair_style lj/class2`: `E = ε [2(σ/r)⁹ − 3(σ/r)⁶]`, σ the
/// position of the minimum; unlike pairs mixed by the style's `mixing`
/// (absent: `sixthpower`, LAMMPS's rule for the style).
#[derive(Clone, Copy, Debug, Default)]
pub struct LjClass2;

impl ScalarForm for LjClass2 {
    fn eval(&self, r: &[F], p: &ParamColumns<'_>, e: &mut [F], de: &mut [F]) {
        eval_by_term(self, r, p, e, de);
    }

    fn pointwise(&self, p: &ParamColumns<'_>) -> Option<impl Fn(F, &[F]) -> (F, F) + '_> {
        let _ = p;
        Some(|r: F, x: &[F]| {
            let (eps, sigma) = (x[0], x[1]);
            let u = sigma / r;
            let u3 = u * u * u;
            let u6 = u3 * u3;
            let u9 = u6 * u3;
            (eps * (2.0 * u9 - 3.0 * u6), -18.0 * eps * (u9 - u6) / r)
        })
    }

    fn inputs(&self) -> Vec<String> {
        ["epsilon", "sigma"].map(String::from).to_vec()
    }
}

#[cfg(test)]
mod tests {

    use crate::ff::compile::ExplicitTerms;
    use crate::ff::ir::Params;
    use crate::ff::potential::Potentials;
    use crate::ff::potential::fixtures::OneShot;
    use molrs::core::Frame;
    use molrs::op::F;

    fn one(eps: F, sigma: F) -> Potentials {
        ExplicitTerms::new("pair", "lj/class2")
            .style_params(Params::from_iter([("cutoff", F::INFINITY)]))
            .term(
                &[0, 1],
                Params::from_iter([("epsilon", eps), ("sigma", sigma)]),
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
    fn energy_at_sigma_is_negative_eps() {
        // At r = sigma: E = eps(2 - 3) = -eps.
        let pot = one(0.5, 1.0);
        let coords: Vec<F> = vec![0.0, 0.0, 0.0, 1.0, 0.0, 0.0];
        assert!((pot.energy_at(&coords) - (-0.5)).abs() < 1e-12);
    }

    #[test]
    fn force_vanishes_at_minimum() {
        // Minimum of 2u^9 - 3u^6 is at u=1 (r=sigma); force is zero there.
        let pot = one(0.5, 1.0);
        let coords: Vec<F> = vec![0.0, 0.0, 0.0, 1.0, 0.0, 0.0];
        let (_, f) = pot.energy_forces_at(&coords);
        for fi in f {
            assert!(fi.abs() < 1e-9, "force {fi}");
        }
    }

    #[test]
    fn forces_match_finite_difference() {
        let pot = one(0.5, 1.0);
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
        let pot = one(0.5, 1.0);
        let coords: Vec<F> = vec![0.0, 0.0, 0.0, 1.2, 0.7, -0.4];
        let (_, f) = pot.energy_forces_at(&coords);
        for dim in 0..3 {
            assert!((f[dim] + f[3 + dim]).abs() < 1e-9, "dim {dim}");
        }
    }

    /// An unlike pair with no cross row is mixed by the style's `mixing`,
    /// absent `sixthpower` — LAMMPS's rule for `lj/class2` — at both doors.
    #[test]
    fn an_unlike_pair_mixes_sixthpower_by_default() {
        use crate::ff::compile::PotentialCompiler;
        use crate::ff::forcefield::ForceField;
        use crate::ff::ir::CombiningRule;
        use molrs::core::Block;
        use ndarray::Array1;

        let mut ff = ForceField::new("t");
        ff.set_special_bonds(crate::ff::ir::SpecialBonds::EXCLUDE_12_13);
        ff.def_style(
            "pair",
            "lj/class2",
            Params::from_iter([("cutoff", F::INFINITY)]),
        )
        .unwrap()
        .def_type(
            "A",
            &["A"],
            Params::from_iter([("epsilon", 0.1), ("sigma", 3.0)]),
        )
        .unwrap()
        .def_type(
            "B",
            &["B"],
            Params::from_iter([("epsilon", 0.4), ("sigma", 3.6)]),
        )
        .unwrap();
        let mut atoms = Block::new();
        atoms
            .insert(
                "type",
                Array1::from_vec(vec!["A".to_string(), "B".to_string()]).into_dyn(),
                None,
            )
            .unwrap();
        for (key, v) in [("x", [0.0, 3.4]), ("y", [0.0, 0.0]), ("z", [0.0, 0.0])] {
            atoms
                .insert(key, Array1::from_vec(v.to_vec()).into_dyn(), None)
                .unwrap();
        }
        let mut frame = Frame::new();
        frame.insert("atoms", atoms);

        let (eps, sigma) = CombiningRule::SixthPower.combine((0.1, 3.0), (0.4, 3.6));
        let s = sigma / 3.4;
        let want = eps * (2.0 * s.powi(9) - 3.0 * s.powi(6));
        let e = PotentialCompiler::new(&ff)
            .compile(&frame)
            .unwrap()
            .energy_at(&[0.0, 0.0, 0.0, 3.4, 0.0, 0.0]);
        assert!((e - want).abs() < 1e-12, "{e} vs {want}");
    }
}
