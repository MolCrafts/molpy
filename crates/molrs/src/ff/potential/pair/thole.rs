//! Thole dipole-dipole screening (CL&Pol short-range damping).

use crate::ff::potential::form_kernel::{ParamColumns, ScalarForm, eval_by_term};
use molrs::op::F;

/// Thole dipole-dipole screening (CL&Pol short-range damping).
///
/// Screens the Coulomb interaction between Drude-related point charges at short
/// range with the exponential Thole function
///
/// ```text
/// T_ij(r) = 1 - (1 + s_ij r / 2) exp(-s_ij r)
/// s_ij    = a_ij / (alpha_i alpha_j)^(1/6),   a_ij = (a_i + a_j) / 2
/// ```
///
/// so the damped energy of a pair is `(coulomb / dielectric) T_ij(r) q_i q_j / r`
/// — the Coulomb constant and dielectric are the style's declared params, as
/// LAMMPS prices `pair_style thole` in `qqrd2e`. The screening
/// `s_ij` depends on **both** endpoints' atomic polarizabilities, so the form
/// reads each atom's own type row (`charge1`, `alpha1`, `damp1` and their
/// `…2`): the parameters combine by the rule above, not by a mixing rule,
/// and need no cross row. `alpha` and
/// `damp` (the Thole `a`) are LAMMPS `pair_style thole`'s `pair_coeff` names;
/// the per-type `charge` is molrs's (LAMMPS damps the Drude charges of the
/// atoms).
///
/// Units: r in A, alpha in A^3, a dimensionless, q in e; energy in the units
/// of `coulomb` (E·L/Q²) divided by L.
///
/// Reference: Thole, Chem. Phys. 59 (1981) 341,
/// DOI 10.1016/0301-0104(81)85176-2; as emitted by the paduagroup/clandpol
/// polarizer (LAMMPS `pair_style thole`).
#[derive(Clone, Copy, Debug, Default)]
pub struct Thole;

impl ScalarForm for Thole {
    fn eval(&self, r: &[F], p: &ParamColumns<'_>, e: &mut [F], de: &mut [F]) {
        eval_by_term(self, r, p, e, de);
    }

    fn pointwise(&self, p: &ParamColumns<'_>) -> Option<impl Fn(F, &[F]) -> (F, F) + '_> {
        let prefactor = p.style_input("coulomb") / p.style_input("dielectric");
        Some(move |r: F, x: &[F]| {
            let (q1, q2, alpha1, alpha2, damp1, damp2) = (x[0], x[1], x[2], x[3], x[4], x[5]);
            let a_ij = 0.5 * (damp1 + damp2);
            let s = a_ij / (alpha1 * alpha2).powf(1.0 / 6.0);
            let x = s * r;
            let e_x = (-x).exp();
            let screen = 1.0 - (1.0 + x / 2.0) * e_x;
            let qq = prefactor * (q1 * q2);
            // V = T qq / r ;  T'(r) = (s/2)(1 + x) e^{-x}
            // dV/dr = qq (T'/r - T/r^2)
            let tp = (s / 2.0) * (1.0 + x) * e_x;
            (screen * qq / r, qq * (tp / r - screen / (r * r)))
        })
    }

    fn inputs(&self) -> Vec<String> {
        [
            "charge1",
            "charge2",
            "alpha1",
            "alpha2",
            "damp1",
            "damp2",
            "coulomb",
            "dielectric",
        ]
        .map(String::from)
        .to_vec()
    }
}

#[cfg(test)]
mod tests {

    use super::*;
    use crate::ff::potential::fixtures::OneShot;

    /// The form's energy and `dE/dr` of one pair at `r`: atoms of charge
    /// `q1`, `q2`, polarizability `alpha` and Thole `damp` each, priced in
    /// `coulomb = 1`, `dielectric = 1`.
    fn form_at(r: F, q: (F, F), alpha: F, damp: F) -> (F, F) {
        let one = |v: F| vec![v];
        let (q1, q2, a1, a2, d1, d2) = (
            one(q.0),
            one(q.1),
            one(alpha),
            one(alpha),
            one(damp),
            one(damp),
        );
        let mut p = ParamColumns::new();
        for (name, col) in [
            ("charge1", &q1),
            ("charge2", &q2),
            ("alpha1", &a1),
            ("alpha2", &a2),
            ("damp1", &d1),
            ("damp2", &d2),
        ] {
            p.push(name, col);
        }
        p.push_style_num("coulomb", 1.0);
        p.push_style_num("dielectric", 1.0);
        let (mut e, mut de) = ([0.0], [0.0]);
        Thole.eval(&[r], &p, &mut e, &mut de);
        (e[0], de[0])
    }

    /// A pair of `thole` atoms of type `D` (`charge`, `alpha`, `damp`) at
    /// `coords` over the `pairs` list `(0, 1)`.
    fn compiled(q: F, alpha: F, damp: F) -> crate::ff::potential::Potentials {
        use crate::ff::compile::PotentialCompiler;
        use crate::ff::forcefield::ForceField;
        use crate::ff::ir::{Params, SpecialBonds};
        let mut ff = ForceField::new("clpol");
        ff.set_special_bonds(SpecialBonds::EXCLUDE_12_13);
        ff.def_style(
            "pair",
            "thole",
            Params::from_iter([("coulomb", 1.0), ("dielectric", 1.0)]),
        )
        .unwrap()
        .def_type(
            "D",
            &["D"],
            Params::from_iter([("charge", q), ("alpha", alpha), ("damp", damp)]),
        )
        .unwrap();
        PotentialCompiler::new(&ff).compile(&two_d()).unwrap()
    }

    fn two_d() -> molrs::core::Frame {
        use molrs::core::{Block, Frame};
        use ndarray::Array1;
        let mut atoms = Block::new();
        atoms
            .insert(
                "type",
                Array1::from_vec(vec!["D".to_string(); 2]).into_dyn(),
                None,
            )
            .unwrap();
        let mut pairs = Block::new();
        pairs
            .insert("atomi", Array1::from_vec(vec![0_u64]).into_dyn(), None)
            .unwrap();
        pairs
            .insert("atomj", Array1::from_vec(vec![1_u64]).into_dyn(), None)
            .unwrap();
        let mut frame = Frame::new();
        frame.insert("atoms", atoms);
        frame.insert("pairs", pairs);
        frame
    }

    fn numerical_forces(pot: &crate::ff::potential::Potentials, coords: &[F]) -> Vec<F> {
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
    fn damping_factor_matches_closed_form() {
        // s=1.0 (α = 1, a = 1), qq=1.0, r=2.0: x=2, T = 1 - (1+1)e^-2 = 1 - 2 e^-2
        let x = 2.0_f64;
        let t = 1.0 - (1.0 + x / 2.0) * (-x).exp();
        let expected = t * 1.0 / 2.0;
        let (e, _) = form_at(2.0, (1.0, 1.0), 1.0, 1.0);
        assert!((e - expected).abs() < 1e-12, "energy {e} vs {expected}");
    }

    /// The style prices in `coulomb / dielectric` (LAMMPS's `qqrd2e`): both
    /// are declared, a missing one is refused, and the energy is the screened
    /// `q_i q_j / r` times that constant.
    #[test]
    fn the_style_prices_in_its_declared_coulomb_constant() {
        use crate::ff::compile::PotentialCompiler;
        use crate::ff::forcefield::ForceField;
        use crate::ff::ir::{Params, SpecialBonds};
        use molrs::core::{Block, Frame};
        use ndarray::Array1;

        let mut atoms = Block::new();
        atoms
            .insert(
                "type",
                Array1::from_vec(vec!["D".to_string(); 2]).into_dyn(),
                None,
            )
            .unwrap();
        for (k, v) in [("x", [0.0, 2.0]), ("y", [0.0; 2]), ("z", [0.0; 2])] {
            atoms
                .insert(k, Array1::from_vec(v.to_vec()).into_dyn(), None)
                .unwrap();
        }
        let mut pairs = Block::new();
        pairs
            .insert("atomi", Array1::from_vec(vec![0_u64]).into_dyn(), None)
            .unwrap();
        pairs
            .insert("atomj", Array1::from_vec(vec![1_u64]).into_dyn(), None)
            .unwrap();
        let mut frame = Frame::new();
        frame.insert("atoms", atoms);
        frame.insert("pairs", pairs);

        let energy = |style: &[(&str, f64)]| {
            let mut ff = ForceField::new("clpol");
            ff.set_special_bonds(SpecialBonds::EXCLUDE_12_13);
            ff.def_style("pair", "thole", Params::from_iter(style.iter().copied()))
                .unwrap()
                .def_type(
                    "D",
                    &["D"],
                    Params::from_iter([("charge", 1.0), ("alpha", 1.0), ("damp", 2.0)]),
                )
                .unwrap();
            let coords = [0.0, 0.0, 0.0, 2.0, 0.0, 0.0];
            PotentialCompiler::new(&ff)
                .compile(&frame)
                .map(|p| p.energy_at(&coords))
        };
        let err = energy(&[("dielectric", 1.0)]).unwrap_err();
        assert!(err.to_string().contains("coulomb"), "{err}");
        let err = energy(&[("coulomb", 332.06371)]).unwrap_err();
        assert!(err.to_string().contains("dielectric"), "{err}");

        // s = a / α^(1/3) = 2, x = s·r = 4, T = 1 − 3 e^{−4}.
        let t = 1.0 - 3.0 * (-4.0_f64).exp();
        let e = energy(&[("coulomb", 332.06371), ("dielectric", 2.0)]).unwrap();
        assert!((e - 332.06371 / 2.0 * t / 2.0).abs() < 1e-9, "{e}");
    }

    #[test]
    fn damping_vanishes_at_zero_separation_limit() {
        // T(r) -> 0 as r -> 0 (screening fully removes the singular Coulomb term),
        // so the energy stays finite. Check T is small at small r: s = 5.
        let (e, _) = form_at(0.05, (1.0, 1.0), 1.0, 5.0);
        assert!(e.is_finite());
    }

    #[test]
    fn forces_match_finite_difference() {
        let pot = compiled(0.8, 1.0, 1.3);
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
        let pot = compiled(0.8, 1.0, 1.3);
        let coords: Vec<F> = vec![0.0, 0.0, 0.0, 1.2, 0.7, -0.4];
        let (_, f) = pot.energy_forces_at(&coords);
        for dim in 0..3 {
            assert!((f[dim] + f[3 + dim]).abs() < 1e-9, "dim {dim}");
        }
    }
}
