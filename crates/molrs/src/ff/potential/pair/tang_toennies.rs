//! Tang-Toennies charge / induced-dipole damping (CL&Pol short-range damping).

use crate::ff::potential::form_kernel::{ParamColumns, ScalarForm, StyleRefusal, eval_by_term};
use molrs::op::F;

/// Tang-Toennies charge / induced-dipole damping (CL&Pol short-range damping).
///
/// Damps the Coulomb interaction between a charge and an induced dipole (a Drude
/// shell) at short range, preventing the polarization catastrophe:
///
/// ```text
/// f_n(r) = 1 - c exp(-b r) sum_{k=0}^{n} (b r)^k / k!
/// ```
///
/// so the damped pair energy is `(coulomb / dielectric) f_n(r) q_i q_j / r`,
/// the Coulomb constant and dielectric being the style's declared params (as
/// LAMMPS prices `coul/tt` in `qqrd2e`). The derivative collapses
/// to a single term: `f'_n(r) = c b exp(-b r) (b r)^n / n!`. CL&Pol canonical
/// settings: `order = 4`, `b = 4.5` (1/A), `c = 1.0` — the pair style's params,
/// `order` a non-negative integer; the charges are each atom's own type row's
/// `charge` (`charge1`, `charge2`).
///
/// Reference: Tang & Toennies, J. Chem. Phys. 80 (1984) 3726,
/// DOI 10.1063/1.447150; as emitted by paduagroup/clandpol `coul_tt`.
#[derive(Clone, Copy, Debug, Default)]
pub struct TangToennies;

impl ScalarForm for TangToennies {
    fn eval(&self, r: &[F], p: &ParamColumns<'_>, e: &mut [F], de: &mut [F]) {
        eval_by_term(self, r, p, e, de);
    }

    fn pointwise(&self, p: &ParamColumns<'_>) -> Option<impl Fn(F, &[F]) -> (F, F) + '_> {
        let prefactor = p.style_input("coulomb") / p.style_input("dielectric");
        let (b, c) = (p.style_input("b"), p.style_input("c"));
        let n = p.style_input("order") as usize;
        Some(move |r: F, x: &[F]| {
            // `(f_n(r), f'_n(r))` — damping factor and its radial derivative:
            // series = sum_{k=0}^n (br)^k / k!, accumulated term-by-term.
            let br = b * r;
            let mut term = 1.0; // k = 0
            let mut series = term;
            for k in 1..=n {
                term *= br / (k as F);
                series += term;
            }
            // term is now (br)^n / n! after the loop's final iteration (k = n).
            let ex = (-br).exp();
            let f = 1.0 - c * ex * series;
            let fp = c * b * ex * term; // c b e^{-br} (br)^n / n!
            let qq = prefactor * (x[0] * x[1]);
            // V = f qq / r ; dV/dr = qq (f'/r - f/r^2)
            (f * qq / r, qq * (fp / r - f / (r * r)))
        })
    }

    fn inputs(&self) -> Vec<String> {
        [
            "charge1",
            "charge2",
            "coulomb",
            "dielectric",
            "b",
            "c",
            "order",
        ]
        .map(String::from)
        .to_vec()
    }

    /// The damping order `n` is a non-negative integer.
    fn check_style(&self, p: &ParamColumns<'_>) -> Result<(), StyleRefusal> {
        let order = p.style_input("order");
        if order < 0.0 || order.fract() != 0.0 {
            return Err(StyleRefusal::new(
                "order",
                format!("= {order} is not a non-negative integer"),
            ));
        }
        Ok(())
    }
}

#[cfg(test)]
mod tests {

    use super::*;
    use crate::ff::potential::fixtures::OneShot;

    /// A pair of `coul/tt` atoms, of type `D` (charge `q`) and of type `E`
    /// (charge one), over the `pairs` list `(0, 1)`, damped by
    /// `(b, order, c)` in `coulomb = 1`, `dielectric = 1`.
    fn compiled(q: F, (b, order, c): (F, F, F)) -> crate::ff::potential::Potentials {
        use crate::ff::compile::PotentialCompiler;
        use crate::ff::forcefield::ForceField;
        use crate::ff::ir::{Params, SpecialBonds};
        use molrs::core::{Block, Frame};
        use ndarray::Array1;
        let mut ff = ForceField::new("clpol");
        ff.set_special_bonds(SpecialBonds::EXCLUDE_12_13);
        ff.def_style(
            "pair",
            "coul/tt",
            Params::from_iter([
                ("coulomb", 1.0),
                ("dielectric", 1.0),
                ("b", b),
                ("order", order),
                ("c", c),
            ]),
        )
        .unwrap()
        .def_type("D", &["D"], Params::from_iter([("charge", q)]))
        .unwrap()
        .def_type("E", &["E"], Params::from_iter([("charge", 1.0)]))
        .unwrap();
        let mut atoms = Block::new();
        atoms
            .insert(
                "type",
                Array1::from_vec(vec!["D".to_string(), "E".to_string()]).into_dyn(),
                None,
            )
            .unwrap();
        let mut frame = Frame::new();
        frame.insert("atoms", atoms);
        PotentialCompiler::new(&ff).compile(&frame).unwrap()
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
    fn damping_matches_closed_form() {
        // n=2, b=1, c=1, r=2: series = 1 + 2 + 2 = 5; f = 1 - e^-2 * 5
        let pot = compiled(1.0, (1.0, 2.0, 1.0));
        let coords: Vec<F> = vec![0.0, 0.0, 0.0, 2.0, 0.0, 0.0];
        let f = 1.0 - (-2.0f64).exp() * 5.0;
        let expected = f * 1.0 / 2.0;
        assert!((pot.energy_at(&coords) - expected).abs() < 1e-12);
    }

    #[test]
    fn forces_match_finite_difference() {
        let pot = compiled(-0.7, (4.5, 4.0, 1.0));
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

    /// `order` is the damping's term count: a non-negative integer.
    #[test]
    fn order_must_be_a_non_negative_integer() {
        let mut p = ParamColumns::new();
        p.push_style_num("order", 2.5);
        let err = TangToennies.check_style(&p).unwrap_err();
        assert_eq!(err.param, "order");
        let mut p = ParamColumns::new();
        p.push_style_num("order", -1.0);
        assert!(TangToennies.check_style(&p).is_err());
        let mut p = ParamColumns::new();
        p.push_style_num("order", 4.0);
        assert!(TangToennies.check_style(&p).is_ok());
    }
}
