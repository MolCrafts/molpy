//! The two halves of LAMMPS's `pair_style lj/charmm/coul/charmm`:
//! `lj/charmm` and `coul/charmm`.
//!
//! molrs splits `lj/charmm/coul/charmm` as it splits `lj/cut/coul/cut`:
//!
//! * `pair lj/charmm` — Lennard-Jones 12-6 with CHARMM's energy switch, per
//!   type `epsilon`, `sigma`, `epsilon14`, `sigma14`;
//! * `pair coul/charmm` — Coulomb with the same switch.
//!
//! Both follow `pair_lj_charmm_coul_charmm.cpp` term for term. With
//! `r_in` = `inner`, `r_c` = `cutoff` and
//!
//! ```text
//! S(r)  = (r_c² − r²)² (r_c² + 2r² − 3r_in²) / (r_c² − r_in²)³     r_in < r < r_c
//! ```
//!
//! (1 below `r_in`, 0 from `r_c` on — the pair kernel truncates at `cutoff`):
//!
//! ```text
//! E_lj   = 4ε[(σ/r)¹² − (σ/r)⁶] · S(r)        F_lj = −dE_lj/dr  (consistent)
//! E_coul = C qᵢqⱼ / r · S(r)                   F_coul = C qᵢqⱼ / r² · S(r)
//! ```
//!
//! `epsilon14` / `sigma14` (absent → `epsilon` / `sigma`, as a two-number
//! `pair_coeff` line in LAMMPS) are not read by the pair style: LAMMPS prices
//! them only inside `dihedral_style charmm`, which molrs routes to the 1-4
//! exceptions kernel ([`PairExceptions`](super::PairExceptions)). A 1-4 pair
//! this style meets on the `pairs` list is weighted by `special_bonds` with
//! the regular `epsilon` / `sigma`, as LAMMPS's pair style does.
//!
//! Cross pairs: an explicit cross row, else the style's `mixing` (LAMMPS's
//! default for the CHARMM styles is `arithmetic`).

use crate::ff::potential::form_kernel::{ParamColumns, ScalarForm, StyleRefusal, eval_by_term};
use molrs::op::F;

/// CHARMM's energy switch between `inner` and `cutoff`.
#[derive(Clone, Copy, Debug)]
struct Switch {
    inner2: F,
    outer2: F,
    denom: F,
}

impl Switch {
    /// The switch of a batch's style parameters.
    fn of(p: &ParamColumns<'_>) -> Self {
        let (inner, outer) = (p.style_input("inner"), p.style_input("cutoff"));
        let (inner2, outer2) = (inner * inner, outer * outer);
        let span = outer2 - inner2;
        Self {
            inner2,
            outer2,
            denom: span * span * span,
        }
    }

    /// `inner` and `cutoff` both finite and positive, `inner` below `cutoff`
    /// (LAMMPS: "Pair inner cutoff >= Pair outer cutoff").
    fn check(p: &ParamColumns<'_>) -> Result<(), StyleRefusal> {
        let (inner, outer) = (p.style_input("inner"), p.style_input("cutoff"));
        if !(inner.is_finite() && outer.is_finite() && inner > 0.0) {
            return Err(StyleRefusal::new(
                "inner",
                format!("= {inner} with cutoff = {outer}: both must be finite and positive"),
            ));
        }
        if inner >= outer {
            return Err(StyleRefusal::new(
                "inner",
                format!("= {inner} is not below the outer cutoff {outer}"),
            ));
        }
        Ok(())
    }

    /// LAMMPS's `switch1`.
    #[inline]
    fn s1(&self, rsq: F) -> F {
        (self.outer2 - rsq) * (self.outer2 - rsq) * (self.outer2 + 2.0 * rsq - 3.0 * self.inner2)
            / self.denom
    }

    /// LAMMPS's `switch2` (= −r·dS/dr).
    #[inline]
    fn s2(&self, rsq: F) -> F {
        12.0 * rsq * (self.outer2 - rsq) * (rsq - self.inner2) / self.denom
    }
}

/// LAMMPS `pair_style lj/charmm/coul/charmm`, van-der-Waals half: the
/// switched 12-6, its force the true gradient (LAMMPS adds `philj · switch2`).
#[derive(Clone, Copy, Debug, Default)]
pub struct LjCharmm;

impl ScalarForm for LjCharmm {
    fn eval(&self, r: &[F], p: &ParamColumns<'_>, e: &mut [F], de: &mut [F]) {
        eval_by_term(self, r, p, e, de);
    }

    fn pointwise(&self, p: &ParamColumns<'_>) -> Option<impl Fn(F, &[F]) -> (F, F) + '_> {
        let law = self.pair_law(p)?;
        Some(move |r: F, x: &[F]| {
            let (e, fpair) = law(r * r, x);
            (e, -fpair * r)
        })
    }

    /// The law in `r²`, term for term LAMMPS's.
    fn pair_law(&self, p: &ParamColumns<'_>) -> Option<impl Fn(F, &[F]) -> (F, F) + '_> {
        let switch = Switch::of(p);
        Some(move |rsq: F, x: &[F]| {
            let (eps, sigma) = (x[0], x[1]);
            // LAMMPS's `lj1..lj4`: `48εσ¹²`, `24εσ⁶`, `4εσ¹²`, `4εσ⁶`.
            let s2 = sigma * sigma;
            let s6 = s2 * s2 * s2;
            let s12 = s6 * s6;
            let (lj1, lj2) = (48.0 * eps * s12, 24.0 * eps * s6);
            let (lj3, lj4) = (4.0 * eps * s12, 4.0 * eps * s6);
            let r2inv = 1.0 / rsq;
            let r6inv = r2inv * r2inv * r2inv;
            let mut forcelj = r6inv * (lj1 * r6inv - lj2);
            let mut philj = r6inv * (lj3 * r6inv - lj4);
            if rsq > switch.inner2 {
                let s1 = switch.s1(rsq);
                forcelj = forcelj * s1 + philj * switch.s2(rsq);
                philj *= s1;
            }
            // LAMMPS's `fpair = forcelj · r2inv`.
            (philj, forcelj * r2inv)
        })
    }

    fn inputs(&self) -> Vec<String> {
        ["epsilon", "sigma", "inner", "cutoff"]
            .map(String::from)
            .to_vec()
    }

    fn check_style(&self, p: &ParamColumns<'_>) -> Result<(), StyleRefusal> {
        Switch::check(p)
    }
}

/// LAMMPS `pair_style lj/charmm/coul/charmm`, Coulomb half.
///
/// The Coulomb force is LAMMPS's: the switched force, **not** the gradient of
/// the switched energy (LAMMPS's `forcecoul *= switch1` drops the `E·S′`
/// term), so the style's spec says its force is no gradient and the form
/// writes the derivative that force is built from. Inside `r_in` the two
/// agree; between `r_in` and `r_c` they differ, exactly as in LAMMPS.
/// `C = coulomb / dielectric`, LAMMPS's `qqrd2e`; `qᵢ`, `qⱼ` the atoms'
/// `charge`.
#[derive(Clone, Copy, Debug, Default)]
pub struct CoulCharmm;

impl ScalarForm for CoulCharmm {
    fn eval(&self, r: &[F], p: &ParamColumns<'_>, e: &mut [F], de: &mut [F]) {
        eval_by_term(self, r, p, e, de);
    }

    fn pointwise(&self, p: &ParamColumns<'_>) -> Option<impl Fn(F, &[F]) -> (F, F) + '_> {
        let k = p.style_input("coulomb") / p.style_input("dielectric");
        let switch = Switch::of(p);
        Some(move |r: F, x: &[F]| {
            let rsq = r * r;
            let r2inv = 1.0 / rsq;
            let mut forcecoul = k * (x[0] * x[1]) * r2inv.sqrt();
            let mut ecoul = forcecoul;
            if rsq > switch.inner2 {
                // LAMMPS switches the force as it switches the energy; the
                // `E·S′` term of the energy's gradient is not in it.
                let s1 = switch.s1(rsq);
                forcecoul *= s1;
                ecoul *= s1;
            }
            (ecoul, -forcecoul * r2inv * r)
        })
    }

    fn inputs(&self) -> Vec<String> {
        ["q1", "q2", "coulomb", "dielectric", "inner", "cutoff"]
            .map(String::from)
            .to_vec()
    }

    fn check_style(&self, p: &ParamColumns<'_>) -> Result<(), StyleRefusal> {
        Switch::check(p)
    }
}

#[cfg(test)]
mod tests {
    use crate::ff::compile::ExplicitTerms;
    use crate::ff::ir::{IrError, Params};
    use crate::ff::potential::fixtures::OneShot;
    use crate::ff::potential::{CompileError, Potentials};
    use molrs::op::F;

    /// LAMMPS's switch, written out from the manual's formula.
    fn s(r: F, ri: F, rc: F) -> F {
        if r <= ri {
            1.0
        } else if r >= rc {
            0.0
        } else {
            let (r2, ri2, rc2) = (r * r, ri * ri, rc * rc);
            (rc2 - r2).powi(2) * (rc2 + 2.0 * r2 - 3.0 * ri2) / (rc2 - ri2).powi(3)
        }
    }

    fn lj(eps: F, sigma: F, r: F) -> F {
        let x = (sigma / r).powi(6);
        4.0 * eps * (x * x - x)
    }

    fn two(r: F) -> Vec<F> {
        vec![0.0, 0.0, 0.0, r, 0.0, 0.0]
    }

    fn lj_charmm(ri: F, rc: F, eps: F, sigma: F) -> Result<Potentials, CompileError> {
        ExplicitTerms::new("pair", "lj/charmm")
            .style_params(Params::from_iter([("inner", ri), ("cutoff", rc)]))
            .term(
                &[0, 1],
                Params::from_iter([("epsilon", eps), ("sigma", sigma)]),
            )
            .compile()
    }

    fn coul_charmm(ri: F, rc: F, k: F, qq: F) -> Result<Potentials, CompileError> {
        ExplicitTerms::new("pair", "coul/charmm")
            .style_params(Params::from_iter([
                ("inner", ri),
                ("cutoff", rc),
                ("coulomb", k),
                ("dielectric", 1.0),
            ]))
            .atoms([vec![0, 1]])
            .charges(vec![1.0, qq])
            .compile()
    }

    #[test]
    fn lj_hand_values_inside_across_and_beyond_the_switch() {
        let (eps, sigma, ri, rc) = (0.2, 3.1, 4.0, 6.0);
        let k = lj_charmm(ri, rc, eps, sigma).unwrap();
        for r in [3.0, 3.9, 4.5, 5.2, 5.99, 6.0, 7.0] {
            let want = lj(eps, sigma, r) * s(r, ri, rc);
            let got = k.energy_at(&two(r));
            assert!(
                (got - want).abs() <= 1e-13 * want.abs().max(1e-12),
                "r = {r}: {got} vs {want}"
            );
        }
    }

    #[test]
    fn lj_force_is_the_gradient_through_the_switch() {
        let k = lj_charmm(4.0, 6.0, 0.2, 3.1).unwrap();
        for r in [3.3, 4.6, 5.5] {
            let (_, f) = k.energy_forces_at(&two(r));
            let h = 1e-6;
            let fd = -(k.energy_at(&two(r + h)) - k.energy_at(&two(r - h))) / (2.0 * h);
            assert!((f[3] - fd).abs() < 1e-7, "r = {r}: {} vs {fd}", f[3]);
            assert!((f[0] + f[3]).abs() < 1e-12);
        }
    }

    #[test]
    fn coulomb_hand_values_and_lammps_switched_force() {
        let (ri, rc, k, qq) = (4.0, 6.0, 332.06371, -0.3);
        let pot = coul_charmm(ri, rc, k, qq).unwrap();
        for r in [2.5, 4.5, 5.5, 6.5] {
            let (e, f) = pot.energy_forces_at(&two(r));
            let sw = s(r, ri, rc);
            let want = k * qq / r * sw;
            assert!(
                (e - want).abs() <= 1e-13 * want.abs().max(1e-12),
                "{e} {want}"
            );
            // LAMMPS's force: k qq / r² · S, the switched force.
            let want_f = k * qq / (r * r) * sw;
            assert!((f[3] - want_f).abs() <= 1e-12 * want_f.abs().max(1e-12));
        }
        // Inside `inner` the force is the gradient.
        let h = 1e-6;
        let fd = -(pot.energy_at(&two(3.0 + h)) - pot.energy_at(&two(3.0 - h))) / (2.0 * h);
        assert!((pot.energy_forces_at(&two(3.0)).1[3] - fd).abs() < 1e-7);
    }

    #[test]
    fn inner_must_be_below_the_cutoff() {
        let bad = |err: CompileError| match err.ir() {
            Some(IrError::BadValue { param, .. }) => param.clone(),
            _ => panic!("not BadValue: {err}"),
        };
        let err = coul_charmm(6.0, 6.0, 1.0, 1.0).map(|_| ()).unwrap_err();
        assert_eq!(bad(err), "inner");
        let err = lj_charmm(7.0, 6.0, 0.1, 3.0).map(|_| ()).unwrap_err();
        assert_eq!(bad(err), "inner");
        let err = ExplicitTerms::new("pair", "lj/charmm")
            .style_params(Params::from_iter([("cutoff", 10.0)]))
            .term(
                &[0, 1],
                Params::from_iter([("epsilon", 0.1), ("sigma", 3.0)]),
            )
            .compile()
            .map(|_| ())
            .unwrap_err();
        assert_eq!(
            err.ir(),
            Some(&IrError::MissingParam {
                style: "lj/charmm".into(),
                type_: String::new(),
                param: "inner".into()
            })
        );
    }
}
