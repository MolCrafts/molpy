//! Buffered Coulomb pair style with a hard distance cutoff (`pair coul/cut`).

use crate::ff::potential::form_kernel::{ParamColumns, ScalarForm, eval_by_term};
use molrs::op::F;

/// Buffered Coulomb pair potential with a hard distance cutoff — `pair/coul/cut`:
///
/// ```text
/// E(r) = k · qᵢqⱼ / (D · (r + δ))      r < r_cut,  else 0
/// ```
///
/// Every constant in that formula is **force-field data, read from the style**:
/// the Coulomb constant `k`, the dielectric `D` and the buffering distance `δ`.
/// The form holds none of them.
///
/// # One kernel, two force fields
///
/// `δ = 0` degenerates the buffered form into the textbook Coulomb `k·qᵢqⱼ/r`,
/// which is what OPLS, LAMMPS and every other non-buffered force field mean. MMFF's
/// electrostatics (Halgren, *MMFF.I* eq. 5) is the *same kernel* at
/// `k = 332.0716`, `D = 1.0`, `δ = 0.05 Å` — a parameterization, not a kernel of
/// its own. The buffer is what keeps `E` finite at `r = 0`, where MMFF's charges
/// sit on the nuclei.
///
/// # Why `k` is a style param and not a shared constant
///
/// MMFF uses Halgren's **332.0716**; OPLS/LAMMPS use CODATA's
/// [`COULOMB_REAL`](molrs::core::constants::COULOMB_REAL) = **332.06371**. The 2.4e-5
/// relative difference is worth 0.0036 kcal/mol on caffeine's −150.48 kcal/mol
/// electrostatic term — **above** the 1e-3 RDKit parity tolerance. Both values are
/// correct; the force field decides. A form holding either one would be choosing a
/// force field for its caller, so there is no default: a style that does not say is
/// an [`Err`] (`MissingParam`), and so is one that states no `dielectric`.
///
/// `qᵢ`, `qⱼ` are the atoms' `charge` (`q1`, `q2`); any 1-4 or special-bonds
/// weight is the pair kernel's.
#[derive(Clone, Copy, Debug, Default)]
pub struct CoulCut;

impl ScalarForm for CoulCut {
    fn eval(&self, r: &[F], p: &ParamColumns<'_>, e: &mut [F], de: &mut [F]) {
        eval_by_term(self, r, p, e, de);
    }

    fn pointwise(&self, p: &ParamColumns<'_>) -> Option<impl Fn(F, &[F]) -> (F, F) + '_> {
        // `k/D` is a style-level constant: hoisted out of the loop. At D = 1
        // this is exactly `k` (IEEE: `x / 1.0 == x`).
        let k_over_d = p.style_input("coulomb") / p.style_input("dielectric");
        let delta = p.style_input("delta");
        Some(move |r: F, x: &[F]| {
            let qq = k_over_d * (x[0] * x[1]);
            let r_buf = r + delta;
            // The force is `−dE/dr · r̂` with `dE/dr = −k·qᵢqⱼ/(D·(r+δ)²)`, i.e.
            // the BUFFERED distance squared in the denominator — not `r²`.
            (qq / r_buf, -qq / (r_buf * r_buf))
        })
    }

    fn inputs(&self) -> Vec<String> {
        ["q1", "q2", "coulomb", "dielectric", "delta"]
            .map(String::from)
            .to_vec()
    }
}

#[cfg(test)]
mod tests {
    use crate::ff::compile::ExplicitTerms;
    use crate::ff::ir::Params;
    use crate::ff::potential::Potentials;
    use crate::ff::potential::fixtures::OneShot;
    use molrs::core::constants::COULOMB_REAL;
    use molrs::op::F;

    /// MMFF's parameterization: Halgren's constant, unit dielectric, 0.05 Å buffer.
    const MMFF: (F, F, F) = (332.0716, 1.0, 0.05);

    /// One pair of charges `1` and `qiqj` (their product `qiqj`) at `k`, `D`,
    /// `δ` and `cutoff`.
    fn pair(qiqj: F, (k, d, delta): (F, F, F), cutoff: F) -> Potentials {
        ExplicitTerms::new("pair", "coul/cut")
            .style_params(Params::from_iter([
                ("coulomb", k),
                ("dielectric", d),
                ("delta", delta),
                ("cutoff", cutoff),
            ]))
            .atoms([vec![0, 1]])
            .charges(vec![1.0, qiqj])
            .compile()
            .unwrap()
    }

    fn unbuffered(qiqj: F, cutoff: F) -> Potentials {
        pair(qiqj, (COULOMB_REAL, 1.0, 0.0), cutoff)
    }

    #[test]
    fn energy_and_sign() {
        // Unit positive charges 2 Å apart: E = k_e / 2.
        let pot = unbuffered(1.0, F::INFINITY);
        let coords: Vec<F> = vec![0.0, 0.0, 0.0, 2.0, 0.0, 0.0];
        let (e, forces) = pot.energy_forces_at(&coords);
        assert!((e - COULOMB_REAL / 2.0).abs() < 1e-9, "E got {e}");
        // Like charges repel: force on j (+x atom) points +x (away from i).
        assert!(
            forces[3] > 0.0,
            "like charges should repel, fxj={}",
            forces[3]
        );
        // Unlike charges attract.
        let pot2 = unbuffered(-1.0, F::INFINITY);
        let (_, f2) = pot2.energy_forces_at(&coords);
        assert!(f2[3] < 0.0, "unlike charges should attract, fxj={}", f2[3]);
    }

    #[test]
    fn cutoff_and_zero_distance() {
        let pot = unbuffered(1.0, 1.5);
        // 2 Å apart, cutoff 1.5 -> excluded, E=0.
        let far: Vec<F> = vec![0.0, 0.0, 0.0, 2.0, 0.0, 0.0];
        assert_eq!(pot.energy_forces_at(&far).0, 0.0);
        // Coincident atoms, UNBUFFERED -> skipped, finite.
        let zero: Vec<F> = vec![0.0, 0.0, 0.0, 0.0, 0.0, 0.0];
        let (e, f) = pot.energy_forces_at(&zero);
        assert_eq!(e, 0.0);
        assert!(f.iter().all(|x| x.is_finite()));
    }

    /// The buffer's reason to exist: at r = 0 the energy is `k·qᵢqⱼ/(D·δ)`, finite.
    #[test]
    fn buffered_zero_distance_is_finite() {
        let (k, d, delta) = MMFF;
        let pot = pair(-0.25, (k, d, delta), F::INFINITY);
        let zero: Vec<F> = vec![0.0, 0.0, 0.0, 0.0, 0.0, 0.0];
        let (e, f) = pot.energy_forces_at(&zero);
        assert!((e - k * -0.25 / (d * delta)).abs() < 1e-9, "E got {e}");
        assert!(f.iter().all(|x| x.is_finite()), "forces {f:?}");
    }

    /// `δ = 0` reproduces the unbuffered kernel bit-for-bit (r = 2 is exact in f64).
    #[test]
    fn delta_zero_is_the_unbuffered_kernel() {
        let pot = unbuffered(-0.48, F::INFINITY);
        let coords: Vec<F> = vec![0.0, 0.0, 0.0, 2.0, 0.0, 0.0];
        let (e, _) = pot.energy_forces_at(&coords);
        assert_eq!(e, COULOMB_REAL * -0.48 / 2.0);
    }

    /// The force must use `(r + δ)²`, not `r²` — buffering the energy and forgetting
    /// the gradient passes every energy assertion and fails here.
    #[test]
    fn numerical_gradient() {
        for (k, d, delta) in [(COULOMB_REAL, 1.0, 0.0), MMFF] {
            let pot = pair(0.8, (k, d, delta), F::INFINITY);
            let coords: Vec<F> = vec![0.0, 0.0, 0.0, 1.7, 0.3, -0.4];
            let (_, forces) = pot.energy_forces_at(&coords);
            let h = 1e-6;
            for dim in 0..coords.len() {
                let mut cp = coords.clone();
                let mut cm = coords.clone();
                cp[dim] += h;
                cm[dim] -= h;
                let fd = -(pot.energy_forces_at(&cp).0 - pot.energy_forces_at(&cm).0) / (2.0 * h);
                assert!(
                    (forces[dim] - fd).abs() < 1e-5,
                    "delta {delta}, comp {dim}: analytic {} vs fd {fd}",
                    forces[dim]
                );
            }
            // Newton's third law.
            for dim in 0..3 {
                assert!((forces[dim] + forces[3 + dim]).abs() < 1e-9);
            }
        }
    }

    /// `coulomb` is the force field's to state: a style without it is
    /// [`IrError::MissingParam`](crate::ff::ir::IrError::MissingParam). The
    /// spec's defaults — `dielectric` 1 (LAMMPS's), `delta` 0, `cutoff` ∞ —
    /// apply when the style leaves them out, exactly as stating them does.
    #[test]
    fn coulomb_and_dielectric_are_required_and_the_spec_defaults_apply() {
        use crate::ff::ir::IrError;
        // Each case states the cutoff, required of its own
        // (`param_reads::a_declared_cutoff_is_stated_and_inf_prices_every_pair`).
        let terms = |style: &[(&str, f64)]| {
            let mut params = Params::from_iter(style.iter().copied());
            params.set("cutoff", F::INFINITY);
            ExplicitTerms::new("pair", "coul/cut")
                .style_params(params)
                .atoms([vec![0, 1]])
                .charges(vec![0.5, -0.4])
                .compile()
        };
        let err = terms(&[]).map(|_| ()).unwrap_err();
        assert_eq!(
            err.ir(),
            Some(&IrError::MissingParam {
                style: "coul/cut".into(),
                type_: String::new(),
                param: "coulomb".into(),
            })
        );
        // `dielectric` is declared too, never assumed to be 1.
        let err = terms(&[("coulomb", COULOMB_REAL)]).map(|_| ()).unwrap_err();
        assert_eq!(
            err.ir(),
            Some(&IrError::MissingParam {
                style: "coul/cut".into(),
                type_: String::new(),
                param: "dielectric".into(),
            })
        );
        let coords = [0.0, 0.0, 0.0, 2.5, 0.0, 0.0];
        let bare = terms(&[("coulomb", COULOMB_REAL), ("dielectric", 1.0)]).unwrap();
        let stated = terms(&[
            ("coulomb", COULOMB_REAL),
            ("dielectric", 1.0),
            ("delta", 0.0),
        ])
        .unwrap();
        let want = COULOMB_REAL * 0.5 * -0.4 / 2.5;
        assert!((bare.energy_at(&coords) - want).abs() < 1e-12);
        assert_eq!(bare.energy_at(&coords), stated.energy_at(&coords));
    }
}
