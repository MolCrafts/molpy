//! UFF angle bend (RDKit `AngleBendContrib`).

use crate::ff::potential::form_kernel::{ParamColumns, ScalarForm};
use molrs::op::F;

/// UFF angle bend, per-instance `ka`, `order` and, for `order = 0`, the
/// Fourier coefficients `c0`, `c1`, `c2` the UFF typifier derives from θ₀:
///
/// ```text
/// order 0:  E = ka (c0 + c1 cos θ + c2 cos 2θ)
/// order 1:  E = ka (1 + cos θ)             (linear centres: minimal at 180°)
/// order n:  E = ka (1 − cos nθ) / n²       (trigonal, square, octahedral centres)
/// ```
#[derive(Clone, Copy, Debug, Default)]
pub struct Uff;

impl ScalarForm for Uff {
    fn eval(&self, theta: &[F], p: &ParamColumns<'_>, e: &mut [F], de: &mut [F]) {
        let (ka, order) = (p.input("ka"), p.input("order"));
        let (c0, c1, c2) = (p.input("c0"), p.input("c1"), p.input("c2"));
        for t in 0..theta.len() {
            if order[t] == 0.0 {
                let (sin_t, cos_t) = theta[t].sin_cos();
                let cos2 = cos_t * cos_t - sin_t * sin_t;
                e[t] = ka[t] * (c0[t] + c1[t] * cos_t + c2[t] * cos2);
                de[t] = -ka[t] * (c1[t] * sin_t + 2.0 * c2[t] * (2.0 * sin_t * cos_t));
            } else if order[t] == 1.0 {
                // A linear centre. `(1 − cos nθ)/n²` rests at 0° for n = 1,
                // so the linear case is `1 + cos θ`, minimal at 180° — the
                // form the MMFF linear branch prices too.
                let (sin_t, cos_t) = theta[t].sin_cos();
                e[t] = ka[t] * (1.0 + cos_t);
                de[t] = -ka[t] * sin_t;
            } else {
                let n = order[t];
                let (sin_n, cos_n) = (n * theta[t]).sin_cos();
                e[t] = ka[t] * (1.0 - cos_n) / (n * n);
                // d(1 − cos nθ)/dθ = n sin nθ; over n², sin nθ / n.
                de[t] = ka[t] * sin_n / n;
            }
        }
    }

    fn inputs(&self) -> Vec<String> {
        ["ka", "order", "c0", "c1", "c2"].map(String::from).to_vec()
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

    fn bent(order: F, c: [F; 3]) -> Potentials {
        ExplicitTerms::new("angle", "uff_angle")
            .term(
                &[0, 1, 2],
                Params::from_iter([
                    ("ka", 100.0),
                    ("order", order),
                    ("c0", c[0]),
                    ("c1", c[1]),
                    ("c2", c[2]),
                ]),
            )
            .compile()
            .unwrap()
    }

    /// A right angle at the centre atom.
    fn right_angle() -> Vec<F> {
        vec![1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0]
    }

    #[test]
    fn fourier_order_zero_is_ka_times_the_cosine_series() {
        // c0 + c1·cosθ + c2·cos2θ at θ = 90° is c0 − c2.
        let pot = bent(0.0, [0.75, 0.3, 0.25]);
        let (e, _) = pot.energy_forces_at(&right_angle());
        assert!((e - 100.0 * (0.75 - 0.25)).abs() < 1e-9);
    }

    /// A linear centre (`order = 1`, what the UFF typifier emits for every
    /// sp atom) rests at 180°: the energy there is the minimum, and a right
    /// angle costs the full `ka`.
    #[test]
    fn a_linear_centre_rests_at_180_degrees() {
        let pot = bent(1.0, [0.0, 0.0, 0.0]);
        let straight = vec![1.0, 0.0, 0.0, 0.0, 0.0, 0.0, -1.0, 0.0, 0.0];
        let (e_straight, _) = pot.energy_forces_at(&straight);
        let (e_right, _) = pot.energy_forces_at(&right_angle());
        assert!(e_straight.abs() < 1e-9, "E(180°) = {e_straight}");
        assert!((e_right - 100.0).abs() < 1e-9, "E(90°) = {e_right}");
    }

    #[test]
    fn forces_are_the_negative_energy_gradient_for_every_order() {
        let coords = vec![1.1, 0.2, -0.1, 0.0, 0.0, 0.0, -0.3, 1.2, 0.4];
        for order in 0..=4 {
            let pot = bent(F::from(order), [0.75, 0.3, 0.25]);
            assert_forces_are_negative_gradient(&pot, &coords, 1e-5);
        }
    }
}
