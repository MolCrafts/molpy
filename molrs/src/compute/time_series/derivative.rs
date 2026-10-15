//! [`Derivative`] — the one finite-difference time derivative.

use molrs::op::F;
use ndarray::{Array2, ArrayView2};

use crate::compute::named_enum::named_enum;
use crate::compute::{Compute, ComputeError};

named_enum! {
    /// The finite-difference stencil of a [`Derivative`].
    pub enum Stencil {
        /// Second-order central difference `(x[t+1] − x[t−1]) / (2 dt)` at the
        /// interior frames only: `T − 2` rows, the first and last frame have
        /// no centred neighbour and are dropped.
        Central => "central",
        /// NumPy's `gradient(…, edge_order=2)`: central differences inside,
        /// second-order one-sided stencils at both ends, `T` rows (first-order
        /// at both ends when `T = 2`).
        Gradient => "gradient",
        /// First-order backward difference `(x[t] − x[t−1]) / dt` for
        /// `t ≥ 1`: `T − 1` rows, the first frame has no predecessor and is
        /// dropped.
        Backward => "backward",
    }
}

/// Time derivative of a sampled series, column by column.
///
/// ```
/// use molrs::compute::{Compute, Derivative, DerivativeInput, Stencil};
/// use ndarray::array;
///
/// // x(t) = t² sampled at t = 0, 1, 2, 3.
/// let x = array![[0.0], [1.0], [4.0], [9.0]];
/// let dx = Derivative::new(Stencil::Central)
///     .compute(DerivativeInput { series: x.view(), dt: 1.0 })
///     .unwrap();
/// assert_eq!(dx, array![[2.0], [4.0]]); // 2t at t = 1, 2
/// ```
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct Derivative {
    stencil: Stencil,
}

/// The data a [`Derivative`] differentiates.
#[derive(Debug, Clone, Copy)]
pub struct DerivativeInput<'a> {
    /// The series, `T × C`: one row per frame, one column per channel.
    pub series: ArrayView2<'a, F>,
    /// Sample interval.
    pub dt: F,
}

impl Derivative {
    /// A derivative with `stencil`.
    pub fn new(stencil: Stencil) -> Self {
        Self { stencil }
    }

    /// The stencil.
    pub fn stencil(&self) -> Stencil {
        self.stencil
    }

    /// Frames the stencil needs.
    fn min_frames(&self) -> usize {
        match self.stencil {
            Stencil::Central => 3,
            Stencil::Gradient | Stencil::Backward => 2,
        }
    }
}

impl Compute for Derivative {
    type Input<'a> = DerivativeInput<'a>;
    type Output = Array2<F>;

    fn compute(&self, input: DerivativeInput<'_>) -> Result<Array2<F>, ComputeError> {
        let x = input.series;
        let (n, n_cols) = x.dim();
        if n_cols == 0 || n < self.min_frames() {
            return Err(ComputeError::EmptyInput);
        }
        if !(input.dt.is_finite() && input.dt > 0.0) {
            return Err(ComputeError::OutOfRange {
                field: "dt",
                value: input.dt.to_string(),
            });
        }
        if let Some(index) = x.iter().position(|v| !v.is_finite()) {
            return Err(ComputeError::NonFinite {
                where_: "series",
                index,
            });
        }
        let inv_h = 1.0 / input.dt;
        let inv_2h = 0.5 * inv_h;
        let central = |t: usize, c: usize| (x[[t + 1, c]] - x[[t - 1, c]]) * inv_2h;
        Ok(match self.stencil {
            Stencil::Central => Array2::from_shape_fn((n - 2, n_cols), |(t, c)| central(t + 1, c)),
            Stencil::Backward => Array2::from_shape_fn((n - 1, n_cols), |(t, c)| {
                (x[[t + 1, c]] - x[[t, c]]) * inv_h
            }),
            Stencil::Gradient if n == 2 => {
                Array2::from_shape_fn((2, n_cols), |(_, c)| (x[[1, c]] - x[[0, c]]) * inv_h)
            }
            Stencil::Gradient => Array2::from_shape_fn((n, n_cols), |(t, c)| {
                if t == 0 {
                    (-1.5 * x[[0, c]] + 2.0 * x[[1, c]] - 0.5 * x[[2, c]]) * inv_h
                } else if t == n - 1 {
                    (0.5 * x[[n - 3, c]] - 2.0 * x[[n - 2, c]] + 1.5 * x[[n - 1, c]]) * inv_h
                } else {
                    central(t, c)
                }
            }),
        })
    }
}

crate::compute::spec::compute_spec! {
    Derivative as "Derivative";
    params |s| {
        stencil: Choice(Stencil::NAMES);
    }
    inputs {
        series: Series;
        dt: Scalar;
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use ndarray::{Array2, array};

    fn derivative(stencil: Stencil, series: &Array2<F>, dt: F) -> Result<Array2<F>, ComputeError> {
        Derivative::new(stencil).compute(DerivativeInput {
            series: series.view(),
            dt,
        })
    }

    /// `x(t) = 3t² − 2t + 1` and `y(t) = −t² + 5` at `t = k·h`: a second-order
    /// stencil is exact on a quadratic.
    fn quadratic(n: usize, h: F) -> (Array2<F>, Array2<F>) {
        let x = Array2::from_shape_fn((n, 2), |(k, c)| {
            let t = k as F * h;
            if c == 0 {
                3.0 * t * t - 2.0 * t + 1.0
            } else {
                -t * t + 5.0
            }
        });
        let dx = Array2::from_shape_fn((n, 2), |(k, c)| {
            let t = k as F * h;
            if c == 0 { 6.0 * t - 2.0 } else { -2.0 * t }
        });
        (x, dx)
    }

    #[test]
    fn central_is_exact_on_a_quadratic_at_the_interior() {
        let (x, dx) = quadratic(7, 0.25);
        let got = derivative(Stencil::Central, &x, 0.25).unwrap();
        assert_eq!(got.dim(), (5, 2));
        for ((t, c), v) in got.indexed_iter() {
            assert!((v - dx[[t + 1, c]]).abs() < 1e-12, "({t}, {c})");
        }
    }

    #[test]
    fn gradient_is_exact_on_a_quadratic_including_both_ends() {
        let (x, dx) = quadratic(6, 0.5);
        let got = derivative(Stencil::Gradient, &x, 0.5).unwrap();
        assert_eq!(got.dim(), (6, 2));
        for ((t, c), v) in got.indexed_iter() {
            assert!((v - dx[[t, c]]).abs() < 1e-12, "({t}, {c})");
        }
    }

    #[test]
    fn backward_is_exact_on_a_line() {
        let x = Array2::from_shape_fn((5, 1), |(k, _)| 4.0 * k as F * 0.1 - 1.0);
        let got = derivative(Stencil::Backward, &x, 0.1).unwrap();
        assert_eq!(got.dim(), (4, 1));
        assert!(got.iter().all(|v| (v - 4.0).abs() < 1e-12));
    }

    #[test]
    fn gradient_matches_numpy_edge_order2_values() {
        let s = Array2::from_shape_fn((5, 1), |(t, _)| (t as F).powi(2));
        let g = derivative(Stencil::Gradient, &s, 1.0).unwrap();
        assert!((g[[1, 0]] - 2.0).abs() < 1e-12);
        assert!((g[[2, 0]] - 4.0).abs() < 1e-12);
        assert!((g[[3, 0]] - 6.0).abs() < 1e-12);
        assert!((g[[0, 0]] - 0.0).abs() < 1e-12);
        assert!((g[[4, 0]] - 8.0).abs() < 1e-12);
    }

    /// NumPy's `edge_order=2` falls back to first order for two samples.
    #[test]
    fn gradient_of_two_samples_is_first_order() {
        let g = derivative(Stencil::Gradient, &array![[1.0], [4.0]], 0.5).unwrap();
        assert_eq!(g, array![[6.0], [6.0]]);
    }

    #[test]
    fn too_short_bad_dt_and_non_finite_are_refused() {
        let two = array![[0.0], [1.0]];
        assert!(matches!(
            derivative(Stencil::Central, &two, 1.0),
            Err(ComputeError::EmptyInput)
        ));
        let one = array![[0.0]];
        for stencil in Stencil::ALL.iter().copied() {
            assert!(matches!(
                derivative(stencil, &one, 1.0),
                Err(ComputeError::EmptyInput)
            ));
        }
        let three = array![[0.0], [1.0], [2.0]];
        assert!(matches!(
            derivative(Stencil::Gradient, &three, 0.0),
            Err(ComputeError::OutOfRange { field: "dt", .. })
        ));
        let nan = array![[0.0], [F::NAN], [2.0]];
        assert!(matches!(
            derivative(Stencil::Central, &nan, 1.0),
            Err(ComputeError::NonFinite { index: 1, .. })
        ));
    }
}
