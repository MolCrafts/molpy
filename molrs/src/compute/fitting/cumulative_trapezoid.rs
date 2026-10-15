//! [`CumulativeTrapezoid`] — the running trapezoidal integral of a curve.

use molrs::op::F;
use ndarray::Array2;

use crate::compute::{Compute, ComputeError, TimeCorrelation};

/// Running trapezoidal integral of one channel of a correlation curve:
///
/// ```text
/// I(t_k) = Σ_{j=1..k} ½ (C(t_{j−1}) + C(t_j)) · dt,     I(0) = 0
/// ```
///
/// The output is itself a curve on the same lag grid, so the Green–Kubo
/// plateau of a [`Vacf`](crate::compute::Vacf) or
/// [`CurrentAcf`](crate::compute::CurrentAcf) is read off it with
/// [`Plateau`](crate::compute::Plateau) without any glue. It is the one
/// trapezoid rule of `compute`: the dielectric spectra, the conductivity sum
/// rule and the hydrogen-bond lifetimes integrate through it.
///
/// ```
/// use molrs::compute::{Compute, CumulativeTrapezoid, CumulativeTrapezoidInput, Vacf, VacfInput};
/// use ndarray::Array2;
///
/// let v = Array2::from_shape_fn((64, 3), |(t, d)| ((t * 3 + d) as f64 * 0.7).sin());
/// let vacf = Vacf::new(16).compute(VacfInput { velocities: v.view(), dt: 1.0 })?;
/// let running = CumulativeTrapezoid::new()
///     .compute(CumulativeTrapezoidInput { correlation: &vacf, channel: 0 })?;
/// assert_eq!(running.n_lags(), 17);
/// assert_eq!(running.values[[0, 0]], 0.0);
/// # Ok::<(), molrs::compute::ComputeError>(())
/// ```
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub struct CumulativeTrapezoid {
    n_lags: Option<usize>,
}

/// The curve a [`CumulativeTrapezoid`] integrates.
#[derive(Debug, Clone, Copy)]
pub struct CumulativeTrapezoidInput<'a> {
    /// The correlation curve.
    pub correlation: &'a TimeCorrelation,
    /// Which of its channels to integrate.
    pub channel: usize,
}

impl CumulativeTrapezoid {
    /// Integrate the whole curve.
    pub fn new() -> Self {
        Self::default()
    }

    /// Integrate only the first `n_lags` samples. A curve shorter than that
    /// is an error, never silently truncated.
    pub fn with_n_lags(mut self, n_lags: usize) -> Self {
        self.n_lags = Some(n_lags);
        self
    }

    /// The number of leading samples integrated; `None` = all.
    pub fn n_lags(&self) -> Option<usize> {
        self.n_lags
    }

    /// The running trapezoid of `y` on intervals of width `step(k)` between
    /// samples `k − 1` and `k`; element `0` is `0`. The one trapezoid
    /// recurrence of `compute`.
    pub(crate) fn running(y: &[F], step: impl Fn(usize) -> F) -> Vec<F> {
        let mut out = vec![0.0; y.len()];
        let mut integral = 0.0;
        for k in 1..y.len() {
            integral += 0.5 * (y[k - 1] + y[k]) * step(k);
            out[k] = integral;
        }
        out
    }

    /// `∫ y` over the whole series on a uniform step (`0` for fewer than two
    /// samples).
    pub(crate) fn total(y: &[F], dt: F) -> F {
        Self::running(y, |_| dt).last().copied().unwrap_or(0.0)
    }
}

impl Compute for CumulativeTrapezoid {
    type Input<'a> = CumulativeTrapezoidInput<'a>;
    type Output = TimeCorrelation;

    /// # Errors
    /// * [`ComputeError::EmptyInput`] for an empty curve or `n_lags = 0`.
    /// * [`ComputeError::OutOfRange`] when `n_lags` exceeds the curve, the
    ///   channel does not exist, or the lag spacing is not positive.
    fn compute(
        &self,
        input: CumulativeTrapezoidInput<'_>,
    ) -> Result<TimeCorrelation, ComputeError> {
        let y = input.correlation.require_channel(input.channel)?;
        if y.is_empty() {
            return Err(ComputeError::EmptyInput);
        }
        let take = match self.n_lags {
            None => y.len(),
            Some(0) => return Err(ComputeError::EmptyInput),
            Some(m) if m > y.len() => {
                return Err(ComputeError::OutOfRange {
                    field: "n_lags (exceeds curve length)",
                    value: format!("{m} > {}", y.len()),
                });
            }
            Some(m) => m,
        };
        let lag_times = input
            .correlation
            .lag_times
            .slice(ndarray::s![..take])
            .to_owned();
        let integral = if take == 1 {
            vec![0.0]
        } else {
            let dt = input.correlation.sample_interval()?;
            let ys: Vec<F> = y.iter().take(take).copied().collect();
            Self::running(&ys, |_| dt)
        };
        Ok(TimeCorrelation {
            lag_times,
            values: Array2::from_shape_vec((take, 1), integral)
                .expect("one value per integrated lag"),
        })
    }
}

crate::compute::spec::compute_spec! {
    CumulativeTrapezoid as "CumulativeTrapezoid";
    params |s| {
        n_lags: Int = CumulativeTrapezoid::new().n_lags => s.n_lags;
    }
    inputs {
        correlation: Upstream("TimeCorrelation");
        channel: Index;
    }
}

#[cfg(test)]
mod tests {
    use super::super::curve;
    use super::*;
    use ndarray::Array1;

    fn integral(c: &TimeCorrelation) -> Array1<F> {
        c.channel(0).unwrap().to_owned()
    }

    fn integrate(y: &[F], dt: F, n_lags: Option<usize>) -> Result<TimeCorrelation, ComputeError> {
        let c = curve(y, dt);
        let fit = match n_lags {
            Some(m) => CumulativeTrapezoid::new().with_n_lags(m),
            None => CumulativeTrapezoid::new(),
        };
        fit.compute(CumulativeTrapezoidInput {
            correlation: &c,
            channel: 0,
        })
    }

    #[test]
    fn constant_curve_linear_integral() {
        // ac-004: constant c on step dt -> integral[k] = c·k·dt.
        let c = 3.0;
        let dt = 0.25;
        let res = integral(&integrate(&[c; 8], dt, None).unwrap());
        for k in 0..8 {
            let expected = c * k as f64 * dt;
            assert!(
                (res[k] - expected).abs() < 1e-12,
                "k={k}: {} != {expected}",
                res[k]
            );
        }
    }

    #[test]
    fn triangle_signal_analytic_integral() {
        // y = t on [0, 4] step 1 -> ∫₀^T t dt = T²/2 by trapezoid (exact for
        // linear integrand).
        let res = integral(&integrate(&[0.0, 1.0, 2.0, 3.0, 4.0], 1.0, None).unwrap());
        for k in 0..5 {
            let expected = (k as f64) * (k as f64) / 2.0;
            assert!((res[k] - expected).abs() < 1e-12);
        }
    }

    #[test]
    fn matches_jacf_running_trapezoid() {
        // ac-004: equals the running trapezoid integral computed inline in
        // the Green–Kubo conductivity running integral on the same JACF + dt.
        let jacf = [1.0, 0.8, 0.5, 0.2, 0.05];
        let dt = 0.5;
        let res = integral(&integrate(&jacf, dt, None).unwrap());
        // Inline jacf.rs recurrence.
        let mut expected = vec![0.0; jacf.len()];
        let mut acc = 0.0;
        for tau in 1..jacf.len() {
            acc += 0.5 * (jacf[tau - 1] + jacf[tau]) * dt;
            expected[tau] = acc;
        }
        for (k, (&got, &want)) in res.iter().zip(expected.iter()).enumerate() {
            assert!((got - want).abs() < 1e-12, "k={k}");
        }
    }

    #[test]
    fn output_is_a_curve_on_the_input_lag_grid() {
        let res = integrate(&[1.0, 1.0, 1.0], 0.5, None).unwrap();
        assert_eq!(res.lag_times.to_vec(), vec![0.0, 0.5, 1.0]);
        assert_eq!(res.n_channels(), 1);
        assert_eq!(CumulativeTrapezoid::total(&[1.0, 1.0, 1.0], 0.5), 1.0);
        assert_eq!(CumulativeTrapezoid::total(&[1.0], 0.5), 0.0);
    }

    #[test]
    fn n_lags_subset_integrates_only_prefix() {
        let res = integrate(&[1.0; 10], 1.0, Some(4)).unwrap();
        assert_eq!(res.n_lags(), 4);
    }

    #[test]
    fn n_lags_exceeding_length_errors() {
        // ac-014: requested lag range exceeds the raw curve -> OutOfRange.
        assert!(matches!(
            integrate(&[1.0, 2.0, 3.0], 1.0, Some(10)),
            Err(ComputeError::OutOfRange { .. })
        ));
    }

    #[test]
    fn empty_curve_errors() {
        assert!(matches!(
            integrate(&[], 1.0, None),
            Err(ComputeError::EmptyInput)
        ));
    }

    #[test]
    fn zero_lags_errors() {
        assert!(matches!(
            integrate(&[1.0, 2.0], 1.0, Some(0)),
            Err(ComputeError::EmptyInput)
        ));
    }

    #[test]
    fn nonpositive_dt_errors() {
        assert!(matches!(
            integrate(&[1.0, 2.0], 0.0, None),
            Err(ComputeError::OutOfRange { .. })
        ));
    }

    #[test]
    fn a_missing_channel_errors() {
        let c = curve(&[1.0, 2.0], 1.0);
        assert!(matches!(
            CumulativeTrapezoid::new().compute(CumulativeTrapezoidInput {
                correlation: &c,
                channel: 1
            }),
            Err(ComputeError::OutOfRange {
                field: "channel",
                ..
            })
        ));
    }
}
