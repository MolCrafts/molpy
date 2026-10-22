//! [`LinearFit`] — ordinary-least-squares line fit over a fractional window.

use molrs::op::F;

use super::FractionalWindow;
use crate::compute::{Compute, ComputeError, TimeCorrelation};

/// Result of a linear (OLS) fit of `C(t) = slope · t + intercept`.
#[derive(Debug, Clone)]
#[cfg_attr(feature = "serde", derive(serde::Serialize))]
pub struct LinearFitResult {
    /// Fitted slope dC/dt. Units: `[C]/[t]`.
    pub slope: F,
    /// Fitted intercept (the value of the line at `t = 0`). Units: `[C]`.
    pub intercept: F,
    /// Coefficient of determination r², dimensionless in `[0, 1]`; `1.0` for
    /// a perfect fit.
    pub r2: F,
    /// Inclusive first lag index of the fitted window.
    pub fit_start: usize,
    /// Inclusive last lag index of the fitted window.
    pub fit_end: usize,
}

/// Ordinary-least-squares line through one channel of a correlation curve,
/// against its lag times, over a fractional window.
///
/// The window `(start, end)` is given as fractions of the last lag index
/// `n − 1`:
///
/// ```text
///     fit_start = round((n − 1) · start)
///     fit_end   = round((n − 1) · end)
/// ```
///
/// A window covering fewer than two lags is an error; it is never widened.
/// Typical uses: the diffusion coefficient `D = slope / 2d` of an
/// [`Msd`](crate::compute::Msd) curve, the Einstein–Helfand conductivity slope
/// of the collective-dipole MSD, the Onsager coefficients of an
/// [`OnsagerCorrelation`](crate::compute::OnsagerCorrelation).
///
/// ```
/// use molrs::compute::{Compute, LinearFit, LinearFitInput, TimeCorrelation};
/// use ndarray::{Array1, Array2};
///
/// // C(t) = 3t + 2 on t = 0 … 9.
/// let lag_times = Array1::from_iter((0..10).map(|k| k as f64));
/// let values = lag_times.mapv(|t| 3.0 * t + 2.0).insert_axis(ndarray::Axis(1));
/// let curve = TimeCorrelation { lag_times, values };
/// let fit = LinearFit::new(0.0, 1.0)?.compute(LinearFitInput { correlation: &curve, channel: 0 })?;
/// assert!((fit.slope - 3.0).abs() < 1e-12);
/// # Ok::<(), molrs::compute::ComputeError>(())
/// ```
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct LinearFit {
    window: FractionalWindow,
}

/// The curve a [`LinearFit`] fits.
#[derive(Debug, Clone, Copy)]
pub struct LinearFitInput<'a> {
    /// The correlation curve; its lag times are the abscissa.
    pub correlation: &'a TimeCorrelation,
    /// Which of its channels to fit.
    pub channel: usize,
}

impl LinearFit {
    /// A fit over the window `(start, end)`, fractions of the last lag.
    ///
    /// # Errors
    /// [`ComputeError::OutOfRange`] unless `0 ≤ start < end ≤ 1`.
    pub fn new(start: F, end: F) -> Result<Self, ComputeError> {
        Ok(Self {
            window: FractionalWindow::new(start, end)?,
        })
    }

    /// The window, `(start, end)` fractions of the last lag.
    pub fn window(&self) -> (F, F) {
        self.window.fractions()
    }

    /// Ordinary least squares of `y = slope · x + intercept` over the
    /// inclusive index range `[start, end]`: `(slope, intercept, r²)`, or
    /// `None` when every `x` in the range is equal (no line is defined).
    pub(crate) fn ordinary_least_squares(
        x: &[F],
        y: &[F],
        start: usize,
        end: usize,
    ) -> Option<(F, F, F)> {
        let np = (end - start + 1) as F;
        let (mut sx, mut sy, mut sxx, mut sxy) = (0.0, 0.0, 0.0, 0.0);
        for i in start..=end {
            let xi = x[i];
            let yi = y[i];
            sx += xi;
            sy += yi;
            sxx += xi * xi;
            sxy += xi * yi;
        }
        let denom = np * sxx - sx * sx;
        if denom.abs() < F::EPSILON {
            return None;
        }
        let slope = (np * sxy - sx * sy) / denom;
        let intercept = (sy - slope * sx) / np;

        // r² = 1 − SS_res / SS_tot.
        let y_mean = sy / np;
        let mut ss_res = 0.0;
        let mut ss_tot = 0.0;
        for i in start..=end {
            let resid = y[i] - (slope * x[i] + intercept);
            ss_res += resid * resid;
            let dev = y[i] - y_mean;
            ss_tot += dev * dev;
        }
        // Perfectly flat y (ss_tot == 0): the line is exact iff residuals vanish.
        let r2 = if ss_tot.abs() < F::EPSILON {
            if ss_res.abs() < F::EPSILON { 1.0 } else { 0.0 }
        } else {
            1.0 - ss_res / ss_tot
        };
        Some((slope, intercept, r2))
    }
}

impl Compute for LinearFit {
    type Input<'a> = LinearFitInput<'a>;
    type Output = LinearFitResult;

    /// # Errors
    /// * [`ComputeError::OutOfRange`] when the window covers fewer than two
    ///   lags, the channel does not exist, or every lag time in the window is
    ///   equal.
    /// * [`ComputeError::NonFinite`] for a non-finite value in the channel.
    fn compute(&self, input: LinearFitInput<'_>) -> Result<LinearFitResult, ComputeError> {
        let y = input.correlation.require_channel(input.channel)?;
        let (fit_start, fit_end) = self.window.indices(y.len())?;
        let xs = input.correlation.lag_times.to_vec();
        let ys = y.to_vec();
        let (slope, intercept, r2) = Self::ordinary_least_squares(&xs, &ys, fit_start, fit_end)
            .ok_or(ComputeError::OutOfRange {
                field: "fit window (degenerate: all lag times equal)",
                value: format!("[{fit_start}, {fit_end}]"),
            })?;
        Ok(LinearFitResult {
            slope,
            intercept,
            r2,
            fit_start,
            fit_end,
        })
    }
}

crate::compute::spec::compute_spec! {
    LinearFit as "LinearFit";
    params |s| {
        start: Float;
        end: Float;
    }
    inputs {
        correlation: Upstream("TimeCorrelation");
        channel: Index;
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use ndarray::{Array1, Array2};

    fn line(x: &[F], y: &[F]) -> TimeCorrelation {
        TimeCorrelation {
            lag_times: Array1::from_vec(x.to_vec()),
            values: Array2::from_shape_vec((y.len(), 1), y.to_vec()).unwrap(),
        }
    }

    fn fit(window: (F, F), x: &[F], y: &[F]) -> Result<LinearFitResult, ComputeError> {
        let c = line(x, y);
        LinearFit::new(window.0, window.1)?.compute(LinearFitInput {
            correlation: &c,
            channel: 0,
        })
    }

    fn linspace(n: usize) -> Vec<F> {
        (0..n).map(|i| i as F).collect()
    }

    #[test]
    fn ols_recovers_exact_line() {
        let x: Vec<f64> = (0..10).map(|i| i as f64).collect();
        let y: Vec<f64> = x.iter().map(|&xi| 3.0 * xi + 2.0).collect();
        let (slope, intercept, r2) = LinearFit::ordinary_least_squares(&x, &y, 0, 9).unwrap();
        assert!((slope - 3.0).abs() < 1e-12);
        assert!((intercept - 2.0).abs() < 1e-12);
        assert!((r2 - 1.0).abs() < 1e-12);
    }

    #[test]
    fn ols_degenerate_returns_none() {
        let x = vec![5.0, 5.0, 5.0];
        let y = vec![1.0, 2.0, 3.0];
        assert!(LinearFit::ordinary_least_squares(&x, &y, 0, 2).is_none());
    }

    #[test]
    fn recovers_known_slope_intercept_r2() {
        // ac-002: y = 3x + 2 on a uniform grid, window (0,1).
        let x = linspace(10);
        let y: Vec<F> = x.iter().map(|&xi| 3.0 * xi + 2.0).collect();
        let res = fit((0.0, 1.0), &x, &y).unwrap();
        assert!((res.slope - 3.0).abs() < 1e-12, "slope {}", res.slope);
        assert!((res.intercept - 2.0).abs() < 1e-12);
        assert!((res.r2 - 1.0).abs() < 1e-12);
        assert_eq!(res.fit_start, 0);
        assert_eq!(res.fit_end, 9);
    }

    #[test]
    fn full_window_uses_whole_curve() {
        let x = linspace(21);
        let y: Vec<F> = x.iter().map(|&xi| -0.5 * xi + 7.0).collect();
        let res = fit((0.0, 1.0), &x, &y).unwrap();
        assert_eq!(res.fit_start, 0);
        assert_eq!(res.fit_end, 20);
        assert!((res.slope + 0.5).abs() < 1e-12);
    }

    #[test]
    fn degenerate_window_start_ge_end_errors() {
        // ac-016: start >= end.
        let x = linspace(10);
        let err = fit((0.7, 0.3), &x, &x).unwrap_err();
        assert!(matches!(err, ComputeError::OutOfRange { .. }));
    }

    #[test]
    fn degenerate_all_x_equal_errors() {
        // ac-016: all x in the window equal -> denom ~ 0.
        let x = vec![5.0; 6];
        let y = vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0];
        let err = fit((0.0, 1.0), &x, &y).unwrap_err();
        assert!(matches!(err, ComputeError::OutOfRange { .. }));
    }

    #[test]
    fn mismatched_lengths_error() {
        let err = fit((0.0, 1.0), &linspace(5), &linspace(4)).unwrap_err();
        assert!(matches!(err, ComputeError::DimensionMismatch { .. }));
    }

    /// Spec §5: the old fit widened a one-lag window to two lags; it is now
    /// refused (`too_short_errors` used to expect `EmptyInput`).
    #[test]
    fn too_short_errors() {
        let err = fit((0.0, 1.0), &linspace(1), &linspace(1)).unwrap_err();
        assert!(matches!(err, ComputeError::OutOfRange { .. }));
        // n = 11, (0.42, 0.46): [round(4.2), round(4.6)] = [4, 5] — two lags, fine.
        assert!(fit((0.42, 0.46), &linspace(11), &linspace(11)).is_ok());
        // n = 11, (0.41, 0.44): [4, 4] — one lag: refused, not widened.
        assert!(matches!(
            fit((0.41, 0.44), &linspace(11), &linspace(11)),
            Err(ComputeError::OutOfRange { .. })
        ));
    }
}
