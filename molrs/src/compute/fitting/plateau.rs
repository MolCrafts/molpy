//! [`Plateau`] — windowed mean (and spread) of a curve.

use molrs::op::F;

use super::FractionalWindow;
use crate::compute::{Compute, ComputeError, TimeCorrelation};

/// Result of a plateau (windowed-mean) read.
#[derive(Debug, Clone)]
pub struct PlateauResult {
    /// Mean of the curve over the window. Units: `[C]`.
    pub value: F,
    /// Number of samples inside the window (`fit_end − fit_start + 1`).
    pub n_samples: usize,
    /// Population standard deviation of the curve over the window. Units: `[C]`.
    pub std: F,
}

/// Windowed mean of one channel of a curve over a fractional window.
///
/// The window `(start, end)` maps to the inclusive index range
/// `[round(start · (n − 1)), round(end · (n − 1))]`; one covering fewer than
/// two samples is an error. Typical uses: the converged tail of a Green–Kubo
/// running integral ([`CumulativeTrapezoid`](crate::compute::CumulativeTrapezoid)),
/// and the non-relaxing plateau `C(∞)` of a dipole ACF that
/// [`DipoleAutocorrelationSpectrum`](crate::compute::DipoleAutocorrelationSpectrum)
/// subtracts.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Plateau {
    window: FractionalWindow,
}

/// The curve a [`Plateau`] reads.
#[derive(Debug, Clone, Copy)]
pub struct PlateauInput<'a> {
    /// The curve.
    pub correlation: &'a TimeCorrelation,
    /// Which of its channels to average.
    pub channel: usize,
}

impl Plateau {
    /// A plateau over the window `(start, end)`, fractions of the last lag.
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

    /// The windowed mean and spread of `y`.
    pub(crate) fn read(&self, y: &[F]) -> Result<PlateauResult, ComputeError> {
        let (fit_start, fit_end) = self.window.indices(y.len())?;
        let n_samples = fit_end - fit_start + 1;
        let window = &y[fit_start..=fit_end];
        let value = window.iter().sum::<F>() / n_samples as F;
        let var = window.iter().map(|v| (v - value) * (v - value)).sum::<F>() / n_samples as F;
        Ok(PlateauResult {
            value,
            n_samples,
            std: var.sqrt(),
        })
    }
}

impl Compute for Plateau {
    type Input<'a> = PlateauInput<'a>;
    type Output = PlateauResult;

    /// # Errors
    /// [`ComputeError::OutOfRange`] when the window covers fewer than two
    /// samples or the channel does not exist; [`ComputeError::NonFinite`] for
    /// a non-finite value in the channel.
    fn compute(&self, input: PlateauInput<'_>) -> Result<PlateauResult, ComputeError> {
        let y = input.correlation.require_channel(input.channel)?;
        self.read(&y.to_vec())
    }
}

crate::compute::spec::compute_spec! {
    Plateau as "Plateau";
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
    use super::super::curve;
    use super::*;

    fn plateau(window: (F, F), y: &[F]) -> Result<PlateauResult, ComputeError> {
        let c = curve(y, 1.0);
        Plateau::new(window.0, window.1)?.compute(PlateauInput {
            correlation: &c,
            channel: 0,
        })
    }

    #[test]
    fn windowed_mean_of_plateau() {
        // ac-005: curve == P over (a, b). n=11, a=0.5, b=1.0 ->
        // [round(5), round(10)] = [5, 10], 6 samples.
        let p = 4.2;
        let mut y = vec![0.0; 11];
        for v in y.iter_mut().skip(5) {
            *v = p;
        }
        let res = plateau((0.5, 1.0), &y).unwrap();
        assert!((res.value - p).abs() < 1e-12);
        assert_eq!(res.n_samples, 6);
        assert!(res.std < 1e-12);
    }

    #[test]
    fn sample_count_matches_rounded_bounds() {
        // n=21, a=0.25, b=0.75 -> [round(5), round(15)] = [5, 15] => 11 samples.
        let y: Vec<F> = (0..21).map(|i| i as f64).collect();
        let res = plateau((0.25, 0.75), &y).unwrap();
        assert_eq!(res.n_samples, 11);
        // Mean of 5..=15 = 10.
        assert!((res.value - 10.0).abs() < 1e-12);
    }

    #[test]
    fn nonzero_std_for_varying_window() {
        let y = [0.0, 0.0, 1.0, 3.0, 5.0];
        let res = plateau((0.5, 1.0), &y).unwrap();
        // window [round(2), round(4)] = [2, 4] => {1,3,5}, mean 3, std sqrt(8/3).
        assert!((res.value - 3.0).abs() < 1e-12);
        assert!((res.std - (8.0_f64 / 3.0).sqrt()).abs() < 1e-12);
    }

    #[test]
    fn degenerate_window_errors() {
        assert!(matches!(
            Plateau::new(0.8, 0.2),
            Err(ComputeError::OutOfRange { .. })
        ));
    }

    /// Spec §5: the old reader collapsed a short curve to its lone point and
    /// returned it; a window of fewer than two samples is now an error.
    #[test]
    fn collapsing_window_errors() {
        assert!(matches!(
            plateau((0.0, 1.0), &[3.0]),
            Err(ComputeError::OutOfRange { .. })
        ));
        // n = 5, (0.9, 1.0): [round(3.6), round(4.0)] = [4, 4].
        assert!(matches!(
            plateau((0.9, 1.0), &[1.0, 2.0, 3.0, 4.0, 5.0]),
            Err(ComputeError::OutOfRange { .. })
        ));
    }

    #[test]
    fn empty_curve_errors() {
        assert!(matches!(
            plateau((0.0, 1.0), &[]),
            Err(ComputeError::OutOfRange { .. })
        ));
    }
}
