//! Fits: the computes that turn a [`TimeCorrelation`] into a coefficient.
//!
//! A raw correlation curve comes from [`Correlator`](crate::compute::Correlator)
//! or one of its surfaces; a fit is a [`Compute`](crate::compute::Compute)
//! whose input is that curve — one channel of it, named by index — and whose
//! configuration is the fit window. Every fit reads the lag grid off the
//! curve, so no sample interval travels separately.
//!
//! | Fit | Reads | Output |
//! |-----|-------|--------|
//! | [`LinearFit`] | `C(t)` over a window | [`LinearFitResult`] (slope / intercept / r²) |
//! | [`CumulativeTrapezoid`] | `C(t)` | the running integral `∫₀ᵗ C`, itself a [`TimeCorrelation`] |
//! | [`Plateau`] | `C(t)` over a window | [`PlateauResult`] (windowed mean / spread) |
//! | [`DebyeFit`] | the leading positive run of `C(t)` | [`DebyeFitResult`] (τ, amplitude) |
//!
//! A window is given as fractions `(start, end)` of the last lag. A window
//! that covers fewer than two samples is an error — a fit never widens or
//! collapses its window to make the data fit.

mod cumulative_trapezoid;
mod debye_fit;
mod linear_fit;
mod plateau;

pub use cumulative_trapezoid::{CumulativeTrapezoid, CumulativeTrapezoidInput};
pub use debye_fit::{DebyeFit, DebyeFitInput, DebyeFitResult};
pub use linear_fit::{LinearFit, LinearFitInput, LinearFitResult};
pub use plateau::{Plateau, PlateauInput, PlateauResult};

use molrs::op::F;

use crate::compute::ComputeError;

#[cfg(test)]
use crate::compute::TimeCorrelation;

/// A fit window `(start, end)` as fractions of a curve's last index, with
/// `0 ≤ start < end ≤ 1`. It maps to the inclusive index range
/// `[round(start · (n − 1)), round(end · (n − 1))]`.
#[derive(Debug, Clone, Copy, PartialEq)]
pub(crate) struct FractionalWindow {
    start: F,
    end: F,
}

impl FractionalWindow {
    /// # Errors
    /// [`ComputeError::OutOfRange`] unless `0 ≤ start < end ≤ 1`.
    pub(crate) fn new(start: F, end: F) -> Result<Self, ComputeError> {
        if !(0.0..=1.0).contains(&start) || !(0.0..=1.0).contains(&end) || start >= end {
            return Err(ComputeError::OutOfRange {
                field: "window (require 0 <= start < end <= 1)",
                value: format!("{start}/{end}"),
            });
        }
        Ok(Self { start, end })
    }

    /// `(start, end)` fractions.
    pub(crate) fn fractions(&self) -> (F, F) {
        (self.start, self.end)
    }

    /// The inclusive index range on a curve of `n` samples.
    ///
    /// # Errors
    /// [`ComputeError::OutOfRange`] when the range holds fewer than two
    /// samples (a curve too short for the window).
    pub(crate) fn indices(&self, n: usize) -> Result<(usize, usize), ComputeError> {
        let last = n.saturating_sub(1) as F;
        let first = (last * self.start).round() as usize;
        let final_ = (last * self.end).round() as usize;
        if n < 2 || final_ <= first {
            return Err(ComputeError::OutOfRange {
                field: "window (covers fewer than two samples of the curve)",
                value: format!(
                    "{}/{} of {n} samples -> [{first}, {final_}]",
                    self.start, self.end
                ),
            });
        }
        Ok((first, final_))
    }
}

/// A single-channel correlation `y(t_k)` on the lag grid `t_k = k · dt`.
#[cfg(test)]
pub(crate) fn curve(y: &[F], dt: F) -> TimeCorrelation {
    TimeCorrelation {
        lag_times: ndarray::Array1::from_iter((0..y.len()).map(|k| k as F * dt)),
        values: ndarray::Array2::from_shape_vec((y.len(), 1), y.to_vec()).unwrap(),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn windows_validate_and_map_to_indices() {
        let w = FractionalWindow::new(0.25, 0.75).unwrap();
        assert_eq!(w.indices(21).unwrap(), (5, 15));
        assert_eq!(w.fractions(), (0.25, 0.75));
        for (a, b) in [(0.7, 0.3), (0.5, 0.5), (-0.1, 0.5), (0.0, 1.5)] {
            assert!(matches!(
                FractionalWindow::new(a, b),
                Err(ComputeError::OutOfRange { .. })
            ));
        }
    }

    /// A window that would cover one sample is refused, never widened.
    #[test]
    fn a_collapsing_window_is_an_error() {
        let w = FractionalWindow::new(0.4, 0.5).unwrap();
        // n = 5: [round(1.6), round(2.0)] = [2, 2] — one sample.
        assert!(matches!(w.indices(5), Err(ComputeError::OutOfRange { .. })));
        assert!(matches!(
            FractionalWindow::new(0.0, 1.0).unwrap().indices(1),
            Err(ComputeError::OutOfRange { .. })
        ));
    }
}
