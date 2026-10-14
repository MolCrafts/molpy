//! [`TimeCorrelation`] — the one output of every time correlation.

use molrs::op::F;
use ndarray::{Array1, Array2, ArrayView1, Axis};

use crate::compute::ComputeError;

/// A time-correlation curve: one row per lag, one column per channel.
///
/// The output of [`Correlator`](super::Correlator) and of every surface over
/// it ([`Msd`](super::Msd), [`Vacf`](super::Vacf), …), and the input of the
/// fits and spectra that turn a curve into a coefficient. Which channels exist
/// is set by the correlator's [`Reduction`](super::Reduction): one for
/// `Mean`, one per item for `PerItem`, one per component for `PerComponent`.
///
/// The lag grid is uniform, `t_k = k · dt`; the fits and spectra read their
/// sample interval off it.
#[derive(Debug, Clone, PartialEq)]
pub struct TimeCorrelation {
    /// Lag times `t_k = k · dt`, `k = 0 … max_lag`, in the unit of `dt`.
    pub lag_times: Array1<F>,
    /// `C(t_k)` per channel, shape `(max_lag + 1, n_channels)`.
    pub values: Array2<F>,
}

impl TimeCorrelation {
    /// Number of lags, `max_lag + 1`.
    pub fn n_lags(&self) -> usize {
        self.values.nrows()
    }

    /// Number of channels.
    pub fn n_channels(&self) -> usize {
        self.values.ncols()
    }

    /// The curve of one channel, indexed by lag.
    ///
    /// # Panics
    /// If `channel >= n_channels()`.
    pub fn channel(&self, channel: usize) -> ArrayView1<'_, F> {
        self.values.index_axis(Axis(1), channel)
    }

    /// The curve of `channel`, after checking that it exists, that the lag
    /// grid matches the values and that every value is finite — the guard
    /// every consumer of a correlation calls first.
    pub(crate) fn require_channel(
        &self,
        channel: usize,
    ) -> Result<ArrayView1<'_, F>, ComputeError> {
        if self.lag_times.len() != self.n_lags() {
            return Err(ComputeError::DimensionMismatch {
                expected: self.n_lags(),
                got: self.lag_times.len(),
                what: "lag_times against the rows of values",
            });
        }
        if channel >= self.n_channels() {
            return Err(ComputeError::OutOfRange {
                field: "channel",
                value: format!("{channel} (the correlation has {})", self.n_channels()),
            });
        }
        let curve = self.channel(channel);
        if let Some(index) = curve.iter().position(|v| !v.is_finite()) {
            return Err(ComputeError::NonFinite {
                where_: "correlation channel",
                index,
            });
        }
        Ok(curve)
    }

    /// The curve of a single-channel correlation (see
    /// [`require_channel`](Self::require_channel)); a correlation with
    /// another channel count is [`ComputeError::DimensionMismatch`].
    pub(crate) fn require_single_channel(&self) -> Result<ArrayView1<'_, F>, ComputeError> {
        if self.n_channels() != 1 {
            return Err(ComputeError::DimensionMismatch {
                expected: 1,
                got: self.n_channels(),
                what: "channels of the correlation (reduce it to one channel first)",
            });
        }
        self.require_channel(0)
    }

    /// The sample interval `dt = t_1 − t_0` of the lag grid.
    ///
    /// # Errors
    /// [`ComputeError::EmptyInput`] for fewer than two lags (no interval is
    /// defined); [`ComputeError::OutOfRange`] for a non-positive or non-finite
    /// interval.
    pub(crate) fn sample_interval(&self) -> Result<F, ComputeError> {
        if self.lag_times.len() < 2 {
            return Err(ComputeError::EmptyInput);
        }
        let dt = self.lag_times[1] - self.lag_times[0];
        if !(dt.is_finite() && dt > 0.0) {
            return Err(ComputeError::OutOfRange {
                field: "lag_times (spacing)",
                value: dt.to_string(),
            });
        }
        Ok(dt)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use ndarray::array;

    fn curve() -> TimeCorrelation {
        TimeCorrelation {
            lag_times: array![0.0, 0.5, 1.0],
            values: array![[1.0, 2.0], [0.5, 1.0], [0.25, 0.5]],
        }
    }

    #[test]
    fn guards_report_their_specific_errors() {
        let c = curve();
        assert_eq!(c.require_channel(1).unwrap().to_vec(), vec![2.0, 1.0, 0.5]);
        assert_eq!(c.sample_interval().unwrap(), 0.5);
        assert!(matches!(
            c.require_channel(2),
            Err(ComputeError::OutOfRange {
                field: "channel",
                ..
            })
        ));
        assert!(matches!(
            c.require_single_channel(),
            Err(ComputeError::DimensionMismatch { .. })
        ));

        let mut nan = curve();
        nan.values[[1, 0]] = F::NAN;
        assert!(matches!(
            nan.require_channel(0),
            Err(ComputeError::NonFinite { index: 1, .. })
        ));

        let one_lag = TimeCorrelation {
            lag_times: array![0.0],
            values: array![[1.0]],
        };
        assert!(matches!(
            one_lag.sample_interval(),
            Err(ComputeError::EmptyInput)
        ));
        let flat = TimeCorrelation {
            lag_times: array![1.0, 1.0],
            values: array![[1.0], [1.0]],
        };
        assert!(matches!(
            flat.sample_interval(),
            Err(ComputeError::OutOfRange { .. })
        ));
        let ragged = TimeCorrelation {
            lag_times: array![0.0, 1.0],
            values: array![[1.0], [1.0], [1.0]],
        };
        assert!(matches!(
            ragged.require_channel(0),
            Err(ComputeError::DimensionMismatch { .. })
        ));
    }
}
