//! [`TimeCorrelation`] — the one output of every time correlation.

use molrs::op::F;
use ndarray::{Array1, Array2, ArrayView1, Axis};

/// A time-correlation curve: one row per lag, one column per channel.
///
/// The output of [`Correlator`](super::Correlator) and of every surface over
/// it ([`Msd`](super::Msd), [`Vacf`](super::Vacf), …), and the input of the
/// fits and spectra that turn a curve into a coefficient. Which channels exist
/// is set by the correlator's [`Reduction`](super::Reduction): one for
/// `Mean`, one per item for `PerItem`, one per component for `PerComponent`.
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
}
