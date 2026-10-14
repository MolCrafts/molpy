//! [`Acf`] — the correlation engine as a [`Compute`], for a series the
//! caller has already built.

use ndarray::Array3;

use super::engine::{TimeCorrelationSpec, autocorrelate};
use super::result::{Normalization, TimeCorrelation};
use crate::compute::Compute;
use crate::compute::ComputeError;
use molrs::core::FrameAccess;

/// Time-autocorrelation of a vector series, averaged over all time origins.
///
/// `C(t) = norm · Σ_i Σ_α δv_iα(τ) · δv_iα(τ+t)` over the
/// `(n_frames T, n_entities N, n_components D)` series, where `δv = v − ⟨v⟩_τ`
/// when `mean_subtract` and `δv = v` otherwise; see [`autocorrelate`] for the
/// estimator and its provenance. The lag axis is in frames (`dt = 1`); scale
/// [`TimeCorrelation::lag_times`] yourself, or build the curve through a
/// quantity front end (`Vacf`, `GreenKuboConductivity`, …) that knows the
/// time step.
///
/// `Default` is no mean subtraction, the unbiased normalization and every
/// time origin. The frames slice is unused: an autocorrelation is a property
/// of the series, and the series need not be positions.
#[derive(Debug, Clone, Copy)]
pub struct Acf {
    /// Subtract each `(entity, component)` column's time mean first.
    pub mean_subtract: bool,
    /// How the lag sums are scaled.
    pub normalization: Normalization,
    /// Time origins every `origin_stride` frames.
    pub origin_stride: usize,
}

impl Default for Acf {
    fn default() -> Self {
        Self {
            mean_subtract: false,
            normalization: Normalization::Unbiased,
            origin_stride: 1,
        }
    }
}

impl Acf {
    /// No mean subtraction, unbiased, every origin.
    pub fn new() -> Self {
        Self::default()
    }

    pub fn with_mean_subtract(mut self, yes: bool) -> Self {
        self.mean_subtract = yes;
        self
    }

    pub fn with_normalization(mut self, normalization: Normalization) -> Self {
        self.normalization = normalization;
        self
    }

    pub fn with_origin_stride(mut self, stride: usize) -> Self {
        self.origin_stride = stride;
        self
    }

    /// The engine settings this front end stands for.
    pub fn spec(&self, max_lag: usize) -> TimeCorrelationSpec<'static> {
        TimeCorrelationSpec::new()
            .max_lag(max_lag)
            .mean_subtract(self.mean_subtract)
            .normalization(self.normalization)
            .origin_stride(self.origin_stride)
    }
}

/// `(series (n_frames, n_entities, n_components), max_lag)` for [`Acf`].
pub type AcfArgs<'a> = (&'a Array3<f64>, usize);

impl Compute for Acf {
    type Args<'a> = AcfArgs<'a>;
    type Output = TimeCorrelation;

    fn compute<'a, FA: FrameAccess + Sync + 'a>(
        &self,
        _frames: &[&'a FA],
        (series, max_lag): Self::Args<'a>,
    ) -> Result<Self::Output, ComputeError> {
        autocorrelate(series.view(), 1.0, &self.spec(max_lag))
    }
}

#[cfg(test)]
mod tests {
    use super::super::engine::tests::series;
    use super::*;
    use molrs::core::Frame;

    #[test]
    fn the_compute_is_the_engine_with_its_settings() {
        let s = series(40, 3, 3);
        let frames: Vec<&Frame> = Vec::new();
        let via_compute = Acf::new()
            .with_mean_subtract(true)
            .with_origin_stride(2)
            .compute(&frames, (&s, 9))
            .unwrap();
        let direct = autocorrelate(
            s.view(),
            1.0,
            &TimeCorrelationSpec::new()
                .max_lag(9)
                .mean_subtract(true)
                .origin_stride(2),
        )
        .unwrap();
        assert_eq!(via_compute.values, direct.values);
        assert_eq!(via_compute.dt, 1.0);
        assert_eq!(via_compute.lag_times()[9], 9.0);
    }
}
