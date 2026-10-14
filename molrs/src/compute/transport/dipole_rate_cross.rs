//! The raw [`DipoleRateCross`] compute and the NumPy-convention gradient it
//! differentiates the dipole with.

use molrs::core::FrameAccess;
use ndarray::{Array1, Array2, Axis};

use crate::compute::Compute;
use crate::compute::ComputeError;
use crate::compute::ComputeResult;
use crate::compute::{Normalization, TimeCorrelationSpec, cross_correlate, lag_times};

/// Second-order finite difference along axis 0, matching NumPy
/// `np.gradient(..., edge_order=2)` (same length as input).
///
/// Interior: central `(f[i+1] − f[i−1]) / (2 h)`.
/// Ends: one-sided order-2 stencils.
///
/// Walks time as the outer loop so row-major `(n_frames, n_comp)` layouts hit
/// consecutive components together.
pub fn gradient_axis0_order2(series: &Array2<f64>, dt: f64) -> Result<Array2<f64>, ComputeError> {
    let n = series.shape()[0];
    let n_comp = series.shape()[1];
    if n < 2 {
        return Err(ComputeError::EmptyInput);
    }
    if dt <= 0.0 {
        return Err(ComputeError::OutOfRange {
            field: "dt",
            value: dt.to_string(),
        });
    }
    let mut out = Array2::<f64>::zeros((n, n_comp));
    let inv_h = 1.0 / dt;
    let inv_2h = 0.5 * inv_h;

    if n == 2 {
        // NumPy edge_order=2 falls back to first-order for n < 3.
        for d in 0..n_comp {
            let g = (series[[1, d]] - series[[0, d]]) * inv_h;
            out[[0, d]] = g;
            out[[1, d]] = g;
        }
        return Ok(out);
    }

    // t = 0: forward order-2
    for d in 0..n_comp {
        out[[0, d]] = (-1.5 * series[[0, d]] + 2.0 * series[[1, d]] - 0.5 * series[[2, d]]) * inv_h;
    }
    // interior
    for t in 1..n - 1 {
        for d in 0..n_comp {
            out[[t, d]] = (series[[t + 1, d]] - series[[t - 1, d]]) * inv_2h;
        }
    }
    // t = n-1: backward order-2
    for d in 0..n_comp {
        out[[n - 1, d]] = (0.5 * series[[n - 3, d]] - 2.0 * series[[n - 2, d]]
            + 1.5 * series[[n - 1, d]])
            * inv_h;
    }
    Ok(out)
}

// ── DipoleRateCross raw Compute ──────────────────────────────────────────────

/// Raw dipole-rate × dipole cross-correlation
/// `C_{ṀM}(τ) = Σ_α ⟨δṀ_α(0) δM_α(τ)⟩`.
///
/// `Ṁ` is a second-order finite difference of the unwrapped total dipole
/// (NumPy `gradient` / edge_order=2 convention — same length as `M`). Both
/// series are mean-subtracted before the cross-correlation, which is
/// [`cross_correlate`] on the two `(n_frames, 1, 3)` views with the unbiased
/// normalization.
///
/// Compose with
/// [`DipoleRateCrossSpectrum`](crate::compute::DipoleRateCrossSpectrum)
/// for ε(ω).
///
/// `Default` is the estimator the spectrum expects: mean-subtracted,
/// unbiased, every time origin.
#[derive(Debug, Clone, Copy)]
pub struct DipoleRateCross {
    /// Subtract the time mean of `Ṁ` and `M` first.
    pub mean_subtract: bool,
    /// How the lag sums are scaled.
    pub normalization: Normalization,
    /// Time origins every `origin_stride` frames.
    pub origin_stride: usize,
}

impl Default for DipoleRateCross {
    fn default() -> Self {
        Self {
            mean_subtract: true,
            normalization: Normalization::Unbiased,
            origin_stride: 1,
        }
    }
}

impl DipoleRateCross {
    /// Mean-subtracted, unbiased, every origin.
    pub fn new() -> Self {
        Self::default()
    }
}

/// Result of [`DipoleRateCross`]: lag grid + unbiased cross correlator.
#[derive(Debug, Clone)]
pub struct DipoleRateCrossResult {
    /// Lag times τ = i·dt, length `max_lag + 1`. Units: `[dt]`.
    pub lag_times: Array1<f64>,
    /// `C_{ṀM}(τ)` summed over Cartesian components. Units: `(e·Å/ps)·(e·Å)`.
    pub cross: Array1<f64>,
}

impl ComputeResult for DipoleRateCrossResult {}

/// `(dipole_moments (n,3), dt, max_correlation_time)` for [`DipoleRateCross`].
pub type DipoleRateCrossArgs<'a> = (&'a Array2<f64>, f64, usize);

impl Compute for DipoleRateCross {
    type Args<'a> = DipoleRateCrossArgs<'a>;
    type Output = DipoleRateCrossResult;

    fn compute<'a, FA: FrameAccess + Sync + 'a>(
        &self,
        _frames: &[&'a FA],
        args: Self::Args<'a>,
    ) -> Result<Self::Output, ComputeError> {
        let (dipole, dt, max_correlation_time) = args;
        let shape = dipole.shape();
        if shape[1] != 3 {
            return Err(ComputeError::DimensionMismatch {
                expected: 3,
                got: shape[1],
                what: "dipole_moments (expected (n_frames, 3))",
            });
        }
        let n_frames = shape[0];
        if n_frames < 2 {
            return Err(ComputeError::EmptyInput);
        }
        if dt <= 0.0 {
            return Err(ComputeError::OutOfRange {
                field: "dt",
                value: dt.to_string(),
            });
        }
        let max_lag = max_correlation_time.min(n_frames - 1);
        let mdot = gradient_axis0_order2(dipole, dt)?;
        let cross = cross_correlate(
            mdot.view().insert_axis(Axis(1)),
            dipole.view().insert_axis(Axis(1)),
            dt,
            &TimeCorrelationSpec::new()
                .max_lag(max_lag)
                .mean_subtract(true),
        )?
        .values;
        Ok(DipoleRateCrossResult {
            lag_times: lag_times(max_lag, dt),
            cross,
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use molrs::core::Frame;
    use ndarray::Array2;
    use rand::{RngExt, SeedableRng};

    fn no_frames() -> Vec<&'static Frame> {
        Vec::new()
    }

    fn rng_series(n: usize, cols: usize, seed: u64) -> Array2<f64> {
        let mut rng = rand::rngs::StdRng::seed_from_u64(seed);
        let mut s = Array2::zeros((n, cols));
        for t in 0..n {
            for c in 0..cols {
                s[[t, c]] = rng.random_range(-1.0..1.0);
            }
        }
        s
    }

    #[test]
    fn gradient_matches_numpy_edge_order2_values() {
        let mut s = Array2::zeros((5, 1));
        for t in 0..5 {
            s[[t, 0]] = (t as f64).powi(2);
        }
        let g = gradient_axis0_order2(&s, 1.0).unwrap();
        assert!((g[[1, 0]] - 2.0).abs() < 1e-12);
        assert!((g[[2, 0]] - 4.0).abs() < 1e-12);
        assert!((g[[3, 0]] - 6.0).abs() < 1e-12);
        assert!((g[[0, 0]] - 0.0).abs() < 1e-12);
        assert!((g[[4, 0]] - 8.0).abs() < 1e-12);
    }

    #[test]
    fn dipole_rate_cross_finite_and_length() {
        let n = 128;
        let dt = 0.01;
        let mct = 40;
        let dm = rng_series(n, 3, 42);
        let raw = DipoleRateCross::new()
            .compute(&no_frames(), (&dm, dt, mct))
            .unwrap();
        assert_eq!(raw.cross.len(), mct + 1);
        assert_eq!(raw.lag_times.len(), mct + 1);
        assert!(raw.cross.iter().all(|v| v.is_finite()));
        assert!((raw.lag_times[1] - dt).abs() < 1e-15);
    }

    #[test]
    fn dipole_rate_cross_matches_the_direct_definition() {
        // Ṁ from the gradient, both series mean-subtracted, unbiased
        // Cartesian-sum cross-correlation written out.
        let n = 96;
        let dt = 0.02;
        let mct = 30;
        let dm = rng_series(n, 3, 7);
        let raw = DipoleRateCross::new()
            .compute(&no_frames(), (&dm, dt, mct))
            .unwrap();
        let mdot = gradient_axis0_order2(&dm, dt).unwrap();
        let mean = |s: &Array2<f64>, d: usize| (0..n).map(|t| s[[t, d]]).sum::<f64>() / n as f64;
        for tau in 0..=mct {
            let count = n - tau;
            let mut acc = 0.0;
            for t in 0..count {
                for d in 0..3 {
                    acc += (mdot[[t, d]] - mean(&mdot, d)) * (dm[[t + tau, d]] - mean(&dm, d));
                }
            }
            let expected = acc / count as f64;
            assert!(
                (raw.cross[tau] - expected).abs() < 1e-10,
                "tau={tau}: {} vs {expected}",
                raw.cross[tau]
            );
        }
    }
}
