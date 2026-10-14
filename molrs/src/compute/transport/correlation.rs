//! Shared Cartesian correlators and the raw [`DipoleRateCross`] compute.

use molrs::core::FrameAccess;
use ndarray::{Array1, Array2};
use rustfft::FftPlanner;
use rustfft::num_complex::Complex64;

use crate::compute::ComputeError;
use crate::compute::ComputeResult;
use crate::compute::LegacyCompute;
use molrs::signal as sig;

// ── Shared primitives ────────────────────────────────────────────────────────

/// Lag grid `τ = i·dt` for `i = 0..=max_lag`.
#[inline]
pub fn lag_times(max_lag: usize, dt: f64) -> Array1<f64> {
    Array1::from_iter((0..=max_lag).map(|i| i as f64 * dt))
}

/// Unbiased time-origin normalisation with an optional overall scale:
/// `C[k] *= scale / (n_frames − k)`.
///
/// The factor `scale/(n−k)` is formed once and applied with one multiply, not
/// `/(n−k)` then `*scale`, so VACF and JACF round identically.
#[inline]
pub fn apply_unbiased_norm(corr: &mut Array1<f64>, n_frames: usize, scale: f64) {
    let max_lag = corr.len().saturating_sub(1);
    for k in 0..=max_lag {
        corr[k] *= scale / (n_frames - k) as f64;
    }
}

/// Per-component trajectory means, length `n_comp`.
pub fn component_means(series: &Array2<f64>) -> Vec<f64> {
    let n_frames = series.shape()[0];
    let n_comp = series.shape()[1];
    let inv_n = 1.0 / n_frames as f64;
    let mut mean = vec![0.0_f64; n_comp];
    // Row-major friendly: walk time outer, components inner.
    for t in 0..n_frames {
        for d in 0..n_comp {
            mean[d] += series[[t, d]];
        }
    }
    for m in mean.iter_mut() {
        *m *= inv_n;
    }
    mean
}

/// Copy column `d` into `out` (length `n_frames`), optionally subtracting `mean`.
#[inline]
fn fill_column(series: &Array2<f64>, d: usize, mean: f64, out: &mut [f64]) {
    debug_assert_eq!(out.len(), series.shape()[0]);
    for (t, slot) in out.iter_mut().enumerate() {
        *slot = series[[t, d]] - mean;
    }
}

// ── Shared Cartesian helpers ─────────────────────────────────────────────────

/// Unbiased Cartesian-sum cross-correlation
/// `C(τ) = Σ_α ⟨a_α(0) b_α(τ)⟩` (optionally of fluctuations).
///
/// Both series must share shape `(n_frames, n_comp)`.
///
/// Several dielectric / transport raw Computes need the same recipe:
///
/// 1. optional per-component mean subtraction,
/// 2. FFT linear ACF / xcorr via [`molrs::signal`] accumulate primitives,
/// 3. sum over Cartesian axes,
/// 4. unbiased `1/(n − τ)` normalisation.
///
/// | Helper / Compute | Correlator | Downstream Fit |
/// |------------------|------------|----------------|
/// | [`autocorrelation`](crate::compute::autocorrelation) | `Σ_α ⟨δa_α(0) δa_α(t)⟩` | Debye / PACF / JACF / VACF |
/// | this function | `Σ_α ⟨δa_α(0) δb_α(t)⟩` | cross spectra |
/// | [`DipoleRateCross`](crate::compute::DipoleRateCross) | `C_{ṀM}(t)` with FD `Ṁ` | [`DipoleRateCrossSpectrum`](crate::compute::DipoleRateCrossSpectrum) |
///
/// Signal kernels stay in [`molrs::signal`]; multi-component assembly lives here.
pub fn unbiased_cartesian_xcorr(
    a: &Array2<f64>,
    b: &Array2<f64>,
    max_lag: usize,
    mean_subtract: bool,
) -> Result<Array1<f64>, ComputeError> {
    let n_frames = a.shape()[0];
    let n_comp = a.shape()[1];
    if b.shape() != a.shape() {
        return Err(ComputeError::BadShape {
            expected: format!("({}, {})", n_frames, n_comp),
            got: format!("{:?}", b.shape()),
        });
    }
    if n_frames < 2 || n_comp == 0 {
        return Err(ComputeError::EmptyInput);
    }
    let max_lag = max_lag.min(n_frames - 1);

    let (mean_a, mean_b) = if mean_subtract {
        (component_means(a), component_means(b))
    } else {
        (vec![0.0; n_comp], vec![0.0; n_comp])
    };

    let mut planner = FftPlanner::new();
    let mut corr = Array1::<f64>::zeros(max_lag + 1);
    let mut ca = vec![0.0_f64; n_frames];
    let mut cb = vec![0.0_f64; n_frames];
    let mut sa: Vec<Complex64> = Vec::new();
    let mut sb: Vec<Complex64> = Vec::new();
    let out = corr.as_slice_mut().expect("corr is contiguous");

    for d in 0..n_comp {
        fill_column(a, d, mean_a[d], &mut ca);
        fill_column(b, d, mean_b[d], &mut cb);
        sig::xcorr_fft_accumulate(&mut planner, &ca, &cb, max_lag, out, &mut sa, &mut sb).map_err(
            |e| ComputeError::OutOfRange {
                field: "xcorr_fft",
                value: e.to_string(),
            },
        )?;
    }
    apply_unbiased_norm(&mut corr, n_frames, 1.0);
    Ok(corr)
}

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
/// series are mean-subtracted before the FFT cross-correlation.
///
/// Compose with
/// [`DipoleRateCrossSpectrum`](crate::compute::DipoleRateCrossSpectrum)
/// for ε(ω).
#[derive(Debug, Clone, Copy, Default)]
pub struct DipoleRateCross;

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

impl LegacyCompute for DipoleRateCross {
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
        let cross = unbiased_cartesian_xcorr(&mdot, dipole, max_lag, true)?;
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
    fn cartesian_acf_matches_direct_sum() {
        let n = 64;
        let max_lag = 20;
        let s = rng_series(n, 3, 11);
        let got =
            crate::compute::autocorrelation(s.view().insert_axis(ndarray::Axis(1)), max_lag, true)
                .unwrap()
                .acf;
        let mut mean = [0.0_f64; 3];
        for t in 0..n {
            for d in 0..3 {
                mean[d] += s[[t, d]];
            }
        }
        for m in mean.iter_mut() {
            *m /= n as f64;
        }
        for tau in 0..=max_lag {
            let count = n - tau;
            let mut acc = 0.0;
            for t in 0..count {
                for d in 0..3 {
                    let a = s[[t, d]] - mean[d];
                    let b = s[[t + tau, d]] - mean[d];
                    acc += a * b;
                }
            }
            let expected = acc / count as f64;
            assert!(
                (got[tau] - expected).abs() < 1e-10,
                "tau={tau}: {} vs {expected}",
                got[tau]
            );
        }
    }

    #[test]
    fn cartesian_xcorr_of_self_matches_acf() {
        let s = rng_series(48, 3, 3);
        let acf = crate::compute::autocorrelation(s.view().insert_axis(ndarray::Axis(1)), 15, true)
            .unwrap()
            .acf;
        let xc = unbiased_cartesian_xcorr(&s, &s, 15, true).unwrap();
        for k in 0..acf.len() {
            assert!((acf[k] - xc[k]).abs() < 1e-12, "k={k}");
        }
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
        let raw = DipoleRateCross
            .compute(&no_frames(), (&dm, dt, mct))
            .unwrap();
        assert_eq!(raw.cross.len(), mct + 1);
        assert_eq!(raw.lag_times.len(), mct + 1);
        assert!(raw.cross.iter().all(|v| v.is_finite()));
        assert!((raw.lag_times[1] - dt).abs() < 1e-15);
    }

    #[test]
    fn dipole_rate_cross_matches_manual_pipeline() {
        let n = 96;
        let dt = 0.02;
        let mct = 30;
        let dm = rng_series(n, 3, 7);
        let raw = DipoleRateCross
            .compute(&no_frames(), (&dm, dt, mct))
            .unwrap();
        let mdot = gradient_axis0_order2(&dm, dt).unwrap();
        let expected = unbiased_cartesian_xcorr(&mdot, &dm, mct, true).unwrap();
        for k in 0..raw.cross.len() {
            assert!((raw.cross[k] - expected[k]).abs() < 1e-14, "k={k}");
        }
    }
}
