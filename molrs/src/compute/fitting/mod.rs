//! Generic curve fitting / smoothing — the [`Fit`](crate::compute::Fit) companion of
//! [`Compute`](crate::compute::Compute).
//!
//! Where a [`Compute`](crate::compute::Compute) measures a raw observable from
//! frames (an MSD curve, a current ACF, a velocity ACF), a
//! [`Fit`](crate::compute::Fit) post-processes
//! that observable into a derived quantity. This module hosts the
//! **domain-agnostic** curve transforms; the raw correlation curves come from
//! [`Correlator`](crate::compute::Correlator) and its surfaces, and the
//! spectral transforms live in `spectroscopy`.
//!
//! | Fit | Input | Output | Lifted from |
//! |-----|-------|--------|-------------|
//! | [`LinearFit`] | `(x, y)` curve | [`LinearFitResult`] (slope/intercept/r²) | Einstein–Helfand conductivity OLS |
//! | [`CumulativeTrapezoid`] | curve + dt | [`CumulativeTrapezoidResult`] (cumulative trapezoid) | Green–Kubo conductivity trapezoid |
//! | [`Plateau`] | curve | [`PlateauResult`] (windowed mean/std) | new |
//! | [`DebyeFit`] | normalised decay + dt | [`DebyeFitResult`] (τ, amplitude) | Debye relaxation |
//!
//! # Shared numerical primitives
//!
//! Helpers are lifted here so the fits share one implementation:
//!
//! - `ols_slope_intercept_r2` — ordinary-least-squares line fit (the
//!   Einstein–Helfand conductivity slope).
//! - `running_trapezoid` — cumulative trapezoidal integral (the Green–Kubo
//!   conductivity integral).
//!
//! The one-sided forward FFT the spectra share is
//! [`Spectrum`](crate::compute::Spectrum).

mod cumulative_trapezoid;
mod debye_fit;
mod linear_fit;
mod plateau;

pub use cumulative_trapezoid::{CumulativeTrapezoid, CumulativeTrapezoidResult};
pub use debye_fit::{DebyeFit, DebyeFitResult};
pub use linear_fit::{LinearFit, LinearFitResult};
pub use plateau::{Plateau, PlateauResult};

/// Ordinary least-squares fit of `y = slope·x + intercept` over the inclusive
/// index range `[start, end]`.
///
/// Lifted verbatim (slope block) from
/// the Einstein–Helfand ionic conductivity:
/// `denom = np·sxx − sx·sx`, `slope = (np·sxy − sx·sy)/denom`, extended with
/// `intercept = (sy − slope·sx)/np` and the coefficient of determination `r²`.
///
/// # Returns
/// `(slope, intercept, r2)`. Units: `slope` is `[y]/[x]`, `intercept` is `[y]`,
/// `r2` is dimensionless in `[0, 1]`.
///
/// # Errors
/// Returns `None` when the design is degenerate (`denom ≈ 0`, i.e. all `x` in
/// the window are equal) — the caller maps this to
/// [`ComputeError::OutOfRange`](crate::compute::ComputeError::OutOfRange).
pub(crate) fn ols_slope_intercept_r2(
    x: &[f64],
    y: &[f64],
    start: usize,
    end: usize,
) -> Option<(f64, f64, f64)> {
    let np = (end - start + 1) as f64;
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
    if denom.abs() < f64::EPSILON {
        return None;
    }
    let slope = (np * sxy - sx * sy) / denom;
    let intercept = (sy - slope * sx) / np;

    // r² = 1 − SS_res / SS_tot.
    let y_mean = sy / np;
    let mut ss_res = 0.0;
    let mut ss_tot = 0.0;
    for i in start..=end {
        let pred = slope * x[i] + intercept;
        let resid = y[i] - pred;
        ss_res += resid * resid;
        let dev = y[i] - y_mean;
        ss_tot += dev * dev;
    }
    // Perfectly-flat y (ss_tot == 0): the line is exact iff residuals vanish.
    let r2 = if ss_tot.abs() < f64::EPSILON {
        if ss_res.abs() < f64::EPSILON {
            1.0
        } else {
            0.0
        }
    } else {
        1.0 - ss_res / ss_tot
    };
    Some((slope, intercept, r2))
}

/// Cumulative trapezoidal integral of `y` on uniform step `dt`.
///
/// Lifted from the Green–Kubo ionic conductivity:
/// `integral += 0.5·(y[k−1] + y[k])·dt; out[k] = integral`. Element `0` is `0`.
///
/// # Returns
/// A length-`y.len()` array; `out[k] = ∫₀^{k·dt} y(t) dt` (trapezoid rule).
pub(crate) fn running_trapezoid(y: &[f64], dt: f64) -> Vec<f64> {
    let n = y.len();
    let mut out = vec![0.0; n];
    let mut integral = 0.0;
    for k in 1..n {
        integral += 0.5 * (y[k - 1] + y[k]) * dt;
        out[k] = integral;
    }
    out
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn ols_recovers_exact_line() {
        let x: Vec<f64> = (0..10).map(|i| i as f64).collect();
        let y: Vec<f64> = x.iter().map(|&xi| 3.0 * xi + 2.0).collect();
        let (slope, intercept, r2) = ols_slope_intercept_r2(&x, &y, 0, 9).unwrap();
        assert!((slope - 3.0).abs() < 1e-12);
        assert!((intercept - 2.0).abs() < 1e-12);
        assert!((r2 - 1.0).abs() < 1e-12);
    }

    #[test]
    fn ols_degenerate_returns_none() {
        let x = vec![5.0, 5.0, 5.0];
        let y = vec![1.0, 2.0, 3.0];
        assert!(ols_slope_intercept_r2(&x, &y, 0, 2).is_none());
    }

    #[test]
    fn trapezoid_of_constant() {
        let y = vec![2.0; 5];
        let out = running_trapezoid(&y, 0.5);
        for (k, &v) in out.iter().enumerate() {
            assert!((v - 2.0 * k as f64 * 0.5).abs() < 1e-12);
        }
    }
}
