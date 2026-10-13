//! [`DebyeFit`] — single-exponential relaxation time of a normalized
//! correlation function.

use ndarray::Array1;

use super::ols_slope_intercept_r2;
use crate::compute::ComputeError;
use crate::compute::ComputeResult;
use crate::compute::Fit;

/// Result of a single-exponential / Debye relaxation fit.
#[derive(Debug, Clone)]
pub struct DebyeFitResult {
    /// Relaxation time τ, units `[dt]`. Positive for a decaying ACF.
    pub tau: f64,
    /// Pre-exponential amplitude A = exp(intercept), dimensionless for a
    /// normalized Φ (≈ 1).
    pub amplitude: f64,
    /// Number of positive samples used in the log-linear fit.
    pub n_samples: usize,
}

impl ComputeResult for DebyeFitResult {}

/// Single-exponential (Debye) relaxation fit of a normalized correlation
/// function: `Φ(t) = A · exp(−t/τ)` by log-linear least squares over the
/// leading run of positive samples.
///
/// Stateless: `dt` travels with the input curve. The Fit stage of
/// [`DebyeRelaxation`](crate::compute::DebyeRelaxation) (τ_D of the dipole
/// ACF), and of any other decaying [`TimeCorrelation`](crate::compute::TimeCorrelation)
/// through [`TimeCorrelation::exponential_fit`](crate::compute::TimeCorrelation::exponential_fit)
/// — the reorientational τ_ℓ of an [`Oacf`](crate::compute::Oacf) curve, say.
#[derive(Debug, Clone, Copy, Default)]
pub struct DebyeFit;

impl Fit for DebyeFit {
    /// `(phi, dt)` — the normalized dipole ACF Φ(t) and its sample step (> 0).
    type Input<'a> = (&'a Array1<f64>, f64);
    type Output = DebyeFitResult;

    /// Fit Φ(t) = A·exp(−t/τ) by log-linear least squares.
    ///
    /// # Errors
    /// * [`ComputeError::OutOfRange`] if `dt <= 0`.
    /// * [`ComputeError::EmptyInput`] if fewer than two positive samples remain
    ///   (cannot fit a line) — falls under invariant (a) when the leading
    ///   positive run is too short.
    /// * [`ComputeError::OutOfRange`] if the fitted slope is non-negative (the
    ///   ACF does not decay → no physical relaxation time).
    fn fit<'a>(&self, input: Self::Input<'a>) -> Result<Self::Output, ComputeError> {
        let (phi, dt) = input;
        if dt <= 0.0 {
            return Err(ComputeError::OutOfRange {
                field: "dt",
                value: dt.to_string(),
            });
        }

        // Use the leading run of strictly-positive samples (the exponential
        // decay before any noise-driven sign change).
        let mut t = Vec::new();
        let mut log_phi = Vec::new();
        for (k, &v) in phi.iter().enumerate() {
            if v <= 0.0 {
                break;
            }
            t.push(k as f64 * dt);
            log_phi.push(v.ln());
        }
        let n_samples = t.len();
        if n_samples < 2 {
            return Err(ComputeError::EmptyInput);
        }

        let (slope, intercept, _r2) = ols_slope_intercept_r2(&t, &log_phi, 0, n_samples - 1)
            .ok_or(ComputeError::OutOfRange {
                field: "debye fit (degenerate time axis)",
                value: format!("n_samples={n_samples}"),
            })?;

        if slope >= 0.0 {
            return Err(ComputeError::OutOfRange {
                field: "debye fit slope (require negative for decay)",
                value: slope.to_string(),
            });
        }

        Ok(DebyeFitResult {
            tau: -1.0 / slope,
            amplitude: intercept.exp(),
            n_samples,
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn recovers_known_tau_and_amplitude() {
        // Φ(t) = exp(−t/τ), τ = 5.0, dt = 0.5.
        let tau = 5.0;
        let dt = 0.5;
        let phi = Array1::from_iter((0..40).map(|k| (-(k as f64) * dt / tau).exp()));
        let res = DebyeFit.fit((&phi, dt)).unwrap();
        assert!((res.tau - tau).abs() < 1e-9, "tau {}", res.tau);
        assert!((res.amplitude - 1.0).abs() < 1e-9, "amp {}", res.amplitude);
    }

    #[test]
    fn recovers_amplitude_below_one() {
        // Φ(t) = 0.8·exp(−t/3).
        let tau = 3.0;
        let dt = 0.25;
        let a = 0.8;
        let phi = Array1::from_iter((0..60).map(|k| a * (-(k as f64) * dt / tau).exp()));
        let res = DebyeFit.fit((&phi, dt)).unwrap();
        assert!((res.tau - tau).abs() < 1e-9);
        assert!((res.amplitude - a).abs() < 1e-9);
    }

    #[test]
    fn stops_at_first_nonpositive_sample() {
        // Positive run of 5, then negative tail.
        let dt = 1.0;
        let mut v: Vec<f64> = (0..5).map(|k| (-(k as f64) / 4.0).exp()).collect();
        v.extend([-0.1, -0.2, 0.05]);
        let phi = Array1::from_vec(v);
        let res = DebyeFit.fit((&phi, dt)).unwrap();
        assert_eq!(res.n_samples, 5);
    }

    #[test]
    fn non_decaying_acf_errors() {
        // Growing ACF -> slope >= 0.
        let phi = Array1::from_iter((0..10).map(|k| (k as f64 / 5.0).exp()));
        assert!(matches!(
            DebyeFit.fit((&phi, 1.0)),
            Err(ComputeError::OutOfRange { .. })
        ));
    }

    #[test]
    fn too_few_positive_samples_errors() {
        let phi = Array1::from_vec(vec![1.0, -1.0, -2.0]);
        assert!(matches!(
            DebyeFit.fit((&phi, 1.0)),
            Err(ComputeError::EmptyInput)
        ));
    }

    #[test]
    fn nonpositive_dt_errors() {
        let phi = Array1::from_vec(vec![1.0, 0.5, 0.25]);
        assert!(matches!(
            DebyeFit.fit((&phi, 0.0)),
            Err(ComputeError::OutOfRange { .. })
        ));
    }
}
