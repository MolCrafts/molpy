//! [`DebyeFit`] — the single-exponential (Debye) relaxation fit.

use molrs::op::F;

use super::LinearFit;
use crate::compute::{Compute, ComputeError, TimeCorrelation};

/// Result of a single-exponential / Debye relaxation fit.
#[derive(Debug, Clone)]
#[cfg_attr(feature = "serde", derive(serde::Serialize))]
pub struct DebyeFitResult {
    /// Relaxation time τ, in the unit of the lag times. Positive for a
    /// decaying curve.
    pub tau: F,
    /// Pre-exponential amplitude `A = exp(intercept)`, in the unit of the
    /// curve (≈ 1 for a normalised Φ(t), `C(0)` for a raw ACF).
    pub amplitude: F,
    /// Number of positive samples used in the log-linear fit.
    pub n_samples: usize,
}

/// Single-exponential (Debye) relaxation fit `C(t) = A · exp(−t/τ)` of one
/// channel of a decaying correlation — typically a
/// [`DipoleAcf`](crate::compute::DipoleAcf), a
/// [`ReorientationAcf`](crate::compute::ReorientationAcf) or a
/// [`RotationalAutocorrelation`](crate::compute::RotationalAutocorrelation).
///
/// The fit is a log-linear least-squares line through the **leading run of
/// strictly positive samples** (the decay before the first noise-driven sign
/// change); τ does not depend on whether the curve was normalised, the
/// amplitude carries the curve's scale.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub struct DebyeFit;

/// The curve a [`DebyeFit`] fits.
#[derive(Debug, Clone, Copy)]
pub struct DebyeFitInput<'a> {
    /// The decaying correlation.
    pub correlation: &'a TimeCorrelation,
    /// Which of its channels to fit.
    pub channel: usize,
}

impl Compute for DebyeFit {
    type Input<'a> = DebyeFitInput<'a>;
    type Output = DebyeFitResult;

    /// # Errors
    /// * [`ComputeError::EmptyInput`] if fewer than two leading samples are
    ///   positive (no line through the log).
    /// * [`ComputeError::OutOfRange`] if the fitted slope is non-negative (the
    ///   curve does not decay, so there is no relaxation time), the lag times
    ///   are degenerate or the channel does not exist.
    fn compute(&self, input: DebyeFitInput<'_>) -> Result<DebyeFitResult, ComputeError> {
        let phi = input.correlation.require_channel(input.channel)?;
        let mut t = Vec::new();
        let mut log_phi = Vec::new();
        for (&time, &v) in input.correlation.lag_times.iter().zip(phi.iter()) {
            if v <= 0.0 {
                break;
            }
            t.push(time);
            log_phi.push(v.ln());
        }
        let n_samples = t.len();
        if n_samples < 2 {
            return Err(ComputeError::EmptyInput);
        }

        let (slope, intercept, _r2) =
            LinearFit::ordinary_least_squares(&t, &log_phi, 0, n_samples - 1).ok_or(
                ComputeError::OutOfRange {
                    field: "debye fit (degenerate time axis)",
                    value: format!("n_samples={n_samples}"),
                },
            )?;
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

crate::compute::spec::compute_spec! {
    DebyeFit as "DebyeFit";
    params |s| {}
    inputs {
        correlation: Upstream("TimeCorrelation");
        channel: Index;
    }
}

#[cfg(test)]
mod tests {
    use super::super::curve;
    use super::*;

    fn debye(phi: &[F], dt: F) -> Result<DebyeFitResult, ComputeError> {
        let c = curve(phi, dt);
        DebyeFit.compute(DebyeFitInput {
            correlation: &c,
            channel: 0,
        })
    }

    #[test]
    fn recovers_known_tau_and_amplitude() {
        // Φ(t) = exp(−t/τ), τ = 5.0, dt = 0.5.
        let tau = 5.0;
        let dt = 0.5;
        let phi: Vec<F> = (0..40).map(|k| (-(k as f64) * dt / tau).exp()).collect();
        let res = debye(&phi, dt).unwrap();
        assert!((res.tau - tau).abs() < 1e-9, "tau {}", res.tau);
        assert!((res.amplitude - 1.0).abs() < 1e-9, "amp {}", res.amplitude);
    }

    #[test]
    fn recovers_amplitude_below_one() {
        // Φ(t) = 0.8·exp(−t/3).
        let tau = 3.0;
        let dt = 0.25;
        let a = 0.8;
        let phi: Vec<F> = (0..60)
            .map(|k| a * (-(k as f64) * dt / tau).exp())
            .collect();
        let res = debye(&phi, dt).unwrap();
        assert!((res.tau - tau).abs() < 1e-9);
        assert!((res.amplitude - a).abs() < 1e-9);
    }

    #[test]
    fn stops_at_first_nonpositive_sample() {
        // Positive run of 5, then negative tail.
        let mut v: Vec<f64> = (0..5).map(|k| (-(k as f64) / 4.0).exp()).collect();
        v.extend([-0.1, -0.2, 0.05]);
        let res = debye(&v, 1.0).unwrap();
        assert_eq!(res.n_samples, 5);
    }

    #[test]
    fn non_decaying_acf_errors() {
        // Growing ACF -> slope >= 0.
        let phi: Vec<F> = (0..10).map(|k| (k as f64 / 5.0).exp()).collect();
        assert!(matches!(
            debye(&phi, 1.0),
            Err(ComputeError::OutOfRange { .. })
        ));
    }

    #[test]
    fn too_few_positive_samples_errors() {
        assert!(matches!(
            debye(&[1.0, -1.0, -2.0], 1.0),
            Err(ComputeError::EmptyInput)
        ));
    }

    /// A zero lag spacing (the old `dt <= 0` refusal) leaves every positive
    /// sample at one time: no line.
    #[test]
    fn nonpositive_dt_errors() {
        assert!(matches!(
            debye(&[1.0, 0.5, 0.25], 0.0),
            Err(ComputeError::OutOfRange { .. })
        ));
    }
}
