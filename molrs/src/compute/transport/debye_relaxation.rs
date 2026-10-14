//! Debye dipole-relaxation raw compute — the dipole-ACF route to ε(ω) and τ_D.

use molrs::core::FrameAccess;
use ndarray::{Array1, Array2, Axis};

use super::correlation::lag_times;
use crate::compute::ComputeError;
use crate::compute::ComputeResult;
use crate::compute::autocorrelation;
use crate::compute::fitting::ols_slope_intercept_r2;
use crate::compute::{Fit, LegacyCompute};

/// Ewald boundary condition under which the dipole fluctuations were sampled.
///
/// The Debye / Neumann–Kirkwood amplitude `ε₀ − ε∞` depends on this boundary
/// condition, so it travels with the raw result (spec invariant (c)).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum EwaldBoundary {
    /// Conducting / "tin-foil" boundary (ε' = ∞) — the common MD default.
    TinFoil,
    /// Vacuum boundary (ε' = 1).
    Vacuum,
}

impl EwaldBoundary {
    /// Parse a boundary-condition name (case-insensitive): `"tinfoil"` /
    /// `"tin-foil"` / `"tin_foil"` / `"conducting"` → [`TinFoil`](Self::TinFoil),
    /// `"vacuum"` → [`Vacuum`](Self::Vacuum).
    pub fn from_name(name: &str) -> Result<Self, ComputeError> {
        match name.to_ascii_lowercase().as_str() {
            "tinfoil" | "tin-foil" | "tin_foil" | "conducting" => Ok(Self::TinFoil),
            "vacuum" => Ok(Self::Vacuum),
            other => Err(ComputeError::OutOfRange {
                field: "EwaldBoundary",
                value: other.to_string(),
            }),
        }
    }
}

/// Raw dipole ACF plus the scalar metadata the Debye amplitude needs.
///
/// Unlike the spectral/diffusion raw results, this carries an **unnormalized**
/// ACF and the zero-lag variance ⟨M(0)²⟩ explicitly (invariant (b)): the
/// normalized Φ(t) gives only the relaxation *shape*/τ via
/// [`DebyeFit`]; the amplitude `ε₀ − ε∞` comes
/// from ⟨M²⟩ together with V, T, and the Ewald boundary condition
/// (invariant (c)).
///
/// All four metadata fields are non-optional, so a `DebyeRelaxationResult`
/// **cannot be constructed without them**.
#[derive(Debug, Clone)]
pub struct DebyeRelaxationResult {
    /// Lag times τ = i·dt, length `max_lag + 1`. Units: `[dt]`.
    pub lag_times: Array1<f64>,
    /// **Unnormalized** dipole ACF `C(τ) = ⟨δM(0)·δM(τ)⟩` summed over the 3
    /// Cartesian components. Units: `(e·Å)²`.
    pub acf: Array1<f64>,
    /// Zero-lag variance ⟨M(0)²⟩ = `acf[0]`, the Debye amplitude scale
    /// (invariant b). Units: `(e·Å)²`.
    pub zero_lag_variance: f64,
    /// System volume V (invariant c). Units: `Å³`.
    pub volume: f64,
    /// Temperature T (invariant c). Units: `K`.
    pub temperature: f64,
    /// Ewald boundary condition the fluctuations were sampled under
    /// (invariant c).
    pub boundary: EwaldBoundary,
}

impl ComputeResult for DebyeRelaxationResult {}

/// Raw dipole-ACF compute for the Debye route. Computes the mean-subtracted,
/// per-component-summed dipole ACF (unbiased estimator) and carries the
/// zero-lag variance + V/T/Ewald-BC metadata the amplitude needs.
#[derive(Debug, Clone, Copy)]
pub struct DebyeRelaxation {
    /// System volume V. Units: `Å³`.
    pub volume: f64,
    /// Temperature T. Units: `K`.
    pub temperature: f64,
    /// Ewald boundary condition.
    pub boundary: EwaldBoundary,
}

/// `(dipole_moments, dt, max_correlation_time)` bundle for [`DebyeRelaxation`].
pub type DebyeRelaxationArgs<'a> = (&'a Array2<f64>, f64, usize);

impl LegacyCompute for DebyeRelaxation {
    type Args<'a> = DebyeRelaxationArgs<'a>;
    type Output = DebyeRelaxationResult;

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
        if self.volume <= 0.0 {
            return Err(ComputeError::OutOfRange {
                field: "volume",
                value: self.volume.to_string(),
            });
        }
        if self.temperature <= 0.0 {
            return Err(ComputeError::OutOfRange {
                field: "temperature",
                value: self.temperature.to_string(),
            });
        }
        let max_lag = max_correlation_time.min(n_frames - 1);

        // Fluctuation ACF so acf[0] = ⟨|δM|²⟩ (shared cartesian helper).
        let acf = autocorrelation(dipole.view().insert_axis(Axis(1)), max_lag, true)?.acf;
        let zero_lag_variance = acf[0];
        Ok(DebyeRelaxationResult {
            lag_times: lag_times(max_lag, dt),
            acf,
            zero_lag_variance,
            volume: self.volume,
            temperature: self.temperature,
            boundary: self.boundary,
        })
    }
}

// ─── Debye fit ───────────────────────────────────────────────────────────
// The Fit stage for this same quantity: `DebyeRelaxation` above produces the
// raw dipole ACF, and `DebyeFit` reads the relaxation time off it. compute,
// fit and check for one physical quantity live in one file.

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

/// Single-exponential (Debye) relaxation fit of a normalized dipole ACF.
///
/// Stateless: `dt` travels with the input curve.
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
    fn debye_relaxation_carries_unnormalized_acf_and_metadata() {
        // ac-013.
        let n = 128;
        let dt = 0.5;
        let dipole = rng_series(n, 3, 9);
        let res = DebyeRelaxation {
            volume: 1234.5,
            temperature: 298.0,
            boundary: EwaldBoundary::TinFoil,
        }
        .compute(&no_frames(), (&dipole, dt, 40))
        .unwrap();
        // zero-lag variance == acf[0] and is non-zero (unnormalized).
        assert!((res.zero_lag_variance - res.acf[0]).abs() < 1e-15);
        assert!(res.zero_lag_variance > 0.0);
        assert_eq!(res.volume, 1234.5);
        assert_eq!(res.temperature, 298.0);
        assert_eq!(res.boundary, EwaldBoundary::TinFoil);
    }

    // ── Debye fit ──

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
