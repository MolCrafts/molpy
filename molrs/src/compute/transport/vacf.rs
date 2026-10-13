//! Raw velocity autocorrelation function — the VDOS / Green–Kubo-diffusion
//! input.

use molrs::core::FrameAccess;
use ndarray::{Array1, Array2, Axis};

use crate::compute::Compute;
use crate::compute::ComputeError;
use crate::compute::ComputeResult;
use crate::compute::lag_times;
use crate::compute::{TimeCorrelationSpec, autocorrelate};

/// Unbiased velocity autocorrelation function result.
#[derive(Debug, Clone)]
pub struct VacfResult {
    /// Lag times τ = i·dt, length `resolution + 1`. Units: `[dt]` (analysis: fs).
    pub lag_times: Array1<f64>,
    /// Unbiased velocity ACF
    /// `C(τ) = (1/n_dof) Σ_d [ 1/(n-τ) Σ_t δv_d(t)·δv_d(t+τ) ]`
    /// with per-DOF trajectory-mean removal. Same curve feeds VDOS
    /// ([`PowerSpectrum`](crate::compute::PowerSpectrum)) and
    /// Green–Kubo D via `D = (1/d) ∫ C(τ) dτ`. Units: `[v]²`.
    pub acf: Array1<f64>,
}

impl ComputeResult for VacfResult {}

/// Raw velocity ACF compute (the VDOS/Green–Kubo-diffusion input).
///
/// Lifts the per-DOF mean-subtract + FFT-ACF + DOF-average block from
/// the VDOS path (the part *before* windowing), returning only the raw ACF.
///
/// [`Vacf`] returns the **unbiased** velocity ACF (per-DOF trajectory-mean
/// removal, then time-origin average `1/(n-τ)`, then DOF average). No windowing
/// and no integrated D — the fit step is the analyst's choice of
/// [`PowerSpectrum`](crate::compute::PowerSpectrum) (VDOS) or
/// [`CumulativeTrapezoid`](crate::compute::CumulativeTrapezoid) + `1/d` (D).
#[derive(Debug, Clone, Copy, Default)]
pub struct Vacf;

/// `(velocities, dt, resolution)` argument bundle for [`Vacf`] /
/// [`GreenKuboDiffusion`](super::GreenKuboDiffusion).
pub type VacfArgs<'a> = (&'a Array2<f64>, f64, usize);

/// Per-DOF mean-subtract → FFT-ACF → DOF-average core, shared by [`Vacf`] and
/// [`GreenKuboDiffusion`](super::GreenKuboDiffusion).
///
/// [`autocorrelation`] with each degree of freedom as one single-component
/// entity, mean-subtracted: the one ACF estimator, on a reshaped view.
pub(super) fn velocity_acf(
    velocities: &Array2<f64>,
    dt: f64,
    resolution: usize,
) -> Result<VacfResult, ComputeError> {
    let n_frames = velocities.shape()[0];
    let n_dof = velocities.shape()[1];
    if n_frames < 2 || n_dof == 0 {
        return Err(ComputeError::EmptyInput);
    }
    if dt <= 0.0 {
        return Err(ComputeError::OutOfRange {
            field: "dt",
            value: dt.to_string(),
        });
    }
    let max_lag = resolution.min(n_frames - 1);

    // ⟨δv_d(0) δv_d(τ)⟩ averaged over the n_dof degrees of freedom: each DOF
    // is one entity of a single component.
    let acf = autocorrelate(
        velocities.view().insert_axis(Axis(2)),
        dt,
        &TimeCorrelationSpec::new()
            .max_lag(max_lag)
            .mean_subtract(true),
    )?
    .values;

    Ok(VacfResult {
        lag_times: lag_times(max_lag, dt),
        acf,
    })
}

impl Compute for Vacf {
    /// `(velocities (n_frames, n_dof), dt, resolution)`. The `frames` slice is
    /// unused.
    type Args<'a> = VacfArgs<'a>;
    type Output = VacfResult;

    fn compute<'a, FA: FrameAccess + Sync + 'a>(
        &self,
        _frames: &[&'a FA],
        args: Self::Args<'a>,
    ) -> Result<Self::Output, ComputeError> {
        let (velocities, dt, resolution) = args;
        velocity_acf(velocities, dt, resolution)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use molrs::core::Frame;
    use molrs::signal as sig;
    use ndarray::{Array1 as A1, Array2};
    use rand::{RngExt, SeedableRng};
    use rustfft::FftPlanner;

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
    fn vacf_equals_unbiased_dof_averaged_acf() {
        // ac-011: VACF == the per-DOF mean-subtract + FFT-ACF + DOF-average curve
        // that the PowerSpectrum (VDOS) transform consumes.
        let n = 512;
        let dt = 0.5;
        let res = 100;
        let v = rng_series(n, 9, 7);

        let max_lag = res.min(n - 1);
        let inv_n_frames = 1.0 / n as f64;
        let mut planner = FftPlanner::new();
        let mut acf_sum = A1::<f64>::zeros(max_lag + 1);
        for d in 0..9 {
            let mut col: A1<f64> = (0..n).map(|t| v[[t, d]]).collect();
            let mean: f64 = col.iter().sum::<f64>() * inv_n_frames;
            for x in col.iter_mut() {
                *x -= mean;
            }
            let acf = sig::acf_fft_with_planner(&mut planner, &col, max_lag).unwrap();
            for k in 0..=max_lag {
                acf_sum[k] += acf[k];
            }
        }
        for k in 0..=max_lag {
            acf_sum[k] *= (1.0 / 9.0) / (n - k) as f64;
        }

        let raw = Vacf.compute(&no_frames(), (&v, dt, res)).unwrap();
        assert_eq!(raw.acf.len(), acf_sum.len());
        for k in 0..raw.acf.len() {
            assert!((raw.acf[k] - acf_sum[k]).abs() < 1e-12, "k={k}");
        }
        assert_eq!(raw.acf.len(), max_lag + 1);
    }

    #[test]
    fn raw_max_lag_exceeds_length_clamps_not_panics() {
        let v = rng_series(8, 3, 1);
        let raw = Vacf.compute(&no_frames(), (&v, 1.0, 1000)).unwrap();
        assert_eq!(raw.acf.len(), 8); // clamped to n_frames - 1 + 1.
    }

    #[test]
    fn vacf_is_cartesian_acf_over_ndof() {
        let n = 64;
        let n_dof = 6;
        let dt = 0.5;
        let res = 20;
        let v = rng_series(n, n_dof, 99);
        let raw = velocity_acf(&v, dt, res).unwrap();
        let cart = autocorrelate(
            v.view().insert_axis(Axis(1)),
            dt,
            &TimeCorrelationSpec::new().max_lag(res).mean_subtract(true),
        )
        .unwrap()
        .values;
        for k in 0..raw.acf.len() {
            // Fused vs sequential DOF scale may differ by 1 ULP; allow tiny tol.
            assert!((raw.acf[k] - cart[k] / n_dof as f64).abs() < 1e-12, "k={k}");
        }
    }
}
