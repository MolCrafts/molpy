//! VCD electric×magnetic dipole cross-correlation raw compute — the
//! VCD-spectrum raw input.

use molrs::core::FrameAccess;
use ndarray::{Array2, Axis};

use super::central_diff_series;
use crate::compute::Compute;
use crate::compute::ComputeError;
use crate::compute::ComputeResult;
use crate::compute::{Normalization, TimeCorrelationSpec, cross_correlate, lag_times};

/// Raw VCD cross-correlation — the VCD-spectrum raw input.
#[derive(Debug, Clone)]
pub struct VcdCrossFluxResult {
    /// Lag times τ = i·dt, length `max_lag + 1`.
    pub lag_times: ndarray::Array1<f64>,
    /// VCD cross-correlation `C(τ) = Σ_d ⟨μ̇_d(0)·ṁ_d(τ)⟩` summed over the 3
    /// Cartesian components — the (signed) ACF the
    /// [`VcdSpectrum`](super::VcdSpectrum) transform consumes.
    pub acf: ndarray::Array1<f64>,
}

impl ComputeResult for VcdCrossFluxResult {}

/// Raw VCD cross-flux compute: cross-correlation of the electric-dipole
/// derivative `μ̇` with the magnetic-dipole derivative `ṁ`.
///
/// Ported from `CROAEngine::ComputeACFPair`, `ROA_SPECTRUM_VCD` branch
/// (`src/roa.cpp`): for each Cartesian component it cross-correlates
/// `m_vaDElDip` (μ̇) with `m_vaDMagDip` (ṁ) and sums the three components.
///
/// **Deviation from the spec's literal `⟨μ̇·m⟩`:** reference implementation correlates the two
/// *derivatives* (`μ̇` × `ṁ`); the two conventions differ only by a frequency-
/// domain factor and share the same peak positions and enantiomer sign law.
#[derive(Debug, Clone, Copy, Default)]
pub struct VcdCrossFlux;

/// `(electric_dipole (n,3), magnetic_dipole (n,3), dt, resolution)` for
/// [`VcdCrossFlux`]. Both series are `(n_frames, 3)`; the central-difference
/// derivatives drop the first and last frame.
pub type VcdCrossFluxArgs<'a> = (&'a Array2<f64>, &'a Array2<f64>, f64, usize);

impl Compute for VcdCrossFlux {
    type Args<'a> = VcdCrossFluxArgs<'a>;
    type Output = VcdCrossFluxResult;

    fn compute<'a, FA: FrameAccess + Sync + 'a>(
        &self,
        _frames: &[&'a FA],
        args: Self::Args<'a>,
    ) -> Result<Self::Output, ComputeError> {
        let (electric, magnetic, dt, resolution) = args;
        let n_frames = electric.shape()[0];
        if electric.shape()[1] != 3 || magnetic.shape()[1] != 3 {
            return Err(ComputeError::DimensionMismatch {
                expected: 3,
                got: electric.shape()[1].max(magnetic.shape()[1]),
                what: "VCD (electric, magnetic) dipoles (expected (n_frames, 3))",
            });
        }
        if magnetic.shape()[0] != n_frames {
            return Err(ComputeError::DimensionMismatch {
                expected: n_frames,
                got: magnetic.shape()[0],
                what: "VCD electric/magnetic frame counts",
            });
        }
        if n_frames < 3 {
            return Err(ComputeError::EmptyInput);
        }
        if dt <= 0.0 {
            return Err(ComputeError::OutOfRange {
                field: "dt",
                value: dt.to_string(),
            });
        }
        let flux_len = n_frames - 2;
        let max_lag = resolution.min(flux_len.saturating_sub(1));
        let mu_dot = central_diff_series(electric, dt);
        let m_dot = central_diff_series(magnetic, dt);
        let acf = cross_correlate(
            mu_dot.view().insert_axis(Axis(1)),
            m_dot.view().insert_axis(Axis(1)),
            dt,
            &TimeCorrelationSpec::new()
                .max_lag(max_lag)
                .normalization(Normalization::Raw),
        )?
        .values;
        Ok(VcdCrossFluxResult {
            lag_times: lag_times(max_lag, dt),
            acf,
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::compute::Compute;
    use molrs::core::Frame;
    use molrs::signal as sig;
    use ndarray::Array1;
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

    /// Pin, bit for bit: central-difference derivatives, per-component
    /// cross-correlation, summed over the 3 components in order.
    #[test]
    fn matches_manual_xcorr_sum_path() {
        let n = 128;
        let dt = 0.5;
        let res = 40;
        let mu = rng_series(n, 3, 31);
        let m = rng_series(n, 3, 32);
        let flux_len = n - 2;
        let max_lag = res.min(flux_len - 1);
        let inv_2dt = 0.5 / dt;
        let diff = |s: &Array2<f64>, c: usize| -> Array1<f64> {
            (1..n - 1)
                .map(|t| (s[[t + 1, c]] - s[[t - 1, c]]) * inv_2dt)
                .collect()
        };
        let mut planner = rustfft::FftPlanner::new();
        let mut want = Array1::<f64>::zeros(max_lag + 1);
        for c in 0..3 {
            let xc =
                sig::xcorr_fft_with_planner(&mut planner, &diff(&mu, c), &diff(&m, c), max_lag)
                    .unwrap();
            for k in 0..=max_lag {
                want[k] += xc[k];
            }
        }
        let raw = VcdCrossFlux
            .compute(&no_frames(), (&mu, &m, dt, res))
            .unwrap();
        assert_eq!(raw.acf, want);
        assert_eq!(raw.lag_times.len(), max_lag + 1);
    }
}
