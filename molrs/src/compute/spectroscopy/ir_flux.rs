//! IR dipole-flux ACF raw compute — the IR-spectrum raw input.

use molrs::core::FrameAccess;
use ndarray::{Array2, Axis};

use super::central_diff_series;
use crate::compute::Compute;
use crate::compute::ComputeError;
use crate::compute::ComputeResult;
use crate::compute::{Normalization, TimeCorrelationSpec, autocorrelate, lag_times};

/// Raw dipole-flux autocorrelation function — the IR-spectrum raw input.
#[derive(Debug, Clone)]
pub struct IrFluxResult {
    /// Lag times τ = i·dt, length `max_lag + 1`. Units: `[dt]`.
    pub lag_times: ndarray::Array1<f64>,
    /// Unnormalized dipole-flux ACF `C(τ) = Σ_d ⟨Ṁ_d(0)·Ṁ_d(τ)⟩`, summed over
    /// the 3 Cartesian components — the ACF the
    /// [`IrSpectrum`](super::IrSpectrum) transform consumes. Units: `[Ṁ]²`.
    pub acf: ndarray::Array1<f64>,
}

impl ComputeResult for IrFluxResult {}

/// Raw dipole-flux-ACF compute (the IR-spectrum input).
///
/// Lifts the central-difference dipole flux + FFT-ACF + component-sum block (the
/// part *before* windowing), returning only the raw ACF. The window + FFT step
/// is then the [`IrSpectrum`](super::IrSpectrum)
/// [`Fit`](crate::compute::Fit).
#[derive(Debug, Clone, Copy, Default)]
pub struct IrFlux;

/// `(dipole_moments, dt, resolution)` argument bundle for [`IrFlux`].
///
/// `dipole_moments` is `(n_frames, 3)`; the central-difference flux loses the
/// first and last frame, so the effective flux length is `n_frames − 2`.
pub type IrFluxArgs<'a> = (&'a Array2<f64>, f64, usize);

impl Compute for IrFlux {
    type Args<'a> = IrFluxArgs<'a>;
    type Output = IrFluxResult;

    fn compute<'a, FA: FrameAccess + Sync + 'a>(
        &self,
        _frames: &[&'a FA],
        args: Self::Args<'a>,
    ) -> Result<Self::Output, ComputeError> {
        let (dipole_moments, dt, resolution) = args;
        let shape = dipole_moments.shape();
        let n_frames = shape[0];
        if shape[1] != 3 {
            return Err(ComputeError::DimensionMismatch {
                expected: 3,
                got: shape[1],
                what: "dipole_moments (expected (n_frames, 3))",
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
        // Shared primitives: central-diff flux → unnormalized Σ_α ACF.
        let flux = central_diff_series(dipole_moments, dt);
        let acf = autocorrelate(
            flux.view().insert_axis(Axis(1)),
            dt,
            &TimeCorrelationSpec::new()
                .max_lag(max_lag)
                .normalization(Normalization::Raw),
        )?
        .values;
        Ok(IrFluxResult {
            lag_times: lag_times(max_lag, dt),
            acf,
        })
    }
}
