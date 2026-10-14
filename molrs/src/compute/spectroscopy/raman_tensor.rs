//! Raman polarizability iso/aniso ACF raw compute — the Raman-spectrum raw
//! input.

use molrs::core::FrameAccess;
use ndarray::{Array1, Array2};
use rustfft::FftPlanner;
use rustfft::num_complex::Complex64;

use super::{acf_accumulate_into, central_diff_series};
use crate::compute::ComputeError;
use crate::compute::ComputeResult;
use crate::compute::LegacyCompute;
use crate::compute::lag_times;

/// Weight for diagonal anisotropy components in the Raman ACF.
pub(super) const DIAG_ANISO_WEIGHT: f64 = 0.5;
/// Weight for off-diagonal anisotropy components in the Raman ACF.
pub(super) const OFFDIAG_ANISO_WEIGHT: f64 = 3.0;

/// Raw isotropic + (weighted) anisotropic polarizability-derivative ACFs — the
/// Raman-spectrum raw input.
#[derive(Debug, Clone)]
pub struct RamanTensorResult {
    /// Lag times τ = i·dt, length `max_lag + 1`. Units: `[dt]`.
    pub lag_times: Array1<f64>,
    /// Isotropic (trace) polarizability-derivative ACF — the `acf_iso` the
    /// [`RamanSpectrum`](super::RamanSpectrum) transform consumes.
    pub acf_iso: Array1<f64>,
    /// Weighted anisotropic (deviatoric) ACF — the `acf_aniso` the
    /// [`RamanSpectrum`](super::RamanSpectrum) transform consumes.
    pub acf_aniso: Array1<f64>,
}

impl ComputeResult for RamanTensorResult {}

/// Raw Raman-tensor-ACF compute (the Raman-spectrum input).
///
/// Lifts the central-difference polarizability derivative + iso/aniso
/// decomposition + FFT-ACF block (the part *before* windowing +
/// cross-section/Bose prefactors), returning only the raw iso/aniso ACFs. The
/// window + FFT + prefactor step is then the
/// [`RamanSpectrum`](super::RamanSpectrum) [`Fit`](crate::compute::Fit).
#[derive(Debug, Clone, Copy, Default)]
pub struct RamanTensor;

/// `(polarizabilities, dt, resolution)` argument bundle for [`RamanTensor`].
///
/// `polarizabilities` is `(n_frames, 6)` in Voigt notation
/// `[α_xx, α_yy, α_zz, α_xy, α_xz, α_yz]`; the central-difference derivative
/// loses the first and last frame.
pub type RamanTensorArgs<'a> = (&'a Array2<f64>, f64, usize);

impl LegacyCompute for RamanTensor {
    type Args<'a> = RamanTensorArgs<'a>;
    type Output = RamanTensorResult;

    fn compute<'a, FA: FrameAccess + Sync + 'a>(
        &self,
        _frames: &[&'a FA],
        args: Self::Args<'a>,
    ) -> Result<Self::Output, ComputeError> {
        let (polarizabilities, dt, resolution) = args;
        let shape = polarizabilities.shape();
        let n_frames = shape[0];
        if shape[1] != 6 {
            return Err(ComputeError::DimensionMismatch {
                expected: 6,
                got: shape[1],
                what: "polarizabilities (expected (n_frames, 6) Voigt)",
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

        // One central-diff pass over the Voigt tensor → (flux_len, 6).
        let dalpha = central_diff_series(polarizabilities, dt);

        // Isotropic: (α̇_xx + α̇_yy + α̇_zz) / 3.
        let mut iso = vec![0.0_f64; flux_len];
        for t in 0..flux_len {
            iso[t] = (dalpha[[t, 0]] + dalpha[[t, 1]] + dalpha[[t, 2]]) / 3.0;
        }

        let mut planner = FftPlanner::new();
        let mut scratch: Vec<Complex64> = Vec::new();
        let mut acf_iso = Array1::<f64>::zeros(max_lag + 1);
        acf_accumulate_into(
            &mut planner,
            &iso,
            max_lag,
            acf_iso.as_slice_mut().unwrap(),
            1.0,
            &mut scratch,
        );

        // Anisotropic: 3 diagonal diffs (weight ½) + 3 off-diagonals (weight 3).
        let mut acf_aniso = Array1::<f64>::zeros(max_lag + 1);
        let aniso_out = acf_aniso.as_slice_mut().unwrap();
        let mut col = vec![0.0_f64; flux_len];

        // Diagonal differences: xx−yy, yy−zz, zz−xx.
        let diag_pairs = [(0usize, 1usize), (1, 2), (2, 0)];
        for (i, j) in diag_pairs {
            for t in 0..flux_len {
                col[t] = dalpha[[t, i]] - dalpha[[t, j]];
            }
            acf_accumulate_into(
                &mut planner,
                &col,
                max_lag,
                aniso_out,
                DIAG_ANISO_WEIGHT,
                &mut scratch,
            );
        }
        // Off-diagonals: xy, xz, yz (Voigt indices 3,4,5).
        for off in 3..6 {
            for t in 0..flux_len {
                col[t] = dalpha[[t, off]];
            }
            acf_accumulate_into(
                &mut planner,
                &col,
                max_lag,
                aniso_out,
                OFFDIAG_ANISO_WEIGHT,
                &mut scratch,
            );
        }

        Ok(RamanTensorResult {
            lag_times: lag_times(max_lag, dt),
            acf_iso,
            acf_aniso,
        })
    }
}
