//! ROA polarizability×optical-activity cross-correlation raw compute — the
//! ROA-spectrum raw input.

use molrs::core::FrameAccess;
use ndarray::{Array1, Array2};
use rustfft::FftPlanner;
use rustfft::num_complex::Complex64;

use super::raman_tensor::{DIAG_ANISO_WEIGHT, OFFDIAG_ANISO_WEIGHT};
use super::{central_diff_series, xcorr_accumulate_into};
use crate::compute::ComputeError;
use crate::compute::ComputeResult;
use crate::compute::LegacyCompute;
use crate::compute::lag_times;

/// Raw ROA cross-correlation iso/aniso curves — the ROA-spectrum raw input.
#[derive(Debug, Clone)]
pub struct RoaCrossTensorResult {
    /// Lag times τ = i·dt, length `max_lag + 1`.
    pub lag_times: Array1<f64>,
    /// Isotropic ROA cross-correlation of `α̇` (electric polarizability
    /// derivative) with `Ġ′` (magnetic-dipole polarizability / optical-activity
    /// tensor derivative).
    pub acf_iso: Array1<f64>,
    /// Weighted anisotropic ROA cross-correlation (same diagonal/off-diagonal
    /// weighting as the Raman anisotropy).
    pub acf_aniso: Array1<f64>,
}

impl ComputeResult for RoaCrossTensorResult {}

/// Raw ROA cross-tensor compute: cross-correlation of the electric
/// polarizability derivative `α̇` with the optical-activity tensor derivative
/// `Ġ′` (and, in the full theory, the electric-quadrupole tensor `A`).
///
/// Ported from `CROAEngine::ComputeACFPair`, `ROA_SPECTRUM_ROA` branch
/// (`src/roa.cpp`): the iso part cross-correlates the polarizability trace with
/// the (negated) `G′` trace, and the anisotropic parts mirror the Raman
/// deviatoric decomposition. Both tensors are passed in Voigt form
/// `[xx, yy, zz, xy, xz, yz]`; the same diagonal (½) / off-diagonal (3) weights
/// as [`RamanTensor`](super::RamanTensor) are used so ROA shares the Raman
/// normal-mode frequencies.
#[derive(Debug, Clone, Copy, Default)]
pub struct RoaCrossTensor;

/// `(electric_pol (n,6), g_tensor (n,6), dt, resolution)` for [`RoaCrossTensor`],
/// both in Voigt notation `[xx, yy, zz, xy, xz, yz]`.
pub type RoaCrossTensorArgs<'a> = (&'a Array2<f64>, &'a Array2<f64>, f64, usize);

impl LegacyCompute for RoaCrossTensor {
    type Args<'a> = RoaCrossTensorArgs<'a>;
    type Output = RoaCrossTensorResult;

    fn compute<'a, FA: FrameAccess + Sync + 'a>(
        &self,
        _frames: &[&'a FA],
        args: Self::Args<'a>,
    ) -> Result<Self::Output, ComputeError> {
        let (el_pol, g_tensor, dt, resolution) = args;
        let n_frames = el_pol.shape()[0];
        if el_pol.shape()[1] != 6 || g_tensor.shape()[1] != 6 {
            return Err(ComputeError::DimensionMismatch {
                expected: 6,
                got: el_pol.shape()[1].max(g_tensor.shape()[1]),
                what: "ROA (electric_pol, g_tensor) (expected (n_frames, 6) Voigt)",
            });
        }
        if g_tensor.shape()[0] != n_frames {
            return Err(ComputeError::DimensionMismatch {
                expected: n_frames,
                got: g_tensor.shape()[0],
                what: "ROA electric_pol/g_tensor frame counts",
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

        // One central-diff pass each — no 12 separate column allocations.
        let a = central_diff_series(el_pol, dt);
        let g = central_diff_series(g_tensor, dt);

        let mut planner = FftPlanner::new();
        let mut sa: Vec<Complex64> = Vec::new();
        let mut sb: Vec<Complex64> = Vec::new();

        // Isotropic: trace × trace.
        let mut a_iso = vec![0.0_f64; flux_len];
        let mut g_iso = vec![0.0_f64; flux_len];
        for t in 0..flux_len {
            a_iso[t] = (a[[t, 0]] + a[[t, 1]] + a[[t, 2]]) / 3.0;
            g_iso[t] = (g[[t, 0]] + g[[t, 1]] + g[[t, 2]]) / 3.0;
        }
        let mut acf_iso = Array1::<f64>::zeros(max_lag + 1);
        xcorr_accumulate_into(
            &mut planner,
            &a_iso,
            &g_iso,
            max_lag,
            acf_iso.as_slice_mut().unwrap(),
            1.0,
            &mut sa,
            &mut sb,
        );

        // Anisotropic: 3 diagonal diffs (½) + 3 off-diagonals (3).
        let mut acf_aniso = Array1::<f64>::zeros(max_lag + 1);
        let aniso_out = acf_aniso.as_slice_mut().unwrap();
        let mut av = vec![0.0_f64; flux_len];
        let mut gv = vec![0.0_f64; flux_len];
        let diag_pairs = [(0usize, 1usize), (1, 2), (2, 0)];
        for (i, j) in diag_pairs {
            for t in 0..flux_len {
                av[t] = a[[t, i]] - a[[t, j]];
                gv[t] = g[[t, i]] - g[[t, j]];
            }
            xcorr_accumulate_into(
                &mut planner,
                &av,
                &gv,
                max_lag,
                aniso_out,
                DIAG_ANISO_WEIGHT,
                &mut sa,
                &mut sb,
            );
        }
        for off in 3..6 {
            for t in 0..flux_len {
                av[t] = a[[t, off]];
                gv[t] = g[[t, off]];
            }
            xcorr_accumulate_into(
                &mut planner,
                &av,
                &gv,
                max_lag,
                aniso_out,
                OFFDIAG_ANISO_WEIGHT,
                &mut sa,
                &mut sb,
            );
        }

        Ok(RoaCrossTensorResult {
            lag_times: lag_times(max_lag, dt),
            acf_iso,
            acf_aniso,
        })
    }
}
