//! ROA polarizability×optical-activity cross-correlation raw compute — the
//! ROA-spectrum raw input.

use molrs::core::FrameAccess;
use ndarray::{Array1, Array2};

use super::raman_tensor::{iso_aniso, raman_parts};
use super::{central_flux, raw_column_correlation};
use crate::compute::ComputeError;
use crate::compute::ComputeResult;
use crate::compute::LegacyCompute;
use crate::compute::Reduction;

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

        // One central-diff pass each, then one unnormalised cross-correlation
        // per Raman part of α̇ against the same part of Ġ′.
        let a = raman_parts(&central_flux(el_pol, dt)?);
        let g = raman_parts(&central_flux(g_tensor, dt)?);
        let raw = raw_column_correlation(&a, Some(&g), max_lag, dt, Reduction::PerComponent)?;
        let (acf_iso, acf_aniso) = iso_aniso(&raw.values);

        Ok(RoaCrossTensorResult {
            lag_times: raw.lag_times,
            acf_iso,
            acf_aniso,
        })
    }
}
