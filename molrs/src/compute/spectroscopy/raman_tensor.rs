//! Raman polarizability iso/aniso ACF raw compute — the Raman-spectrum raw
//! input.

use molrs::core::FrameAccess;
use ndarray::{Array1, Array2};

use super::{central_flux, raw_column_correlation};
use crate::compute::ComputeError;
use crate::compute::ComputeResult;
use crate::compute::LegacyCompute;
use crate::compute::Reduction;

/// Weight of each anisotropy part in the Raman ACF: the three diagonal
/// differences `xx−yy, yy−zz, zz−xx` (½) and the off-diagonals `xy, xz, yz` (3).
const ANISO_WEIGHTS: [f64; 6] = [0.5, 0.5, 0.5, 3.0, 3.0, 3.0];

/// The seven Raman parts of a Voigt tensor-derivative series `(T, 6)`:
/// column 0 the isotropic trace / 3, columns 1–6 the anisotropy parts in
/// [`ANISO_WEIGHTS`] order.
pub(super) fn raman_parts(dalpha: &Array2<f64>) -> Array2<f64> {
    Array2::from_shape_fn((dalpha.nrows(), 7), |(t, c)| match c {
        0 => (dalpha[[t, 0]] + dalpha[[t, 1]] + dalpha[[t, 2]]) / 3.0,
        1 => dalpha[[t, 0]] - dalpha[[t, 1]],
        2 => dalpha[[t, 1]] - dalpha[[t, 2]],
        3 => dalpha[[t, 2]] - dalpha[[t, 0]],
        voigt => dalpha[[t, voigt - 1]],
    })
}

/// `(iso, aniso)` from the per-part correlations `(max_lag + 1, 7)`: part 0,
/// and the [`ANISO_WEIGHTS`]-weighted sum of parts 1–6.
pub(super) fn iso_aniso(parts: &Array2<f64>) -> (Array1<f64>, Array1<f64>) {
    let mut aniso = Array1::<f64>::zeros(parts.nrows());
    for (offset, weight) in ANISO_WEIGHTS.iter().enumerate() {
        aniso.scaled_add(*weight, &parts.column(offset + 1));
    }
    (parts.column(0).to_owned(), aniso)
}

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

        // One central-diff pass over the Voigt tensor → (flux_len, 6), then
        // one unnormalised ACF per Raman part.
        let parts = raman_parts(&central_flux(polarizabilities, dt)?);
        let raw = raw_column_correlation(&parts, None, max_lag, dt, Reduction::PerComponent)?;
        let (acf_iso, acf_aniso) = iso_aniso(&raw.values);

        Ok(RamanTensorResult {
            lag_times: raw.lag_times,
            acf_iso,
            acf_aniso,
        })
    }
}
