//! Raman polarizability iso/aniso ACF raw compute — the Raman-spectrum raw
//! input.

use molrs::core::FrameAccess;
use ndarray::{Array1, Array2, Array3};

use super::central_diff_series;
use crate::compute::Compute;
use crate::compute::ComputeError;
use crate::compute::ComputeResult;
use crate::compute::{Normalization, TimeCorrelationSpec, autocorrelate, lag_times};

/// Weight for diagonal anisotropy components in the Raman ACF.
pub(super) const DIAG_ANISO_WEIGHT: f64 = 0.5;
/// Weight for off-diagonal anisotropy components in the Raman ACF.
pub(super) const OFFDIAG_ANISO_WEIGHT: f64 = 3.0;

/// Per-column weights of the anisotropic series built by
/// [`raman_invariant_series`]: the three diagonal differences, then the three
/// off-diagonals.
pub(super) const ANISO_WEIGHTS: [f64; 6] = [
    DIAG_ANISO_WEIGHT,
    DIAG_ANISO_WEIGHT,
    DIAG_ANISO_WEIGHT,
    OFFDIAG_ANISO_WEIGHT,
    OFFDIAG_ANISO_WEIGHT,
    OFFDIAG_ANISO_WEIGHT,
];

/// The two rotational invariants of a `(n, 6)` Voigt tensor series, as the
/// `(n, 1, 1)` isotropic series `(xx + yy + zz)/3` and the `(n, 1, 6)`
/// anisotropic series `(xx−yy, yy−zz, zz−xx, xy, xz, yz)`, whose weighted
/// autocorrelation (weights [`ANISO_WEIGHTS`]) is the anisotropic
/// correlator.
pub(super) fn raman_invariant_series(voigt: &Array2<f64>) -> (Array3<f64>, Array3<f64>) {
    let n = voigt.shape()[0];
    let mut iso = Array3::<f64>::zeros((n, 1, 1));
    let mut aniso = Array3::<f64>::zeros((n, 1, 6));
    for t in 0..n {
        iso[[t, 0, 0]] = (voigt[[t, 0]] + voigt[[t, 1]] + voigt[[t, 2]]) / 3.0;
        aniso[[t, 0, 0]] = voigt[[t, 0]] - voigt[[t, 1]];
        aniso[[t, 0, 1]] = voigt[[t, 1]] - voigt[[t, 2]];
        aniso[[t, 0, 2]] = voigt[[t, 2]] - voigt[[t, 0]];
        aniso[[t, 0, 3]] = voigt[[t, 3]];
        aniso[[t, 0, 4]] = voigt[[t, 4]];
        aniso[[t, 0, 5]] = voigt[[t, 5]];
    }
    (iso, aniso)
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

impl Compute for RamanTensor {
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
        let (iso, aniso) = raman_invariant_series(&dalpha);
        let raw = TimeCorrelationSpec::new()
            .max_lag(max_lag)
            .normalization(Normalization::Raw);
        let acf_iso = autocorrelate(iso.view(), dt, &raw)?.values;
        let acf_aniso =
            autocorrelate(aniso.view(), dt, &raw.component_weights(&ANISO_WEIGHTS))?.values;

        Ok(RamanTensorResult {
            lag_times: lag_times(max_lag, dt),
            acf_iso,
            acf_aniso,
        })
    }
}
