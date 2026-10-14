//! VCD spectrum transform.

use ndarray::Array1;

use super::power_spectrum::PowerSpectrum;
use super::spectra::SpectrumResult;
use crate::compute::ComputeError;
use crate::compute::Fit;
use crate::signal::WindowType;

/// VCD (vibrational circular dichroism) spectrum transform of a **raw VCD
/// cross-correlation** `⟨μ̇(0)·ṁ(τ)⟩` from [`VcdCrossFlux`](super::VcdCrossFlux).
///
/// Identical window + one-sided FFT pipeline as [`IrSpectrum`](super::IrSpectrum)
/// (both delegate to [`PowerSpectrum`](super::PowerSpectrum)), so the cm⁻¹
/// grid matches IR/Raman exactly — only the supplied cross-correlation
/// differs. The resulting intensities are **signed**: enantiomers produce
/// sign-flipped spectra and an achiral system gives ≈ 0.
///
/// `Default` is the cosine² window with a pad factor of 4.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct VcdSpectrum {
    /// Taper applied to the one-sided ACF before the FFT.
    pub window: WindowType,
    /// Zero-padding factor of the transform.
    pub pad_factor: usize,
}

impl Default for VcdSpectrum {
    fn default() -> Self {
        let PowerSpectrum { window, pad_factor } = PowerSpectrum::default();
        Self { window, pad_factor }
    }
}

impl VcdSpectrum {
    /// The default transform with another window.
    pub fn with_window(window: WindowType) -> Self {
        Self {
            window,
            ..Self::default()
        }
    }
}

impl Fit for VcdSpectrum {
    /// `(acf, dt_fs)` — the raw correlation (1D) and timestep (fs, > 0).
    type Input<'a> = (&'a Array1<f64>, f64);
    type Output = SpectrumResult;

    /// Window + FFT the raw correlation into a spectrum; see
    /// [`PowerSpectrum::fit`](super::PowerSpectrum) for the errors.
    fn fit<'a>(&self, input: Self::Input<'a>) -> Result<Self::Output, ComputeError> {
        PowerSpectrum {
            window: self.window,
            pad_factor: self.pad_factor,
        }
        .fit(input)
    }
}
