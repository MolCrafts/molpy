//! Resonance-Raman spectrum transform.

use ndarray::Array1;

use super::raman_spectrum::RamanSpectrum;
use super::spectra::RamanSpectrumResult;
use crate::compute::ComputeError;
use crate::compute::Fit;
use crate::signal::WindowType;

/// Resonance-Raman spectrum transform of **raw resonant iso/aniso ACFs** from
/// [`ResonanceRamanTensor`](super::ResonanceRamanTensor).
///
/// Identical to [`RamanSpectrum`] (same window + FFT + cross-section/Bose
/// prefactors and cm⁻¹ grid); the only difference is upstream — the ACFs come
/// from a resonant (frequency-dependent) polarizability series.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct ResonanceRamanSpectrum {
    /// Laser / excitation frequency, cm⁻¹. `0.0` to skip the cross-section.
    pub incident_frequency_cm1: f64,
    /// Temperature, K, for the Bose factor. `0.0` to skip.
    pub temperature_k: f64,
    /// If `true`, also emit parallel / perpendicular components.
    pub averaged: bool,
    /// Taper applied to both one-sided ACFs before the FFT.
    pub window: WindowType,
    /// Zero-padding factor of the transform.
    pub pad_factor: usize,
}

impl ResonanceRamanSpectrum {
    /// Prefactors as given, cosine² window, pad factor 4.
    pub fn new(incident_frequency_cm1: f64, temperature_k: f64, averaged: bool) -> Self {
        Self {
            incident_frequency_cm1,
            temperature_k,
            averaged,
            window: WindowType::CosineSq,
            pad_factor: 4,
        }
    }

    pub fn with_window(mut self, window: WindowType) -> Self {
        self.window = window;
        self
    }
}

impl Fit for ResonanceRamanSpectrum {
    /// `(acf_iso, acf_aniso, dt_fs)` — the raw resonant iso/aniso ACFs and dt.
    type Input<'a> = (&'a Array1<f64>, &'a Array1<f64>, f64);
    type Output = RamanSpectrumResult;

    fn fit<'a>(&self, input: Self::Input<'a>) -> Result<Self::Output, ComputeError> {
        RamanSpectrum {
            incident_frequency_cm1: self.incident_frequency_cm1,
            temperature_k: self.temperature_k,
            averaged: self.averaged,
            window: self.window,
            pad_factor: self.pad_factor,
        }
        .fit(input)
    }
}
