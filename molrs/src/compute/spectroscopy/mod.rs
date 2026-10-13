//! Vibrational and chiral spectroscopy: raw flux/tensor correlation
//! [`Compute`](crate::compute::Compute)s and the spectral
//! [`Fit`](crate::compute::Fit) transforms that turn them into
//! frequency-domain spectra.
//!
//! # Shared spectral primitives
//!
//! The window + one-sided-FFT machinery (`window_and_fft`, `windowed_acf`,
//! `acf_to_spectrum`, `acf_to_intensities`), the physical helper
//! `bose_factor`, and the flux primitive `central_diff_series` live here so
//! every spectral method shares one implementation. Window coefficients
//! always route through [`molrs::signal`] (never reimplemented); the pad +
//! forward-FFT core is the crate-shared `forward_fft_onesided`. The raw
//! correlators themselves are the one engine in
//! [`correlation`](crate::compute::correlation) (`autocorrelate` /
//! `cross_correlate` with [`Normalization::Raw`](crate::compute::Normalization)
//! and per-component weights).

mod dielectric_spectrum;
mod ir_flux;
mod ir_spectrum;
mod power_spectrum;
mod raman_spectrum;
mod raman_tensor;
mod resonance_raman_spectrum;
mod resonance_raman_tensor;
mod roa_cross_tensor;
mod roa_spectrum;
mod spectra;
mod vcd_cross_flux;
mod vcd_spectrum;

pub use dielectric_spectrum::{
    ConductivitySumRule, DielectricSpectrumResult, DipoleAutocorrelationSpectrum,
    DipoleRateCrossSpectrum, EinsteinHelfandSpectrum, GreenKuboSpectrum, KramersKronig,
    KramersKronigCheck, RouteAgreement, RouteAgreementCheck, SumRuleCheck,
};
pub use ir_flux::{IrFlux, IrFluxArgs, IrFluxResult};
pub use ir_spectrum::IrSpectrum;
pub use power_spectrum::PowerSpectrum;
pub use raman_spectrum::RamanSpectrum;
pub use raman_tensor::{RamanTensor, RamanTensorArgs, RamanTensorResult};
pub use resonance_raman_spectrum::ResonanceRamanSpectrum;
pub use resonance_raman_tensor::{ResonanceRamanTensor, ResonanceRamanTensorArgs};
pub use roa_cross_tensor::{RoaCrossTensor, RoaCrossTensorArgs, RoaCrossTensorResult};
pub use roa_spectrum::RoaSpectrum;
pub use spectra::{RamanSpectrumResult, SpectrumResult};
pub use vcd_cross_flux::{VcdCrossFlux, VcdCrossFluxArgs, VcdCrossFluxResult};
pub use vcd_spectrum::VcdSpectrum;

use ndarray::{Array1, Array2};
use rustfft::FftPlanner;

use crate::compute::ComputeError;
use crate::signal::forward_fft_onesided;
use molrs::core::constants::{SECOND_RADIATION_CONSTANT, SPEED_OF_LIGHT};
use molrs::core::unit_factors::M_PER_S_TO_CM_PER_FS;
use molrs::signal as sig;

/// Conversion from angular frequency (rad / fs) to wavenumber (cm⁻¹):
/// ν̃ = ω / (2π · c), with `c` in cm/fs.
pub(crate) fn angular_freq_to_cm1() -> f64 {
    1.0 / (2.0 * std::f64::consts::PI * SPEED_OF_LIGHT * M_PER_S_TO_CM_PER_FS.get())
}

/// Largest exponent such that `exp(x)` does not overflow f64.
const MAX_EXP_ARG: f64 = 700.0;

// ── Spectral window + FFT helpers ─────────────────────────────────────────────

/// Window a raw ACF, zero-pad to `(pad_factor · n).next_power_of_two()`, and
/// forward-FFT it into a `(frequencies_cm1, intensities)` spectrum.
///
/// The window coefficients come from
/// [`window_coefficients`](crate::signal::window_coefficients) and the FFT
/// from the shared [`forward_fft_onesided`] core. `WindowType::CosineSq`
/// with `pad_factor = 4` is the transform every molrs spectrum used before
/// the window became selectable.
pub(crate) fn window_and_fft(
    planner: &mut FftPlanner<f64>,
    acf: &Array1<f64>,
    dt_fs: f64,
    window: sig::WindowType,
    pad_factor: usize,
) -> Result<(Array1<f64>, Array1<f64>), ComputeError> {
    let n = acf.len();
    if pad_factor == 0 {
        return Err(ComputeError::OutOfRange {
            field: "pad_factor",
            value: "0".to_string(),
        });
    }
    let windowed = windowed_acf(acf, window);
    let n_pad = (pad_factor * n).next_power_of_two();
    Ok(acf_to_spectrum(planner, &windowed, dt_fs, n_pad))
}

/// `acf[k] · w[k]` for the `n = acf.len()` coefficients of `window`.
pub(crate) fn windowed_acf(acf: &Array1<f64>, window: sig::WindowType) -> Array1<f64> {
    let w = sig::window_coefficients(window, acf.len());
    acf.iter().zip(&w).map(|(a, w)| a * w).collect()
}

/// Convert a windowed one-sided ACF to a `(frequencies_cm1, intensities_raw)`
/// spectrum. The caller applies any physical prefactors (cross-section + Bose
/// for Raman). Intensities use the spectra-flavoured `1/n_pad` scaling.
pub(crate) fn acf_to_spectrum(
    planner: &mut FftPlanner<f64>,
    acf: &Array1<f64>,
    dt_fs: f64,
    n_pad: usize,
) -> (Array1<f64>, Array1<f64>) {
    let freqs_rad = sig::frequency_grid(n_pad, dt_fs);
    let intensities = acf_to_intensities(planner, acf, n_pad);
    let n_freq = intensities.len();
    let mut frequencies_cm1 = Array1::zeros(n_freq);
    for j in 0..n_freq {
        frequencies_cm1[j] = freqs_rad[j] * angular_freq_to_cm1();
    }
    (frequencies_cm1, intensities)
}

/// FFT a windowed ACF and return only the intensity spectrum (no frequency
/// grid). Used for the second Raman component so we don't allocate a second
/// identical frequency array. Delegates the pad+forward-FFT step to the shared
/// [`forward_fft_onesided`](crate::compute::fitting::forward_fft_onesided) core,
/// then applies the spectra-flavoured `1/n_pad` real-part scaling.
pub(crate) fn acf_to_intensities(
    planner: &mut FftPlanner<f64>,
    acf: &Array1<f64>,
    n_pad: usize,
) -> Array1<f64> {
    let acf_vec;
    let acf_slice = match acf.as_slice() {
        Some(s) => s,
        None => {
            acf_vec = acf.to_vec();
            &acf_vec
        }
    };
    let bins = forward_fft_onesided(planner, acf_slice, n_pad);
    let mut intensities = Array1::zeros(bins.len());
    for (j, b) in bins.iter().enumerate() {
        intensities[j] = b.re / n_pad as f64;
    }
    intensities
}

/// Evaluate the Bose-Einstein factor at frequency `nu` (cm⁻¹) and temperature
/// `T` (K). Returns 1.0 when `nu <= 0` or the exponent would underflow.
pub(crate) fn bose_factor(nu: f64, temperature_k: f64) -> f64 {
    if nu <= 0.0 || temperature_k <= 0.0 {
        return 1.0;
    }
    let exponent = -SECOND_RADIATION_CONSTANT * nu / temperature_k;
    if exponent > -MAX_EXP_ARG {
        1.0 / (1.0 - exponent.exp())
    } else {
        1.0
    }
}

// ── Flux primitive (IR / Raman / VCD / ROA) ──────────────────────────────────

/// Central-difference time derivative of every column of an `(n_frames, n_cols)`
/// series, dropping first and last frame → shape `(n_frames − 2, n_cols)`.
///
/// `ẋ[t] = (x[t+1] − x[t−1]) / (2·dt)` — the IR / Raman / VCD / ROA flux
/// convention. Row-major walk (time outer) for cache locality.
pub(crate) fn central_diff_series(series: &Array2<f64>, dt: f64) -> Array2<f64> {
    let n = series.shape()[0];
    let n_cols = series.shape()[1];
    debug_assert!(n >= 3 && dt > 0.0);
    let inv_2dt = 0.5 / dt;
    let mut out = Array2::<f64>::zeros((n - 2, n_cols));
    for t in 1..n - 1 {
        let o = t - 1;
        for c in 0..n_cols {
            out[[o, c]] = (series[[t + 1, c]] - series[[t - 1, c]]) * inv_2dt;
        }
    }
    out
}
