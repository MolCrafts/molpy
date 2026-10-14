//! Vibrational and chiral spectroscopy: raw flux/tensor correlation
//! [`Compute`](crate::compute::Compute)s and the spectral
//! [`Fit`](crate::compute::Fit) transforms that turn them into
//! frequency-domain spectra.
//!
//! # Shared spectral compositions
//!
//! The spectra transform through [`Spectrum`] (`cosine_sq_spectrum`), the
//! raw correlators differentiate through [`Derivative`] (`central_flux`) and
//! correlate through [`Correlator`] (`raw_column_correlation`); `bose_factor`
//! is the one physical helper kept here.

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

use ndarray::{Array1, Array2, Axis};

use crate::compute::{
    Compute, ComputeError, Correlator, CorrelatorInput, Derivative, DerivativeInput, LagKernel,
    Normalization, Reduction, Spectrum, SpectrumInput, Stencil, TimeCorrelation, Window,
};
use molrs::core::constants::{SECOND_RADIATION_CONSTANT, SPEED_OF_LIGHT};
use molrs::core::unit_factors::M_PER_S_TO_CM_PER_FS;

/// Conversion from angular frequency (rad / fs) to wavenumber (cm⁻¹):
/// ν̃ = ω / (2π · c), with `c` in cm/fs.
pub(crate) fn angular_freq_to_cm1() -> f64 {
    1.0 / (2.0 * std::f64::consts::PI * SPEED_OF_LIGHT * M_PER_S_TO_CM_PER_FS.get())
}

/// Largest exponent such that `exp(x)` does not overflow f64.
const MAX_EXP_ARG: f64 = 700.0;

// ── Spectral transform ────────────────────────────────────────────────────────

/// The vibrational-spectrum transform of a raw correlation curve: the
/// [`Spectrum`] with a cos² window and 4× zero padding, as
/// `(frequencies_cm1, intensities)` with `intensities = Re X_k / n_fft` and
/// `dt_fs` in fs.
pub(crate) fn cosine_sq_spectrum(
    acf: &Array1<f64>,
    dt_fs: f64,
) -> Result<(Array1<f64>, Array1<f64>), ComputeError> {
    let spectrum = Spectrum::new(Window::CosineSq, 4)?.compute(SpectrumInput {
        signal: acf.view(),
        dt: dt_fs,
    })?;
    let n_fft = spectrum.n_fft as f64;
    let to_cm1 = angular_freq_to_cm1();
    Ok((
        spectrum.frequencies.mapv(|omega| omega * to_cm1),
        spectrum.real.mapv(|re| re / n_fft),
    ))
}

// ── Flux + correlator compositions (IR / Raman / VCD / ROA) ──────────────────

/// Central-difference time derivative of every column of an `(n_frames,
/// n_cols)` series — `n_frames − 2` rows, the IR / Raman / VCD / ROA flux
/// convention.
pub(crate) fn central_flux(series: &Array2<f64>, dt: f64) -> Result<Array2<f64>, ComputeError> {
    Derivative::new(Stencil::Central).compute(DerivativeInput {
        series: series.view(),
        dt,
    })
}

/// Unnormalised lagged-product sums of the columns of `a` (against `b` when
/// given): `C(k) = Σ_τ a(τ) · b(τ + k)` — summed over columns (`reduce =
/// Mean` on one item) or one channel per column (`PerComponent`).
pub(crate) fn raw_column_correlation(
    a: &Array2<f64>,
    b: Option<&Array2<f64>>,
    max_lag: usize,
    dt: f64,
    reduce: Reduction,
) -> Result<TimeCorrelation, ComputeError> {
    Correlator::new(LagKernel::Product, max_lag)
        .with_normalization(Normalization::None)
        .with_reduction(reduce)
        .compute(CorrelatorInput {
            a: a.view().insert_axis(Axis(1)),
            b: b.map(|b| b.view().insert_axis(Axis(1))),
            dt,
            boxes: None,
        })
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
