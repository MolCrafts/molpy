//! Vibrational-spectrum result types and shared input validation.
//!
//! The window + one-sided-FFT machinery lives in
//! [`compute::spectroscopy`](crate::compute::spectroscopy) (a windowed
//! transform is a fit), routing every window through [`Window`](crate::compute::Window). This
//! file holds the two result types and the shared input validator.

use ndarray::Array1;

use crate::compute::ComputeResult;

// ── Result types ─────────────────────────────────────────────────────────────

/// Single-spectrum result (VDOS, IR).
///
/// The three vibrational spectra (VDOS / IR / Raman) are the explicit
/// composition of a **raw-ACF compute** with a **spectral
/// [`Fit`](crate::compute::Fit) transform**,
/// keeping "what was measured" separate from "how the analyst transforms it":
///
/// | spectrum | raw compute (raw ACF) | transform |
/// |----------|-----------------------|-----------|
/// | VDOS  | [`Vacf`](crate::compute::Vacf) (velocity ACF) | [`PowerSpectrum`](super::PowerSpectrum) |
/// | IR    | [`IrFlux`](super::IrFlux) (dipole-flux ACF) | [`IrSpectrum`](super::IrSpectrum) |
/// | Raman | [`RamanTensor`](super::RamanTensor) (polarizability iso/aniso ACFs) | [`RamanSpectrum`](super::RamanSpectrum) |
///
/// Reference: Dickey & Paskin, *Phys. Rev.* **188**, 1407 (1969).
///
/// # Units
///
/// | quantity      | unit   |
/// |---------------|--------|
/// | time / dt     | fs     |
/// | frequency     | cm⁻¹   |
/// | intensity     | arb.   |
/// | temperature   | K      |
#[derive(Debug, Clone)]
pub struct SpectrumResult {
    /// Frequency grid in cm⁻¹, length `n_pad / 2 + 1`.
    pub frequencies_cm1: Array1<f64>,
    /// Spectral intensities (arbitrary units).
    pub intensities: Array1<f64>,
    /// Number of ACF lags retained before windowing.
    pub resolution: usize,
    /// Number of input frames.
    pub n_frames: usize,
}

/// Raman spectrum result with isotropic / anisotropic decomposition.
#[derive(Debug, Clone)]
pub struct RamanSpectrumResult {
    /// Frequency grid in cm⁻¹.
    pub frequencies_cm1: Array1<f64>,
    /// Isotropic (trace) contribution.
    pub isotropic: Array1<f64>,
    /// Anisotropic (deviatoric) contribution.
    pub anisotropic: Array1<f64>,
    /// Parallel polarization `I_∥ = I_iso + (4/45)·I_aniso`. Set when
    /// `averaged = true`.
    pub parallel: Option<Array1<f64>>,
    /// Perpendicular polarization `I_⊥ = I_aniso / 15`. Set when
    /// `averaged = true`.
    pub perpendicular: Option<Array1<f64>>,
    /// Number of ACF lags retained.
    pub resolution: usize,
    /// Number of input frames.
    pub n_frames: usize,
}

impl ComputeResult for SpectrumResult {}
impl ComputeResult for RamanSpectrumResult {}
