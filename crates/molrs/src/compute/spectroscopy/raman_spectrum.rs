//! [`RamanSpectrum`] — window + FFT + cross-section / Bose prefactors of the
//! isotropic and anisotropic Raman correlations.

use molrs::core::constants::{SECOND_RADIATION_CONSTANT, SPEED_OF_LIGHT};
use molrs::core::unit_factors::M_PER_S_TO_CM_PER_FS;
use molrs::op::F;
use ndarray::{Array1, ArrayView1};

use crate::compute::{Compute, ComputeError, Spectrum, SpectrumInput, TimeCorrelation, Window};

/// Parallel polarization: `I_∥ = I_iso + (4/45)·I_aniso`.
const PARALLEL_ANISO_COEFF: F = 4.0 / 45.0;
/// Perpendicular polarization denominator: `I_⊥ = I_aniso / 15`.
const PERPENDICULAR_DENOM: F = 15.0;
/// Largest exponent magnitude for which `exp(−x)` is still distinguishable
/// from zero; beyond it the Bose factor is exactly its limit 1.
const MAX_EXP_ARG: F = 700.0;

/// Raman spectrum with its isotropic / anisotropic decomposition.
#[derive(Debug, Clone)]
#[cfg_attr(feature = "serde", derive(serde::Serialize))]
pub struct RamanSpectrumResult {
    /// Frequency grid in cm⁻¹, `n_fft / 2 + 1` points.
    pub frequencies_cm1: Array1<F>,
    /// Isotropic (trace) contribution, arbitrary units.
    pub isotropic: Array1<F>,
    /// Anisotropic (deviatoric) contribution, arbitrary units.
    pub anisotropic: Array1<F>,
    /// Parallel polarization `I_∥ = I_iso + (4/45)·I_aniso`; set when the
    /// spectrum is [averaged](RamanSpectrum::with_averaged).
    pub parallel: Option<Array1<F>>,
    /// Perpendicular polarization `I_⊥ = I_aniso / 15`; set when the spectrum
    /// is [averaged](RamanSpectrum::with_averaged).
    pub perpendicular: Option<Array1<F>>,
    /// The largest lag of the input correlations.
    pub resolution: usize,
}

/// Raman spectrum of the isotropic and anisotropic polarizability
/// correlations.
///
/// Each correlation is turned into a spectrum by the one [`Spectrum`]
/// (window, zero padding, one-sided FFT), scaled by `1 / n_fft`, on the
/// wavenumber grid `ν̃ = ω / (2π c)` (lag times in **fs**, wavenumbers in
/// **cm⁻¹**). Two optional physical corrections multiply every bin `ν̃ > 0`:
///
/// - the scattering cross-section `(ν̃₀ − ν̃)⁴ / ν̃` for an incident laser at
///   `ν̃₀` ([`with_incident_frequency`](Self::with_incident_frequency)), and
/// - the Bose–Einstein factor `1 / (1 − exp(−h c ν̃ / k_B T))` at the state's
///   temperature, given to [`compute`](Compute::compute).
///
/// The DC bin is left uncorrected (both factors diverge there).
///
/// The correlations are the [`Correlator`](crate::compute::Correlator)
/// `Product` correlations of the [`PolarizabilityParts`](super::PolarizabilityParts)
/// of the polarizability derivative. One transform serves the whole family —
/// only the correlations differ:
///
/// | Spectrum | isotropic / anisotropic input |
/// |---|---|
/// | Raman | autocorrelations of the parts of `α̇` |
/// | Resonance Raman | autocorrelations of the parts of a resonant `α̇(ω_L)` |
/// | Raman optical activity | cross-correlations of the parts of `α̇` against those of `Ġ′` (signed, flips between enantiomers) |
///
/// Reference: Dickey & Paskin, *Phys. Rev.* **188**, 1407 (1969).
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct RamanSpectrum {
    spectrum: Spectrum,
    incident_frequency_cm1: Option<F>,
    averaged: bool,
}

/// The correlations a [`RamanSpectrum`] transforms.
#[derive(Debug, Clone, Copy)]
pub struct RamanSpectrumInput<'a> {
    /// Isotropic correlation, one channel, lag times in fs.
    pub isotropic: &'a TimeCorrelation,
    /// Anisotropic correlation on the same lag grid, one channel.
    pub anisotropic: &'a TimeCorrelation,
    /// Temperature of the state in K for the Bose factor; `None` applies no
    /// Bose factor.
    pub temperature: Option<F>,
}

impl Default for RamanSpectrum {
    /// The default [`Window`] taper ([`Window::CosineSq`]),
    /// [`Spectrum::DEFAULT_PAD`] zero padding, no cross-section, not averaged.
    fn default() -> Self {
        Self {
            spectrum: Spectrum::new(Window::default(), Spectrum::DEFAULT_PAD)
                .expect("the default pad is valid"),
            incident_frequency_cm1: None,
            averaged: false,
        }
    }
}

impl RamanSpectrum {
    /// The default transform (see [`Default`]); tune it with the setters.
    pub fn new() -> Self {
        Self::default()
    }

    /// Taper the correlations with `window`.
    pub fn with_window(mut self, window: Window) -> Self {
        self.spectrum = self.spectrum.with_window(window);
        self
    }

    /// Zero-pad to `pad` times the correlation length (rounded up to a power
    /// of two).
    ///
    /// # Errors
    /// [`ComputeError::OutOfRange`] for `pad == 0`.
    pub fn with_pad(mut self, pad: usize) -> Result<Self, ComputeError> {
        self.spectrum = Spectrum::new(self.spectrum.window(), pad)?;
        Ok(self)
    }

    /// Apply the scattering cross-section `(ν̃₀ − ν̃)⁴ / ν̃` of a laser at
    /// `nu0_cm1` (cm⁻¹).
    ///
    /// # Errors
    /// [`ComputeError::OutOfRange`] unless `nu0_cm1` is finite and positive.
    pub fn with_incident_frequency(mut self, nu0_cm1: F) -> Result<Self, ComputeError> {
        if !(nu0_cm1.is_finite() && nu0_cm1 > 0.0) {
            return Err(ComputeError::OutOfRange {
                field: "incident_frequency_cm1",
                value: nu0_cm1.to_string(),
            });
        }
        self.incident_frequency_cm1 = Some(nu0_cm1);
        Ok(self)
    }

    /// Also emit the parallel / perpendicular polarized spectra.
    pub fn with_averaged(mut self, averaged: bool) -> Self {
        self.averaged = averaged;
        self
    }

    /// The window + FFT configuration.
    pub fn spectrum(&self) -> Spectrum {
        self.spectrum
    }

    /// The incident laser wavenumber, cm⁻¹, if the cross-section applies.
    pub fn incident_frequency(&self) -> Option<F> {
        self.incident_frequency_cm1
    }

    /// Whether polarized spectra are emitted.
    pub fn averaged(&self) -> bool {
        self.averaged
    }

    /// cm⁻¹ per rad·fs⁻¹: `ν̃ = ω / (2π c)` with `c` in cm/fs.
    pub(crate) fn wavenumber_per_angular() -> F {
        1.0 / (2.0 * std::f64::consts::PI * SPEED_OF_LIGHT * M_PER_S_TO_CM_PER_FS.get())
    }

    /// The Bose–Einstein factor at `nu` cm⁻¹ and a validated temperature
    /// `t_k > 0`; `1` at the DC bin.
    fn bose_factor(nu: F, t_k: F) -> F {
        if nu <= 0.0 {
            return 1.0;
        }
        let exponent = -SECOND_RADIATION_CONSTANT * nu / t_k;
        if exponent > -MAX_EXP_ARG {
            1.0 / (1.0 - exponent.exp())
        } else {
            1.0
        }
    }

    /// `(wavenumbers, Re X_k / n_fft)` of one correlation.
    fn transform(
        &self,
        acf: ArrayView1<'_, F>,
        dt: F,
    ) -> Result<(Array1<F>, Array1<F>), ComputeError> {
        let out = self.spectrum.compute(SpectrumInput { signal: acf, dt })?;
        let n_fft = out.n_fft as F;
        let to_cm1 = Self::wavenumber_per_angular();
        Ok((
            out.frequencies.mapv(|omega| omega * to_cm1),
            out.real.mapv(|re| re / n_fft),
        ))
    }
}

impl Compute for RamanSpectrum {
    type Input<'a> = RamanSpectrumInput<'a>;
    type Output = RamanSpectrumResult;

    /// # Errors
    /// * [`ComputeError::DimensionMismatch`] when a correlation does not have
    ///   exactly one channel or the two differ in length.
    /// * [`ComputeError::OutOfRange`] for a temperature that is not finite
    ///   and positive, or a non-positive lag spacing.
    /// * [`ComputeError::EmptyInput`] for fewer than two lags.
    fn compute(&self, input: RamanSpectrumInput<'_>) -> Result<RamanSpectrumResult, ComputeError> {
        let acf_iso = input.isotropic.require_single_channel()?;
        let acf_aniso = input.anisotropic.require_single_channel()?;
        if acf_aniso.len() != acf_iso.len() {
            return Err(ComputeError::DimensionMismatch {
                expected: acf_iso.len(),
                got: acf_aniso.len(),
                what: "Raman (isotropic, anisotropic) lags",
            });
        }
        let dt = input.isotropic.sample_interval()?;
        if let Some(t) = input.temperature
            && !(t.is_finite() && t > 0.0)
        {
            return Err(ComputeError::OutOfRange {
                field: "temperature",
                value: t.to_string(),
            });
        }

        let (frequencies_cm1, mut iso_int) = self.transform(acf_iso, dt)?;
        let (_, mut aniso_int) = self.transform(acf_aniso, dt)?;
        for j in 0..frequencies_cm1.len() {
            let nu = frequencies_cm1[j];
            let mut correction = 1.0;
            if let Some(nu0) = self.incident_frequency_cm1
                && nu > 0.0
            {
                correction *= (nu0 - nu).powi(4) / nu;
            }
            if let Some(t) = input.temperature {
                correction *= Self::bose_factor(nu, t);
            }
            iso_int[j] *= correction;
            aniso_int[j] *= correction;
        }

        let (parallel, perpendicular) = if self.averaged {
            (
                Some(&iso_int + &aniso_int.mapv(|a| PARALLEL_ANISO_COEFF * a)),
                Some(aniso_int.mapv(|a| a / PERPENDICULAR_DENOM)),
            )
        } else {
            (None, None)
        };
        Ok(RamanSpectrumResult {
            frequencies_cm1,
            isotropic: iso_int,
            anisotropic: aniso_int,
            parallel,
            perpendicular,
            resolution: acf_iso.len() - 1,
        })
    }
}

crate::compute::spec::compute_spec! {
    RamanSpectrum as "RamanSpectrum";
    params |s| {
        window: Choice(Window::NAMES)
            = RamanSpectrum::new().spectrum.window()
            => s.spectrum.window();
        pad: Int = RamanSpectrum::new().spectrum.pad() => s.spectrum.pad();
        incident_frequency: Float
            = RamanSpectrum::new().incident_frequency_cm1
            => s.incident_frequency_cm1;
        averaged: Bool = RamanSpectrum::new().averaged => s.averaged;
    }
    inputs {
        isotropic: Upstream("TimeCorrelation");
        anisotropic: Upstream("TimeCorrelation");
        temperature: Scalar, optional;
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::compute::fitting::curve;

    #[test]
    fn raman_rejects_mismatched_acf_lengths() {
        // ac-016: shape mismatch -> DimensionMismatch.
        let iso = curve(&[1.0, 2.0, 3.0], 0.5);
        let aniso = curve(&[1.0, 2.0], 0.5);
        let err = RamanSpectrum::default()
            .compute(RamanSpectrumInput {
                isotropic: &iso,
                anisotropic: &aniso,
                temperature: None,
            })
            .unwrap_err();
        assert!(matches!(err, ComputeError::DimensionMismatch { .. }));
    }

    /// Spec §5: a non-positive temperature used to switch the Bose factor
    /// off silently (`bose_factor` returned 1); it is now an error, and "no
    /// Bose factor" is `None`.
    #[test]
    fn a_non_positive_temperature_is_an_error() {
        let c = curve(&[1.0, 0.5, 0.25, 0.1], 0.5);
        for t in [0.0, -300.0, F::NAN] {
            assert!(matches!(
                RamanSpectrum::default().compute(RamanSpectrumInput {
                    isotropic: &c,
                    anisotropic: &c,
                    temperature: Some(t),
                }),
                Err(ComputeError::OutOfRange {
                    field: "temperature",
                    ..
                })
            ));
        }
        assert!(
            RamanSpectrum::default()
                .compute(RamanSpectrumInput {
                    isotropic: &c,
                    anisotropic: &c,
                    temperature: None,
                })
                .is_ok()
        );
    }

    /// Spec §5: the 0.0 "skip" sentinel of the incident frequency is gone —
    /// no cross-section is the default, and a non-positive laser is refused.
    #[test]
    fn the_incident_frequency_is_an_option_not_a_sentinel() {
        assert_eq!(RamanSpectrum::default().incident_frequency(), None);
        for nu0 in [0.0, -1.0, F::INFINITY] {
            assert!(matches!(
                RamanSpectrum::default().with_incident_frequency(nu0),
                Err(ComputeError::OutOfRange { .. })
            ));
        }
        assert_eq!(
            RamanSpectrum::default()
                .with_incident_frequency(10000.0)
                .unwrap()
                .incident_frequency(),
            Some(10000.0)
        );
    }

    #[test]
    fn bose_factor_has_its_limits() {
        // High frequency / low temperature: the factor tends to 1.
        assert_eq!(RamanSpectrum::bose_factor(1.0e6, 1.0), 1.0);
        assert_eq!(RamanSpectrum::bose_factor(0.0, 300.0), 1.0);
        let x = SECOND_RADIATION_CONSTANT * 500.0 / 300.0;
        assert!(
            (RamanSpectrum::bose_factor(500.0, 300.0) - 1.0 / (1.0 - (-x).exp())).abs() < 1e-15
        );
    }
}
