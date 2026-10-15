//! Spectroscopy (`molrs::compute::spectroscopy`): the four ε(ω) routes
//! (`EinsteinHelfandSpectrum`, `GreenKuboSpectrum`, `DipoleRateCrossSpectrum`,
//! `DipoleAutocorrelationSpectrum`), their checks (`KramersKronig`,
//! `ConductivitySumRule`, `RouteAgreement`) and the Raman family
//! (`PolarizabilityParts`, `RamanSpectrum`); their outputs.

use crate::array::{ArrayArg1, ArrayArg2};
use molrs::compute::{
    Compute, ConductivitySumRule, DielectricSpectrumInput, DielectricSpectrumResult,
    DipoleAutocorrelationSpectrum, DipoleRateCrossSpectrum, EinsteinHelfandSpectrum,
    GreenKuboSpectrum, KramersKronig, KramersKronigCheck, KramersKronigInput, Plateau,
    PolarizabilityParts, PolarizabilityPartsInput, PolarizabilityPartsResult, RamanSpectrum,
    RamanSpectrumInput, RamanSpectrumResult, RouteAgreement, RouteAgreementCheck,
    RouteAgreementInput, SumRuleCheck, SumRuleInput,
};

use pyo3::prelude::*;

use super::args;
use super::error::Result;
use super::output::py_result;
use super::spec::described;
use super::time_series::PyTimeCorrelation;

py_result! {
    /// A complex permittivity ε(ω) = ``eps_real`` − i ``eps_imag`` at
    /// ``frequencies`` (``molrs::compute::DielectricSpectrumResult``).
    PyDielectricSpectrumResult(DielectricSpectrumResult) as "DielectricSpectrumResult" {
        frequencies, eps_real, eps_imag, epsilon_inf
    }
}

py_result! {
    /// The Kramers–Kronig consistency of a spectrum
    /// (``molrs::compute::KramersKronigCheck``): ε′ ``recovered`` from ε″, its
    /// mean absolute error against ε′, the tolerance and the verdict.
    PyKramersKronigCheck(KramersKronigCheck) as "KramersKronigCheck" {
        recovered, mae, tolerance, passed
    }
}

py_result! {
    /// The conductivity sum rule (``molrs::compute::SumRuleCheck``): the
    /// integral, its expected value, their relative error and the verdict.
    PySumRuleCheck(SumRuleCheck) as "SumRuleCheck" { integral, expected, relative_error, passed }
}

py_result! {
    /// The agreement of several ε(ω) routes
    /// (``molrs::compute::RouteAgreementCheck``): the RMS difference of every
    /// pair ``(i, j, rms)``, the largest and the verdict.
    PyRouteAgreementCheck(RouteAgreementCheck) as "RouteAgreementCheck" {
        pairwise, max_rms, passed
    }
}

py_result! {
    /// The rotational invariants of a polarizability series
    /// (``molrs::compute::PolarizabilityPartsResult``): ``isotropic`` ``(T, 1)``
    /// and ``anisotropic`` ``(T, 6)``.
    PyPolarizabilityPartsResult(PolarizabilityPartsResult) as "PolarizabilityPartsResult" {
        isotropic, anisotropic
    }
}

py_result! {
    /// A Raman spectrum (``molrs::compute::RamanSpectrumResult``): the
    /// isotropic and anisotropic intensities at ``frequencies_cm1``, the
    /// polarized / depolarized combinations when averaged, and the spectral
    /// resolution.
    PyRamanSpectrumResult(RamanSpectrumResult) as "RamanSpectrumResult" {
        frequencies_cm1, isotropic, anisotropic, parallel, perpendicular, resolution
    }
}

/// One ε(ω) route class: the pyclass over the core route, its `compute` over
/// a `DielectricSpectrumInput`, its descriptor methods, and `configure`, the
/// settings every route shares (units, window, pad, ε_∞). Each class adds its
/// own `#[new]` on top of `configure`.
macro_rules! dielectric_route {
    ($(#[$meta:meta])* $py:ident($core:ident) as $name:literal) => {
        $(#[$meta])*
        #[pyclass(module = "molrs.compute", name = $name, frozen)]
        pub(crate) struct $py {
            inner: $core,
        }

        impl $py {
            /// The core route with the shared settings applied where given.
            fn configure(
                units: &Bound<'_, PyAny>,
                window: Option<&str>,
                pad: Option<usize>,
                epsilon_inf: Option<f64>,
            ) -> Result<$core> {
                let mut inner = $core::new(args::units(units)?);
                if let Some(window) = window {
                    inner = inner.with_window(window.parse()?);
                }
                if let Some(pad) = pad {
                    inner = inner.with_pad(pad)?;
                }
                if let Some(epsilon_inf) = epsilon_inf {
                    inner = inner.with_epsilon_inf(epsilon_inf);
                }
                Ok(inner)
            }
        }

        #[pymethods]
        impl $py {
            #[pyo3(signature = (*, correlation, volume, temperature, boundary))]
            fn compute(
                &self,
                correlation: PyRef<'_, PyTimeCorrelation>,
                volume: f64,
                temperature: f64,
                boundary: &str,
            ) -> Result<PyDielectricSpectrumResult> {
                let out = self.inner.compute(DielectricSpectrumInput {
                    correlation: &correlation.inner,
                    volume,
                    temperature,
                    boundary: boundary.parse()?,
                })?;
                Ok(out.into())
            }
        }

        described!($py, $core);
    };
}

dielectric_route! {
    /// ε(ω) by the Einstein–Helfand route from the fluctuation dipole ACF
    /// (``molrs::compute::EinsteinHelfandSpectrum``), e.g. ``DipoleAcf``'s
    /// curve.
    ///
    /// Parameters
    /// ----------
    /// units : str or UnitPreset
    /// window : str, optional
    ///     ``"hann"``, ``"blackman"``, ``"cosine_sq"`` or ``"none"``.
    /// pad : int, optional
    ///     Zero-padding factor of the transform.
    /// epsilon_inf : float, optional
    ///     ε_∞.
    ///
    /// ``None`` keeps the core default. ``compute(correlation=, volume=,
    /// temperature=, boundary=)`` returns a ``DielectricSpectrumResult``;
    /// ``boundary`` is ``"tin_foil"`` or ``"vacuum"``.
    PyEinsteinHelfandSpectrum(EinsteinHelfandSpectrum) as "EinsteinHelfandSpectrum"
}

dielectric_route! {
    /// ε(ω) by the Green–Kubo route from the current ACF
    /// (``molrs::compute::GreenKuboSpectrum``), e.g. ``CurrentAcf``'s curve.
    ///
    /// Parameters and ``compute`` as ``EinsteinHelfandSpectrum``.
    PyGreenKuboSpectrum(GreenKuboSpectrum) as "GreenKuboSpectrum"
}

dielectric_route! {
    /// ε(ω) by the dipole–current cross-correlation route
    /// (``molrs::compute::DipoleRateCrossSpectrum``).
    ///
    /// Parameters and ``compute`` as ``EinsteinHelfandSpectrum``.
    PyDipoleRateCrossSpectrum(DipoleRateCrossSpectrum) as "DipoleRateCrossSpectrum"
}

dielectric_route! {
    /// ε(ω) by the dipole-autocorrelation route
    /// (``molrs::compute::DipoleAutocorrelationSpectrum``).
    ///
    /// Parameters as ``EinsteinHelfandSpectrum``, plus ``plateau``: an optional
    /// ``(start, end)`` fraction window whose value is subtracted from the
    /// correlation (a conductor's plateau), and ``plateau_value``: ``"mean"``
    /// (the window's mean) or ``"end"`` (its last sample). ``compute`` as
    /// ``EinsteinHelfandSpectrum``.
    PyDipoleAutocorrelationSpectrum(DipoleAutocorrelationSpectrum)
        as "DipoleAutocorrelationSpectrum"
}

#[pymethods]
impl PyEinsteinHelfandSpectrum {
    #[new]
    #[pyo3(signature = (units, *, window = None, pad = None, epsilon_inf = None))]
    fn new(
        units: &Bound<'_, PyAny>,
        window: Option<&str>,
        pad: Option<usize>,
        epsilon_inf: Option<f64>,
    ) -> Result<Self> {
        Ok(Self {
            inner: Self::configure(units, window, pad, epsilon_inf)?,
        })
    }
}

#[pymethods]
impl PyGreenKuboSpectrum {
    #[new]
    #[pyo3(signature = (units, *, window = None, pad = None, epsilon_inf = None))]
    fn new(
        units: &Bound<'_, PyAny>,
        window: Option<&str>,
        pad: Option<usize>,
        epsilon_inf: Option<f64>,
    ) -> Result<Self> {
        Ok(Self {
            inner: Self::configure(units, window, pad, epsilon_inf)?,
        })
    }
}

#[pymethods]
impl PyDipoleRateCrossSpectrum {
    #[new]
    #[pyo3(signature = (units, *, window = None, pad = None, epsilon_inf = None))]
    fn new(
        units: &Bound<'_, PyAny>,
        window: Option<&str>,
        pad: Option<usize>,
        epsilon_inf: Option<f64>,
    ) -> Result<Self> {
        Ok(Self {
            inner: Self::configure(units, window, pad, epsilon_inf)?,
        })
    }
}

#[pymethods]
impl PyDipoleAutocorrelationSpectrum {
    #[new]
    #[pyo3(signature = (
        units,
        *,
        window = None,
        pad = None,
        epsilon_inf = None,
        plateau = None,
        plateau_value = None,
    ))]
    fn new(
        units: &Bound<'_, PyAny>,
        window: Option<&str>,
        pad: Option<usize>,
        epsilon_inf: Option<f64>,
        plateau: Option<(f64, f64)>,
        plateau_value: Option<&str>,
    ) -> Result<Self> {
        let mut inner = Self::configure(units, window, pad, epsilon_inf)?;
        if let Some((start, end)) = plateau {
            inner = inner.with_plateau(Plateau::new(start, end)?);
        }
        if let Some(plateau_value) = plateau_value {
            inner = inner.with_plateau_value(plateau_value.parse()?);
        }
        Ok(Self { inner })
    }
}

/// Kramers–Kronig consistency check of a ε(ω) spectrum
/// (``molrs::compute::KramersKronig``).
///
/// Parameters
/// ----------
/// relative_tolerance, absolute_tolerance : float, optional
///     The pass tolerance; one not given keeps the core's value.
///
/// ``compute(spectrum=)`` over a ``DielectricSpectrumResult`` returns a
/// ``KramersKronigCheck``.
#[pyclass(module = "molrs.compute", name = "KramersKronig", frozen)]
pub(crate) struct PyKramersKronig {
    inner: KramersKronig,
}

#[pymethods]
impl PyKramersKronig {
    #[new]
    #[pyo3(signature = (*, relative_tolerance = None, absolute_tolerance = None))]
    fn new(relative_tolerance: Option<f64>, absolute_tolerance: Option<f64>) -> Result<Self> {
        let mut inner = KramersKronig::new();
        if relative_tolerance.is_some() || absolute_tolerance.is_some() {
            // The core sets both together; the one not given keeps its value.
            let (relative, absolute) = inner.tolerance();
            inner = inner.with_tolerance(
                relative_tolerance.unwrap_or(relative),
                absolute_tolerance.unwrap_or(absolute),
            )?;
        }
        Ok(Self { inner })
    }

    #[pyo3(signature = (*, spectrum))]
    fn compute(
        &self,
        spectrum: PyRef<'_, PyDielectricSpectrumResult>,
    ) -> Result<PyKramersKronigCheck> {
        let out = self.inner.compute(KramersKronigInput {
            spectrum: &spectrum.inner,
        })?;
        Ok(out.into())
    }
}

described!(PyKramersKronig, KramersKronig);

/// The conductivity sum rule ``∫ σ(ω) dω`` against its static value
/// (``molrs::compute::ConductivitySumRule``).
///
/// Parameters
/// ----------
/// units : str or UnitPreset
/// tolerance : float, optional
///     The pass tolerance on the relative error; ``None`` keeps the core
///     default.
///
/// ``compute(frequencies=, conductivity=, current_sq_mean=, volume=,
/// temperature=)`` returns a ``SumRuleCheck``.
#[pyclass(module = "molrs.compute", name = "ConductivitySumRule", frozen)]
pub(crate) struct PyConductivitySumRule {
    inner: ConductivitySumRule,
}

#[pymethods]
impl PyConductivitySumRule {
    #[new]
    #[pyo3(signature = (units, *, tolerance = None))]
    fn new(units: &Bound<'_, PyAny>, tolerance: Option<f64>) -> Result<Self> {
        let mut inner = ConductivitySumRule::new(args::units(units)?);
        if let Some(tolerance) = tolerance {
            inner = inner.with_tolerance(tolerance)?;
        }
        Ok(Self { inner })
    }

    #[pyo3(signature = (*, frequencies, conductivity, current_sq_mean, volume, temperature))]
    fn compute(
        &self,
        frequencies: ArrayArg1<'_, f64>,
        conductivity: ArrayArg1<'_, f64>,
        current_sq_mean: f64,
        volume: f64,
        temperature: f64,
    ) -> Result<PySumRuleCheck> {
        let out = self.inner.compute(SumRuleInput {
            frequencies: frequencies.as_array(),
            conductivity: conductivity.as_array(),
            current_sq_mean,
            volume,
            temperature,
        })?;
        Ok(out.into())
    }
}

described!(PyConductivitySumRule, ConductivitySumRule);

/// Agreement of several ε(ω) routes on one system
/// (``molrs::compute::RouteAgreement``).
///
/// Parameters
/// ----------
/// part : str, optional
///     ``"imaginary"`` or ``"real"``.
/// tolerance : float, optional
///
/// ``None`` keeps the core default. ``compute(spectra=)`` over a list of
/// ``DielectricSpectrumResult`` returns a ``RouteAgreementCheck``.
#[pyclass(module = "molrs.compute", name = "RouteAgreement", frozen)]
pub(crate) struct PyRouteAgreement {
    inner: RouteAgreement,
}

#[pymethods]
impl PyRouteAgreement {
    #[new]
    #[pyo3(signature = (*, part = None, tolerance = None))]
    fn new(part: Option<&str>, tolerance: Option<f64>) -> Result<Self> {
        let mut inner = RouteAgreement::new();
        if let Some(part) = part {
            inner = inner.with_part(part.parse()?);
        }
        if let Some(tolerance) = tolerance {
            inner = inner.with_tolerance(tolerance)?;
        }
        Ok(Self { inner })
    }

    #[pyo3(signature = (*, spectra))]
    fn compute(
        &self,
        spectra: Vec<PyRef<'_, PyDielectricSpectrumResult>>,
    ) -> Result<PyRouteAgreementCheck> {
        let spectra: Vec<&DielectricSpectrumResult> =
            spectra.iter().map(|spectrum| &spectrum.inner).collect();
        let out = self
            .inner
            .compute(RouteAgreementInput { spectra: &spectra })?;
        Ok(out.into())
    }
}

described!(PyRouteAgreement, RouteAgreement);

/// Split a polarizability series into its Raman invariants
/// (``molrs::compute::PolarizabilityParts``).
///
/// ``compute(alpha=)`` over a ``(T, 6)`` series in Voigt order
/// ``[xx, yy, zz, xy, xz, yz]`` returns a ``PolarizabilityPartsResult``.
#[pyclass(module = "molrs.compute", name = "PolarizabilityParts", frozen)]
pub(crate) struct PyPolarizabilityParts {
    inner: PolarizabilityParts,
}

#[pymethods]
impl PyPolarizabilityParts {
    #[new]
    fn new() -> Self {
        Self {
            inner: PolarizabilityParts::new(),
        }
    }

    #[pyo3(signature = (*, alpha))]
    fn compute(&self, alpha: ArrayArg2<'_, f64>) -> Result<PyPolarizabilityPartsResult> {
        let out = self.inner.compute(PolarizabilityPartsInput {
            alpha: alpha.as_array(),
        })?;
        Ok(out.into())
    }
}

described!(PyPolarizabilityParts, PolarizabilityParts);

/// Raman spectrum from the isotropic and anisotropic polarizability
/// correlations (``molrs::compute::RamanSpectrum``).
///
/// Parameters
/// ----------
/// window : str, optional
///     ``"hann"``, ``"blackman"``, ``"cosine_sq"`` or ``"none"``.
/// pad : int, optional
///     Zero-padding factor of the transform.
/// incident_frequency : float, optional
///     The laser wavenumber (cm⁻¹) of the scattering cross-section prefactor.
/// averaged : bool, optional
///     Report the orientation-averaged polarized / depolarized intensities.
///
/// ``None`` keeps the core default. ``compute(isotropic=, anisotropic=,
/// temperature=None)`` over two single-channel ``TimeCorrelation`` curves
/// returns a ``RamanSpectrumResult``; a ``temperature`` applies the Bose
/// factor.
#[pyclass(module = "molrs.compute", name = "RamanSpectrum", frozen)]
pub(crate) struct PyRamanSpectrum {
    inner: RamanSpectrum,
}

#[pymethods]
impl PyRamanSpectrum {
    #[new]
    #[pyo3(signature = (*, window = None, pad = None, incident_frequency = None, averaged = None))]
    fn new(
        window: Option<&str>,
        pad: Option<usize>,
        incident_frequency: Option<f64>,
        averaged: Option<bool>,
    ) -> Result<Self> {
        let mut inner = RamanSpectrum::new();
        if let Some(window) = window {
            inner = inner.with_window(window.parse()?);
        }
        if let Some(pad) = pad {
            inner = inner.with_pad(pad)?;
        }
        if let Some(incident_frequency) = incident_frequency {
            inner = inner.with_incident_frequency(incident_frequency)?;
        }
        if let Some(averaged) = averaged {
            inner = inner.with_averaged(averaged);
        }
        Ok(Self { inner })
    }

    #[pyo3(signature = (*, isotropic, anisotropic, temperature = None))]
    fn compute(
        &self,
        isotropic: PyRef<'_, PyTimeCorrelation>,
        anisotropic: PyRef<'_, PyTimeCorrelation>,
        temperature: Option<f64>,
    ) -> Result<PyRamanSpectrumResult> {
        let out = self.inner.compute(RamanSpectrumInput {
            isotropic: &isotropic.inner,
            anisotropic: &anisotropic.inner,
            temperature,
        })?;
        Ok(out.into())
    }
}

described!(PyRamanSpectrum, RamanSpectrum);

pub(crate) fn register(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_class::<PyDielectricSpectrumResult>()?;
    m.add_class::<PyKramersKronigCheck>()?;
    m.add_class::<PySumRuleCheck>()?;
    m.add_class::<PyRouteAgreementCheck>()?;
    m.add_class::<PyPolarizabilityPartsResult>()?;
    m.add_class::<PyRamanSpectrumResult>()?;
    m.add_class::<PyEinsteinHelfandSpectrum>()?;
    m.add_class::<PyGreenKuboSpectrum>()?;
    m.add_class::<PyDipoleRateCrossSpectrum>()?;
    m.add_class::<PyDipoleAutocorrelationSpectrum>()?;
    m.add_class::<PyKramersKronig>()?;
    m.add_class::<PyConductivitySumRule>()?;
    m.add_class::<PyRouteAgreement>()?;
    m.add_class::<PyPolarizabilityParts>()?;
    m.add_class::<PyRamanSpectrum>()?;
    Ok(())
}
