//! Spectra from raw correlation functions and polarizabilities
//! (`molrs::compute::spectroscopy`), and the consistency checks that judge them
//! (Kramers–Kronig, the conductivity sum rule, route agreement).

use super::correlation::{PySpectrum, parse_window};
use crate::error::py_value_err;
use molrs::compute::{
    Check, ConductivitySumRule, DipoleAutocorrelationSpectrum, DipoleRateCrossSpectrum,
    EinsteinHelfandSpectrum, Fit, GreenKuboSpectrum, IrSpectrum, KramersKronig, PowerSpectrum,
    RamanSpectrum, ResonanceRamanSpectrum, RoaSpectrum, RouteAgreement, VcdSpectrum,
};
use molrs::signal::WindowType;
use ndarray::Array1;
use numpy::{IntoPyArray, PyReadonlyArray1};
use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;
use pyo3::types::{PyAny, PyAnyMethods, PyDict, PyDictMethods};

/// The window + pad settings every one-sided spectrum transform takes.
#[derive(Clone, Copy)]
struct Taper {
    window: WindowType,
    pad_factor: usize,
}

impl Taper {
    fn parse(window: &str, pad_factor: usize) -> PyResult<Self> {
        if pad_factor == 0 {
            return Err(PyValueError::new_err("pad_factor must be >= 1"));
        }
        Ok(Self {
            window: parse_window(window)?,
            pad_factor,
        })
    }

    fn power(self) -> PowerSpectrum {
        PowerSpectrum {
            window: self.window,
            pad_factor: self.pad_factor,
        }
    }
}

fn one_sided(taper: Taper, acf: PyReadonlyArray1<'_, f64>, dt_fs: f64) -> PyResult<PySpectrum> {
    let a = acf.as_array().to_owned();
    let r = taper.power().fit((&a, dt_fs)).map_err(py_value_err)?;
    Ok(PySpectrum::from_result(r, taper.window, dt_fs))
}

// ── PowerSpectrum (VDOS) ──────────────────────────────────────────────────────

/// Velocity power spectrum (VDOS) transform of a raw velocity ACF: window
/// the one-sided curve, zero-pad to ``pad_factor`` times its length (rounded
/// up to a power of two) and forward-FFT it. ``window`` is ``"cosine_sq"``
/// (default; keeps ``C(0)``), ``"hann"``, ``"blackman"`` or ``"none"``.
/// :meth:`TimeCorrelation.spectrum` is this transform on a curve.
#[pyclass(module = "molrs.compute", name = "PowerSpectrum")]
pub struct PyPowerSpectrum {
    taper: Taper,
}

#[pymethods]
impl PyPowerSpectrum {
    #[new]
    #[pyo3(signature = (window="cosine_sq", pad_factor=4))]
    fn new(window: &str, pad_factor: usize) -> PyResult<Self> {
        Ok(Self {
            taper: Taper::parse(window, pad_factor)?,
        })
    }

    /// Window + FFT the raw ACF (timestep ``dt_fs``) into a :class:`Spectrum`.
    fn fit(&self, acf: PyReadonlyArray1<'_, f64>, dt_fs: f64) -> PyResult<PySpectrum> {
        one_sided(self.taper, acf, dt_fs)
    }

    #[getter]
    fn window(&self) -> &'static str {
        self.taper.window.name()
    }

    #[getter]
    fn pad_factor(&self) -> usize {
        self.taper.pad_factor
    }
}

// ── IrSpectrum ───────────────────────────────────────────────────────────────

/// Infrared absorption spectrum transform of a raw dipole-flux ACF: the
/// :class:`PowerSpectrum` transform; only the supplied ACF differs.
#[pyclass(module = "molrs.compute", name = "IrSpectrum")]
pub struct PyIrSpectrum {
    taper: Taper,
}

#[pymethods]
impl PyIrSpectrum {
    #[new]
    #[pyo3(signature = (window="cosine_sq", pad_factor=4))]
    fn new(window: &str, pad_factor: usize) -> PyResult<Self> {
        Ok(Self {
            taper: Taper::parse(window, pad_factor)?,
        })
    }

    /// Window + FFT the raw flux ACF (timestep ``dt_fs``) into a :class:`Spectrum`.
    fn fit(&self, acf: PyReadonlyArray1<'_, f64>, dt_fs: f64) -> PyResult<PySpectrum> {
        let a = acf.as_array().to_owned();
        let r = IrSpectrum {
            window: self.taper.window,
            pad_factor: self.taper.pad_factor,
        }
        .fit((&a, dt_fs))
        .map_err(py_value_err)?;
        Ok(PySpectrum::from_result(r, self.taper.window, dt_fs))
    }

    #[getter]
    fn window(&self) -> &'static str {
        self.taper.window.name()
    }

    #[getter]
    fn pad_factor(&self) -> usize {
        self.taper.pad_factor
    }
}

// ── RamanSpectrum ────────────────────────────────────────────────────────────

/// Raman spectrum transform of **raw isotropic + anisotropic ACFs**
/// (one window per ACF — ``"cosine_sq"`` by default — FFT both, then the
/// cross-section + Bose prefactors).
#[pyclass(module = "molrs.compute", name = "RamanSpectrum")]
pub struct PyRamanSpectrum {
    inner: RamanSpectrum,
}

#[pymethods]
impl PyRamanSpectrum {
    /// ``RamanSpectrum(incident_frequency_cm1=0.0, temperature_k=0.0,
    /// averaged=False, window="cosine_sq", pad_factor=4)`` — set
    /// ``incident_frequency_cm1``/``temperature_k`` to ``0.0`` to skip the
    /// cross-section / Bose correction.
    #[new]
    #[pyo3(signature = (incident_frequency_cm1=0.0, temperature_k=0.0, averaged=false, window="cosine_sq", pad_factor=4))]
    fn new(
        incident_frequency_cm1: f64,
        temperature_k: f64,
        averaged: bool,
        window: &str,
        pad_factor: usize,
    ) -> PyResult<Self> {
        let taper = Taper::parse(window, pad_factor)?;
        Ok(Self {
            inner: RamanSpectrum {
                incident_frequency_cm1,
                temperature_k,
                averaged,
                window: taper.window,
                pad_factor: taper.pad_factor,
            },
        })
    }

    /// Window + FFT + prefactors on the raw iso/aniso ACFs (timestep ``dt_fs``).
    ///
    /// Returns ``{"frequencies_cm1", "isotropic", "anisotropic", "parallel",
    /// "perpendicular", "resolution", "n_frames"}`` (``parallel`` /
    /// ``perpendicular`` are ``None`` unless ``averaged=True``).
    fn fit<'py>(
        &self,
        py: Python<'py>,
        acf_iso: PyReadonlyArray1<'py, f64>,
        acf_aniso: PyReadonlyArray1<'py, f64>,
        dt_fs: f64,
    ) -> PyResult<Bound<'py, PyDict>> {
        let iso = acf_iso.as_array().to_owned();
        let aniso = acf_aniso.as_array().to_owned();
        let r = self
            .inner
            .fit((&iso, &aniso, dt_fs))
            .map_err(py_value_err)?;
        let d = PyDict::new(py);
        d.set_item("frequencies_cm1", r.frequencies_cm1.into_pyarray(py))?;
        d.set_item("isotropic", r.isotropic.into_pyarray(py))?;
        d.set_item("anisotropic", r.anisotropic.into_pyarray(py))?;
        match r.parallel {
            Some(p) => d.set_item("parallel", p.into_pyarray(py))?,
            None => d.set_item("parallel", py.None())?,
        }
        match r.perpendicular {
            Some(p) => d.set_item("perpendicular", p.into_pyarray(py))?,
            None => d.set_item("perpendicular", py.None())?,
        }
        d.set_item("resolution", r.resolution)?;
        d.set_item("n_frames", r.n_frames)?;
        Ok(d)
    }
}

// ── EinsteinHelfandSpectrum (dielectric ε(ω), dipole route) ───────────────────

fn dielectric_dict<'py>(
    py: Python<'py>,
    frequencies: Array1<f64>,
    eps_real: Array1<f64>,
    eps_imag: Array1<f64>,
) -> PyResult<Bound<'py, PyDict>> {
    let d = PyDict::new(py);
    d.set_item("frequencies", frequencies.into_pyarray(py))?;
    d.set_item("eps_real", eps_real.into_pyarray(py))?;
    d.set_item("eps_imag", eps_imag.into_pyarray(py))?;
    Ok(d)
}

/// Einstein–Helfand ε(ω) transform of a **raw fluctuation dipole ACF** (the
/// [`DebyeRelaxation`](PyDebyeRelaxation) ACF): one-sided cos² taper +
/// derivative-FT + the `4π·KAPPA/(3·V·k_B·T)` prefactor.
#[pyclass(module = "molrs.compute", name = "EinsteinHelfandSpectrum")]

pub struct PyEinsteinHelfandSpectrum {
    inner: EinsteinHelfandSpectrum,
}

#[pymethods]
impl PyEinsteinHelfandSpectrum {
    /// ``EinsteinHelfandSpectrum(dt, volume, temperature, epsilon_inf,
    /// zero_lag_variance)`` — ``zero_lag_variance`` is ⟨|δM|²⟩ (the
    /// ``DebyeRelaxation`` ``zero_lag_variance``), which pins the exact DC bin.
    #[new]
    fn new(
        dt: f64,
        volume: f64,
        temperature: f64,
        epsilon_inf: f64,
        zero_lag_variance: f64,
    ) -> Self {
        Self {
            inner: EinsteinHelfandSpectrum {
                dt,
                volume,
                temperature,
                epsilon_inf,
                zero_lag_variance,
            },
        }
    }

    /// Transform the raw fluctuation dipole ``acf`` into ε(ω).
    ///
    /// Returns ``{"frequencies", "eps_real", "eps_imag"}`` (float64 arrays).
    fn fit<'py>(
        &self,
        py: Python<'py>,
        acf: PyReadonlyArray1<'py, f64>,
    ) -> PyResult<Bound<'py, PyDict>> {
        let a = acf.as_array().to_owned();
        let r = self.inner.fit(&a).map_err(py_value_err)?;
        dielectric_dict(py, r.frequencies, r.eps_real, r.eps_imag)
    }
}

// ── GreenKuboSpectrum (dielectric ε(ω), current route) ────────────────────────

/// Green–Kubo ε(ω) transform of a **raw current ACF** (the
/// [`GreenKuboConductivity`](PyGreenKuboConductivity) ACF over the post-NaN
/// series): window + FFT → σ(ω) → ε(ω).
#[pyclass(module = "molrs.compute", name = "GreenKuboSpectrum")]
pub struct PyGreenKuboSpectrum {
    inner: GreenKuboSpectrum,
}

#[pymethods]
impl PyGreenKuboSpectrum {
    /// ``GreenKuboSpectrum(dt, volume, temperature, epsilon_inf,
    /// window_type="hann")`` — ``window_type`` is ``"cosine_sq"``, ``"hann"``,
    /// or ``"blackman"``.
    #[new]
    #[pyo3(signature = (dt, volume, temperature, epsilon_inf, window_type="hann"))]
    fn new(dt: f64, volume: f64, temperature: f64, epsilon_inf: f64, window_type: &str) -> Self {
        Self {
            inner: GreenKuboSpectrum {
                dt,
                volume,
                temperature,
                epsilon_inf,
                window_type: window_type.to_string(),
            },
        }
    }

    /// Transform the raw current ``acf`` into ε(ω).
    ///
    /// Returns ``{"frequencies", "eps_real", "eps_imag"}`` (float64 arrays).
    fn fit<'py>(
        &self,
        py: Python<'py>,
        acf: PyReadonlyArray1<'py, f64>,
    ) -> PyResult<Bound<'py, PyDict>> {
        let a = acf.as_array().to_owned();
        let r = self.inner.fit(&a).map_err(py_value_err)?;
        dielectric_dict(py, r.frequencies, r.eps_real, r.eps_imag)
    }
}

// ── DipoleRateCrossSpectrum (⟨Ṁ(0)·M(t)⟩ → ε(ω)) ─────────────────────────────

/// ε(ω) from the dipole-rate × dipole cross-correlation
/// `C(t) = Σ_α ⟨δṀ_α(0) δM_α(t)⟩` (piecewise-linear FT + conducting-BC prefactor).
#[pyclass(module = "molrs.compute", name = "DipoleRateCrossSpectrum")]

pub struct PyDipoleRateCrossSpectrum {
    inner: DipoleRateCrossSpectrum,
}

#[pymethods]
impl PyDipoleRateCrossSpectrum {
    /// ``DipoleRateCrossSpectrum(dt, volume, temperature, epsilon_inf,
    /// window_type="none")``.
    #[new]
    #[pyo3(signature = (dt, volume, temperature, epsilon_inf, window_type="none"))]
    fn new(dt: f64, volume: f64, temperature: f64, epsilon_inf: f64, window_type: &str) -> Self {
        Self {
            inner: DipoleRateCrossSpectrum {
                dt,
                volume,
                temperature,
                epsilon_inf,
                window_type: window_type.to_string(),
            },
        }
    }

    /// Transform the raw cross correlator into ε(ω).
    ///
    /// Returns ``{"frequencies", "eps_real", "eps_imag"}``.
    fn fit<'py>(
        &self,
        py: Python<'py>,
        cross: PyReadonlyArray1<'py, f64>,
    ) -> PyResult<Bound<'py, PyDict>> {
        let a = cross.as_array().to_owned();
        let r = self.inner.fit(&a).map_err(py_value_err)?;
        dielectric_dict(py, r.frequencies, r.eps_real, r.eps_imag)
    }
}

// ── DipoleAutocorrelationSpectrum (PACF → ε(ω) via C(0)−iωĈ) ─────────────────

/// ε(ω) from the fluctuation dipole ACF via `χ = A [C(0) − iω Ĉ(ω)]`.
///
/// Optional conductive-plateau handling: when ``subtract_plateau`` is
/// ``None`` (default), subtract ``C(∞)`` automatically if
/// ``|C(∞)/C(0)| > 0.1``.
#[pyclass(module = "molrs.compute", name = "DipoleAutocorrelationSpectrum")]

pub struct PyDipoleAutocorrelationSpectrum {
    inner: DipoleAutocorrelationSpectrum,
}

#[pymethods]
impl PyDipoleAutocorrelationSpectrum {
    /// ``DipoleAutocorrelationSpectrum(dt, volume, temperature, epsilon_inf,
    /// window_type="none", subtract_plateau=None)``.
    #[new]
    #[pyo3(signature = (dt, volume, temperature, epsilon_inf, window_type="none", subtract_plateau=None))]
    fn new(
        dt: f64,
        volume: f64,
        temperature: f64,
        epsilon_inf: f64,
        window_type: &str,
        subtract_plateau: Option<bool>,
    ) -> Self {
        Self {
            inner: DipoleAutocorrelationSpectrum {
                dt,
                volume,
                temperature,
                epsilon_inf,
                window_type: window_type.to_string(),
                subtract_plateau,
            },
        }
    }

    /// Transform the raw dipole PACF into ε(ω).
    ///
    /// Returns ``{"frequencies", "eps_real", "eps_imag"}``.
    fn fit<'py>(
        &self,
        py: Python<'py>,
        acf: PyReadonlyArray1<'py, f64>,
    ) -> PyResult<Bound<'py, PyDict>> {
        let a = acf.as_array().to_owned();
        let r = self.inner.fit(&a).map_err(py_value_err)?;
        dielectric_dict(py, r.frequencies, r.eps_real, r.eps_imag)
    }
}

// ═══════════════════════════════════════════════════════════════════════════
// Registration
// ═══════════════════════════════════════════════════════════════════════════

// ── Chiral / resonance spectra (analysis parity) ───────────────────────────────

/// Build the Raman-family result dict (shared by ROA / resonance-Raman).
fn raman_dict<'py>(
    py: Python<'py>,
    r: molrs::compute::RamanSpectrumResult,
) -> PyResult<Bound<'py, PyDict>> {
    let d = PyDict::new(py);
    d.set_item("frequencies_cm1", r.frequencies_cm1.into_pyarray(py))?;
    d.set_item("isotropic", r.isotropic.into_pyarray(py))?;
    d.set_item("anisotropic", r.anisotropic.into_pyarray(py))?;
    match r.parallel {
        Some(p) => d.set_item("parallel", p.into_pyarray(py))?,
        None => d.set_item("parallel", py.None())?,
    }
    match r.perpendicular {
        Some(p) => d.set_item("perpendicular", p.into_pyarray(py))?,
        None => d.set_item("perpendicular", py.None())?,
    }
    d.set_item("resolution", r.resolution)?;
    d.set_item("n_frames", r.n_frames)?;
    Ok(d)
}

/// Vibrational circular dichroism (VCD) transform of a **raw electric/magnetic
/// dipole cross-flux ACF** (the :class:`IrSpectrum` window+FFT pipeline; the
/// signed cross-flux ACF differs).
#[pyclass(module = "molrs.compute", name = "VcdSpectrum")]
pub struct PyVcdSpectrum {
    taper: Taper,
}

#[pymethods]
impl PyVcdSpectrum {
    #[new]
    #[pyo3(signature = (window="cosine_sq", pad_factor=4))]
    fn new(window: &str, pad_factor: usize) -> PyResult<Self> {
        Ok(Self {
            taper: Taper::parse(window, pad_factor)?,
        })
    }

    /// Window + FFT the raw VCD cross-flux ACF (timestep ``dt_fs``) into a
    /// (signed) :class:`Spectrum`.
    fn fit(&self, acf: PyReadonlyArray1<'_, f64>, dt_fs: f64) -> PyResult<PySpectrum> {
        let a = acf.as_array().to_owned();
        let r = VcdSpectrum {
            window: self.taper.window,
            pad_factor: self.taper.pad_factor,
        }
        .fit((&a, dt_fs))
        .map_err(py_value_err)?;
        Ok(PySpectrum::from_result(r, self.taper.window, dt_fs))
    }

    #[getter]
    fn window(&self) -> &'static str {
        self.taper.window.name()
    }

    #[getter]
    fn pad_factor(&self) -> usize {
        self.taper.pad_factor
    }
}

/// Raman optical activity (ROA) transform of **raw iso/aniso ROA
/// cross-correlations** (shares the Raman window+FFT+prefactor pipeline).
#[pyclass(module = "molrs.compute", name = "RoaSpectrum")]
pub struct PyRoaSpectrum {
    inner: RoaSpectrum,
}

#[pymethods]
impl PyRoaSpectrum {
    /// ``RoaSpectrum(incident_frequency_cm1=0.0, temperature_k=0.0, averaged=False,
    /// window="cosine_sq", pad_factor=4)``.
    #[new]
    #[pyo3(signature = (incident_frequency_cm1=0.0, temperature_k=0.0, averaged=false, window="cosine_sq", pad_factor=4))]
    fn new(
        incident_frequency_cm1: f64,
        temperature_k: f64,
        averaged: bool,
        window: &str,
        pad_factor: usize,
    ) -> PyResult<Self> {
        let taper = Taper::parse(window, pad_factor)?;
        Ok(Self {
            inner: RoaSpectrum {
                incident_frequency_cm1,
                temperature_k,
                averaged,
                window: taper.window,
                pad_factor: taper.pad_factor,
            },
        })
    }

    /// Window + FFT the raw iso/aniso ROA ACFs (timestep ``dt_fs``).
    ///
    /// Returns ``{"frequencies_cm1", "isotropic", "anisotropic", "parallel",
    /// "perpendicular", "resolution", "n_frames"}``.
    fn fit<'py>(
        &self,
        py: Python<'py>,
        acf_iso: PyReadonlyArray1<'py, f64>,
        acf_aniso: PyReadonlyArray1<'py, f64>,
        dt_fs: f64,
    ) -> PyResult<Bound<'py, PyDict>> {
        let iso = acf_iso.as_array().to_owned();
        let aniso = acf_aniso.as_array().to_owned();
        let r = self
            .inner
            .fit((&iso, &aniso, dt_fs))
            .map_err(py_value_err)?;
        raman_dict(py, r)
    }
}

/// Resonance-Raman transform of **raw resonant iso/aniso ACFs** (identical
/// pipeline to `RamanSpectrum`; the ACFs come from a resonant polarizability
/// series upstream).
#[pyclass(module = "molrs.compute", name = "ResonanceRamanSpectrum")]
pub struct PyResonanceRamanSpectrum {
    inner: ResonanceRamanSpectrum,
}

#[pymethods]
impl PyResonanceRamanSpectrum {
    /// ``ResonanceRamanSpectrum(incident_frequency_cm1=0.0, temperature_k=0.0, averaged=False,
    /// window="cosine_sq", pad_factor=4)``.
    #[new]
    #[pyo3(signature = (incident_frequency_cm1=0.0, temperature_k=0.0, averaged=false, window="cosine_sq", pad_factor=4))]
    fn new(
        incident_frequency_cm1: f64,
        temperature_k: f64,
        averaged: bool,
        window: &str,
        pad_factor: usize,
    ) -> PyResult<Self> {
        let taper = Taper::parse(window, pad_factor)?;
        Ok(Self {
            inner: ResonanceRamanSpectrum {
                incident_frequency_cm1,
                temperature_k,
                averaged,
                window: taper.window,
                pad_factor: taper.pad_factor,
            },
        })
    }

    /// Window + FFT the raw resonant iso/aniso ACFs (timestep ``dt_fs``).
    ///
    /// Returns the same dict as ``RoaSpectrum.fit``.
    fn fit<'py>(
        &self,
        py: Python<'py>,
        acf_iso: PyReadonlyArray1<'py, f64>,
        acf_aniso: PyReadonlyArray1<'py, f64>,
        dt_fs: f64,
    ) -> PyResult<Bound<'py, PyDict>> {
        let iso = acf_iso.as_array().to_owned();
        let aniso = acf_aniso.as_array().to_owned();
        let r = self
            .inner
            .fit((&iso, &aniso, dt_fs))
            .map_err(py_value_err)?;
        raman_dict(py, r)
    }
}

/// Kramers–Kronig consistency check of a dielectric spectrum: rebuilds
/// ``eps_real`` from ``eps_imag`` and compares.
#[pyclass(module = "molrs.compute", name = "KramersKronig", frozen)]
pub struct PyKramersKronig {
    inner: KramersKronig,
}

#[pymethods]
impl PyKramersKronig {
    #[new]
    fn new(eps_inf: f64) -> Self {
        Self {
            inner: KramersKronig { eps_inf },
        }
    }

    /// Returns ``{"passed", "mae", "eps_real_recovered"}``.
    fn check<'py>(
        &self,
        py: Python<'py>,
        frequency: PyReadonlyArray1<'py, f64>,
        eps_real: PyReadonlyArray1<'py, f64>,
        eps_imag: PyReadonlyArray1<'py, f64>,
    ) -> PyResult<Py<PyAny>> {
        let out = self
            .inner
            .check((
                &frequency.as_array().to_owned(),
                &eps_real.as_array().to_owned(),
                &eps_imag.as_array().to_owned(),
            ))
            .map_err(py_value_err)?;
        let dict = PyDict::new(py);
        dict.set_item("passed", out.passed)?;
        dict.set_item("mae", out.mae)?;
        dict.set_item("eps_real_recovered", out.recovered.into_pyarray(py))?;
        Ok(dict.into())
    }
}

/// Conductivity sum rule: the integral of Re σ(ω) against the equal-time
/// current fluctuation.
#[pyclass(module = "molrs.compute", name = "ConductivitySumRule", frozen)]
pub struct PyConductivitySumRule {
    inner: ConductivitySumRule,
}

#[pymethods]
impl PyConductivitySumRule {
    #[new]
    fn new(current_sq_mean: f64, volume: f64, temperature: f64) -> Self {
        Self {
            inner: ConductivitySumRule {
                current_sq_mean,
                volume,
                temperature,
            },
        }
    }

    /// Returns ``{"passed", "relative_error", "integral", "expected"}``.
    fn check<'py>(
        &self,
        py: Python<'py>,
        frequency: PyReadonlyArray1<'py, f64>,
        conductivity: PyReadonlyArray1<'py, f64>,
    ) -> PyResult<Py<PyAny>> {
        let out = self
            .inner
            .check((
                &frequency.as_array().to_owned(),
                &conductivity.as_array().to_owned(),
            ))
            .map_err(py_value_err)?;
        let dict = PyDict::new(py);
        dict.set_item("passed", out.passed)?;
        dict.set_item("relative_error", out.relative_error)?;
        dict.set_item("integral", out.integral)?;
        dict.set_item("expected", out.expected)?;
        Ok(dict.into())
    }
}

/// Agreement of the same spectrum computed by several routes (pairwise RMS).
#[pyclass(module = "molrs.compute", name = "RouteAgreement", frozen)]
pub struct PyRouteAgreement;

#[pymethods]
impl PyRouteAgreement {
    #[new]
    fn new() -> Self {
        Self
    }

    /// ``results`` maps a route name to its 1-D float64 curve. Returns
    /// ``{"passed", "pairwise_rms"}``.
    fn check<'py>(&self, py: Python<'py>, results: &Bound<'py, PyDict>) -> PyResult<Py<PyAny>> {
        let mut entries = Vec::with_capacity(results.len());
        for (key, value) in results.iter() {
            let name: String = key.extract()?;
            let arr: PyReadonlyArray1<f64> = value.extract().map_err(|_| {
                PyValueError::new_err(format!(
                    "RouteAgreement: value for '{name}' must be a 1-D float64 array"
                ))
            })?;
            entries.push((name, arr.as_array().to_owned()));
        }
        let out = RouteAgreement.check(&entries).map_err(py_value_err)?;
        let pairwise = PyDict::new(py);
        for (label, rms) in &out.pairwise {
            pairwise.set_item(label, rms)?;
        }
        let dict = PyDict::new(py);
        dict.set_item("passed", out.passed)?;
        dict.set_item("pairwise_rms", pairwise)?;
        Ok(dict.into())
    }
}

/// Register this domain's classes and functions.
pub(crate) fn register(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_class::<PyPowerSpectrum>()?;
    m.add_class::<PyIrSpectrum>()?;
    m.add_class::<PyRamanSpectrum>()?;
    m.add_class::<PyEinsteinHelfandSpectrum>()?;
    m.add_class::<PyGreenKuboSpectrum>()?;
    m.add_class::<PyDipoleRateCrossSpectrum>()?;
    m.add_class::<PyDipoleAutocorrelationSpectrum>()?;
    m.add_class::<PyVcdSpectrum>()?;
    m.add_class::<PyRoaSpectrum>()?;
    m.add_class::<PyResonanceRamanSpectrum>()?;
    m.add_class::<PyKramersKronig>()?;
    m.add_class::<PyConductivitySumRule>()?;
    m.add_class::<PyRouteAgreement>()?;
    Ok(())
}
