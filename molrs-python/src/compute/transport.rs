//! Transport raw computes (`molrs::compute::transport`): velocity and current
//! correlations, Einstein / Green–Kubo diffusion and conductivity, Debye
//! relaxation and its fit, Onsager cross-correlation. Each returns the raw
//! curve; the coefficient is the caller's explicit fit.

use super::{EMPTY_FRAMES, collect_frames};
use crate::error::py_value_err;
use molrs::compute::{
    DebyeFit, DebyeRelaxation, DipoleRateCross, EinsteinConductivity, EinsteinDiffusion,
    EinsteinDiffusionArgs, EwaldBoundary, Fit, GreenKuboConductivity, GreenKuboDiffusion,
    LegacyCompute, OnsagerCorrelation, Vacf,
};
use molrs::core::Frame as CoreFrame;
use numpy::{IntoPyArray, PyReadonlyArray1, PyReadonlyArray2};
use pyo3::prelude::*;
use pyo3::types::{PyAny, PyDict, PyDictMethods};

// ═══════════════════════════════════════════════════════════════════════════
// Raw computes
// ═══════════════════════════════════════════════════════════════════════════

// ── VACF ─────────────────────────────────────────────────────────────────────

/// Raw unnormalized velocity autocorrelation function (the VDOS /
/// Green–Kubo-diffusion input). Returns only the raw ACF curve — compose with
/// [`PowerSpectrum`](PyPowerSpectrum) for VDOS or
/// [`CumulativeTrapezoid`](PyCumulativeTrapezoid) for D.
#[pyclass(module = "molrs.compute", name = "Vacf")]
pub struct PyVacf;

#[pymethods]
impl PyVacf {
    #[new]
    fn new() -> Self {
        Self
    }

    /// Compute the raw velocity ACF.
    ///
    /// Returns a dict ``{"lag_times", "acf"}`` of float64 arrays. **No** fitted
    /// scalar.
    fn compute<'py>(
        &self,
        py: Python<'py>,
        velocities: PyReadonlyArray2<'py, f64>,
        dt: f64,
        resolution: usize,
    ) -> PyResult<Bound<'py, PyDict>> {
        let v = velocities.as_array().to_owned();
        let r = Vacf
            .compute(EMPTY_FRAMES, (&v, dt, resolution))
            .map_err(py_value_err)?;
        let d = PyDict::new(py);
        d.set_item("lag_times", r.lag_times.into_pyarray(py))?;
        d.set_item("acf", r.acf.into_pyarray(py))?;
        Ok(d)
    }
}

// ── GreenKuboDiffusion (raw velocity ACF, diffusion route) ────────────────────

/// Raw velocity ACF for the Green–Kubo diffusion route (same raw curve as
/// [`Vacf`](PyVacf)). `D = (1/d)·∫ Vacf dt` is then a
/// [`CumulativeTrapezoid`](PyCumulativeTrapezoid) + scale step.
#[pyclass(module = "molrs.compute", name = "GreenKuboDiffusion")]
pub struct PyGreenKuboDiffusion;

#[pymethods]
impl PyGreenKuboDiffusion {
    #[new]
    fn new() -> Self {
        Self
    }

    /// Returns ``{"lag_times", "acf"}``; no fitted D.
    fn compute<'py>(
        &self,
        py: Python<'py>,
        velocities: PyReadonlyArray2<'py, f64>,
        dt: f64,
        resolution: usize,
    ) -> PyResult<Bound<'py, PyDict>> {
        let v = velocities.as_array().to_owned();
        let r = GreenKuboDiffusion
            .compute(EMPTY_FRAMES, (&v, dt, resolution))
            .map_err(py_value_err)?;
        let d = PyDict::new(py);
        d.set_item("lag_times", r.lag_times.into_pyarray(py))?;
        d.set_item("acf", r.acf.into_pyarray(py))?;
        Ok(d)
    }
}

// ── EinsteinDiffusion (raw self-MSD; consumes frames) ─────────────────────────

/// Raw self-MSD for the Einstein diffusion route. Delegates to
/// `Msd::windowed` — MSD math is NOT re-derived. `D = slope/(2d)` is then a
/// [`LinearFit`](PyLinearFit) + scale step.
#[pyclass(module = "molrs.compute", name = "EinsteinDiffusion")]
pub struct PyEinsteinDiffusion;

#[pymethods]
impl PyEinsteinDiffusion {
    #[new]
    fn new() -> Self {
        Self
    }

    /// Compute the raw self-MSD over ``frames`` (a `Frame` or list of `Frame`).
    ///
    /// Returns ``{"lag_times", "msd"}``; no fitted D.
    fn compute<'py>(
        &self,
        py: Python<'py>,
        frames: &Bound<'py, PyAny>,
        dt: f64,
    ) -> PyResult<Bound<'py, PyDict>> {
        let owned = collect_frames(frames)?;
        let refs: Vec<&CoreFrame> = owned.iter().collect();
        let r = EinsteinDiffusion
            .compute(&refs, EinsteinDiffusionArgs { dt })
            .map_err(py_value_err)?;
        let d = PyDict::new(py);
        d.set_item("lag_times", r.lag_times.into_pyarray(py))?;
        d.set_item("msd", r.msd.into_pyarray(py))?;
        Ok(d)
    }
}

// ── EinsteinConductivity (raw collective charge-dipole MSD) ───────────────────

/// Raw collective charge-dipole MSD, with **no** fitted sigma/slope.
/// `σ = slope/(6·V·k_B·T)·prefactor` is a downstream
/// [`LinearFit`](PyLinearFit) + scale step.
#[pyclass(module = "molrs.compute", name = "EinsteinConductivity")]
pub struct PyEinsteinConductivity;

#[pymethods]
impl PyEinsteinConductivity {
    #[new]
    fn new() -> Self {
        Self
    }

    /// Compute the raw collective-dipole MSD.
    ///
    /// ``translational_dipole`` is ``(n_frames, 3)``. Returns
    /// ``{"lag_times", "msd"}``; **no** ``sigma``/``slope``.
    fn compute<'py>(
        &self,
        py: Python<'py>,
        translational_dipole: PyReadonlyArray2<'py, f64>,
        dt: f64,
        max_correlation_time: usize,
    ) -> PyResult<Bound<'py, PyDict>> {
        let td = translational_dipole.as_array().to_owned();
        let r = EinsteinConductivity
            .compute(EMPTY_FRAMES, (&td, dt, max_correlation_time))
            .map_err(py_value_err)?;
        let d = PyDict::new(py);
        d.set_item("lag_times", r.lag_times.into_pyarray(py))?;
        d.set_item("msd", r.msd.into_pyarray(py))?;
        Ok(d)
    }
}

// ── GreenKuboConductivity (raw current ACF) ───────────────────────────────────

/// Raw current autocorrelation function, with **no** fitted sigma. The
/// σ = (1/(3·V·k_B·T))·∫⟨JJ⟩ step is a downstream
/// [`CumulativeTrapezoid`](PyCumulativeTrapezoid) + scale.
#[pyclass(module = "molrs.compute", name = "GreenKuboConductivity")]
pub struct PyGreenKuboConductivity;

#[pymethods]
impl PyGreenKuboConductivity {
    #[new]
    fn new() -> Self {
        Self
    }

    /// Compute the raw current ACF.
    ///
    /// ``current`` is ``(n_frames, 3)``. Returns ``{"lag_times", "jacf"}``;
    /// **no** ``sigma``/``sigma_running``.
    fn compute<'py>(
        &self,
        py: Python<'py>,
        current: PyReadonlyArray2<'py, f64>,
        dt: f64,
        max_correlation_time: usize,
    ) -> PyResult<Bound<'py, PyDict>> {
        let j = current.as_array().to_owned();
        let r = GreenKuboConductivity
            .compute(EMPTY_FRAMES, (&j, dt, max_correlation_time))
            .map_err(py_value_err)?;
        let d = PyDict::new(py);
        d.set_item("lag_times", r.lag_times.into_pyarray(py))?;
        d.set_item("jacf", r.jacf.into_pyarray(py))?;
        Ok(d)
    }
}

// ── DipoleRateCross (raw C_ṀM correlator) ─────────────────────────────────────

/// Raw dipole-rate × dipole cross-correlation `C(t) = Σ_α ⟨δṀ_α(0) δM_α(t)⟩`.
///
/// Finite-difference `Ṁ` (NumPy `gradient` / edge_order=2), mean-subtracted
/// Cartesian xcorr. Compose with [`DipoleRateCrossSpectrum`] for ε(ω).
#[pyclass(module = "molrs.compute", name = "DipoleRateCross")]
pub struct PyDipoleRateCross;

#[pymethods]
impl PyDipoleRateCross {
    #[new]
    fn new() -> Self {
        Self
    }

    /// Compute the raw cross correlator from ``dipole_moments`` ``(n_frames, 3)``.
    ///
    /// Returns ``{"lag_times", "cross"}``.
    fn compute<'py>(
        &self,
        py: Python<'py>,
        dipole_moments: PyReadonlyArray2<'py, f64>,
        dt: f64,
        max_correlation_time: usize,
    ) -> PyResult<Bound<'py, PyDict>> {
        let dm = dipole_moments.as_array().to_owned();
        let r = DipoleRateCross
            .compute(EMPTY_FRAMES, (&dm, dt, max_correlation_time))
            .map_err(py_value_err)?;
        let d = PyDict::new(py);
        d.set_item("lag_times", r.lag_times.into_pyarray(py))?;
        d.set_item("cross", r.cross.into_pyarray(py))?;
        Ok(d)
    }
}

// ── DebyeRelaxation (raw dipole ACF + V/T/BC metadata) ────────────────────────

/// Raw dipole-ACF compute for the Debye relaxation route. Carries the
/// unnormalized ACF, the zero-lag variance ⟨M(0)²⟩, and the V/T/Ewald-BC
/// metadata the Debye amplitude needs (invariants b, c). The relaxation *shape*
/// τ comes from [`DebyeFit`](PyDebyeFit) applied to the **normalized** ACF.
/// The same ``acf`` feeds [`DipoleAutocorrelationSpectrum`].
#[pyclass(module = "molrs.compute", name = "DebyeRelaxation")]
pub struct PyDebyeRelaxation {
    inner: DebyeRelaxation,
}

#[pymethods]
impl PyDebyeRelaxation {
    /// ``DebyeRelaxation(volume, temperature, boundary="tinfoil")``.
    #[new]
    #[pyo3(signature = (volume, temperature, boundary="tinfoil"))]
    fn new(volume: f64, temperature: f64, boundary: &str) -> PyResult<Self> {
        Ok(Self {
            inner: DebyeRelaxation {
                volume,
                temperature,
                boundary: EwaldBoundary::from_name(boundary).map_err(py_value_err)?,
            },
        })
    }

    /// Compute the raw dipole ACF + metadata.
    ///
    /// ``dipole_moments`` is ``(n_frames, 3)``. Returns ``{"lag_times", "acf",
    /// "zero_lag_variance", "volume", "temperature", "boundary"}``; the ACF is
    /// **unnormalized** (``acf[0] == zero_lag_variance``).
    fn compute<'py>(
        &self,
        py: Python<'py>,
        dipole_moments: PyReadonlyArray2<'py, f64>,
        dt: f64,
        max_correlation_time: usize,
    ) -> PyResult<Bound<'py, PyDict>> {
        let dm = dipole_moments.as_array().to_owned();
        let r = self
            .inner
            .compute(EMPTY_FRAMES, (&dm, dt, max_correlation_time))
            .map_err(py_value_err)?;
        let d = PyDict::new(py);
        d.set_item("lag_times", r.lag_times.into_pyarray(py))?;
        d.set_item("acf", r.acf.into_pyarray(py))?;
        d.set_item("zero_lag_variance", r.zero_lag_variance)?;
        d.set_item("volume", r.volume)?;
        d.set_item("temperature", r.temperature)?;
        d.set_item(
            "boundary",
            match r.boundary {
                EwaldBoundary::TinFoil => "tinfoil",
                EwaldBoundary::Vacuum => "vacuum",
            },
        )?;
        Ok(d)
    }
}

// ── DebyeFit (time-domain log-linear ACF fit) ─────────────────────────────────

/// Single-exponential (Debye) relaxation fit of a **normalized** dipole ACF
/// Φ(t) = A·exp(−t/τ), by log-linear least squares over the leading positive
/// run. This is the **time-domain** ACF fit consolidated from molpy; it returns
/// τ and the amplitude A.
#[pyclass(module = "molrs.compute", name = "DebyeFit")]
pub struct PyDebyeFit;

#[pymethods]
impl PyDebyeFit {
    #[new]
    fn new() -> Self {
        Self
    }

    /// Fit Φ(t) = A·exp(−t/τ) on the normalized ACF ``phi`` (sample step ``dt``).
    ///
    /// Returns ``{"tau", "amplitude", "n_samples"}``.
    fn fit<'py>(
        &self,
        py: Python<'py>,
        phi: PyReadonlyArray1<'py, f64>,
        dt: f64,
    ) -> PyResult<Bound<'py, PyDict>> {
        let p = phi.as_array().to_owned();
        let r = DebyeFit.fit((&p, dt)).map_err(py_value_err)?;
        let d = PyDict::new(py);
        d.set_item("tau", r.tau)?;
        d.set_item("amplitude", r.amplitude)?;
        d.set_item("n_samples", r.n_samples)?;
        Ok(d)
    }
}

/// Onsager collective mean-displacement cross-correlation
/// `⟨ΔP_i(t) · ΔP_j(t)⟩` (the integrand of the Onsager coefficients `L_ij`).
#[pyclass(module = "molrs.compute", name = "OnsagerCorrelation", frozen)]
pub struct PyOnsagerCorrelation;

#[pymethods]
impl PyOnsagerCorrelation {
    #[new]
    fn new() -> Self {
        Self
    }

    /// Returns a dict ``{"lag_times", "correlation"}`` of float64 arrays.
    #[pyo3(signature = (p_i, p_j, dt, max_correlation_time))]
    fn compute<'py>(
        &self,
        py: Python<'py>,
        p_i: PyReadonlyArray2<'py, f64>,
        p_j: PyReadonlyArray2<'py, f64>,
        dt: f64,
        max_correlation_time: usize,
    ) -> PyResult<Py<PyAny>> {
        let pi = p_i.as_array().to_owned();
        let pj = p_j.as_array().to_owned();
        let result = OnsagerCorrelation
            .compute(EMPTY_FRAMES, (&pi, &pj, dt, max_correlation_time))
            .map_err(py_value_err)?;
        let dict = pyo3::types::PyDict::new(py);
        dict.set_item("lag_times", result.lag_times.into_pyarray(py))?;
        dict.set_item("correlation", result.correlation.into_pyarray(py))?;
        Ok(dict.into())
    }
}

/// Register this domain's classes and functions.
pub(crate) fn register(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_class::<PyVacf>()?;
    m.add_class::<PyGreenKuboDiffusion>()?;
    m.add_class::<PyEinsteinDiffusion>()?;
    m.add_class::<PyEinsteinConductivity>()?;
    m.add_class::<PyGreenKuboConductivity>()?;
    m.add_class::<PyDipoleRateCross>()?;
    m.add_class::<PyDebyeRelaxation>()?;
    m.add_class::<PyDebyeFit>()?;
    m.add_class::<PyOnsagerCorrelation>()?;
    Ok(())
}
