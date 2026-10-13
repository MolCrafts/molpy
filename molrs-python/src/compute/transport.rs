//! Transport raw computes (`molrs::compute::transport`): velocity and current
//! correlations, Einstein / Green–Kubo diffusion and conductivity, Debye
//! relaxation and its fit, Onsager cross-correlation. Each returns a
//! [`TimeCorrelation`](super::correlation::PyTimeCorrelation); the coefficient
//! is the caller's explicit fit (`.diffusion()`, `.linear_fit()`,
//! `.integrate()`, `DebyeFit`).

use super::EMPTY_FRAMES;
use super::correlation::{PyTimeCorrelation, Settings, cartesian_series, finish, origin_counts};
use super::msd::positions_of;
use crate::error::py_value_err;
use molrs::compute::{
    Compute, DebyeFit, DipoleRateCross, EwaldBoundary, Fit, TimeCorrelationSpec, autocorrelate,
    displacement_correlate,
};
use ndarray::{Array3, Axis};
use numpy::{PyReadonlyArray1, PyReadonlyArray2, PyReadonlyArray3};
use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;
use pyo3::types::{PyAny, PyDict, PyDictMethods};

/// A velocity array as the `(n_frames, n_dof, 1)` series the VACF averages
/// over: `(n_frames, n_dof)` as is, `(n_frames, n_atoms, 3)` flattened.
fn dof_series(velocities: &Bound<'_, PyAny>) -> PyResult<Array3<f64>> {
    if let Ok(v2) = velocities.extract::<PyReadonlyArray2<'_, f64>>() {
        return Ok(v2.as_array().to_owned().insert_axis(Axis(2)));
    }
    if let Ok(v3) = velocities.extract::<PyReadonlyArray3<'_, f64>>() {
        let a = v3.as_array();
        let (t, n, d) = a.dim();
        let flat = a
            .to_owned()
            .into_shape_with_order((t, n * d, 1))
            .map_err(|e| PyValueError::new_err(e.to_string()))?;
        return Ok(flat);
    }
    Err(PyValueError::new_err(
        "velocities must be a float64 array of shape (n_frames, n_dof) or (n_frames, n_atoms, 3)",
    ))
}

fn vacf(
    settings: Settings,
    velocities: &Bound<'_, PyAny>,
    dt: f64,
    max_lag: usize,
    per_entity: bool,
) -> PyResult<PyTimeCorrelation> {
    let series = dof_series(velocities)?;
    let tc = autocorrelate(series.view(), dt, &settings.spec(max_lag, per_entity))
        .map_err(py_value_err)?;
    Ok(finish(tc, Some(dt), "vacf", settings, series.dim().0, None))
}

// ── Vacf ─────────────────────────────────────────────────────────────────────

/// Velocity autocorrelation function, ``C(t) = ⟨v(τ) · v(τ+t)⟩`` averaged
/// over degrees of freedom and time origins.
///
/// ``velocities`` is ``(n_frames, n_dof)`` or ``(n_frames, n_atoms, 3)``;
/// each degree of freedom loses its time mean first (``mean_subtract=True``)
/// and the curve is the unbiased all-origins estimator — :class:`Acf` on the
/// ``(n_frames, n_dof, 1)`` view. Compose with
/// :meth:`TimeCorrelation.spectrum` for the VDOS or
/// :meth:`TimeCorrelation.diffusion` (Green–Kubo) for D.
#[pyclass(module = "molrs.compute", name = "Vacf")]
pub struct PyVacf {
    settings: Settings,
}

#[pymethods]
impl PyVacf {
    #[new]
    #[pyo3(signature = (mean_subtract=true, normalization="unbiased", stride=1))]
    fn new(mean_subtract: bool, normalization: &str, stride: usize) -> PyResult<Self> {
        Ok(Self {
            settings: Settings::parse(mean_subtract, normalization, stride)?,
        })
    }

    /// The VACF for lags ``0..=max_lag`` (clamped to ``n_frames - 1``), with
    /// ``dt`` the frame spacing; kind ``"vacf"``.
    #[pyo3(signature = (velocities, dt, max_lag, per_entity=false))]
    fn compute(
        &self,
        velocities: &Bound<'_, PyAny>,
        dt: f64,
        max_lag: usize,
        per_entity: bool,
    ) -> PyResult<PyTimeCorrelation> {
        vacf(self.settings, velocities, dt, max_lag, per_entity)
    }
}

// ── GreenKuboDiffusion ────────────────────────────────────────────────────────

/// The velocity ACF for the Green–Kubo diffusion route: the same curve as
/// :class:`Vacf` with its defaults. ``D = (1/d) ∫ C(t) dt`` is then
/// :meth:`TimeCorrelation.diffusion`.
#[pyclass(module = "molrs.compute", name = "GreenKuboDiffusion")]
pub struct PyGreenKuboDiffusion;

#[pymethods]
impl PyGreenKuboDiffusion {
    #[new]
    fn new() -> Self {
        Self
    }

    /// ``Vacf().compute(velocities, dt, max_lag)``.
    #[pyo3(signature = (velocities, dt, max_lag))]
    fn compute(
        &self,
        velocities: &Bound<'_, PyAny>,
        dt: f64,
        max_lag: usize,
    ) -> PyResult<PyTimeCorrelation> {
        let settings = Settings::parse(true, "unbiased", 1)?;
        vacf(settings, velocities, dt, max_lag, false)
    }
}

// ── EinsteinDiffusion ─────────────────────────────────────────────────────────

/// The self-MSD for the Einstein diffusion route: ``Msd(method="window")``
/// over every lag, atom-averaged. ``D = slope / (2d)`` is then
/// :meth:`TimeCorrelation.diffusion`.
#[pyclass(module = "molrs.compute", name = "EinsteinDiffusion")]
pub struct PyEinsteinDiffusion;

#[pymethods]
impl PyEinsteinDiffusion {
    #[new]
    fn new() -> Self {
        Self
    }

    /// The windowed MSD of ``frames_or_positions`` (a ``Frame``, a sequence
    /// of frames, or an ``(n_frames, n_atoms, 3)`` unwrapped array) with
    /// ``dt`` the frame spacing; kind ``"msd"``.
    fn compute(
        &self,
        frames_or_positions: &Bound<'_, PyAny>,
        dt: f64,
    ) -> PyResult<PyTimeCorrelation> {
        let positions = positions_of(frames_or_positions)?;
        let settings = Settings::parse(false, "unbiased", 1)?;
        let tc = displacement_correlate(positions.view(), None, dt, &TimeCorrelationSpec::new())
            .map_err(py_value_err)?;
        Ok(finish(
            tc,
            Some(dt),
            "msd",
            settings,
            positions.dim().0,
            None,
        ))
    }
}

// ── EinsteinConductivity ──────────────────────────────────────────────────────

/// The collective charge-dipole MSD ``⟨|M_J(τ+t) − M_J(τ)|²⟩`` of the
/// Einstein–Helfand conductivity route, with **no** fitted σ:
/// ``σ = slope / (6 V k_B T)`` is the caller's :meth:`TimeCorrelation.linear_fit`
/// plus prefactor.
#[pyclass(module = "molrs.compute", name = "EinsteinConductivity")]
pub struct PyEinsteinConductivity {
    stride: usize,
}

#[pymethods]
impl PyEinsteinConductivity {
    #[new]
    #[pyo3(signature = (stride=1))]
    fn new(stride: usize) -> PyResult<Self> {
        if stride == 0 {
            return Err(PyValueError::new_err("stride must be >= 1"));
        }
        Ok(Self { stride })
    }

    /// ``translational_dipole`` is ``(n_frames, 3)``; kind ``"msd"``.
    #[pyo3(signature = (translational_dipole, dt, max_lag))]
    fn compute(
        &self,
        translational_dipole: PyReadonlyArray2<'_, f64>,
        dt: f64,
        max_lag: usize,
    ) -> PyResult<PyTimeCorrelation> {
        let m = cartesian_series(&translational_dipole, "translational_dipole")?;
        let settings = Settings::parse(false, "unbiased", self.stride)?;
        let tc = displacement_correlate(
            m.view().insert_axis(Axis(1)),
            None,
            dt,
            &settings.spec(max_lag, false),
        )
        .map_err(py_value_err)?;
        Ok(finish(tc, Some(dt), "msd", settings, m.dim().0, None))
    }
}

// ── GreenKuboConductivity ─────────────────────────────────────────────────────

/// The current autocorrelation ``⟨J(τ) · J(τ+t)⟩`` of the Green–Kubo
/// conductivity route, with **no** fitted σ: ``σ = (1 / (3 V k_B T)) ∫ C dt``
/// is the caller's :meth:`TimeCorrelation.integrate` plus prefactor. The
/// current is not mean-subtracted by default.
#[pyclass(module = "molrs.compute", name = "GreenKuboConductivity")]
pub struct PyGreenKuboConductivity {
    settings: Settings,
}

#[pymethods]
impl PyGreenKuboConductivity {
    #[new]
    #[pyo3(signature = (mean_subtract=false, normalization="unbiased", stride=1))]
    fn new(mean_subtract: bool, normalization: &str, stride: usize) -> PyResult<Self> {
        Ok(Self {
            settings: Settings::parse(mean_subtract, normalization, stride)?,
        })
    }

    /// ``current`` is ``(n_frames, 3)``; kind ``"jacf"``.
    #[pyo3(signature = (current, dt, max_lag))]
    fn compute(
        &self,
        current: PyReadonlyArray2<'_, f64>,
        dt: f64,
        max_lag: usize,
    ) -> PyResult<PyTimeCorrelation> {
        let j = cartesian_series(&current, "current")?;
        let tc = autocorrelate(
            j.view().insert_axis(Axis(1)),
            dt,
            &self.settings.spec(max_lag, false),
        )
        .map_err(py_value_err)?;
        Ok(finish(tc, Some(dt), "jacf", self.settings, j.dim().0, None))
    }
}

// ── DipoleRateCross ───────────────────────────────────────────────────────────

/// Dipole-rate × dipole cross-correlation ``C(t) = Σ_α ⟨δṀ_α(τ) δM_α(τ+t)⟩``,
/// with ``Ṁ`` the second-order finite difference of ``M`` (NumPy ``gradient``,
/// ``edge_order=2``). Compose with ``DipoleRateCrossSpectrum`` for ε(ω).
#[pyclass(module = "molrs.compute", name = "DipoleRateCross")]
pub struct PyDipoleRateCross {
    settings: Settings,
}

#[pymethods]
impl PyDipoleRateCross {
    #[new]
    #[pyo3(signature = (mean_subtract=true, normalization="unbiased", stride=1))]
    fn new(mean_subtract: bool, normalization: &str, stride: usize) -> PyResult<Self> {
        Ok(Self {
            settings: Settings::parse(mean_subtract, normalization, stride)?,
        })
    }

    /// ``dipole_moments`` is ``(n_frames, 3)``; kind ``"cross"``.
    #[pyo3(signature = (dipole_moments, dt, max_lag))]
    fn compute(
        &self,
        dipole_moments: PyReadonlyArray2<'_, f64>,
        dt: f64,
        max_lag: usize,
    ) -> PyResult<PyTimeCorrelation> {
        let dm = cartesian_series(&dipole_moments, "dipole_moments")?;
        let n_frames = dm.dim().0;
        let r = DipoleRateCross {
            mean_subtract: self.settings.mean_subtract,
            normalization: self.settings.normalization,
            origin_stride: self.settings.stride,
        }
        .compute(EMPTY_FRAMES, (&dm, dt, max_lag))
        .map_err(py_value_err)?;
        let n_lags = r.cross.len();
        Ok(PyTimeCorrelation::from_curve(
            r.cross,
            Some(dt),
            "cross",
            self.settings,
            Some(origin_counts(n_frames, n_lags - 1, self.settings.stride)),
            n_frames,
        ))
    }
}

// ── DebyeRelaxation ───────────────────────────────────────────────────────────

/// The fluctuation dipole ACF ``⟨δM(τ) · δM(τ+t)⟩`` of the Debye relaxation
/// route, carrying the metadata the Debye amplitude needs in ``extras``:
/// ``volume``, ``temperature``, ``boundary`` and ``zero_lag_variance``
/// (``values[0]``). τ is :meth:`TimeCorrelation.fit_exponential`; the same
/// curve feeds ``EinsteinHelfandSpectrum`` and
/// ``DipoleAutocorrelationSpectrum``.
#[pyclass(module = "molrs.compute", name = "DebyeRelaxation")]
pub struct PyDebyeRelaxation {
    volume: f64,
    temperature: f64,
    boundary: EwaldBoundary,
    settings: Settings,
}

#[pymethods]
impl PyDebyeRelaxation {
    /// ``DebyeRelaxation(volume, temperature, boundary="tinfoil",
    /// mean_subtract=True, normalization="unbiased", stride=1)``.
    #[new]
    #[pyo3(signature = (volume, temperature, boundary="tinfoil", mean_subtract=true, normalization="unbiased", stride=1))]
    fn new(
        volume: f64,
        temperature: f64,
        boundary: &str,
        mean_subtract: bool,
        normalization: &str,
        stride: usize,
    ) -> PyResult<Self> {
        if volume <= 0.0 {
            return Err(PyValueError::new_err(format!(
                "volume must be positive, got {volume}"
            )));
        }
        if temperature <= 0.0 {
            return Err(PyValueError::new_err(format!(
                "temperature must be positive, got {temperature}"
            )));
        }
        Ok(Self {
            volume,
            temperature,
            boundary: EwaldBoundary::from_name(boundary).map_err(py_value_err)?,
            settings: Settings::parse(mean_subtract, normalization, stride)?,
        })
    }

    /// ``dipole_moments`` is ``(n_frames, 3)``; kind ``"dipole_acf"``.
    #[pyo3(signature = (dipole_moments, dt, max_lag))]
    fn compute(
        &self,
        py: Python<'_>,
        dipole_moments: PyReadonlyArray2<'_, f64>,
        dt: f64,
        max_lag: usize,
    ) -> PyResult<PyTimeCorrelation> {
        let dm = cartesian_series(&dipole_moments, "dipole_moments")?;
        let tc = autocorrelate(
            dm.view().insert_axis(Axis(1)),
            dt,
            &self.settings.spec(max_lag, false),
        )
        .map_err(py_value_err)?;
        let zero_lag_variance = tc.values[0];
        let mut out = finish(tc, Some(dt), "dipole_acf", self.settings, dm.dim().0, None);
        let extras = PyDict::new(py);
        extras.set_item("volume", self.volume)?;
        extras.set_item("temperature", self.temperature)?;
        extras.set_item(
            "boundary",
            match self.boundary {
                EwaldBoundary::TinFoil => "tinfoil",
                EwaldBoundary::Vacuum => "vacuum",
            },
        )?;
        extras.set_item("zero_lag_variance", zero_lag_variance)?;
        out.set_extras(extras.unbind());
        Ok(out)
    }
}

// ── DebyeFit ──────────────────────────────────────────────────────────────────

/// Single-exponential (Debye) relaxation fit of a **normalized** correlation
/// function ``Φ(t) = A·exp(−t/τ)``, by log-linear least squares over the
/// leading positive run; returns τ and the amplitude A.
/// :meth:`TimeCorrelation.fit_exponential` is this fit on a curve.
#[pyclass(module = "molrs.compute", name = "DebyeFit")]
pub struct PyDebyeFit;

#[pymethods]
impl PyDebyeFit {
    #[new]
    fn new() -> Self {
        Self
    }

    /// Fit Φ(t) = A·exp(−t/τ) on the normalized curve ``phi`` (sample step ``dt``).
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

// ── OnsagerCorrelation ────────────────────────────────────────────────────────

/// Onsager collective mean-displacement cross-correlation
/// ``L_ij(t) = ⟨ΔP_i(t) · ΔP_j(t)⟩`` over all time origins (the integrand of
/// the Onsager coefficients); ``p_i == p_j`` is that species' collective MSD.
#[pyclass(module = "molrs.compute", name = "OnsagerCorrelation", frozen)]
pub struct PyOnsagerCorrelation {
    stride: usize,
}

#[pymethods]
impl PyOnsagerCorrelation {
    #[new]
    #[pyo3(signature = (stride=1))]
    fn new(stride: usize) -> PyResult<Self> {
        if stride == 0 {
            return Err(PyValueError::new_err("stride must be >= 1"));
        }
        Ok(Self { stride })
    }

    /// ``p_i`` and ``p_j`` are ``(n_frames, 3)`` unwrapped collective
    /// coordinates; kind ``"onsager"``.
    #[pyo3(signature = (p_i, p_j, dt, max_lag))]
    fn compute(
        &self,
        p_i: PyReadonlyArray2<'_, f64>,
        p_j: PyReadonlyArray2<'_, f64>,
        dt: f64,
        max_lag: usize,
    ) -> PyResult<PyTimeCorrelation> {
        let pi = cartesian_series(&p_i, "p_i")?;
        let pj = cartesian_series(&p_j, "p_j")?;
        if pi.dim().0 != pj.dim().0 {
            return Err(PyValueError::new_err(format!(
                "p_i and p_j must have one frame count, got {} and {}",
                pi.dim().0,
                pj.dim().0
            )));
        }
        let settings = Settings::parse(false, "unbiased", self.stride)?;
        let tc = displacement_correlate(
            pi.view().insert_axis(Axis(1)),
            Some(pj.view().insert_axis(Axis(1))),
            dt,
            &settings.spec(max_lag, false),
        )
        .map_err(py_value_err)?;
        Ok(finish(tc, Some(dt), "onsager", settings, pi.dim().0, None))
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
