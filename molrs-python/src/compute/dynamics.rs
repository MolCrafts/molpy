//! Time-correlated dynamics (`molrs::compute::dynamics`): autocorrelation, Van
//! Hove, pair persistence.

use super::collect_frames;
use crate::error::py_value_err;
use molrs::compute::{
    AcfResult, LegacyCompute, SurvivalMethod, VanHove, VanHoveResult, pair_survival_tcf,
};
use molrs::core::Frame as CoreFrame;
use numpy::{IntoPyArray, PyArray1, PyArray2, PyReadonlyArray2, PyReadonlyArray3};
use pyo3::prelude::*;
use pyo3::types::{PyAny, PyDictMethods};

// ---------------------------------------------------------------------------
// Van Hove correlation function
// ---------------------------------------------------------------------------

#[pyclass(module = "molrs.compute", name = "VanHoveResult")]
pub struct PyVanHoveResult {
    inner: VanHoveResult,
}

#[pymethods]
impl PyVanHoveResult {
    #[getter]
    fn r_edges<'py>(&self, py: Python<'py>) -> Bound<'py, PyArray1<f64>> {
        self.inner.r_edges.clone().into_pyarray(py)
    }
    #[getter]
    fn r_centers<'py>(&self, py: Python<'py>) -> Bound<'py, PyArray1<f64>> {
        self.inner.r_centers.clone().into_pyarray(py)
    }
    #[getter]
    fn lags(&self) -> Vec<usize> {
        self.inner.lags.clone()
    }
    #[getter]
    fn g_self<'py>(&self, py: Python<'py>) -> Bound<'py, PyArray2<f64>> {
        self.inner.g_self.clone().into_pyarray(py)
    }
    #[getter]
    fn g_distinct<'py>(&self, py: Python<'py>) -> Bound<'py, PyArray2<f64>> {
        self.inner.g_distinct.clone().into_pyarray(py)
    }
    #[getter]
    fn dr(&self) -> f64 {
        self.inner.dr
    }
    #[getter]
    fn has_distinct(&self) -> bool {
        self.inner.has_distinct
    }
}

/// Van Hove correlation function `G(r, t)` (self + distinct parts).
#[pyclass(module = "molrs.compute", name = "VanHove")]
pub struct PyVanHove {
    inner: VanHove,
}

#[pymethods]
impl PyVanHove {
    #[new]
    #[pyo3(signature = (n_rbins, r_max, lags, stride=1))]
    fn new(n_rbins: usize, r_max: f64, lags: Vec<usize>, stride: usize) -> PyResult<Self> {
        let inner = VanHove::new(n_rbins, r_max, lags)
            .map_err(py_value_err)?
            .with_stride(stride);
        Ok(Self { inner })
    }

    /// Compute `G(r, t)` from a trajectory (list of frames, time-ordered).
    fn compute(&self, frames: &Bound<'_, PyAny>) -> PyResult<PyVanHoveResult> {
        let owned = collect_frames(frames)?;
        let refs: Vec<&CoreFrame> = owned.iter().collect();
        let inner = self.inner.compute(&refs, ()).map_err(py_value_err)?;
        Ok(PyVanHoveResult { inner })
    }
}

/// Autocorrelation curve, one entry per lag.
#[pyclass(module = "molrs.compute", name = "AcfResult")]
pub struct PyAcfResult {
    inner: AcfResult,
}

#[pymethods]
impl PyAcfResult {
    /// Lags ``t = 0, 1, …, max_lag``, in frames.
    #[getter]
    fn lags(&self) -> Vec<usize> {
        self.inner.lags.to_vec()
    }

    /// ``C(t)``, in units of the input squared.
    #[getter]
    fn acf<'py>(&self, py: Python<'py>) -> Bound<'py, PyArray1<f64>> {
        self.inner.acf.clone().into_pyarray(py)
    }

    fn __repr__(&self) -> String {
        format!("AcfResult(n_lags={})", self.inner.acf.len())
    }
}

/// Time-autocorrelation of a vector series, averaged over **all** time origins.
///
/// ``C(t) = 1/(N·(T−t)) · Σ_i Σ_τ v_i(τ)·v_i(τ+t)`` — inner product over
/// components, sum over entities, unbiased average over the `T−t` origins that
/// exist for each lag. Computed by the Wiener–Khinchin (FFT) route in
/// O(T log T); see ``molrs::compute::dynamics::acf`` for the references.
///
/// Not the same estimator as :class:`Vacf`, which mean-subtracts each degree of
/// freedom, averages over degrees of freedom, and uses the biased
/// normalisation because it feeds the VDOS spectrum.
///
/// Examples
/// --------
/// >>> molrs.compute.Acf().compute(velocities, max_lag=50).acf
#[pyclass(module = "molrs.compute", name = "Acf")]
pub struct PyAcf;

#[pymethods]
impl PyAcf {
    #[new]
    fn new() -> Self {
        Self
    }

    /// Compute ``C(t)`` for a ``(n_frames, n_entities, n_components)`` series.
    ///
    /// Parameters
    /// ----------
    /// series : ndarray, shape (n_frames, n_entities, n_components)
    /// max_lag : int
    ///     Clamped to ``n_frames - 1``.
    fn compute(&self, series: PyReadonlyArray3<'_, f64>, max_lag: usize) -> PyResult<PyAcfResult> {
        let owned = series.as_array().to_owned();
        let inner =
            molrs::compute::autocorrelation(owned.view(), max_lag, false).map_err(py_value_err)?;
        Ok(PyAcfResult { inner })
    }

    fn __repr__(&self) -> String {
        "Acf()".to_string()
    }
}

/// Pair-survival time-correlation function of the `(i, j)` pairs that start
/// within `[r0, r1]`; returns ``{"lag_times", "correlation"}``.
#[pyfunction(name = "pair_survival_tcf")]
#[pyo3(signature = (coords_i, coords_j, box_lengths, r0, r1, method, dt, max_correlation_time, exclude_self=false))]
#[allow(clippy::too_many_arguments)]
fn pair_survival_tcf_py<'py>(
    py: Python<'py>,
    coords_i: PyReadonlyArray3<'py, f64>,
    coords_j: PyReadonlyArray3<'py, f64>,
    box_lengths: PyReadonlyArray2<'py, f64>,
    r0: f64,
    r1: f64,
    method: &str,
    dt: f64,
    max_correlation_time: usize,
    exclude_self: bool,
) -> PyResult<Py<PyAny>> {
    let ci = coords_i.as_array().to_owned();
    let cj = coords_j.as_array().to_owned();
    let bl = box_lengths.as_array().to_owned();
    let m = SurvivalMethod::parse(method).map_err(py_value_err)?;
    let result = pair_survival_tcf(
        &ci,
        &cj,
        &bl,
        r0,
        r1,
        m,
        dt,
        max_correlation_time,
        exclude_self,
    )
    .map_err(py_value_err)?;
    let dict = pyo3::types::PyDict::new(py);
    dict.set_item("lag_times", result.lag_times.into_pyarray(py))?;
    dict.set_item("correlation", result.correlation.into_pyarray(py))?;
    Ok(dict.into())
}

/// Register this domain's classes and functions.
pub(crate) fn register(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_class::<PyVanHoveResult>()?;
    m.add_class::<PyVanHove>()?;
    m.add_class::<PyAcfResult>()?;
    m.add_class::<PyAcf>()?;
    crate::add_function(
        m,
        "molrs.compute",
        wrap_pyfunction!(pair_survival_tcf_py, m)?,
    )?;
    Ok(())
}
