//! Time-correlated dynamics (`molrs::compute::dynamics`): Van Hove and pair
//! persistence. The correlators (`Acf`, `Oacf`) are in `correlation.rs`.

use super::collect_frames;
use super::correlation::{PyTimeCorrelation, Settings, origin_counts};
use crate::error::py_value_err;
use molrs::compute::{
    Compute, Normalization, SurvivalMethod, VanHove, VanHoveResult, pair_survival_tcf,
};
use molrs::core::Frame as CoreFrame;
use numpy::{IntoPyArray, PyArray1, PyArray2, PyReadonlyArray2, PyReadonlyArray3};
use pyo3::prelude::*;
use pyo3::types::PyAny;

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

/// Pair-survival time-correlation function of the `(i, j)` pairs that start
/// within `[r0, r1]`, over all time origins up to ``max_lag`` frames.
///
/// Returns a :class:`TimeCorrelation` of kind ``"survival"``.
#[pyfunction(name = "pair_survival_tcf")]
#[pyo3(signature = (coords_i, coords_j, box_lengths, r0, r1, method, dt, max_lag, exclude_self=false))]
#[allow(clippy::too_many_arguments)]
fn pair_survival_tcf_py<'py>(
    coords_i: PyReadonlyArray3<'py, f64>,
    coords_j: PyReadonlyArray3<'py, f64>,
    box_lengths: PyReadonlyArray2<'py, f64>,
    r0: f64,
    r1: f64,
    method: &str,
    dt: f64,
    max_lag: usize,
    exclude_self: bool,
) -> PyResult<PyTimeCorrelation> {
    let ci = coords_i.as_array().to_owned();
    let cj = coords_j.as_array().to_owned();
    let bl = box_lengths.as_array().to_owned();
    let m = SurvivalMethod::parse(method).map_err(py_value_err)?;
    let n_frames = ci.dim().0;
    let result = pair_survival_tcf(&ci, &cj, &bl, r0, r1, m, dt, max_lag, exclude_self)
        .map_err(py_value_err)?;
    let n_lags = result.correlation.len();
    let settings = Settings {
        mean_subtract: false,
        normalization: Normalization::Unbiased,
        stride: 1,
    };
    Ok(PyTimeCorrelation::from_curve(
        result.correlation,
        Some(dt),
        "survival",
        settings,
        Some(origin_counts(n_frames, n_lags - 1, 1)),
        n_frames,
    ))
}

/// Register this domain's classes and functions.
pub(crate) fn register(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_class::<PyVanHoveResult>()?;
    m.add_class::<PyVanHove>()?;
    crate::add_function(
        m,
        "molrs.compute",
        wrap_pyfunction!(pair_survival_tcf_py, m)?,
    )?;
    Ok(())
}
