//! Matrix decomposition (`molrs::compute::Pca`): `Pca` and `PcaResult`.

use super::analysis_contract::PyDescriptorRow;
use crate::error::py_value_err;
use molrs::compute::{LegacyCompute, Pca, PcaResult};
use molrs::core::Frame as CoreFrame;
use numpy::{IntoPyArray, PyArray2};
use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;

/// Two-component PCA result.
#[pyclass(module = "molrs.compute", name = "PcaResult", from_py_object)]
#[derive(Clone)]
pub struct PyPcaResult {
    pub(crate) inner: PcaResult,
}

#[pymethods]
impl PyPcaResult {
    /// Row-major `(n_rows, 2)` projected coordinates.
    #[getter]
    fn coords<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyArray2<f64>>> {
        let n = self.inner.coords.len() / 2;
        let flat: Vec<f64> = self.inner.coords.to_vec();
        Ok(ndarray::Array2::from_shape_vec((n, 2), flat)
            .map_err(|e| PyValueError::new_err(e.to_string()))?
            .into_pyarray(py))
    }

    /// `[var_pc1, var_pc2]` — explained variance per component.
    #[getter]
    fn variance(&self) -> (f64, f64) {
        (self.inner.variance[0], self.inner.variance[1])
    }

    fn __repr__(&self) -> String {
        format!(
            "PcaResult(n_rows={}, variance=[{:.4}, {:.4}])",
            self.inner.coords.len() / 2,
            self.inner.variance[0],
            self.inner.variance[1],
        )
    }
}

/// Two-component PCA calculator.
#[pyclass(module = "molrs.compute", name = "Pca")]
pub struct PyPca {
    inner: Pca<PyDescriptorRow>,
}

#[pymethods]
impl PyPca {
    #[new]
    fn new() -> Self {
        Self {
            inner: Pca::<PyDescriptorRow>::new(),
        }
    }

    /// Compute PCA over a list of `DescriptorRow` objects.
    fn compute(&self, rows: Vec<PyRef<'_, PyDescriptorRow>>) -> PyResult<PyPcaResult> {
        let owned: Vec<PyDescriptorRow> = rows.iter().map(|r| (*r).clone()).collect();
        // Wrap an empty FrameAccess slice — Pca does not touch frames.
        let frames: [&CoreFrame; 0] = [];
        let pca = self.inner.compute(&frames, &owned).map_err(py_value_err)?;
        Ok(PyPcaResult { inner: pca })
    }

    fn __repr__(&self) -> String {
        "Pca()".to_string()
    }
}

// ---------------------------------------------------------------------------
// Kmeans
// ---------------------------------------------------------------------------

pub(crate) fn register(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_class::<PyPcaResult>()?;
    m.add_class::<PyPca>()?;
    Ok(())
}
