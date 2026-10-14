//! Clustering (`molrs::compute::Kmeans`): `Kmeans` and `KmeansResult`.

use super::decomposition::PyPcaResult;
use crate::error::py_value_err;
use molrs::compute::{Kmeans, KmeansResult, LegacyCompute};
use molrs::core::Frame as CoreFrame;
use numpy::{IntoPyArray, PyArray1};
use pyo3::prelude::*;

/// k-means cluster labels.
#[pyclass(module = "molrs.compute", name = "KmeansResult")]
pub struct PyKmeansResult {
    inner: KmeansResult,
}

#[pymethods]
impl PyKmeansResult {
    #[getter]
    fn labels<'py>(&self, py: Python<'py>) -> Bound<'py, PyArray1<i32>> {
        ndarray::Array1::from_vec(self.inner.0.clone()).into_pyarray(py)
    }

    fn __len__(&self) -> usize {
        self.inner.0.len()
    }

    fn __repr__(&self) -> String {
        format!("KmeansResult(n={})", self.inner.0.len())
    }
}

/// k-means clustering over a PCA projection (2-D).
#[pyclass(module = "molrs.compute", name = "Kmeans")]
pub struct PyKmeans {
    inner: Kmeans,
}

#[pymethods]
impl PyKmeans {
    #[new]
    #[pyo3(signature = (k, max_iter = 100, seed = 0))]
    fn new(k: usize, max_iter: usize, seed: u64) -> PyResult<Self> {
        Ok(Self {
            inner: Kmeans::new(k, max_iter, seed).map_err(py_value_err)?,
        })
    }

    /// Cluster a `PcaResult`.
    fn compute(&self, pca: &PyPcaResult) -> PyResult<PyKmeansResult> {
        let frames: [&CoreFrame; 0] = [];
        let out = self
            .inner
            .compute(&frames, &pca.inner)
            .map_err(py_value_err)?;
        Ok(PyKmeansResult { inner: out })
    }

    fn __repr__(&self) -> String {
        "Kmeans(...)".to_string()
    }
}

/// Register this domain's classes and functions.
pub(crate) fn register(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_class::<PyKmeansResult>()?;
    m.add_class::<PyKmeans>()?;
    Ok(())
}
