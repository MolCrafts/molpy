//! Radial distribution function (`molrs::compute::rdf`): `Rdf` and its
//! `RdfResult`.

use super::{collect_frames, collect_neighbors};
use crate::error::py_value_err;
use molrs::compute::{LegacyCompute, Rdf, RdfResult};
use molrs::core::Frame as CoreFrame;
use molrs::core::Neighbors;
use numpy::{IntoPyArray, PyArray1};
use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;
use pyo3::types::PyAny;

// ---------------------------------------------------------------------------
// RDF
// ---------------------------------------------------------------------------

/// Radial distribution function g(r) result.
///
/// `rdf` is already normalized (RDF.compute finalizes eagerly).
#[pyclass(module = "molrs.compute", name = "RdfResult")]
pub struct PyRdfResult {
    inner: RdfResult,
}

#[pymethods]
impl PyRdfResult {
    #[getter]
    fn bin_centers<'py>(&self, py: Python<'py>) -> Bound<'py, PyArray1<f64>> {
        self.inner.bin_centers.clone().into_pyarray(py)
    }

    #[getter]
    fn rdf<'py>(&self, py: Python<'py>) -> Bound<'py, PyArray1<f64>> {
        self.inner.rdf.clone().into_pyarray(py)
    }

    #[getter]
    fn bin_edges<'py>(&self, py: Python<'py>) -> Bound<'py, PyArray1<f64>> {
        self.inner.bin_edges.clone().into_pyarray(py)
    }

    #[getter]
    fn n_r<'py>(&self, py: Python<'py>) -> Bound<'py, PyArray1<f64>> {
        self.inner.n_r.clone().into_pyarray(py)
    }

    #[getter]
    fn volume(&self) -> f64 {
        self.inner.volume
    }

    #[getter]
    fn r_min(&self) -> f64 {
        self.inner.r_min
    }

    #[getter]
    fn n_points(&self) -> usize {
        self.inner.n_points
    }

    #[getter]
    fn n_frames(&self) -> usize {
        self.inner.n_frames
    }

    fn __repr__(&self) -> String {
        format!(
            "RdfResult(n_bins={}, n_frames={}, n_points={})",
            self.inner.bin_centers.len(),
            self.inner.n_frames,
            self.inner.n_points,
        )
    }
}

/// Radial distribution function calculator.
///
/// Accepts either a single `(frame, nlist)` pair or a list of each. Results
/// accumulate across frames and are ideal-gas normalized on return.
#[pyclass(module = "molrs.compute", name = "Rdf")]
pub struct PyRdf {
    inner: Rdf,
}

#[pymethods]
impl PyRdf {
    #[new]
    #[pyo3(signature = (n_bins, r_max, r_min = 0.0))]
    fn new(n_bins: usize, r_max: f64, r_min: f64) -> PyResult<Self> {
        let inner = Rdf::new(n_bins, r_max, r_min).map_err(py_value_err)?;
        Ok(Self { inner })
    }

    /// Compute g(r) from a batch of frames + neighbor lists.
    ///
    /// Parameters
    /// ----------
    /// frames : Frame | list[Frame]
    /// nlists : Neighbors | list[Neighbors]
    ///     One neighbor list per frame.
    ///
    /// Returns
    /// -------
    /// RdfResult
    fn compute(
        &self,
        frames: &Bound<'_, PyAny>,
        nlists: &Bound<'_, PyAny>,
    ) -> PyResult<PyRdfResult> {
        let owned = collect_frames(frames)?;
        let refs: Vec<&CoreFrame> = owned.iter().collect();
        let nlists_vec: Vec<Neighbors> = collect_neighbors(nlists)?;
        if nlists_vec.len() != refs.len() {
            return Err(PyValueError::new_err(format!(
                "len(nlists)={} must equal len(frames)={}",
                nlists_vec.len(),
                refs.len()
            )));
        }
        let result = self
            .inner
            .compute(&refs, &nlists_vec)
            .map_err(py_value_err)?;
        Ok(PyRdfResult { inner: result })
    }

    fn __repr__(&self) -> String {
        "Rdf(...)".to_string()
    }
}

/// Register this domain's classes and functions.
pub(crate) fn register(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_class::<PyRdfResult>()?;
    m.add_class::<PyRdf>()?;
    Ok(())
}
