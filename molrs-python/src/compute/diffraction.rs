//! Diffraction (`molrs::compute::diffraction`): the Debye static structure
//! factor.

#![allow(clippy::type_complexity)]

use super::collect_frames;
use crate::error::py_value_err;
use molrs::compute::{LegacyCompute, StaticStructureFactorDebye};
use molrs::core::Frame as CoreFrame;
use numpy::{IntoPyArray, PyArray1};
use pyo3::prelude::*;
use pyo3::types::PyAny;

// ---------------------------------------------------------------------------
// StaticStructureFactorDebye
// ---------------------------------------------------------------------------

#[pyclass(module = "molrs.compute", name = "StaticStructureFactorDebye")]

pub struct PyStaticStructureFactorDebye {
    inner: StaticStructureFactorDebye,
}

#[pymethods]
impl PyStaticStructureFactorDebye {
    #[new]
    fn new(k_values: Vec<f64>) -> PyResult<Self> {
        Ok(Self {
            inner: StaticStructureFactorDebye::new(&k_values).map_err(py_value_err)?,
        })
    }

    #[staticmethod]
    fn linspace(k_min: f64, k_max: f64, n: usize) -> PyResult<Self> {
        Ok(Self {
            inner: StaticStructureFactorDebye::linspace(k_min, k_max, n).map_err(py_value_err)?,
        })
    }

    /// Returns per-frame `(k_values, S(k), n_particles)`.
    fn compute<'py>(
        &self,
        py: Python<'py>,
        frames: &Bound<'py, PyAny>,
    ) -> PyResult<Vec<(Bound<'py, PyArray1<f64>>, Bound<'py, PyArray1<f64>>, usize)>> {
        let owned = collect_frames(frames)?;
        let refs: Vec<&CoreFrame> = owned.iter().collect();
        let results = self.inner.compute(&refs, ()).map_err(py_value_err)?;
        Ok(results
            .into_iter()
            .map(|r| {
                (
                    r.k_values.into_pyarray(py),
                    r.sk.into_pyarray(py),
                    r.n_particles,
                )
            })
            .collect())
    }
}

/// Register this domain's classes and functions.
pub(crate) fn register(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_class::<PyStaticStructureFactorDebye>()?;
    Ok(())
}
