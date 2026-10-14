//! Local environment (`molrs::compute::environment`): `BondOrientationalOrder`.

#![allow(clippy::type_complexity)]

use super::{collect_frames, collect_neighbors};
use crate::error::py_value_err;
use molrs::compute::{BondOrientationalOrder, LegacyCompute};
use molrs::core::Frame as CoreFrame;
use ndarray::Array1;
use numpy::{IntoPyArray, PyArray1, PyArray2};
use pyo3::prelude::*;
use pyo3::types::PyAny;

// ---------------------------------------------------------------------------
// BondOrientationalOrder
// ---------------------------------------------------------------------------

#[pyclass(module = "molrs.compute", name = "BondOrientationalOrder")]
pub struct PyBondOrientationalOrder {
    inner: BondOrientationalOrder,
}

#[pymethods]
impl PyBondOrientationalOrder {
    #[new]
    fn new(n_theta: usize, n_phi: usize) -> PyResult<Self> {
        Ok(Self {
            inner: BondOrientationalOrder::new(n_theta, n_phi).map_err(py_value_err)?,
        })
    }

    /// Returns per-frame `(raw_counts, bond_order, theta_edges, phi_edges)`.
    fn compute<'py>(
        &self,
        py: Python<'py>,
        frames: &Bound<'py, PyAny>,
        nlists: &Bound<'py, PyAny>,
    ) -> PyResult<
        Vec<(
            Bound<'py, PyArray2<u64>>,
            Bound<'py, PyArray2<f64>>,
            Bound<'py, PyArray1<f64>>,
            Bound<'py, PyArray1<f64>>,
        )>,
    > {
        let owned = collect_frames(frames)?;
        let refs: Vec<&CoreFrame> = owned.iter().collect();
        let neighbors = collect_neighbors(nlists)?;
        let results = self
            .inner
            .compute(&refs, &neighbors)
            .map_err(py_value_err)?;
        Ok(results
            .into_iter()
            .map(|r| {
                (
                    r.raw_counts.into_pyarray(py),
                    r.bond_order.into_pyarray(py),
                    Array1::from_vec(r.theta_edges).into_pyarray(py),
                    Array1::from_vec(r.phi_edges).into_pyarray(py),
                )
            })
            .collect())
    }
}

/// Register this domain's classes and functions.
pub(crate) fn register(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_class::<PyBondOrientationalOrder>()?;
    Ok(())
}
