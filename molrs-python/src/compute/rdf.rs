//! Radial distribution function (`molrs::compute::Rdf`) and its output
//! `RdfResult`.

use molrs::compute::{Compute, Rdf, RdfResult};
use pyo3::prelude::*;

use super::args::{self, PairSourceArg};
use super::error::Result;
use super::output::py_result;
use super::spec::described;

py_result! {
    /// A radial distribution function (``molrs::compute::RdfResult``):
    /// ``rdf`` and the cumulative count ``n_r`` on the bins, with the
    /// normalisation it used.
    PyRdfResult(RdfResult) as "RdfResult" {
        bin_edges,
        bin_centers,
        rdf,
        n_r,
        n_points,
        n_query_points,
        self_query,
        volume,
        r_min,
        n_frames,
        dimensionality,
    }
}

/// Radial distribution function g(r) (``molrs::compute::Rdf``).
///
/// Parameters
/// ----------
/// n_bins : int
/// r_max, r_min : float
///     The binned range ``[r_min, r_max)``.
/// dimensionality : int, optional
///     ``2`` or ``3``; ``None`` keeps the core default.
///
/// ``compute(pairs=)`` folds one pair source per frame — a ``Neighbors``
/// table or a built ``NeighborList`` — into one ``RdfResult``.
#[pyclass(module = "molrs.compute", name = "Rdf", frozen)]
pub(crate) struct PyRdf {
    inner: Rdf,
}

#[pymethods]
impl PyRdf {
    #[new]
    #[pyo3(signature = (n_bins, r_max, r_min, *, dimensionality = None))]
    fn new(n_bins: usize, r_max: f64, r_min: f64, dimensionality: Option<u8>) -> Result<Self> {
        let mut inner = Rdf::new(n_bins, r_max, r_min)?;
        if let Some(dimensionality) = dimensionality {
            inner = inner.with_dimensionality(dimensionality)?;
        }
        Ok(Self { inner })
    }

    #[pyo3(signature = (*, pairs))]
    fn compute(&self, pairs: Vec<Bound<'_, PyAny>>) -> Result<PyRdfResult> {
        let pairs = PairSourceArg::extract_all(&pairs)?;
        let frames = args::sources(&pairs)?;
        Ok(self.inner.compute(&frames)?.into())
    }
}

described!(PyRdf, Rdf);

pub(crate) fn register(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_class::<PyRdfResult>()?;
    m.add_class::<PyRdf>()?;
    Ok(())
}
