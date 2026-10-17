//! Radial distribution function (`molrs::compute::Rdf`) and its output
//! `RdfResult`.

use molrs::compute::{Accumulate, Rdf, RdfAccumulator, RdfResult, Stream};
use pyo3::prelude::*;

use super::args::PairSourceArg;
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

/// Radial distribution function g(r) (``molrs::compute::Rdf``) — a stream
/// only: :meth:`accumulator` hands out an :class:`RdfAccumulator`; push one
/// pair source per frame and :meth:`RdfAccumulator.finish` reads the
/// ``RdfResult``.
///
/// Parameters
/// ----------
/// n_bins : int
/// r_max, r_min : float
///     The binned range ``[r_min, r_max)``.
/// dimensionality : int, optional
///     ``2`` or ``3``; ``None`` keeps the core default.
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

    /// A fresh accumulator configured like this ``Rdf``
    /// (``Stream::accumulator``).
    fn accumulator(&self) -> PyRdfAccumulator {
        PyRdfAccumulator {
            inner: self.inner.accumulator(),
        }
    }
}

/// The streaming g(r) accumulator (``molrs::compute::RdfAccumulator``), made
/// by :meth:`Rdf.accumulator`.
///
/// :meth:`push` folds one frame's pair source; :meth:`finish` normalizes the
/// running histogram into an ``RdfResult`` without consuming it, so a run can
/// be inspected and continued. Memory is ``O(n_bins)``.
#[pyclass(module = "molrs.compute", name = "RdfAccumulator")]
pub(crate) struct PyRdfAccumulator {
    inner: RdfAccumulator,
}

#[pymethods]
impl PyRdfAccumulator {
    /// Fold one frame's pairs — a ``Neighbors`` table or a built
    /// ``NeighborList``, carrying its box and ``dist_sq`` — into the running
    /// histogram. A rejected frame leaves the state unchanged.
    ///
    /// Raises
    /// ------
    /// ValueError
    ///     The pairing (self vs cross query) differs from frame 0's, the
    ///     source has no box or no ``dist_sq``, the box volume is not
    ///     positive, or a distance is not finite.
    fn push(&mut self, pairs: &Bound<'_, PyAny>) -> Result<()> {
        let pairs = PairSourceArg::extract(pairs)?;
        self.inner.push(pairs.source()?)?;
        Ok(())
    }

    /// The g(r) over every frame pushed so far.
    fn finish(&self) -> Result<PyRdfResult> {
        Ok(self.inner.finish()?.into())
    }

    /// Number of frames pushed so far.
    #[getter]
    fn n_frames(&self) -> usize {
        self.inner.n_frames()
    }
}

described!(PyRdf, Rdf);

pub(crate) fn register(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_class::<PyRdfResult>()?;
    m.add_class::<PyRdf>()?;
    m.add_class::<PyRdfAccumulator>()?;
    Ok(())
}
