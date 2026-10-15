//! Decomposition of a feature matrix (`molrs::compute::Pca`): `Pca` and its
//! output `PcaResult`, itself a feature matrix.

use molrs::compute::{Compute, Pca, PcaInput, PcaResult};
use pyo3::prelude::*;

use super::args;
use super::error::Result;
use super::output::py_result;
use super::spec::described;

py_result! {
    /// Principal components (``molrs::compute::PcaResult``): the
    /// ``(n_rows, n_components)`` projected ``coords`` and the ``variance``
    /// of each component. A feature matrix.
    PyPcaResult(PcaResult) as "PcaResult" { coords, variance }
}

/// Principal component analysis of the rows of a feature matrix
/// (``molrs::compute::Pca``).
///
/// Parameters
/// ----------
/// n_components : int
///     The number of components kept.
///
/// ``compute(features=)`` decomposes a 2-D float array or a descriptor result
/// (``SteinhardtResult``, ``LocalDescriptorsResult``, ``PcaResult``) and
/// returns a ``PcaResult``.
#[pyclass(module = "molrs.compute", name = "Pca", frozen)]
pub(crate) struct PyPca {
    inner: Pca,
}

#[pymethods]
impl PyPca {
    #[new]
    fn new(n_components: usize) -> Result<Self> {
        Ok(Self {
            inner: Pca::new(n_components)?,
        })
    }

    #[pyo3(signature = (*, features))]
    fn compute<'py>(&self, features: &Bound<'py, PyAny>) -> Result<PyPcaResult> {
        let features = args::FeatureArg::extract(features)?;
        let out = self.inner.compute(PcaInput {
            features: features.features(),
        })?;
        Ok(out.into())
    }
}

described!(PyPca, Pca);

pub(crate) fn register(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_class::<PyPcaResult>()?;
    m.add_class::<PyPca>()?;
    Ok(())
}
