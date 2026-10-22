//! Clustering of feature rows (`molrs::compute::Kmeans`): `Kmeans` and its
//! output `KmeansResult`, a partition.

use molrs::compute::{Compute, Kmeans, KmeansInput, KmeansResult};
use pyo3::prelude::*;

use super::args;
use super::error::Result;
use super::output::py_result;
use super::spec::described;

py_result! {
    /// The k-means partition (``molrs::compute::KmeansResult``): one label per
    /// row and the ``(k, n_features)`` centroids. A partition.
    PyKmeansResult(KmeansResult) as "KmeansResult" { labels, centroids }
}

/// K-means clustering of the rows of a feature matrix
/// (``molrs::compute::Kmeans``).
///
/// Parameters
/// ----------
/// k : int
///     The number of clusters.
/// max_iter : int, optional
///     The iteration cap; ``None`` keeps the core default.
/// seed : int, optional
///     The initialisation seed; ``None`` keeps the core default.
///
/// ``compute(features=)`` clusters a 2-D float array or a descriptor result
/// (``SteinhardtResult``, ``LocalDescriptorsResult``, ``PcaResult``) and
/// returns a ``KmeansResult``.
#[pyclass(module = "molpy.compute", name = "Kmeans", frozen)]
pub(crate) struct PyKmeans {
    inner: Kmeans,
}

#[pymethods]
impl PyKmeans {
    #[new]
    #[pyo3(signature = (k, *, max_iter = None, seed = None))]
    fn new(k: usize, max_iter: Option<usize>, seed: Option<u64>) -> Result<Self> {
        let mut inner = Kmeans::new(k)?;
        if let Some(max_iter) = max_iter {
            inner = inner.with_max_iter(max_iter)?;
        }
        if let Some(seed) = seed {
            inner = inner.with_seed(seed);
        }
        Ok(Self { inner })
    }

    #[pyo3(signature = (*, features))]
    fn compute<'py>(&self, features: &Bound<'py, PyAny>) -> Result<PyKmeansResult> {
        let features = args::FeatureArg::extract(features)?;
        let out = self.inner.compute(KmeansInput {
            features: features.features(),
        })?;
        Ok(out.into())
    }
}

described!(PyKmeans, Kmeans);

pub(crate) fn register(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_class::<PyKmeansResult>()?;
    m.add_class::<PyKmeans>()?;
    Ok(())
}
