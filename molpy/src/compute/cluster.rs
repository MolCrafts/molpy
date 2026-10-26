//! Clusters (`molrs::compute::Cluster`): connected components of a pair
//! table, and their output `ClusterResult`, a partition.

use molrs::compute::{Cluster, ClusterInput, ClusterResult, Compute, Partition};
use pyo3::prelude::*;

use super::args;
use super::error::Result;
use super::output::py_result;
use super::spec::described;
use crate::core::neighborlist::PyNeighbors;

py_result! {
    /// Connected clusters (``molrs::compute::ClusterResult``); a partition.
    ///
    /// ``cluster_idx[i]`` is item ``i``'s cluster, ``cluster_sizes`` and
    /// ``cluster_keys`` are per cluster. Pass it as the ``partition=`` of a
    /// group compute (``CenterOfMass``, ``GyrationTensor``, …).
    PyClusterResult(ClusterResult) as "ClusterResult" {
        cluster_idx, n_clusters, cluster_sizes, cluster_keys
    }
}

#[pymethods]
impl PyClusterResult {
    /// The size of every group, as ``[(group, size), …]`` by ascending group.
    fn sizes(&self) -> Vec<(u64, usize)> {
        self.inner.sizes().into_iter().collect()
    }
}

/// Connected components of a neighbour table (``molrs::compute::Cluster``).
///
/// Parameters
/// ----------
/// min_cluster_size : int, optional
///     Drop clusters smaller than this; ``None`` keeps the core default.
///
/// ``compute(pairs=, merge=None)`` clusters the items joined by the pairs of
/// the ``Neighbors`` table ``pairs``; ``merge`` (a partition) joins items of
/// the same group first. Returns a ``ClusterResult``.
#[pyclass(module = "molpy.compute", name = "Cluster", frozen)]
pub(crate) struct PyCluster {
    inner: Cluster,
}

#[pymethods]
impl PyCluster {
    #[new]
    #[pyo3(signature = (*, min_cluster_size = None))]
    fn new(min_cluster_size: Option<usize>) -> Result<Self> {
        let mut inner = Cluster::new();
        if let Some(min_cluster_size) = min_cluster_size {
            inner = inner.with_min_cluster_size(min_cluster_size)?;
        }
        Ok(Self { inner })
    }

    #[pyo3(signature = (*, pairs, merge = None))]
    fn compute<'py>(
        &self,
        pairs: PyRef<'py, PyNeighbors>,
        merge: Option<Bound<'py, PyAny>>,
    ) -> Result<PyClusterResult> {
        let merge = merge
            .as_ref()
            .map(args::PartitionArg::extract)
            .transpose()?;
        let out = self.inner.compute(ClusterInput {
            pairs: &pairs.inner,
            merge: merge.as_ref().map(args::PartitionArg::partition),
        })?;
        Ok(out.into())
    }
}

described!(PyCluster, Cluster);

pub(crate) fn register(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_class::<PyClusterResult>()?;
    m.add_class::<PyCluster>()?;
    Ok(())
}
