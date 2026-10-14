//! Clustering (`molrs::compute::cluster`): `Cluster`, `ClusterResult`,
//! `ClusterProperties`.

use super::{collect_frames, collect_neighbors, was_batched};
use crate::error::py_value_err;
use molrs::compute::{Cluster, ClusterProperties, ClusterResult, LegacyCompute};
use molrs::core::Frame as CoreFrame;
use molrs::op::F;
use ndarray::{Array1, Array2, Array3};
use numpy::{IntoPyArray, PyArray1};
use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;
use pyo3::types::{PyAny, PyAnyMethods, PyDict, PyDictMethods};

// ---------------------------------------------------------------------------
// Cluster
// ---------------------------------------------------------------------------

/// Per-frame cluster assignment.
#[pyclass(module = "molrs.compute", name = "ClusterResult", from_py_object)]
#[derive(Clone)]
pub struct PyClusterResult {
    pub(crate) inner: ClusterResult,
}

#[pymethods]
impl PyClusterResult {
    #[getter]
    fn n_clusters(&self) -> usize {
        self.inner.n_clusters
    }

    #[getter]
    fn cluster_idx<'py>(&self, py: Python<'py>) -> Bound<'py, PyArray1<i64>> {
        self.inner.cluster_idx.clone().into_pyarray(py)
    }

    #[getter]
    fn cluster_sizes(&self) -> Vec<usize> {
        self.inner.cluster_sizes.clone()
    }

    /// The membership keys present in each cluster (freud's `cluster_keys`).
    /// Empty for spatial clustering; one key per cluster for `keys=`-based
    /// grouping.
    #[getter]
    fn cluster_keys(&self) -> Vec<Vec<u64>> {
        self.inner.cluster_keys.clone()
    }

    fn __repr__(&self) -> String {
        format!(
            "ClusterResult(n_clusters={}, largest={})",
            self.inner.n_clusters,
            self.inner.cluster_sizes.iter().max().unwrap_or(&0),
        )
    }
}

/// Distance-based cluster analysis.
#[pyclass(module = "molrs.compute", name = "Cluster")]
pub struct PyCluster {
    inner: Cluster,
}

#[pymethods]
impl PyCluster {
    #[new]
    fn new(min_cluster_size: usize) -> Self {
        Self {
            inner: Cluster::new(min_cluster_size),
        }
    }

    /// Compute one cluster result per input frame.
    ///
    /// Two modes (mirroring freud's `Cluster.compute`):
    /// * spatial — pass `nlists` (a `Neighbors` table per frame); particles within
    ///   the cutoff and transitively connected form a cluster.
    /// * by key — pass `keys` (one non-negative integer per atom, e.g. a
    ///   molecule id); all atoms sharing a key form one cluster, independent of
    ///   geometry. Robust for per-molecule properties (e.g. per-chain Rg) even
    ///   when molecules overlap or a bond exceeds any spatial cutoff. `nlists`
    ///   is then ignored.
    ///
    /// Returns a single `ClusterResult` when a single frame is passed, or a
    /// `list[ClusterResult]` when a list is passed.
    #[pyo3(signature = (frames, nlists=None, keys=None))]
    fn compute<'py>(
        &self,
        py: Python<'py>,
        frames: &Bound<'py, PyAny>,
        nlists: Option<&Bound<'py, PyAny>>,
        keys: Option<&Bound<'py, PyAny>>,
    ) -> PyResult<Py<PyAny>> {
        let batched = was_batched(frames);
        let owned = collect_frames(frames)?;
        let refs: Vec<&CoreFrame> = owned.iter().collect();
        let out = if let Some(keys_obj) = keys {
            let keys_i: Vec<i64> = keys_obj
                .extract()
                .map_err(|_| PyValueError::new_err("keys must be a 1-D sequence of integers"))?;
            let mut keys_u: Vec<u64> = Vec::with_capacity(keys_i.len());
            for k in keys_i {
                if k < 0 {
                    return Err(PyValueError::new_err("keys must be non-negative"));
                }
                keys_u.push(k as u64);
            }
            self.inner
                .compute_keyed(&refs, &keys_u)
                .map_err(py_value_err)?
        } else {
            let nlists = nlists.ok_or_else(|| {
                PyValueError::new_err(
                    "compute requires either nlists (spatial) or keys (group-by-key)",
                )
            })?;
            let nlists_vec = collect_neighbors(nlists)?;
            if nlists_vec.len() != refs.len() {
                return Err(PyValueError::new_err(format!(
                    "len(nlists)={} must equal len(frames)={}",
                    nlists_vec.len(),
                    refs.len()
                )));
            }
            self.inner
                .compute(&refs, &nlists_vec)
                .map_err(py_value_err)?
        };
        if !batched {
            let single = out.into_iter().next().unwrap();
            return Ok(Py::new(py, PyClusterResult { inner: single })?.into_any());
        }
        let wrapped: Vec<Py<PyClusterResult>> = out
            .into_iter()
            .map(|r| Py::new(py, PyClusterResult { inner: r }))
            .collect::<PyResult<_>>()?;
        Ok(pyo3::types::PyList::new(py, wrapped)?.into_any().unbind())
    }

    fn __repr__(&self) -> String {
        "Cluster(...)".to_string()
    }
}

// ---------------------------------------------------------------------------
// ClusterProperties
// ---------------------------------------------------------------------------

#[pyclass(module = "molrs.compute", name = "ClusterProperties")]
pub struct PyClusterProperties {
    inner: ClusterProperties,
}

#[pymethods]
impl PyClusterProperties {
    #[new]
    fn new() -> Self {
        Self {
            inner: ClusterProperties::new(),
        }
    }

    fn with_masses(&self, masses: Vec<f64>) -> Self {
        Self {
            inner: self.inner.clone().with_masses(&masses),
        }
    }

    /// Returns a dict per frame with keys
    /// `sizes`, `centers`, `centers_of_mass`, `cluster_masses`,
    /// `gyration_tensors`, `radii_of_gyration`.
    fn compute<'py>(
        &self,
        py: Python<'py>,
        frames: &Bound<'py, PyAny>,
        clusters: Vec<PyRef<'_, PyClusterResult>>,
    ) -> PyResult<Vec<Bound<'py, PyDict>>> {
        let owned = collect_frames(frames)?;
        let refs: Vec<&CoreFrame> = owned.iter().collect();
        let cl: Vec<_> = clusters.iter().map(|c| c.inner.clone()).collect();
        let results = self.inner.compute(&refs, &cl).map_err(py_value_err)?;
        results
            .into_iter()
            .map(|r| {
                let d = PyDict::new(py);
                d.set_item("sizes", r.sizes)?;
                d.set_item(
                    "centers",
                    Array2::from_shape_vec(
                        (r.centers.len(), 3),
                        r.centers.iter().flatten().copied().collect::<Vec<F>>(),
                    )
                    .map_err(py_value_err)?
                    .into_pyarray(py),
                )?;
                d.set_item(
                    "centers_of_mass",
                    Array2::from_shape_vec(
                        (r.centers_of_mass.len(), 3),
                        r.centers_of_mass
                            .iter()
                            .flatten()
                            .copied()
                            .collect::<Vec<F>>(),
                    )
                    .map_err(py_value_err)?
                    .into_pyarray(py),
                )?;
                d.set_item(
                    "cluster_masses",
                    Array1::from_vec(r.cluster_masses).into_pyarray(py),
                )?;
                let n = r.gyration_tensors.len();
                let mut g = Array3::<F>::zeros((n, 3, 3));
                for (c, t) in r.gyration_tensors.iter().enumerate() {
                    for a in 0..3 {
                        for b in 0..3 {
                            g[[c, a, b]] = t[a][b];
                        }
                    }
                }
                d.set_item("gyration_tensors", g.into_pyarray(py))?;
                d.set_item(
                    "radii_of_gyration",
                    Array1::from_vec(r.radii_of_gyration).into_pyarray(py),
                )?;
                Ok(d)
            })
            .collect()
    }
}

/// Register this domain's classes and functions.
pub(crate) fn register(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_class::<PyClusterResult>()?;
    m.add_class::<PyCluster>()?;
    m.add_class::<PyClusterProperties>()?;
    Ok(())
}
