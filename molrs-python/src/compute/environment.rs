//! Local environments (`molrs::compute::environment`):
//! `BondOrientationalOrder`, `LocalDescriptors`, `LocalBondProjection`,
//! `EnvironmentMatch`, `AngularSeparationGlobal` and
//! `AngularSeparationNeighbor`.

use crate::array::ArrayArg2;
use molrs::compute::{
    AngularSeparationGlobal, AngularSeparationGlobalInput, AngularSeparationGlobalResult,
    AngularSeparationNeighbor, AngularSeparationNeighborInput, AngularSeparationNeighborResult,
    BondOrientationalOrder, BondOrientationalOrderInput, BondOrientationalOrderResult, Compute,
    EnvironmentMatch, EnvironmentMatchInput, EnvironmentMatchResult, FeatureMatrix,
    LocalBondProjection, LocalBondProjectionInput, LocalBondProjectionResult, LocalDescriptors,
    LocalDescriptorsInput, LocalDescriptorsResult,
};
use ndarray::Array2;
use pyo3::prelude::*;
use pyo3::types::PyComplex;

use super::error::Result;
use super::output::{ToPy, py_result};
use super::spec::described;
use crate::core::neighborlist::PyNeighbors;

py_result! {
    /// Environment clusters (``molrs::compute::EnvironmentMatchResult``); a
    /// partition.
    PyEnvironmentMatchResult(EnvironmentMatchResult) as "EnvironmentMatchResult" {
        cluster_idx, n_clusters, fingerprints
    }
}

py_result! {
    /// Per-bond spherical-harmonic descriptors
    /// (``molrs::compute::LocalDescriptorsResult``); a feature matrix.
    PyLocalDescriptorsResult(LocalDescriptorsResult) as "LocalDescriptorsResult" { l_max, n_sphs }
}

#[pymethods]
impl PyLocalDescriptorsResult {
    /// The ``(n_bonds, 2 * n_sphs)`` feature matrix (real and imaginary parts
    /// interleaved), one row per bond.
    #[getter]
    fn features<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyAny>> {
        let features = &self.inner;
        let mut matrix = Array2::zeros((features.n_rows(), features.n_cols()));
        for (row, mut out) in matrix.rows_mut().into_iter().enumerate() {
            out.assign(&features.row(row));
        }
        matrix.to_py(py)
    }

    /// The descriptor of bond ``pair`` at ``(l, m)``, or ``None`` out of range.
    fn descriptor<'py>(
        &self,
        py: Python<'py>,
        pair: usize,
        l: u32,
        m: i32,
    ) -> Option<Bound<'py, PyComplex>> {
        self.inner
            .descriptor(pair, l, m)
            .map(|z| PyComplex::from_doubles(py, z.re, z.im))
    }
}

py_result! {
    /// The bond-orientation histogram on the sphere
    /// (``molrs::compute::BondOrientationalOrderResult``).
    PyBondOrientationalOrderResult(BondOrientationalOrderResult) as "BondOrientationalOrderResult" {
        bond_order, raw_counts, theta_edges, phi_edges
    }
}

py_result! {
    /// Bond projections onto the given axes
    /// (``molrs::compute::LocalBondProjectionResult``).
    PyLocalBondProjectionResult(LocalBondProjectionResult) as "LocalBondProjectionResult" {
        projections
    }
}

py_result! {
    /// Angles between query and reference orientations
    /// (``molrs::compute::AngularSeparationGlobalResult``).
    PyAngularSeparationGlobalResult(AngularSeparationGlobalResult) as "AngularSeparationGlobalResult" {
        angles
    }
}

py_result! {
    /// Angles between neighbouring orientations
    /// (``molrs::compute::AngularSeparationNeighborResult``).
    PyAngularSeparationNeighborResult(AngularSeparationNeighborResult) as "AngularSeparationNeighborResult" {
        angles
    }
}

/// Bond-orientational order diagram on a ``(theta, phi)`` grid
/// (``molrs::compute::BondOrientationalOrder``).
///
/// ``compute(neighbors=)``.
#[pyclass(module = "molrs.compute", name = "BondOrientationalOrder", frozen)]
pub(crate) struct PyBondOrientationalOrder {
    inner: BondOrientationalOrder,
}

#[pymethods]
impl PyBondOrientationalOrder {
    #[new]
    fn new(n_theta: usize, n_phi: usize) -> Result<Self> {
        Ok(Self {
            inner: BondOrientationalOrder::new(n_theta, n_phi)?,
        })
    }

    #[pyo3(signature = (*, neighbors))]
    fn compute(&self, neighbors: PyRef<'_, PyNeighbors>) -> Result<PyBondOrientationalOrderResult> {
        let out = self.inner.compute(BondOrientationalOrderInput {
            neighbors: &neighbors.inner,
        })?;
        Ok(out.into())
    }
}

described!(PyBondOrientationalOrder, BondOrientationalOrder);

/// Spherical-harmonic descriptors of every bond up to ``l_max``
/// (``molrs::compute::LocalDescriptors``).
///
/// ``compute(neighbors=)``.
#[pyclass(module = "molrs.compute", name = "LocalDescriptors", frozen)]
pub(crate) struct PyLocalDescriptors {
    inner: LocalDescriptors,
}

#[pymethods]
impl PyLocalDescriptors {
    #[new]
    fn new(l_max: u32) -> Self {
        Self {
            inner: LocalDescriptors::new(l_max),
        }
    }

    #[pyo3(signature = (*, neighbors))]
    fn compute(&self, neighbors: PyRef<'_, PyNeighbors>) -> Result<PyLocalDescriptorsResult> {
        let out = self.inner.compute(LocalDescriptorsInput {
            neighbors: &neighbors.inner,
        })?;
        Ok(out.into())
    }
}

described!(PyLocalDescriptors, LocalDescriptors);

/// Projections of each bond onto ``(M, 3)`` axes, in the particle frame when
/// ``(N, 4)`` orientations are given (``molrs::compute::LocalBondProjection``).
///
/// ``compute(neighbors=, proj_vectors=, orientations=None)``.
#[pyclass(module = "molrs.compute", name = "LocalBondProjection", frozen)]
pub(crate) struct PyLocalBondProjection {
    inner: LocalBondProjection,
}

#[pymethods]
impl PyLocalBondProjection {
    #[new]
    fn new() -> Self {
        Self {
            inner: LocalBondProjection::new(),
        }
    }

    #[pyo3(signature = (*, neighbors, proj_vectors, orientations = None))]
    fn compute(
        &self,
        neighbors: PyRef<'_, PyNeighbors>,
        proj_vectors: ArrayArg2<'_, f64>,
        orientations: Option<ArrayArg2<'_, f64>>,
    ) -> Result<PyLocalBondProjectionResult> {
        let out = self.inner.compute(LocalBondProjectionInput {
            neighbors: &neighbors.inner,
            proj_vectors: proj_vectors.as_array(),
            orientations: orientations.as_ref().map(|o| o.as_array()),
        })?;
        Ok(out.into())
    }
}

described!(PyLocalBondProjection, LocalBondProjection);

/// Cluster particles whose neighbour environments match within
/// ``rmsd_threshold`` (``molrs::compute::EnvironmentMatch``).
///
/// Parameters
/// ----------
/// rmsd_threshold : float
/// registration : bool, optional
/// max_neighbors_for_registration : int, optional
///
/// ``compute(neighbors=)`` returns an ``EnvironmentMatchResult`` (a partition).
#[pyclass(module = "molrs.compute", name = "EnvironmentMatch", frozen)]
pub(crate) struct PyEnvironmentMatch {
    inner: EnvironmentMatch,
}

#[pymethods]
impl PyEnvironmentMatch {
    #[new]
    #[pyo3(signature = (rmsd_threshold, *, registration = None, max_neighbors_for_registration = None))]
    fn new(
        rmsd_threshold: f64,
        registration: Option<bool>,
        max_neighbors_for_registration: Option<usize>,
    ) -> Result<Self> {
        let mut inner = EnvironmentMatch::new(rmsd_threshold)?;
        if let Some(registration) = registration {
            inner = inner.with_registration(registration);
        }
        if let Some(n) = max_neighbors_for_registration {
            inner = inner.with_max_neighbors_for_registration(n)?;
        }
        Ok(Self { inner })
    }

    #[pyo3(signature = (*, neighbors))]
    fn compute(&self, neighbors: PyRef<'_, PyNeighbors>) -> Result<PyEnvironmentMatchResult> {
        let out = self.inner.compute(EnvironmentMatchInput {
            neighbors: &neighbors.inner,
        })?;
        Ok(out.into())
    }
}

described!(PyEnvironmentMatch, EnvironmentMatch);

/// Angles between ``(N, 4)`` query quaternions and ``(M, 4)`` reference ones
/// (``molrs::compute::AngularSeparationGlobal``).
///
/// ``compute(query=, reference=)``.
#[pyclass(module = "molrs.compute", name = "AngularSeparationGlobal", frozen)]
pub(crate) struct PyAngularSeparationGlobal {
    inner: AngularSeparationGlobal,
}

#[pymethods]
impl PyAngularSeparationGlobal {
    #[new]
    #[pyo3(signature = (*, equivalent_orientations = None))]
    fn new(equivalent_orientations: Option<bool>) -> Self {
        let mut inner = AngularSeparationGlobal::new();
        if let Some(on) = equivalent_orientations {
            inner = inner.with_equivalent_orientations(on);
        }
        Self { inner }
    }

    #[pyo3(signature = (*, query, reference))]
    fn compute(
        &self,
        query: ArrayArg2<'_, f64>,
        reference: ArrayArg2<'_, f64>,
    ) -> Result<PyAngularSeparationGlobalResult> {
        let out = self.inner.compute(AngularSeparationGlobalInput {
            query: query.as_array(),
            reference: reference.as_array(),
        })?;
        Ok(out.into())
    }
}

described!(PyAngularSeparationGlobal, AngularSeparationGlobal);

/// Angles between the orientations of neighbouring particles
/// (``molrs::compute::AngularSeparationNeighbor``).
///
/// ``compute(neighbors=, query_orientations=, point_orientations=)``.
#[pyclass(module = "molrs.compute", name = "AngularSeparationNeighbor", frozen)]
pub(crate) struct PyAngularSeparationNeighbor {
    inner: AngularSeparationNeighbor,
}

#[pymethods]
impl PyAngularSeparationNeighbor {
    #[new]
    #[pyo3(signature = (*, equivalent_orientations = None))]
    fn new(equivalent_orientations: Option<bool>) -> Self {
        let mut inner = AngularSeparationNeighbor::new();
        if let Some(on) = equivalent_orientations {
            inner = inner.with_equivalent_orientations(on);
        }
        Self { inner }
    }

    #[pyo3(signature = (*, neighbors, query_orientations, point_orientations))]
    fn compute(
        &self,
        neighbors: PyRef<'_, PyNeighbors>,
        query_orientations: ArrayArg2<'_, f64>,
        point_orientations: ArrayArg2<'_, f64>,
    ) -> Result<PyAngularSeparationNeighborResult> {
        let out = self.inner.compute(AngularSeparationNeighborInput {
            neighbors: &neighbors.inner,
            query_orientations: query_orientations.as_array(),
            point_orientations: point_orientations.as_array(),
        })?;
        Ok(out.into())
    }
}

described!(PyAngularSeparationNeighbor, AngularSeparationNeighbor);

pub(crate) fn register(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_class::<PyEnvironmentMatchResult>()?;
    m.add_class::<PyLocalDescriptorsResult>()?;
    m.add_class::<PyBondOrientationalOrderResult>()?;
    m.add_class::<PyLocalBondProjectionResult>()?;
    m.add_class::<PyAngularSeparationGlobalResult>()?;
    m.add_class::<PyAngularSeparationNeighborResult>()?;
    m.add_class::<PyBondOrientationalOrder>()?;
    m.add_class::<PyLocalDescriptors>()?;
    m.add_class::<PyLocalBondProjection>()?;
    m.add_class::<PyEnvironmentMatch>()?;
    m.add_class::<PyAngularSeparationGlobal>()?;
    m.add_class::<PyAngularSeparationNeighbor>()?;
    Ok(())
}
