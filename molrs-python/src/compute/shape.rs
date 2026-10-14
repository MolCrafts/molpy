//! Per-cluster shape descriptors (`molrs::compute::shape`): cluster centres,
//! centre of mass, gyration and inertia tensors, radius of gyration.

use super::cluster::PyClusterResult;
use super::{collect_frames, was_batched};
use crate::error::py_value_err;
use molrs::compute::{
    CenterOfMass, CenterOfMassResult, ClusterCenters, ClusterCentersResult, ClusterResult,
    GyrationTensor, InertiaTensor, LegacyCompute, RadiusOfGyration, RadiusOfGyrationResult,
};
use molrs::core::Frame as CoreFrame;
use molrs::op::F;
use numpy::{IntoPyArray, PyArray1, PyArray2, PyArrayDyn, PyReadonlyArray1};
use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;
use pyo3::types::{PyAny, PyAnyMethods};

// ---------------------------------------------------------------------------
// ClusterCenters
// ---------------------------------------------------------------------------

/// Geometric cluster centers for a single frame.
#[pyclass(
    module = "molrs.compute",
    name = "ClusterCentersResult",
    from_py_object
)]
#[derive(Clone)]
pub struct PyClusterCentersResult {
    pub(crate) inner: ClusterCentersResult,
}

#[pymethods]
impl PyClusterCentersResult {
    #[getter]
    fn centers<'py>(&self, py: Python<'py>) -> Bound<'py, PyArray2<f64>> {
        let nc = self.inner.centers.len();
        let flat: Vec<f64> = self
            .inner
            .centers
            .iter()
            .flat_map(|c| [c[0], c[1], c[2]])
            .collect();
        ndarray::Array2::from_shape_vec((nc, 3), flat)
            .expect("centers shape")
            .into_pyarray(py)
    }

    fn __len__(&self) -> usize {
        self.inner.centers.len()
    }

    fn __repr__(&self) -> String {
        format!(
            "ClusterCentersResult(n_clusters={})",
            self.inner.centers.len()
        )
    }
}

fn extract_cluster_vec(arg: &Bound<'_, PyAny>) -> PyResult<(bool, Vec<ClusterResult>)> {
    if let Ok(single) = arg.extract::<PyRef<'_, PyClusterResult>>() {
        return Ok((false, vec![single.inner.clone()]));
    }
    let list: Vec<PyRef<'_, PyClusterResult>> = arg.extract()?;
    Ok((true, list.iter().map(|r| r.inner.clone()).collect()))
}

fn extract_centers_vec(arg: &Bound<'_, PyAny>) -> PyResult<Vec<ClusterCentersResult>> {
    if let Ok(single) = arg.extract::<PyRef<'_, PyClusterCentersResult>>() {
        return Ok(vec![single.inner.clone()]);
    }
    let list: Vec<PyRef<'_, PyClusterCentersResult>> = arg.extract()?;
    Ok(list.iter().map(|r| r.inner.clone()).collect())
}

#[pyclass(module = "molrs.compute", name = "ClusterCenters")]
pub struct PyClusterCenters {
    inner: ClusterCenters,
}

#[pymethods]
impl PyClusterCenters {
    #[new]
    fn new() -> Self {
        Self {
            inner: ClusterCenters::new(),
        }
    }

    fn compute<'py>(
        &self,
        py: Python<'py>,
        frames: &Bound<'py, PyAny>,
        clusters: &Bound<'py, PyAny>,
    ) -> PyResult<Py<PyAny>> {
        let batched = was_batched(frames);
        let owned = collect_frames(frames)?;
        let refs: Vec<&CoreFrame> = owned.iter().collect();
        let (_, cl_vec) = extract_cluster_vec(clusters)?;
        if cl_vec.len() != refs.len() {
            return Err(PyValueError::new_err(
                "len(clusters) must equal len(frames)",
            ));
        }
        let out = self.inner.compute(&refs, &cl_vec).map_err(py_value_err)?;
        if !batched {
            let single = out.into_iter().next().unwrap();
            return Ok(Py::new(py, PyClusterCentersResult { inner: single })?.into_any());
        }
        let wrapped: Vec<Py<PyClusterCentersResult>> = out
            .into_iter()
            .map(|r| Py::new(py, PyClusterCentersResult { inner: r }))
            .collect::<PyResult<_>>()?;
        Ok(pyo3::types::PyList::new(py, wrapped)?.into_any().unbind())
    }

    fn __repr__(&self) -> String {
        "ClusterCenters()".to_string()
    }
}

// ---------------------------------------------------------------------------
// CenterOfMass
// ---------------------------------------------------------------------------

/// Per-frame mass-weighted cluster centers and total cluster masses.
#[pyclass(module = "molrs.compute", name = "CenterOfMassResult", from_py_object)]
#[derive(Clone)]
pub struct PyCenterOfMassResult {
    pub(crate) inner: CenterOfMassResult,
}

#[pymethods]
impl PyCenterOfMassResult {
    #[getter]
    fn centers_of_mass<'py>(&self, py: Python<'py>) -> Bound<'py, PyArray2<f64>> {
        let nc = self.inner.centers_of_mass.len();
        let flat: Vec<f64> = self
            .inner
            .centers_of_mass
            .iter()
            .flat_map(|c| [c[0], c[1], c[2]])
            .collect();
        ndarray::Array2::from_shape_vec((nc, 3), flat)
            .expect("com shape")
            .into_pyarray(py)
    }

    #[getter]
    fn cluster_masses<'py>(&self, py: Python<'py>) -> Bound<'py, PyArray1<f64>> {
        let v: Vec<f64> = self.inner.cluster_masses.to_vec();
        ndarray::Array1::from_vec(v).into_pyarray(py)
    }

    fn __repr__(&self) -> String {
        format!(
            "CenterOfMassResult(n_clusters={})",
            self.inner.centers_of_mass.len()
        )
    }
}

fn extract_com_vec(arg: &Bound<'_, PyAny>) -> PyResult<Vec<CenterOfMassResult>> {
    if let Ok(single) = arg.extract::<PyRef<'_, PyCenterOfMassResult>>() {
        return Ok(vec![single.inner.clone()]);
    }
    let list: Vec<PyRef<'_, PyCenterOfMassResult>> = arg.extract()?;
    Ok(list.iter().map(|r| r.inner.clone()).collect())
}

#[pyclass(module = "molrs.compute", name = "CenterOfMass")]
pub struct PyCenterOfMass {
    masses: Option<Vec<F>>,
}

#[pymethods]
impl PyCenterOfMass {
    #[new]
    #[pyo3(signature = (masses=None))]
    fn new(masses: Option<PyReadonlyArray1<'_, f64>>) -> Self {
        let masses = masses.map(|m| {
            m.as_slice()
                .unwrap()
                .iter()
                .map(|&v| v as F)
                .collect::<Vec<F>>()
        });
        Self { masses }
    }

    fn compute<'py>(
        &self,
        py: Python<'py>,
        frames: &Bound<'py, PyAny>,
        clusters: &Bound<'py, PyAny>,
    ) -> PyResult<Py<PyAny>> {
        let batched = was_batched(frames);
        let owned = collect_frames(frames)?;
        let refs: Vec<&CoreFrame> = owned.iter().collect();
        let (_, cl_vec) = extract_cluster_vec(clusters)?;
        if cl_vec.len() != refs.len() {
            return Err(PyValueError::new_err(
                "len(clusters) must equal len(frames)",
            ));
        }
        let calc = if let Some(ref ms) = self.masses {
            CenterOfMass::new().with_masses(ms)
        } else {
            CenterOfMass::new()
        };
        let out = calc.compute(&refs, &cl_vec).map_err(py_value_err)?;
        if !batched {
            let single = out.into_iter().next().unwrap();
            return Ok(Py::new(py, PyCenterOfMassResult { inner: single })?.into_any());
        }
        let wrapped: Vec<Py<PyCenterOfMassResult>> = out
            .into_iter()
            .map(|r| Py::new(py, PyCenterOfMassResult { inner: r }))
            .collect::<PyResult<_>>()?;
        Ok(pyo3::types::PyList::new(py, wrapped)?.into_any().unbind())
    }

    fn __repr__(&self) -> String {
        "CenterOfMass(...)".to_string()
    }
}

// ---------------------------------------------------------------------------
// GyrationTensor
// ---------------------------------------------------------------------------

fn tensor_list_into_pyarray<'py>(
    py: Python<'py>,
    _batched: bool,
    tensors: Vec<[[F; 3]; 3]>,
) -> Bound<'py, PyArrayDyn<f64>> {
    let n = tensors.len();
    let flat: Vec<f64> = tensors
        .iter()
        .flat_map(|t| t.iter().flat_map(|row| row.iter().copied()))
        .collect();
    ndarray::ArrayD::from_shape_vec(vec![n, 3, 3], flat)
        .expect("tensor shape")
        .into_pyarray(py)
}

/// Gyration tensor per cluster.
///
/// ``compute(frames, clusters, centers)``:
/// - single frame → shape `(n_clusters, 3, 3)` **but wrapped as a list of length 1**
///   when a list of frames is passed. For a single frame you get a `(n_clusters, 3, 3)` ndarray.
/// - list of frames → ndarray of shape `(n_frames, n_clusters, 3, 3)` only if all frames
///   have identical cluster counts; otherwise a Python list of per-frame ndarrays.
#[pyclass(module = "molrs.compute", name = "GyrationTensor")]
pub struct PyGyrationTensor {
    inner: GyrationTensor,
}

#[pymethods]
impl PyGyrationTensor {
    #[new]
    fn new() -> Self {
        Self {
            inner: GyrationTensor::new(),
        }
    }

    fn compute<'py>(
        &self,
        py: Python<'py>,
        frames: &Bound<'py, PyAny>,
        clusters: &Bound<'py, PyAny>,
        centers: &Bound<'py, PyAny>,
    ) -> PyResult<Py<PyAny>> {
        let batched = was_batched(frames);
        let owned = collect_frames(frames)?;
        let refs: Vec<&CoreFrame> = owned.iter().collect();
        let (_, cl_vec) = extract_cluster_vec(clusters)?;
        let cc_vec = extract_centers_vec(centers)?;
        if cl_vec.len() != refs.len() || cc_vec.len() != refs.len() {
            return Err(PyValueError::new_err(
                "clusters and centers must match len(frames)",
            ));
        }
        let out = self
            .inner
            .compute(&refs, (&cl_vec, &cc_vec))
            .map_err(py_value_err)?;
        if !batched {
            let tensors = out.into_iter().next().unwrap().0;
            return Ok(tensor_list_into_pyarray(py, false, tensors)
                .into_any()
                .unbind());
        }
        let arrays: Vec<Py<PyArrayDyn<f64>>> = out
            .into_iter()
            .map(|r| tensor_list_into_pyarray(py, false, r.0).unbind())
            .collect();
        Ok(pyo3::types::PyList::new(py, arrays)?.into_any().unbind())
    }

    fn __repr__(&self) -> String {
        "GyrationTensor()".to_string()
    }
}

// ---------------------------------------------------------------------------
// InertiaTensor
// ---------------------------------------------------------------------------

#[pyclass(module = "molrs.compute", name = "InertiaTensor")]
pub struct PyInertiaTensor {
    masses: Option<Vec<F>>,
}

#[pymethods]
impl PyInertiaTensor {
    #[new]
    #[pyo3(signature = (masses=None))]
    fn new(masses: Option<PyReadonlyArray1<'_, f64>>) -> Self {
        let masses = masses.map(|m| {
            m.as_slice()
                .unwrap()
                .iter()
                .map(|&v| v as F)
                .collect::<Vec<F>>()
        });
        Self { masses }
    }

    fn compute<'py>(
        &self,
        py: Python<'py>,
        frames: &Bound<'py, PyAny>,
        clusters: &Bound<'py, PyAny>,
        com: &Bound<'py, PyAny>,
    ) -> PyResult<Py<PyAny>> {
        let batched = was_batched(frames);
        let owned = collect_frames(frames)?;
        let refs: Vec<&CoreFrame> = owned.iter().collect();
        let (_, cl_vec) = extract_cluster_vec(clusters)?;
        let com_vec = extract_com_vec(com)?;
        if cl_vec.len() != refs.len() || com_vec.len() != refs.len() {
            return Err(PyValueError::new_err(
                "clusters and com must match len(frames)",
            ));
        }
        let calc = if let Some(ref ms) = self.masses {
            InertiaTensor::new().with_masses(ms)
        } else {
            InertiaTensor::new()
        };
        let out = calc
            .compute(&refs, (&cl_vec, &com_vec))
            .map_err(py_value_err)?;
        if !batched {
            let tensors = out.into_iter().next().unwrap().0;
            return Ok(tensor_list_into_pyarray(py, false, tensors)
                .into_any()
                .unbind());
        }
        let arrays: Vec<Py<PyArrayDyn<f64>>> = out
            .into_iter()
            .map(|r| tensor_list_into_pyarray(py, false, r.0).unbind())
            .collect();
        Ok(pyo3::types::PyList::new(py, arrays)?.into_any().unbind())
    }

    fn __repr__(&self) -> String {
        "InertiaTensor(...)".to_string()
    }
}

// ---------------------------------------------------------------------------
// RadiusOfGyration
// ---------------------------------------------------------------------------

#[pyclass(module = "molrs.compute", name = "RadiusOfGyration")]
pub struct PyRadiusOfGyration {
    masses: Option<Vec<F>>,
}

#[pymethods]
impl PyRadiusOfGyration {
    #[new]
    #[pyo3(signature = (masses=None))]
    fn new(masses: Option<PyReadonlyArray1<'_, f64>>) -> Self {
        let masses = masses.map(|m| {
            m.as_slice()
                .unwrap()
                .iter()
                .map(|&v| v as F)
                .collect::<Vec<F>>()
        });
        Self { masses }
    }

    /// Compute Rg per cluster.
    ///
    /// Returns a `(n_clusters,)` ndarray for a single frame, or a
    /// `(n_frames, n_clusters)` ndarray for a batch (clusters per frame
    /// must be identical; otherwise a Python list is returned).
    fn compute<'py>(
        &self,
        py: Python<'py>,
        frames: &Bound<'py, PyAny>,
        clusters: &Bound<'py, PyAny>,
        com: &Bound<'py, PyAny>,
    ) -> PyResult<Py<PyAny>> {
        let batched = was_batched(frames);
        let owned = collect_frames(frames)?;
        let refs: Vec<&CoreFrame> = owned.iter().collect();
        let (_, cl_vec) = extract_cluster_vec(clusters)?;
        let com_vec = extract_com_vec(com)?;
        if cl_vec.len() != refs.len() || com_vec.len() != refs.len() {
            return Err(PyValueError::new_err(
                "clusters and com must match len(frames)",
            ));
        }
        let calc = if let Some(ref ms) = self.masses {
            RadiusOfGyration::new().with_masses(ms)
        } else {
            RadiusOfGyration::new()
        };
        let out: Vec<RadiusOfGyrationResult> = calc
            .compute(&refs, (&cl_vec, &com_vec))
            .map_err(py_value_err)?;

        if !batched {
            let v: Vec<f64> = out.into_iter().next().unwrap().0.to_vec();
            return Ok(ndarray::Array1::from_vec(v)
                .into_pyarray(py)
                .into_any()
                .unbind());
        }

        // Try a rectangular (n_frames, nc) packing if widths agree.
        let widths: Vec<usize> = out.iter().map(|r| r.0.len()).collect();
        let uniform = widths.iter().all(|&w| w == widths[0]);
        if uniform {
            let n_frames = out.len();
            let nc = widths.first().copied().unwrap_or(0);
            let mut flat: Vec<f64> = Vec::with_capacity(n_frames * nc);
            for row in &out {
                flat.extend(row.0.iter().copied());
            }
            let arr =
                ndarray::Array2::from_shape_vec((n_frames, nc), flat).expect("rg batch shape");
            return Ok(arr.into_pyarray(py).into_any().unbind());
        }

        let arrays: Vec<Py<PyArray1<f64>>> = out
            .into_iter()
            .map(|r| {
                let v: Vec<f64> = r.0.to_vec();
                ndarray::Array1::from_vec(v).into_pyarray(py).unbind()
            })
            .collect();
        Ok(pyo3::types::PyList::new(py, arrays)?.into_any().unbind())
    }

    fn __repr__(&self) -> String {
        "RadiusOfGyration(...)".to_string()
    }
}

/// Register this domain's classes and functions.
pub(crate) fn register(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_class::<PyClusterCentersResult>()?;
    m.add_class::<PyClusterCenters>()?;
    m.add_class::<PyCenterOfMassResult>()?;
    m.add_class::<PyCenterOfMass>()?;
    m.add_class::<PyGyrationTensor>()?;
    m.add_class::<PyInertiaTensor>()?;
    m.add_class::<PyRadiusOfGyration>()?;
    Ok(())
}
