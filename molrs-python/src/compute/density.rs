//! Density fields (`molrs::compute::density`): local and Gaussian density,
//! spatial distribution.

#![allow(clippy::type_complexity)]

use super::{collect_frames, collect_neighbors};
use crate::error::py_value_err;
use molrs::compute::{
    AtomGroups, GaussianDensity, GridSpec, LegacyCompute, LocalDensity, SpatialDistribution,
    SpatialDistributionResult,
};
use molrs::core::{Frame as CoreFrame, FrameAccess};
use molrs::op::F;
use ndarray::{Array1, Array2};
use numpy::{IntoPyArray, PyArray1, PyArray3, PyArray4, PyReadonlyArray2};
use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;
use pyo3::types::PyAny;

// ---------------------------------------------------------------------------
// LocalDensity
// ---------------------------------------------------------------------------

#[pyclass(module = "molrs.compute", name = "LocalDensity")]
pub struct PyLocalDensity {
    inner: LocalDensity,
}

#[pymethods]
impl PyLocalDensity {
    #[new]
    #[pyo3(signature = (r_max, diameter=0.0))]
    fn new(r_max: f64, diameter: f64) -> PyResult<Self> {
        let inner = LocalDensity::new(r_max)
            .map_err(py_value_err)?
            .with_diameter(diameter);
        Ok(Self { inner })
    }

    /// Returns `(n_neighbors, density)` ndarrays per frame.
    fn compute<'py>(
        &self,
        py: Python<'py>,
        frames: &Bound<'py, PyAny>,
        nlists: &Bound<'py, PyAny>,
    ) -> PyResult<Vec<(Bound<'py, PyArray1<f64>>, Bound<'py, PyArray1<f64>>)>> {
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
                    Array1::from_vec(r.n_neighbors).into_pyarray(py),
                    Array1::from_vec(r.density).into_pyarray(py),
                )
            })
            .collect())
    }
}

// ---------------------------------------------------------------------------
// GaussianDensity
// ---------------------------------------------------------------------------

#[pyclass(module = "molrs.compute", name = "GaussianDensity")]
pub struct PyGaussianDensity {
    inner: GaussianDensity,
}

#[pymethods]
impl PyGaussianDensity {
    #[new]
    fn new(nx: usize, ny: usize, nz: usize, sigma: f64) -> PyResult<Self> {
        Ok(Self {
            inner: GaussianDensity::new(nx, ny, nz, sigma).map_err(py_value_err)?,
        })
    }

    /// Returns a list of 3-D density grids `(nx, ny, nz)`, one per frame.
    fn compute<'py>(
        &self,
        py: Python<'py>,
        frames: &Bound<'py, PyAny>,
    ) -> PyResult<Vec<Bound<'py, PyArray3<f64>>>> {
        let owned = collect_frames(frames)?;
        let refs: Vec<&CoreFrame> = owned.iter().collect();
        let results = self.inner.compute(&refs, ()).map_err(py_value_err)?;
        Ok(results
            .into_iter()
            .map(|r| r.density.into_pyarray(py))
            .collect())
    }
}

// ---------------------------------------------------------------------------
// Spatial distribution function (SDF)
// ---------------------------------------------------------------------------

#[pyclass(module = "molrs.compute", name = "SpatialDistributionResult")]
pub struct PySpatialDistributionResult {
    inner: SpatialDistributionResult,
}

#[pymethods]
impl PySpatialDistributionResult {
    #[getter]
    fn counts<'py>(&self, py: Python<'py>) -> Bound<'py, PyArray3<f64>> {
        self.inner.counts.clone().into_pyarray(py)
    }
    #[getter]
    fn density<'py>(&self, py: Python<'py>) -> Bound<'py, PyArray3<f64>> {
        self.inner.density.clone().into_pyarray(py)
    }
    #[getter]
    fn g_sdf<'py>(&self, py: Python<'py>) -> Option<Bound<'py, PyArray3<f64>>> {
        self.inner
            .g_sdf
            .as_ref()
            .map(|g| g.clone().into_pyarray(py))
    }
    /// Per-voxel mean body-frame orientation `(nx, ny, nz, 3)` (only present
    /// when the frames carried an `"orientations"` topology block).
    #[getter]
    fn orientation<'py>(&self, py: Python<'py>) -> Option<Bound<'py, PyArray4<f64>>> {
        self.inner
            .orientation
            .as_ref()
            .map(|o| o.clone().into_pyarray(py))
    }
    #[getter]
    fn n(&self) -> [usize; 3] {
        self.inner.n
    }
    #[getter]
    fn extent(&self) -> [f64; 3] {
        self.inner.extent
    }
    #[getter]
    fn voxel_volume(&self) -> f64 {
        self.inner.voxel_volume
    }
    #[getter]
    fn n_frames(&self) -> usize {
        self.inner.n_frames
    }
}

/// Spatial distribution function: target-atom density on a body-fixed grid,
/// aligned to a reference template via Kabsch superposition.
#[pyclass(module = "molrs.compute", name = "SpatialDistribution")]
pub struct PySpatialDistribution {
    inner: SpatialDistribution,
}

#[pymethods]
impl PySpatialDistribution {
    /// Parameters
    /// ----------
    /// reference, target : list[int]
    ///     Reference (≥3, for alignment) and target (binned) atom indices.
    /// template : (R, 3) float array
    ///     Body-frame template coordinates for the reference atoms.
    /// n : (int, int, int)
    ///     Grid voxel counts per axis.
    /// extent : (float, float, float)
    ///     Grid extent (Å) per axis.
    /// bulk_density : float, optional
    ///     If set, also produce ``g_sdf = density / bulk_density``.
    ///
    /// A per-voxel mean orientation field is produced when the frames carry an
    /// ``"orientations"`` topology block — one `(head, tail)` atom pair per
    /// target atom — read at compute time (no constructor array).
    #[new]
    #[pyo3(signature = (reference, template, target, n, extent, bulk_density=None))]
    fn new(
        reference: Vec<usize>,
        template: PyReadonlyArray2<'_, f64>,
        target: Vec<usize>,
        n: [usize; 3],
        extent: [F; 3],
        bulk_density: Option<F>,
    ) -> PyResult<Self> {
        let tmpl: Array2<F> = template.as_array().to_owned();
        let grid = GridSpec { n, extent };
        let mut sdf =
            SpatialDistribution::new(reference, tmpl, target, grid).map_err(py_value_err)?;
        if let Some(rho) = bulk_density {
            sdf = sdf.with_bulk_density(rho);
        }
        Ok(Self { inner: sdf })
    }

    /// Accumulate the SDF over a trajectory (list of frames).
    ///
    /// If the first frame carries an `"orientations"` topology block (one
    /// `(head, tail)` atom pair per target atom, in `target` order), a per-voxel
    /// mean body-frame orientation of the unit `head − tail` vector is
    /// accumulated; otherwise the SDF is orientation-free.
    fn compute(&self, frames: &Bound<'_, PyAny>) -> PyResult<PySpatialDistributionResult> {
        let owned = collect_frames(frames)?;
        let refs: Vec<&CoreFrame> = owned.iter().collect();
        let first = refs
            .first()
            .copied()
            .ok_or_else(|| PyValueError::new_err("no frames provided"))?;
        let inner = if first.contains_block("orientations") {
            let groups = AtomGroups::from_frame(first, "orientations", 2).map_err(py_value_err)?;
            // The block stores `(head, tail)`; `with_orientation` expects
            // `(tail, head)` and forms the unit `head − tail` vector internally
            // (with minimum-image), so swap the endpoint order.
            let tuples: Vec<(usize, usize)> = (0..groups.len())
                .map(|i| {
                    let t = groups.tuple(i);
                    (t[1] as usize, t[0] as usize)
                })
                .collect();
            self.inner
                .clone()
                .with_orientation(tuples)
                .compute(&refs, ())
                .map_err(py_value_err)?
        } else {
            self.inner.compute(&refs, ()).map_err(py_value_err)?
        };
        Ok(PySpatialDistributionResult { inner })
    }
}

/// Register this domain's classes and functions.
pub(crate) fn register(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_class::<PyLocalDensity>()?;
    m.add_class::<PyGaussianDensity>()?;
    m.add_class::<PySpatialDistributionResult>()?;
    m.add_class::<PySpatialDistribution>()?;
    Ok(())
}
