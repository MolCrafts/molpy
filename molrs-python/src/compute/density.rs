//! Densities and pair correlations on bins and grids
//! (`molrs::compute::density`): `CorrelationFunction`, `LocalDensity`,
//! `GaussianDensity`, `SphereVoxelization` (one snapshot each) and
//! `SpatialDistribution` (folded over frames), with their outputs.

use molrs::compute::{
    Compute, CorrelationFunction, CorrelationFunctionInput, CorrelationFunctionResult,
    GaussianDensity, GaussianDensityInput, GaussianDensityResult, LocalDensity, LocalDensityInput,
    LocalDensityResult, SpatialDistribution, SpatialDistributionFrame, SpatialDistributionResult,
    SphereVoxelization, SphereVoxelizationInput, SphereVoxelizationResult,
};
use numpy::{PyReadonlyArray1, PyReadonlyArray2};
use pyo3::prelude::*;

use super::args::PairSourceArg;
use super::distribution::{frame_boxes, same_frames};
use super::error::Result;
use super::output::py_result;
use super::spec::described;
use super::values::PyRegularGrid;
use crate::core::simbox::PyBox;

py_result! {
    /// A pair correlation of per-point values on radial bins
    /// (``molrs::compute::CorrelationFunctionResult``).
    PyCorrelationFunctionResult(CorrelationFunctionResult) as "CorrelationFunctionResult" {
        bin_edges,
        bin_centers,
        bin_counts,
        correlation,
    }
}

py_result! {
    /// Per-point neighbour count and local number density
    /// (``molrs::compute::LocalDensityResult``).
    PyLocalDensityResult(LocalDensityResult) as "LocalDensityResult" { n_neighbors, density }
}

py_result! {
    /// A Gaussian-smeared density on a grid
    /// (``molrs::compute::GaussianDensityResult``).
    PyGaussianDensityResult(GaussianDensityResult) as "GaussianDensityResult" { density, grid }
}

py_result! {
    /// Occupied voxels of spheres on a grid
    /// (``molrs::compute::SphereVoxelizationResult``).
    PySphereVoxelizationResult(SphereVoxelizationResult) as "SphereVoxelizationResult" {
        voxels,
        raw_counts,
        grid,
    }
}

py_result! {
    /// A spatial distribution in a reference body frame
    /// (``molrs::compute::SpatialDistributionResult``).
    PySpatialDistributionResult(SpatialDistributionResult) as "SpatialDistributionResult" {
        counts,
        density,
        g_sdf,
        orientation,
        grid,
        voxel_volume,
        n_frames,
    }
}

/// Radial correlation of two per-point value arrays over a pair source
/// (``molrs::compute::CorrelationFunction``).
///
/// Parameters
/// ----------
/// n_bins : int
/// r_max, r_min : float
///
/// ``compute(pairs=, values_a=, values_b=)``: ``pairs`` is a ``Neighbors`` or
/// a ``NeighborList``; ``values_a`` / ``values_b`` are per query / reference
/// point.
#[pyclass(module = "molrs.compute", name = "CorrelationFunction", frozen)]
pub(crate) struct PyCorrelationFunction {
    inner: CorrelationFunction,
}

#[pymethods]
impl PyCorrelationFunction {
    #[new]
    fn new(n_bins: usize, r_max: f64, r_min: f64) -> Result<Self> {
        Ok(Self {
            inner: CorrelationFunction::new(n_bins, r_max, r_min)?,
        })
    }

    #[pyo3(signature = (*, pairs, values_a, values_b))]
    fn compute(
        &self,
        pairs: &Bound<'_, PyAny>,
        values_a: PyReadonlyArray1<'_, f64>,
        values_b: PyReadonlyArray1<'_, f64>,
    ) -> Result<PyCorrelationFunctionResult> {
        let pairs = PairSourceArg::extract(pairs)?;
        let out = self.inner.compute(CorrelationFunctionInput {
            pairs: pairs.source()?,
            values_a: values_a.as_array(),
            values_b: values_b.as_array(),
        })?;
        Ok(out.into())
    }
}

described!(PyCorrelationFunction, CorrelationFunction);

/// Local number density within ``r_max`` of each point
/// (``molrs::compute::LocalDensity``).
///
/// Parameters
/// ----------
/// r_max : float
/// diameter : float, optional
///     The particle diameter smoothing the sphere edge.
///
/// ``compute(pairs=)``: a ``Neighbors`` or a ``NeighborList``.
#[pyclass(module = "molrs.compute", name = "LocalDensity", frozen)]
pub(crate) struct PyLocalDensity {
    inner: LocalDensity,
}

#[pymethods]
impl PyLocalDensity {
    #[new]
    #[pyo3(signature = (r_max, *, diameter = None))]
    fn new(r_max: f64, diameter: Option<f64>) -> Result<Self> {
        let mut inner = LocalDensity::new(r_max)?;
        if let Some(diameter) = diameter {
            inner = inner.with_diameter(diameter)?;
        }
        Ok(Self { inner })
    }

    #[pyo3(signature = (*, pairs))]
    fn compute(&self, pairs: &Bound<'_, PyAny>) -> Result<PyLocalDensityResult> {
        let pairs = PairSourceArg::extract(pairs)?;
        let out = self.inner.compute(LocalDensityInput {
            pairs: pairs.source()?,
        })?;
        Ok(out.into())
    }
}

described!(PyLocalDensity, LocalDensity);

/// Gaussian-smeared density of points on a grid over the box
/// (``molrs::compute::GaussianDensity``).
///
/// Parameters
/// ----------
/// dims : tuple[int, int, int]
///     Voxels per cell vector.
/// sigma : float
///     The Gaussian width.
/// r_max : float, optional
///     The cutoff of the Gaussian.
/// reach : str, optional
///     ``"ceil"`` (every voxel centre within ``r_max``) or ``"floor"``
///     (freud's narrower stencil).
///
/// ``compute(positions=, simbox=)``.
#[pyclass(module = "molrs.compute", name = "GaussianDensity", frozen)]
pub(crate) struct PyGaussianDensity {
    inner: GaussianDensity,
}

#[pymethods]
impl PyGaussianDensity {
    #[new]
    #[pyo3(signature = (dims, sigma, *, r_max = None, reach = None))]
    fn new(dims: [usize; 3], sigma: f64, r_max: Option<f64>, reach: Option<&str>) -> Result<Self> {
        let mut inner = GaussianDensity::new(dims, sigma)?;
        if let Some(r_max) = r_max {
            inner = inner.with_r_max(r_max)?;
        }
        if let Some(reach) = reach {
            inner = inner.with_reach(reach.parse()?);
        }
        Ok(Self { inner })
    }

    #[pyo3(signature = (*, positions, simbox))]
    fn compute(
        &self,
        positions: PyReadonlyArray2<'_, f64>,
        simbox: PyRef<'_, PyBox>,
    ) -> Result<PyGaussianDensityResult> {
        let out = self.inner.compute(GaussianDensityInput {
            positions: positions.as_array(),
            simbox: &simbox.inner,
        })?;
        Ok(out.into())
    }
}

described!(PyGaussianDensity, GaussianDensity);

/// Voxels covered by spheres of radius ``r_max`` around the points
/// (``molrs::compute::SphereVoxelization``).
///
/// Parameters
/// ----------
/// dims : tuple[int, int, int]
/// r_max : float
///
/// ``compute(positions=, simbox=)``.
#[pyclass(module = "molrs.compute", name = "SphereVoxelization", frozen)]
pub(crate) struct PySphereVoxelization {
    inner: SphereVoxelization,
}

#[pymethods]
impl PySphereVoxelization {
    #[new]
    fn new(dims: [usize; 3], r_max: f64) -> Result<Self> {
        Ok(Self {
            inner: SphereVoxelization::new(dims, r_max)?,
        })
    }

    #[pyo3(signature = (*, positions, simbox))]
    fn compute(
        &self,
        positions: PyReadonlyArray2<'_, f64>,
        simbox: PyRef<'_, PyBox>,
    ) -> Result<PySphereVoxelizationResult> {
        let out = self.inner.compute(SphereVoxelizationInput {
            positions: positions.as_array(),
            simbox: &simbox.inner,
        })?;
        Ok(out.into())
    }
}

described!(PySphereVoxelization, SphereVoxelization);

/// Spatial distribution of target atoms in the body frame of a rigid
/// reference, over frames (``molrs::compute::SpatialDistribution``).
///
/// Parameters
/// ----------
/// grid : RegularGrid
///     The body-frame grid.
/// bulk_density : float, optional
///     Normalise ``g_sdf`` by this bulk density.
///
/// ``compute(positions=, simbox=None, reference=, template=, target=,
/// orientation=None)`` takes one entry per frame in every list: positions
/// ``(N, 3)``, a box (or ``None``), reference indices, the ``(n_ref, 3)``
/// body-frame template, target indices and optional ``(head, tail)`` pairs.
#[pyclass(module = "molrs.compute", name = "SpatialDistribution", frozen)]
pub(crate) struct PySpatialDistribution {
    inner: SpatialDistribution,
}

#[pymethods]
impl PySpatialDistribution {
    #[new]
    #[pyo3(signature = (grid, *, bulk_density = None))]
    fn new(grid: PyRef<'_, PyRegularGrid>, bulk_density: Option<f64>) -> Result<Self> {
        let mut inner = SpatialDistribution::new(grid.inner.clone());
        if let Some(bulk_density) = bulk_density {
            inner = inner.with_bulk_density(bulk_density)?;
        }
        Ok(Self { inner })
    }

    #[allow(clippy::too_many_arguments, reason = "one argument per input field")]
    #[pyo3(signature = (
        *,
        positions,
        simbox = None,
        reference,
        template,
        target,
        orientation = None,
    ))]
    fn compute(
        &self,
        positions: Vec<PyReadonlyArray2<'_, f64>>,
        simbox: Option<Vec<Option<PyRef<'_, PyBox>>>>,
        reference: Vec<Vec<usize>>,
        template: Vec<PyReadonlyArray2<'_, f64>>,
        target: Vec<Vec<usize>>,
        orientation: Option<Vec<Option<Vec<[usize; 2]>>>>,
    ) -> Result<PySpatialDistributionResult> {
        let n_frames = positions.len();
        let boxes = frame_boxes(&simbox, n_frames)?;
        same_frames(n_frames, reference.len(), "reference")?;
        same_frames(n_frames, template.len(), "template")?;
        same_frames(n_frames, target.len(), "target")?;
        let orientation: Vec<Option<Vec<(usize, usize)>>> = match orientation {
            None => vec![None; n_frames],
            Some(frames) => {
                same_frames(n_frames, frames.len(), "orientation")?;
                frames
                    .into_iter()
                    .map(|pairs| {
                        pairs.map(|pairs| pairs.into_iter().map(|[h, t]| (h, t)).collect())
                    })
                    .collect()
            }
        };
        let frames: Vec<SpatialDistributionFrame<'_>> = (0..n_frames)
            .map(|t| SpatialDistributionFrame {
                positions: positions[t].as_array(),
                simbox: boxes[t],
                reference: &reference[t],
                template: template[t].as_array(),
                target: &target[t],
                orientation: orientation[t].as_deref(),
            })
            .collect();
        Ok(self.inner.compute(&frames)?.into())
    }
}

described!(PySpatialDistribution, SpatialDistribution);

pub(crate) fn register(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_class::<PyCorrelationFunctionResult>()?;
    m.add_class::<PyLocalDensityResult>()?;
    m.add_class::<PyGaussianDensityResult>()?;
    m.add_class::<PySphereVoxelizationResult>()?;
    m.add_class::<PySpatialDistributionResult>()?;
    m.add_class::<PyCorrelationFunction>()?;
    m.add_class::<PyLocalDensity>()?;
    m.add_class::<PyGaussianDensity>()?;
    m.add_class::<PySphereVoxelization>()?;
    m.add_class::<PySpatialDistribution>()?;
    Ok(())
}
