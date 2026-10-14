//! Geometric distributions read from a frame's topology blocks
//! (`molrs::compute::distribution`).

use super::collect_frames;
use crate::error::py_value_err;
use molrs::compute::{
    AngleObservable, AtomGroups, AxisSpec, CombinedDistribution, CombinedDistributionResult,
    DihedralObservable, DistanceObservable, DistributionFunction, DistributionResult,
    InternalCoordinate, LegacyCompute,
};
use molrs::core::Frame as CoreFrame;
use numpy::{IntoPyArray, PyArray1};
use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;
use pyo3::types::PyAny;

// ---------------------------------------------------------------------------
// Distribution functions (ADF / DDF / distance-DF)
// ---------------------------------------------------------------------------

/// Shared result of the geometric distribution functions.
#[pyclass(module = "molrs.compute", name = "DistributionResult")]
pub struct PyDistributionResult {
    inner: DistributionResult,
}

#[pymethods]
impl PyDistributionResult {
    #[getter]
    fn bin_centers<'py>(&self, py: Python<'py>) -> Bound<'py, PyArray1<f64>> {
        self.inner.bin_centers.clone().into_pyarray(py)
    }
    #[getter]
    fn bin_edges<'py>(&self, py: Python<'py>) -> Bound<'py, PyArray1<f64>> {
        self.inner.bin_edges.clone().into_pyarray(py)
    }
    #[getter]
    fn counts<'py>(&self, py: Python<'py>) -> Bound<'py, PyArray1<f64>> {
        self.inner.counts.clone().into_pyarray(py)
    }
    #[getter]
    fn density<'py>(&self, py: Python<'py>) -> Bound<'py, PyArray1<f64>> {
        self.inner.density.clone().into_pyarray(py)
    }
    #[getter]
    fn density_sin_corrected<'py>(&self, py: Python<'py>) -> Option<Bound<'py, PyArray1<f64>>> {
        self.inner
            .density_sin_corrected
            .as_ref()
            .map(|d| d.clone().into_pyarray(py))
    }
    #[getter]
    fn bin_width(&self) -> f64 {
        self.inner.bin_width
    }
    #[getter]
    fn n_binned(&self) -> f64 {
        self.inner.n_binned
    }
    #[getter]
    fn n_raw_samples(&self) -> usize {
        self.inner.n_raw_samples
    }
    #[getter]
    fn n_frames(&self) -> usize {
        self.inner.n_frames
    }
    #[getter]
    fn angular(&self) -> bool {
        self.inner.angular
    }
}

/// The three instantiations of the generic Rust `DistributionFunction<O>`.
enum DistributionKernel {
    Distance(DistributionFunction<DistanceObservable>),
    Angle(DistributionFunction<AngleObservable>),
    Dihedral(DistributionFunction<DihedralObservable>),
}

/// One-dimensional distribution function of an internal coordinate over the
/// atom groups of a frame's topology — Rust `DistributionFunction<O>` with the
/// observable named by ``observable``:
///
/// * ``"distance"`` — pairs from the ``bonds`` block; ``min`` and ``max`` (in
///   the coordinates' length unit) are **required**, a distance has no natural
///   range.
/// * ``"angle"`` — triplets from the ``angles`` block, angle at the middle atom
///   (ADF). Bounds in **radians**; omit both for the natural range ``[0, π]``.
///   The sin θ correction divides by a vanishing quantity at both ends, so the
///   corrected density amplifies counting noise near θ = 0 and θ = π.
/// * ``"dihedral"`` — quadruplets from the ``dihedrals`` block (DDF). Bounds in
///   **radians**; omit both for the natural range ``(−π, π]``. The default stays
///   **signed**: folding to ``|φ|`` would collapse g+ onto g− and cannot be
///   undone.
///
/// Supplying exactly one bound is a ``ValueError``.
#[pyclass(module = "molrs.compute", name = "DistributionFunction")]
pub struct PyDistributionFunction {
    kernel: DistributionKernel,
}

/// Build a kernel over `min..max`, or over the observable's natural range when
/// both are omitted.
fn ranged<O: molrs::compute::Observable>(
    observable: O,
    n_bins: usize,
    min: Option<f64>,
    max: Option<f64>,
    name: &str,
) -> PyResult<DistributionFunction<O>> {
    match (min, max) {
        (None, None) => DistributionFunction::over_natural_range(observable, n_bins),
        (Some(min), Some(max)) => DistributionFunction::new(observable, n_bins, min, max),
        _ => {
            return Err(PyValueError::new_err(format!(
                "DistributionFunction({name:?}): supply both `min` and `max` (radians), \
                 or neither to use the observable's natural range"
            )));
        }
    }
    .map_err(py_value_err)
}

#[pymethods]
impl PyDistributionFunction {
    #[new]
    #[pyo3(signature = (observable, n_bins, min=None, max=None))]
    fn new(observable: &str, n_bins: usize, min: Option<f64>, max: Option<f64>) -> PyResult<Self> {
        let kernel = match observable {
            "distance" => {
                let (Some(min), Some(max)) = (min, max) else {
                    return Err(PyValueError::new_err(
                        "DistributionFunction(\"distance\"): `min` and `max` are required; \
                         a distance has no natural range",
                    ));
                };
                DistributionKernel::Distance(
                    DistributionFunction::new(DistanceObservable, n_bins, min, max)
                        .map_err(py_value_err)?,
                )
            }
            "angle" => {
                DistributionKernel::Angle(ranged(AngleObservable, n_bins, min, max, observable)?)
            }
            "dihedral" => DistributionKernel::Dihedral(ranged(
                DihedralObservable,
                n_bins,
                min,
                max,
                observable,
            )?),
            other => {
                return Err(PyValueError::new_err(format!(
                    "DistributionFunction: unknown observable {other:?} \
                     (expected \"distance\", \"angle\" or \"dihedral\")"
                )));
            }
        };
        Ok(Self { kernel })
    }

    /// The atom groups are read from the first frame's topology block
    /// (``bonds`` / ``angles`` / ``dihedrals``).
    fn compute(&self, frames: &Bound<'_, PyAny>) -> PyResult<PyDistributionResult> {
        let owned = collect_frames(frames)?;
        let refs: Vec<&CoreFrame> = owned.iter().collect();
        let first = refs
            .first()
            .copied()
            .ok_or_else(|| PyValueError::new_err("no frames provided"))?;
        let inner = match &self.kernel {
            DistributionKernel::Distance(k) => {
                let groups = AtomGroups::from_frame(first, "bonds", 2).map_err(py_value_err)?;
                k.compute(&refs, &groups)
            }
            DistributionKernel::Angle(k) => {
                let groups = AtomGroups::from_frame(first, "angles", 3).map_err(py_value_err)?;
                k.compute(&refs, &groups)
            }
            DistributionKernel::Dihedral(k) => {
                let groups = AtomGroups::from_frame(first, "dihedrals", 4).map_err(py_value_err)?;
                k.compute(&refs, &groups)
            }
        }
        .map_err(py_value_err)?;
        Ok(PyDistributionResult { inner })
    }
}

// ---------------------------------------------------------------------------
// Combined (multi-axis) distribution
// ---------------------------------------------------------------------------

/// Map an observable-kind string to its `InternalCoordinate` variant + arity.
#[pyclass(module = "molrs.compute", name = "CombinedDistributionResult")]

pub struct PyCombinedDistributionResult {
    inner: CombinedDistributionResult,
}

#[pymethods]
impl PyCombinedDistributionResult {
    /// Per-axis bin edges (`bins + 1` each).
    #[getter]
    fn edges<'py>(&self, py: Python<'py>) -> Vec<Bound<'py, PyArray1<f64>>> {
        self.inner
            .edges
            .iter()
            .map(|e| e.clone().into_pyarray(py))
            .collect()
    }
    /// Per-axis bin centers (`bins` each).
    #[getter]
    fn centers<'py>(&self, py: Python<'py>) -> Vec<Bound<'py, PyArray1<f64>>> {
        self.inner
            .centers
            .iter()
            .map(|c| c.clone().into_pyarray(py))
            .collect()
    }
    /// Flat row-major counts (axis 0 fastest).
    #[getter]
    fn counts<'py>(&self, py: Python<'py>) -> Bound<'py, PyArray1<f64>> {
        self.inner.counts.clone().into_pyarray(py)
    }
    /// Flat row-major normalized joint density.
    #[getter]
    fn density<'py>(&self, py: Python<'py>) -> Bound<'py, PyArray1<f64>> {
        self.inner.density.clone().into_pyarray(py)
    }
    #[getter]
    fn binned(&self) -> f64 {
        self.inner.binned
    }
    #[getter]
    fn n_raw_samples(&self) -> usize {
        self.inner.n_raw_samples
    }
    #[getter]
    fn n_frames(&self) -> usize {
        self.inner.n_frames
    }
    #[getter]
    fn ndim(&self) -> usize {
        self.inner.ndim()
    }
    /// Flat row-major index for a multi-axis bin coordinate.
    fn flat_index(&self, idx: Vec<usize>) -> usize {
        self.inner.flat_index(&idx)
    }
    /// Product of all axis bin widths (the N-D cell "volume").
    fn bin_width_product(&self) -> f64 {
        self.inner.bin_width_product()
    }
}

/// Joint multi-axis distribution over several geometric observables (the reference implementation
/// combined-DF). Each axis is `(kind, bins, min, max, sin_weight)` where `kind`
/// is ``"distance"`` / ``"angle"`` / ``"dihedral"``.
#[pyclass(module = "molrs.compute", name = "CombinedDistribution")]
pub struct PyCombinedDistribution {
    inner: CombinedDistribution,
    arities: Vec<usize>,
}

#[pymethods]
impl PyCombinedDistribution {
    /// `axes`: list of `(kind, bins, min, max, sin_weight)` — one per dimension.
    #[new]
    fn new(axes: Vec<(String, usize, f64, f64, bool)>) -> PyResult<Self> {
        let mut observables = Vec::with_capacity(axes.len());
        let mut specs = Vec::with_capacity(axes.len());
        let mut arities = Vec::with_capacity(axes.len());
        for (kind, bins, min, max, sin_weight) in &axes {
            let (obs, arity) = InternalCoordinate::from_kind(kind).map_err(py_value_err)?;
            observables.push(obs);
            arities.push(arity);
            specs.push(
                AxisSpec::new(*bins, *min, *max)
                    .map_err(py_value_err)?
                    .with_sin_weight(*sin_weight),
            );
        }
        let inner = CombinedDistribution::new(observables, specs).map_err(py_value_err)?;
        Ok(Self { inner, arities })
    }

    /// Per-axis atom groups are read from the first frame's topology block that
    /// matches each axis arity (2 → `bonds`, 3 → `angles`, 4 → `dihedrals`).
    fn compute(&self, frames: &Bound<'_, PyAny>) -> PyResult<PyCombinedDistributionResult> {
        let owned = collect_frames(frames)?;
        let refs: Vec<&CoreFrame> = owned.iter().collect();
        let first = refs
            .first()
            .copied()
            .ok_or_else(|| PyValueError::new_err("no frames provided"))?;
        let group_objs: Vec<AtomGroups> = self
            .arities
            .iter()
            .map(|&arity| {
                let block = match arity {
                    2 => "bonds",
                    3 => "angles",
                    4 => "dihedrals",
                    _ => {
                        return Err(PyValueError::new_err(format!(
                            "no topology block for axis arity {arity}"
                        )));
                    }
                };
                AtomGroups::from_frame(first, block, arity).map_err(py_value_err)
            })
            .collect::<PyResult<_>>()?;
        let inner = self
            .inner
            .compute(&refs, &group_objs)
            .map_err(py_value_err)?;
        Ok(PyCombinedDistributionResult { inner })
    }
}

/// Register this domain's classes and functions.
pub(crate) fn register(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_class::<PyDistributionResult>()?;
    m.add_class::<PyDistributionFunction>()?;
    m.add_class::<PyCombinedDistributionResult>()?;
    m.add_class::<PyCombinedDistribution>()?;
    Ok(())
}
