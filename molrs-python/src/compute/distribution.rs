//! Internal-coordinate distributions (`molrs::compute::distribution`):
//! `DistributionFunction` (one coordinate) and `CombinedDistribution` (2 or
//! 3 coordinates jointly), both folded over frames; their outputs
//! `DistributionResult` and `CombinedDistributionResult`.
//!
//! Also the frames-form plumbing the other per-frame computes of this binding
//! share: every field is one list, an entry per frame, of the same length.

use molrs::compute::{
    Axis, CombinedDistribution, CombinedDistributionFrame, CombinedDistributionResult, Compute,
    ComputeError, Described, DistributionFrame, DistributionFunction, DistributionResult,
    InternalCoordinate,
};
use molrs::core::SimBox;
use ndarray::Array2;
use numpy::{IntoPyArray, PyArrayDyn, PyReadonlyArray2};
use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;

use super::args;
use super::error::Result;
use super::output::{ToPy, py_result};
use super::spec::{self, PyComputeSpec, described};
use super::values::PyAxis;
use crate::core::simbox::PyBox;

/// Refuse a per-frame list whose length is not the frame count.
pub(crate) fn same_frames(n_frames: usize, got: usize, what: &'static str) -> Result<()> {
    if got == n_frames {
        Ok(())
    } else {
        Err(ComputeError::DimensionMismatch {
            expected: n_frames,
            got,
            what,
        }
        .into())
    }
}

/// The per-frame boxes of an optional `simbox` field: `None` is no box in any
/// frame, a list gives each frame's box (or `None`).
pub(crate) fn frame_boxes<'a>(
    simbox: &'a Option<Vec<Option<PyRef<'_, PyBox>>>>,
    n_frames: usize,
) -> Result<Vec<Option<&'a SimBox>>> {
    match simbox {
        None => Ok(vec![None; n_frames]),
        Some(boxes) => {
            same_frames(n_frames, boxes.len(), "simbox")?;
            Ok(boxes
                .iter()
                .map(|simbox| simbox.as_ref().map(|simbox| &simbox.inner))
                .collect())
        }
    }
}

py_result! {
    /// A 1-D distribution (``molrs::compute::DistributionResult``): the raw
    /// ``histogram``, the normalised ``density`` and, for an angular
    /// coordinate, the sin θ-corrected density.
    PyDistributionResult(DistributionResult) as "DistributionResult" {
        histogram,
        density,
        density_sin_corrected,
        n_raw_samples,
        n_frames,
    }
}

/// Distribution of one internal coordinate over frames
/// (``molrs::compute::DistributionFunction``).
///
/// Parameters
/// ----------
/// observable : str
///     ``"distance"``, ``"angle"`` or ``"dihedral"``.
/// axis : Axis, optional
///     Bin on this axis (the core ``new``).
/// n_bins : int, optional
///     Bin the coordinate's natural range into ``n_bins`` (the core
///     ``over_natural_range``). Give exactly one of ``axis`` / ``n_bins``.
///
/// ``compute(positions=, simbox=None, tuples=)`` takes one ``(N, 3)``
/// position array, box (or ``None``) and ``(M, arity)`` index table per frame.
#[pyclass(module = "molrs.compute", name = "DistributionFunction", frozen)]
pub(crate) struct PyDistributionFunction {
    inner: DistributionFunction<InternalCoordinate>,
}

#[pymethods]
impl PyDistributionFunction {
    #[new]
    #[pyo3(signature = (observable, *, axis = None, n_bins = None))]
    fn new(
        observable: &str,
        axis: Option<PyRef<'_, PyAxis>>,
        n_bins: Option<usize>,
    ) -> Result<Self> {
        let observable: InternalCoordinate = observable.parse()?;
        let inner = match (axis, n_bins) {
            (Some(axis), None) => DistributionFunction::new(observable, axis.inner),
            (None, Some(n_bins)) => DistributionFunction::over_natural_range(observable, n_bins)?,
            _ => {
                return Err(PyValueError::new_err(
                    "DistributionFunction takes exactly one of axis (bin on it) or n_bins \
                     (bin the natural range)",
                )
                .into());
            }
        };
        Ok(Self { inner })
    }

    #[pyo3(signature = (*, positions, simbox = None, tuples))]
    fn compute(
        &self,
        positions: Vec<PyReadonlyArray2<'_, f64>>,
        simbox: Option<Vec<Option<PyRef<'_, PyBox>>>>,
        tuples: Vec<Vec<Vec<usize>>>,
    ) -> Result<PyDistributionResult> {
        let n_frames = positions.len();
        let boxes = frame_boxes(&simbox, n_frames)?;
        same_frames(n_frames, tuples.len(), "tuples")?;
        let tables = tuples
            .into_iter()
            .map(|rows| args::tuples(rows, "tuples"))
            .collect::<Result<Vec<_>>>()?;
        let frames: Vec<DistributionFrame<'_>> = positions
            .iter()
            .zip(&boxes)
            .zip(&tables)
            .map(|((positions, &simbox), tuples)| DistributionFrame {
                positions: positions.as_array(),
                simbox,
                tuples: tuples.view(),
            })
            .collect();
        Ok(self.inner.compute(&frames)?.into())
    }
}

described!(
    PyDistributionFunction,
    DistributionFunction<InternalCoordinate>
);

/// A `CombinedDistribution` of a dimension this binding instantiates.
enum Combined {
    Two(CombinedDistribution<2>),
    Three(CombinedDistribution<3>),
}

/// A `CombinedDistributionResult` of either dimension.
enum CombinedResult {
    Two(CombinedDistributionResult<2>),
    Three(CombinedDistributionResult<3>),
}

/// Evaluate `$body` with `$result` bound to the result of either dimension.
macro_rules! with_result {
    ($self:expr, $result:ident => $body:expr) => {
        match &$self.inner {
            CombinedResult::Two($result) => $body,
            CombinedResult::Three($result) => $body,
        }
    };
}

fn build<const N: usize>(
    observables: Vec<InternalCoordinate>,
    axes: Vec<Axis>,
    sin_weight: &Option<Vec<usize>>,
) -> Result<CombinedDistribution<N>> {
    let n_axes = axes.len();
    let observables: [InternalCoordinate; N] =
        observables
            .try_into()
            .map_err(|given: Vec<_>| ComputeError::DimensionMismatch {
                expected: N,
                got: given.len(),
                what: "observables",
            })?;
    let axes: [Axis; N] = axes
        .try_into()
        .map_err(|_| ComputeError::DimensionMismatch {
            expected: N,
            got: n_axes,
            what: "axes",
        })?;
    let mut inner = CombinedDistribution::new(observables, axes)?;
    for &axis in sin_weight.iter().flatten() {
        inner = inner.with_sin_weight(axis)?;
    }
    Ok(inner)
}

fn run<const N: usize>(
    inner: &CombinedDistribution<N>,
    positions: &[PyReadonlyArray2<'_, f64>],
    boxes: &[Option<&SimBox>],
    tables: &[Vec<Array2<usize>>],
) -> Result<CombinedDistributionResult<N>> {
    for frame in tables {
        same_frames(N, frame.len(), "tuples")?;
    }
    let frames: Vec<CombinedDistributionFrame<'_, N>> = positions
        .iter()
        .zip(boxes)
        .zip(tables)
        .map(|((positions, &simbox), tables)| CombinedDistributionFrame {
            positions: positions.as_array(),
            simbox,
            tuples: std::array::from_fn(|axis| tables[axis].view()),
        })
        .collect();
    Ok(inner.compute(&frames)?)
}

/// Joint distribution of 2 or 3 internal coordinates over frames
/// (``molrs::compute::CombinedDistribution``).
///
/// Parameters
/// ----------
/// observables : list[str]
///     One of ``"distance"``, ``"angle"``, ``"dihedral"`` per axis (2 or 3).
/// axes : list[Axis]
///     One binning axis per observable.
/// sin_weight : list[int], optional
///     The angular axes whose marginal carries the sin θ-corrected density.
///
/// ``compute(positions=, simbox=None, tuples=)`` takes, per frame, the
/// ``(N, 3)`` positions, a box (or ``None``) and one index table per axis.
#[pyclass(module = "molrs.compute", name = "CombinedDistribution", frozen)]
pub(crate) struct PyCombinedDistribution {
    inner: Combined,
}

#[pymethods]
impl PyCombinedDistribution {
    #[new]
    #[pyo3(signature = (observables, axes, *, sin_weight = None))]
    fn new(
        observables: Vec<String>,
        axes: Vec<PyRef<'_, PyAxis>>,
        sin_weight: Option<Vec<usize>>,
    ) -> Result<Self> {
        let parsed = observables
            .iter()
            .map(|name| name.parse::<InternalCoordinate>())
            .collect::<std::result::Result<Vec<_>, _>>()?;
        let axes: Vec<Axis> = axes.iter().map(|axis| axis.inner).collect();
        let inner = match parsed.len() {
            2 => Combined::Two(build(parsed, axes, &sin_weight)?),
            3 => Combined::Three(build(parsed, axes, &sin_weight)?),
            n => {
                return Err(ComputeError::OutOfRange {
                    field: "CombinedDistribution observables",
                    value: format!("{n} (only 2-D and 3-D supported)"),
                }
                .into());
            }
        };
        Ok(Self { inner })
    }

    #[pyo3(signature = (*, positions, simbox = None, tuples))]
    fn compute(
        &self,
        positions: Vec<PyReadonlyArray2<'_, f64>>,
        simbox: Option<Vec<Option<PyRef<'_, PyBox>>>>,
        tuples: Vec<Vec<Vec<Vec<usize>>>>,
    ) -> Result<PyCombinedDistributionResult> {
        let n_frames = positions.len();
        let boxes = frame_boxes(&simbox, n_frames)?;
        same_frames(n_frames, tuples.len(), "tuples")?;
        let tables = tuples
            .into_iter()
            .map(|frame| {
                frame
                    .into_iter()
                    .map(|rows| args::tuples(rows, "tuples"))
                    .collect::<Result<Vec<_>>>()
            })
            .collect::<Result<Vec<_>>>()?;
        let inner = match &self.inner {
            Combined::Two(inner) => CombinedResult::Two(run(inner, &positions, &boxes, &tables)?),
            Combined::Three(inner) => {
                CombinedResult::Three(run(inner, &positions, &boxes, &tables)?)
            }
        };
        Ok(PyCombinedDistributionResult { inner })
    }

    /// The descriptor of this compute type.
    #[staticmethod]
    fn spec() -> PyComputeSpec {
        PyComputeSpec {
            inner: <CombinedDistribution<2> as Described>::spec(),
        }
    }

    /// The value this instance holds for every parameter that has a default,
    /// as ``[(name, value), …]`` in spec order.
    fn tunables<'py>(&self, py: Python<'py>) -> PyResult<Vec<(&'static str, Bound<'py, PyAny>)>> {
        let values = match &self.inner {
            Combined::Two(inner) => inner.tunables(),
            Combined::Three(inner) => inner.tunables(),
        };
        spec::tunables(py, values)
    }
}

/// A joint distribution (``molrs::compute::CombinedDistributionResult``):
/// the N-D ``histogram`` and normalised ``density``, with which axes are
/// sin θ-corrected.
#[pyclass(module = "molrs.compute", name = "CombinedDistributionResult", frozen)]
pub(crate) struct PyCombinedDistributionResult {
    inner: CombinedResult,
}

#[pymethods]
impl PyCombinedDistributionResult {
    #[getter]
    fn histogram<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyAny>> {
        with_result!(self, result => result.histogram.to_py(py))
    }

    #[getter]
    fn density<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyAny>> {
        with_result!(self, result => result.density.to_py(py))
    }

    #[getter]
    fn n_raw_samples(&self) -> usize {
        with_result!(self, result => result.n_raw_samples)
    }

    #[getter]
    fn n_frames(&self) -> usize {
        with_result!(self, result => result.n_frames)
    }

    #[getter]
    fn sin_corrected<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyAny>> {
        with_result!(self, result => result.sin_corrected.to_py(py))
    }

    /// The 1-D distribution along ``axis``, summed over the others.
    fn marginal(&self, axis: usize) -> Result<PyDistributionResult> {
        Ok(with_result!(self, result => result.marginal(axis))?.into())
    }

    /// The free-energy surface ``-kT ln p`` at ``temperature`` (K), as the
    /// core ``free_energy`` defines it.
    fn free_energy<'py>(&self, py: Python<'py>, temperature: f64) -> Bound<'py, PyArrayDyn<f64>> {
        with_result!(self, result => result.free_energy(temperature)).into_pyarray(py)
    }

    fn __repr__(&self) -> String {
        "CombinedDistributionResult(histogram, density, n_raw_samples, n_frames, sin_corrected)"
            .to_string()
    }
}

pub(crate) fn register(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_class::<PyDistributionResult>()?;
    m.add_class::<PyDistributionFunction>()?;
    m.add_class::<PyCombinedDistributionResult>()?;
    m.add_class::<PyCombinedDistribution>()?;
    Ok(())
}
