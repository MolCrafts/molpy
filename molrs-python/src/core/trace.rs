//! Python wrapper for [`molrs::core::Trace`], an ordered path of 3D points.

use crate::array::ArrayArg2;
use molrs::core::Trace;
use ndarray::Array2;
use numpy::{IntoPyArray, PyArray2};
use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;

/// An ordered path of 3D points, with no chemistry — ``molrs.core.Trace``.
///
/// A trace says *where* consecutive units of a chain sit (for example the
/// site positions of one coarse-grained chain), not *what* sits there.
/// Frozen.
///
/// Parameters
/// ----------
/// points : numpy.ndarray, shape (k, 3), float64
///     Every point, in order (Å). ``(0, 3)`` is the empty trace.
///
/// Raises
/// ------
/// ValueError
///     If ``points`` is not ``(k, 3)``.
///
/// Examples
/// --------
/// >>> trace = molrs.core.Trace(np.array([[0.0, 0.0, 0.0], [1.5, 0.0, 0.0]]))
/// >>> len(trace)
/// 2
#[pyclass(
    module = "molrs.core",
    name = "Trace",
    frozen,
    skip_from_py_object,
    subclass
)]
pub struct PyTrace {
    pub(crate) inner: Trace,
}

#[pymethods]
impl PyTrace {
    #[new]
    fn new(points: ArrayArg2<'_, f64>) -> PyResult<Self> {
        let array = points.as_array();
        if array.ncols() != 3 {
            return Err(PyValueError::new_err(format!(
                "points must have shape (k, 3), got ({}, {})",
                array.nrows(),
                array.ncols()
            )));
        }
        let points = array
            .rows()
            .into_iter()
            .map(|row| [row[0], row[1], row[2]])
            .collect();
        Ok(Self {
            inner: Trace::from_points(points),
        })
    }

    /// Every point, in order, as a float64 ``(k, 3)`` copy (Å).
    #[getter]
    fn points<'py>(&self, py: Python<'py>) -> Bound<'py, PyArray2<f64>> {
        let points = self.inner.points();
        Array2::from_shape_fn((points.len(), 3), |(row, axis)| points[row][axis]).into_pyarray(py)
    }

    /// The number of points, k.
    fn __len__(&self) -> usize {
        self.inner.points().len()
    }

    fn __repr__(&self) -> String {
        format!("<Trace n_points={}>", self.inner.points().len())
    }
}
