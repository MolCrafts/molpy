//! The value types a compute is configured with or returns: `Axis` (a binning
//! axis), `RegularGrid` (a voxel grid) and `Histogram` (an N-D histogram on
//! axes). None of them is a compute.

use molrs::compute::{Axis, RegularGrid};
use ndarray::ArrayD;
use pyo3::prelude::*;

use super::error::Result;
use super::output::ToPy;
use crate::core::simbox::PyBox;

/// A uniform binning axis: ``n_bins`` half-open bins over ``[min, max)``
/// (``molrs::compute::Axis``).
///
/// Parameters
/// ----------
/// min, max : float
///     The axis range.
/// n_bins : int
///     The number of bins.
#[pyclass(module = "molrs.compute", name = "Axis", frozen, skip_from_py_object)]
#[derive(Clone)]
pub(crate) struct PyAxis {
    pub(crate) inner: Axis,
}

#[pymethods]
impl PyAxis {
    #[new]
    fn new(min: f64, max: f64, n_bins: usize) -> Result<Self> {
        Ok(Self {
            inner: Axis::new(min, max, n_bins)?,
        })
    }

    #[getter]
    fn min(&self) -> f64 {
        self.inner.min()
    }

    #[getter]
    fn max(&self) -> f64 {
        self.inner.max()
    }

    #[getter]
    fn n_bins(&self) -> usize {
        self.inner.n_bins()
    }

    #[getter]
    fn width(&self) -> f64 {
        self.inner.width()
    }

    /// The ``n_bins + 1`` bin edges.
    #[getter]
    fn edges<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyAny>> {
        self.inner.edges().to_py(py)
    }

    /// The ``n_bins`` bin centres.
    #[getter]
    fn centers<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyAny>> {
        self.inner.centers().to_py(py)
    }

    fn __repr__(&self) -> String {
        format!(
            "Axis(min={}, max={}, n_bins={})",
            self.inner.min(),
            self.inner.max(),
            self.inner.n_bins()
        )
    }
}

/// A regular 3-D voxel grid (``molrs::compute::RegularGrid``): ``origin``,
/// the three cell ``basis`` vectors (rows), ``dims`` voxels per axis and the
/// periodicity ``pbc`` of each axis.
#[pyclass(
    module = "molrs.compute",
    name = "RegularGrid",
    frozen,
    skip_from_py_object
)]
#[derive(Clone)]
pub(crate) struct PyRegularGrid {
    pub(crate) inner: RegularGrid<3>,
}

impl ToPy for RegularGrid<3> {
    fn to_py<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyAny>> {
        Ok(Bound::new(
            py,
            PyRegularGrid {
                inner: self.clone(),
            },
        )?
        .into_any())
    }
}

#[pymethods]
impl PyRegularGrid {
    #[new]
    fn new(
        origin: [f64; 3],
        basis: [[f64; 3]; 3],
        dims: [usize; 3],
        pbc: [bool; 3],
    ) -> Result<Self> {
        Ok(Self {
            inner: RegularGrid::new(origin, basis, dims, pbc)?,
        })
    }

    /// A non-periodic orthogonal grid of total ``extent`` centred on the
    /// origin.
    #[staticmethod]
    fn centered(extent: [f64; 3], dims: [usize; 3]) -> Result<Self> {
        Ok(Self {
            inner: RegularGrid::centered(extent, dims)?,
        })
    }

    /// The grid spanning ``box`` with ``dims`` voxels per cell vector, periodic
    /// where the box is.
    #[staticmethod]
    fn over_box(r#box: PyRef<'_, PyBox>, dims: [usize; 3]) -> Result<Self> {
        Ok(Self {
            inner: RegularGrid::over_box(&r#box.inner, dims)?,
        })
    }

    #[getter]
    fn origin(&self) -> [f64; 3] {
        self.inner.origin()
    }

    #[getter]
    fn basis<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyAny>> {
        self.inner.basis().to_py(py)
    }

    #[getter]
    fn dims(&self) -> [usize; 3] {
        self.inner.dims()
    }

    #[getter]
    fn pbc(&self) -> [bool; 3] {
        self.inner.pbc()
    }

    #[getter]
    fn voxel_volume(&self) -> f64 {
        self.inner.voxel_volume()
    }

    fn __repr__(&self) -> String {
        format!(
            "RegularGrid(origin={:?}, dims={:?}, pbc={:?})",
            self.inner.origin(),
            self.inner.dims(),
            self.inner.pbc()
        )
    }
}

/// An N-D histogram (``molrs::compute::Histogram``): raw ``counts`` on
/// ``axes``, the number of samples that fell in a bin (``binned``) and the
/// volume of one bin.
#[pyclass(module = "molrs.compute", name = "Histogram", frozen)]
pub(crate) struct PyHistogram {
    pub(crate) axes: Vec<Axis>,
    pub(crate) counts: ArrayD<f64>,
    pub(crate) binned: f64,
    pub(crate) bin_volume: f64,
}

#[pymethods]
impl PyHistogram {
    #[getter]
    fn axes(&self) -> Vec<PyAxis> {
        self.axes.iter().map(|&inner| PyAxis { inner }).collect()
    }

    #[getter]
    fn counts<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyAny>> {
        self.counts.to_py(py)
    }

    #[getter]
    fn binned(&self) -> f64 {
        self.binned
    }

    #[getter]
    fn bin_volume(&self) -> f64 {
        self.bin_volume
    }

    fn __repr__(&self) -> String {
        format!("Histogram(shape={:?})", self.counts.shape())
    }
}

pub(crate) fn register(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_class::<PyAxis>()?;
    m.add_class::<PyRegularGrid>()?;
    m.add_class::<PyHistogram>()?;
    Ok(())
}
