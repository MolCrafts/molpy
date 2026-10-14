//! Core outputs as Python objects: the [`ToPy`] conversion of every field
//! type a core result holds, and [`py_result!`], the one macro that turns a
//! core result type into a frozen class with a read-only getter per field.
//!
//! A getter converts on access: an array field is a fresh `numpy` array, a
//! nested core value (a `QlmTable`, an `HBond`, a `RegularGrid`) is its
//! Python class.

use molrs::compute::{Axis, Histogram};
use molrs::core::Complex;
use ndarray::{Array, ArrayD, Dimension};
use numpy::{Element, IntoPyArray, PyArray1};
use pyo3::IntoPyObjectExt;
use pyo3::prelude::*;
use pyo3::types::PyList;

use super::values::{PyAxis, PyHistogram};

/// A value a result getter hands to Python.
pub(crate) trait ToPy {
    fn to_py<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyAny>>;
}

macro_rules! scalar_to_py {
    ($($t:ty),+ $(,)?) => {$(
        impl ToPy for $t {
            fn to_py<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyAny>> {
                self.clone().into_bound_py_any(py)
            }
        }
    )+};
}

scalar_to_py!(
    f64,
    bool,
    u8,
    u32,
    u64,
    usize,
    i64,
    String,
    (usize, usize, f64)
);

/// A flat numeric list is a 1-D array.
macro_rules! vec_to_array {
    ($($t:ty),+ $(,)?) => {$(
        impl ToPy for Vec<$t> {
            fn to_py<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyAny>> {
                Ok(PyArray1::from_slice(py, self).into_any())
            }
        }
    )+};
}

vec_to_array!(f64, bool, u32, u64, usize);

/// Complex values are a `complex128` array.
impl ToPy for Vec<Complex> {
    fn to_py<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyAny>> {
        complex_array(py, ndarray::ArrayView1::from(self.as_slice()))
    }
}

/// A `complex128` array of core complex values.
pub(crate) fn complex_array<'py, D: Dimension>(
    py: Python<'py>,
    values: ndarray::ArrayView<'_, Complex, D>,
) -> PyResult<Bound<'py, PyAny>> {
    Ok(values
        .mapv(|z| num_complex::Complex64::new(z.re, z.im))
        .into_pyarray(py)
        .into_any())
}

/// A list of values that are not numbers is a Python list.
pub(crate) fn list<'py, T: ToPy>(py: Python<'py>, items: &[T]) -> PyResult<Bound<'py, PyAny>> {
    let items = items
        .iter()
        .map(|item| item.to_py(py))
        .collect::<PyResult<Vec<_>>>()?;
    Ok(PyList::new(py, items)?.into_any())
}

macro_rules! vec_to_list {
    ($($t:ty),+ $(,)?) => {$(
        impl ToPy for Vec<$t> {
            fn to_py<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyAny>> {
                list(py, self)
            }
        }
    )+};
}

vec_to_list!(
    Vec<f64>,
    Vec<u64>,
    Option<usize>,
    (usize, usize, f64),
    ndarray::Array1<f64>,
);

impl<A: Element + Clone, D: Dimension> ToPy for Array<A, D> {
    fn to_py<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyAny>> {
        Ok(self.clone().into_pyarray(py).into_any())
    }
}

impl<const N: usize> ToPy for [f64; N] {
    fn to_py<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyAny>> {
        Ok(PyArray1::from_slice(py, self).into_any())
    }
}

impl<const N: usize> ToPy for [bool; N] {
    fn to_py<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyAny>> {
        Ok(PyArray1::from_slice(py, self).into_any())
    }
}

/// A 3 × 3 matrix is a `(3, 3)` array.
impl ToPy for [[f64; 3]; 3] {
    fn to_py<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyAny>> {
        ndarray::Array2::from(self.to_vec()).to_py(py)
    }
}

impl<T: ToPy> ToPy for Option<T> {
    fn to_py<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyAny>> {
        match self {
            Some(value) => value.to_py(py),
            None => Ok(py.None().into_bound(py)),
        }
    }
}

impl ToPy for Axis {
    fn to_py<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyAny>> {
        Ok(Bound::new(py, PyAxis { inner: *self })?.into_any())
    }
}

impl<const N: usize> ToPy for Histogram<N> {
    fn to_py<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyAny>> {
        let histogram = PyHistogram {
            axes: self.axes().to_vec(),
            counts: ArrayD::clone(self.counts()),
            binned: self.binned(),
            bin_volume: self.bin_volume(),
        };
        Ok(Bound::new(py, histogram)?.into_any())
    }
}

/// Declare the Python class of a core result type.
///
/// ```ignore
/// py_result! {
///     /// The k-means partition.
///     PyKmeansResult(KmeansResult) as "KmeansResult" { labels, centroids }
/// }
/// ```
///
/// generates a frozen `#[pyclass]` holding the core value as `inner`, one
/// read-only getter per listed (public) field converted through [`ToPy`], a
/// `__repr__` naming the fields, `From<core>` for the class, and `ToPy` for
/// the core type, so a result nested in another result converts to its class.
/// Methods beyond the fields go in a separate `#[pymethods]` block.
macro_rules! py_result {
    (
        $(#[$meta:meta])*
        $py:ident($core:ty) as $name:literal { $($field:ident),* $(,)? }
    ) => {
        $(#[$meta])*
        #[pyclass(module = "molrs.compute", name = $name, frozen)]
        pub(crate) struct $py {
            pub(crate) inner: $core,
        }

        impl From<$core> for $py {
            fn from(inner: $core) -> Self {
                Self { inner }
            }
        }

        impl $crate::compute::output::ToPy for $core {
            fn to_py<'py>(
                &self,
                py: pyo3::Python<'py>,
            ) -> pyo3::PyResult<pyo3::Bound<'py, pyo3::PyAny>> {
                Ok(pyo3::Bound::new(py, $py { inner: self.clone() })?.into_any())
            }
        }

        #[pymethods]
        impl $py {
            $(
                #[getter]
                fn $field<'py>(
                    &self,
                    py: pyo3::Python<'py>,
                ) -> pyo3::PyResult<pyo3::Bound<'py, pyo3::PyAny>> {
                    $crate::compute::output::ToPy::to_py(&self.inner.$field, py)
                }
            )*

            fn __repr__(&self) -> String {
                let fields: &[&str] = &[$(stringify!($field)),*];
                format!("{}({})", $name, fields.join(", "))
            }
        }
    };
}

pub(crate) use py_result;
