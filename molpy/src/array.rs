//! Array arguments: a numpy ``ndarray`` of one element type and rank, borrowed
//! read-only, as every binding takes them.
//!
//! [`ArrayArg`] is the one place an array argument is extracted. numpy's own
//! extraction fails a wrong dtype or rank with a downcast message that names
//! the type twice (``'ndarray' object is not an instance of 'ndarray'``) and
//! says neither what was expected nor what arrived. Here the failure says
//! both — ``expected a 2-D float64 array, got a 3-D int64 array of shape
//! (100, 32, 3)`` — and pyo3 prefixes the argument's name. Nothing is
//! converted: a wrong dtype or rank is an error, never a silent cast.

use std::ops::Deref;

use numpy::ndarray::{Dimension, Ix1, Ix2, Ix3, IxDyn};
use numpy::{Element, PyArray, PyArrayDescrMethods, PyArrayMethods, PyReadonlyArray};
use pyo3::exceptions::PyTypeError;
use pyo3::prelude::*;
use pyo3::{Borrowed, FromPyObject};

/// A read-only numpy array argument of element `T` and dimension `D`.
///
/// Dereferences to numpy's [`PyReadonlyArray`], so `as_array()`,
/// `as_slice()` and the rest read as before.
pub(crate) struct ArrayArg<'py, T: Element, D: Dimension>(PyReadonlyArray<'py, T, D>);

/// A 1-D array argument.
pub(crate) type ArrayArg1<'py, T> = ArrayArg<'py, T, Ix1>;
/// A 2-D array argument.
pub(crate) type ArrayArg2<'py, T> = ArrayArg<'py, T, Ix2>;
/// A 3-D array argument.
pub(crate) type ArrayArg3<'py, T> = ArrayArg<'py, T, Ix3>;
/// An array argument of any rank.
pub(crate) type ArrayArgDyn<'py, T> = ArrayArg<'py, T, IxDyn>;

impl<'py, T: Element, D: Dimension> Deref for ArrayArg<'py, T, D> {
    type Target = PyReadonlyArray<'py, T, D>;

    fn deref(&self) -> &Self::Target {
        &self.0
    }
}

impl<'a, 'py, T: Element + 'a, D: Dimension + 'a> FromPyObject<'a, 'py> for ArrayArg<'py, T, D> {
    type Error = PyErr;

    fn extract(obj: Borrowed<'a, 'py, PyAny>) -> PyResult<Self> {
        match obj.cast::<PyArray<T, D>>() {
            Ok(array) => Ok(Self(array.readonly())),
            Err(_) => Err(PyTypeError::new_err(mismatch::<T, D>(&obj))),
        }
    }
}

/// `expected <rank> <dtype> array, got <what obj is>`.
fn mismatch<T: Element, D: Dimension>(obj: &Bound<'_, PyAny>) -> String {
    let dtype = numpy::dtype::<T>(obj.py());
    let rank = match D::NDIM {
        Some(n) => format!("a {n}-D"),
        None => "an".to_string(),
    };
    format!(
        "expected {rank} {} array, got {}",
        dtype_name(&dtype),
        describe(obj)
    )
}

/// What `obj` is: an array's rank, dtype and shape, or any other object's
/// type name.
fn describe(obj: &Bound<'_, PyAny>) -> String {
    if let Ok(array) = obj.cast::<numpy::PyUntypedArray>() {
        use numpy::PyUntypedArrayMethods;
        let shape = array.shape();
        let dims = shape
            .iter()
            .map(usize::to_string)
            .collect::<Vec<_>>()
            .join(", ");
        let shape = if shape.len() == 1 {
            format!("({dims},)")
        } else {
            format!("({dims})")
        };
        return format!(
            "a {}-D {} array of shape {shape}",
            array.ndim(),
            dtype_name(&array.dtype())
        );
    }
    obj.get_type()
        .name()
        .map(|name| format!("{name}"))
        .unwrap_or_else(|_| "an object of unknown type".to_string())
}

fn dtype_name(dtype: &Bound<'_, numpy::PyArrayDescr>) -> String {
    dtype
        .str()
        .map(|name| name.to_string())
        .unwrap_or_else(|_| format!("{:?}", dtype.typeobj()))
}
