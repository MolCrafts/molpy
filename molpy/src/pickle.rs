//! The pickle protocol shared by every picklable class.
//!
//! A class's ``__reduce__`` hands its constructor arguments (and, for a class
//! with its own ``__setstate__``, its state) to one of these; a Python
//! subclass instance's ``__dict__`` rides along either way.

use pyo3::conversion::IntoPyObjectExt;
use pyo3::intern;
use pyo3::prelude::*;
use pyo3::types::{PyDict, PyTuple};

/// The ``__dict__`` of a Python subclass instance, when it holds anything.
///
/// A native instance has no ``__dict__``; an instance of a Python subclass of
/// a ``subclass`` pyclass does, and pickling must carry it.
fn instance_dict<'py>(slf: &Bound<'py, PyAny>) -> Option<Bound<'py, PyDict>> {
    slf.getattr(intern!(slf.py(), "__dict__"))
        .ok()?
        .cast_into::<PyDict>()
        .ok()
        .filter(|d| !d.is_empty())
}

/// Pickle protocol: reconstruct with ``type(self)(*args)``.
///
/// The state is a Python subclass instance's ``__dict__`` (``None`` for a
/// native instance), which pickle's default restore merges back.
pub fn reduce_via_type<'py>(
    slf: &Bound<'py, PyAny>,
    args: impl IntoPyObject<'py>,
) -> PyResult<Bound<'py, PyTuple>> {
    let py = slf.py();
    let packed = args.into_bound_py_any(py)?;
    let tuple = if let Ok(tuple) = packed.cast::<PyTuple>() {
        tuple.to_owned()
    } else {
        PyTuple::new(py, [packed])?
    };
    let state = instance_dict(slf).map_or_else(|| py.None().into_bound(py), Bound::into_any);
    PyTuple::new(py, [slf.get_type().into_any(), tuple.into_any(), state])
}

/// Pickle protocol for a class with its own ``__setstate__``: reconstruct
/// with ``type(self)(*args)``, then ``__setstate__(state)``.
///
/// A Python subclass instance's ``__dict__`` rides along: the reduce value
/// then names [`_restore_pickled_state`] as its state setter, which runs
/// ``__setstate__`` and merges the dict back.
pub fn reduce_with_state<'py>(
    slf: &Bound<'py, PyAny>,
    args: Bound<'py, PyTuple>,
    state: Bound<'py, PyAny>,
) -> PyResult<Bound<'py, PyTuple>> {
    let py = slf.py();
    let cls = slf.get_type().into_any();
    let Some(dict) = instance_dict(slf) else {
        return PyTuple::new(py, [cls, args.into_any(), state]);
    };
    let setter = py
        .import(intern!(py, "molpy._core"))?
        .getattr(intern!(py, "_restore_pickled_state"))?;
    let both = PyTuple::new(py, [state, dict.into_any()])?.into_any();
    PyTuple::new(
        py,
        [
            cls,
            args.into_any(),
            both,
            py.None().into_bound(py),
            py.None().into_bound(py),
            setter,
        ],
    )
}

/// The state setter of [`reduce_with_state`]: ``obj.__setstate__(state)``,
/// then ``obj.__dict__.update(instance_dict)``.
#[pyfunction]
pub fn _restore_pickled_state(
    obj: &Bound<'_, PyAny>,
    state: (Bound<'_, PyAny>, Bound<'_, PyDict>),
) -> PyResult<()> {
    let (native, dict) = state;
    obj.call_method1(intern!(obj.py(), "__setstate__"), (native,))?;
    obj.getattr(intern!(obj.py(), "__dict__"))?
        .call_method1(intern!(obj.py(), "update"), (dict,))?;
    Ok(())
}
