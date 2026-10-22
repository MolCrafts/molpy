//! The one mapping from a core [`ComputeError`] to a Python exception.
//!
//! A binding method returns [`Result`], so `?` lifts both a core refusal and
//! a Python-side extraction failure; PyO3 turns the error into the exception.
//! A [`ComputeError`] is always a `ValueError` carrying the core message.

use molrs::compute::ComputeError;
use pyo3::PyErr;
use pyo3::exceptions::PyValueError;

/// A refusal raised by a compute binding: the core's, or Python's own.
pub(crate) enum Error {
    /// The core refused the configuration or the data.
    Compute(ComputeError),
    /// A Python-side failure (an argument of the wrong type, …).
    Py(PyErr),
}

/// What every compute binding method returns.
pub(crate) type Result<T> = std::result::Result<T, Error>;

impl From<ComputeError> for Error {
    fn from(error: ComputeError) -> Self {
        Self::Compute(error)
    }
}

impl From<PyErr> for Error {
    fn from(error: PyErr) -> Self {
        Self::Py(error)
    }
}

impl From<Error> for PyErr {
    fn from(error: Error) -> Self {
        match error {
            Error::Compute(error) => PyValueError::new_err(error.to_string()),
            Error::Py(error) => error,
        }
    }
}
