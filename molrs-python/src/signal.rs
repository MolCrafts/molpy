//! Python bindings for `molrs::signal` (`molrs.signal`): FFT
//! autocorrelation and cross-correlation, window functions and frequency
//! grids.

use molrs::signal::{self as sig, WindowType};
use numpy::{IntoPyArray, PyArray1, PyArrayDyn, PyReadonlyArray1, PyReadonlyArrayDyn};
use pyo3::prelude::*;

use crate::error::py_value_err;

#[pyfunction]
#[pyo3(signature = (data, max_lag))]
pub(crate) fn acf_fft<'py>(
    py: Python<'py>,
    data: PyReadonlyArray1<'py, f64>,
    max_lag: usize,
) -> PyResult<Bound<'py, PyArray1<f64>>> {
    let arr = data.as_array().to_owned();
    let result = sig::acf_fft(&arr, max_lag).map_err(py_value_err)?;
    Ok(result.into_pyarray(py))
}

#[pyfunction]
#[pyo3(signature = (data, window_type, axis = 0))]
pub(crate) fn apply_window<'py>(
    py: Python<'py>,
    data: PyReadonlyArrayDyn<'py, f64>,
    window_type: &str,
    axis: usize,
) -> PyResult<Bound<'py, PyArrayDyn<f64>>> {
    let wt = WindowType::from_name(window_type).map_err(py_value_err)?;
    let arr = data.as_array().to_owned();
    let result = sig::apply_window(&arr, wt, axis).map_err(py_value_err)?;
    Ok(result.into_pyarray(py))
}

/// Cross-correlation via FFT: `C[k] = Σ_t a[t]·b[t+k]` (Wiener–Khinchin).
///
/// For vector series pass each Cartesian component separately and sum.
#[pyfunction]
#[pyo3(signature = (a, b, max_lag))]
pub(crate) fn xcorr_fft<'py>(
    py: Python<'py>,
    a: PyReadonlyArray1<'py, f64>,
    b: PyReadonlyArray1<'py, f64>,
    max_lag: usize,
) -> PyResult<Bound<'py, PyArray1<f64>>> {
    let aa = a.as_array().to_owned();
    let bb = b.as_array().to_owned();
    let result = sig::xcorr_fft(&aa, &bb, max_lag).map_err(py_value_err)?;
    Ok(result.into_pyarray(py))
}

#[pyfunction]
pub(crate) fn frequency_grid<'py>(
    py: Python<'py>,
    n_fft: usize,
    dt: f64,
) -> Bound<'py, PyArray1<f64>> {
    let result = sig::frequency_grid(n_fft, dt);
    result.into_pyarray(py)
}

/// Register `molrs.signal`.
pub fn register(m: &Bound<'_, PyModule>) -> PyResult<()> {
    crate::add_function(m, "molrs.signal", wrap_pyfunction!(acf_fft, m)?)?;
    crate::add_function(m, "molrs.signal", wrap_pyfunction!(xcorr_fft, m)?)?;
    crate::add_function(m, "molrs.signal", wrap_pyfunction!(apply_window, m)?)?;
    crate::add_function(m, "molrs.signal", wrap_pyfunction!(frequency_grid, m)?)?;
    Ok(())
}
