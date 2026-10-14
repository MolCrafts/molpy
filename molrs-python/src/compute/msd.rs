//! Mean-squared displacement (`molrs::compute::msd`): `Msd`, `MsdResult`,
//! `MsdTimeSeries`.

use super::collect_frames;
use crate::error::py_value_err;
use molrs::compute::{LegacyCompute, Msd, MsdMode, MsdResult, MsdTimeSeries};
use molrs::core::Frame as CoreFrame;
use numpy::{IntoPyArray, PyArray1, PyArray2};
use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;
use pyo3::types::PyAny;

// ---------------------------------------------------------------------------
// MSD
// ---------------------------------------------------------------------------

/// Per-frame MSD result (from a single time point).
#[pyclass(module = "molrs.compute", name = "MsdResult")]
pub struct PyMsdResult {
    inner: MsdResult,
}

#[pymethods]
impl PyMsdResult {
    #[getter]
    fn mean(&self) -> f64 {
        self.inner.mean
    }

    #[getter]
    fn per_particle<'py>(&self, py: Python<'py>) -> Bound<'py, PyArray1<f64>> {
        self.inner.per_particle.clone().into_pyarray(py)
    }

    fn __repr__(&self) -> String {
        format!(
            "MsdResult(mean={:.4}, n_particles={})",
            self.inner.mean,
            self.inner.per_particle.len(),
        )
    }
}

/// MSD time series aligned with the input frame list.
///
/// `series.per_frame[0]` is the reference frame (mean = 0); `series.per_frame[i]`
/// compares frame `i` against frame `0`.
#[pyclass(module = "molrs.compute", name = "MsdTimeSeries")]
pub struct PyMsdTimeSeries {
    inner: MsdTimeSeries,
}

#[pymethods]
impl PyMsdTimeSeries {
    #[getter]
    fn mean<'py>(&self, py: Python<'py>) -> Bound<'py, PyArray1<f64>> {
        let v: Vec<f64> = self.inner.per_frame.iter().map(|r| r.mean).collect();
        ndarray::Array1::from_vec(v).into_pyarray(py)
    }

    #[getter]
    fn per_particle<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyArray2<f64>>> {
        let t = self.inner.per_frame.len();
        if t == 0 {
            return Ok(ndarray::Array2::from_shape_vec((0, 0), vec![])
                .unwrap()
                .into_pyarray(py));
        }
        let n = self.inner.per_frame[0].per_particle.len();
        let mut flat: Vec<f64> = Vec::with_capacity(t * n);
        for row in &self.inner.per_frame {
            if row.per_particle.len() != n {
                return Err(PyValueError::new_err(
                    "Msd per-particle width not constant across frames",
                ));
            }
            flat.extend(row.per_particle.iter().copied());
        }
        Ok(ndarray::Array2::from_shape_vec((t, n), flat)
            .expect("Msd shape")
            .into_pyarray(py))
    }

    fn __len__(&self) -> usize {
        self.inner.per_frame.len()
    }

    fn __getitem__(&self, i: isize) -> PyResult<PyMsdResult> {
        let n = self.inner.per_frame.len() as isize;
        let idx = if i < 0 { i + n } else { i };
        if idx < 0 || idx >= n {
            return Err(PyValueError::new_err("Msd index out of range"));
        }
        Ok(PyMsdResult {
            inner: self.inner.per_frame[idx as usize].clone(),
        })
    }

    fn __repr__(&self) -> String {
        format!("MsdTimeSeries(n_frames={})", self.inner.per_frame.len())
    }
}

/// Mean squared displacement.
///
/// Two estimators of the same quantity, chosen by ``method``; they are not
/// interchangeable and there is no default that suits both:
///
/// - ``"direct"`` (default) — ``Msd(t) = ⟨|r(t) − r(0)|²⟩``, frame 0 as the one
///   time origin.
/// - ``"window"`` — ``Msd(t) = ⟨|r(τ+t) − r(τ)|²⟩`` averaged over **every** time
///   origin τ. Far better statistics at long lag, which is what a diffusion
///   coefficient needs, and O(T log T) via the Wiener–Khinchin identity rather
///   than the O(T²) nested loop. Conventions match ``freud.msd``.
///
/// ``compute(frames)`` returns an ``MsdTimeSeries`` as long as ``frames``
/// either way.
///
/// Parameters
/// ----------
/// method : {"direct", "window"}, optional
///
/// Examples
/// --------
/// >>> molrs.compute.Msd(method="window").compute(frames).mean
#[pyclass(module = "molrs.compute", name = "Msd")]
pub struct PyMsd {
    inner: Msd,
}

#[pymethods]
impl PyMsd {
    #[new]
    #[pyo3(signature = (method = "direct"))]
    fn new(method: &str) -> PyResult<Self> {
        let mode = match method {
            "direct" => MsdMode::Direct,
            "window" => MsdMode::Window,
            other => {
                return Err(PyValueError::new_err(format!(
                    "unknown MSD method {other:?}; expected 'direct' or 'window'"
                )));
            }
        };
        Ok(Self {
            inner: Msd::with_mode(mode),
        })
    }

    /// The estimator this analyzer uses: ``"direct"`` or ``"window"``.
    #[getter]
    fn method(&self) -> &'static str {
        match self.inner.mode() {
            MsdMode::Direct => "direct",
            MsdMode::Window => "window",
        }
    }

    /// Compute the MSD time series.
    fn compute(&self, frames: &Bound<'_, PyAny>) -> PyResult<PyMsdTimeSeries> {
        let owned = collect_frames(frames)?;
        if owned.is_empty() {
            return Err(PyValueError::new_err("Msd.compute requires >= 1 frame"));
        }
        let refs: Vec<&CoreFrame> = owned.iter().collect();
        let series = self.inner.compute(&refs, ()).map_err(py_value_err)?;
        Ok(PyMsdTimeSeries { inner: series })
    }

    fn __repr__(&self) -> String {
        format!("Msd(method={:?})", self.method())
    }
}

/// Register this domain's classes and functions.
pub(crate) fn register(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_class::<PyMsdResult>()?;
    m.add_class::<PyMsdTimeSeries>()?;
    m.add_class::<PyMsd>()?;
    Ok(())
}
