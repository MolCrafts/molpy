//! Fits of a correlation curve (`molrs::compute::fitting`): `CumulativeTrapezoid`,
//! `DebyeFit`, `LinearFit` and `Plateau`, each reading one channel of a
//! `TimeCorrelation`; their outputs `DebyeFitResult`, `LinearFitResult` and
//! `PlateauResult`.

use molrs::compute::{
    Compute, CumulativeTrapezoid, CumulativeTrapezoidInput, DebyeFit, DebyeFitInput,
    DebyeFitResult, LinearFit, LinearFitInput, LinearFitResult, Plateau, PlateauInput,
    PlateauResult,
};
use pyo3::prelude::*;

use super::error::Result;
use super::output::py_result;
use super::spec::described;
use super::time_series::PyTimeCorrelation;

py_result! {
    /// A single-exponential fit ``C(t) = amplitude · exp(−t / tau)``
    /// (``molrs::compute::DebyeFitResult``) over ``n_samples`` lags.
    PyDebyeFitResult(DebyeFitResult) as "DebyeFitResult" { tau, amplitude, n_samples }
}

py_result! {
    /// A least-squares line ``slope · t + intercept``
    /// (``molrs::compute::LinearFitResult``) with its ``r2``, fitted over the
    /// lags ``[fit_start, fit_end)``.
    PyLinearFitResult(LinearFitResult) as "LinearFitResult" {
        slope, intercept, r2, fit_start, fit_end
    }
}

py_result! {
    /// The mean ``value`` (and its ``std``) of a curve over a window of
    /// ``n_samples`` lags (``molrs::compute::PlateauResult``).
    PyPlateauResult(PlateauResult) as "PlateauResult" { value, n_samples, std }
}

/// The running trapezoid integral of one channel of a correlation
/// (``molrs::compute::CumulativeTrapezoid``) — a Green–Kubo integral.
///
/// Parameters
/// ----------
/// n_lags : int, optional
///     Integrate only the first ``n_lags`` lags; ``None`` keeps the core
///     default (every lag).
///
/// ``compute(correlation=, channel=)`` returns the integral as a
/// ``TimeCorrelation``.
#[pyclass(module = "molpy.compute", name = "CumulativeTrapezoid", frozen)]
pub(crate) struct PyCumulativeTrapezoid {
    inner: CumulativeTrapezoid,
}

#[pymethods]
impl PyCumulativeTrapezoid {
    #[new]
    #[pyo3(signature = (*, n_lags = None))]
    fn new(n_lags: Option<usize>) -> Self {
        let mut inner = CumulativeTrapezoid::new();
        if let Some(n_lags) = n_lags {
            inner = inner.with_n_lags(n_lags);
        }
        Self { inner }
    }

    #[pyo3(signature = (*, correlation, channel))]
    fn compute(
        &self,
        correlation: PyRef<'_, PyTimeCorrelation>,
        channel: usize,
    ) -> Result<PyTimeCorrelation> {
        let out = self.inner.compute(CumulativeTrapezoidInput {
            correlation: &correlation.inner,
            channel,
        })?;
        Ok(out.into())
    }
}

described!(PyCumulativeTrapezoid, CumulativeTrapezoid);

/// Single-exponential (Debye) relaxation fit of one channel of a correlation
/// (``molrs::compute::DebyeFit``).
///
/// ``compute(correlation=, channel=)`` returns a ``DebyeFitResult``.
#[pyclass(module = "molpy.compute", name = "DebyeFit", frozen)]
pub(crate) struct PyDebyeFit {
    inner: DebyeFit,
}

#[pymethods]
impl PyDebyeFit {
    #[new]
    fn new() -> Self {
        Self { inner: DebyeFit }
    }

    #[pyo3(signature = (*, correlation, channel))]
    fn compute(
        &self,
        correlation: PyRef<'_, PyTimeCorrelation>,
        channel: usize,
    ) -> Result<PyDebyeFitResult> {
        let out = self.inner.compute(DebyeFitInput {
            correlation: &correlation.inner,
            channel,
        })?;
        Ok(out.into())
    }
}

described!(PyDebyeFit, DebyeFit);

/// Least-squares line through one channel of a correlation over a window of
/// its lags (``molrs::compute::LinearFit``) — e.g. the diffusive regime of an
/// MSD.
///
/// Parameters
/// ----------
/// start, end : float
///     The window, as fractions of the curve length.
///
/// ``compute(correlation=, channel=)`` returns a ``LinearFitResult``.
#[pyclass(module = "molpy.compute", name = "LinearFit", frozen)]
pub(crate) struct PyLinearFit {
    inner: LinearFit,
}

#[pymethods]
impl PyLinearFit {
    #[new]
    fn new(start: f64, end: f64) -> Result<Self> {
        Ok(Self {
            inner: LinearFit::new(start, end)?,
        })
    }

    #[pyo3(signature = (*, correlation, channel))]
    fn compute(
        &self,
        correlation: PyRef<'_, PyTimeCorrelation>,
        channel: usize,
    ) -> Result<PyLinearFitResult> {
        let out = self.inner.compute(LinearFitInput {
            correlation: &correlation.inner,
            channel,
        })?;
        Ok(out.into())
    }
}

described!(PyLinearFit, LinearFit);

/// The mean of one channel of a correlation over a window of its lags
/// (``molrs::compute::Plateau``) — e.g. the plateau of a Green–Kubo integral.
///
/// Parameters
/// ----------
/// start, end : float
///     The window, as fractions of the curve length.
///
/// ``compute(correlation=, channel=)`` returns a ``PlateauResult``.
#[pyclass(module = "molpy.compute", name = "Plateau", frozen)]
pub(crate) struct PyPlateau {
    inner: Plateau,
}

#[pymethods]
impl PyPlateau {
    #[new]
    fn new(start: f64, end: f64) -> Result<Self> {
        Ok(Self {
            inner: Plateau::new(start, end)?,
        })
    }

    #[pyo3(signature = (*, correlation, channel))]
    fn compute(
        &self,
        correlation: PyRef<'_, PyTimeCorrelation>,
        channel: usize,
    ) -> Result<PyPlateauResult> {
        let out = self.inner.compute(PlateauInput {
            correlation: &correlation.inner,
            channel,
        })?;
        Ok(out.into())
    }
}

described!(PyPlateau, Plateau);

pub(crate) fn register(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_class::<PyDebyeFitResult>()?;
    m.add_class::<PyLinearFitResult>()?;
    m.add_class::<PyPlateauResult>()?;
    m.add_class::<PyCumulativeTrapezoid>()?;
    m.add_class::<PyDebyeFit>()?;
    m.add_class::<PyLinearFit>()?;
    m.add_class::<PyPlateau>()?;
    Ok(())
}
