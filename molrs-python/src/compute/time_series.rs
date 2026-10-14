//! Time series (`molrs::compute::time_series`): the one estimator
//! `Correlator`, its fixed-configuration surfaces (`Msd`, `Vacf`,
//! `OnsagerCorrelation`, `DipoleAcf`, `CurrentAcf`, `ReorientationAcf`,
//! `RotationalAutocorrelation`, `Survival`), the series operations
//! `Derivative` and `Unwrap`, and the transform `Spectrum`; their outputs
//! `TimeCorrelation` and `OneSidedSpectrum`.

use molrs::compute::{
    Compute, Correlator, CorrelatorInput, CurrentAcf, CurrentAcfInput, Derivative, DerivativeInput,
    DipoleAcf, DipoleAcfInput, LagKernel, Msd, MsdInput, OneSidedSpectrum, OnsagerCorrelation,
    OnsagerInput, ReorientationAcf, ReorientationAcfInput, RotationalAutocorrelation,
    RotationalAutocorrelationInput, Spectrum, SpectrumInput, Survival, SurvivalInput,
    TimeCorrelation, Unwrap, UnwrapInput, Vacf, VacfInput,
};
use numpy::{
    IntoPyArray, PyArray2, PyArray3, PyReadonlyArray1, PyReadonlyArray2, PyReadonlyArray3,
};
use pyo3::prelude::*;

use super::args;
use super::error::Result;
use super::output::{ToPy, py_result};
use super::spec::described;
use crate::core::simbox::PyBox;

py_result! {
    /// A correlation curve (``molrs::compute::TimeCorrelation``):
    /// ``values[lag, channel]`` at ``lag_times[lag]``.
    ///
    /// The output of ``Correlator`` and its surfaces, and the input of the
    /// fits, ``Spectrum``-based routes and checks. Build one from your own
    /// curve with ``TimeCorrelation(lag_times, values)``.
    PyTimeCorrelation(TimeCorrelation) as "TimeCorrelation" { lag_times, values }
}

#[pymethods]
impl PyTimeCorrelation {
    #[new]
    fn new(lag_times: PyReadonlyArray1<'_, f64>, values: PyReadonlyArray2<'_, f64>) -> Self {
        Self {
            inner: TimeCorrelation {
                lag_times: lag_times.as_array().to_owned(),
                values: values.as_array().to_owned(),
            },
        }
    }

    #[getter]
    fn n_lags(&self) -> usize {
        self.inner.n_lags()
    }

    #[getter]
    fn n_channels(&self) -> usize {
        self.inner.n_channels()
    }

    /// The curve of one channel.
    fn channel<'py>(&self, py: Python<'py>, channel: usize) -> Result<Bound<'py, PyAny>> {
        Ok(self.inner.channel(channel)?.to_owned().to_py(py)?)
    }
}

py_result! {
    /// A one-sided spectrum (``molrs::compute::OneSidedSpectrum``):
    /// ``real`` and ``imag`` parts at ``frequencies``, from an ``n_fft``-point
    /// transform.
    PyOneSidedSpectrum(OneSidedSpectrum) as "OneSidedSpectrum" { frequencies, real, imag, n_fft }
}

/// The one time-correlation estimator (``molrs::compute::Correlator``).
///
/// Parameters
/// ----------
/// kernel : str
///     ``"product"``, ``"squared_displacement"``, ``"legendre"``,
///     ``"wigner_d"`` or ``"persistence"`` (the core ``LagKernelKind``).
/// max_lag : int
///     The last lag, in frames.
/// l : int, optional
///     The order ``legendre`` / ``wigner_d`` require and the others refuse.
/// origins, mean, normalization, reduction : str, optional
///     The core choices (``Origins``, ``MeanPolicy``, ``Normalization``,
///     ``Reduction``); ``None`` keeps the core default.
/// dimension : int, optional
///     Use only the first ``dimension`` components.
///
/// ``compute(a=, b=None, dt=, boxes=None)`` correlates the ``(T, N, D)``
/// stack ``a`` (with ``b`` for a cross-correlation) at frame spacing ``dt``.
#[pyclass(module = "molrs.compute", name = "Correlator", frozen)]
pub(crate) struct PyCorrelator {
    inner: Correlator,
}

#[pymethods]
impl PyCorrelator {
    #[new]
    #[pyo3(signature = (
        kernel,
        max_lag,
        *,
        l = None,
        origins = None,
        mean = None,
        normalization = None,
        reduction = None,
        dimension = None,
    ))]
    #[allow(clippy::too_many_arguments, reason = "Public Python keyword arguments")]
    fn new(
        kernel: &str,
        max_lag: usize,
        l: Option<u32>,
        origins: Option<&str>,
        mean: Option<&str>,
        normalization: Option<&str>,
        reduction: Option<&str>,
        dimension: Option<usize>,
    ) -> Result<Self> {
        let mut inner = Correlator::new(LagKernel::new(kernel.parse()?, l)?, max_lag);
        if let Some(origins) = origins {
            inner = inner.with_origins(origins.parse()?);
        }
        if let Some(mean) = mean {
            inner = inner.with_mean(mean.parse()?);
        }
        if let Some(normalization) = normalization {
            inner = inner.with_normalization(normalization.parse()?);
        }
        if let Some(reduction) = reduction {
            inner = inner.with_reduction(reduction.parse()?);
        }
        if let Some(dimension) = dimension {
            inner = inner.with_dimension(dimension);
        }
        Ok(Self { inner })
    }

    #[pyo3(signature = (*, a, b = None, dt, boxes = None))]
    fn compute(
        &self,
        a: PyReadonlyArray3<'_, f64>,
        b: Option<PyReadonlyArray3<'_, f64>>,
        dt: f64,
        boxes: Option<Vec<PyRef<'_, PyBox>>>,
    ) -> Result<PyTimeCorrelation> {
        let boxes = boxes.as_deref().map(args::boxes);
        let out = self.inner.compute(CorrelatorInput {
            a: a.as_array(),
            b: b.as_ref().map(|b| b.as_array()),
            dt,
            boxes: boxes.as_deref(),
        })?;
        Ok(out.into())
    }
}

described!(PyCorrelator, Correlator);

/// Mean-squared displacement (``molrs::compute::Msd``), the
/// ``squared_displacement`` correlator.
///
/// Parameters
/// ----------
/// max_lag : int
/// origins, reduction : str, optional
///
/// ``compute(positions=, dt=, boxes=None)`` over an unwrapped ``(T, N, 3)``
/// trajectory; with ``boxes`` a wrapped one is refused.
#[pyclass(module = "molrs.compute", name = "Msd", frozen)]
pub(crate) struct PyMsd {
    inner: Msd,
}

#[pymethods]
impl PyMsd {
    #[new]
    #[pyo3(signature = (max_lag, *, origins = None, reduction = None))]
    fn new(max_lag: usize, origins: Option<&str>, reduction: Option<&str>) -> Result<Self> {
        let mut inner = Msd::new(max_lag);
        if let Some(origins) = origins {
            inner = inner.with_origins(origins.parse()?);
        }
        if let Some(reduction) = reduction {
            inner = inner.with_reduction(reduction.parse()?);
        }
        Ok(Self { inner })
    }

    #[pyo3(signature = (*, positions, dt, boxes = None))]
    fn compute(
        &self,
        positions: PyReadonlyArray3<'_, f64>,
        dt: f64,
        boxes: Option<Vec<PyRef<'_, PyBox>>>,
    ) -> Result<PyTimeCorrelation> {
        let boxes = boxes.as_deref().map(args::boxes);
        let out = self.inner.compute(MsdInput {
            positions: positions.as_array(),
            dt,
            boxes: boxes.as_deref(),
        })?;
        Ok(out.into())
    }
}

described!(PyMsd, Msd);

/// Velocity autocorrelation (``molrs::compute::Vacf``).
///
/// ``compute(velocities=, dt=)`` over a ``(T, C)`` velocity series.
#[pyclass(module = "molrs.compute", name = "Vacf", frozen)]
pub(crate) struct PyVacf {
    inner: Vacf,
}

#[pymethods]
impl PyVacf {
    #[new]
    fn new(max_lag: usize) -> Self {
        Self {
            inner: Vacf::new(max_lag),
        }
    }

    #[pyo3(signature = (*, velocities, dt))]
    fn compute(&self, velocities: PyReadonlyArray2<'_, f64>, dt: f64) -> Result<PyTimeCorrelation> {
        let out = self.inner.compute(VacfInput {
            velocities: velocities.as_array(),
            dt,
        })?;
        Ok(out.into())
    }
}

described!(PyVacf, Vacf);

/// Onsager cross-correlation of two ``(T, C)`` series
/// (``molrs::compute::OnsagerCorrelation``).
///
/// ``compute(a=, b=, dt=, boxes=None)``.
#[pyclass(module = "molrs.compute", name = "OnsagerCorrelation", frozen)]
pub(crate) struct PyOnsagerCorrelation {
    inner: OnsagerCorrelation,
}

#[pymethods]
impl PyOnsagerCorrelation {
    #[new]
    fn new(max_lag: usize) -> Self {
        Self {
            inner: OnsagerCorrelation::new(max_lag),
        }
    }

    #[pyo3(signature = (*, a, b, dt, boxes = None))]
    fn compute(
        &self,
        a: PyReadonlyArray2<'_, f64>,
        b: PyReadonlyArray2<'_, f64>,
        dt: f64,
        boxes: Option<Vec<PyRef<'_, PyBox>>>,
    ) -> Result<PyTimeCorrelation> {
        let boxes = boxes.as_deref().map(args::boxes);
        let out = self.inner.compute(OnsagerInput {
            a: a.as_array(),
            b: b.as_array(),
            dt,
            boxes: boxes.as_deref(),
        })?;
        Ok(out.into())
    }
}

described!(PyOnsagerCorrelation, OnsagerCorrelation);

/// Dipole autocorrelation of a ``(T, 3)`` dipole series
/// (``molrs::compute::DipoleAcf``).
///
/// ``compute(dipoles=, dt=)``.
#[pyclass(module = "molrs.compute", name = "DipoleAcf", frozen)]
pub(crate) struct PyDipoleAcf {
    inner: DipoleAcf,
}

#[pymethods]
impl PyDipoleAcf {
    #[new]
    fn new(max_lag: usize) -> Self {
        Self {
            inner: DipoleAcf::new(max_lag),
        }
    }

    #[pyo3(signature = (*, dipoles, dt))]
    fn compute(&self, dipoles: PyReadonlyArray2<'_, f64>, dt: f64) -> Result<PyTimeCorrelation> {
        let out = self.inner.compute(DipoleAcfInput {
            dipoles: dipoles.as_array(),
            dt,
        })?;
        Ok(out.into())
    }
}

described!(PyDipoleAcf, DipoleAcf);

/// Current autocorrelation of a ``(T, 3)`` current series
/// (``molrs::compute::CurrentAcf``).
///
/// ``compute(current=, dt=)``.
#[pyclass(module = "molrs.compute", name = "CurrentAcf", frozen)]
pub(crate) struct PyCurrentAcf {
    inner: CurrentAcf,
}

#[pymethods]
impl PyCurrentAcf {
    #[new]
    fn new(max_lag: usize) -> Self {
        Self {
            inner: CurrentAcf::new(max_lag),
        }
    }

    #[pyo3(signature = (*, current, dt))]
    fn compute(&self, current: PyReadonlyArray2<'_, f64>, dt: f64) -> Result<PyTimeCorrelation> {
        let out = self.inner.compute(CurrentAcfInput {
            current: current.as_array(),
            dt,
        })?;
        Ok(out.into())
    }
}

described!(PyCurrentAcf, CurrentAcf);

/// Legendre reorientation correlation ``⟨P_l(u(0)·u(t))⟩`` of ``(T, N, 3)``
/// unit vectors (``molrs::compute::ReorientationAcf``).
///
/// ``compute(vectors=, dt=)``.
#[pyclass(module = "molrs.compute", name = "ReorientationAcf", frozen)]
pub(crate) struct PyReorientationAcf {
    inner: ReorientationAcf,
}

#[pymethods]
impl PyReorientationAcf {
    #[new]
    fn new(l: u32, max_lag: usize) -> Self {
        Self {
            inner: ReorientationAcf::new(l, max_lag),
        }
    }

    #[pyo3(signature = (*, vectors, dt))]
    fn compute(&self, vectors: PyReadonlyArray3<'_, f64>, dt: f64) -> Result<PyTimeCorrelation> {
        let out = self.inner.compute(ReorientationAcfInput {
            vectors: vectors.as_array(),
            dt,
        })?;
        Ok(out.into())
    }
}

described!(PyReorientationAcf, ReorientationAcf);

/// Rotational autocorrelation of ``(T, N, 4)`` unit quaternions
/// (``molrs::compute::RotationalAutocorrelation``).
///
/// ``compute(orientations=, dt=)``.
#[pyclass(module = "molrs.compute", name = "RotationalAutocorrelation", frozen)]
pub(crate) struct PyRotationalAutocorrelation {
    inner: RotationalAutocorrelation,
}

#[pymethods]
impl PyRotationalAutocorrelation {
    #[new]
    #[pyo3(signature = (l, max_lag, *, reduction = None))]
    fn new(l: u32, max_lag: usize, reduction: Option<&str>) -> Result<Self> {
        let mut inner = RotationalAutocorrelation::new(l, max_lag);
        if let Some(reduction) = reduction {
            inner = inner.with_reduction(reduction.parse()?);
        }
        Ok(Self { inner })
    }

    #[pyo3(signature = (*, orientations, dt))]
    fn compute(
        &self,
        orientations: PyReadonlyArray3<'_, f64>,
        dt: f64,
    ) -> Result<PyTimeCorrelation> {
        let out = self.inner.compute(RotationalAutocorrelationInput {
            orientations: orientations.as_array(),
            dt,
        })?;
        Ok(out.into())
    }
}

described!(PyRotationalAutocorrelation, RotationalAutocorrelation);

/// Survival correlation of a ``(T, C)`` presence series
/// (``molrs::compute::Survival``), e.g. ``PairPresence``'s ``values``.
///
/// Parameters
/// ----------
/// kind : str
///     ``"continuous"`` or ``"intermittent"``.
/// max_lag : int
///
/// ``compute(presence=, dt=)``.
#[pyclass(module = "molrs.compute", name = "Survival", frozen)]
pub(crate) struct PySurvival {
    inner: Survival,
}

#[pymethods]
impl PySurvival {
    #[new]
    fn new(kind: &str, max_lag: usize) -> Result<Self> {
        Ok(Self {
            inner: Survival::new(kind.parse()?, max_lag),
        })
    }

    #[pyo3(signature = (*, presence, dt))]
    fn compute(&self, presence: PyReadonlyArray2<'_, f64>, dt: f64) -> Result<PyTimeCorrelation> {
        let out = self.inner.compute(SurvivalInput {
            presence: presence.as_array(),
            dt,
        })?;
        Ok(out.into())
    }
}

described!(PySurvival, Survival);

/// Finite-difference time derivative of a ``(T, C)`` series
/// (``molrs::compute::Derivative``).
///
/// Parameters
/// ----------
/// stencil : str
///     ``"central"``, ``"gradient"`` or ``"backward"``.
///
/// ``compute(series=, dt=)`` returns the ``(T, C)`` derivative.
#[pyclass(module = "molrs.compute", name = "Derivative", frozen)]
pub(crate) struct PyDerivative {
    inner: Derivative,
}

#[pymethods]
impl PyDerivative {
    #[new]
    fn new(stencil: &str) -> Result<Self> {
        Ok(Self {
            inner: Derivative::new(stencil.parse()?),
        })
    }

    #[pyo3(signature = (*, series, dt))]
    fn compute<'py>(
        &self,
        py: Python<'py>,
        series: PyReadonlyArray2<'py, f64>,
        dt: f64,
    ) -> Result<Bound<'py, PyArray2<f64>>> {
        let out = self.inner.compute(DerivativeInput {
            series: series.as_array(),
            dt,
        })?;
        Ok(out.into_pyarray(py))
    }
}

described!(PyDerivative, Derivative);

/// Unwrap a wrapped ``(T, N, 3)`` trajectory (``molrs::compute::Unwrap``),
/// from int32 image flags when given, else by minimum image between frames.
///
/// ``compute(positions=, images=None, boxes=)`` returns the unwrapped
/// ``(T, N, 3)`` positions.
#[pyclass(module = "molrs.compute", name = "Unwrap", frozen)]
pub(crate) struct PyUnwrap {
    inner: Unwrap,
}

#[pymethods]
impl PyUnwrap {
    #[new]
    fn new() -> Self {
        Self {
            inner: Unwrap::new(),
        }
    }

    #[pyo3(signature = (*, positions, images = None, boxes))]
    fn compute<'py>(
        &self,
        py: Python<'py>,
        positions: PyReadonlyArray3<'py, f64>,
        images: Option<PyReadonlyArray3<'py, i32>>,
        boxes: Vec<PyRef<'py, PyBox>>,
    ) -> Result<Bound<'py, PyArray3<f64>>> {
        let boxes = args::boxes(&boxes);
        let out = self.inner.compute(UnwrapInput {
            positions: positions.as_array(),
            images: images.as_ref().map(|images| images.as_array()),
            boxes: &boxes,
        })?;
        Ok(out.into_pyarray(py))
    }
}

described!(PyUnwrap, Unwrap);

/// The one-sided Fourier transform of a real signal
/// (``molrs::compute::Spectrum``).
///
/// Parameters
/// ----------
/// window : str
///     ``"hann"``, ``"blackman"``, ``"cosine_sq"`` or ``"none"``.
/// pad : int
///     Zero-pad to ``pad`` times the signal length (rounded up to a power of
///     two).
///
/// ``compute(signal=, dt=)`` returns a ``OneSidedSpectrum``.
#[pyclass(module = "molrs.compute", name = "Spectrum", frozen)]
pub(crate) struct PySpectrum {
    inner: Spectrum,
}

#[pymethods]
impl PySpectrum {
    #[new]
    fn new(window: &str, pad: usize) -> Result<Self> {
        Ok(Self {
            inner: Spectrum::new(window.parse()?, pad)?,
        })
    }

    #[pyo3(signature = (*, signal, dt))]
    fn compute(&self, signal: PyReadonlyArray1<'_, f64>, dt: f64) -> Result<PyOneSidedSpectrum> {
        let out = self.inner.compute(SpectrumInput {
            signal: signal.as_array(),
            dt,
        })?;
        Ok(out.into())
    }
}

described!(PySpectrum, Spectrum);

pub(crate) fn register(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_class::<PyTimeCorrelation>()?;
    m.add_class::<PyOneSidedSpectrum>()?;
    m.add_class::<PyCorrelator>()?;
    m.add_class::<PyMsd>()?;
    m.add_class::<PyVacf>()?;
    m.add_class::<PyOnsagerCorrelation>()?;
    m.add_class::<PyDipoleAcf>()?;
    m.add_class::<PyCurrentAcf>()?;
    m.add_class::<PyReorientationAcf>()?;
    m.add_class::<PyRotationalAutocorrelation>()?;
    m.add_class::<PySurvival>()?;
    m.add_class::<PyDerivative>()?;
    m.add_class::<PyUnwrap>()?;
    m.add_class::<PySpectrum>()?;
    Ok(())
}
