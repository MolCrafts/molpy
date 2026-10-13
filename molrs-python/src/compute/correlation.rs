//! The time-correlation engine (`molrs::compute::correlation`) for Python:
//! [`TimeCorrelation`](PyTimeCorrelation), the one result every correlator
//! returns; [`Spectrum`](PySpectrum), what a windowed transform of one
//! returns; and the two bare front ends [`Acf`](PyAcf) and [`Oacf`](PyOacf).
//! The quantity front ends (`Vacf`, `Msd`, `OnsagerCorrelation`, …) live in
//! their own domain files and build their result through [`finish`].

use crate::error::py_value_err;
use molrs::compute::{
    CumulativeTrapezoid, DebyeFit, Fit, LinearFit, Normalization, Oacf, Plateau, PowerSpectrum,
    TimeCorrelation, TimeCorrelationSpec, autocorrelate,
};
use molrs::signal::WindowType;
use ndarray::{Array1, Array2, Array3};
use numpy::{
    IntoPyArray, PyArray1, PyArray2, PyReadonlyArray1, PyReadonlyArray2, PyReadonlyArray3,
};
use pyo3::conversion::IntoPyObjectExt;
use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;
use pyo3::types::{PyAny, PyDict, PyDictMethods, PyTuple};

// ═══════════════════════════════════════════════════════════════════════════
// Shared vocabulary
// ═══════════════════════════════════════════════════════════════════════════

/// Estimator settings every producer takes: mean subtraction, normalization
/// name, origin stride.
#[derive(Debug, Clone, Copy)]
pub(crate) struct Settings {
    pub mean_subtract: bool,
    pub normalization: Normalization,
    pub stride: usize,
}

impl Settings {
    pub(crate) fn parse(mean_subtract: bool, normalization: &str, stride: usize) -> PyResult<Self> {
        let normalization = parse_normalization(normalization)?;
        if stride == 0 {
            return Err(PyValueError::new_err("stride must be >= 1"));
        }
        Ok(Self {
            mean_subtract,
            normalization,
            stride,
        })
    }

    pub(crate) fn spec(&self, max_lag: usize, per_entity: bool) -> TimeCorrelationSpec<'static> {
        TimeCorrelationSpec::new()
            .max_lag(max_lag)
            .mean_subtract(self.mean_subtract)
            .normalization(self.normalization)
            .origin_stride(self.stride)
            .per_entity(per_entity)
    }
}

pub(crate) fn parse_normalization(name: &str) -> PyResult<Normalization> {
    Normalization::from_name(name).ok_or_else(|| {
        PyValueError::new_err(format!(
            "unknown normalization {name:?}; expected 'unbiased', 'biased' or 'raw'"
        ))
    })
}

pub(crate) fn parse_window(name: &str) -> PyResult<WindowType> {
    WindowType::from_name(name).map_err(py_value_err)
}

/// A `(n_frames, 3)` Cartesian series, checked.
pub(crate) fn cartesian_series(
    series: &PyReadonlyArray2<'_, f64>,
    what: &str,
) -> PyResult<Array2<f64>> {
    let a = series.as_array();
    if a.shape()[1] != 3 {
        return Err(PyValueError::new_err(format!(
            "{what} must have shape (n_frames, 3), got {:?}",
            a.shape()
        )));
    }
    if a.shape()[0] < 2 {
        return Err(PyValueError::new_err(format!(
            "{what} needs at least two frames, got {}",
            a.shape()[0]
        )));
    }
    if let Some(index) = a.iter().position(|v| !v.is_finite()) {
        return Err(PyValueError::new_err(format!(
            "{what} has a non-finite value at flat index {index}"
        )));
    }
    Ok(a.to_owned())
}

/// Time origins per lag when origins are taken every `stride` frames.
pub(crate) fn origin_counts(n_frames: usize, max_lag: usize, stride: usize) -> Array1<usize> {
    Array1::from_iter((0..=max_lag).map(|t| (n_frames - 1 - t) / stride + 1))
}

/// Wrap an engine result as the Python object, with the producer's metadata.
pub(crate) fn finish(
    tc: TimeCorrelation,
    dt: Option<f64>,
    kind: &str,
    settings: Settings,
    n_frames: usize,
    order: Option<u32>,
) -> PyTimeCorrelation {
    PyTimeCorrelation {
        values: tc.values,
        lags: tc.lags,
        counts: Some(tc.counts),
        per_entity: tc.per_entity,
        dt,
        kind: kind.to_string(),
        normalization: tc.normalization,
        mean_subtract: settings.mean_subtract,
        stride: settings.stride,
        order,
        n_frames: Some(n_frames),
        n_entities: tc.n_entities,
        extras: None,
    }
}

impl PyTimeCorrelation {
    /// A curve a front end computed without the engine (one origin, a
    /// survival count): lags `0..n`, the given counts.
    pub(crate) fn from_curve(
        values: Array1<f64>,
        dt: Option<f64>,
        kind: &str,
        settings: Settings,
        counts: Option<Array1<usize>>,
        n_frames: usize,
    ) -> Self {
        let n = values.len();
        Self {
            values,
            lags: Array1::from_iter(0..n),
            counts,
            per_entity: None,
            dt,
            kind: kind.to_string(),
            normalization: settings.normalization,
            mean_subtract: settings.mean_subtract,
            stride: settings.stride,
            order: None,
            n_frames: Some(n_frames),
            n_entities: 1,
            extras: None,
        }
    }

    /// Attach per-entity rows `(n_entities, n_lags)` and the entity count.
    pub(crate) fn with_rows(mut self, rows: Array2<f64>) -> Self {
        self.n_entities = rows.nrows();
        self.per_entity = Some(rows);
        self
    }

    /// Record the entity count of a curve without rows.
    pub(crate) fn with_entities(mut self, n_entities: usize) -> Self {
        self.n_entities = n_entities;
        self
    }

    /// Keep only the first `n` lags.
    pub(crate) fn truncated(mut self, n: usize) -> Self {
        let n = n.min(self.values.len()).max(1);
        self.values = self.values.slice(ndarray::s![..n]).to_owned();
        self.lags = self.lags.slice(ndarray::s![..n]).to_owned();
        self.counts = self.counts.map(|c| c.slice(ndarray::s![..n]).to_owned());
        self.per_entity = self
            .per_entity
            .map(|r| r.slice(ndarray::s![.., ..n]).to_owned());
        self
    }
}

fn name_of(normalization: Normalization) -> &'static str {
    normalization.name()
}

fn window_name(window: WindowType) -> &'static str {
    window.name()
}

// ═══════════════════════════════════════════════════════════════════════════
// TimeCorrelation
// ═══════════════════════════════════════════════════════════════════════════

/// One time-correlation curve and the estimator that produced it.
///
/// Every correlator in ``molrs.compute`` — ``Acf``, ``Vacf``, ``Oacf``,
/// ``Msd``, ``EinsteinDiffusion``, ``GreenKuboDiffusion``,
/// ``EinsteinConductivity``, ``GreenKuboConductivity``, ``DebyeRelaxation``,
/// ``DipoleRateCross``, ``OnsagerCorrelation``, ``pair_survival_tcf`` —
/// returns one of these. ``values[t]`` is ``C(t)`` at ``lags[t]`` frames;
/// ``lag_times`` is ``lags * dt`` when the producer was given a ``dt``.
/// ``counts[t]`` is the number of time origins behind lag ``t``.
///
/// The derived quantities are methods, each the matching ``Fit`` on the same
/// arrays: :meth:`normalized`, :meth:`integrate`, :meth:`linear_fit`,
/// :meth:`fit_exponential`, :meth:`diffusion`, :meth:`spectrum`.
///
/// ``np.asarray(tc)`` is ``tc.values``. The object pickles.
///
/// Parameters
/// ----------
/// values : ndarray, shape (n_lags,)
/// lags : ndarray of int, optional
///     Defaults to ``0, 1, …, n_lags - 1``.
/// dt : float, optional
///     Frame spacing; ``None`` means the lag axis is in frames.
/// kind : str
///     What was correlated: ``"acf"``, ``"vacf"``, ``"oacf"``, ``"msd"``,
///     ``"jacf"``, ``"dipole_acf"``, ``"cross"``, ``"onsager"``,
///     ``"survival"`` or ``"custom"``.
/// counts : ndarray of int, optional
///     Time origins per lag.
/// per_entity : ndarray, shape (n_lags, n_entities), optional
#[pyclass(module = "molrs.compute", name = "TimeCorrelation")]
pub struct PyTimeCorrelation {
    values: Array1<f64>,
    lags: Array1<usize>,
    counts: Option<Array1<usize>>,
    /// `(n_entities, n_lags)`, as the engine lays it out.
    per_entity: Option<Array2<f64>>,
    dt: Option<f64>,
    kind: String,
    normalization: Normalization,
    mean_subtract: bool,
    stride: usize,
    order: Option<u32>,
    n_frames: Option<usize>,
    n_entities: usize,
    extras: Option<Py<PyDict>>,
}

impl PyTimeCorrelation {
    fn dt_or_frames(&self) -> f64 {
        self.dt.unwrap_or(1.0)
    }

    fn lag_times_array(&self) -> Array1<f64> {
        let dt = self.dt_or_frames();
        self.lags.mapv(|l| l as f64 * dt)
    }

    fn require_dt(&self, what: &str) -> PyResult<f64> {
        self.dt.ok_or_else(|| {
            PyValueError::new_err(format!(
                "{what} needs the frame spacing: pass dt to the producer or use with_dt(dt)"
            ))
        })
    }

    fn clone_with(
        &self,
        py: Python<'_>,
        values: Array1<f64>,
        per_entity: Option<Array2<f64>>,
    ) -> Self {
        Self {
            values,
            lags: self.lags.clone(),
            counts: self.counts.clone(),
            per_entity,
            dt: self.dt,
            kind: self.kind.clone(),
            normalization: self.normalization,
            mean_subtract: self.mean_subtract,
            stride: self.stride,
            order: self.order,
            n_frames: self.n_frames,
            n_entities: self.n_entities,
            extras: self.extras.as_ref().map(|d| d.clone_ref(py)),
        }
    }

    pub(crate) fn set_extras(&mut self, extras: Py<PyDict>) {
        self.extras = Some(extras);
    }

    fn normalized_arrays(&self) -> PyResult<(Array1<f64>, Option<Array2<f64>>)> {
        let c0 = self.values[0];
        if !(c0.is_finite() && c0 != 0.0) {
            return Err(PyValueError::new_err(format!(
                "cannot normalize: C(0) = {c0}"
            )));
        }
        let values = self.values.mapv(|v| v / c0);
        let per_entity = self.per_entity.as_ref().map(|rows| {
            let mut out = rows.clone();
            for mut row in out.rows_mut() {
                let r0 = row[0];
                if r0.is_finite() && r0 != 0.0 {
                    row.mapv_inplace(|v| v / r0);
                }
            }
            out
        });
        Ok((values, per_entity))
    }
}

#[pymethods]
impl PyTimeCorrelation {
    #[new]
    #[pyo3(signature = (values, lags=None, dt=None, kind="custom", counts=None, per_entity=None, normalization="unbiased", mean_subtract=false, stride=1, order=None, n_frames=None, n_entities=None, extras=None))]
    #[allow(clippy::too_many_arguments)]
    fn new(
        py: Python<'_>,
        values: PyReadonlyArray1<'_, f64>,
        lags: Option<PyReadonlyArray1<'_, i64>>,
        dt: Option<f64>,
        kind: &str,
        counts: Option<PyReadonlyArray1<'_, i64>>,
        per_entity: Option<PyReadonlyArray2<'_, f64>>,
        normalization: &str,
        mean_subtract: bool,
        stride: usize,
        order: Option<u32>,
        n_frames: Option<usize>,
        n_entities: Option<usize>,
        extras: Option<Bound<'_, PyDict>>,
    ) -> PyResult<Self> {
        let values = values.as_array().to_owned();
        let n_lags = values.len();
        if n_lags == 0 {
            return Err(PyValueError::new_err("values must not be empty"));
        }
        let to_usize = |a: PyReadonlyArray1<'_, i64>, what: &str| -> PyResult<Array1<usize>> {
            let a = a.as_array();
            if a.len() != n_lags {
                return Err(PyValueError::new_err(format!(
                    "{what} must have the length of values ({n_lags}), got {}",
                    a.len()
                )));
            }
            a.iter()
                .map(|&v| {
                    usize::try_from(v).map_err(|_| {
                        PyValueError::new_err(format!("{what} must be non-negative, got {v}"))
                    })
                })
                .collect()
        };
        let lags = match lags {
            Some(l) => to_usize(l, "lags")?,
            None => Array1::from_iter(0..n_lags),
        };
        let counts = counts.map(|c| to_usize(c, "counts")).transpose()?;
        let per_entity = per_entity
            .map(|p| {
                let p = p.as_array();
                if p.shape()[0] != n_lags {
                    return Err(PyValueError::new_err(format!(
                        "per_entity must have shape (n_lags, n_entities) = ({n_lags}, …), got {:?}",
                        p.shape()
                    )));
                }
                Ok(p.t().to_owned())
            })
            .transpose()?;
        if let Some(dt) = dt
            && !(dt > 0.0 && dt.is_finite())
        {
            return Err(PyValueError::new_err(format!(
                "dt must be positive, got {dt}"
            )));
        }
        if stride == 0 {
            return Err(PyValueError::new_err("stride must be >= 1"));
        }
        let n_entities = match (n_entities, per_entity.as_ref()) {
            (Some(n), _) => n,
            (None, Some(rows)) => rows.nrows(),
            (None, None) => 1,
        };
        Ok(Self {
            values,
            lags,
            counts,
            per_entity,
            dt,
            kind: kind.to_string(),
            normalization: parse_normalization(normalization)?,
            mean_subtract,
            stride,
            order,
            n_frames,
            n_entities,
            extras: extras.map(|d| d.unbind()),
        })
        .map(|mut s| {
            // Keep the dict our own: a later mutation of the caller's dict
            // must not change a result.
            if let Some(d) = s.extras.take() {
                s.extras = d.bind(py).copy().ok().map(Bound::unbind);
            }
            s
        })
    }

    /// Lags in frames, ``int64``.
    #[getter]
    fn lags<'py>(&self, py: Python<'py>) -> Bound<'py, PyArray1<i64>> {
        self.lags.mapv(|l| l as i64).into_pyarray(py)
    }

    /// ``lags * dt``, or ``None`` when no ``dt`` is known.
    #[getter]
    fn lag_times<'py>(&self, py: Python<'py>) -> Option<Bound<'py, PyArray1<f64>>> {
        self.dt.map(|_| self.lag_times_array().into_pyarray(py))
    }

    /// ``C(t)``, one value per lag.
    #[getter]
    fn values<'py>(&self, py: Python<'py>) -> Bound<'py, PyArray1<f64>> {
        self.values.clone().into_pyarray(py)
    }

    /// Time origins per lag (``int64``), or ``None`` when unknown.
    #[getter]
    fn counts<'py>(&self, py: Python<'py>) -> Option<Bound<'py, PyArray1<i64>>> {
        self.counts
            .as_ref()
            .map(|c| c.mapv(|v| v as i64).into_pyarray(py))
    }

    /// Per-entity curves, shape ``(n_lags, n_entities)``, when they were
    /// requested; ``values`` is their mean (sum, for ``"raw"``).
    #[getter]
    fn per_entity<'py>(&self, py: Python<'py>) -> Option<Bound<'py, PyArray2<f64>>> {
        self.per_entity
            .as_ref()
            .map(|rows| rows.t().to_owned().into_pyarray(py))
    }

    /// Frame spacing, or ``None``.
    #[getter]
    fn dt(&self) -> Option<f64> {
        self.dt
    }

    /// What was correlated.
    #[getter]
    fn kind(&self) -> &str {
        &self.kind
    }

    /// ``"unbiased"``, ``"biased"`` or ``"raw"``.
    #[getter]
    fn normalization(&self) -> &'static str {
        name_of(self.normalization)
    }

    /// Whether each column's time mean was removed before correlating.
    #[getter]
    fn mean_subtract(&self) -> bool {
        self.mean_subtract
    }

    /// Spacing of the time origins, in frames.
    #[getter]
    fn stride(&self) -> usize {
        self.stride
    }

    /// Legendre order of an orientational correlation; ``None`` otherwise.
    #[getter]
    fn order(&self) -> Option<u32> {
        self.order
    }

    /// Frames in the series the curve came from, when known.
    #[getter]
    fn n_frames(&self) -> Option<usize> {
        self.n_frames
    }

    /// Entities (atoms, molecules, vectors) averaged over.
    #[getter]
    fn n_entities(&self) -> usize {
        self.n_entities
    }

    /// Producer-specific scalars (``DebyeRelaxation``: ``volume``,
    /// ``temperature``, ``boundary``, ``zero_lag_variance``), or ``None``.
    #[getter]
    fn extras<'py>(&self, py: Python<'py>) -> PyResult<Option<Bound<'py, PyDict>>> {
        self.extras.as_ref().map(|d| d.bind(py).copy()).transpose()
    }

    /// The same curve with the frame spacing ``dt`` attached.
    fn with_dt(&self, py: Python<'_>, dt: f64) -> PyResult<Self> {
        if !(dt > 0.0 && dt.is_finite()) {
            return Err(PyValueError::new_err(format!(
                "dt must be positive, got {dt}"
            )));
        }
        let mut out = self.clone_with(py, self.values.clone(), self.per_entity.clone());
        out.dt = Some(dt);
        Ok(out)
    }

    /// ``C(t) / C(0)`` (rows included). ``ValueError`` when ``C(0)`` is zero —
    /// a displacement correlation has no normalized form.
    fn normalized(&self, py: Python<'_>) -> PyResult<Self> {
        let (values, per_entity) = self.normalized_arrays()?;
        Ok(self.clone_with(py, values, per_entity))
    }

    /// Trapezoidal integral of the curve over its first ``n_lags`` samples
    /// (all by default) on the ``dt`` grid (frames when no ``dt``).
    ///
    /// Returns the total as a float, or the running integral (an array,
    /// ``integral[0] == 0``) with ``running=True``. ``CumulativeTrapezoid``
    /// on the same arrays.
    #[pyo3(signature = (n_lags=None, running=false))]
    fn integrate<'py>(
        &self,
        py: Python<'py>,
        n_lags: Option<usize>,
        running: bool,
    ) -> PyResult<Bound<'py, PyAny>> {
        let r = CumulativeTrapezoid
            .fit((&self.values, self.dt_or_frames(), n_lags))
            .map_err(py_value_err)?;
        if running {
            r.integral.into_pyarray(py).into_bound_py_any(py)
        } else {
            r.integral[r.integral.len() - 1].into_bound_py_any(py)
        }
    }

    /// Ordinary least squares ``C(t) = slope * t + intercept`` over
    /// ``window`` (fractions of the last lag) on the ``lag_times`` axis:
    /// ``LinearFit`` on the same arrays.
    ///
    /// Returns ``{"slope", "intercept", "r2", "fit_start", "fit_end"}``.
    fn linear_fit<'py>(&self, py: Python<'py>, window: (f64, f64)) -> PyResult<Bound<'py, PyDict>> {
        let r = LinearFit { window }
            .fit((&self.lag_times_array(), &self.values))
            .map_err(py_value_err)?;
        let d = PyDict::new(py);
        d.set_item("slope", r.slope)?;
        d.set_item("intercept", r.intercept)?;
        d.set_item("r2", r.r2)?;
        d.set_item("fit_start", r.fit_start)?;
        d.set_item("fit_end", r.fit_end)?;
        Ok(d)
    }

    /// Single-exponential fit ``C(t)/C(0) = A * exp(-t / tau)``: ``DebyeFit``
    /// on the normalized curve with this ``dt``.
    ///
    /// Returns ``{"tau", "amplitude", "n_samples"}``.
    fn fit_exponential<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyDict>> {
        let (phi, _) = self.normalized_arrays()?;
        let r = DebyeFit
            .fit((&phi, self.dt_or_frames()))
            .map_err(py_value_err)?;
        let d = PyDict::new(py);
        d.set_item("tau", r.tau)?;
        d.set_item("amplitude", r.amplitude)?;
        d.set_item("n_samples", r.n_samples)?;
        Ok(d)
    }

    /// The diffusion coefficient this curve gives, in ``[values] / [dt]``
    /// units (``[length]² / [dt]`` for a displacement).
    ///
    /// * ``"msd"`` / ``"onsager"`` — Einstein: the ``linear_fit`` slope over
    ///   ``window``, divided by ``2 * n_dims``.
    /// * ``"vacf"`` / ``"acf"`` — Green–Kubo: the plateau (``Plateau`` over
    ///   ``window``) of the running integral, divided by ``n_dims``.
    ///
    /// Any other kind raises ``ValueError``.
    fn diffusion(&self, n_dims: usize, window: (f64, f64)) -> PyResult<f64> {
        if n_dims == 0 {
            return Err(PyValueError::new_err("n_dims must be >= 1"));
        }
        match self.kind.as_str() {
            "msd" | "onsager" => {
                let fit = LinearFit { window }
                    .fit((&self.lag_times_array(), &self.values))
                    .map_err(py_value_err)?;
                Ok(fit.slope / (2.0 * n_dims as f64))
            }
            "vacf" | "acf" => {
                let running = CumulativeTrapezoid
                    .fit((&self.values, self.dt_or_frames(), None))
                    .map_err(py_value_err)?
                    .integral;
                let plateau = Plateau { window }.fit(&running).map_err(py_value_err)?;
                Ok(plateau.value / n_dims as f64)
            }
            other => Err(PyValueError::new_err(format!(
                "diffusion() needs a curve of kind 'msd', 'onsager', 'vacf' or 'acf', not {other:?}"
            ))),
        }
    }

    /// Windowed one-sided power spectrum of the curve, ``dt`` read as
    /// femtoseconds: ``PowerSpectrum(window, pad_factor).fit(values, dt)``.
    #[pyo3(signature = (window="cosine_sq", pad_factor=4))]
    fn spectrum(&self, window: &str, pad_factor: usize) -> PyResult<PySpectrum> {
        let dt_fs = self.require_dt("spectrum()")?;
        let window = parse_window(window)?;
        let r = PowerSpectrum { window, pad_factor }
            .fit((&self.values, dt_fs))
            .map_err(py_value_err)?;
        Ok(PySpectrum {
            frequencies_cm1: r.frequencies_cm1,
            intensities: r.intensities,
            resolution: r.resolution,
            n_frames: r.n_frames,
            window,
            dt_fs: Some(dt_fs),
        })
    }

    /// Every field as a plain ``dict`` of arrays and scalars.
    fn to_dict<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyDict>> {
        let d = PyDict::new(py);
        d.set_item("lags", self.lags(py))?;
        d.set_item("lag_times", self.lag_times(py))?;
        d.set_item("values", self.values(py))?;
        d.set_item("counts", self.counts(py))?;
        d.set_item("per_entity", self.per_entity(py))?;
        d.set_item("dt", self.dt)?;
        d.set_item("kind", &self.kind)?;
        d.set_item("normalization", self.normalization())?;
        d.set_item("mean_subtract", self.mean_subtract)?;
        d.set_item("stride", self.stride)?;
        d.set_item("order", self.order)?;
        d.set_item("n_frames", self.n_frames)?;
        d.set_item("n_entities", self.n_entities)?;
        d.set_item("extras", self.extras(py)?)?;
        Ok(d)
    }

    fn __len__(&self) -> usize {
        self.values.len()
    }

    #[pyo3(signature = (dtype=None, copy=None))]
    fn __array__<'py>(
        &self,
        py: Python<'py>,
        dtype: Option<&Bound<'py, PyAny>>,
        copy: Option<bool>,
    ) -> PyResult<Bound<'py, PyAny>> {
        let _ = copy;
        let arr = self.values(py).into_any();
        match dtype {
            Some(dtype) => arr.call_method1("astype", (dtype,)),
            None => Ok(arr),
        }
    }

    fn __repr__(&self) -> String {
        let order = self.order.map_or(String::new(), |o| format!(", order={o}"));
        let dt = self.dt.map_or("None".to_string(), |d| d.to_string());
        format!(
            "TimeCorrelation(kind={:?}, n_lags={}, n_entities={}, dt={}, normalization={:?}{})",
            self.kind,
            self.values.len(),
            self.n_entities,
            dt,
            self.normalization(),
            order
        )
    }

    fn __reduce__<'py>(slf: &Bound<'py, Self>) -> PyResult<Bound<'py, PyTuple>> {
        let py = slf.py();
        let this = slf.borrow();
        let args = PyTuple::new(
            py,
            [
                this.values(py).into_any(),
                this.lags(py).into_any(),
                this.dt.into_bound_py_any(py)?,
                this.kind.as_str().into_bound_py_any(py)?,
                this.counts(py).into_bound_py_any(py)?,
                this.per_entity(py).into_bound_py_any(py)?,
                this.normalization().into_bound_py_any(py)?,
                this.mean_subtract.into_bound_py_any(py)?,
                this.stride.into_bound_py_any(py)?,
                this.order.into_bound_py_any(py)?,
                this.n_frames.into_bound_py_any(py)?,
                this.n_entities.into_bound_py_any(py)?,
                this.extras(py)?.into_bound_py_any(py)?,
            ],
        )?;
        crate::pickle::reduce_via_type(slf.as_any(), args)
    }
}

// ═══════════════════════════════════════════════════════════════════════════
// Spectrum
// ═══════════════════════════════════════════════════════════════════════════

/// A one-sided spectrum on a wavenumber grid: what ``PowerSpectrum``,
/// ``IrSpectrum``, ``VcdSpectrum`` and ``TimeCorrelation.spectrum`` return.
///
/// ``resolution`` is the number of lags the correlation had (a count, not a
/// frequency) and ``n_frames`` that plus one; ``window`` and ``dt_fs`` are
/// the transform's settings. The object pickles.
#[pyclass(module = "molrs.compute", name = "Spectrum")]
pub struct PySpectrum {
    frequencies_cm1: Array1<f64>,
    intensities: Array1<f64>,
    resolution: usize,
    n_frames: usize,
    window: WindowType,
    dt_fs: Option<f64>,
}

impl PySpectrum {
    pub(crate) fn from_result(
        r: molrs::compute::SpectrumResult,
        window: WindowType,
        dt_fs: f64,
    ) -> Self {
        Self {
            frequencies_cm1: r.frequencies_cm1,
            intensities: r.intensities,
            resolution: r.resolution,
            n_frames: r.n_frames,
            window,
            dt_fs: Some(dt_fs),
        }
    }
}

#[pymethods]
impl PySpectrum {
    #[new]
    #[pyo3(signature = (frequencies_cm1, intensities, resolution=None, n_frames=None, window="cosine_sq", dt_fs=None))]
    fn new(
        frequencies_cm1: PyReadonlyArray1<'_, f64>,
        intensities: PyReadonlyArray1<'_, f64>,
        resolution: Option<usize>,
        n_frames: Option<usize>,
        window: &str,
        dt_fs: Option<f64>,
    ) -> PyResult<Self> {
        let frequencies_cm1 = frequencies_cm1.as_array().to_owned();
        let intensities = intensities.as_array().to_owned();
        if frequencies_cm1.len() != intensities.len() {
            return Err(PyValueError::new_err(format!(
                "frequencies_cm1 and intensities must have one length, got {} and {}",
                frequencies_cm1.len(),
                intensities.len()
            )));
        }
        let n = intensities.len();
        Ok(Self {
            frequencies_cm1,
            intensities,
            resolution: resolution.unwrap_or(n.saturating_sub(1)),
            n_frames: n_frames.unwrap_or(n),
            window: parse_window(window)?,
            dt_fs,
        })
    }

    #[getter]
    fn frequencies_cm1<'py>(&self, py: Python<'py>) -> Bound<'py, PyArray1<f64>> {
        self.frequencies_cm1.clone().into_pyarray(py)
    }

    #[getter]
    fn intensities<'py>(&self, py: Python<'py>) -> Bound<'py, PyArray1<f64>> {
        self.intensities.clone().into_pyarray(py)
    }

    /// Lags of the correlation the spectrum came from (a count).
    #[getter]
    fn resolution(&self) -> usize {
        self.resolution
    }

    #[getter]
    fn n_frames(&self) -> usize {
        self.n_frames
    }

    /// The taper applied before the FFT.
    #[getter]
    fn window(&self) -> &'static str {
        window_name(self.window)
    }

    /// The time step the frequency grid was built from, in fs.
    #[getter]
    fn dt_fs(&self) -> Option<f64> {
        self.dt_fs
    }

    /// ``{"frequencies_cm1", "intensities", "resolution", "n_frames",
    /// "window", "dt_fs"}``.
    fn to_dict<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyDict>> {
        let d = PyDict::new(py);
        d.set_item("frequencies_cm1", self.frequencies_cm1(py))?;
        d.set_item("intensities", self.intensities(py))?;
        d.set_item("resolution", self.resolution)?;
        d.set_item("n_frames", self.n_frames)?;
        d.set_item("window", self.window())?;
        d.set_item("dt_fs", self.dt_fs)?;
        Ok(d)
    }

    fn __len__(&self) -> usize {
        self.intensities.len()
    }

    fn __repr__(&self) -> String {
        format!(
            "Spectrum(n_bins={}, window={:?}, resolution={}, dt_fs={})",
            self.intensities.len(),
            self.window(),
            self.resolution,
            self.dt_fs.map_or("None".to_string(), |d| d.to_string())
        )
    }

    fn __reduce__<'py>(slf: &Bound<'py, Self>) -> PyResult<Bound<'py, PyTuple>> {
        let py = slf.py();
        let this = slf.borrow();
        let args = PyTuple::new(
            py,
            [
                this.frequencies_cm1(py).into_any(),
                this.intensities(py).into_any(),
                this.resolution.into_bound_py_any(py)?,
                this.n_frames.into_bound_py_any(py)?,
                this.window().into_bound_py_any(py)?,
                this.dt_fs.into_bound_py_any(py)?,
            ],
        )?;
        crate::pickle::reduce_via_type(slf.as_any(), args)
    }
}

// ═══════════════════════════════════════════════════════════════════════════
// Acf
// ═══════════════════════════════════════════════════════════════════════════

/// Time-autocorrelation of a vector series, averaged over all time origins.
///
/// ``C(t) = norm · Σ_i Σ_c Σ_τ v_ic(τ) · v_ic(τ+t)`` over an
/// ``(n_frames, n_entities, n_components)`` series: the inner product runs
/// over components, the sum over entities, the average over the time
/// origins ``τ = 0, stride, 2·stride, …``. ``normalization`` is
/// ``"unbiased"`` (``Σ / (N · counts[t])``, the multiple-time-origin
/// estimator), ``"biased"`` (``Σ / (N · counts[0])``) or ``"raw"`` (the
/// plain sum). With ``mean_subtract`` each ``(entity, component)`` column
/// loses its own time mean first. Wiener–Khinchin (FFT) route, O(T log T).
///
/// :class:`Vacf` is this estimator with ``mean_subtract=True`` on the
/// ``(n_frames, n_dof, 1)`` view of a velocity array.
///
/// Examples
/// --------
/// >>> c = molrs.compute.Acf().compute(series, max_lag=50, dt=0.5)
/// >>> c.values[0], c.integrate()
#[pyclass(module = "molrs.compute", name = "Acf")]
pub struct PyAcf {
    settings: Settings,
}

#[pymethods]
impl PyAcf {
    #[new]
    #[pyo3(signature = (mean_subtract=false, normalization="unbiased", stride=1))]
    fn new(mean_subtract: bool, normalization: &str, stride: usize) -> PyResult<Self> {
        Ok(Self {
            settings: Settings::parse(mean_subtract, normalization, stride)?,
        })
    }

    /// ``C(t)`` for ``t = 0..=max_lag`` (clamped to ``n_frames - 1``).
    ///
    /// Parameters
    /// ----------
    /// series : ndarray, shape (n_frames, n_entities, n_components)
    /// max_lag : int
    /// dt : float, optional
    ///     Frame spacing for ``lag_times``; frames when omitted.
    /// per_entity : bool
    ///     Also return one curve per entity.
    #[pyo3(signature = (series, max_lag, dt=None, per_entity=false))]
    fn compute(
        &self,
        series: PyReadonlyArray3<'_, f64>,
        max_lag: usize,
        dt: Option<f64>,
        per_entity: bool,
    ) -> PyResult<PyTimeCorrelation> {
        let owned = series.as_array().to_owned();
        let tc = autocorrelate(
            owned.view(),
            dt.unwrap_or(1.0),
            &self.settings.spec(max_lag, per_entity),
        )
        .map_err(py_value_err)?;
        Ok(finish(tc, dt, "acf", self.settings, owned.dim().0, None))
    }

    #[getter]
    fn mean_subtract(&self) -> bool {
        self.settings.mean_subtract
    }

    #[getter]
    fn normalization(&self) -> &'static str {
        name_of(self.settings.normalization)
    }

    #[getter]
    fn stride(&self) -> usize {
        self.settings.stride
    }

    fn __repr__(&self) -> String {
        format!(
            "Acf(mean_subtract={}, normalization={:?}, stride={})",
            if self.settings.mean_subtract {
                "True"
            } else {
                "False"
            },
            self.normalization(),
            self.settings.stride
        )
    }
}

// ═══════════════════════════════════════════════════════════════════════════
// Oacf
// ═══════════════════════════════════════════════════════════════════════════

/// Orientational autocorrelation function of a unit-vector series,
/// ``C_ℓ(t) = ⟨P_ℓ(û(τ) · û(τ+t))⟩`` averaged over vectors and time origins.
///
/// ``order`` is the Legendre rank: 1 (``⟨cos θ⟩``, what dielectric
/// relaxation and infrared absorption see) or 2 (``⟨(3cos²θ − 1)/2⟩``, what
/// NMR, Raman and fluorescence anisotropy see); higher orders fall back to
/// the direct origin sum. ``C_ℓ(0) = 1`` and the integral of the curve is
/// the reorientational correlation time ``τ_ℓ``
/// (:meth:`TimeCorrelation.integrate`).
///
/// You build the unit vectors — which atoms, which minimum-image
/// convention — and pass them as an ``(n_frames, n_vectors, 3)`` array;
/// ``|û|² ≠ 1`` is a ``ValueError``, never silently renormalized.
/// ``LegendreReorientation`` is the front end that builds them from a
/// frame's ``bonds`` block.
///
/// Examples
/// --------
/// >>> u = np.stack([h - o for ...]); u /= np.linalg.norm(u, axis=-1, keepdims=True)
/// >>> c2 = molrs.compute.Oacf(order=2).compute(u, max_lag=2000, dt=0.05)
/// >>> tau2 = c2.integrate()
#[pyclass(module = "molrs.compute", name = "Oacf")]
pub struct PyOacf {
    order: u32,
    stride: usize,
}

#[pymethods]
impl PyOacf {
    #[new]
    #[pyo3(signature = (order=1, stride=1))]
    fn new(order: u32, stride: usize) -> PyResult<Self> {
        if order == 0 {
            return Err(PyValueError::new_err("order must be >= 1"));
        }
        if stride == 0 {
            return Err(PyValueError::new_err("stride must be >= 1"));
        }
        Ok(Self { order, stride })
    }

    /// ``C_ℓ(t)`` for ``t = 0..=max_lag`` (clamped to ``n_frames - 1``).
    ///
    /// Parameters
    /// ----------
    /// vectors : ndarray, shape (n_frames, n_vectors, 3)
    ///     Unit vectors.
    /// max_lag : int
    /// dt : float, optional
    /// per_entity : bool
    ///     Also return one curve per vector.
    #[pyo3(signature = (vectors, max_lag, dt=None, per_entity=false))]
    fn compute(
        &self,
        vectors: PyReadonlyArray3<'_, f64>,
        max_lag: usize,
        dt: Option<f64>,
        per_entity: bool,
    ) -> PyResult<PyTimeCorrelation> {
        let owned: Array3<f64> = vectors.as_array().to_owned();
        let tc = Oacf::new(self.order, max_lag)
            .with_origin_stride(self.stride)
            .with_per_entity(per_entity)
            .correlate(&owned, dt.unwrap_or(1.0))
            .map_err(py_value_err)?;
        let settings = Settings {
            mean_subtract: false,
            normalization: Normalization::Unbiased,
            stride: self.stride,
        };
        Ok(finish(
            tc,
            dt,
            "oacf",
            settings,
            owned.dim().0,
            Some(self.order),
        ))
    }

    #[getter]
    fn order(&self) -> u32 {
        self.order
    }

    #[getter]
    fn stride(&self) -> usize {
        self.stride
    }

    fn __repr__(&self) -> String {
        format!("Oacf(order={}, stride={})", self.order, self.stride)
    }
}

/// Register this domain's classes.
pub(crate) fn register(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_class::<PyTimeCorrelation>()?;
    m.add_class::<PySpectrum>()?;
    m.add_class::<PyAcf>()?;
    m.add_class::<PyOacf>()?;
    Ok(())
}
