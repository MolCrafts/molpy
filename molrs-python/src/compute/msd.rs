//! Mean-squared displacement (`molrs::compute::msd`): `Msd`, a
//! `TimeCorrelation` of kind ``"msd"``.

use super::collect_frames;
use super::correlation::{PyTimeCorrelation, Settings, finish};
use crate::error::py_value_err;
use molrs::compute::{
    Compute, Msd, MsdMode, Normalization, TimeCorrelationSpec, displacement_correlate,
    position_series,
};
use molrs::core::{Block, Frame as CoreFrame};
use ndarray::{Array1, Array2, Array3};
use numpy::PyReadonlyArray3;
use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;
use pyo3::types::PyAny;

/// The `(n_frames, n_atoms, 3)` positions of `frames_or_positions`: an
/// ndarray as given, frames through [`position_series`].
pub(crate) fn positions_of(frames_or_positions: &Bound<'_, PyAny>) -> PyResult<Array3<f64>> {
    if let Ok(arr) = frames_or_positions.extract::<PyReadonlyArray3<'_, f64>>() {
        let a = arr.as_array();
        if a.shape()[2] != 3 {
            return Err(PyValueError::new_err(format!(
                "positions must have shape (n_frames, n_atoms, 3), got {:?}",
                a.shape()
            )));
        }
        return Ok(a.to_owned());
    }
    let owned = collect_frames(frames_or_positions)?;
    let refs: Vec<&CoreFrame> = owned.iter().collect();
    position_series(&refs).map_err(py_value_err)
}

/// Frames carrying only `x`/`y`/`z`, for the single-origin estimator.
fn frames_from_positions(positions: &Array3<f64>) -> Vec<CoreFrame> {
    let (t, _, _) = positions.dim();
    (0..t)
        .map(|k| {
            let mut block = Block::new();
            for (c, name) in ["x", "y", "z"].into_iter().enumerate() {
                let column: Array1<f64> = positions.slice(ndarray::s![k, .., c]).to_owned();
                block
                    .insert(name, column.into_dyn())
                    .expect("fresh block accepts a column");
            }
            let mut frame = CoreFrame::new();
            frame.insert("atoms", block);
            frame
        })
        .collect()
}

/// Mean squared displacement, ``MSD(t) = ⟨|r(τ+t) − r(τ)|²⟩``.
///
/// Two estimators of the same quantity, chosen by ``method``; they are not
/// interchangeable and there is no default that suits both:
///
/// - ``"direct"`` (default) — frame 0 is the one time origin:
///   ``MSD(t) = ⟨|r(t) − r(0)|²⟩``.
/// - ``"window"`` — averaged over **every** time origin ``τ`` (every
///   ``stride``-th frame). Far better statistics at long lag, which is what a
///   diffusion coefficient needs, and O(T log T) through the correlation
///   engine's displacement route rather than the O(T²) nested loop.
///   Conventions match ``freud.msd``.
///
/// ``compute`` takes a ``Frame``, a sequence of frames, or an
/// ``(n_frames, n_atoms, 3)`` array of **unwrapped** positions, and returns a
/// :class:`TimeCorrelation` of kind ``"msd"`` whose ``values`` is the
/// atom-averaged curve and whose ``per_entity`` holds one curve per atom.
///
/// Examples
/// --------
/// >>> msd = molrs.compute.Msd(method="window").compute(frames, dt=1.0)
/// >>> D = msd.diffusion(3, window=(0.2, 0.8))
#[pyclass(module = "molrs.compute", name = "Msd")]
pub struct PyMsd {
    mode: MsdMode,
    stride: usize,
}

#[pymethods]
impl PyMsd {
    #[new]
    #[pyo3(signature = (method = "direct", stride = 1))]
    fn new(method: &str, stride: usize) -> PyResult<Self> {
        let mode = match method {
            "direct" => MsdMode::Direct,
            "window" => MsdMode::Window,
            other => {
                return Err(PyValueError::new_err(format!(
                    "unknown MSD method {other:?}; expected 'direct' or 'window'"
                )));
            }
        };
        if stride == 0 {
            return Err(PyValueError::new_err("stride must be >= 1"));
        }
        if mode == MsdMode::Direct && stride != 1 {
            return Err(PyValueError::new_err(
                "stride applies to method='window'; the direct estimator has one origin",
            ));
        }
        Ok(Self { mode, stride })
    }

    /// The estimator this analyzer uses: ``"direct"`` or ``"window"``.
    #[getter]
    fn method(&self) -> &'static str {
        match self.mode {
            MsdMode::Direct => "direct",
            MsdMode::Window => "window",
        }
    }

    /// Spacing of the time origins (window mode).
    #[getter]
    fn stride(&self) -> usize {
        self.stride
    }

    /// The MSD curve.
    ///
    /// Parameters
    /// ----------
    /// frames_or_positions : Frame | Sequence[Frame] | ndarray (n_frames, n_atoms, 3)
    /// dt : float, optional
    ///     Frame spacing for ``lag_times``; frames when omitted.
    /// max_lag : int, optional
    ///     Longest lag in frames; every lag by default.
    /// per_entity : bool
    ///     Also return one curve per atom (default ``True``).
    #[pyo3(signature = (frames_or_positions, dt=None, max_lag=None, per_entity=true))]
    fn compute(
        &self,
        frames_or_positions: &Bound<'_, PyAny>,
        dt: Option<f64>,
        max_lag: Option<usize>,
        per_entity: bool,
    ) -> PyResult<PyTimeCorrelation> {
        let positions = positions_of(frames_or_positions)?;
        let (n_frames, n_atoms, _) = positions.dim();
        if n_frames == 0 {
            return Err(PyValueError::new_err("Msd.compute requires >= 1 frame"));
        }
        let settings = Settings {
            mean_subtract: false,
            normalization: Normalization::Unbiased,
            stride: self.stride,
        };
        match self.mode {
            MsdMode::Window => {
                if n_frames < 2 {
                    return Err(PyValueError::new_err(
                        "the window estimator needs at least two frames",
                    ));
                }
                let spec = TimeCorrelationSpec::new()
                    .max_lag(max_lag.unwrap_or(usize::MAX))
                    .origin_stride(self.stride)
                    .per_entity(per_entity);
                let tc = displacement_correlate(positions.view(), None, dt.unwrap_or(1.0), &spec)
                    .map_err(py_value_err)?;
                Ok(finish(tc, dt, "msd", settings, n_frames, None))
            }
            MsdMode::Direct => {
                let frames = frames_from_positions(&positions);
                let refs: Vec<&CoreFrame> = frames.iter().collect();
                let series = Msd::with_mode(MsdMode::Direct)
                    .compute(&refs, ())
                    .map_err(py_value_err)?;
                let values = Array1::from_iter(series.per_frame.iter().map(|r| r.mean));
                let mut out = PyTimeCorrelation::from_curve(
                    values,
                    dt,
                    "msd",
                    settings,
                    Some(Array1::from_elem(n_frames, 1)),
                    n_frames,
                );
                if per_entity {
                    let mut rows = Array2::<f64>::zeros((n_atoms, n_frames));
                    for (t, r) in series.per_frame.iter().enumerate() {
                        for (i, v) in r.per_particle.iter().enumerate() {
                            rows[[i, t]] = *v;
                        }
                    }
                    out = out.with_rows(rows);
                } else {
                    out = out.with_entities(n_atoms);
                }
                Ok(match max_lag {
                    Some(m) => out.truncated(m + 1),
                    None => out,
                })
            }
        }
    }

    fn __repr__(&self) -> String {
        format!("Msd(method={:?}, stride={})", self.method(), self.stride)
    }
}

/// Register this domain's classes and functions.
pub(crate) fn register(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_class::<PyMsd>()?;
    Ok(())
}
