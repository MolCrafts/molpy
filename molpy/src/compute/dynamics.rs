//! Pair dynamics (`molrs::compute::dynamics`): `PairPresence` (which pairs
//! are bound at each frame, with hysteresis) and the van Hove correlation
//! `VanHove`; the `Presence` series `Survival` consumes.

use crate::array::ArrayArg3;
use molrs::compute::{
    Compute, PairPresence, PairPresenceInput, Presence, VanHove, VanHoveInput, VanHoveResult,
};
use ndarray::Array2;
use pyo3::prelude::*;

use super::args::{self, PairSourceArg};
use super::error::Result;
use super::output::{ToPy, py_result};
use super::spec::described;
use super::values::PyAxis;
use crate::core::simbox::PyBox;

/// A list of index pairs is an ``(n, 2)`` uint32 array.
impl ToPy for Vec<(u32, u32)> {
    fn to_py<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyAny>> {
        let flat: Vec<u32> = self.iter().flat_map(|&(i, j)| [i, j]).collect();
        Array2::from_shape_vec((self.len(), 2), flat)
            .expect("two columns per pair")
            .to_py(py)
    }
}

py_result! {
    /// A presence series (``molrs::compute::Presence``): ``values[t, k]`` is
    /// 1 when pair ``pairs[k]`` is present at frame ``t``. Pass ``values`` to
    /// ``Survival`` as its ``presence``.
    PyPresence(Presence) as "Presence" { pairs, values }
}

py_result! {
    /// The self and distinct van Hove functions at each lag
    /// (``molrs::compute::VanHoveResult``).
    PyVanHoveResult(VanHoveResult) as "VanHoveResult" {
        r_edges,
        r_centers,
        lags,
        g_self,
        g_distinct,
        dr,
    }
}

/// Pair presence with hysteresis over frames
/// (``molrs::compute::PairPresence``): a pair enters at distance ``≤ r0`` and
/// stays while within ``r1``.
///
/// Parameters
/// ----------
/// r0 : float
/// exclude_self : bool
///     Drop ``i == j`` pairs.
/// r1 : float, optional
///     The exit distance (``≥ r0``).
///
/// ``compute(frames=)`` takes one pair source (``Neighbors`` or
/// ``NeighborList``) per frame and returns a ``Presence``, or ``None`` when no
/// pair is ever present.
#[pyclass(module = "molpy.compute", name = "PairPresence", frozen)]
pub(crate) struct PyPairPresence {
    inner: PairPresence,
}

#[pymethods]
impl PyPairPresence {
    #[new]
    #[pyo3(signature = (r0, exclude_self, *, r1 = None))]
    fn new(r0: f64, exclude_self: bool, r1: Option<f64>) -> Result<Self> {
        let mut inner = PairPresence::new(r0, exclude_self)?;
        if let Some(r1) = r1 {
            inner = inner.with_r1(r1)?;
        }
        Ok(Self { inner })
    }

    #[pyo3(signature = (*, frames))]
    fn compute(&self, frames: Vec<Bound<'_, PyAny>>) -> Result<Option<PyPresence>> {
        let frames = PairSourceArg::extract_all(&frames)?;
        let frames = args::sources(&frames)?;
        let out = self.inner.compute(PairPresenceInput { frames: &frames })?;
        Ok(out.map(PyPresence::from))
    }
}

described!(PyPairPresence, PairPresence);

/// The van Hove correlation G(r, t) over a trajectory
/// (``molrs::compute::VanHove``).
///
/// Parameters
/// ----------
/// axis : Axis
///     The distance bins.
/// lags : list[int]
///     The lags, in frames.
/// origins : str, optional
///     ``"all"`` or ``"first"``.
/// stride : int, optional
///     Origin stride.
/// upper_edge : str, optional
///     ``"open"`` (half-open bins) or ``"closed"`` (``r_max`` in the last
///     bin).
///
/// ``compute(positions=, boxes=)``: an unwrapped ``(T, N, 3)`` trajectory
/// and one box per frame.
#[pyclass(module = "molpy.compute", name = "VanHove", frozen)]
pub(crate) struct PyVanHove {
    inner: VanHove,
}

#[pymethods]
impl PyVanHove {
    #[new]
    #[pyo3(signature = (axis, lags, *, origins = None, stride = None, upper_edge = None))]
    fn new(
        axis: PyRef<'_, PyAxis>,
        lags: Vec<usize>,
        origins: Option<&str>,
        stride: Option<usize>,
        upper_edge: Option<&str>,
    ) -> Result<Self> {
        let mut inner = VanHove::new(axis.inner, lags)?;
        if let Some(origins) = origins {
            inner = inner.with_origins(origins.parse()?)?;
        }
        if let Some(stride) = stride {
            inner = inner.with_stride(stride)?;
        }
        if let Some(upper_edge) = upper_edge {
            inner = inner.with_upper_edge(upper_edge.parse()?);
        }
        Ok(Self { inner })
    }

    #[pyo3(signature = (*, positions, boxes))]
    fn compute(
        &self,
        positions: ArrayArg3<'_, f64>,
        boxes: Vec<PyRef<'_, PyBox>>,
    ) -> Result<PyVanHoveResult> {
        let boxes = args::boxes(&boxes);
        let out = self.inner.compute(VanHoveInput {
            positions: positions.as_array(),
            boxes: &boxes,
        })?;
        Ok(out.into())
    }
}

described!(PyVanHove, VanHove);

pub(crate) fn register(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_class::<PyPresence>()?;
    m.add_class::<PyVanHoveResult>()?;
    m.add_class::<PyPairPresence>()?;
    m.add_class::<PyVanHove>()?;
    Ok(())
}
