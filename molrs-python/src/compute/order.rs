//! Order parameters (`molrs::compute::order`): Steinhardt, nematic, hexatic,
//! solid-liquid and Legendre reorientation.

#![allow(clippy::type_complexity)]

use super::{collect_frames, collect_neighbors};
use crate::error::py_value_err;
use molrs::compute::{
    AtomGroups, Hexatic, LegacyCompute, LegendreReorientation, LegendreReorientationResult,
    Nematic, SolidLiquid, Steinhardt,
};
use molrs::core::Frame as CoreFrame;
use molrs::op::F;
use ndarray::{Array1, Array2};
use numpy::{IntoPyArray, PyArray1, PyArray2};
use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;
use pyo3::types::{PyAny, PyDict, PyDictMethods};

// ---------------------------------------------------------------------------
// Shared: per-particle orientation axes from a frame's `orientations` block
// ---------------------------------------------------------------------------

/// Read the per-particle `(head, tail)` orientation-axis pairs from `frame`'s
/// `"orientations"` topology block (same on-disk schema as `bonds`: the two
/// endpoint columns `atomi`/`atomj`). Each row is one particle's axis; the
/// director vector is the internal expansion `pos[head] − pos[tail]`.
fn orientation_pairs(frame: &CoreFrame) -> PyResult<Vec<(usize, usize)>> {
    let groups = AtomGroups::from_frame(frame, "orientations", 2).map_err(py_value_err)?;
    Ok((0..groups.len())
        .map(|i| {
            let t = groups.tuple(i);
            (t[0] as usize, t[1] as usize)
        })
        .collect())
}

// ---------------------------------------------------------------------------
// Steinhardt
// ---------------------------------------------------------------------------

#[pyclass(module = "molrs.compute", name = "Steinhardt")]
pub struct PySteinhardt {
    inner: Steinhardt,
}

#[pymethods]
impl PySteinhardt {
    #[new]
    #[pyo3(signature = (l, average=false, wl=false, wl_normalize=false))]
    fn new(l: Vec<u32>, average: bool, wl: bool, wl_normalize: bool) -> PyResult<Self> {
        let inner = Steinhardt::new(&l)
            .map_err(py_value_err)?
            .with_average(average)
            .with_wl(wl)
            .with_wl_normalize(wl_normalize);
        Ok(Self { inner })
    }

    fn compute<'py>(
        &self,
        py: Python<'py>,
        frames: &Bound<'py, PyAny>,
        nlists: &Bound<'py, PyAny>,
    ) -> PyResult<Vec<Bound<'py, PyDict>>> {
        let owned = collect_frames(frames)?;
        let refs: Vec<&CoreFrame> = owned.iter().collect();
        let neighbors = collect_neighbors(nlists)?;
        let results = self
            .inner
            .compute(&refs, &neighbors)
            .map_err(py_value_err)?;
        results
            .into_iter()
            .map(|r| {
                let d = PyDict::new(py);
                let ls: Vec<u32> = r.l.clone();
                d.set_item("l", ls)?;
                let ql: Vec<Bound<'py, PyArray1<f64>>> =
                    r.ql.into_iter()
                        .map(|v| Array1::from_vec(v).into_pyarray(py))
                        .collect();
                d.set_item("ql", ql)?;
                if let Some(wl) = r.wl {
                    let wls: Vec<Bound<'py, PyArray1<f64>>> = wl
                        .into_iter()
                        .map(|v| Array1::from_vec(v).into_pyarray(py))
                        .collect();
                    d.set_item("wl", wls)?;
                }
                Ok(d)
            })
            .collect()
    }
}

// ---------------------------------------------------------------------------
// Nematic
// ---------------------------------------------------------------------------

#[pyclass(module = "molrs.compute", name = "Nematic")]
pub struct PyNematic {
    inner: Nematic,
}

#[pymethods]
impl PyNematic {
    #[new]
    fn new() -> Self {
        Self {
            inner: Nematic::new(),
        }
    }

    /// Per-particle orientation directors are the unit `head − tail` vectors of
    /// the `"orientations"` topology block, read from the first frame (one
    /// `(head, tail)` atom pair per row). No external director array is passed.
    ///
    /// Returns `(order: float, eigenvalues: ndarray[3], director: ndarray[3], q_tensor: ndarray[3,3])`.
    fn compute<'py>(
        &self,
        py: Python<'py>,
        frames: &Bound<'py, PyAny>,
    ) -> PyResult<(
        f64,
        Bound<'py, PyArray1<f64>>,
        Bound<'py, PyArray1<f64>>,
        Bound<'py, PyArray2<f64>>,
    )> {
        let owned = collect_frames(frames)?;
        let refs: Vec<&CoreFrame> = owned.iter().collect();
        let first = refs
            .first()
            .copied()
            .ok_or_else(|| PyValueError::new_err("no frames provided"))?;
        let pairs = orientation_pairs(first)?;
        let xyz = first.coords().map_err(py_value_err)?;
        let n = xyz.nrows();
        let mut directors: Vec<[F; 3]> = Vec::with_capacity(pairs.len());
        for (head, tail) in pairs {
            if head >= n || tail >= n {
                return Err(PyValueError::new_err(
                    "orientations atom index out of range",
                ));
            }
            directors.push(std::array::from_fn(|d| xyz[[head, d]] - xyz[[tail, d]]));
        }
        let mut results = self
            .inner
            .compute(&refs, directors.as_slice())
            .map_err(py_value_err)?;
        let r = results.remove(0);
        let q = Array2::from_shape_vec(
            (3, 3),
            r.q_tensor.iter().flatten().copied().collect::<Vec<F>>(),
        )
        .map_err(py_value_err)?;
        Ok((
            r.order,
            Array1::from_vec(r.eigenvalues.to_vec()).into_pyarray(py),
            Array1::from_vec(r.director.to_vec()).into_pyarray(py),
            q.into_pyarray(py),
        ))
    }
}

// ---------------------------------------------------------------------------
// Hexatic
// ---------------------------------------------------------------------------

#[pyclass(module = "molrs.compute", name = "Hexatic")]
pub struct PyHexatic {
    inner: Hexatic,
}

#[pymethods]
impl PyHexatic {
    #[new]
    fn new(k: u32) -> PyResult<Self> {
        Ok(Self {
            inner: Hexatic::new(k).map_err(py_value_err)?,
        })
    }

    /// Returns one numpy array per frame: `complex64` per-particle ψ_k
    /// (interleaved real/imag pairs in an `(N, 2)` `float64` array).
    fn compute<'py>(
        &self,
        py: Python<'py>,
        frames: &Bound<'py, PyAny>,
        nlists: &Bound<'py, PyAny>,
    ) -> PyResult<Vec<Bound<'py, PyArray2<f64>>>> {
        let owned = collect_frames(frames)?;
        let refs: Vec<&CoreFrame> = owned.iter().collect();
        let neighbors = collect_neighbors(nlists)?;
        let results = self
            .inner
            .compute(&refs, &neighbors)
            .map_err(py_value_err)?;
        Ok(results
            .into_iter()
            .map(|r| {
                let n = r.psi.len();
                let mut arr = Array2::<F>::zeros((n, 2));
                for (i, c) in r.psi.into_iter().enumerate() {
                    arr[[i, 0]] = c.re;
                    arr[[i, 1]] = c.im;
                }
                arr.into_pyarray(py)
            })
            .collect())
    }
}

// ---------------------------------------------------------------------------
// SolidLiquid
// ---------------------------------------------------------------------------

#[pyclass(module = "molrs.compute", name = "SolidLiquid")]
pub struct PySolidLiquid {
    inner: SolidLiquid,
}

#[pymethods]
impl PySolidLiquid {
    #[new]
    #[pyo3(signature = (l, q_threshold=0.7, n_threshold=6))]
    fn new(l: u32, q_threshold: f64, n_threshold: u32) -> Self {
        let inner = SolidLiquid::new(l)
            .with_q_threshold(q_threshold)
            .with_n_threshold(n_threshold);
        Self { inner }
    }

    /// Returns `(n_solid_bonds[i] : u32 ndarray, is_solid[i] : bool list)` per frame.
    fn compute<'py>(
        &self,
        py: Python<'py>,
        frames: &Bound<'py, PyAny>,
        nlists: &Bound<'py, PyAny>,
    ) -> PyResult<Vec<(Bound<'py, PyArray1<u32>>, Vec<bool>)>> {
        let owned = collect_frames(frames)?;
        let refs: Vec<&CoreFrame> = owned.iter().collect();
        let neighbors = collect_neighbors(nlists)?;
        let results = self
            .inner
            .compute(&refs, &neighbors)
            .map_err(py_value_err)?;
        Ok(results
            .into_iter()
            .map(|r| {
                (
                    Array1::from_vec(r.n_solid_bonds).into_pyarray(py),
                    r.is_solid,
                )
            })
            .collect())
    }
}

// ---------------------------------------------------------------------------
// Legendre reorientation correlation (C1 / C2)
// ---------------------------------------------------------------------------

#[pyclass(module = "molrs.compute", name = "LegendreReorientationResult")]
pub struct PyLegendreReorientationResult {
    inner: LegendreReorientationResult,
}

#[pymethods]
impl PyLegendreReorientationResult {
    #[getter]
    fn lags(&self) -> Vec<usize> {
        self.inner.lags.clone()
    }
    #[getter]
    fn c1<'py>(&self, py: Python<'py>) -> Bound<'py, PyArray1<f64>> {
        self.inner.c1.clone().into_pyarray(py)
    }
    #[getter]
    fn c2<'py>(&self, py: Python<'py>) -> Bound<'py, PyArray1<f64>> {
        self.inner.c2.clone().into_pyarray(py)
    }
}

/// First/second Legendre reorientational TCFs `C₁(t)`, `C₂(t)` of bond vectors.
#[pyclass(module = "molrs.compute", name = "LegendreReorientation")]
pub struct PyLegendreReorientation {
    inner: LegendreReorientation,
}

#[pymethods]
impl PyLegendreReorientation {
    #[new]
    #[pyo3(signature = (max_lag, stride=1))]
    fn new(max_lag: usize, stride: usize) -> Self {
        Self {
            inner: LegendreReorientation::new(max_lag).with_stride(stride),
        }
    }

    /// The `(tail, head)` atom-index pairs defining each tracked bond vector are
    /// read from the `bonds` topology block of the first frame. `frames` are
    /// time-ordered.
    fn compute(&self, frames: &Bound<'_, PyAny>) -> PyResult<PyLegendreReorientationResult> {
        let owned = collect_frames(frames)?;
        let refs: Vec<&CoreFrame> = owned.iter().collect();
        let first = refs
            .first()
            .copied()
            .ok_or_else(|| PyValueError::new_err("no frames provided"))?;
        let groups = AtomGroups::from_frame(first, "bonds", 2).map_err(py_value_err)?;
        let tuples: Vec<(u32, u32)> = (0..groups.len())
            .map(|i| {
                let t = groups.tuple(i);
                (t[0] as u32, t[1] as u32)
            })
            .collect();
        let inner = self.inner.compute(&refs, &tuples).map_err(py_value_err)?;
        Ok(PyLegendreReorientationResult { inner })
    }
}

/// Register this domain's classes and functions.
pub(crate) fn register(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_class::<PySteinhardt>()?;
    m.add_class::<PyNematic>()?;
    m.add_class::<PyHexatic>()?;
    m.add_class::<PySolidLiquid>()?;
    m.add_class::<PyLegendreReorientationResult>()?;
    m.add_class::<PyLegendreReorientation>()?;
    Ok(())
}
