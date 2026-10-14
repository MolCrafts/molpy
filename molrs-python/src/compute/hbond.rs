//! Hydrogen bonds (`molrs::compute::hbond`).

use super::collect_frames;
use crate::error::py_value_err;
use molrs::compute::{HBondCriterion, HBondDistanceKind, HBonds, HBondsResult, LegacyCompute};
use molrs::core::Frame as CoreFrame;
use numpy::{PyReadonlyArray1, PyReadonlyArray2};
use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;
use pyo3::types::{PyAny, PyAnyMethods};

// ---------------------------------------------------------------------------
// Hydrogen bonds
// ---------------------------------------------------------------------------

/// Geometric hydrogen-bond criterion (Luzar–Chandler defaults).
#[pyclass(module = "molrs.compute", name = "HBondCriterion", from_py_object)]
#[derive(Clone)]
pub struct PyHBondCriterion {
    inner: HBondCriterion,
}

#[pymethods]
impl PyHBondCriterion {
    /// ``HBondCriterion(dist_cutoff=3.5, dist_kind="donor_acceptor", angle_cutoff=150.0)``.
    /// `dist_kind` is ``"donor_acceptor"`` or ``"hydrogen_acceptor"``.
    #[new]
    #[pyo3(signature = (dist_cutoff=3.5, dist_kind="donor_acceptor".to_string(), angle_cutoff=150.0))]
    fn new(dist_cutoff: f64, dist_kind: String, angle_cutoff: f64) -> PyResult<Self> {
        let kind = match dist_kind.as_str() {
            "donor_acceptor" => HBondDistanceKind::DonorAcceptor,
            "hydrogen_acceptor" => HBondDistanceKind::HydrogenAcceptor,
            other => {
                return Err(PyValueError::new_err(format!(
                    "dist_kind must be 'donor_acceptor' or 'hydrogen_acceptor', got {other:?}"
                )));
            }
        };
        Ok(Self {
            inner: HBondCriterion::new(dist_cutoff, kind, angle_cutoff),
        })
    }
}

#[pyclass(module = "molrs.compute", name = "HBondsResult")]
pub struct PyHBondsResult {
    inner: HBondsResult,
}

/// Per-frame hydrogen bonds: lists of `(donor, hydrogen, acceptor, distance, angle)`.
type PerFrameHBonds = Vec<Vec<(u32, u32, u32, f64, f64)>>;

#[pymethods]
impl PyHBondsResult {
    /// Per-frame hydrogen bonds as a list of lists of
    /// `(donor, hydrogen, acceptor, distance, angle)` tuples.
    #[getter]
    fn per_frame(&self) -> PerFrameHBonds {
        self.inner
            .per_frame
            .iter()
            .map(|frame| {
                frame
                    .iter()
                    .map(|b| (b.donor, b.hydrogen, b.acceptor, b.distance, b.angle))
                    .collect()
            })
            .collect()
    }

    /// H-bond count per frame.
    #[getter]
    fn counts(&self) -> Vec<usize> {
        self.inner.counts.clone()
    }
}

/// Detect hydrogen bonds per frame from explicit donor `(D, H)` pairs and
/// acceptor atoms under a geometric criterion.
#[pyclass(module = "molrs.compute", name = "HBonds")]
pub struct PyHBonds {
    inner: HBonds,
}

#[pymethods]
impl PyHBonds {
    /// `donors`: `(N, 2)` int array of `(donor, hydrogen)` pairs.
    /// `acceptors`: 1-D int array of acceptor atom indices.
    #[new]
    #[pyo3(signature = (donors, acceptors, criterion=None))]
    fn new(
        donors: PyReadonlyArray2<'_, i64>,
        acceptors: PyReadonlyArray1<'_, i64>,
        criterion: Option<PyHBondCriterion>,
    ) -> PyResult<Self> {
        let da = donors.as_array();
        if da.ncols() != 2 {
            return Err(PyValueError::new_err(
                "donors must have 2 columns (donor, hydrogen)",
            ));
        }
        let mut donor_pairs: Vec<(u32, u32)> = Vec::with_capacity(da.nrows());
        for row in da.rows() {
            if row[0] < 0 || row[1] < 0 {
                return Err(PyValueError::new_err("atom indices must be non-negative"));
            }
            donor_pairs.push((row[0] as u32, row[1] as u32));
        }
        let mut acc: Vec<u32> = Vec::with_capacity(acceptors.len()?);
        for &v in acceptors.as_array().iter() {
            if v < 0 {
                return Err(PyValueError::new_err("atom indices must be non-negative"));
            }
            acc.push(v as u32);
        }
        let crit = criterion.map(|c| c.inner).unwrap_or_default();
        Ok(Self {
            inner: HBonds::new(donor_pairs, acc, crit),
        })
    }

    /// Detect hydrogen bonds in each frame.
    fn compute(&self, frames: &Bound<'_, PyAny>) -> PyResult<PyHBondsResult> {
        let owned = collect_frames(frames)?;
        let refs: Vec<&CoreFrame> = owned.iter().collect();
        let inner = self.inner.compute(&refs, ()).map_err(py_value_err)?;
        Ok(PyHBondsResult { inner })
    }
}

/// Register this domain's classes and functions.
pub(crate) fn register(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_class::<PyHBondCriterion>()?;
    m.add_class::<PyHBondsResult>()?;
    m.add_class::<PyHBonds>()?;
    Ok(())
}
