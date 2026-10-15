//! Geometric hydrogen bonds (`molrs::compute::HBonds`): the bonds of every
//! frame, their presence series and per-frame networks.

use molrs::compute::{Compute, HBond, HBondCriterion, HBonds, HBondsInput, HBondsResult};
use numpy::{PyArray1, PyReadonlyArray3};
use pyo3::prelude::*;

use super::dynamics::PyPresence;
use super::error::Result;
use super::output::{ToPy, list, py_result};
use super::spec::described;
use crate::core::neighborlist::PyNeighbors;

py_result! {
    /// One hydrogen bond (``molrs::compute::HBond``): donor, hydrogen and
    /// acceptor indices, the D···A distance (Å) and the D–H···A angle (deg).
    PyHBond(HBond) as "HBond" { donor, hydrogen, acceptor, distance, angle }
}

impl ToPy for Vec<HBond> {
    fn to_py<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyAny>> {
        list(py, self)
    }
}

impl ToPy for Vec<Vec<HBond>> {
    fn to_py<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyAny>> {
        list(py, self)
    }
}

py_result! {
    /// The hydrogen bonds of every frame (``molrs::compute::HBondsResult``):
    /// ``per_frame[t]`` lists frame ``t``'s ``HBond``s, ``counts[t]`` their
    /// number.
    PyHBondsResult(HBondsResult) as "HBondsResult" { per_frame, counts, n_atoms }
}

#[pymethods]
impl PyHBondsResult {
    /// The (donor, acceptor) presence series, for ``Survival``; ``None`` when
    /// no bond is ever present.
    fn presence(&self) -> Option<PyPresence> {
        self.inner.presence().map(PyPresence::from)
    }

    /// The hydrogen-bond network component label of every atom in ``frame``.
    fn components<'py>(&self, py: Python<'py>, frame: usize) -> Result<Bound<'py, PyArray1<u32>>> {
        Ok(PyArray1::from_vec(py, self.inner.components(frame)?))
    }
}

/// Geometric hydrogen-bond detection, frame by frame
/// (``molrs::compute::HBonds``).
///
/// Parameters
/// ----------
/// dist_cutoff : float, optional
/// dist_kind : str, optional
///     ``"donor_acceptor"`` or ``"hydrogen_acceptor"``.
/// angle_cutoff : float, optional
/// donor_acceptor_margin : float, optional
///     The criterion fields; each ``None`` keeps the core default criterion.
///
/// ``compute(positions=, donors=, acceptors=, pairs=)``: a ``(T, N, 3)``
/// trajectory, ``(donor, hydrogen)`` index pairs, acceptor indices and one
/// ``Neighbors`` cross-query per frame (donor sites against acceptors).
#[pyclass(module = "molrs.compute", name = "HBonds", frozen)]
pub(crate) struct PyHBonds {
    inner: HBonds,
}

#[pymethods]
impl PyHBonds {
    #[new]
    #[pyo3(signature = (
        *,
        dist_cutoff = None,
        dist_kind = None,
        angle_cutoff = None,
        donor_acceptor_margin = None,
    ))]
    fn new(
        dist_cutoff: Option<f64>,
        dist_kind: Option<&str>,
        angle_cutoff: Option<f64>,
        donor_acceptor_margin: Option<f64>,
    ) -> Result<Self> {
        let mut criterion = HBondCriterion::default();
        if let Some(dist_cutoff) = dist_cutoff {
            criterion.dist_cutoff = dist_cutoff;
        }
        if let Some(dist_kind) = dist_kind {
            criterion.dist_kind = dist_kind.parse()?;
        }
        if let Some(angle_cutoff) = angle_cutoff {
            criterion.angle_cutoff = angle_cutoff;
        }
        if let Some(margin) = donor_acceptor_margin {
            criterion = criterion.with_donor_acceptor_margin(margin);
        }
        Ok(Self {
            inner: HBonds::new(criterion)?,
        })
    }

    #[pyo3(signature = (*, positions, donors, acceptors, pairs))]
    fn compute(
        &self,
        positions: PyReadonlyArray3<'_, f64>,
        donors: Vec<[u32; 2]>,
        acceptors: Vec<u32>,
        pairs: Vec<PyRef<'_, PyNeighbors>>,
    ) -> Result<PyHBondsResult> {
        let donors: Vec<(u32, u32)> = donors.into_iter().map(|[d, h]| (d, h)).collect();
        let pairs: Vec<_> = pairs.iter().map(|table| &table.inner).collect();
        let out = self.inner.compute(HBondsInput {
            positions: positions.as_array(),
            donors: &donors,
            acceptors: &acceptors,
            pairs: &pairs,
        })?;
        Ok(out.into())
    }
}

described!(PyHBonds, HBonds);

pub(crate) fn register(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_class::<PyHBond>()?;
    m.add_class::<PyHBondsResult>()?;
    m.add_class::<PyHBonds>()?;
    Ok(())
}
