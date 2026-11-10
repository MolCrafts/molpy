//! Bonds threaded through rings under periodic boundaries
//! (`molrs::compute::RingPiercing`).

use molrs::compute::{Compute, RingPiercing, RingPiercingInput, RingPiercingResult};
use pyo3::prelude::*;

use super::error::Result;
use super::output::py_result;
use super::spec::described;
use crate::array::ArrayArg2;
use crate::core::simbox::PyBox;

py_result! {
    /// Ring piercings (``molrs::compute::RingPiercingResult``).
    ///
    /// Row ``k`` is one piercing: bond ``bond_idx[k]`` (an index into the
    /// input bonds) passes through ring ``ring_idx[k]`` (an index into the
    /// input rings); sorted by ring, then bond.
    PyRingPiercingResult(RingPiercingResult) as "RingPiercingResult" {
        ring_idx, bond_idx
    }
}

#[pymethods]
impl PyRingPiercingResult {
    /// Number of piercings.
    #[getter]
    fn n_piercings(&self) -> usize {
        self.inner.n_piercings()
    }
}

/// Bonds that pass through a ring, read by minimum image
/// (``molrs::compute::RingPiercing``).
///
/// A bond pierces a ring when its segment crosses the ring's least-squares
/// plane closer to the ring's centroid than ``radius_fraction`` times the
/// ring's mean radius, and shares no atom with it. Bonds never break in a
/// minimization or a classical run, so a piercing is never undone there.
///
/// Parameters
/// ----------
/// radius_fraction : float
///     Share of a ring's mean radius inside which a crossing pierces it
///     (``1.0``: the ring's own mean radius). Positive and finite.
///
/// ``compute(positions=, simbox=, bonds=, rings=)`` takes ``(N, 3)``
/// positions (Å, wrapped or not), the cell, bonds as ``(i, j)`` atom index
/// pairs and rings as atom index lists of three atoms or more (e.g. the
/// small rings of ``molpy.perceive.perceive_rings``). Returns a
/// ``RingPiercingResult``.
#[pyclass(module = "molpy.compute", name = "RingPiercing", frozen)]
pub(crate) struct PyRingPiercing {
    inner: RingPiercing,
}

#[pymethods]
impl PyRingPiercing {
    #[new]
    #[pyo3(signature = (*, radius_fraction))]
    fn new(radius_fraction: f64) -> Result<Self> {
        Ok(Self {
            inner: RingPiercing::new(radius_fraction)?,
        })
    }

    #[pyo3(signature = (*, positions, simbox, bonds, rings))]
    fn compute<'py>(
        &self,
        py: Python<'py>,
        positions: ArrayArg2<'py, f64>,
        simbox: PyRef<'py, PyBox>,
        bonds: Vec<[usize; 2]>,
        rings: Vec<Vec<usize>>,
    ) -> Result<PyRingPiercingResult> {
        let out = {
            let input = RingPiercingInput {
                positions: positions.as_array(),
                simbox: &simbox.inner,
                bonds: &bonds,
                rings: &rings,
            };
            py.detach(|| self.inner.compute(input))
        }?;
        Ok(out.into())
    }
}

described!(PyRingPiercing, RingPiercing);

pub(crate) fn register(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_class::<PyRingPiercingResult>()?;
    m.add_class::<PyRingPiercing>()?;
    Ok(())
}
