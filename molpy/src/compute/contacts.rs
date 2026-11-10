//! Close contacts of a bonded snapshot (`molrs::compute::CloseContacts`).

use molrs::compute::{CloseContacts, CloseContactsInput, CloseContactsResult, Compute};
use pyo3::prelude::*;

use super::error::Result;
use super::output::py_result;
use super::spec::described;
use crate::core::neighborlist::PyNeighbors;

py_result! {
    /// Close contacts (``molrs::compute::CloseContactsResult``).
    ///
    /// ``pairs`` is ``(n, 2)``, one contact ``(i, j)`` per row with ``i < j``,
    /// sorted; ``distances`` their minimum-image distances (Å).
    PyCloseContactsResult(CloseContactsResult) as "CloseContactsResult" {
        pairs, distances
    }
}

#[pymethods]
impl PyCloseContactsResult {
    /// Number of contacts.
    #[getter]
    fn n_contacts(&self) -> usize {
        self.inner.n_contacts()
    }
}

/// Pairs of a neighbour table that the bond graph does not join within
/// ``exclude_bonds`` bonds (``molrs::compute::CloseContacts``).
///
/// Parameters
/// ----------
/// exclude_bonds : int
///     Pairs at most this many bonds apart are not contacts: 1 excludes
///     bonded pairs, 2 also angle ends, 3 also torsion ends; 0 excludes none.
///
/// ``compute(pairs=, bonds=)`` reads the half-shell self-query ``Neighbors``
/// table ``pairs`` (its cutoff is the contact distance; it needs the
/// ``dist_sq`` column) and the bonds as ``(i, j)`` atom index pairs. Returns a
/// ``CloseContactsResult``.
#[pyclass(module = "molpy.compute", name = "CloseContacts", frozen)]
pub(crate) struct PyCloseContacts {
    inner: CloseContacts,
}

#[pymethods]
impl PyCloseContacts {
    #[new]
    #[pyo3(signature = (*, exclude_bonds))]
    fn new(exclude_bonds: usize) -> Self {
        Self {
            inner: CloseContacts::new(exclude_bonds),
        }
    }

    #[pyo3(signature = (*, pairs, bonds))]
    fn compute<'py>(
        &self,
        py: Python<'py>,
        pairs: PyRef<'py, PyNeighbors>,
        bonds: Vec<[usize; 2]>,
    ) -> Result<PyCloseContactsResult> {
        let out = {
            let input = CloseContactsInput {
                pairs: &pairs.inner,
                bonds: &bonds,
            };
            py.detach(|| self.inner.compute(input))
        }?;
        Ok(out.into())
    }
}

described!(PyCloseContacts, CloseContacts);

pub(crate) fn register(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_class::<PyCloseContactsResult>()?;
    m.add_class::<PyCloseContacts>()?;
    Ok(())
}
