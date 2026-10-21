//! Ring facts (`molrs::perceive::perceive_rings`): [`PyRingSet`] reports the SSSR
//! rings of a molecule and the systems they fuse into, without touching it.

use molrs::core::NodeId;
use pyo3::prelude::*;

use crate::core::graph_views::{Leaf, PyNodeRef};
use crate::core::molgraph::PyAtomistic;

/// The ring facts of a molecule: SSSR rings and the systems they fuse into.
///
/// Returned by :func:`perceive_rings`, which runs the perception once; every
/// method reads the result. Atoms are the molecule's own views: a ring is a
/// list of :class:`~molrs.core.Atom`, and an atom argument must be an atom
/// of that molecule.
///
/// Not to be confused with :func:`assign_rings`, which answers a
/// different question — it *decorates* the graph with ring flags, in place.
/// This type *reports*, and never touches the molecule.
///
/// Examples
/// --------
/// >>> rings = molrs.perceive.perceive_rings(molrs.io.smiles.read_smiles("c1ccccc1").to_atomistic())
/// >>> rings.n_rings()
/// 1
/// >>> rings.ring_sizes()
/// [6]
#[pyclass(module = "molrs.perceive", name = "RingSet", subclass)]
pub struct PyRingSet {
    inner: molrs::perceive::RingSet,
    /// The molecule the rings were perceived on: the world of every view.
    mol: Py<PyAtomistic>,
}

impl PyRingSet {
    /// `groups` as views on the perceived molecule.
    fn views<'py>(
        &self,
        py: Python<'py>,
        groups: &[Vec<NodeId>],
    ) -> PyResult<Vec<Vec<Bound<'py, PyAny>>>> {
        let leaf = Leaf::of(self.mol.bind(py).as_any())?;
        groups
            .iter()
            .map(|group| super::atom_views(&leaf, group))
            .collect()
    }

    /// The node `atom` names, checked to be an atom of the perceived molecule.
    fn own(&self, atom: &Bound<'_, PyNodeRef>) -> PyResult<NodeId> {
        Leaf::of(self.mol.bind(atom.py()).as_any())?.own_node(atom)
    }
}

#[pymethods]
impl PyRingSet {
    /// Every ring, as the atoms of a closed path.
    fn rings<'py>(&self, py: Python<'py>) -> PyResult<Vec<Vec<Bound<'py, PyAny>>>> {
        self.views(py, self.inner.rings())
    }

    /// Number of rings.
    fn n_rings(&self) -> usize {
        self.inner.n_rings()
    }

    /// Atom count of every ring, ascending.
    fn ring_sizes(&self) -> Vec<usize> {
        self.inner.ring_sizes()
    }

    /// Rings that share at least one atom, unioned: benzene → one system of 6,
    /// naphthalene → one of 10, biphenyl → two of 6.
    fn ring_systems<'py>(&self, py: Python<'py>) -> PyResult<Vec<Vec<Bound<'py, PyAny>>>> {
        self.views(py, &self.inner.ring_systems())
    }

    /// Atom count of the largest fused / bridged ring system (naphthalene →
    /// 10); ``0`` for an acyclic molecule.
    fn max_ring_system_size(&self) -> usize {
        self.inner.max_ring_system_size()
    }

    /// Whether `atom` belongs to any ring.
    ///
    /// Raises
    /// ------
    /// ValueError
    ///     If `atom` is not an atom of the perceived molecule.
    fn is_atom_in_ring(&self, atom: &Bound<'_, PyNodeRef>) -> PyResult<bool> {
        Ok(self.inner.is_atom_in_ring(self.own(atom)?))
    }

    /// Number of rings containing `atom`.
    fn n_atom_rings(&self, atom: &Bound<'_, PyNodeRef>) -> PyResult<usize> {
        Ok(self.inner.n_atom_rings(self.own(atom)?))
    }

    /// Size of the smallest ring containing `atom`, or ``None``.
    fn smallest_ring_containing_atom(
        &self,
        atom: &Bound<'_, PyNodeRef>,
    ) -> PyResult<Option<usize>> {
        Ok(self.inner.smallest_ring_containing_atom(self.own(atom)?))
    }

    fn __repr__(&self) -> String {
        format!(
            "RingSet(n_rings={}, sizes={:?})",
            self.inner.n_rings(),
            self.inner.ring_sizes()
        )
    }
}

/// Perceive the rings of ``mol`` (SSSR / minimum cycle basis) as a
/// :class:`RingSet` side table — a query: ``mol`` is left untouched.
#[pyfunction(name = "perceive_rings")]
pub(super) fn perceive_rings_py(mol: &Bound<'_, PyAtomistic>) -> PyResult<PyRingSet> {
    Ok(PyRingSet {
        inner: molrs::perceive::perceive_rings(mol.try_borrow()?.core()),
        mol: mol.clone().unbind(),
    })
}
