//! Python binding for the index-only bond graph [`Topology`].

use molrs::core::Topology;
use numpy::{IntoPyArray, PyArray1, PyArray2};
use pyo3::exceptions::{PyIndexError, PyValueError};
use pyo3::prelude::*;
use pyo3::types::PyType;

use crate::core::frame::PyFrame;

/// Index-only bond graph: atoms are the indices ``0..n_atoms``, bonds index
/// pairs. Angles, dihedrals and impropers are derived from the bonds.
///
/// ``Topology(n_atoms)`` is ``n_atoms`` unbonded atoms (``Topology(0)`` the
/// empty graph); :meth:`from_frame` reads the bond graph of a frame (atoms
/// the rows of ``frame["atoms"]``, bonds the ``atomi`` / ``atomj`` rows of
/// ``frame["bonds"]``). Deleting is swap-remove: the last atom (bond) moves
/// into the freed slot; :meth:`del_atoms` / :meth:`del_bonds` take indices of
/// the graph *before* the call.
#[pyclass(
    module = "molrs.core",
    name = "Topology",
    skip_from_py_object,
    subclass
)]
pub struct PyTopology {
    inner: Topology,
}

impl PyTopology {
    fn check_atom(&self, idx: usize) -> PyResult<()> {
        if idx >= self.inner.n_atoms() {
            return Err(PyIndexError::new_err(format!(
                "atom {idx} out of range (n_atoms = {})",
                self.inner.n_atoms()
            )));
        }
        Ok(())
    }

    fn check_bond(&self, idx: usize) -> PyResult<()> {
        if idx >= self.inner.n_bonds() {
            return Err(PyIndexError::new_err(format!(
                "bond {idx} out of range (n_bonds = {})",
                self.inner.n_bonds()
            )));
        }
        Ok(())
    }
}

/// `rows` (fixed-width index tuples) as an ``(n, K)`` ``uint64`` array.
fn index_rows<'py, const K: usize>(
    py: Python<'py>,
    rows: Vec<[usize; K]>,
) -> Bound<'py, PyArray2<u64>> {
    let flat: Vec<u64> = rows.iter().flatten().map(|&i| i as u64).collect();
    ndarray::Array2::from_shape_vec((rows.len(), K), flat)
        .expect("K indices per row")
        .into_pyarray(py)
}

#[pymethods]
impl PyTopology {
    /// ``n_atoms`` unbonded atoms (indices ``0..n_atoms``).
    #[new]
    fn new(n_atoms: usize) -> Self {
        Self {
            inner: Topology::with_atoms(n_atoms),
        }
    }

    /// Read the bond graph of ``frame``.
    ///
    /// Raises
    /// ------
    /// ValueError
    ///     ``frame`` has no atoms, the bonds block lacks
    ///     ``atomi`` / ``atomj``, or a bond names a row outside the frame.
    #[classmethod]
    fn from_frame(_cls: &Bound<'_, PyType>, frame: &PyFrame) -> PyResult<Self> {
        let inner = frame
            .with_frame(Topology::from_frame)?
            .map_err(|e| PyValueError::new_err(e.to_string()))?;
        Ok(Self { inner })
    }

    /// Number of atoms (graph nodes).
    #[getter]
    fn n_atoms(&self) -> usize {
        self.inner.n_atoms()
    }

    /// Number of bonds (graph edges).
    #[getter]
    fn n_bonds(&self) -> usize {
        self.inner.n_bonds()
    }

    /// Number of unique angles ``i-j-k``.
    #[getter]
    fn n_angles(&self) -> usize {
        self.inner.n_angles()
    }

    /// Number of unique proper dihedrals ``i-j-k-l``.
    #[getter]
    fn n_dihedrals(&self) -> usize {
        self.inner.n_dihedrals()
    }

    /// Number of connected components; an unbonded atom is its own.
    #[getter]
    fn n_components(&self) -> usize {
        self.inner.n_components()
    }

    /// Every bond ``[i, j]`` in insertion order, an ``(n_bonds, 2)`` ``uint64``
    /// array; row ``b`` is bond index ``b``.
    fn bonds<'py>(&self, py: Python<'py>) -> Bound<'py, PyArray2<u64>> {
        index_rows(py, self.inner.bonds())
    }

    /// Every unique angle ``[i, j, k]`` (``i < k``), ``(n_angles, 3)`` ``uint64``.
    fn angles<'py>(&self, py: Python<'py>) -> Bound<'py, PyArray2<u64>> {
        index_rows(py, self.inner.angles())
    }

    /// Every unique proper dihedral ``[i, j, k, l]`` (``j < k``),
    /// ``(n_dihedrals, 4)`` ``uint64``.
    fn dihedrals<'py>(&self, py: Python<'py>) -> Bound<'py, PyArray2<u64>> {
        index_rows(py, self.inner.dihedrals())
    }

    /// Every improper ``[center, i, j, k]``: each sorted 3-combination of the
    /// neighbours of every atom of degree >= 3, ``(n, 4)`` ``uint64``.
    fn impropers<'py>(&self, py: Python<'py>) -> Bound<'py, PyArray2<u64>> {
        index_rows(py, self.inner.impropers())
    }

    /// Neighbour indices of atom ``idx``, in insertion order.
    ///
    /// Raises
    /// ------
    /// IndexError
    ///     ``idx`` is not an atom index.
    fn neighbors(&self, idx: usize) -> PyResult<Vec<usize>> {
        self.check_atom(idx)?;
        Ok(self.inner.neighbors(idx))
    }

    /// Number of bonds of atom ``idx``.
    ///
    /// Raises
    /// ------
    /// IndexError
    ///     ``idx`` is not an atom index.
    fn degree(&self, idx: usize) -> PyResult<usize> {
        self.check_atom(idx)?;
        Ok(self.inner.degree(idx))
    }

    /// Whether atoms ``i`` and ``j`` are directly bonded.
    ///
    /// Raises
    /// ------
    /// IndexError
    ///     ``i`` or ``j`` is not an atom index.
    fn are_bonded(&self, i: usize, j: usize) -> PyResult<bool> {
        self.check_atom(i)?;
        self.check_atom(j)?;
        Ok(self.inner.are_bonded(i, j))
    }

    /// Per-atom connected-component label, ``0..n_components`` in order of
    /// each component's first atom, as a ``uint32`` array.
    fn connected_components<'py>(&self, py: Python<'py>) -> Bound<'py, PyArray1<u32>> {
        self.inner
            .connected_components()
            .into_iter()
            .map(|c| u32::try_from(c).expect("a component label is below n_atoms"))
            .collect::<Vec<_>>()
            .into_pyarray(py)
    }

    /// Bond-hop distance from atom ``source`` to every atom, an ``int64``
    /// array; ``-1`` marks an atom in another component.
    ///
    /// Raises
    /// ------
    /// IndexError
    ///     ``source`` is not an atom index.
    fn distances<'py>(
        &self,
        py: Python<'py>,
        source: usize,
    ) -> PyResult<Bound<'py, PyArray1<i64>>> {
        self.check_atom(source)?;
        Ok(self.inner.distances(source).into_pyarray(py))
    }

    /// Append ``n`` unbonded atoms, indexed after the existing ones.
    fn add_atoms(&mut self, n: usize) {
        self.inner.add_atoms(n);
    }

    /// Add every bond ``(i, j)`` of ``pairs`` in order, skipping one already
    /// present (either orientation). One bond is ``add_bonds([(i, j)])``.
    ///
    /// Raises
    /// ------
    /// IndexError
    ///     An endpoint is not an atom index (nothing is added).
    fn add_bonds(&mut self, pairs: Vec<[usize; 2]>) -> PyResult<()> {
        for &[i, j] in &pairs {
            self.check_atom(i)?;
            self.check_atom(j)?;
        }
        self.inner.add_bonds(&pairs);
        Ok(())
    }

    /// Delete atom ``idx`` and every bond incident to it. The last atom moves
    /// into slot ``idx``.
    ///
    /// Raises
    /// ------
    /// IndexError
    ///     ``idx`` is not an atom index.
    fn del_atom(&mut self, idx: usize) -> PyResult<()> {
        self.check_atom(idx)?;
        self.inner.del_atom(idx);
        Ok(())
    }

    /// Delete every atom of ``idxs`` (indices before the call; duplicates
    /// name one atom) and every bond incident to any of them.
    ///
    /// Raises
    /// ------
    /// IndexError
    ///     An index is not an atom index (nothing is deleted).
    fn del_atoms(&mut self, idxs: Vec<usize>) -> PyResult<()> {
        for &idx in &idxs {
            self.check_atom(idx)?;
        }
        self.inner.del_atoms(&idxs);
        Ok(())
    }

    /// Delete bond ``idx`` (a row of :meth:`bonds`). The last bond moves into
    /// slot ``idx``.
    ///
    /// Raises
    /// ------
    /// IndexError
    ///     ``idx`` is not a bond index.
    fn del_bond(&mut self, idx: usize) -> PyResult<()> {
        self.check_bond(idx)?;
        self.inner.del_bond(idx);
        Ok(())
    }

    /// Delete every bond of ``idxs`` (rows of :meth:`bonds` before the call;
    /// duplicates name one bond).
    ///
    /// Raises
    /// ------
    /// IndexError
    ///     An index is not a bond index (nothing is deleted).
    fn del_bonds(&mut self, idxs: Vec<usize>) -> PyResult<()> {
        for &idx in &idxs {
            self.check_bond(idx)?;
        }
        self.inner.del_bonds(&idxs);
        Ok(())
    }

    fn __repr__(&self) -> String {
        format!(
            "Topology(n_atoms={}, n_bonds={})",
            self.inner.n_atoms(),
            self.inner.n_bonds()
        )
    }
}
