//! Python wrappers for neighbor search: `NeighborList`, `Neighbors`,
//! `NeighborQuery`.
//!
//! The surface mirrors the Rust contract (`molrs::core::NeighborList`) one to
//! one, because the seam is the place where a renamed concept turns into a
//! silently different answer:
//!
//! - [`PyNeighborList`] (`molrs.core.NeighborList`) is the **engine**: it owns the
//!   cutoff and the backend that indexes space. `build` / `update` index
//!   coordinates and enumerate nothing; `neighbors(...)` materializes a table.
//! - [`PyNeighbors`] (`molrs.core.Neighbors`) is the **table**: read-only columns,
//!   one row per pair.
//! - [`PyNeighborQuery`] (`molrs.core.NeighborQuery`) is the **cross** door: a set of
//!   query points searched against a separate reference set.
//!
//! Two conventions travel with every table and are worth stating here, because
//! both have been a source of silent double counting:
//!
//! - A self search is **half-shell**: each unordered pair appears exactly once,
//!   with `i < j`, and never as `i == j`. A cross-query is directed and has no
//!   such rule.
//! - `dist_sq=True, disp=True` (the binder default, "FULL") names the *columns*
//!   a table keeps. It never means a bidirectional pair list. A column that was
//!   not stored reads back as `None` — never as a fabricated zero, which would
//!   be a physically meaningful value (coincident particles).
//!
//! `disp` is the **unnormalized** minimum-image displacement `r_j - r_i`: its
//! length is the pair distance, not 1. All lengths are in the unit of the input
//! coordinates, which is Å everywhere in molrs, so `dist_sq` is in Å².

use crate::array::ArrayArg2;
use crate::core::simbox::PyBox;
use molrs::core::SimBox;
use molrs::core::{
    NeighborColumns, NeighborList as RsNeighborList, NeighborPair, NeighborPolicy,
    NeighborQuery as RsNeighborQuery, Neighbors as RsNeighbors, QueryMode, SkinError,
    VerletSkin as RsVerletSkin,
};
use ndarray::{Array2, ArrayView1};
use numpy::{IntoPyArray, PyArray1, PyArray2};
use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;
use pyo3::types::PyTuple;

/// Reject a coordinate array that is not `(N, 3)` before it reaches the core,
/// whose own check is an `assert!` — and a panic must not cross this seam.
/// `label` names the offending argument in the error.
fn check_points(points: &ArrayArg2<'_, f64>, label: &str) -> PyResult<()> {
    if points.as_array().ncols() != 3 {
        return Err(PyValueError::new_err(format!(
            "{label} must have shape (N, 3)"
        )));
    }
    Ok(())
}

/// Reject a non-positive cutoff before the core asserts on it.
fn check_cutoff(cutoff: f64) -> PyResult<()> {
    if cutoff.is_nan() || cutoff <= 0.0 {
        return Err(PyValueError::new_err(
            "cutoff must be a positive length in Å",
        ));
    }
    Ok(())
}

// ---------------------------------------------------------------------------
// PyNeighbors — the materialized pair table
// ---------------------------------------------------------------------------

/// Materialized neighbor pair table, exposed to Python as `molrs.core.Neighbors`.
///
/// A column store of every pair within the cutoff: two index columns that are
/// always present, plus whichever physical columns the search was told to keep.
/// Row ``k`` of every column describes the same pair.
///
/// A self search is half-shell — each unordered pair appears once, with
/// ``i < j``. A cross-query (:class:`NeighborQuery`) is directed and has no
/// such rule.
///
/// Optional columns are methods returning ``None`` when the search did not
/// store them; ``None`` is deliberately distinct from a column of zeros, which
/// would mean coincident particles.
///
/// Methods
/// -------
/// query_point_indices() : numpy.ndarray[uint32], shape (n_pairs,)
///     Query-point index ``i`` per pair.
/// point_indices() : numpy.ndarray[uint32], shape (n_pairs,)
///     Reference-point index ``j`` per pair.
/// dist_sq() : numpy.ndarray[float64], shape (n_pairs,), or None
///     Squared minimum-image distance in Å². Take ``np.sqrt`` at the call site
///     for a distance — nothing here hides a per-pair square root.
/// disp() : numpy.ndarray[float64], shape (n_pairs, 3), or None
///     Minimum-image displacement ``r_j - r_i`` in Å, **unnormalized**.
///
/// Attributes
/// ----------
/// n_pairs : int
///     Number of pairs — the row count every column shares.
/// n_points : int
///     Number of reference points.
/// n_query_points : int
///     Number of query points (equal to ``n_points`` for a self search).
/// is_self_query : bool
///     ``True`` when both index columns address the same point set.
///
/// Examples
/// --------
/// >>> neigh = nl.neighbors()
/// >>> i, j = neigh.query_point_indices(), neigh.point_indices()
/// >>> distances = np.sqrt(neigh.dist_sq())
/// >>> directions = neigh.disp() / distances[:, None]
#[pyclass(module = "molrs.core", name = "Neighbors", subclass)]
pub struct PyNeighbors {
    pub(crate) inner: RsNeighbors,
}

#[pymethods]
impl PyNeighbors {
    #[new]
    #[allow(
        clippy::too_many_arguments,
        reason = "Pair-table constructor arguments"
    )]
    #[pyo3(signature = (is_self_query, n_points, n_query_points, idx_i, idx_j, dist_sq=None, disp=None))]
    fn new(
        is_self_query: bool,
        n_points: usize,
        n_query_points: usize,
        idx_i: Vec<u32>,
        idx_j: Vec<u32>,
        dist_sq: Option<Vec<f64>>,
        disp: Option<Vec<[f64; 3]>>,
    ) -> PyResult<Self> {
        let n = idx_i.len();
        if idx_j.len() != n
            || dist_sq.as_ref().is_some_and(|values| values.len() != n)
            || disp.as_ref().is_some_and(|values| values.len() != n)
        {
            return Err(PyValueError::new_err(
                "inconsistent Neighbors pickle columns",
            ));
        }
        let columns = NeighborColumns {
            dist_sq: dist_sq.is_some(),
            disp: disp.is_some(),
        };
        let mode = if is_self_query {
            QueryMode::SelfQuery { n_points }
        } else {
            QueryMode::CrossQuery {
                n_query_points,
                n_points,
            }
        };
        let pairs = (0..n).map(|row| NeighborPair {
            i: idx_i[row],
            j: idx_j[row],
            dist_sq: dist_sq.as_ref().map_or(0.0, |values| values[row]),
            disp: disp.as_ref().map_or([0.0; 3], |values| values[row]),
        });
        Ok(Self {
            inner: RsNeighbors::from_pairs(pairs, columns, mode),
        })
    }

    fn __reduce__<'py>(slf: &Bound<'py, Self>) -> PyResult<Bound<'py, PyTuple>> {
        let inner = &slf.borrow().inner;
        crate::pickle::reduce_via_type(
            slf.as_any(),
            (
                matches!(inner.mode(), QueryMode::SelfQuery { .. }),
                inner.n_points(),
                inner.n_query_points(),
                inner.query_point_indices().to_vec(),
                inner.point_indices().to_vec(),
                inner.dist_sq().map(|values| values.to_vec()),
                inner.disp().map(|values| {
                    values
                        .outer_iter()
                        .map(|row| [row[0], row[1], row[2]])
                        .collect::<Vec<_>>()
                }),
            ),
        )
    }

    /// Query-point indices ``i``, one per pair.
    ///
    /// Zero-copy numpy view into the underlying Rust ``Vec<u32>``, pinned alive
    /// through numpy's ``.base`` mechanism: the returned array keeps this table
    /// alive for as long as Python references it.
    ///
    /// Returns
    /// -------
    /// numpy.ndarray, shape (n_pairs,), dtype uint32
    fn query_point_indices<'py>(slf: Bound<'py, Self>) -> Bound<'py, PyArray1<u32>> {
        let owner = slf.clone().into_any();
        let borrowed = slf.borrow();
        let view = ArrayView1::from(borrowed.inner.query_point_indices());
        unsafe { PyArray1::<u32>::borrow_from_array(&view, owner) }
    }

    /// Reference-point indices ``j``, one per pair.
    ///
    /// Zero-copy view; see [`query_point_indices`](Self::query_point_indices).
    ///
    /// Returns
    /// -------
    /// numpy.ndarray, shape (n_pairs,), dtype uint32
    fn point_indices<'py>(slf: Bound<'py, Self>) -> Bound<'py, PyArray1<u32>> {
        let owner = slf.clone().into_any();
        let borrowed = slf.borrow();
        let view = ArrayView1::from(borrowed.inner.point_indices());
        unsafe { PyArray1::<u32>::borrow_from_array(&view, owner) }
    }

    /// Squared minimum-image distances (Å²), one per pair.
    ///
    /// Zero-copy view, or ``None`` when this table never stored the column.
    /// The square root is left to the caller: most consumers compare against a
    /// squared cutoff, and an accessor that hid a per-pair ``sqrt`` would make
    /// that cost invisible.
    ///
    /// Returns
    /// -------
    /// numpy.ndarray, shape (n_pairs,), dtype float64, or None
    fn dist_sq<'py>(slf: Bound<'py, Self>) -> Option<Bound<'py, PyArray1<f64>>> {
        let owner = slf.clone().into_any();
        let borrowed = slf.borrow();
        let column = borrowed.inner.dist_sq()?;
        let view = ArrayView1::from(column);
        Some(unsafe { PyArray1::<f64>::borrow_from_array(&view, owner) })
    }

    /// Minimum-image displacements ``r_j - r_i`` (Å) as an ``(n_pairs, 3)``
    /// array.
    ///
    /// Row ``k`` points from ``i`` to ``j`` and is **not** normalized — its
    /// length is the pair distance. Zero-copy view, or ``None`` when this table
    /// never stored the column (which is not the same as a zero vector: that
    /// would mean two coincident particles).
    ///
    /// Returns
    /// -------
    /// numpy.ndarray, shape (n_pairs, 3), dtype float64, or None
    fn disp<'py>(slf: Bound<'py, Self>) -> Option<Bound<'py, PyArray2<f64>>> {
        let owner = slf.clone().into_any();
        let borrowed = slf.borrow();
        let view = borrowed.inner.disp()?;
        Some(unsafe { PyArray2::<f64>::borrow_from_array(&view, owner) })
    }

    /// Number of neighbor pairs — the row count every column shares.
    #[getter]
    fn n_pairs(&self) -> usize {
        self.inner.n_pairs()
    }

    /// Number of reference points the search indexed.
    #[getter]
    fn n_points(&self) -> usize {
        self.inner.n_points()
    }

    /// Number of query points; equal to ``n_points`` for a self search.
    #[getter]
    fn n_query_points(&self) -> usize {
        self.inner.n_query_points()
    }

    /// Whether both index columns address the same point set (half-shell).
    #[getter]
    fn is_self_query(&self) -> bool {
        matches!(self.inner.mode(), QueryMode::SelfQuery { .. })
    }

    fn __repr__(&self) -> String {
        let mode = if matches!(self.inner.mode(), QueryMode::SelfQuery { .. }) {
            "self"
        } else {
            "cross"
        };
        let columns = self.inner.columns();
        format!(
            "Neighbors(n_pairs={}, mode={}, n_points={}, n_query_points={}, \
             dist_sq={}, disp={})",
            self.inner.n_pairs(),
            mode,
            self.inner.n_points(),
            self.inner.n_query_points(),
            columns.dist_sq,
            columns.disp,
        )
    }
}

// ---------------------------------------------------------------------------
// PyNeighborList — the engine
// ---------------------------------------------------------------------------

/// Neighbor search over one point set, exposed to Python as
/// `molrs.core.NeighborList`.
///
/// The engine owns the cutoff, the backend that indexes space, and the box the
/// index was built for, and keeps the two halves of the job apart: ``build`` /
/// ``update`` index coordinates and enumerate nothing, ``neighbors`` turns the
/// index into a :class:`Neighbors` table. The pairs are half-shell — each
/// unordered pair once, with ``i < j``. A directed search against a *second*
/// point set is :class:`NeighborQuery`.
///
/// Parameters
/// ----------
/// cutoff : float
///     Interaction radius in Å. Fixed at construction because it sets the cell
///     width of the index.
/// brute_force : bool, optional
///     Use the O(N²) all-pairs backend instead of the O(N) cell list: the same
///     pairs — that is what makes it useful as an oracle — at a cost that
///     grows with the square of the particle count. Default ``False``.
///
/// Raises
/// ------
/// ValueError
///     If ``cutoff`` is not positive.
///
/// Passing the engine into :class:`VerletSkin` **moves** it: the list is
/// consumed, and every later method call on it raises ``ValueError`` — build a
/// new ``NeighborList`` instead.
///
/// Examples
/// --------
/// >>> nl = molrs.core.NeighborList(3.0)          # O(N) cell-list backend
/// >>> oracle = molrs.core.NeighborList(3.0, brute_force=True)
/// >>> nl.build(points, box)                 # index only — no pair table
/// >>> neigh = nl.neighbors()                # both columns (the default)
/// >>> lean = nl.neighbors(disp=False)       # indices + d² only
/// >>> nl.update(moved_points)               # re-index in the same box
#[pyclass(module = "molrs.core", name = "NeighborList", subclass)]
pub struct PyNeighborList {
    pub(crate) inner: Option<RsNeighborList>,
    brute_force: bool,
    points: Option<Array2<f64>>,
    simbox: Option<SimBox>,
}

fn engine_moved_err() -> PyErr {
    PyValueError::new_err("NeighborList has been moved into a VerletSkin; build a new NeighborList")
}

impl PyNeighborList {
    pub(crate) fn take(&mut self) -> PyResult<RsNeighborList> {
        self.inner.take().ok_or_else(engine_moved_err)
    }

    fn get(&self) -> PyResult<&RsNeighborList> {
        self.inner.as_ref().ok_or_else(engine_moved_err)
    }

    fn get_mut(&mut self) -> PyResult<&mut RsNeighborList> {
        self.inner.as_mut().ok_or_else(engine_moved_err)
    }
}

#[pymethods]
impl PyNeighborList {
    /// Build an engine: the O(N) cell-list backend (the production choice),
    /// or the O(N²) all-pairs one with ``brute_force=True``.
    #[new]
    #[pyo3(signature = (cutoff, brute_force=false))]
    fn new(cutoff: f64, brute_force: bool) -> PyResult<Self> {
        check_cutoff(cutoff)?;
        Ok(Self {
            inner: Some(if brute_force {
                RsNeighborList::brute_force(cutoff)
            } else {
                RsNeighborList::new(cutoff)
            }),
            brute_force,
            points: None,
            simbox: None,
        })
    }

    /// The cutoff distance (Å) fixed at construction.
    #[getter]
    fn cutoff(&self) -> PyResult<f64> {
        Ok(self.get()?.cutoff())
    }

    /// Index ``points`` in ``box`` — coordinates **and** box.
    ///
    /// Builds the spatial index and nothing else: no pairs are enumerated and
    /// no table is allocated. ``box`` is retained, so a later :meth:`update`
    /// can re-index new coordinates in the same box; a run whose box changes
    /// every step (a barostat) must call ``build`` again rather than
    /// ``update``.
    ///
    /// Parameters
    /// ----------
    /// points : numpy.ndarray, shape (N, 3), dtype float64
    ///     Cartesian coordinates in Å.
    /// box : Box
    ///     Simulation box supplying the periodicity used for minimum images.
    ///
    /// Raises
    /// ------
    /// ValueError
    ///     If ``points`` does not have shape ``(N, 3)``.
    fn build(&mut self, points: ArrayArg2<'_, f64>, r#box: &PyBox) -> PyResult<()> {
        check_points(&points, "points")?;
        self.get_mut()?.build(points.as_array(), &r#box.inner);
        self.points = Some(points.as_array().to_owned());
        self.simbox = Some(r#box.inner.clone());
        Ok(())
    }

    /// Re-index new coordinates in the box captured by the last :meth:`build`.
    ///
    /// The natural per-step call at fixed volume: the positions move, the box
    /// does not. Index-only, like ``build``. There is no skin — this re-indexes
    /// every time rather than deciding for you that the previous index was
    /// still good enough.
    ///
    /// Parameters
    /// ----------
    /// points : numpy.ndarray, shape (N, 3), dtype float64
    ///     New Cartesian coordinates in Å.
    ///
    /// Raises
    /// ------
    /// ValueError
    ///     If ``points`` does not have shape ``(N, 3)``, or if no ``build`` has
    ///     run yet — the box is then unknown, and guessing one would fold
    ///     minimum images against a box the caller never named.
    fn update(&mut self, points: ArrayArg2<'_, f64>) -> PyResult<()> {
        check_points(&points, "points")?;
        let engine = self.get_mut()?;
        // The core panics on an update before a build (the box is unknown);
        // a panic must not cross this seam, so check the engine's own state
        // and raise instead.
        if !engine.is_built() {
            return Err(PyValueError::new_err(
                "NeighborList.update reuses the box of the previous build: \
                 call build(points, box) first",
            ));
        }
        engine.update(points.as_array());
        self.points = Some(points.as_array().to_owned());
        Ok(())
    }

    /// Materialize the pairs into a :class:`Neighbors` table.
    ///
    /// Both physical columns are kept by default, so no downstream analysis is
    /// surprised by a missing one; pass ``False`` for a column this call site
    /// will not read. A dropped column cannot be added afterwards — rerun the
    /// search instead.
    ///
    /// Row order is unspecified: the cell-list backend materializes in
    /// parallel, and the work split decides the order.
    ///
    /// Parameters
    /// ----------
    /// dist_sq : bool, default True
    ///     Keep the squared minimum-image distance column (Å²), 8 B/pair.
    /// disp : bool, default True
    ///     Keep the minimum-image displacement column (Å), 24 B/pair.
    ///
    /// Returns
    /// -------
    /// Neighbors
    #[pyo3(signature = (dist_sq=true, disp=true))]
    fn neighbors(&self, dist_sq: bool, disp: bool) -> PyResult<PyNeighbors> {
        Ok(PyNeighbors {
            inner: self.get()?.neighbors(NeighborColumns { dist_sq, disp }),
        })
    }

    fn __repr__(&self) -> String {
        match &self.inner {
            Some(engine) => format!(
                "NeighborList(cutoff={}, built={})",
                engine.cutoff(),
                engine.is_built(),
            ),
            None => "NeighborList(<moved into VerletSkin>)".into(),
        }
    }

    /// Pickle as ``NeighborList(cutoff, brute_force)`` plus the last build's
    /// ``(points, box)`` (``None`` before a build), replayed by
    /// :meth:`__setstate__`.
    fn __reduce__<'py>(slf: &Bound<'py, Self>) -> PyResult<Bound<'py, PyTuple>> {
        let py = slf.py();
        let this = slf.borrow();
        let args = PyTuple::new(
            py,
            [
                this.cutoff()?.into_pyobject(py)?.into_any(),
                pyo3::types::PyBool::new(py, this.brute_force)
                    .to_owned()
                    .into_any(),
            ],
        )?;
        let state = match (&this.points, &this.simbox) {
            (Some(points), Some(simbox)) => (
                points.clone().into_pyarray(py),
                PyBox {
                    inner: simbox.clone(),
                },
            )
                .into_pyobject(py)?
                .into_any(),
            _ => py.None().into_bound(py),
        };
        crate::pickle::reduce_with_state(slf.as_any(), args, state)
    }

    /// Replay a pickled build: ``state`` is ``(points, box)`` or ``None``.
    fn __setstate__(
        &mut self,
        state: Option<(ArrayArg2<'_, f64>, PyRef<'_, PyBox>)>,
    ) -> PyResult<()> {
        if let Some((points, r#box)) = state {
            self.build(points, &r#box)?;
        }
        Ok(())
    }
}

// ---------------------------------------------------------------------------
// PyNeighborQuery — the cross door
// ---------------------------------------------------------------------------

/// Cross-query neighbor search, exposed to Python as `molrs.core.NeighborQuery`.
///
/// Build a spatial index from reference points once, then answer repeated
/// queries against it. This is the door for the *cross* question — "which
/// reference points lie within the cutoff of **these other** points?" — and is
/// the reason the class exists. A search of one point set against itself
/// belongs to :class:`NeighborList`, which can additionally re-index moved
/// coordinates and choose which columns a table keeps.
///
/// Every table returned here carries both physical columns.
///
/// Parameters
/// ----------
/// box : Box
///     Simulation box describing geometry and periodic boundaries. A
///     free-boundary search of an isolated cluster takes the non-periodic
///     bounding box padded by the cutoff:
///     ``Box.bounding(points, (cutoff,) * 3, (False,) * 3)``.
/// points : numpy.ndarray, shape (N, 3), dtype float64
///     Reference point positions in Å.
/// cutoff : float
///     Cutoff radius in Å.
///
/// Examples
/// --------
/// >>> nq = molrs.core.NeighborQuery(box, positions, cutoff=3.0)
/// >>> cross = nq.query(query_positions)   # directed, no ``i < j`` rule
/// >>> half = nq.query_self()              # half-shell over the reference set
#[pyclass(module = "molrs.core", name = "NeighborQuery", subclass)]
pub struct PyNeighborQuery {
    inner: RsNeighborQuery,
}

#[pymethods]
impl PyNeighborQuery {
    /// Build the spatial index.
    ///
    /// Raises
    /// ------
    /// ValueError
    ///     If ``points`` does not have shape ``(N, 3)`` or ``cutoff`` is not
    ///     positive.
    #[new]
    #[pyo3(signature = (r#box, points, cutoff))]
    fn new(r#box: &PyBox, points: ArrayArg2<'_, f64>, cutoff: f64) -> PyResult<Self> {
        check_points(&points, "points")?;
        check_cutoff(cutoff)?;
        Ok(Self {
            inner: RsNeighborQuery::new(&r#box.inner, points.as_array(), cutoff),
        })
    }

    /// Find all pairs where ``i`` indexes ``query_points`` and ``j`` indexes
    /// the reference points.
    ///
    /// Directed: every query point reports all of its reference neighbors, with
    /// no ``i < j`` constraint. Passing the reference coordinates back in means
    /// each point finds itself at distance zero and each pair appears twice —
    /// use :meth:`query_self` when each unordered pair should appear once.
    ///
    /// Returns
    /// -------
    /// Neighbors
    ///
    /// Raises
    /// ------
    /// ValueError
    ///     If ``query_points`` does not have shape ``(M, 3)``.
    fn query(&self, query_points: ArrayArg2<'_, f64>) -> PyResult<PyNeighbors> {
        check_points(&query_points, "query_points")?;
        Ok(PyNeighbors {
            inner: self.inner.query(query_points.as_array()),
        })
    }

    /// Find each unordered pair within the reference point set once (``i < j``).
    ///
    /// Returns
    /// -------
    /// Neighbors
    fn query_self(&self) -> PyNeighbors {
        PyNeighbors {
            inner: self.inner.query_self(),
        }
    }

    fn __repr__(&self) -> String {
        format!(
            "NeighborQuery(n_points={}, cutoff={})",
            self.inner.points().nrows(),
            self.inner.cutoff(),
        )
    }

    fn __reduce__<'py>(slf: &Bound<'py, Self>) -> PyResult<Bound<'py, PyTuple>> {
        let py = slf.py();
        let this = slf.borrow();
        crate::pickle::reduce_via_type(
            slf.as_any(),
            (
                PyBox {
                    inner: this.inner.simbox().clone(),
                },
                this.inner.points().to_owned().into_pyarray(py),
                this.inner.cutoff(),
            ),
        )
    }
}

fn skin_err(err: SkinError) -> PyErr {
    PyValueError::new_err(err.to_string())
}

/// A :class:`NeighborList` with Verlet skin — ``VerletSkin(NeighborList, …)``.
///
/// Constructed from a search engine whose cutoff is ``cutoff + skin``. The
/// engine is **moved** into this object. Passing the skin into
/// ``VelocityVerlet(..., neighbors=skin)`` moves it again into the integrator.
#[pyclass(module = "molrs.core", name = "VerletSkin", subclass)]
pub struct PyVerletSkin {
    pub(crate) inner: Option<RsVerletSkin>,
    brute_force: bool,
    x_hold: Array2<f64>,
    simbox: SimBox,
    every: usize,
    delay: usize,
    check: bool,
}

impl PyVerletSkin {
    pub(crate) fn take(&mut self) -> PyResult<RsVerletSkin> {
        self.inner
            .take()
            .ok_or_else(|| PyValueError::new_err("VerletSkin has already been moved"))
    }

    fn get(&self) -> PyResult<&RsVerletSkin> {
        self.inner
            .as_ref()
            .ok_or_else(|| PyValueError::new_err("VerletSkin has already been moved"))
    }

    pub(crate) fn get_mut(&mut self) -> PyResult<&mut RsVerletSkin> {
        self.inner
            .as_mut()
            .ok_or_else(|| PyValueError::new_err("VerletSkin has already been moved"))
    }
}

#[pymethods]
impl PyVerletSkin {
    /// Wrap ``neighbors`` (cutoff must equal ``cutoff + skin``) with Verlet policy.
    #[new]
    #[pyo3(signature = (neighbors, cutoff, positions, r#box, skin=0.0, every=1, delay=0, check=true, ago=0, n_rebuilds=0, ndanger=0))]
    #[allow(clippy::too_many_arguments, reason = "Public Python keyword arguments")]
    fn new(
        neighbors: &mut PyNeighborList,
        cutoff: f64,
        positions: ArrayArg2<'_, f64>,
        r#box: &PyBox,
        skin: f64,
        every: usize,
        delay: usize,
        check: bool,
        ago: usize,
        n_rebuilds: usize,
        ndanger: usize,
    ) -> PyResult<Self> {
        check_points(&positions, "positions")?;
        if cutoff <= 0.0 {
            return Err(PyValueError::new_err("cutoff must be > 0 Å"));
        }
        // Move the engine out; the emptied NeighborList raises on any later use.
        let search = neighbors.take()?;
        let policy = NeighborPolicy {
            skin,
            every,
            delay,
            check,
        };
        let mut inner = RsVerletSkin::new(
            search,
            cutoff,
            policy,
            positions.as_array(),
            r#box.inner.clone(),
        )
        .map_err(skin_err)?;
        if ago != 0 || n_rebuilds != 0 || ndanger != 0 {
            inner.restore_progress(ago, n_rebuilds, ndanger);
        }
        Ok(Self {
            inner: Some(inner),
            brute_force: neighbors.brute_force,
            x_hold: positions.as_array().to_owned(),
            simbox: r#box.inner.clone(),
            every,
            delay,
            check,
        })
    }

    #[getter]
    fn cutoff(&self) -> PyResult<f64> {
        Ok(self.get()?.cutoff())
    }

    #[getter]
    fn skin(&self) -> PyResult<f64> {
        Ok(self.get()?.skin())
    }

    #[getter]
    fn n_edges(&self) -> PyResult<usize> {
        Ok(self.get()?.n_edges())
    }

    #[getter]
    fn n_rebuilds(&self) -> PyResult<usize> {
        Ok(self.get()?.n_rebuilds())
    }

    #[getter]
    fn ago(&self) -> PyResult<usize> {
        Ok(self.get()?.ago())
    }

    fn update(&mut self, positions: ArrayArg2<'_, f64>) -> PyResult<bool> {
        check_points(&positions, "positions")?;
        let rebuilt = self
            .get_mut()?
            .update(positions.as_array())
            .map_err(skin_err)?;
        if rebuilt {
            self.x_hold = positions.as_array().to_owned();
        }
        Ok(rebuilt)
    }

    fn rebuild(&mut self, positions: ArrayArg2<'_, f64>) -> PyResult<()> {
        check_points(&positions, "positions")?;
        self.get_mut()?
            .rebuild(positions.as_array())
            .map_err(skin_err)?;
        self.x_hold = positions.as_array().to_owned();
        Ok(())
    }

    fn __repr__(&self) -> String {
        match &self.inner {
            Some(s) => format!(
                "VerletSkin(cutoff={}, skin={}, edges={}, rebuilds={})",
                s.cutoff(),
                s.skin(),
                s.n_edges(),
                s.n_rebuilds()
            ),
            None => "VerletSkin(<moved>)".into(),
        }
    }

    fn __reduce__<'py>(slf: &Bound<'py, Self>) -> PyResult<Bound<'py, PyTuple>> {
        let py = slf.py();
        let this = slf.borrow();
        let inner = this.get()?;
        let neighbors = PyNeighborList::new(inner.cutoff() + inner.skin(), this.brute_force)?;
        crate::pickle::reduce_via_type(
            slf.as_any(),
            (
                neighbors,
                inner.cutoff(),
                this.x_hold.clone().into_pyarray(py),
                PyBox {
                    inner: this.simbox.clone(),
                },
                inner.skin(),
                this.every,
                this.delay,
                this.check,
                inner.ago(),
                inner.n_rebuilds(),
                inner.ndanger(),
            ),
        )
    }
}
