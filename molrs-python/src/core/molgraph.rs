//! Python bindings for the ECS molecular graph.
//!
//! The core is an ECS *world*: entities are stable opaque handles, their data
//! lives in aligned component columns, and topology is kind-tagged relations.
//! This module exposes that faithfully:
//!
//! - [`PyMolGraph`] (`molrs.core.MolGraph`) — the domain-agnostic world: stable-handle
//!   entities, by-name component get/set, and the kind-tagged relation API.
//! - [`PyAtomistic`] (`molrs.core.Atomistic`) / [`PyCoarseGrain`]
//!   (`molrs.core.CoarseGrain`) — peer leaves that **hold a core [`Atomistic`] /
//!   [`CoarseGrain`] from construction** (never converted from a `MolGraph`,
//!   never converted into each other). They add the
//!   domain builders (`add_atom`/`add_bond`/…), own `to_frame` /
//!   `from_frame` (`self.inner.to_frame()`, zero conversion), carry the
//!   graph's `props` and live views ([`super::views`]), and are
//!   subclassable, like every core data class. They subclass `MolGraph`;
//!   the generic graph API is shared via the [`graph_world_impl!`] macro,
//!   which always operates on the receiver's *own* graph (`self.mol()` /
//!   `self.mol_mut()`), so the leaf's graph is the single data slot.
//!
//! Handles are stable opaque `int`s (generational slotmap keys); removing one
//! entity never invalidates another, and a stale handle raises.
//!
//! A leaf is built natively ([`PyAtomistic::from_core`] for a new graph,
//! [`PyAtomistic::derive`] for one derived from another, which keeps its
//! `props`). The empty base `MolGraph` such a construction carries is
//! structural, not waste: PyO3 builds a subclass base-then-subclass and
//! `PyMolGraph`'s only field is a `MolGraph`, so an instance of a class declaring
//! `extends = PyMolGraph` necessarily has one, and the leaf-first accessors above
//! make sure nothing ever reads it.

use crate::array::ArrayArgDyn;
use std::collections::HashMap;

use std::str::FromStr;

use ndarray::Array2;

use numpy::{IntoPyArray, PyArray1, PyArray2};

use pyo3::exceptions::{PyTypeError, PyValueError};

use pyo3::prelude::*;

use pyo3::types::{PyDict, PyList, PyTuple, PyType};

use pyo3::{PyTraverseError, PyVisit};

use molrs::core::CenterError;

use molrs::core::keys;

use molrs::core::LinkError;

use molrs::core::PortKind;

use molrs::core::EntityCell;

use molrs::core::{Atomistic, ExtractedAtomistic};

use molrs::core::{BondNumber, BondOrder};

use molrs::core::{CoarseGrain, ExtractedCoarseGrain};

use molrs::core::{
    KindId, MolGraph, NodeId, PropValue, node_from_u64, node_to_u64, relation_from_u64,
    relation_to_u64,
};

use super::graph_views::{Leaf, PyNodeRef, PyRefs, PyRelationBuckets, ViewCache};
use molrs::core::keys::BEAD_ATOMS;

use crate::core::frame::PyFrame;

use crate::error::molrs_error_to_pyerr;

use crate::op::vector_to_py;

// ---------------------------------------------------------------------------
// Value conversion helpers
// ---------------------------------------------------------------------------

/// Convert a Python scalar to a [`PropValue`].
///
/// `bool` is tried before `int` because a Python `bool` is a subclass of `int`
/// (so `extract::<i64>()` would silently collapse `True`→`1`); `int` is tried
/// before `float` so an integer literal doesn't become a float. Anything that is
/// not `bool` / `int` / `float` / `str` is rejected fail-fast — non-representable
/// values (lists, `None`, arbitrary objects) MUST raise, never be stashed. An
/// integer outside the stored 32-bit range raises `OverflowError` rather than
/// wrapping (or, past 64 bits, silently becoming a float).
pub(crate) fn py_to_prop(value: &Bound<'_, PyAny>) -> PyResult<PropValue> {
    // `extract::<bool>()` matches only a genuine Python `bool`, not an `int`.
    if let Ok(b) = value.extract::<bool>() {
        Ok(PropValue::Bool(b))
    } else if value.hasattr(pyo3::intern!(value.py(), "__index__"))? {
        // An integer (Python `int`, numpy integer). Extracting straight to the
        // stored width raises `OverflowError` out of range instead of wrapping.
        value.extract::<i32>().map(PropValue::Int)
    } else if let Ok(f) = value.extract::<f64>() {
        Ok(PropValue::F64(f))
    } else if let Ok(s) = value.extract::<String>() {
        Ok(PropValue::Str(s))
    } else {
        Err(PyTypeError::new_err(
            "component value must be bool, int, float, or str",
        ))
    }
}

pub(crate) fn cell_to_py(py: Python<'_>, cell: EntityCell<'_>) -> PyResult<Py<PyAny>> {
    Ok(match cell {
        EntityCell::F64(v) => v.into_pyobject(py)?.into_any().unbind(),
        EntityCell::I32(v) => v.into_pyobject(py)?.into_any().unbind(),
        EntityCell::Str(s) => s.into_pyobject(py)?.into_any().unbind(),
        EntityCell::Bool(b) => b.into_pyobject(py)?.to_owned().into_any().unbind(),
    })
}

pub(crate) fn prop_to_py(py: Python<'_>, value: &PropValue) -> PyResult<Py<PyAny>> {
    Ok(match value {
        PropValue::F64(v) => v.into_pyobject(py)?.into_any().unbind(),
        PropValue::Int(v) => v.into_pyobject(py)?.into_any().unbind(),
        PropValue::Str(s) => s.into_pyobject(py)?.into_any().unbind(),
        PropValue::Bool(b) => b.into_pyobject(py)?.to_owned().into_any().unbind(),
    })
}

/// Resolve a kind name to a [`KindId`], or raise a Python `ValueError`.
fn kind_id_checked(mol: &MolGraph, kind: &str) -> PyResult<KindId> {
    mol.kind_id(kind)
        .ok_or_else(|| PyValueError::new_err(format!("kind '{kind}' is not registered")))
}

/// Render a [`CenterError`] as a Python `ValueError`, through
/// [`center_error_message`].
fn center_error_to_pyerr(e: CenterError) -> PyErr {
    PyValueError::new_err(center_error_message(e))
}

/// The message of a [`CenterError`] as Python sees it.
///
/// Node ids cross as the `int` handles Python holds (`node_to_u64`), never as
/// the Rust debug form `NodeId(3v1)`. Every variant is matched by name, so a
/// new variant is a compile error here rather than a silent fallback.
pub(crate) fn center_error_message(e: CenterError) -> String {
    match e {
        CenterError::Empty => "center of an empty node set".to_owned(),
        CenterError::NotFound { node } => {
            format!("node {} is not in this graph", node_to_u64(node))
        }
        CenterError::BadPosition { node } => format!(
            "node {} has a missing or non-finite '{}'/'{}'/'{}'",
            node_to_u64(node),
            keys::X,
            keys::Y,
            keys::Z
        ),
        CenterError::BadMass { node } => format!(
            "node {} has a missing, negative or non-finite '{}'",
            node_to_u64(node),
            keys::MASS
        ),
        CenterError::ZeroMass => "total mass is not positive and finite".to_owned(),
    }
}

/// Render a [`LinkError`] as a Python `ValueError`, through
/// [`link_error_message`].
fn link_error_to_pyerr(e: LinkError) -> PyErr {
    PyValueError::new_err(link_error_message(e))
}

/// The message of a [`LinkError`] as Python sees it.
///
/// Port and atom ids cross as the `int` handles Python holds
/// (`relation_to_u64` / `node_to_u64`), never as `RelationId(..)` / `NodeId(..)`.
/// Every variant is matched by name, with no catch-all arm.
pub(crate) fn link_error_message(e: LinkError) -> String {
    let port = relation_to_u64;
    let atom = node_to_u64;
    match e {
        LinkError::Port(inner) => format!("port does not read back: {inner}"),
        LinkError::StalePort { port: p } => format!(
            "port {} is stale: its anchor–handle bond no longer exists",
            port(p)
        ),
        LinkError::Incompatible { a, b } => {
            format!("ports {} and {} are not compatible", port(a), port(b))
        }
        LinkError::SameAnchor { a, b } => format!(
            "ports {} and {} share one anchor; a bond cannot join an atom to itself",
            port(a),
            port(b)
        ),
        LinkError::AlreadyBonded { a, b } => {
            format!("anchors {} and {} are already bonded", atom(a), atom(b))
        }
        LinkError::BranchReachesAnchor { port: p } => {
            format!("port {}'s handle branch reaches its own anchor", port(p))
        }
        LinkError::BranchesOverlap => "the two ports' handle branches overlap".to_owned(),
        LinkError::OneSidedCharge { anchor } => format!(
            "charge is present on only part of anchor {} and its handle branch",
            atom(anchor)
        ),
        LinkError::Graph(inner) => format!("graph refused a read: {inner}"),
    }
}

// ---------------------------------------------------------------------------
// Shared generic-world method body
// ---------------------------------------------------------------------------

/// Emits the generic ECS world `#[pymethods]` for a graph type. Always operates
/// on `self.mol()` / `self.mol_mut()` (the receiver's own graph), so each
/// concrete type's graph is the single data slot — a leaf's methods read/write
/// the leaf's own core graph, never an empty base.
macro_rules! graph_world_impl {
    ($ty:ty) => {
        #[pymethods]
        impl $ty {
            // ---- ports: named attachment points any graph may carry ----

            /// Record a descriptor on the ``(anchor, handle)`` valence; the
            /// ``ports`` relation kind is registered on first use.
            ///
            /// Parameters
            /// ----------
            /// anchor : int
            ///     The node that keeps its place in the product.
            /// handle : int
            ///     A node bonded to ``anchor``: the root of the leaving group.
            ///     Endpoint order is load-bearing.
            /// kind : str
            ///     The notation glyph — one of ``"$"``, ``"<"``, ``">"``, ``"!"``.
            /// label : str, optional
            ///     Free-form descriptor label; ``""`` (the default) means unnamed.
            /// order : int, optional
            ///     Multiplicity of the bond this port will form, ``1`` to ``4``
            ///     (default ``1``).
            ///
            /// Returns
            /// -------
            /// int
            ///     The port's stable relation handle.
            ///
            /// Raises
            /// ------
            /// ValueError
            ///     If ``kind`` is not one of the four glyphs, ``handle`` is not
            ///     bonded to ``anchor``, ``order`` is not a definite bond
            ///     number, the valence already carries a port, or a handle is
            ///     stale or unknown.
            #[pyo3(signature = (anchor, handle, kind, label="", order=1))]
            fn add_port(
                &mut self,
                anchor: u64,
                handle: u64,
                kind: &str,
                label: &str,
                order: u32,
            ) -> PyResult<u64> {
                let kind = PortKind::from_str(kind).map_err(molrs_error_to_pyerr)?;
                self.mol_mut()
                    .add_port(
                        node_from_u64(anchor),
                        node_from_u64(handle),
                        kind,
                        label,
                        BondNumber::from_code(order),
                    )
                    .map(relation_to_u64)
                    .map_err(molrs_error_to_pyerr)
            }

            /// Number of ports (``0`` when the graph carries none).
            #[getter]
            fn n_ports(&self) -> usize {
                self.mol().n_ports()
            }

            /// Record the unit instance ``node`` came from, under ``frag_id``.
            ///
            /// Raises
            /// ------
            /// ValueError
            ///     If ``id`` exceeds the widest identifier a node column stores,
            ///     or ``node`` is stale or unknown.
            fn set_frag_id(&mut self, node: u64, id: u32) -> PyResult<()> {
                self.mol_mut()
                    .set_frag_id(node_from_u64(node), id)
                    .map_err(molrs_error_to_pyerr)
            }

            /// The unit instance ``node`` came from, or ``None``.
            fn frag_id(&self, node: u64) -> Option<u32> {
                self.mol().frag_id(node_from_u64(node))
            }

            /// Propagate each ``frag_id`` to the unlabelled degree-1 nodes
            /// hanging off a labelled one (one pass); returns how many were
            /// labelled. The relabel step after a conformer added hydrogens.
            fn inherit_frag_ids(&mut self) -> usize {
                self.mol_mut().inherit_frag_ids()
            }

            /// Join port ``a`` to port ``b`` with a new anchor–anchor bond.
            ///
            /// Both leaving groups are removed, their partial charge (e) folds
            /// onto the anchors, and the anchors are bonded with the port
            /// order. No coordinate moves.
            ///
            /// Returns
            /// -------
            /// int
            ///     The new bond's relation handle.
            ///
            /// Raises
            /// ------
            /// ValueError
            ///     If a handle names no live port, a port is stale, the two
            ///     ports do not accept each other, share an anchor, their
            ///     anchors are already bonded, or the leaving groups overlap or
            ///     reach an anchor; the graph is unchanged.
            /// OverflowError
            ///     If ``a`` or ``b`` is negative.
            fn link(&mut self, a: u64, b: u64) -> PyResult<u64> {
                self.mol_mut()
                    .link(relation_from_u64(a), relation_from_u64(b))
                    .map(relation_to_u64)
                    .map_err(link_error_to_pyerr)
            }

            // ---- entities ----

            /// Spawn a new entity, returning its stable handle.
            fn spawn(&mut self) -> u64 {
                node_to_u64(self.mol_mut().add_node())
            }

            /// Remove an entity (cascades incident relations). Errors if stale.
            fn despawn(&mut self, h: u64) -> PyResult<()> {
                self.mol_mut()
                    .remove_node(node_from_u64(h))
                    .map(|_| ())
                    .map_err(molrs_error_to_pyerr)
            }

            /// All live entity handles, in row order.
            fn entities(&self) -> Vec<u64> {
                self.mol().node_ids().map(node_to_u64).collect()
            }

            /// Whether `h` is a live entity handle.
            fn has_entity(&self, h: u64) -> bool {
                self.mol().node_table().contains(node_from_u64(h))
            }

            /// Number of entities.
            #[getter]
            fn n_nodes(&self) -> usize {
                self.mol().n_nodes()
            }

            // ---- components ----

            /// Read entity `h`'s component `key` (``None`` if absent).
            ///
            /// `key` is a :class:`molrs.core.keys.Key` or ``str``.
            fn get(&self, py: Python<'_>, h: u64, key: &Bound<'_, PyAny>) -> PyResult<Py<PyAny>> {
                let key = crate::core::schema::extract_column_key(key)?;
                match self.mol().node_table().value(node_from_u64(h), &key) {
                    Some(cell) => cell_to_py(py, cell),
                    None => Ok(py.None()),
                }
            }

            /// Set entity `h`'s component `key` (``value`` is int|float|str).
            ///
            /// `key` is a :class:`molrs.core.keys.Key` or ``str``.
            fn set(
                &mut self,
                h: u64,
                key: &Bound<'_, PyAny>,
                value: &Bound<'_, PyAny>,
            ) -> PyResult<()> {
                let key = crate::core::schema::extract_column_key(key)?;
                let pv = py_to_prop(value)?;
                self.mol_mut()
                    .set_node(node_from_u64(h), &key, pv)
                    .map_err(molrs_error_to_pyerr)
            }

            /// Whether entity `h` has component `key`.
            ///
            /// `key` is a :class:`molrs.core.keys.Key` or ``str``.
            fn has(&self, h: u64, key: &Bound<'_, PyAny>) -> PyResult<bool> {
                let key = crate::core::schema::extract_column_key(key)?;
                Ok(self.mol().node_table().has(node_from_u64(h), &key))
            }

            /// Clear entity `h`'s component `key` (no-op if absent).
            ///
            /// `key` is a :class:`molrs.core.keys.Key` or ``str``.
            fn delete(&mut self, h: u64, key: &Bound<'_, PyAny>) -> PyResult<()> {
                let key = crate::core::schema::extract_column_key(key)?;
                self.mol_mut()
                    .clear_node(node_from_u64(h), &key)
                    .map_err(molrs_error_to_pyerr)
            }

            /// Component keys currently set on entity `h`, in column order.
            fn node_keys(&self, h: u64) -> Vec<String> {
                self.mol()
                    .node_table()
                    .row_cells(node_from_u64(h))
                    .map(|(k, _)| k.to_owned())
                    .collect()
            }

            // ---- relations ----

            /// Register a relation kind (idempotent for a matching arity).
            fn register_kind(&mut self, kind: &str, arity: usize) -> PyResult<()> {
                let m = self.mol_mut();
                if let Some(kid) = m.kind_id(kind) {
                    let existing = m.arity(kid);
                    if existing != arity {
                        return Err(PyValueError::new_err(format!(
                            "kind '{kind}' already registered with arity {existing}, got {arity}"
                        )));
                    }
                    return Ok(());
                }
                m.register_kind(kind, arity);
                Ok(())
            }

            /// Names of all registered relation kinds.
            fn kinds(&self) -> Vec<String> {
                self.mol()
                    .kind_ids()
                    .map(|kid| self.mol().kind_name(kid).to_owned())
                    .collect()
            }

            /// Fixed endpoint count for a registered relation kind.
            fn kind_arity(&self, kind: &str) -> PyResult<usize> {
                let kid = kind_id_checked(self.mol(), kind)?;
                Ok(self.mol().arity(kid))
            }

            /// Add a relation of `kind` over node handles, returning its handle.
            fn add_relation(&mut self, kind: &str, nodes: Vec<u64>) -> PyResult<u64> {
                let kid = kind_id_checked(self.mol(), kind)?;
                let nids: Vec<NodeId> = nodes.into_iter().map(node_from_u64).collect();
                let rid = self
                    .mol_mut()
                    .add_relation(kid, &nids)
                    .map_err(molrs_error_to_pyerr)?;
                Ok(relation_to_u64(rid))
            }

            /// Endpoint node handles of relation `rh` of `kind`.
            fn relation_nodes(&self, kind: &str, rh: u64) -> PyResult<Vec<u64>> {
                let kid = kind_id_checked(self.mol(), kind)?;
                let nodes = self
                    .mol()
                    .relation_nodes(kid, relation_from_u64(rh))
                    .map_err(molrs_error_to_pyerr)?;
                Ok(nodes.iter().map(|&n| node_to_u64(n)).collect())
            }

            /// Relations of `kind` incident to node `nh`, as
            /// `(relation_handle, other_node_handle)` pairs, via the adjacency
            /// index (O(degree)). Only arity-2 kinds are tracked in adjacency.
            fn incident_relations(&self, nh: u64, kind: &str) -> PyResult<Vec<(u64, u64)>> {
                let kid = kind_id_checked(self.mol(), kind)?;
                let nid = node_from_u64(nh);
                Ok(self
                    .mol()
                    .neighbor_relations(nid)
                    .filter(|(k, _, _)| *k == kid)
                    .map(|(_, rid, other)| (relation_to_u64(rid), node_to_u64(other)))
                    .collect())
            }

            /// Set a property on relation `rh` of `kind`.
            fn set_relation_prop(
                &mut self,
                kind: &str,
                rh: u64,
                key: &str,
                value: &Bound<'_, PyAny>,
            ) -> PyResult<()> {
                let kid = kind_id_checked(self.mol(), kind)?;
                let pv = py_to_prop(value)?;
                self.mol_mut()
                    .set_relation_prop(kid, relation_from_u64(rh), key, pv)
                    .map_err(molrs_error_to_pyerr)
            }

            /// Read a property of relation `rh` of `kind` (``None`` if absent).
            fn get_relation_prop(
                &self,
                py: Python<'_>,
                kind: &str,
                rh: u64,
                key: &str,
            ) -> PyResult<Py<PyAny>> {
                let kid = kind_id_checked(self.mol(), kind)?;
                let rel = self
                    .mol()
                    .get_relation(kid, relation_from_u64(rh))
                    .map_err(molrs_error_to_pyerr)?;
                match rel.props.get(key) {
                    Some(v) => prop_to_py(py, v),
                    None => Ok(py.None()),
                }
            }

            /// Property keys currently set on relation `rh` of `kind`.
            fn relation_keys(&self, kind: &str, rh: u64) -> PyResult<Vec<String>> {
                let kid = kind_id_checked(self.mol(), kind)?;
                let rel = self
                    .mol()
                    .get_relation(kid, relation_from_u64(rh))
                    .map_err(molrs_error_to_pyerr)?;
                Ok(rel.props.keys().map(|k| k.to_owned()).collect())
            }

            /// Clear property `key` on relation `rh` of `kind` (no-op if absent).
            fn delete_relation_prop(&mut self, kind: &str, rh: u64, key: &str) -> PyResult<()> {
                let kid = kind_id_checked(self.mol(), kind)?;
                self.mol_mut()
                    .clear_relation_prop(kid, relation_from_u64(rh), key)
                    .map_err(molrs_error_to_pyerr)
            }

            /// Remove relation `rh` of `kind`.
            fn remove_relation(&mut self, kind: &str, rh: u64) -> PyResult<()> {
                let kid = kind_id_checked(self.mol(), kind)?;
                self.mol_mut()
                    .remove_relation(kid, relation_from_u64(rh))
                    .map(|_| ())
                    .map_err(molrs_error_to_pyerr)
            }

            /// Number of relations of `kind`.
            fn n_relations(&self, kind: &str) -> PyResult<usize> {
                let kid = kind_id_checked(self.mol(), kind)?;
                Ok(self.mol().n_relations(kid))
            }

            /// Live relation handles of `kind`, in row order.
            ///
            /// Authoritative enumeration — callers must not probe opaque handle
            /// ranges. Returns an empty list for a registered kind with no
            /// relations; errors only if `kind` is unregistered.
            fn relation_ids(&self, kind: &str) -> PyResult<Vec<u64>> {
                let kid = kind_id_checked(self.mol(), kind)?;
                Ok(self.mol().relation_ids(kid).map(relation_to_u64).collect())
            }

            // ---- zero-copy columns ----

            /// The component column `key`, aligned to row order (length ==
            /// `n_nodes`), as a numpy array of the column's own element type.
            ///
            /// A `f64` column is a **zero-copy view** that writes through to the
            /// world (`col[i] = v` updates the entity at row `i`); `i32`, `bool`
            /// and `str` columns are copied. The view borrows the world's storage;
            /// structural mutation (`spawn`/`despawn`) may reallocate or reorder
            /// the column and invalidate an outstanding view — re-fetch after
            /// such ops.
            ///
            /// Every entity must carry the component: a column with a hole is a
            /// `KeyError` naming how many entities lack it, never a silently
            /// zero-filled array. Use :meth:`validity` to find the holes and
            /// :meth:`get` for entity-wise reads.
            fn column<'py>(
                slf: Bound<'py, $ty>,
                key: &Bound<'_, PyAny>,
            ) -> PyResult<Bound<'py, PyAny>> {
                use molrs::core::EntityColumn;

                let key = crate::core::schema::extract_column_key(key)?;
                let py = slf.py();
                let this = slf.borrow();
                let table = this.mol().node_table();
                let validity = table.col_validity(&key).ok_or_else(|| {
                    pyo3::exceptions::PyKeyError::new_err(format!(
                        "component '{key}' is absent from every entity"
                    ))
                })?;
                let holes = validity.as_slice().iter().filter(|v| !**v).count();
                if holes > 0 {
                    return Err(pyo3::exceptions::PyKeyError::new_err(format!(
                        "component '{key}' is absent on {holes} of {} entities; a column \
                         needs it on every one (see validity())",
                        validity.len()
                    )));
                }
                let column = table.column(&key).ok_or_else(|| {
                    pyo3::exceptions::PyKeyError::new_err(format!(
                        "component '{key}' is absent from every entity"
                    ))
                })?;
                match column {
                    EntityColumn::F64(data, _) => {
                        let (ptr, len) = (data.as_ptr(), data.len());
                        drop(this);
                        // SAFETY: `slf` owns the backing Vec and is held as the array's base
                        // object, so the memory stays valid for the array's lifetime; the
                        // documented contract forbids structural mutation while held.
                        let view = unsafe { numpy::ndarray::ArrayView1::from_shape_ptr(len, ptr) };
                        Ok(
                            unsafe { numpy::PyArray1::borrow_from_array(&view, slf.into_any()) }
                                .into_any(),
                        )
                    }
                    EntityColumn::I32(data, _) => {
                        Ok(numpy::PyArray1::from_slice(py, data).into_any())
                    }
                    EntityColumn::Bool(data, _) => {
                        Ok(numpy::PyArray1::from_slice(py, data).into_any())
                    }
                    EntityColumn::Str(data, _) => {
                        let list = pyo3::types::PyList::new(py, data.iter().map(String::as_str))?;
                        py.import("numpy")?.call_method1("array", (list,))
                    }
                }
            }

            /// Names of every component column registered on the node table
            /// (a component set on at least one entity), in no particular order.
            fn columns(&self) -> Vec<String> {
                self.mol()
                    .node_table()
                    .columns()
                    .map(str::to_owned)
                    .collect()
            }

            /// Validity mask (numpy `bool` array, copied) of component column `key`,
            /// aligned to row order. `True` where the entity at that row has the
            /// component set.
            ///
            /// `key` is a :class:`molrs.core.keys.Key` or ``str``.
            fn validity<'py>(
                &self,
                py: Python<'py>,
                key: &Bound<'_, PyAny>,
            ) -> PyResult<Bound<'py, numpy::PyArray1<bool>>> {
                let key = crate::core::schema::extract_column_key(key)?;
                let valid =
                    self.mol().node_table().col_validity(&key).ok_or_else(|| {
                        PyValueError::new_err(format!("column '{key}' is absent"))
                    })?;
                Ok(numpy::PyArray1::from_slice(py, valid.as_slice()))
            }

            // ---- zero-copy adopt ----

            /// Zero-copy adopt: **move** `other`'s graph storage into `self`,
            /// leaving `other` empty. Handles in the adopted graph stay valid
            /// (the whole generational slotmap is moved, not reindexed). For
            /// taking ownership of a graph produced elsewhere without a per-node
            /// copy. Defined per leaf so it swaps the leaf's own backing store.
            /// Views of either graph taken before are stale afterwards.
            fn adopt(&mut self, other: &mut $ty) {
                self.inner = std::mem::take(&mut other.inner);
                self.forget_views();
                other.forget_views();
            }
        }
    };
}

// ---------------------------------------------------------------------------
// ExtractedSubgraph — result of radius-ball extraction
// ---------------------------------------------------------------------------

/// Result of :meth:`Atomistic.extract_subgraph` / :meth:`CoarseGrain.extract_subgraph`.
///
/// * ``graph`` — the extracted leaf (``Atomistic`` or ``CoarseGrain``)
/// * ``boundary`` — parent handles with a neighbor outside the ball
/// * ``parent_of`` — ``{new_handle: parent_handle}``
/// * ``hops`` — ``{parent_handle: hops_from_nearest_center}``
/// * ``node_map`` — ``{parent_handle: new_handle}``
#[pyclass(
    module = "molrs.core",
    name = "ExtractedSubgraph",
    skip_from_py_object,
    subclass
)]

pub struct PyExtractedSubgraph {
    graph: Py<PyAny>,
    boundary: Vec<u64>,
    parent_of: HashMap<u64, u64>,
    hops: HashMap<u64, i64>,
    node_map: HashMap<u64, u64>,
}

#[pymethods]
impl PyExtractedSubgraph {
    #[new]
    fn new(
        graph: Py<PyAny>,
        boundary: Vec<u64>,
        parent_of: HashMap<u64, u64>,
        hops: HashMap<u64, i64>,
        node_map: HashMap<u64, u64>,
    ) -> Self {
        Self {
            graph,
            boundary,
            parent_of,
            hops,
            node_map,
        }
    }

    fn __reduce__<'py>(slf: &Bound<'py, Self>) -> PyResult<Bound<'py, PyTuple>> {
        crate::pickle::reduce_via_type(
            slf.as_any(),
            (
                slf.getattr("graph")?,
                slf.getattr("boundary")?,
                slf.getattr("parent_of")?,
                slf.getattr("hops")?,
                slf.getattr("node_map")?,
            ),
        )
    }

    #[getter]
    fn graph(&self, py: Python<'_>) -> Py<PyAny> {
        self.graph.clone_ref(py)
    }

    #[getter]
    fn boundary(&self) -> Vec<u64> {
        self.boundary.clone()
    }

    #[getter]
    fn parent_of(&self) -> HashMap<u64, u64> {
        self.parent_of.clone()
    }

    #[getter]
    fn hops(&self) -> HashMap<u64, i64> {
        self.hops.clone()
    }

    #[getter]
    fn node_map(&self) -> HashMap<u64, u64> {
        self.node_map.clone()
    }

    fn __repr__(&self) -> String {
        format!(
            "ExtractedSubgraph(n_boundary={}, n_nodes_mapped={})",
            self.boundary.len(),
            self.node_map.len()
        )
    }
}

impl PyExtractedSubgraph {
    fn from_atomistic(
        py: Python<'_>,
        source: &PyAtomistic,
        ext: ExtractedAtomistic,
    ) -> PyResult<Self> {
        let graph = source.derive(py, ext.graph)?.into_any();
        Ok(Self::with_maps(
            graph,
            ext.boundary,
            ext.parent_of,
            ext.hops,
            ext.node_map,
        ))
    }

    fn from_coarsegrain(
        py: Python<'_>,
        source: &PyCoarseGrain,
        ext: ExtractedCoarseGrain,
    ) -> PyResult<Self> {
        let graph = source.derive(py, ext.graph)?.into_any();
        Ok(Self::with_maps(
            graph,
            ext.boundary,
            ext.parent_of,
            ext.hops,
            ext.node_map,
        ))
    }

    fn with_maps(
        graph: Py<PyAny>,
        boundary: Vec<NodeId>,
        parent_of: HashMap<NodeId, NodeId>,
        hops: HashMap<NodeId, i64>,
        node_map: HashMap<NodeId, NodeId>,
    ) -> Self {
        let pairs = |map: HashMap<NodeId, NodeId>| {
            map.into_iter()
                .map(|(k, v)| (node_to_u64(k), node_to_u64(v)))
                .collect()
        };
        Self {
            graph,
            boundary: boundary.into_iter().map(node_to_u64).collect(),
            parent_of: pairs(parent_of),
            hops: hops.into_iter().map(|(k, v)| (node_to_u64(k), v)).collect(),
            node_map: pairs(node_map),
        }
    }
}

// ---------------------------------------------------------------------------
// Pickling: a graph's content as plain Python data
// ---------------------------------------------------------------------------

/// A graph's content as plain Python data — node fields in row order and,
/// per relation kind, its arity and its relations as endpoint *rows* plus
/// fields. Handles are not kept (a restored graph mints fresh ones), which
/// is why a view pickles by row.
trait GraphContent {
    fn dump_content<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyTuple>>;
    fn load_content(&mut self, content: &Bound<'_, PyAny>) -> PyResult<()>;
}

impl GraphContent for MolGraph {
    fn dump_content<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyTuple>> {
        let table = self.node_table();
        let nodes = PyList::empty(py);
        let mut row_of: HashMap<NodeId, usize> = HashMap::new();
        for (row, id) in self.node_ids().enumerate() {
            row_of.insert(id, row);
            let fields = PyDict::new(py);
            for (key, cell) in table.row_cells(id) {
                fields.set_item(key, cell_to_py(py, cell)?)?;
            }
            nodes.append(fields)?;
        }
        let kinds = PyList::empty(py);
        for kid in self.kind_ids() {
            let relations = PyList::empty(py);
            for (_, relation) in self.relations(kid) {
                let ends: Vec<usize> = relation.nodes.iter().map(|node| row_of[node]).collect();
                let fields = PyDict::new(py);
                for (key, value) in &relation.props {
                    fields.set_item(key, prop_to_py(py, value)?)?;
                }
                relations.append((ends, fields))?;
            }
            kinds.append((self.kind_name(kid), self.arity(kid), relations))?;
        }
        (nodes, kinds).into_pyobject(py)
    }

    #[allow(
        clippy::type_complexity,
        reason = "the pickled kind table: (name, arity, [(endpoint rows, fields)])"
    )]
    fn load_content(&mut self, content: &Bound<'_, PyAny>) -> PyResult<()> {
        let (nodes, kinds): (
            Vec<Bound<'_, PyDict>>,
            Vec<(String, usize, Vec<(Vec<usize>, Bound<'_, PyDict>)>)>,
        ) = content.extract()?;
        let mut ids = Vec::with_capacity(nodes.len());
        for fields in nodes {
            let id = self.add_node();
            for (key, value) in fields.iter() {
                self.set_node(id, &key.extract::<String>()?, py_to_prop(&value)?)
                    .map_err(molrs_error_to_pyerr)?;
            }
            ids.push(id);
        }
        for (name, arity, relations) in kinds {
            let kid = match self.kind_id(&name) {
                Some(kid) => kid,
                None => self.register_kind(&name, arity),
            };
            for (ends, fields) in relations {
                let nodes = ends
                    .iter()
                    .map(|&row| {
                        ids.get(row).copied().ok_or_else(|| {
                            PyValueError::new_err(format!("relation endpoint row {row} is no node"))
                        })
                    })
                    .collect::<PyResult<Vec<NodeId>>>()?;
                let rid = self
                    .add_relation(kid, &nodes)
                    .map_err(molrs_error_to_pyerr)?;
                for (key, value) in fields.iter() {
                    self.set_relation_prop(
                        kid,
                        rid,
                        &key.extract::<String>()?,
                        py_to_prop(&value)?,
                    )
                    .map_err(molrs_error_to_pyerr)?;
                }
            }
        }
        Ok(())
    }
}

// ---------------------------------------------------------------------------
// PyMolGraph — the generic world
// ---------------------------------------------------------------------------

/// Domain-agnostic ECS world, exposed to Python as `molrs.core.MolGraph`.
#[pyclass(module = "molrs.core", name = "MolGraph", subclass)]
pub struct PyMolGraph {
    inner: MolGraph,
}

impl PyMolGraph {
    fn mol(&self) -> &MolGraph {
        &self.inner
    }
    fn mol_mut(&mut self) -> &mut MolGraph {
        &mut self.inner
    }
    /// The base every leaf carries (see the module docs).
    fn base() -> Self {
        Self {
            inner: MolGraph::new(),
        }
    }
    /// A `MolGraph` has no views.
    fn forget_views(&mut self) {}
}

#[pymethods]
impl PyMolGraph {
    /// Create an empty world.
    #[new]
    fn new() -> Self {
        Self::base()
    }

    fn __reduce__<'py>(slf: &Bound<'py, Self>) -> PyResult<Bound<'py, PyTuple>> {
        let py = slf.py();
        let content = slf.borrow().inner.dump_content(py)?;
        crate::pickle::reduce_with_state(
            slf.as_any(),
            PyTuple::empty(py),
            PyTuple::new(py, [content])?.into_any(),
        )
    }

    fn __setstate__(&mut self, state: (Bound<'_, PyAny>,)) -> PyResult<()> {
        let mut inner = MolGraph::new();
        inner.load_content(&state.0)?;
        self.inner = inner;
        Ok(())
    }
}

graph_world_impl!(PyMolGraph);

/// Any graph object: a `MolGraph`, an `Atomistic` or a `CoarseGrain`.
pub(crate) enum AnyGraph<'py> {
    MolGraph(Bound<'py, PyMolGraph>),
    Atomistic(Bound<'py, PyAtomistic>),
    CoarseGrain(Bound<'py, PyCoarseGrain>),
}

impl<'py> AnyGraph<'py> {
    /// The graph `obj` is.
    ///
    /// # Errors
    ///
    /// `TypeError` when `obj` is no graph.
    pub(crate) fn of(obj: &Bound<'py, PyAny>) -> PyResult<Self> {
        // Leaves first: a leaf is also a `MolGraph` (its empty base).
        if let Ok(leaf) = obj.cast::<PyAtomistic>() {
            return Ok(Self::Atomistic(leaf.clone()));
        }
        if let Ok(leaf) = obj.cast::<PyCoarseGrain>() {
            return Ok(Self::CoarseGrain(leaf.clone()));
        }
        if let Ok(graph) = obj.cast::<PyMolGraph>() {
            return Ok(Self::MolGraph(graph.clone()));
        }
        Err(PyTypeError::new_err(format!(
            "expected a graph (MolGraph, Atomistic, CoarseGrain), not {}",
            obj.get_type().name()?
        )))
    }

    /// A copy of the graph this object holds, as a bare [`MolGraph`]. Graph
    /// types are peers: this reads the object's own graph, it converts
    /// nothing.
    pub(crate) fn to_molgraph(&self) -> PyResult<MolGraph> {
        Ok(match self {
            Self::MolGraph(graph) => graph.try_borrow()?.inner.clone(),
            Self::Atomistic(leaf) => leaf.try_borrow()?.mol().clone(),
            Self::CoarseGrain(leaf) => leaf.try_borrow()?.mol().clone(),
        })
    }
}

/// The graph class a graph-producing API is asked to build.
pub(crate) enum GraphClass {
    MolGraph,
    Atomistic,
    CoarseGrain,
}

impl GraphClass {
    /// `cls` as a graph class; `None` means `MolGraph`.
    ///
    /// # Errors
    ///
    /// `TypeError` when `cls` is not `MolGraph`, `Atomistic` or `CoarseGrain`.
    pub(crate) fn of(py: Python<'_>, cls: Option<&Bound<'_, PyType>>) -> PyResult<Self> {
        let Some(cls) = cls else {
            return Ok(Self::MolGraph);
        };
        if cls.is(py.get_type::<PyAtomistic>()) {
            Ok(Self::Atomistic)
        } else if cls.is(py.get_type::<PyCoarseGrain>()) {
            Ok(Self::CoarseGrain)
        } else if cls.is(py.get_type::<PyMolGraph>()) {
            Ok(Self::MolGraph)
        } else {
            Err(PyTypeError::new_err(format!(
                "cls must be a graph class (MolGraph, Atomistic, CoarseGrain), not {}",
                cls.name()?
            )))
        }
    }

    /// `graph` as an instance of this class.
    ///
    /// # Errors
    ///
    /// `ValueError` when `graph` breaks the class's invariant (an `Atomistic`
    /// node without `element`).
    pub(crate) fn build(self, py: Python<'_>, graph: MolGraph) -> PyResult<Py<PyAny>> {
        Ok(match self {
            Self::MolGraph => Py::new(py, PyMolGraph { inner: graph })?.into_any(),
            Self::Atomistic => {
                let leaf = Atomistic::try_from_molgraph(graph).map_err(molrs_error_to_pyerr)?;
                PyAtomistic::from_core(py, leaf)?.into_any()
            }
            Self::CoarseGrain => {
                let leaf = CoarseGrain::try_from_molgraph(graph).map_err(molrs_error_to_pyerr)?;
                PyCoarseGrain::from_core(py, leaf)?.into_any()
            }
        })
    }
}

/// The keyword arguments of a node factory as one dict: `mapping` (any
/// mapping, or pairs) updated by `attrs`.
fn node_fields<'py>(
    py: Python<'py>,
    mapping: Option<&Bound<'py, PyAny>>,
    attrs: Option<&Bound<'py, PyDict>>,
) -> PyResult<Bound<'py, PyDict>> {
    let fields = PyDict::new(py);
    if let Some(mapping) = mapping {
        fields.call_method1(pyo3::intern!(py, "update"), (mapping,))?;
    }
    if let Some(attrs) = attrs {
        fields.update(attrs.as_mapping())?;
    }
    Ok(fields)
}

/// The members of a graph with views: `props`, `links` and `remove_link`.
macro_rules! leaf_views_impl {
    ($ty:ty) => {
        #[pymethods]
        impl $ty {
            /// Whole-graph annotations (a name, a provenance tag): the keyword
            /// arguments the graph was built with, as a live dict. A copy, a
            /// pickle and every graph derived from this one (perception,
            /// typing, conformers, subgraphs) carry them.
            #[getter]
            fn props(&self, py: Python<'_>) -> Py<PyDict> {
                self.props.clone_ref(py)
            }

            /// The graph's relations, selected by view class
            /// (``graph.links.exact_bucket(Bond)``).
            #[getter]
            fn links(slf: &Bound<'_, Self>) -> PyRelationBuckets {
                PyRelationBuckets::new(slf.as_any())
            }

            /// Remove each of ``links`` (relation views of this graph).
            ///
            /// Raises
            /// ------
            /// ValueError
            ///     If a relation belongs to another graph.
            #[pyo3(signature = (*links))]
            fn remove_link(slf: &Bound<'_, Self>, links: &Bound<'_, PyTuple>) -> PyResult<()> {
                Leaf::of(slf.as_any())?.remove_links(links)
            }
        }
    };
}

// ---------------------------------------------------------------------------
// PyAtomistic — all-atom leaf (holds a core Atomistic)
// ---------------------------------------------------------------------------

/// All-atom molecular graph, exposed to Python as `molrs.core.Atomistic`.
///
/// Holds a core [`Atomistic`] from construction; it is never converted from a
/// `MolGraph`. Subclasses `MolGraph`; the generic API operates on this leaf's own
/// graph. ``Atomistic(**props)``: the keywords are the graph's :attr:`props`.
#[pyclass(module = "molrs.core", name = "Atomistic", extends = PyMolGraph, subclass)]
pub struct PyAtomistic {
    inner: Atomistic,
    props: Py<PyDict>,
    pub(super) views: ViewCache,
}

impl PyAtomistic {
    pub(crate) fn mol(&self) -> &MolGraph {
        self.inner.as_molgraph()
    }
    pub(crate) fn mol_mut(&mut self) -> &mut MolGraph {
        self.inner.as_molgraph_mut()
    }
    fn forget_views(&mut self) {
        self.views.clear();
    }
}

#[pymethods]
impl PyAtomistic {
    #[new]
    #[pyo3(signature = (**props))]
    fn new(py: Python<'_>, props: Option<Bound<'_, PyDict>>) -> (Self, PyMolGraph) {
        let leaf = Self {
            inner: Atomistic::new(),
            props: props.unwrap_or_else(|| PyDict::new(py)).unbind(),
            views: ViewCache::default(),
        };
        (leaf, PyMolGraph::base())
    }

    // ---- live views ----

    /// Every atom, in row order.
    #[getter]
    fn atoms(slf: &Bound<'_, Self>) -> PyResult<PyRefs> {
        PyRefs::nodes_of(slf.as_any())
    }

    /// Every bond, in row order.
    #[getter]
    fn bonds(slf: &Bound<'_, Self>) -> PyResult<PyRefs> {
        PyRefs::relations_of(slf.as_any(), "bonds")
    }

    /// Every angle, in row order.
    #[getter]
    fn angles(slf: &Bound<'_, Self>) -> PyResult<PyRefs> {
        PyRefs::relations_of(slf.as_any(), "angles")
    }

    /// Every proper dihedral, in row order.
    #[getter]
    fn dihedrals(slf: &Bound<'_, Self>) -> PyResult<PyRefs> {
        PyRefs::relations_of(slf.as_any(), "dihedrals")
    }

    /// Every improper, in row order.
    #[getter]
    fn impropers(slf: &Bound<'_, Self>) -> PyResult<PyRefs> {
        PyRefs::relations_of(slf.as_any(), "impropers")
    }

    /// Every port, in row order (empty when the graph carries none).
    #[getter]
    fn ports(slf: &Bound<'_, Self>) -> PyResult<PyRefs> {
        PyRefs::relations_of(slf.as_any(), "ports")
    }

    /// Add an atom carrying the fields of `mapping` and `attrs` (a ``None``
    /// value is skipped) and return its view.
    ///
    /// Raises
    /// ------
    /// TypeError
    ///     If a value is not a bool, int, float or str; the graph is then
    ///     unchanged.
    #[pyo3(signature = (mapping = None, /, **attrs))]
    fn def_atom<'py>(
        slf: &Bound<'py, Self>,
        mapping: Option<&Bound<'py, PyAny>>,
        attrs: Option<&Bound<'py, PyDict>>,
    ) -> PyResult<Bound<'py, PyAny>> {
        Leaf::of(slf.as_any())?.create_node(&node_fields(slf.py(), mapping, attrs)?)
    }

    /// Add a virtual site — an atom with a ``vsite`` field naming its kind —
    /// and return its view, of class `kind` (``VirtualSite`` by default,
    /// ``DrudeParticle`` or ``MasslessSite``). An explicit ``vsite`` in
    /// `attrs` wins over the one `kind` implies.
    ///
    /// Raises
    /// ------
    /// TypeError
    ///     If `kind` is not one of the three virtual-site classes.
    #[pyo3(signature = (mapping = None, /, *, kind = None, **attrs))]
    fn def_virtual_site<'py>(
        slf: &Bound<'py, Self>,
        mapping: Option<&Bound<'py, PyAny>>,
        kind: Option<&Bound<'py, PyType>>,
        attrs: Option<&Bound<'py, PyDict>>,
    ) -> PyResult<Bound<'py, PyAny>> {
        let fields = node_fields(slf.py(), mapping, attrs)?;
        Leaf::of(slf.as_any())?.create_virtual_site(&fields, kind)
    }

    /// Add a single bond between two atoms of this graph and return its view.
    ///
    /// Routes through the native writer, so the bond carries both bond facts
    /// — ``bond_type = 1`` and ``bond_number = 1`` — before ``attrs`` are
    /// applied.
    ///
    /// Raises
    /// ------
    /// ValueError
    ///     If an endpoint belongs to another graph, or an attr is refused.
    #[pyo3(signature = (a, b, /, **attrs))]
    fn def_bond<'py>(
        slf: &Bound<'py, Self>,
        a: &Bound<'py, PyNodeRef>,
        b: &Bound<'py, PyNodeRef>,
        attrs: Option<&Bound<'py, PyDict>>,
    ) -> PyResult<Bound<'py, PyAny>> {
        let leaf = Leaf::of(slf.as_any())?;
        let (a, b) = (leaf.own_node(a)?, leaf.own_node(b)?);
        let handle = slf
            .try_borrow_mut()?
            .inner
            .add_bond(node_from_u64(a), node_from_u64(b))
            .map_err(molrs_error_to_pyerr)?;
        leaf.adopt_relation("bonds", relation_to_u64(handle), attrs)
    }

    /// Add an angle ``a–b–c`` (``b`` the vertex) and return its view.
    ///
    /// Raises
    /// ------
    /// ValueError
    ///     If an endpoint belongs to another graph, or an attr is refused.
    #[pyo3(signature = (a, b, c, /, **attrs))]
    fn def_angle<'py>(
        slf: &Bound<'py, Self>,
        a: &Bound<'py, PyNodeRef>,
        b: &Bound<'py, PyNodeRef>,
        c: &Bound<'py, PyNodeRef>,
        attrs: Option<&Bound<'py, PyDict>>,
    ) -> PyResult<Bound<'py, PyAny>> {
        let leaf = Leaf::of(slf.as_any())?;
        let ends = [leaf.own_node(a)?, leaf.own_node(b)?, leaf.own_node(c)?].map(node_from_u64);
        let handle = slf
            .try_borrow_mut()?
            .inner
            .add_angle(ends[0], ends[1], ends[2])
            .map_err(molrs_error_to_pyerr)?;
        leaf.adopt_relation("angles", relation_to_u64(handle), attrs)
    }

    /// Add a proper dihedral ``a–b–c–d`` and return its view.
    ///
    /// Raises
    /// ------
    /// ValueError
    ///     If an endpoint belongs to another graph, or an attr is refused.
    #[pyo3(signature = (a, b, c, d, /, **attrs))]
    fn def_dihedral<'py>(
        slf: &Bound<'py, Self>,
        a: &Bound<'py, PyNodeRef>,
        b: &Bound<'py, PyNodeRef>,
        c: &Bound<'py, PyNodeRef>,
        d: &Bound<'py, PyNodeRef>,
        attrs: Option<&Bound<'py, PyDict>>,
    ) -> PyResult<Bound<'py, PyAny>> {
        let leaf = Leaf::of(slf.as_any())?;
        let [a, b, c, d] = [a, b, c, d].map(|end| leaf.own_node(end).map(node_from_u64));
        let handle = slf
            .try_borrow_mut()?
            .inner
            .add_dihedral(a?, b?, c?, d?)
            .map_err(molrs_error_to_pyerr)?;
        leaf.adopt_relation("dihedrals", relation_to_u64(handle), attrs)
    }

    /// Add an improper over ``a, b, c, d`` in its style's slot order and
    /// return its view.
    ///
    /// Raises
    /// ------
    /// ValueError
    ///     If an endpoint belongs to another graph, or an attr is refused.
    #[pyo3(signature = (a, b, c, d, /, **attrs))]
    fn def_improper<'py>(
        slf: &Bound<'py, Self>,
        a: &Bound<'py, PyNodeRef>,
        b: &Bound<'py, PyNodeRef>,
        c: &Bound<'py, PyNodeRef>,
        d: &Bound<'py, PyNodeRef>,
        attrs: Option<&Bound<'py, PyDict>>,
    ) -> PyResult<Bound<'py, PyAny>> {
        let leaf = Leaf::of(slf.as_any())?;
        let [a, b, c, d] = [a, b, c, d].map(|end| leaf.own_node(end).map(node_from_u64));
        let handle = slf
            .try_borrow_mut()?
            .inner
            .add_improper(a?, b?, c?, d?)
            .map_err(molrs_error_to_pyerr)?;
        leaf.adopt_relation("impropers", relation_to_u64(handle), attrs)
    }

    /// Remove each of `atoms` and every relation touching it. Their views go
    /// stale: a read raises.
    ///
    /// Raises
    /// ------
    /// ValueError
    ///     If an atom belongs to another graph or was removed already.
    #[pyo3(signature = (*atoms))]
    fn del_atom(slf: &Bound<'_, Self>, atoms: &Bound<'_, PyTuple>) -> PyResult<()> {
        Leaf::of(slf.as_any())?.remove_nodes(atoms)
    }

    /// Record a bonding descriptor on the ``(anchor, handle_atom)`` valence
    /// and return the port's view.
    ///
    /// Routes through the native writer, which is where the validation
    /// lives: the anchor–handle bond check, the one-port-per-valence check,
    /// the glyph parse and the order check.
    ///
    /// Parameters
    /// ----------
    /// anchor : Atom
    ///     The atom that keeps its place in the product molecule.
    /// handle_atom : Atom
    ///     The atom bonded to `anchor` that roots the leaving group.
    /// kind : str
    ///     The notation glyph -- one of ``"$"``, ``"<"``, ``">"``, ``"!"``.
    /// label : str, optional
    ///     Descriptor label; ``""`` (the default) means unnamed.
    /// order : int, optional
    ///     Multiplicity of the bond this port will form, ``1`` to ``4``.
    ///
    /// Raises
    /// ------
    /// ValueError
    ///     If an endpoint belongs to another graph, `kind` is not one of the
    ///     four glyphs, `handle_atom` is not bonded to `anchor`, that valence
    ///     already carries a port, or `order` is not a definite bond number.
    #[pyo3(signature = (anchor, handle_atom, kind, label = "", order = 1))]
    fn def_port<'py>(
        slf: &Bound<'py, Self>,
        anchor: &Bound<'py, PyNodeRef>,
        handle_atom: &Bound<'py, PyNodeRef>,
        kind: &str,
        label: &str,
        order: u32,
    ) -> PyResult<Bound<'py, PyAny>> {
        let leaf = Leaf::of(slf.as_any())?;
        let (anchor, handle_atom) = (leaf.own_node(anchor)?, leaf.own_node(handle_atom)?);
        let kind = PortKind::from_str(kind).map_err(molrs_error_to_pyerr)?;
        let handle = slf
            .try_borrow_mut()?
            .mol_mut()
            .add_port(
                node_from_u64(anchor),
                node_from_u64(handle_atom),
                kind,
                label,
                BondNumber::from_code(order),
            )
            .map_err(molrs_error_to_pyerr)?;
        leaf.relation("ports", relation_to_u64(handle))
    }

    // ---- pickling ----

    fn __reduce__<'py>(slf: &Bound<'py, Self>) -> PyResult<Bound<'py, PyTuple>> {
        let py = slf.py();
        let this = slf.try_borrow()?;
        let state =
            (this.props.bind(py).clone(), this.mol().dump_content(py)?).into_pyobject(py)?;
        crate::pickle::reduce_with_state(slf.as_any(), PyTuple::empty(py), state.into_any())
    }

    fn __setstate__(&mut self, state: (Bound<'_, PyDict>, Bound<'_, PyAny>)) -> PyResult<()> {
        let (props, content) = state;
        let mut inner = Atomistic::new();
        inner.as_molgraph_mut().load_content(&content)?;
        self.inner = inner;
        self.props = props.unbind();
        self.forget_views();
        Ok(())
    }

    fn __traverse__(&self, visit: PyVisit<'_>) -> Result<(), PyTraverseError> {
        visit.call(&self.props)
    }

    fn __clear__(&mut self) {
        self.props = Python::attach(|py| PyDict::new(py).unbind());
    }

    // ---- domain builders (operate on the core Atomistic directly) ----

    /// Add an atom with element `symbol` and optional coordinates. Returns its
    /// stable handle.
    #[pyo3(signature = (symbol, x=None, y=None, z=None))]
    fn add_atom(&mut self, symbol: &str, x: Option<f64>, y: Option<f64>, z: Option<f64>) -> u64 {
        let id = match (x, y, z) {
            (Some(x), Some(y), Some(z)) => self.inner.add_atom_xyz(symbol, x, y, z),
            _ => self.inner.add_atom_bare(symbol),
        };
        node_to_u64(id)
    }

    /// Add a bond between two atom handles (default order 1.0). Returns its handle.
    fn add_bond(&mut self, a: u64, b: u64) -> PyResult<u64> {
        self.inner
            .add_bond(node_from_u64(a), node_from_u64(b))
            .map(relation_to_u64)
            .map_err(molrs_error_to_pyerr)
    }

    /// Add an angle over three atom handles (`j` central).
    fn add_angle(&mut self, i: u64, j: u64, k: u64) -> PyResult<u64> {
        self.inner
            .add_angle(node_from_u64(i), node_from_u64(j), node_from_u64(k))
            .map(relation_to_u64)
            .map_err(molrs_error_to_pyerr)
    }

    /// Add a dihedral over four atom handles.
    fn add_dihedral(&mut self, i: u64, j: u64, k: u64, l: u64) -> PyResult<u64> {
        self.inner
            .add_dihedral(
                node_from_u64(i),
                node_from_u64(j),
                node_from_u64(k),
                node_from_u64(l),
            )
            .map(relation_to_u64)
            .map_err(molrs_error_to_pyerr)
    }

    /// Add an improper over four atom handles.
    fn add_improper(&mut self, i: u64, j: u64, k: u64, l: u64) -> PyResult<u64> {
        self.inner
            .add_improper(
                node_from_u64(i),
                node_from_u64(j),
                node_from_u64(k),
                node_from_u64(l),
            )
            .map(relation_to_u64)
            .map_err(molrs_error_to_pyerr)
    }

    /// Perceive angle, dihedral and improper relations from the bond graph.
    ///
    /// Angles are 2-edge paths ``i-j-k`` and proper dihedrals 3-edge paths
    /// ``i-j-k-l`` over the bonds (graph-theory via the native `Topology`-backed
    /// ``Topology``). Impropers are the molecular-mechanics reading: one
    /// ``[centre, i, j, k]`` quartet per atom with **exactly three** neighbours,
    /// centre first, peripherals sorted — not every 3-combination at every
    /// centre of degree >= 3, which would hand an sp3 carbon four quartets.
    /// Whether a trivalent centre is planar enough to carry the term is
    /// force-field data, not a graph property, so every one is emitted.
    ///
    /// Idempotent; ``clear_existing`` wipes existing relations of the requested
    /// kinds first. Returns ``(n_angles_added, n_dihedrals_added,
    /// n_impropers_added)``.
    #[pyo3(signature = (gen_angle=true, gen_dihedral=true, gen_improper=false, clear_existing=false))]
    fn generate_topology(
        &mut self,
        gen_angle: bool,
        gen_dihedral: bool,
        gen_improper: bool,
        clear_existing: bool,
    ) -> PyResult<(usize, usize, usize)> {
        self.inner
            .generate_topology(gen_angle, gen_dihedral, gen_improper, clear_existing)
            .map_err(molrs_error_to_pyerr)
    }

    /// Single-source shortest-path (BFS) distances over the bond graph from
    /// `source` (a node handle), as `(node_handle, hops)` pairs for every atom
    /// reachable from `source` (including `source` itself at distance 0).
    /// Unreachable atoms (a different connected component) are omitted; an
    /// unknown `source` handle yields an empty list.
    #[pyo3(signature = (source, max_hops = None))]
    fn topo_distances(&self, source: u64, max_hops: Option<i64>) -> Vec<(u64, i64)> {
        self.inner
            .topo_distances(node_from_u64(source), max_hops)
            .into_iter()
            .map(|(a, d)| (node_to_u64(a), d))
            .collect()
    }

    /// Number of atoms.
    #[getter]
    fn n_atoms(&self) -> usize {
        self.inner.n_atoms()
    }

    /// Number of bonds.
    #[getter]
    fn n_bonds(&self) -> usize {
        self.inner.n_bonds()
    }

    /// Export to a tabular [`Frame`] (atoms / bonds / angles / dihedrals /
    /// impropers blocks). Leaf-owned — `self.inner.to_frame()`, zero conversion.
    ///
    /// Raises
    /// ------
    /// ValueError
    ///     If an atom or relation property contradicts the dtype the Frame
    ///     schema declares for its key (a string under ``"x"``).
    #[pyo3(signature = (atom_fields = None))]
    fn to_frame(&self, atom_fields: Option<Vec<String>>) -> PyResult<PyFrame> {
        let frame = self.inner.to_frame().map_err(molrs_error_to_pyerr)?;
        PyFrame::from_core_frame(select_atom_fields(frame, atom_fields)?)
    }

    /// Build an `Atomistic` from a [`Frame`] (registers the chemistry kinds,
    /// then reads the relation blocks). A leaf constructor, not a conversion.
    #[staticmethod]
    fn from_frame(py: Python<'_>, frame: &PyFrame) -> PyResult<Py<PyAtomistic>> {
        let core = frame.clone_core_frame()?;
        let inner = Atomistic::from_frame(&core).map_err(molrs_error_to_pyerr)?;
        PyAtomistic::from_core(py, inner)
    }

    // ---- graph-edit conveniences (forward to core Atomistic fns) ----

    /// Remove an atom by handle, cascading incident bonds / angles / dihedrals /
    /// impropers. Errors if the handle is stale.
    fn remove_atom(&mut self, handle: u64) -> PyResult<()> {
        self.inner
            .remove_atom(node_from_u64(handle))
            .map(|_| ())
            .map_err(molrs_error_to_pyerr)
    }

    /// Remove a bond by handle (a relation handle from :meth:`add_bond`).
    fn remove_bond(&mut self, handle: u64) -> PyResult<()> {
        self.inner
            .remove_bond(relation_from_u64(handle))
            .map(|_| ())
            .map_err(molrs_error_to_pyerr)
    }

    /// Set a bond's chemical class and its localized bond number.
    ///
    /// The two are set together because they are only meaningful together: a
    /// class without a number leaves the bond un-standardized, and a number
    /// without a class leaves a consumer no way to tell aromatic from double.
    ///
    /// Parameters
    /// ----------
    /// handle : int
    ///     A relation handle from :meth:`add_bond`.
    /// bond_type : int
    ///     0 unknown, 1 single, 2 double, 3 triple, 4 aromatic.
    /// bond_number : int
    ///     0 unknown, 1 single, 2 double, 3 triple, 4 quadruple.
    fn set_bond_class(&mut self, handle: u64, bond_type: u32, bond_number: u32) -> PyResult<()> {
        self.inner
            .set_bond_class(
                relation_from_u64(handle),
                BondOrder::from_code(bond_type),
                BondNumber::from_code(bond_number),
            )
            .map_err(molrs_error_to_pyerr)
    }

    /// Set a plain (non-aromatic) bond, whose class implies its number.
    ///
    /// ``Aromatic`` (4) implies no number — the phase is a Kekulé assignment's
    /// to decide — so it goes through :meth:`set_bond_class`.
    fn set_bond_type(&mut self, handle: u64, bond_type: u32) -> PyResult<()> {
        self.inner
            .set_bond_type(relation_from_u64(handle), BondOrder::from_code(bond_type))
            .map_err(molrs_error_to_pyerr)
    }

    /// The bond's chemical class code; ``0`` when it has none.
    fn bond_type(&self, handle: u64) -> u32 {
        self.inner.bond_type(relation_from_u64(handle)).code()
    }

    /// The bond's localized (Kekulé) bond number; ``0`` when it has none.
    fn bond_number(&self, handle: u64) -> u32 {
        self.inner.bond_number(relation_from_u64(handle)).code()
    }

    /// Return an independent deep copy of this `Atomistic`, with a copy of
    /// its :attr:`props`.
    ///
    /// **Handles are preserved** (same generational keys as in ``self``).
    fn copy(&self, py: Python<'_>) -> PyResult<Py<PyAtomistic>> {
        self.derive(py, self.inner.clone())
    }

    /// Structural merge of ``other`` into ``self``.
    ///
    /// Consumes ``other``'s storage (``other`` is left empty). Every node handle
    /// from ``other`` is remapped; returns ``{old_handle: new_handle}``.
    ///
    /// Raises
    /// ------
    /// ValueError
    ///     If a property of ``other`` contradicts the type this graph holds
    ///     for that key (a string ``tag`` into an int ``tag`` column).
    fn merge(&mut self, other: &mut Self) -> PyResult<HashMap<u64, u64>> {
        let taken = std::mem::take(&mut other.inner);
        other.forget_views();
        Ok(self
            .inner
            .merge(taken)
            .map_err(molrs_error_to_pyerr)?
            .into_iter()
            .map(|(k, v)| (node_to_u64(k), node_to_u64(v)))
            .collect())
    }

    /// Induced subgraph on an explicit list of atom handles.
    ///
    /// Returns ``(subgraph, {parent_handle: new_handle})``. Stale handles raise.
    fn induced_subgraph(
        &self,
        py: Python<'_>,
        nodes: Vec<u64>,
    ) -> PyResult<(Py<PyAtomistic>, HashMap<u64, u64>)> {
        let ids: Vec<_> = nodes.into_iter().map(node_from_u64).collect();
        let (sub, map) = self
            .inner
            .induced_subgraph(&ids)
            .map_err(molrs_error_to_pyerr)?;
        let py_sub = self.derive(py, sub)?;
        let py_map = map
            .into_iter()
            .map(|(k, v)| (node_to_u64(k), node_to_u64(v)))
            .collect();
        Ok((py_sub, py_map))
    }

    /// Radius ball around ``centers`` over the bond graph.
    ///
    /// When ``regenerate_topology`` is true, only bonds are copied and
    /// angles/dihedrals are perceived on the ball. When false, higher-order
    /// terms fully inside the ball are copied from the parent.
    ///
    /// ``max_ring_size`` closes the ball on ring systems: a ring of at most
    /// that many atoms which the radius only partly reaches comes along whole,
    /// together with everything fused or bridged to it. A cut ring is not a
    /// smaller molecule — ring perception, and with it aromaticity and every
    /// ring-aware atom type, reads the remainder as something else. The added
    /// atoms lie beyond ``radius``, so they arrive as context: their ``hops``
    /// are larger than the radius asked for.
    ///
    /// The bound is not a tuning knob, it is the claim's domain. Cutting a
    /// benzene changes the chemistry; cutting a 5000-membered macrocycle
    /// changes nothing any typifier can see, while closing on it would drag the
    /// whole loop into a ball that asked for nine atoms. Pass the largest ring
    /// your typifier can actually distinguish. ``None`` disables closure; the
    /// cost then, and with a bound, follows the ball and its neighbourhood —
    /// never the size of the parent graph.
    #[pyo3(signature = (centers, radius, *, regenerate_topology=false, max_ring_size=None))]
    fn extract_subgraph(
        &self,
        py: Python<'_>,
        centers: Vec<u64>,
        radius: i64,
        regenerate_topology: bool,
        max_ring_size: Option<usize>,
    ) -> PyResult<PyExtractedSubgraph> {
        let ids: Vec<_> = centers.into_iter().map(node_from_u64).collect();
        let groups = match max_ring_size {
            Some(max_ring_size) => {
                molrs::perceive::small_ring_closure(&self.inner, &ids, radius, max_ring_size)
            }
            None => Vec::new(),
        };
        let ext = self
            .inner
            .extract_subgraph(&ids, radius, regenerate_topology, &groups)
            .map_err(molrs_error_to_pyerr)?;
        PyExtractedSubgraph::from_atomistic(py, self, ext)
    }

    // ---- structural graph hash (WL) ----

    /// Isomorphism-invariant Weisfeiler–Lehman structural hash (``int``).
    ///
    /// A stable, reproducible dedup key over the molecular graph
    /// (element / charge / aromatic node labels, bond-order edge labels):
    /// identical for a node-permuted copy, sensitive to any label or
    /// connectivity change. Reproducible across runs and processes.
    fn structural_hash(&self) -> u64 {
        self.inner.structural_hash()
    }

    /// Deterministic canonical atom ordering (a list of atom handles) from the
    /// WL refinement, so two isomorphic molecules line up node-by-node.
    fn canonical_order(&self) -> Vec<u64> {
        self.inner
            .canonical_order()
            .into_iter()
            .map(node_to_u64)
            .collect()
    }

    /// Whether `self` and `other` are isomorphic as labeled molecular graphs.
    fn is_isomorphic(&self, other: &PyAtomistic) -> bool {
        self.inner.is_isomorphic(other.core())
    }

    /// Mass-weighted centre of every atom.
    ///
    /// ``R = sum_i m_i r_i / sum_i m_i`` over all atoms, reading positions
    /// from ``x`` / ``y`` / ``z`` (Å) and masses from ``mass`` (g/mol). No
    /// periodic imaging: a molecule split across a box face must be unwrapped
    /// first (:meth:`Box.unwrap`).
    ///
    /// Returns
    /// -------
    /// numpy.ndarray, shape (3,), float64
    ///     The centre in Å.
    ///
    /// Raises
    /// ------
    /// ValueError
    ///     If the molecule has no atoms; if an atom lacks a finite ``x`` /
    ///     ``y`` / ``z`` or a finite, non-negative ``mass`` (the message names
    ///     its int handle); or if the total mass is not positive.
    fn center<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyArray1<f64>>> {
        let atoms: Vec<NodeId> = self.inner.node_ids().collect();
        let center = self
            .inner
            .as_molgraph()
            .center(&atoms)
            .map_err(center_error_to_pyerr)?;
        Ok(vector_to_py(py, &center))
    }
}

graph_world_impl!(PyAtomistic);

leaf_views_impl!(PyAtomistic);

impl PyAtomistic {
    /// A new `Atomistic` holding `inner`, with no props: the graph-out door
    /// of every API that builds a graph from something that is not one
    /// (a reader, a SMILES string, a frame).
    pub(crate) fn from_core(py: Python<'_>, inner: Atomistic) -> PyResult<Py<PyAtomistic>> {
        let leaf = Self {
            inner,
            props: PyDict::new(py).unbind(),
            views: ViewCache::default(),
        };
        Py::new(py, (leaf, PyMolGraph::base()))
    }

    /// A new `Atomistic` holding `inner`, derived from this one: it carries
    /// a copy of this graph's :attr:`props`. The graph-out door of every API
    /// that maps a graph to a graph (copy, perception, typing, conformers,
    /// subgraphs).
    pub(crate) fn derive(&self, py: Python<'_>, inner: Atomistic) -> PyResult<Py<PyAtomistic>> {
        let leaf = Self {
            inner,
            props: self.props.bind(py).copy()?.unbind(),
            views: ViewCache::default(),
        };
        Py::new(py, (leaf, PyMolGraph::base()))
    }

    /// Borrow the held core [`Atomistic`] (for domain consumers like the
    /// conformer / force-field typifier that operate on atomistic chemistry).
    pub(crate) fn core(&self) -> &Atomistic {
        &self.inner
    }

    /// Mutably borrow the held core [`Atomistic`] (for in-place chemistry
    /// systems like `perceive_aromaticity` / `compute_gasteiger_charges`).
    pub(crate) fn core_mut(&mut self) -> &mut Atomistic {
        &mut self.inner
    }
}

// ---------------------------------------------------------------------------
// PyCoarseGrain — coarse-grained leaf (holds a core CoarseGrain)
// ---------------------------------------------------------------------------

/// Coarse-grained molecular graph, exposed to Python as `molrs.core.CoarseGrain`.
///
/// ``CoarseGrain(**props)``: the keywords are the graph's :attr:`props`. A
/// bead built with ``def_bead(atoms=...)`` groups atom views of one source
/// graph (its *member world*); ``bead["atoms"]`` answers with those views.
#[pyclass(module = "molrs.core", name = "CoarseGrain", extends = PyMolGraph, subclass)]
pub struct PyCoarseGrain {
    inner: CoarseGrain,
    props: Py<PyDict>,
    pub(super) views: ViewCache,
    /// The graph whose atoms the bead memberships name, once one is given.
    member_world: Option<Py<PyAny>>,
}

impl PyCoarseGrain {
    pub(crate) fn mol(&self) -> &MolGraph {
        self.inner.as_molgraph()
    }
    pub(crate) fn mol_mut(&mut self) -> &mut MolGraph {
        self.inner.as_molgraph_mut()
    }
    fn forget_views(&mut self) {
        self.views.clear();
    }

    /// The atom views bead `bead` groups, or `None` when it groups none or
    /// no member world was given (raw ``set_bead_members`` handles).
    pub(crate) fn bead_atoms<'py>(
        graph: &Bound<'py, Self>,
        bead: u64,
    ) -> PyResult<Option<Bound<'py, PyTuple>>> {
        let py = graph.py();
        let (members, world) = {
            let this = graph.try_borrow()?;
            let members = this.inner.bead_members(node_from_u64(bead)).to_vec();
            (members, this.member_world.as_ref().map(|w| w.clone_ref(py)))
        };
        let Some(world) = world.filter(|_| !members.is_empty()) else {
            return Ok(None);
        };
        let leaf = Leaf::of(world.bind(py))?;
        let atoms = members
            .into_iter()
            .map(|atom| leaf.node(atom))
            .collect::<PyResult<Vec<_>>>()?;
        Ok(Some(PyTuple::new(py, atoms)?))
    }

    /// The handles of the atom views in `atoms`, which must all come from one
    /// graph — the member world, which the first membership fixes.
    fn member_handles(slf: &Bound<'_, Self>, atoms: &Bound<'_, PyAny>) -> PyResult<Vec<u64>> {
        let py = slf.py();
        let mut handles = Vec::new();
        for atom in atoms.try_iter()? {
            let atom = atom?;
            let atom = atom.cast::<PyNodeRef>()?.get();
            let mut this = slf.try_borrow_mut()?;
            match &this.member_world {
                None => this.member_world = Some(atom.world(py).clone().unbind()),
                Some(world) if world.bind(py).is(atom.world(py)) => {}
                Some(_) => {
                    return Err(PyValueError::new_err(
                        "bead membership atoms must all come from the same source world",
                    ));
                }
            }
            handles.push(atom.handle());
        }
        Ok(handles)
    }
}

#[pymethods]
impl PyCoarseGrain {
    #[new]
    #[pyo3(signature = (**props))]
    fn new(py: Python<'_>, props: Option<Bound<'_, PyDict>>) -> (Self, PyMolGraph) {
        let leaf = Self {
            inner: CoarseGrain::new(),
            props: props.unwrap_or_else(|| PyDict::new(py)).unbind(),
            views: ViewCache::default(),
            member_world: None,
        };
        (leaf, PyMolGraph::base())
    }

    // ---- live views ----

    /// Every bead, in row order.
    #[getter]
    fn beads(slf: &Bound<'_, Self>) -> PyResult<PyRefs> {
        PyRefs::nodes_of(slf.as_any())
    }

    /// Every CG bond, in row order.
    #[getter]
    fn cgbonds(slf: &Bound<'_, Self>) -> PyResult<PyRefs> {
        PyRefs::relations_of(slf.as_any(), "bonds")
    }

    /// Add a bead carrying the fields of `mapping` and `attrs` (a ``None``
    /// value is skipped) and return its view. An ``atoms`` field is the
    /// bead's membership: atom views of one source graph.
    ///
    /// Raises
    /// ------
    /// ValueError
    ///     If the member atoms do not all come from the graph earlier
    ///     memberships came from.
    /// TypeError
    ///     If a value is not a bool, int, float or str.
    #[pyo3(signature = (mapping = None, /, **attrs))]
    fn def_bead<'py>(
        slf: &Bound<'py, Self>,
        mapping: Option<&Bound<'py, PyAny>>,
        attrs: Option<&Bound<'py, PyDict>>,
    ) -> PyResult<Bound<'py, PyAny>> {
        let fields = node_fields(slf.py(), mapping, attrs)?;
        let members = match fields.get_item(BEAD_ATOMS)? {
            Some(atoms) => {
                fields.del_item(BEAD_ATOMS)?;
                if atoms.is_none() {
                    None
                } else {
                    Some(Self::member_handles(slf, &atoms)?)
                }
            }
            None => None,
        };
        let bead = Leaf::of(slf.as_any())?.create_node(&fields)?;
        if let Some(members) = members {
            let handle = bead.cast::<PyNodeRef>()?.get().handle();
            slf.try_borrow_mut()?
                .inner
                .set_bead_members(node_from_u64(handle), members);
        }
        Ok(bead)
    }

    /// Add a CG bond between two beads of this graph and return its view.
    ///
    /// Raises
    /// ------
    /// ValueError
    ///     If an endpoint belongs to another graph, or an attr is refused.
    #[pyo3(signature = (a, b, /, **attrs))]
    fn def_cgbond<'py>(
        slf: &Bound<'py, Self>,
        a: &Bound<'py, PyNodeRef>,
        b: &Bound<'py, PyNodeRef>,
        attrs: Option<&Bound<'py, PyDict>>,
    ) -> PyResult<Bound<'py, PyAny>> {
        let leaf = Leaf::of(slf.as_any())?;
        let (a, b) = (leaf.own_node(a)?, leaf.own_node(b)?);
        let handle = slf
            .try_borrow_mut()?
            .inner
            .add_bond(node_from_u64(a), node_from_u64(b))
            .map_err(molrs_error_to_pyerr)?;
        leaf.adopt_relation("bonds", relation_to_u64(handle), attrs)
    }

    // ---- pickling ----

    fn __reduce__<'py>(slf: &Bound<'py, Self>) -> PyResult<Bound<'py, PyTuple>> {
        let py = slf.py();
        let this = slf.try_borrow()?;
        let world = this.member_world.as_ref().map(|w| w.bind(py).clone());
        // Memberships by bead row; each member by its row in the member
        // world, or as the raw handle when there is none.
        let member_rows = match &world {
            Some(world) => Some(Leaf::of(world)?.read(|mol| {
                mol.node_ids()
                    .enumerate()
                    .map(|(row, id)| (node_to_u64(id), row as u64))
                    .collect::<HashMap<u64, u64>>()
            })?),
            None => None,
        };
        let mut memberships: Vec<(usize, Vec<u64>)> = Vec::new();
        for (row, bead) in this.mol().node_ids().enumerate() {
            let members = this.inner.bead_members(bead);
            if members.is_empty() {
                continue;
            }
            let members = match &member_rows {
                Some(rows) => members
                    .iter()
                    .map(|atom| {
                        rows.get(atom).copied().ok_or_else(|| {
                            PyValueError::new_err(format!(
                                "bead member {atom} is not an atom of the member world"
                            ))
                        })
                    })
                    .collect::<PyResult<Vec<u64>>>()?,
                None => members.to_vec(),
            };
            memberships.push((row, members));
        }
        let state = (
            this.props.bind(py).clone(),
            this.mol().dump_content(py)?,
            world,
            memberships,
        )
            .into_pyobject(py)?;
        crate::pickle::reduce_with_state(slf.as_any(), PyTuple::empty(py), state.into_any())
    }

    #[allow(
        clippy::type_complexity,
        reason = "the pickled state: (props, content, member world, [(bead row, members)])"
    )]
    fn __setstate__(
        slf: &Bound<'_, Self>,
        state: (
            Bound<'_, PyDict>,
            Bound<'_, PyAny>,
            Option<Bound<'_, PyAny>>,
            Vec<(usize, Vec<u64>)>,
        ),
    ) -> PyResult<()> {
        let (props, content, world, memberships) = state;
        let world_handles: Option<Vec<u64>> = match &world {
            Some(world) => {
                Some(Leaf::of(world)?.read(|mol| mol.node_ids().map(node_to_u64).collect())?)
            }
            None => None,
        };
        let mut inner = CoarseGrain::new();
        inner.as_molgraph_mut().load_content(&content)?;
        let beads: Vec<NodeId> = inner.as_molgraph().node_ids().collect();
        for (row, members) in memberships {
            let bead = *beads
                .get(row)
                .ok_or_else(|| PyValueError::new_err(format!("no bead at row {row}")))?;
            let members = match &world_handles {
                Some(handles) => members
                    .iter()
                    .map(|&member| {
                        handles.get(member as usize).copied().ok_or_else(|| {
                            PyValueError::new_err(format!("no member-world atom at row {member}"))
                        })
                    })
                    .collect::<PyResult<Vec<u64>>>()?,
                None => members,
            };
            inner.set_bead_members(bead, members);
        }
        let mut this = slf.try_borrow_mut()?;
        this.inner = inner;
        this.props = props.unbind();
        this.member_world = world.map(Bound::unbind);
        this.forget_views();
        Ok(())
    }

    fn __traverse__(&self, visit: PyVisit<'_>) -> Result<(), PyTraverseError> {
        visit.call(&self.props)?;
        if let Some(world) = &self.member_world {
            visit.call(world)?;
        }
        Ok(())
    }

    fn __clear__(&mut self) {
        self.member_world = None;
        self.props = Python::attach(|py| PyDict::new(py).unbind());
    }

    /// Add a bead with `bead_type` and optional coordinates. Returns its handle.
    #[pyo3(signature = (bead_type, x=None, y=None, z=None))]
    fn add_bead(&mut self, bead_type: &str, x: Option<f64>, y: Option<f64>, z: Option<f64>) -> u64 {
        let id = match (x, y, z) {
            (Some(x), Some(y), Some(z)) => self.inner.add_bead(bead_type, x, y, z),
            _ => self.inner.add_bead_bare(bead_type),
        };
        node_to_u64(id)
    }

    /// Add a CG bond between two bead handles. Returns its handle.
    fn add_bond(&mut self, a: u64, b: u64) -> PyResult<u64> {
        self.inner
            .add_bond(node_from_u64(a), node_from_u64(b))
            .map(relation_to_u64)
            .map_err(molrs_error_to_pyerr)
    }

    /// Number of beads.
    #[getter]
    fn n_beads(&self) -> usize {
        self.inner.n_beads()
    }

    /// Record the atom handles a bead groups (its membership), replacing any
    /// previous set. An empty list clears the membership. Handles are opaque —
    /// they belong to the caller's source (all-atom) world.
    fn set_bead_members(&mut self, bead: u64, atoms: Vec<u64>) {
        self.inner.set_bead_members(node_from_u64(bead), atoms);
    }

    /// The atom handles a bead groups (empty if none recorded).
    fn bead_members(&self, bead: u64) -> Vec<u64> {
        self.inner.bead_members(node_from_u64(bead)).to_vec()
    }

    /// Bead handles whose membership includes `atom`, in bead-handle order.
    fn beads_of_atom(&self, atom: u64) -> Vec<u64> {
        self.inner
            .beads_of_atom(atom)
            .into_iter()
            .map(node_to_u64)
            .collect()
    }

    /// Export to a tabular :class:`~molrs.core.Frame` in the shared vocabulary.
    ///
    /// One ``atoms`` row per bead and one ``bonds`` row per CG bond
    /// (``atomi`` / ``atomj`` are the endpoint beads' ``atoms`` rows), plus a
    /// ``members`` block when any bead records membership: one row per
    /// bead–atom pair, ``ibead`` being the bead's row in ``atoms`` and
    /// ``atom`` the opaque atom handle. Coordinates stay in Å.
    ///
    /// Every bead property is written, ``mol_id`` included, so a CoarseGrain
    /// whose beads carry two ``mol_id`` values does not round-trip through
    /// :meth:`from_frame`; select one molecule with ``Frame.subset`` first.
    ///
    /// Returns
    /// -------
    /// Frame
    ///
    /// Raises
    /// ------
    /// ValueError
    ///     If a bead or bond property contradicts the dtype the Frame schema
    ///     declares for its key (a string under ``"x"``).
    #[pyo3(signature = (atom_fields = None))]
    fn to_frame(&self, atom_fields: Option<Vec<String>>) -> PyResult<PyFrame> {
        let frame = self.inner.to_frame().map_err(molrs_error_to_pyerr)?;
        PyFrame::from_core_frame(select_atom_fields(frame, atom_fields)?)
    }

    /// Build a `CoarseGrain` from a frame holding **one molecule**.
    ///
    /// Every ``atoms`` row becomes a bead and every ``bonds`` row a CG bond
    /// (``atomi`` / ``atomj`` are 0-based ``atoms`` rows); a ``members``
    /// block, when present, is read back as bead membership. ``bead_type``
    /// comes from the first column present: ``bead_type``, then ``type``,
    /// then ``type_id`` (rendered in decimal). Other blocks are ignored.
    ///
    /// Parameters
    /// ----------
    /// frame : Frame
    ///     One molecule in the ``atoms`` / ``bonds`` vocabulary. Select one
    ///     molecule of a many-molecule frame with ``Frame.subset`` first.
    ///
    /// Returns
    /// -------
    /// CoarseGrain
    ///
    /// Raises
    /// ------
    /// ValueError
    ///     If the frame has no ``atoms`` rows; if ``mol_id`` holds a null row,
    ///     is not an unsigned-int column, or holds more than one distinct
    ///     value (the message points at ``Frame.subset``); if none of
    ///     ``bead_type`` / ``type`` / ``type_id`` is present, or the chosen one
    ///     has a null row or the wrong dtype (string for ``bead_type`` /
    ///     ``type``, unsigned int for ``type_id``); if a bond row names no bead
    ///     row; or if the ``members`` block lacks an unsigned-int ``ibead`` or
    ///     ``atom`` column, names no bead row, or repeats an
    ///     ``(ibead, atom)`` pair.
    #[staticmethod]
    fn from_frame(py: Python<'_>, frame: &PyFrame) -> PyResult<Py<PyCoarseGrain>> {
        let core = frame.clone_core_frame()?;
        let inner = CoarseGrain::from_frame(&core).map_err(molrs_error_to_pyerr)?;
        PyCoarseGrain::from_core(py, inner)
    }

    // ---- structural graph hash (WL) ----

    /// Isomorphism-invariant Weisfeiler–Lehman structural hash (``int``) of the
    /// bead graph (bead-type node labels, bond-order edge labels). Shares the
    /// same [`MolGraph`] primitive that serves the all-atom case.
    fn structural_hash(&self) -> u64 {
        self.inner.structural_hash()
    }

    /// Deterministic canonical bead ordering (a list of bead handles) from the
    /// WL refinement.
    fn canonical_order(&self) -> Vec<u64> {
        self.inner
            .canonical_order()
            .into_iter()
            .map(node_to_u64)
            .collect()
    }

    /// Whether `self` and `other` are isomorphic as labeled bead graphs.
    fn is_isomorphic(&self, other: &PyCoarseGrain) -> bool {
        self.inner.is_isomorphic(&other.inner)
    }

    /// Independent deep copy, with a copy of its :attr:`props` and the same
    /// member world. **Handles are preserved**.
    fn copy(&self, py: Python<'_>) -> PyResult<Py<PyCoarseGrain>> {
        self.derive(py, self.inner.clone())
    }

    /// Structural merge of ``other`` into ``self``; ``other`` is emptied.
    /// Returns ``{old_handle: new_handle}``.
    ///
    /// Raises
    /// ------
    /// ValueError
    ///     If a property of ``other`` contradicts the type this graph holds
    ///     for that key (a string ``tag`` into an int ``tag`` column).
    fn merge(&mut self, other: &mut Self) -> PyResult<HashMap<u64, u64>> {
        let taken = std::mem::take(&mut other.inner);
        other.forget_views();
        Ok(self
            .inner
            .merge(taken)
            .map_err(molrs_error_to_pyerr)?
            .into_iter()
            .map(|(k, v)| (node_to_u64(k), node_to_u64(v)))
            .collect())
    }

    /// Induced subgraph on bead handles. Returns ``(subgraph, node_map)``.
    fn induced_subgraph(
        &self,
        py: Python<'_>,
        nodes: Vec<u64>,
    ) -> PyResult<(Py<PyCoarseGrain>, HashMap<u64, u64>)> {
        let ids: Vec<_> = nodes.into_iter().map(node_from_u64).collect();
        let (sub, map) = self
            .inner
            .induced_subgraph(&ids)
            .map_err(molrs_error_to_pyerr)?;
        let py_sub = self.derive(py, sub)?;
        let py_map = map
            .into_iter()
            .map(|(k, v)| (node_to_u64(k), node_to_u64(v)))
            .collect();
        Ok((py_sub, py_map))
    }

    /// Radius ball around bead ``centers`` over CG bonds.
    #[pyo3(signature = (centers, radius))]
    fn extract_subgraph(
        &self,
        py: Python<'_>,
        centers: Vec<u64>,
        radius: i64,
    ) -> PyResult<PyExtractedSubgraph> {
        let ids: Vec<_> = centers.into_iter().map(node_from_u64).collect();
        let ext = self
            .inner
            .extract_subgraph(&ids, radius)
            .map_err(molrs_error_to_pyerr)?;
        PyExtractedSubgraph::from_coarsegrain(py, self, ext)
    }

    /// Mass-weighted centre of the bead group ``group``.
    ///
    /// ``R = sum_i m_i r_i / sum_i m_i`` over the listed beads, reading
    /// positions from ``x`` / ``y`` / ``z`` (Å) and masses from ``mass``
    /// (g/mol). A bead listed twice counts twice. :meth:`add_bead` writes no
    /// ``mass``, so give each bead one (or read it from a frame's ``mass``
    /// column with :meth:`from_frame`) before asking for a centre.
    ///
    /// No periodic imaging: the group's coordinates are used as stored. A
    /// group that straddles a box face must be unwrapped first
    /// (:meth:`Box.unwrap`), otherwise the centre lands between the images.
    ///
    /// Parameters
    /// ----------
    /// group : Sequence[int]
    ///     Bead handles, e.g. one group from ``SubgraphMatcher.find``.
    ///
    /// Returns
    /// -------
    /// numpy.ndarray, shape (3,), float64
    ///     The centre in Å.
    ///
    /// Raises
    /// ------
    /// ValueError
    ///     If ``group`` is empty; if a handle is not a live bead of this graph,
    ///     or a bead lacks a finite ``x`` / ``y`` / ``z`` or a finite,
    ///     non-negative ``mass`` (the message names its int handle); or if the
    ///     total mass is not positive.
    /// OverflowError
    ///     If a handle in ``group`` is negative (handles are unsigned ints).
    fn center<'py>(&self, py: Python<'py>, group: Vec<u64>) -> PyResult<Bound<'py, PyArray1<f64>>> {
        let group: Vec<NodeId> = group.into_iter().map(node_from_u64).collect();
        let center = self
            .inner
            .as_molgraph()
            .center(&group)
            .map_err(center_error_to_pyerr)?;
        Ok(vector_to_py(py, &center))
    }

    /// The positions of ``beads``, one row per listed bead, in the listed
    /// order (a bead listed twice appears twice).
    ///
    /// Parameters
    /// ----------
    /// beads : Sequence[int]
    ///     Bead handles.
    ///
    /// Returns
    /// -------
    /// numpy.ndarray, shape (k, 3), float64
    ///     ``x`` / ``y`` / ``z`` as stored (Å).
    ///
    /// Raises
    /// ------
    /// ValueError
    ///     If a handle is not a live bead of this graph, or a bead lacks a
    ///     finite ``x`` / ``y`` / ``z``; the message names its int handle.
    fn positions<'py>(
        &self,
        py: Python<'py>,
        beads: Vec<u64>,
    ) -> PyResult<Bound<'py, PyArray2<f64>>> {
        let beads: Vec<NodeId> = beads.into_iter().map(node_from_u64).collect();
        let points = self.inner.positions(&beads).map_err(molrs_error_to_pyerr)?;
        Ok(
            Array2::from_shape_fn((points.len(), 3), |(row, axis)| points[row][axis])
                .into_pyarray(py),
        )
    }

    /// The site axes of ``beads``, one row per listed bead, in the listed
    /// order. A site from ``Coarsener.coarsen`` carries the vector from the
    /// first bead of its group to the site; a one-bead site's axis is zero.
    ///
    /// Parameters
    /// ----------
    /// beads : Sequence[int]
    ///     Bead handles.
    ///
    /// Returns
    /// -------
    /// numpy.ndarray, shape (k, 3), float64
    ///     ``axis_x`` / ``axis_y`` / ``axis_z`` as stored (Å).
    ///
    /// Raises
    /// ------
    /// ValueError
    ///     If a handle is not a live bead of this graph, or a bead lacks a
    ///     finite axis; the message names its int handle.
    fn axes<'py>(&self, py: Python<'py>, beads: Vec<u64>) -> PyResult<Bound<'py, PyArray2<f64>>> {
        let beads: Vec<NodeId> = beads.into_iter().map(node_from_u64).collect();
        let axes = self.inner.axes(&beads).map_err(molrs_error_to_pyerr)?;
        Ok(Array2::from_shape_fn((axes.len(), 3), |(row, c)| axes[row][c]).into_pyarray(py))
    }

    /// The ``bead_type`` of each of ``beads``, in the listed order (a bead
    /// listed twice appears twice).
    ///
    /// Parameters
    /// ----------
    /// beads : Sequence[int]
    ///     Bead handles.
    ///
    /// Returns
    /// -------
    /// list[str]
    ///     One type per listed bead.
    ///
    /// Raises
    /// ------
    /// ValueError
    ///     If a handle is not a live bead of this graph, or a bead carries no
    ///     ``bead_type``; the message names its int handle.
    fn bead_types(&self, beads: Vec<u64>) -> PyResult<Vec<String>> {
        let beads: Vec<NodeId> = beads.into_iter().map(node_from_u64).collect();
        self.inner.bead_types(&beads).map_err(molrs_error_to_pyerr)
    }
}

graph_world_impl!(PyCoarseGrain);

leaf_views_impl!(PyCoarseGrain);

impl PyCoarseGrain {
    /// A new `CoarseGrain` holding `inner`, with no props and no member
    /// world (see [`PyAtomistic::from_core`]).
    pub(crate) fn from_core(py: Python<'_>, inner: CoarseGrain) -> PyResult<Py<PyCoarseGrain>> {
        let leaf = Self {
            inner,
            props: PyDict::new(py).unbind(),
            views: ViewCache::default(),
            member_world: None,
        };
        Py::new(py, (leaf, PyMolGraph::base()))
    }

    /// A new `CoarseGrain` holding `inner`, derived from this one: a copy of
    /// its props and the same member world (see [`PyAtomistic::derive`]).
    pub(crate) fn derive(&self, py: Python<'_>, inner: CoarseGrain) -> PyResult<Py<PyCoarseGrain>> {
        let leaf = Self {
            inner,
            props: self.props.bind(py).copy()?.unbind(),
            views: ViewCache::default(),
            member_world: self.member_world.as_ref().map(|w| w.clone_ref(py)),
        };
        Py::new(py, (leaf, PyMolGraph::base()))
    }

    /// Borrow the held core [`CoarseGrain`] (for the bead-pattern matcher,
    /// which reads the pattern and the target graph).
    pub(crate) fn core(&self) -> &CoarseGrain {
        &self.inner
    }
}

/// The shared argument seam of the two leaf ``replicate`` methods:
/// ``rotations (N,3,3)`` + ``translations (N,3)`` as rigid motions, and
/// ``frag_ids`` as ``i32`` (an ``int32`` array read directly, any other
/// integer sequence element by element, out-of-range values refused).
fn replicate_args(
    rotations: &ArrayArgDyn<'_, f64>,
    translations: &ArrayArgDyn<'_, f64>,
    frag_ids: &Bound<'_, PyAny>,
) -> PyResult<(Vec<molrs::op::Rigid>, Vec<i32>)> {
    let transforms = crate::op::rigids_from_arrays(rotations, translations)?;
    let frag_ids = match frag_ids.extract::<ArrayArgDyn<'_, i32>>() {
        Ok(array) => {
            let view = array.as_array();
            if view.ndim() != 1 {
                return Err(PyValueError::new_err(format!(
                    "frag_ids must have shape (N,), got {:?}",
                    view.shape()
                )));
            }
            view.iter().copied().collect()
        }
        Err(_) => frag_ids.extract::<Vec<i32>>()?,
    };
    Ok((transforms, frag_ids))
}

// ---------------------------------------------------------------------------
// Rigid replication
// ---------------------------------------------------------------------------

/// The leaf ``replicate`` method: `$leaf` is the Python class name, `$node`
/// what its nodes are called, `$kept` what else a copy carries.
macro_rules! replicate_impl {
    ($ty:ty, $leaf:literal, $node:literal, $kept:literal) => {
        #[pymethods]
        impl $ty {
            /// Grow this graph by one rigid copy of ``template`` per transform.
            ///
            /// Copy ``c`` is ``template`` moved by ``rotations[c] @ r + translations[c]``
            /// and every node of it is stamped ``frag_id = frag_ids[c]``. Every
            #[doc = concat!("relation kind of ``template`` is copied with it", $kept, ". Column-wise and")]
            /// atomic: on an error this graph is unchanged. ``template`` is never
            /// mutated. The GIL is released while the copies are written.
            ///
            /// Parameters
            /// ----------
            #[doc = concat!("template : ", $leaf)]
            ///     Another graph: this graph cannot be its own template.
            /// rotations : ndarray, shape (N, 3, 3), float64
            /// translations : ndarray, shape (N, 3), float64
            /// frag_ids : ndarray, shape (N,), int32
            ///
            /// Returns
            /// -------
            /// list[int]
            #[doc = concat!("    The new ", $node, " handles, copy-major.")]
            ///
            /// Raises
            /// ------
            /// ValueError
            ///     If ``template`` is this graph, on a wrong shape, disagreeing
            ///     counts, or a column of ``template`` whose type contradicts
            ///     the one this graph holds.
            fn replicate(
                slf: &Bound<'_, Self>,
                template: &Bound<'_, Self>,
                rotations: ArrayArgDyn<'_, f64>,
                translations: ArrayArgDyn<'_, f64>,
                frag_ids: &Bound<'_, PyAny>,
            ) -> PyResult<Vec<u64>> {
                // Checked before either borrow: borrowing one object both
                // ways would be PyO3's borrow error, not this refusal.
                if slf.is(template) {
                    return Err(PyValueError::new_err(
                        "a graph cannot replicate itself; pass a copy as the template",
                    ));
                }
                let (transforms, frag_ids) = replicate_args(&rotations, &translations, frag_ids)?;
                let template = template.try_borrow()?;
                let mut this = slf.try_borrow_mut()?;
                let (target, source) = (&mut this.inner, &template.inner);
                let added = slf
                    .py()
                    .detach(|| target.replicate(source, &transforms, &frag_ids))
                    .map_err(molrs_error_to_pyerr)?;
                Ok(added.into_iter().map(node_to_u64).collect())
            }
        }
    };
}

replicate_impl!(PyAtomistic, "Atomistic", "atom", "");

replicate_impl!(PyCoarseGrain, "CoarseGrain", "bead", "");

// ---------------------------------------------------------------------------
// Rigid-body moves
// ---------------------------------------------------------------------------
//
// `translate` / `rotate` / `scale` are leaf methods: each resolves to the
// leaf's *own* graph (never the empty base it carries for `issubclass`).

macro_rules! rigid_body_impl {
    ($ty:ty) => {
        #[pymethods]
        impl $ty {
            /// Translate every node that has coordinates by `delta`. Returns
            /// this graph, so moves chain.
            fn translate(mut slf: PyRefMut<'_, Self>, delta: [f64; 3]) -> PyRefMut<'_, Self> {
                slf.mol_mut().translate(delta);
                slf
            }

            /// Rotate every node that has coordinates by `angle` radians about
            /// `axis`, pivoting on `about` (default: the origin). Returns this
            /// graph, so moves chain.
            #[pyo3(signature = (axis, angle, about=None))]
            fn rotate(
                mut slf: PyRefMut<'_, Self>,
                axis: [f64; 3],
                angle: f64,
                about: Option<[f64; 3]>,
            ) -> PyResult<PyRefMut<'_, Self>> {
                slf.mol_mut()
                    .rotate(axis, angle, about)
                    .map_err(|error| PyValueError::new_err(error.to_string()))?;
                Ok(slf)
            }

            /// Scale every node that has coordinates by a per-axis `factor`
            /// about `about` (default: the origin). Pass `[s, s, s]` for a
            /// uniform scale. Returns this graph, so moves chain.
            #[pyo3(signature = (factor, about=None))]
            fn scale(
                mut slf: PyRefMut<'_, Self>,
                factor: [f64; 3],
                about: Option<[f64; 3]>,
            ) -> PyRefMut<'_, Self> {
                slf.mol_mut().scale(factor, about);
                slf
            }
        }
    };
}

rigid_body_impl!(PyAtomistic);

rigid_body_impl!(PyCoarseGrain);

/// Keep only the `atom_fields` columns of `frame`'s ``atoms`` block; `None`
/// keeps them all. A requested column the block lacks raises
/// ``ValueError`` naming it.
fn select_atom_fields(
    mut frame: molrs::core::Frame,
    atom_fields: Option<Vec<String>>,
) -> PyResult<molrs::core::Frame> {
    let Some(fields) = atom_fields else {
        return Ok(frame);
    };
    let keys: Vec<&str> = fields.iter().map(String::as_str).collect();
    if let Some(atoms) = frame.remove("atoms") {
        let selected = atoms
            .select_columns(&keys)
            .map_err(|e| PyValueError::new_err(e.to_string()))?;
        frame.insert("atoms", selected);
    } else if let Some(first) = keys.first() {
        return Err(PyValueError::new_err(format!(
            "column '{first}' not found: the frame has no atoms block"
        )));
    }
    Ok(frame)
}
