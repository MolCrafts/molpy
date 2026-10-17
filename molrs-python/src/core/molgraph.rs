//! Python bindings for the molecular graphs.
//!
//! - [`PyMolGraph`] (`molrs.core.MolGraph`) — the domain-agnostic ECS world:
//!   stable `int` node and relation handles, by-name fields, and the
//!   kind-tagged relation API.
//! - [`PyAtomistic`] (`molrs.core.Atomistic`) / [`PyCoarseGrain`]
//!   (`molrs.core.CoarseGrain`) — peer leaves that **hold a core [`Atomistic`]
//!   / [`CoarseGrain`]** and speak only its typed API. They are not
//!   `MolGraph`s: no generic world method reaches their graph, so every
//!   invariant the core keeps (an atom's valid ``atomic_number``, a bead's
//!   string ``bead_type``, a bond class set only by ``set_bond_class``) holds
//!   for every Python door. Every entity a leaf hands out or takes is a live
//!   view ([`super::graph_views`]); a view's mapping is the one door to its
//!   fields.
//!
//! The methods both leaves share (bonds, ports, geometry, composition,
//! hashing, frames, copying) are emitted once by [`leaf_impl!`].

use crate::array::ArrayArgDyn;
use std::collections::HashMap;
use std::str::FromStr;

use ndarray::Array2;

use numpy::{IntoPyArray, PyArray1, PyArray2};

use pyo3::exceptions::{PyTypeError, PyValueError};
use pyo3::prelude::*;
use pyo3::types::{PyDict, PyList, PyTuple};
use pyo3::{PyTraverseError, PyVisit};

use molrs::core::keys;
use molrs::core::keys::VSITE;
use molrs::core::{
    Atomistic, BondNumber, BondOrder, CenterError, CoarseGrain, EntityCell, ExtractedAtomistic,
    ExtractedCoarseGrain, KindId, LinkError, MolGraph, NodeId, PortKind, PropValue, Props,
    RelationId,
};

use super::bond_order::{PyBondNumber, PyBondOrder};
use super::graph_views::{Leaf, LeafGraph, PyNodeRef, PyRefs, PyRelationRef, ViewCache, vsite_of};

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

/// The keyword arguments of a builder (``**props``) as one [`Props`] bag.
///
/// Every value goes through [`py_to_prop`], so a value that is not a
/// ``bool`` / ``int`` / ``float`` / ``str`` (``None`` included) raises
/// ``TypeError`` before anything is written.
pub(crate) fn py_props(props: Option<&Bound<'_, PyDict>>) -> PyResult<Props> {
    let mut bag = Props::new();
    if let Some(props) = props {
        for (key, value) in props.iter() {
            bag.set(&key.extract::<String>()?, py_to_prop(&value)?);
        }
    }
    Ok(bag)
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

/// A property bag as a Python ``dict``.
fn props_to_py<'py>(py: Python<'py>, props: &Props) -> PyResult<Bound<'py, PyDict>> {
    let dict = PyDict::new(py);
    for (key, value) in props.iter() {
        dict.set_item(key, prop_to_py(py, value)?)?;
    }
    Ok(dict)
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
/// Node ids cross as the `int` handles a view carries (`view.handle`), never
/// as the Rust debug form `NodeId(3v1)`. Every variant is matched by name, so
/// a new variant is a compile error here rather than a silent fallback.
pub(crate) fn center_error_message(e: CenterError) -> String {
    match e {
        CenterError::Empty => "center of an empty node set".to_owned(),
        CenterError::NotFound { node } => {
            format!("node {} is not in this graph", u64::from(node))
        }
        CenterError::BadPosition { node } => format!(
            "node {} has a missing or non-finite '{}'/'{}'/'{}'",
            u64::from(node),
            keys::X,
            keys::Y,
            keys::Z
        ),
        CenterError::BadMass { node } => format!(
            "node {} has a missing, negative or non-finite '{}'",
            u64::from(node),
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
/// Port and atom ids cross as the `int` handles a view carries, never as
/// `RelationId(..)` / `NodeId(..)`. Every variant is matched by name, with no
/// catch-all arm.
pub(crate) fn link_error_message(e: LinkError) -> String {
    let port = u64::from;
    let atom = u64::from;
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
// Entity arguments of the leaves
// ---------------------------------------------------------------------------

/// The node `view` names, checked to be a node view of `leaf`.
fn own_node(leaf: &Leaf<'_>, view: &Bound<'_, PyAny>) -> PyResult<NodeId> {
    let node = view
        .cast::<PyNodeRef>()
        .map_err(|_| PyTypeError::new_err("expected a node view (Atom / Bead) of this graph"))?;
    leaf.own_node(node)
}

/// The relation `view` names, checked to be a `kind` view of `leaf`.
fn own_relation(leaf: &Leaf<'_>, view: &Bound<'_, PyAny>, kind: &str) -> PyResult<RelationId> {
    let relation = view
        .cast::<PyRelationRef>()
        .map_err(|_| PyTypeError::new_err(format!("expected a '{kind}' view of this graph")))?;
    leaf.own_relation(relation, kind)
}

/// `nodes` as views of `leaf`, in order.
fn node_views<'py>(
    leaf: &Leaf<'py>,
    nodes: impl IntoIterator<Item = NodeId>,
) -> PyResult<Vec<Bound<'py, PyAny>>> {
    nodes
        .into_iter()
        .map(|id| leaf.node(u64::from(id)))
        .collect()
}

/// A node map between two graphs as ``{from view: to view}``.
fn view_map<'py>(
    py: Python<'py>,
    from: &Leaf<'py>,
    to: &Leaf<'py>,
    map: &HashMap<NodeId, NodeId>,
) -> PyResult<Bound<'py, PyDict>> {
    let dict = PyDict::new(py);
    for (&k, &v) in map {
        dict.set_item(from.node(u64::from(k))?, to.node(u64::from(v))?)?;
    }
    Ok(dict)
}

// ---------------------------------------------------------------------------
// ExtractedSubgraph — result of radius-ball extraction
// ---------------------------------------------------------------------------

/// Result of :meth:`Atomistic.extract_subgraph` / :meth:`CoarseGrain.extract_subgraph`.
///
/// * ``graph`` — the extracted leaf (``Atomistic`` or ``CoarseGrain``)
/// * ``boundary`` — parent nodes with a neighbour outside the ball
/// * ``parent_of`` — ``{new node: parent node}``
/// * ``hops`` — ``{parent node: hops from the nearest centre}``
/// * ``node_map`` — ``{parent node: new node}``
///
/// Nodes are views; a parent node is a view of the graph the ball was cut
/// from.
#[pyclass(
    module = "molrs.core",
    name = "ExtractedSubgraph",
    frozen,
    skip_from_py_object,
    subclass
)]
pub struct PyExtractedSubgraph {
    graph: Py<PyAny>,
    boundary: Py<PyList>,
    parent_of: Py<PyDict>,
    hops: Py<PyDict>,
    node_map: Py<PyDict>,
}

#[pymethods]
impl PyExtractedSubgraph {
    #[new]
    fn new(
        graph: Py<PyAny>,
        boundary: Py<PyList>,
        parent_of: Py<PyDict>,
        hops: Py<PyDict>,
        node_map: Py<PyDict>,
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
        let py = slf.py();
        let this = slf.get();
        crate::pickle::reduce_via_type(
            slf.as_any(),
            (
                this.graph.clone_ref(py),
                this.boundary.clone_ref(py),
                this.parent_of.clone_ref(py),
                this.hops.clone_ref(py),
                this.node_map.clone_ref(py),
            ),
        )
    }

    /// The extracted graph.
    #[getter]
    fn graph(&self, py: Python<'_>) -> Py<PyAny> {
        self.graph.clone_ref(py)
    }

    /// Parent nodes with a neighbour outside the ball (a new list).
    #[getter]
    fn boundary<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyList>> {
        PyList::new(py, self.boundary.bind(py).iter())
    }

    /// ``{new node: parent node}`` (a new dict).
    #[getter]
    fn parent_of<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyDict>> {
        self.parent_of.bind(py).copy()
    }

    /// ``{parent node: hops from the nearest centre}`` (a new dict).
    #[getter]
    fn hops<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyDict>> {
        self.hops.bind(py).copy()
    }

    /// ``{parent node: new node}`` (a new dict).
    #[getter]
    fn node_map<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyDict>> {
        self.node_map.bind(py).copy()
    }

    fn __repr__(&self, py: Python<'_>) -> String {
        format!(
            "ExtractedSubgraph(n_boundary={}, n_nodes_mapped={})",
            self.boundary.bind(py).len(),
            self.node_map.bind(py).len()
        )
    }

    fn __traverse__(&self, visit: PyVisit<'_>) -> Result<(), PyTraverseError> {
        visit.call(&self.graph)?;
        visit.call(&self.boundary)?;
        visit.call(&self.parent_of)?;
        visit.call(&self.hops)?;
        visit.call(&self.node_map)
    }
}

impl PyExtractedSubgraph {
    /// The result record of cutting `graph` out of `parent`.
    fn of(
        parent: &Bound<'_, PyAny>,
        graph: Py<PyAny>,
        boundary: Vec<NodeId>,
        parent_of: HashMap<NodeId, NodeId>,
        hops: HashMap<NodeId, i64>,
        node_map: HashMap<NodeId, NodeId>,
    ) -> PyResult<Self> {
        let py = parent.py();
        let from = Leaf::of(parent)?;
        let to = Leaf::of(graph.bind(py))?;
        let boundary = PyList::new(py, node_views(&from, boundary)?)?;
        let hop_map = PyDict::new(py);
        for (node, hop) in hops {
            hop_map.set_item(from.node(u64::from(node))?, hop)?;
        }
        Ok(Self {
            parent_of: view_map(py, &to, &from, &parent_of)?.unbind(),
            node_map: view_map(py, &from, &to, &node_map)?.unbind(),
            graph,
            boundary: boundary.unbind(),
            hops: hop_map.unbind(),
        })
    }

    fn from_atomistic(slf: &Bound<'_, PyAtomistic>, ext: ExtractedAtomistic) -> PyResult<Self> {
        let graph = slf.borrow().derive(slf.py(), ext.graph)?.into_any();
        Self::of(
            slf.as_any(),
            graph,
            ext.boundary,
            ext.parent_of,
            ext.hops,
            ext.node_map,
        )
    }

    fn from_coarsegrain(
        slf: &Bound<'_, PyCoarseGrain>,
        ext: ExtractedCoarseGrain,
    ) -> PyResult<Self> {
        let graph = slf.borrow().derive(slf.py(), ext.graph)?.into_any();
        Self::of(
            slf.as_any(),
            graph,
            ext.boundary,
            ext.parent_of,
            ext.hops,
            ext.node_map,
        )
    }
}

// ---------------------------------------------------------------------------
// Pickling a MolGraph: its content as plain Python data
// ---------------------------------------------------------------------------

/// A world's content as plain Python data — node fields in row order and,
/// per relation kind, its arity and its relations as endpoint *rows* plus
/// fields. Handles are not kept (a restored world mints fresh ones).
fn dump_molgraph<'py>(graph: &MolGraph, py: Python<'py>) -> PyResult<Bound<'py, PyTuple>> {
    let table = graph.node_table();
    let nodes = PyList::empty(py);
    let mut row_of: HashMap<NodeId, usize> = HashMap::new();
    for (row, id) in graph.node_ids().enumerate() {
        row_of.insert(id, row);
        let fields = PyDict::new(py);
        for (key, cell) in table.row_cells(id) {
            fields.set_item(key, cell_to_py(py, cell)?)?;
        }
        nodes.append(fields)?;
    }
    let kinds = PyList::empty(py);
    for kid in graph.kind_ids() {
        let relations = PyList::empty(py);
        for (_, relation) in graph.relations(kid) {
            let ends: Vec<usize> = relation.nodes.iter().map(|node| row_of[node]).collect();
            relations.append((ends, props_to_py(py, &relation.props)?))?;
        }
        kinds.append((graph.kind_name(kid), graph.arity(kid), relations))?;
    }
    (nodes, kinds).into_pyobject(py)
}

/// Rebuild [`dump_molgraph`]'s `content` into a new world.
#[allow(
    clippy::type_complexity,
    reason = "the pickled kind table: (name, arity, [(endpoint rows, fields)])"
)]
fn load_molgraph(content: &Bound<'_, PyAny>) -> PyResult<MolGraph> {
    let (nodes, kinds): (
        Vec<Bound<'_, PyDict>>,
        Vec<(String, usize, Vec<(Vec<usize>, Bound<'_, PyDict>)>)>,
    ) = content.extract()?;
    let mut graph = MolGraph::new();
    let mut ids = Vec::with_capacity(nodes.len());
    for fields in nodes {
        ids.push(
            graph
                .add_node(py_props(Some(&fields))?)
                .map_err(molrs_error_to_pyerr)?,
        );
    }
    for (name, arity, relations) in kinds {
        let kid = match graph.kind_id(&name) {
            Some(kid) => kid,
            None => graph.register_kind(&name, arity),
        };
        for (ends, fields) in relations {
            let nodes = rows_to_nodes(&ids, &ends)?;
            graph
                .add_relation(kid, &nodes, py_props(Some(&fields))?)
                .map_err(molrs_error_to_pyerr)?;
        }
    }
    Ok(graph)
}

/// Pickled endpoint rows as the nodes minted for them.
fn rows_to_nodes(ids: &[NodeId], rows: &[usize]) -> PyResult<Vec<NodeId>> {
    rows.iter()
        .map(|&row| {
            ids.get(row).copied().ok_or_else(|| {
                PyValueError::new_err(format!("relation endpoint row {row} is no node"))
            })
        })
        .collect()
}

// ---------------------------------------------------------------------------
// Pickling a leaf: through its typed API
// ---------------------------------------------------------------------------

/// A leaf's content as plain Python data: node fields in row order and, per
/// relation kind of `kinds` (in that order), its relations as endpoint rows
/// plus fields.
fn dump_leaf<'py>(
    py: Python<'py>,
    graph: &dyn LeafGraph,
    kinds: &[String],
) -> PyResult<Bound<'py, PyTuple>> {
    let table = graph.table();
    let nodes = PyList::empty(py);
    let mut row_of: HashMap<NodeId, usize> = HashMap::new();
    for (row, id) in table.handles().enumerate() {
        row_of.insert(id, row);
        let fields = PyDict::new(py);
        for (key, cell) in table.row_cells(id) {
            fields.set_item(key, cell_to_py(py, cell)?)?;
        }
        nodes.append(fields)?;
    }
    let dumped = PyList::empty(py);
    for name in kinds {
        let kind = graph.kind(name)?;
        let relations = PyList::empty(py);
        for id in graph.relation_ids(&kind).map_err(molrs_error_to_pyerr)? {
            let (ends, props) = graph
                .get_relation(&kind, id)
                .map_err(molrs_error_to_pyerr)?;
            let ends: Vec<usize> = ends.iter().map(|node| row_of[node]).collect();
            relations.append((ends, props_to_py(py, &props)?))?;
        }
        dumped.append((name, relations))?;
    }
    (nodes, dumped).into_pyobject(py)
}

/// The pickled leaf content: node fields, then `(kind, [(rows, fields)])`.
type LeafContent<'py> = (
    Vec<Bound<'py, PyDict>>,
    Vec<(String, Vec<(Vec<usize>, Bound<'py, PyDict>)>)>,
);

/// Re-add a pickled port through the typed `add_port`.
fn port_args(fields: &Props) -> PyResult<(PortKind, String, BondNumber)> {
    let missing = |key: &str| PyValueError::new_err(format!("pickled port lacks '{key}'"));
    let kind = fields
        .get_str("port_kind")
        .ok_or_else(|| missing("port_kind"))?;
    let kind = PortKind::from_str(kind).map_err(molrs_error_to_pyerr)?;
    let label = fields
        .get_str("port_label")
        .ok_or_else(|| missing("port_label"))?
        .to_owned();
    let order: i32 = fields
        .get_int("port_order")
        .ok_or_else(|| missing("port_order"))?;
    let order = u32::try_from(order)
        .map_err(|_| PyValueError::new_err(format!("pickled port_order {order} is negative")))?;
    Ok((kind, label, BondNumber::from_code(order)))
}

/// Endpoint `i` of a pickled relation.
fn end(nodes: &[NodeId], i: usize, kind: &str) -> PyResult<NodeId> {
    nodes.get(i).copied().ok_or_else(|| {
        PyValueError::new_err(format!("pickled '{kind}' relation has too few endpoints"))
    })
}

/// Rebuild an `Atomistic` from [`dump_leaf`]'s content: every atom and
/// relation through the typed adds, so a pickle cannot smuggle in an atom
/// without a valid ``atomic_number``.
fn load_atomistic(content: &Bound<'_, PyAny>) -> PyResult<Atomistic> {
    let (nodes, kinds): LeafContent<'_> = content.extract()?;
    let mut mol = Atomistic::new();
    let mut ids = Vec::with_capacity(nodes.len());
    for fields in nodes {
        ids.push(
            mol.add_atom(py_props(Some(&fields))?)
                .map_err(molrs_error_to_pyerr)?,
        );
    }
    for (kind, relations) in kinds {
        for (rows, fields) in relations {
            let n = rows_to_nodes(&ids, &rows)?;
            let props = py_props(Some(&fields))?;
            let e = |i| end(&n, i, &kind);
            match kind.as_str() {
                "bonds" => mol.add_bond(e(0)?, e(1)?, props).map(drop),
                "angles" => mol.add_angle(e(0)?, e(1)?, e(2)?, props).map(drop),
                "dihedrals" => mol
                    .add_dihedral(e(0)?, e(1)?, e(2)?, e(3)?, props)
                    .map(drop),
                "impropers" => mol
                    .add_improper(e(0)?, e(1)?, e(2)?, e(3)?, props)
                    .map(drop),
                "ports" => {
                    let (glyph, label, order) = port_args(&props)?;
                    mol.add_port(e(0)?, e(1)?, glyph, &label, order).map(drop)
                }
                block => mol.add_term(block, &n, props).map(drop),
            }
            .map_err(molrs_error_to_pyerr)?;
        }
    }
    Ok(mol)
}

/// Rebuild a `CoarseGrain` from [`dump_leaf`]'s content: every bead through
/// the typed add, so a pickle cannot smuggle in a bead without a string
/// ``bead_type``.
fn load_coarsegrain(content: &Bound<'_, PyAny>) -> PyResult<CoarseGrain> {
    let (nodes, kinds): LeafContent<'_> = content.extract()?;
    let mut cg = CoarseGrain::new();
    let mut ids = Vec::with_capacity(nodes.len());
    for fields in nodes {
        ids.push(
            cg.add_bead(py_props(Some(&fields))?)
                .map_err(molrs_error_to_pyerr)?,
        );
    }
    for (kind, relations) in kinds {
        for (rows, fields) in relations {
            let n = rows_to_nodes(&ids, &rows)?;
            let props = py_props(Some(&fields))?;
            let e = |i| end(&n, i, &kind);
            match kind.as_str() {
                "bonds" => cg.add_bond(e(0)?, e(1)?, props).map(drop),
                "ports" => {
                    let (glyph, label, order) = port_args(&props)?;
                    cg.add_port(e(0)?, e(1)?, glyph, &label, order).map(drop)
                }
                other => {
                    return Err(PyValueError::new_err(format!(
                        "a pickled CoarseGrain has no relation kind '{other}'"
                    )));
                }
            }
            .map_err(molrs_error_to_pyerr)?;
        }
    }
    Ok(cg)
}

// ---------------------------------------------------------------------------
// PyMolGraph — the generic world
// ---------------------------------------------------------------------------

/// Domain-agnostic ECS world, exposed to Python as `molrs.core.MolGraph`.
///
/// Nodes and relations are addressed by stable opaque ``int`` handles
/// (generational slotmap keys): deleting one entity never invalidates
/// another, and a stale handle raises.
#[pyclass(module = "molrs.core", name = "MolGraph", subclass)]
pub struct PyMolGraph {
    inner: MolGraph,
}

impl PyMolGraph {
    /// A new world holding `inner`.
    pub(crate) fn from_core(py: Python<'_>, inner: MolGraph) -> PyResult<Py<PyMolGraph>> {
        Py::new(py, Self { inner })
    }

    /// Borrow the held core [`MolGraph`].
    pub(crate) fn core(&self) -> &MolGraph {
        &self.inner
    }
}

#[pymethods]
impl PyMolGraph {
    /// Create an empty world.
    #[new]
    fn new() -> Self {
        Self {
            inner: MolGraph::new(),
        }
    }

    fn __reduce__<'py>(slf: &Bound<'py, Self>) -> PyResult<Bound<'py, PyTuple>> {
        let py = slf.py();
        let content = dump_molgraph(&slf.borrow().inner, py)?;
        crate::pickle::reduce_with_state(
            slf.as_any(),
            PyTuple::empty(py),
            PyTuple::new(py, [content])?.into_any(),
        )
    }

    fn __setstate__(&mut self, state: (Bound<'_, PyAny>,)) -> PyResult<()> {
        self.inner = load_molgraph(&state.0)?;
        Ok(())
    }

    // ---- nodes ----

    /// Add a node carrying ``props`` and return its handle.
    ///
    /// Parameters
    /// ----------
    /// **props : bool | int | float | str
    ///     The node's fields.
    ///
    /// Returns
    /// -------
    /// int
    ///     The new node's handle.
    ///
    /// Raises
    /// ------
    /// TypeError
    ///     If a value is not a bool, int, float or str.
    /// ValueError
    ///     If a value contradicts the type an existing column holds for its
    ///     key; the world is then unchanged.
    #[pyo3(signature = (**props))]
    fn add_node(&mut self, props: Option<&Bound<'_, PyDict>>) -> PyResult<u64> {
        let props = py_props(props)?;
        self.inner
            .add_node(props)
            .map(u64::from)
            .map_err(molrs_error_to_pyerr)
    }

    /// Delete node ``node`` and every relation touching it.
    ///
    /// Raises
    /// ------
    /// ValueError
    ///     If ``node`` is stale or unknown.
    fn del_node(&mut self, node: u64) -> PyResult<()> {
        self.inner
            .del_node(NodeId::from(node))
            .map(drop)
            .map_err(molrs_error_to_pyerr)
    }

    /// Delete every node of ``nodes`` and every relation touching one.
    /// Atomic: on an error (a stale handle, a handle listed twice) nothing
    /// is deleted.
    fn del_nodes(&mut self, nodes: Vec<u64>) -> PyResult<()> {
        let ids: Vec<NodeId> = nodes.into_iter().map(NodeId::from).collect();
        self.inner
            .del_nodes(&ids)
            .map(drop)
            .map_err(molrs_error_to_pyerr)
    }

    /// Every field of node ``node``, as a new ``dict``.
    ///
    /// Raises
    /// ------
    /// ValueError
    ///     If ``node`` is stale or unknown.
    fn get_node<'py>(&self, py: Python<'py>, node: u64) -> PyResult<Bound<'py, PyDict>> {
        let props = self
            .inner
            .get_node(NodeId::from(node))
            .map_err(molrs_error_to_pyerr)?;
        props_to_py(py, &props)
    }

    /// Set field ``key`` of node ``node`` (``value`` is bool|int|float|str).
    ///
    /// ``key`` is a :class:`molrs.core.keys.Key` or ``str``.
    fn set_node(
        &mut self,
        node: u64,
        key: &Bound<'_, PyAny>,
        value: &Bound<'_, PyAny>,
    ) -> PyResult<()> {
        let key = crate::core::schema::extract_column_key(key)?;
        let value = py_to_prop(value)?;
        self.inner
            .set_node(NodeId::from(node), &key, value)
            .map_err(molrs_error_to_pyerr)
    }

    /// Clear field ``key`` of node ``node`` (no-op if absent).
    ///
    /// ``key`` is a :class:`molrs.core.keys.Key` or ``str``.
    fn clear_node(&mut self, node: u64, key: &Bound<'_, PyAny>) -> PyResult<()> {
        let key = crate::core::schema::extract_column_key(key)?;
        self.inner
            .clear_node(NodeId::from(node), &key)
            .map_err(molrs_error_to_pyerr)
    }

    /// All live node handles, in row order.
    fn node_ids(&self) -> Vec<u64> {
        self.inner.node_ids().map(u64::from).collect()
    }

    /// Number of nodes.
    #[getter]
    fn n_nodes(&self) -> usize {
        self.inner.n_nodes()
    }

    // ---- relations ----

    /// Register a relation kind (idempotent for a matching arity).
    fn register_kind(&mut self, kind: &str, arity: usize) -> PyResult<()> {
        if let Some(kid) = self.inner.kind_id(kind) {
            let existing = self.inner.arity(kid);
            if existing != arity {
                return Err(PyValueError::new_err(format!(
                    "kind '{kind}' already registered with arity {existing}, got {arity}"
                )));
            }
            return Ok(());
        }
        self.inner.register_kind(kind, arity);
        Ok(())
    }

    /// Names of all registered relation kinds, in registration order.
    fn kinds(&self) -> Vec<String> {
        self.inner
            .kind_ids()
            .map(|kid| self.inner.kind_name(kid).to_owned())
            .collect()
    }

    /// Fixed endpoint count of a registered relation kind.
    fn kind_arity(&self, kind: &str) -> PyResult<usize> {
        let kid = kind_id_checked(&self.inner, kind)?;
        Ok(self.inner.arity(kid))
    }

    /// Add a relation of ``kind`` over node handles, carrying ``props``.
    ///
    /// Parameters
    /// ----------
    /// kind : str
    ///     A registered relation kind.
    /// nodes : Sequence[int]
    ///     Endpoint node handles, as many as the kind's arity.
    /// **props : bool | int | float | str
    ///     The relation's fields.
    ///
    /// Returns
    /// -------
    /// int
    ///     The new relation's handle.
    ///
    /// Raises
    /// ------
    /// TypeError
    ///     If a value is not a bool, int, float or str.
    /// ValueError
    ///     If ``kind`` is not registered, the endpoint count is not its
    ///     arity, an endpoint is stale, or a value contradicts the type
    ///     an existing column holds; the world is then unchanged.
    #[pyo3(signature = (kind, nodes, /, **props))]
    fn add_relation(
        &mut self,
        kind: &str,
        nodes: Vec<u64>,
        props: Option<&Bound<'_, PyDict>>,
    ) -> PyResult<u64> {
        let kid = kind_id_checked(&self.inner, kind)?;
        let nids: Vec<NodeId> = nodes.into_iter().map(NodeId::from).collect();
        let props = py_props(props)?;
        self.inner
            .add_relation(kid, &nids, props)
            .map(u64::from)
            .map_err(molrs_error_to_pyerr)
    }

    /// Relation ``rel`` of ``kind``: its endpoint node handles, in order, and
    /// a new ``dict`` of its fields.
    fn get_relation<'py>(
        &self,
        py: Python<'py>,
        kind: &str,
        rel: u64,
    ) -> PyResult<(Vec<u64>, Bound<'py, PyDict>)> {
        let kid = kind_id_checked(&self.inner, kind)?;
        let relation = self
            .inner
            .get_relation(kid, RelationId::from(rel))
            .map_err(molrs_error_to_pyerr)?;
        let nodes = relation.nodes.iter().map(|&n| u64::from(n)).collect();
        Ok((nodes, props_to_py(py, &relation.props)?))
    }

    /// Relations of ``kind`` incident to node ``node``, as
    /// ``(relation_handle, other_node_handle)`` pairs, via the adjacency
    /// index (O(degree)). Only arity-2 kinds are tracked in adjacency.
    fn incident_relations(&self, node: u64, kind: &str) -> PyResult<Vec<(u64, u64)>> {
        let kid = kind_id_checked(&self.inner, kind)?;
        Ok(self
            .inner
            .neighbor_relations(NodeId::from(node))
            .filter(|(k, _, _)| *k == kid)
            .map(|(_, rid, other)| (u64::from(rid), u64::from(other)))
            .collect())
    }

    /// Set field ``key`` of relation ``rel`` of ``kind``.
    fn set_relation_prop(
        &mut self,
        kind: &str,
        rel: u64,
        key: &str,
        value: &Bound<'_, PyAny>,
    ) -> PyResult<()> {
        let kid = kind_id_checked(&self.inner, kind)?;
        let value = py_to_prop(value)?;
        self.inner
            .set_relation_prop(kid, RelationId::from(rel), key, value)
            .map_err(molrs_error_to_pyerr)
    }

    /// Clear field ``key`` of relation ``rel`` of ``kind`` (no-op if absent).
    fn clear_relation_prop(&mut self, kind: &str, rel: u64, key: &str) -> PyResult<()> {
        let kid = kind_id_checked(&self.inner, kind)?;
        self.inner
            .clear_relation_prop(kid, RelationId::from(rel), key)
            .map_err(molrs_error_to_pyerr)
    }

    /// Delete relation ``rel`` of ``kind``.
    fn del_relation(&mut self, kind: &str, rel: u64) -> PyResult<()> {
        let kid = kind_id_checked(&self.inner, kind)?;
        self.inner
            .del_relation(kid, RelationId::from(rel))
            .map(drop)
            .map_err(molrs_error_to_pyerr)
    }

    /// Delete every relation of ``rels`` of ``kind``. Atomic: on an error
    /// nothing is deleted.
    fn del_relations(&mut self, kind: &str, rels: Vec<u64>) -> PyResult<()> {
        let kid = kind_id_checked(&self.inner, kind)?;
        let ids: Vec<RelationId> = rels.into_iter().map(RelationId::from).collect();
        self.inner
            .del_relations(kid, &ids)
            .map(drop)
            .map_err(molrs_error_to_pyerr)
    }

    /// Number of relations of ``kind``.
    fn n_relations(&self, kind: &str) -> PyResult<usize> {
        let kid = kind_id_checked(&self.inner, kind)?;
        Ok(self.inner.n_relations(kid))
    }

    /// Live relation handles of ``kind``, in row order.
    ///
    /// Authoritative enumeration — callers must not probe opaque handle
    /// ranges. Returns an empty list for a registered kind with no
    /// relations; errors only if ``kind`` is unregistered.
    fn relation_ids(&self, kind: &str) -> PyResult<Vec<u64>> {
        let kid = kind_id_checked(&self.inner, kind)?;
        Ok(self.inner.relation_ids(kid).map(u64::from).collect())
    }

    // ---- columns ----

    /// The field column ``key``, aligned to row order (length ==
    /// ``n_nodes``), as a numpy array of the column's own element type.
    ///
    /// A ``f64`` column is a **zero-copy view** that writes through to the
    /// world (``col[i] = v`` updates the node at row ``i``); ``i32``, ``bool``
    /// and ``str`` columns are copied. The view borrows the world's storage;
    /// structural mutation (``add_node`` / ``del_node``) may reallocate or
    /// reorder the column and invalidate an outstanding view — re-fetch after
    /// such ops.
    ///
    /// Every node must carry the field: a column with a hole is a
    /// ``KeyError`` naming how many nodes lack it, never a silently
    /// zero-filled array. Use :meth:`validity` to find the holes and
    /// :meth:`get_node` for node-wise reads.
    fn column<'py>(slf: Bound<'py, Self>, key: &Bound<'_, PyAny>) -> PyResult<Bound<'py, PyAny>> {
        use molrs::core::EntityColumn;

        let key = crate::core::schema::extract_column_key(key)?;
        let py = slf.py();
        let this = slf.borrow();
        let table = this.inner.node_table();
        let absent = || {
            pyo3::exceptions::PyKeyError::new_err(format!(
                "component '{key}' is absent from every entity"
            ))
        };
        let validity = table.col_validity(&key).ok_or_else(absent)?;
        let holes = validity.as_slice().iter().filter(|v| !**v).count();
        if holes > 0 {
            return Err(pyo3::exceptions::PyKeyError::new_err(format!(
                "component '{key}' is absent on {holes} of {} entities; a column \
                 needs it on every one (see validity())",
                validity.len()
            )));
        }
        match table.entity_column(&key).ok_or_else(absent)? {
            EntityColumn::F64(data, _) => {
                let (ptr, len) = (data.as_ptr(), data.len());
                drop(this);
                // SAFETY: `slf` owns the backing Vec and is held as the array's base
                // object, so the memory stays valid for the array's lifetime; the
                // documented contract forbids structural mutation while held.
                let view = unsafe { numpy::ndarray::ArrayView1::from_shape_ptr(len, ptr) };
                Ok(unsafe { numpy::PyArray1::borrow_from_array(&view, slf.into_any()) }.into_any())
            }
            EntityColumn::I32(data, _) => Ok(numpy::PyArray1::from_slice(py, data).into_any()),
            EntityColumn::Bool(data, _) => Ok(numpy::PyArray1::from_slice(py, data).into_any()),
            EntityColumn::Str(data, _) => {
                let list = PyList::new(py, data.iter().map(String::as_str))?;
                py.import("numpy")?.call_method1("array", (list,))
            }
        }
    }

    /// Names of every field column registered on the node table (a field set
    /// on at least one node), in no particular order.
    fn columns(&self) -> Vec<String> {
        self.inner
            .node_table()
            .columns()
            .map(str::to_owned)
            .collect()
    }

    /// Validity mask (numpy ``bool`` array, copied) of field column ``key``,
    /// aligned to row order: ``True`` where the node at that row has the
    /// field set.
    ///
    /// ``key`` is a :class:`molrs.core.keys.Key` or ``str``.
    fn validity<'py>(
        &self,
        py: Python<'py>,
        key: &Bound<'_, PyAny>,
    ) -> PyResult<Bound<'py, numpy::PyArray1<bool>>> {
        let key = crate::core::schema::extract_column_key(key)?;
        let valid = self
            .inner
            .node_table()
            .col_validity(&key)
            .ok_or_else(|| PyValueError::new_err(format!("column '{key}' is absent")))?;
        Ok(numpy::PyArray1::from_slice(py, valid.as_slice()))
    }

    /// Zero-copy adopt: **move** ``other``'s storage into ``self``, leaving
    /// ``other`` empty. Handles in the adopted world stay valid (the whole
    /// generational slotmap is moved, not reindexed).
    fn adopt(&mut self, other: &mut PyMolGraph) {
        self.inner = std::mem::take(&mut other.inner);
    }

    // ---- ports ----

    /// Number of ports (``0`` when the world carries none).
    #[getter]
    fn n_ports(&self) -> usize {
        self.inner.n_ports()
    }

    /// Join port ``a`` to port ``b`` with a new anchor–anchor bond and
    /// return the bond's relation handle.
    ///
    /// Both leaving groups are deleted, their partial charge (e) folds
    /// onto the anchors, and the anchors are bonded with the port order. No
    /// coordinate moves.
    ///
    /// Raises
    /// ------
    /// ValueError
    ///     If a handle names no live port, a port is stale, the two ports do
    ///     not accept each other, share an anchor, their anchors are already
    ///     bonded, or the leaving groups overlap or reach an anchor; the
    ///     world is unchanged.
    fn link(&mut self, a: u64, b: u64) -> PyResult<u64> {
        self.inner
            .link(RelationId::from(a), RelationId::from(b))
            .map(u64::from)
            .map_err(link_error_to_pyerr)
    }

    /// Propagate each ``frag_id`` to the unlabelled degree-1 nodes hanging
    /// off a labelled one (one pass); returns how many were labelled.
    fn inherit_frag_ids(&mut self) -> usize {
        self.inner.inherit_frag_ids()
    }
}

// ---------------------------------------------------------------------------
// The methods both leaves share
// ---------------------------------------------------------------------------

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

/// Emits the `#[pymethods]` both leaves share. `$noun` is what the leaf's
/// nodes are called, `$leaf` its Python class name.
macro_rules! leaf_impl {
    ($ty:ty, $leaf:literal, $noun:literal) => {
        #[pymethods]
        impl $ty {
            // ---- graph annotations and copies ----

            /// Whole-graph annotations (a name, a provenance tag): the keyword
            /// arguments the graph was built with, as a live dict. A copy, a
            /// pickle and every graph derived from this one (perception,
            /// typing, conformers, subgraphs) carry them.
            #[getter]
            fn props(&self, py: Python<'_>) -> Py<PyDict> {
                self.props.clone_ref(py)
            }

            /// ``copy.copy(graph)``: an independent copy with a copy of its
            /// :attr:`props`. **Handles are preserved** (``view.handle`` of a
            /// node names the same node in the copy).
            fn __copy__(&self, py: Python<'_>) -> PyResult<Py<Self>> {
                self.derive(py, self.inner.clone())
            }

            /// ``copy.deepcopy(graph)``: the same independent copy as
            /// ``copy.copy`` — a graph holds no Python objects to recurse
            /// into.
            #[pyo3(signature = (_memo, /))]
            fn __deepcopy__(&self, py: Python<'_>, _memo: &Bound<'_, PyAny>) -> PyResult<Py<Self>> {
                self.derive(py, self.inner.clone())
            }

            // ---- bonds ----

            #[doc = concat!("Add a bond between ", $noun, "s ``a`` and ``b`` of this graph, carrying")]
            /// ``props``, and return its view.
            ///
            /// Raises
            /// ------
            /// TypeError
            ///     If a value is not a bool, int, float or str.
            /// ValueError
            ///     If an endpoint belongs to another graph, or a value contradicts
            ///     the type an existing column holds; the graph is then unchanged.
            #[pyo3(signature = (a, b, /, **props))]
            fn add_bond<'py>(
                slf: &Bound<'py, Self>,
                a: &Bound<'py, PyAny>,
                b: &Bound<'py, PyAny>,
                props: Option<&Bound<'py, PyDict>>,
            ) -> PyResult<Bound<'py, PyAny>> {
                let leaf = Leaf::of(slf.as_any())?;
                let (a, b) = (own_node(&leaf, a)?, own_node(&leaf, b)?);
                let props = py_props(props)?;
                let id = slf
                    .try_borrow_mut()?
                    .inner
                    .add_bond(a, b, props)
                    .map_err(molrs_error_to_pyerr)?;
                leaf.relation("bonds", u64::from(id))
            }

            /// Delete ``bond``; its view goes stale.
            fn del_bond(slf: &Bound<'_, Self>, bond: &Bound<'_, PyAny>) -> PyResult<()> {
                let leaf = Leaf::of(slf.as_any())?;
                let id = own_relation(&leaf, bond, "bonds")?;
                slf.try_borrow_mut()?
                    .inner
                    .del_bond(id)
                    .map(drop)
                    .map_err(molrs_error_to_pyerr)
            }

            /// Delete every bond of ``bonds`` (views or a ``Refs``). Atomic: on
            /// an error nothing is deleted.
            fn del_bonds(slf: &Bound<'_, Self>, bonds: &Bound<'_, PyAny>) -> PyResult<()> {
                let leaf = Leaf::of(slf.as_any())?;
                let ids = leaf.own_relations(bonds, "bonds")?;
                slf.try_borrow_mut()?
                    .inner
                    .del_bonds(&ids)
                    .map(drop)
                    .map_err(molrs_error_to_pyerr)
            }

            /// Every bond, in row order.
            #[getter]
            fn bonds(slf: &Bound<'_, Self>) -> PyResult<PyRefs> {
                Leaf::of(slf.as_any())?.relations("bonds")
            }

            /// Number of bonds.
            #[getter]
            fn n_bonds(&self) -> usize {
                self.inner.n_bonds()
            }

            #[doc = concat!("The bonds of ``", $noun, "``, as ``(neighbour, bond)`` pairs.")]
            fn neighbor_bonds<'py>(
                slf: &Bound<'py, Self>,
                node: &Bound<'py, PyAny>,
            ) -> PyResult<Vec<(Bound<'py, PyAny>, Bound<'py, PyAny>)>> {
                let leaf = Leaf::of(slf.as_any())?;
                let id = own_node(&leaf, node)?;
                let pairs: Vec<(NodeId, RelationId)> =
                    slf.try_borrow()?.inner.neighbor_bonds(id).collect();
                pairs
                    .into_iter()
                    .map(|(other, bond)| {
                        Ok((
                            leaf.node(u64::from(other))?,
                            leaf.relation("bonds", u64::from(bond))?,
                        ))
                    })
                    .collect()
            }

            // ---- ports ----

            /// Record a bonding descriptor on the ``(anchor, handle_atom)``
            /// valence and return the port's view.
            ///
            /// Parameters
            /// ----------
            /// anchor
            ///     The node that keeps its place in the product (positional-only).
            /// handle_atom
            ///     The node bonded to ``anchor`` that roots the leaving group
            ///     (positional-only). Endpoint order is load-bearing.
            /// kind : str
            ///     The notation glyph -- one of ``"$"``, ``"<"``, ``">"``, ``"!"``.
            /// label : str, optional
            ///     Descriptor label; ``""`` (the default) means unnamed.
            /// order : int, optional
            ///     Multiplicity of the bond this port will form, ``1`` to ``4``
            ///     (default ``1``).
            ///
            /// Raises
            /// ------
            /// ValueError
            ///     If an endpoint belongs to another graph, ``kind`` is not one of
            ///     the four glyphs, ``handle_atom`` is not bonded to ``anchor``, that
            ///     valence already carries a port, or ``order`` is not a definite
            ///     bond number.
            #[pyo3(signature = (anchor, handle_atom, /, kind, label = "", order = 1))]
            fn add_port<'py>(
                slf: &Bound<'py, Self>,
                anchor: &Bound<'py, PyAny>,
                handle_atom: &Bound<'py, PyAny>,
                kind: &str,
                label: &str,
                order: u32,
            ) -> PyResult<Bound<'py, PyAny>> {
                let leaf = Leaf::of(slf.as_any())?;
                let (anchor, handle) = (own_node(&leaf, anchor)?, own_node(&leaf, handle_atom)?);
                let kind = PortKind::from_str(kind).map_err(molrs_error_to_pyerr)?;
                let id = slf
                    .try_borrow_mut()?
                    .inner
                    .add_port(anchor, handle, kind, label, BondNumber::from_code(order))
                    .map_err(molrs_error_to_pyerr)?;
                leaf.relation("ports", u64::from(id))
            }

            /// Delete ``port`` (its bond stays); its view goes stale.
            fn del_port(slf: &Bound<'_, Self>, port: &Bound<'_, PyAny>) -> PyResult<()> {
                let leaf = Leaf::of(slf.as_any())?;
                let id = own_relation(&leaf, port, "ports")?;
                slf.try_borrow_mut()?
                    .inner
                    .del_port(id)
                    .map(drop)
                    .map_err(molrs_error_to_pyerr)
            }

            /// Delete every port of ``ports`` (their bonds stay). Atomic.
            fn del_ports(slf: &Bound<'_, Self>, ports: &Bound<'_, PyAny>) -> PyResult<()> {
                let leaf = Leaf::of(slf.as_any())?;
                let ids = leaf.own_relations(ports, "ports")?;
                slf.try_borrow_mut()?
                    .inner
                    .del_ports(&ids)
                    .map(drop)
                    .map_err(molrs_error_to_pyerr)
            }

            /// Every port, in row order (empty when the graph carries none).
            #[getter]
            fn ports(slf: &Bound<'_, Self>) -> PyResult<PyRefs> {
                Leaf::of(slf.as_any())?.relations("ports")
            }

            /// Number of ports.
            #[getter]
            fn n_ports(&self) -> usize {
                self.inner.n_ports()
            }

            /// The port on the ``(anchor, handle_atom)`` valence, or ``None``.
            fn port_on<'py>(
                slf: &Bound<'py, Self>,
                anchor: &Bound<'py, PyAny>,
                handle_atom: &Bound<'py, PyAny>,
            ) -> PyResult<Option<Bound<'py, PyAny>>> {
                let leaf = Leaf::of(slf.as_any())?;
                let (anchor, handle) = (own_node(&leaf, anchor)?, own_node(&leaf, handle_atom)?);
                let port = slf.try_borrow()?.inner.port_on(anchor, handle);
                port.map(|id| leaf.relation("ports", u64::from(id)))
                    .transpose()
            }

            #[doc = concat!("The leaving group of ``port``: the ", $noun, "s of its handle's connected")]
            /// component once the anchor–handle bond is cut, in handle order.
            fn leaving_group<'py>(
                slf: &Bound<'py, Self>,
                port: &Bound<'py, PyAny>,
            ) -> PyResult<Vec<Bound<'py, PyAny>>> {
                let leaf = Leaf::of(slf.as_any())?;
                let id = own_relation(&leaf, port, "ports")?;
                let group = {
                    let this = slf.try_borrow()?;
                    let port = this.inner.get_port(id).map_err(molrs_error_to_pyerr)?;
                    this.inner.leaving_group(&port).map_err(molrs_error_to_pyerr)?
                };
                node_views(&leaf, group)
            }

            /// Join port ``a`` to port ``b`` and return the new anchor–anchor
            /// bond.
            ///
            /// Both leaving groups are deleted, their partial charge (e) folds
            /// onto the anchors, and the anchors are bonded with the port
            /// order. No coordinate moves.
            ///
            /// Raises
            /// ------
            /// ValueError
            ///     If a port belongs to another graph or is stale, the two ports
            ///     do not accept each other, share an anchor, their anchors are
            ///     already bonded, or the leaving groups overlap or reach an
            ///     anchor; the graph is unchanged.
            fn link<'py>(
                slf: &Bound<'py, Self>,
                a: &Bound<'py, PyAny>,
                b: &Bound<'py, PyAny>,
            ) -> PyResult<Bound<'py, PyAny>> {
                let leaf = Leaf::of(slf.as_any())?;
                let (a, b) = (own_relation(&leaf, a, "ports")?, own_relation(&leaf, b, "ports")?);
                let bond = slf
                    .try_borrow_mut()?
                    .inner
                    .link(a, b)
                    .map_err(link_error_to_pyerr)?;
                leaf.relation("bonds", u64::from(bond))
            }

            #[doc = concat!("Propagate each ``frag_id`` to the unlabelled degree-1 ", $noun, "s bonded")]
            /// to a labelled one (one pass); returns how many were labelled.
            fn inherit_frag_ids(&mut self) -> usize {
                self.inner.inherit_frag_ids()
            }

            // ---- geometry ----

            #[doc = concat!("Translate every ", $noun, " that has coordinates by ``delta`` (Å).")]
            fn translate(&mut self, delta: [f64; 3]) {
                self.inner.translate(delta);
            }

            #[doc = concat!("Rotate every ", $noun, " that has coordinates by ``angle`` radians about")]
            /// ``axis``, pivoting on ``about`` (Å; ``None``: the origin).
            ///
            /// Raises
            /// ------
            /// ValueError
            ///     If ``axis`` has no direction or ``angle`` is not finite;
            ///     nothing moves then.
            #[pyo3(signature = (axis, angle, about = None))]
            fn rotate(&mut self, axis: [f64; 3], angle: f64, about: Option<[f64; 3]>) -> PyResult<()> {
                self.inner
                    .rotate(axis, angle, about)
                    .map_err(molrs_error_to_pyerr)
            }

            #[doc = concat!("Scale every ", $noun, " that has coordinates by the per-axis ``factor``")]
            /// about ``about`` (Å; ``None``: the origin). Pass ``[s, s, s]`` for a
            /// uniform scale.
            #[pyo3(signature = (factor, about = None))]
            fn scale(&mut self, factor: [f64; 3], about: Option<[f64; 3]>) {
                self.inner.scale(factor, about);
            }

            #[doc = concat!("Mass-weighted centre (Å) of ``", $noun, "s`` (views or a ``Refs``).")]
            ///
            /// ``R = sum_i m_i r_i / sum_i m_i``, reading positions from ``x`` /
            /// ``y`` / ``z`` and masses from ``mass``; a node listed twice
            /// counts twice. No periodic imaging: a group split across a box
            /// face must be unwrapped first (:meth:`Box.unwrap`).
            ///
            /// Returns
            /// -------
            /// numpy.ndarray, shape (3,), float64
            ///
            /// Raises
            /// ------
            /// ValueError
            ///     If the set is empty, a node belongs to another graph, lacks a
            ///     finite ``x`` / ``y`` / ``z`` or a finite, non-negative ``mass``
            ///     (the message names its handle), or the total mass is not
            ///     positive.
            fn center<'py>(
                slf: &Bound<'py, Self>,
                nodes: &Bound<'py, PyAny>,
            ) -> PyResult<Bound<'py, PyArray1<f64>>> {
                let ids = Leaf::of(slf.as_any())?.own_nodes(nodes)?;
                let center = slf
                    .try_borrow()?
                    .inner
                    .center(&ids)
                    .map_err(center_error_to_pyerr)?;
                Ok(vector_to_py(slf.py(), &center))
            }

            // ---- composition ----

            /// Grow this graph by one rigid copy of ``template`` per transform.
            ///
            /// Copy ``c`` is ``template`` moved by
            /// ``rotations[c] @ r + translations[c]`` and every node of it is
            /// stamped ``frag_id = frag_ids[c]``. Every relation of ``template``
            /// is copied with it. Column-wise and atomic: on an error this graph
            /// is unchanged. ``template`` is never mutated. The GIL is released
            /// while the copies are written.
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
            /// Refs
            #[doc = concat!("    The new ", $noun, "s, copy-major.")]
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
            ) -> PyResult<PyRefs> {
                // Checked before either borrow: borrowing one object both
                // ways would be PyO3's borrow error, not this refusal.
                if slf.is(template) {
                    return Err(PyValueError::new_err(
                        "a graph cannot replicate itself; pass a copy as the template",
                    ));
                }
                let (transforms, frag_ids) = replicate_args(&rotations, &translations, frag_ids)?;
                let added = {
                    let template = template.try_borrow()?;
                    let mut this = slf.try_borrow_mut()?;
                    let (target, source) = (&mut this.inner, &template.inner);
                    slf.py()
                        .detach(|| target.replicate(source, &transforms, &frag_ids))
                        .map_err(molrs_error_to_pyerr)?
                };
                Ok(Leaf::of(slf.as_any())?.node_refs(added.into_iter().map(u64::from).collect()))
            }

            #[doc = concat!("The subgraph induced by ``", $noun, "s`` (views or a ``Refs``), with the map")]
            /// ``{node of this graph: node of the subgraph}``.
            fn induced_subgraph<'py>(
                slf: &Bound<'py, Self>,
                nodes: &Bound<'py, PyAny>,
            ) -> PyResult<(Py<Self>, Bound<'py, PyDict>)> {
                let py = slf.py();
                let leaf = Leaf::of(slf.as_any())?;
                let ids = leaf.own_nodes(nodes)?;
                let (sub, map) = {
                    let this = slf.try_borrow()?;
                    let (sub, map) = this.inner.induced_subgraph(&ids).map_err(molrs_error_to_pyerr)?;
                    (this.derive(py, sub)?, map)
                };
                let child = Leaf::of(sub.bind(py).as_any())?;
                let map = view_map(py, &leaf, &child, &map)?;
                Ok((sub, map))
            }

            // ---- structural graph hash (WL) ----

            /// Isomorphism-invariant Weisfeiler–Lehman structural hash (``int``).
            ///
            /// A stable, reproducible dedup key over the graph's node and bond
            /// labels: identical for a node-permuted copy, sensitive to any label
            /// or connectivity change, reproducible across runs and processes.
            fn structural_hash(&self) -> u64 {
                self.inner.structural_hash()
            }

            #[doc = concat!("Deterministic canonical ", $noun, " order from the WL refinement, so two")]
            /// isomorphic graphs line up node by node.
            fn canonical_order(slf: &Bound<'_, Self>) -> PyResult<PyRefs> {
                let order: Vec<u64> = slf
                    .try_borrow()?
                    .inner
                    .canonical_order()
                    .into_iter()
                    .map(u64::from)
                    .collect();
                Ok(Leaf::of(slf.as_any())?.node_refs(order))
            }

            /// Whether ``self`` and ``other`` are isomorphic as labelled graphs.
            fn is_isomorphic(&self, other: PyRef<'_, Self>) -> bool {
                self.inner.is_isomorphic(&other.inner)
            }

            // ---- frames ----

            /// Export to a tabular :class:`~molrs.core.Frame` in the shared
            /// vocabulary. Select columns on the frame's blocks afterwards.
            ///
            /// Raises
            /// ------
            /// ValueError
            ///     If a node or relation property contradicts the dtype the Frame
            ///     schema declares for its key (a string under ``"x"``).
            fn to_frame(&self) -> PyResult<PyFrame> {
                let frame = self.inner.to_frame().map_err(molrs_error_to_pyerr)?;
                PyFrame::from_core_frame(frame)
            }
        }
    };
}

// ---------------------------------------------------------------------------
// PyAtomistic — all-atom leaf (holds a core Atomistic)
// ---------------------------------------------------------------------------

/// All-atom molecular graph, exposed to Python as `molrs.core.Atomistic`.
///
/// ``Atomistic(**props)``: the keywords are the graph's :attr:`props`. Every
/// atom, bond, angle, dihedral, improper, port and term is a live view; a
/// view's mapping is the one door to its fields (``atom["x"] = 1.0``,
/// ``del bond["k"]``).
#[pyclass(module = "molrs.core", name = "Atomistic", subclass)]
pub struct PyAtomistic {
    inner: Atomistic,
    props: Py<PyDict>,
    pub(super) views: ViewCache,
}

/// The terms of `terms` (views of one term block of `leaf`): the block and
/// the handles.
fn own_terms(
    leaf: &Leaf<'_>,
    terms: &Bound<'_, PyAny>,
) -> PyResult<Option<(String, Vec<RelationId>)>> {
    let mut block: Option<String> = None;
    let mut ids = Vec::new();
    for term in terms.try_iter()? {
        let term = term?;
        let view = term
            .cast::<PyRelationRef>()
            .map_err(|_| PyTypeError::new_err("expected term views of this graph"))?;
        let kind = view.get().kind_name().to_owned();
        match &block {
            None => block = Some(kind.clone()),
            Some(first) if *first == kind => {}
            Some(first) => {
                return Err(PyValueError::new_err(format!(
                    "del_terms deletes terms of one block per call; got '{first}' and '{kind}'"
                )));
            }
        }
        ids.push(leaf.own_relation(view, &kind)?);
    }
    Ok(block.map(|block| (block, ids)))
}

/// The six doors of one typed topology block of an `Atomistic` besides its
/// `add_*` (whose arity differs): delete one, delete many, enumerate, count.
macro_rules! topology_block_impl {
    ($kind:literal, $one:literal, $del:ident, $dels:ident, $all:ident, $n:ident,
     $core_del:ident, $core_dels:ident, $core_n:ident) => {
        #[pymethods]
        impl PyAtomistic {
            #[doc = concat!("Delete ``", $one, "``; its view goes stale.")]
            fn $del(slf: &Bound<'_, Self>, relation: &Bound<'_, PyAny>) -> PyResult<()> {
                let leaf = Leaf::of(slf.as_any())?;
                let id = own_relation(&leaf, relation, $kind)?;
                slf.try_borrow_mut()?
                    .inner
                    .$core_del(id)
                    .map(drop)
                    .map_err(molrs_error_to_pyerr)
            }

            #[doc = concat!("Delete every ", $one, " of ``", $kind, "`` (views or a ``Refs``). Atomic.")]
            fn $dels(slf: &Bound<'_, Self>, relations: &Bound<'_, PyAny>) -> PyResult<()> {
                let leaf = Leaf::of(slf.as_any())?;
                let ids = leaf.own_relations(relations, $kind)?;
                slf.try_borrow_mut()?
                    .inner
                    .$core_dels(&ids)
                    .map(drop)
                    .map_err(molrs_error_to_pyerr)
            }

            #[doc = concat!("Every ", $one, ", in row order.")]
            #[getter]
            fn $all(slf: &Bound<'_, Self>) -> PyResult<PyRefs> {
                Leaf::of(slf.as_any())?.relations($kind)
            }

            #[doc = concat!("Number of ", $kind, ".")]
            #[getter]
            fn $n(&self) -> usize {
                self.inner.$core_n()
            }
        }
    };
}

topology_block_impl!(
    "angles", "angle", del_angle, del_angles, angles, n_angles, del_angle, del_angles, n_angles
);
topology_block_impl!(
    "dihedrals",
    "dihedral",
    del_dihedral,
    del_dihedrals,
    dihedrals,
    n_dihedrals,
    del_dihedral,
    del_dihedrals,
    n_dihedrals
);
topology_block_impl!(
    "impropers",
    "improper",
    del_improper,
    del_impropers,
    impropers,
    n_impropers,
    del_improper,
    del_impropers,
    n_impropers
);

#[pymethods]
impl PyAtomistic {
    #[new]
    #[pyo3(signature = (**props))]
    fn new(py: Python<'_>, props: Option<Bound<'_, PyDict>>) -> Self {
        Self {
            inner: Atomistic::new(),
            props: props.unwrap_or_else(|| PyDict::new(py)).unbind(),
            views: ViewCache::default(),
        }
    }

    // ---- atoms ----

    /// Every atom, in row order.
    #[getter]
    fn atoms(slf: &Bound<'_, Self>) -> PyResult<PyRefs> {
        Leaf::of(slf.as_any())?.nodes()
    }

    /// Number of atoms.
    #[getter]
    fn n_atoms(&self) -> usize {
        self.inner.n_atoms()
    }

    /// Add an atom carrying ``props`` and return its view.
    ///
    /// Parameters
    /// ----------
    /// kind : type[VirtualSite], optional
    ///     Keyword-only. Make the atom a virtual site of this class
    ///     (``VirtualSite``, ``DrudeParticle`` or ``MasslessSite``): its
    ///     ``vsite`` field is stamped with the kind the class names. ``None``
    ///     (the default) makes a plain atom.
    /// **props : bool | int | float | str
    ///     The atom's fields. ``atomic_number`` (0..=118, 0 = wildcard) is
    ///     required; a symbol is converted explicitly first
    ///     (``Element.by_symbol("C").z``).
    ///
    /// Returns
    /// -------
    /// Atom
    ///     The new atom's view, of class ``kind`` when one is given.
    ///
    /// Raises
    /// ------
    /// TypeError
    ///     If a value is not a bool, int, float or str, or ``kind`` is not
    ///     one of the three virtual-site classes.
    /// ValueError
    ///     If ``props`` has no valid ``atomic_number``, gives ``vsite``
    ///     together with ``kind``, or a value contradicts the type an
    ///     existing column holds; the graph is then unchanged.
    #[pyo3(signature = (*, kind = None, **props))]
    fn add_atom<'py>(
        slf: &Bound<'py, Self>,
        kind: Option<&Bound<'py, pyo3::types::PyType>>,
        props: Option<&Bound<'py, PyDict>>,
    ) -> PyResult<Bound<'py, PyAny>> {
        let mut props = py_props(props)?;
        if let Some(cls) = kind {
            if props.contains_key(VSITE) {
                return Err(PyValueError::new_err(format!(
                    "give a virtual site's kind once: kind= or '{VSITE}', not both"
                )));
            }
            props.set(VSITE, vsite_of(cls)?);
        }
        Leaf::of(slf.as_any())?.create_node(props)
    }

    /// Delete ``atom`` and every relation touching it; their views go stale.
    ///
    /// Raises
    /// ------
    /// ValueError
    ///     If ``atom`` belongs to another graph or was deleted already.
    fn del_atom(slf: &Bound<'_, Self>, atom: &Bound<'_, PyAny>) -> PyResult<()> {
        let leaf = Leaf::of(slf.as_any())?;
        let id = own_node(&leaf, atom)?;
        slf.try_borrow_mut()?
            .inner
            .del_atom(id)
            .map(drop)
            .map_err(molrs_error_to_pyerr)
    }

    /// Delete every atom of ``atoms`` (views or a ``Refs``) and every
    /// relation touching one. Atomic: on an error (an atom of another graph,
    /// a deleted atom, an atom listed twice) nothing is deleted.
    fn del_atoms(slf: &Bound<'_, Self>, atoms: &Bound<'_, PyAny>) -> PyResult<()> {
        let leaf = Leaf::of(slf.as_any())?;
        let ids = leaf.own_nodes(atoms)?;
        slf.try_borrow_mut()?
            .inner
            .del_atoms(&ids)
            .map(drop)
            .map_err(molrs_error_to_pyerr)
    }

    // ---- bonds: the Atomistic-only doors ----

    /// Set a bond's chemical class and its localized bond number — the one
    /// door that writes ``bond_type`` / ``bond_number`` (a view refuses
    /// them).
    ///
    /// The two are set together because they are only meaningful together:
    /// a class without a number leaves the bond un-standardized, and a
    /// number without a class leaves a consumer no way to tell aromatic from
    /// double.
    fn set_bond_class(
        slf: &Bound<'_, Self>,
        bond: &Bound<'_, PyAny>,
        bond_type: &Bound<'_, PyBondOrder>,
        bond_number: &Bound<'_, PyBondNumber>,
    ) -> PyResult<()> {
        let leaf = Leaf::of(slf.as_any())?;
        let id = own_relation(&leaf, bond, "bonds")?;
        let order = BondOrder::from(*bond_type.get());
        let number = BondNumber::from_code(*bond_number.get() as u32);
        slf.try_borrow_mut()?
            .inner
            .set_bond_class(id, order, number)
            .map_err(molrs_error_to_pyerr)
    }

    /// The bond joining atoms ``a`` and ``b``, or ``None``.
    fn bond_between<'py>(
        slf: &Bound<'py, Self>,
        a: &Bound<'py, PyAny>,
        b: &Bound<'py, PyAny>,
    ) -> PyResult<Option<Bound<'py, PyAny>>> {
        let leaf = Leaf::of(slf.as_any())?;
        let (a, b) = (own_node(&leaf, a)?, own_node(&leaf, b)?);
        let bond = slf.try_borrow()?.inner.bond_between(a, b);
        bond.map(|id| leaf.relation("bonds", u64::from(id)))
            .transpose()
    }

    /// Single-source shortest-path (BFS) distances over the bond graph from
    /// ``atom``, as ``(atom, hops)`` pairs for every atom reachable from it
    /// (``atom`` itself at 0), up to ``max_hops`` when given. Atoms of
    /// another connected component are omitted.
    #[pyo3(signature = (atom, max_hops = None))]
    fn topo_distances<'py>(
        slf: &Bound<'py, Self>,
        atom: &Bound<'py, PyAny>,
        max_hops: Option<i64>,
    ) -> PyResult<Vec<(Bound<'py, PyAny>, i64)>> {
        let leaf = Leaf::of(slf.as_any())?;
        let id = own_node(&leaf, atom)?;
        let distances = slf.try_borrow()?.inner.topo_distances(id, max_hops);
        distances
            .into_iter()
            .map(|(node, hops)| Ok((leaf.node(u64::from(node))?, hops)))
            .collect()
    }

    // ---- angles / dihedrals / impropers ----

    /// Add an angle ``a–b–c`` (``b`` the vertex), carrying ``props``, and
    /// return its view.
    ///
    /// Raises
    /// ------
    /// TypeError
    ///     If a value is not a bool, int, float or str.
    /// ValueError
    ///     If an endpoint belongs to another graph, or a value contradicts
    ///     the type an existing column holds; the graph is then unchanged.
    #[pyo3(signature = (a, b, c, /, **props))]
    fn add_angle<'py>(
        slf: &Bound<'py, Self>,
        a: &Bound<'py, PyAny>,
        b: &Bound<'py, PyAny>,
        c: &Bound<'py, PyAny>,
        props: Option<&Bound<'py, PyDict>>,
    ) -> PyResult<Bound<'py, PyAny>> {
        let leaf = Leaf::of(slf.as_any())?;
        let (a, b, c) = (
            own_node(&leaf, a)?,
            own_node(&leaf, b)?,
            own_node(&leaf, c)?,
        );
        let props = py_props(props)?;
        let id = slf
            .try_borrow_mut()?
            .inner
            .add_angle(a, b, c, props)
            .map_err(molrs_error_to_pyerr)?;
        leaf.relation("angles", u64::from(id))
    }

    /// Add a proper dihedral ``a–b–c–d``, carrying ``props``, and return its
    /// view.
    ///
    /// Raises
    /// ------
    /// TypeError
    ///     If a value is not a bool, int, float or str.
    /// ValueError
    ///     If an endpoint belongs to another graph, or a value contradicts
    ///     the type an existing column holds; the graph is then unchanged.
    #[pyo3(signature = (a, b, c, d, /, **props))]
    fn add_dihedral<'py>(
        slf: &Bound<'py, Self>,
        a: &Bound<'py, PyAny>,
        b: &Bound<'py, PyAny>,
        c: &Bound<'py, PyAny>,
        d: &Bound<'py, PyAny>,
        props: Option<&Bound<'py, PyDict>>,
    ) -> PyResult<Bound<'py, PyAny>> {
        let leaf = Leaf::of(slf.as_any())?;
        let [a, b, c, d] = [a, b, c, d].map(|end| own_node(&leaf, end));
        let (a, b, c, d) = (a?, b?, c?, d?);
        let props = py_props(props)?;
        let id = slf
            .try_borrow_mut()?
            .inner
            .add_dihedral(a, b, c, d, props)
            .map_err(molrs_error_to_pyerr)?;
        leaf.relation("dihedrals", u64::from(id))
    }

    /// Add an improper over ``a, b, c, d`` in its style's slot order,
    /// carrying ``props``, and return its view.
    ///
    /// Raises
    /// ------
    /// TypeError
    ///     If a value is not a bool, int, float or str.
    /// ValueError
    ///     If an endpoint belongs to another graph, or a value contradicts
    ///     the type an existing column holds; the graph is then unchanged.
    #[pyo3(signature = (a, b, c, d, /, **props))]
    fn add_improper<'py>(
        slf: &Bound<'py, Self>,
        a: &Bound<'py, PyAny>,
        b: &Bound<'py, PyAny>,
        c: &Bound<'py, PyAny>,
        d: &Bound<'py, PyAny>,
        props: Option<&Bound<'py, PyDict>>,
    ) -> PyResult<Bound<'py, PyAny>> {
        let leaf = Leaf::of(slf.as_any())?;
        let [a, b, c, d] = [a, b, c, d].map(|end| own_node(&leaf, end));
        let (a, b, c, d) = (a?, b?, c?, d?);
        let props = py_props(props)?;
        let id = slf
            .try_borrow_mut()?
            .inner
            .add_improper(a, b, c, d, props)
            .map_err(molrs_error_to_pyerr)?;
        leaf.relation("impropers", u64::from(id))
    }

    /// Perceive angle, dihedral and improper relations from the bond graph.
    ///
    /// Angles are 2-edge paths ``i-j-k`` and proper dihedrals 3-edge paths
    /// ``i-j-k-l`` over the bonds. Impropers are the molecular-mechanics
    /// reading: one ``[centre, i, j, k]`` quartet per atom with **exactly
    /// three** neighbours, centre first, peripherals sorted — not every
    /// 3-combination at every centre of degree >= 3, which would hand an sp3
    /// carbon four quartets. Whether a trivalent centre is planar enough to
    /// carry the term is force-field data, not a graph property, so every
    /// one is emitted.
    ///
    /// Idempotent; ``clear_existing`` deletes existing relations of the
    /// requested kinds first. Returns ``(n_angles_added, n_dihedrals_added,
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

    // ---- terms: every other relation block ----

    /// Declare the term block ``block`` with ``arity`` atoms per row, so
    /// :meth:`add_term` accepts a block the Frame schema does not define (a
    /// force-field term such as ``urey_bradleys``). Re-declaring a block at
    /// the same arity is a no-op.
    ///
    /// Raises
    /// ------
    /// ValueError
    ///     If ``block`` has typed methods, ``arity`` is 0, the Frame schema
    ///     defines ``block`` at another arity, or the graph already holds it
    ///     at another arity.
    fn declare_term(&mut self, block: &str, arity: usize) -> PyResult<()> {
        self.inner
            .declare_term(block, arity)
            .map_err(molrs_error_to_pyerr)
    }

    /// Add a row to the term block ``block`` (``cmaps``, ``constraints``,
    /// ``drudes``, ``exclusions``, ``pairs``, ``virtual_sites``, a block
    /// declared with :meth:`declare_term`, or any block this graph already
    /// holds) over ``atoms``, carrying ``props``, and return its view.
    ///
    /// The block's arity is the one this graph registered for it, else the
    /// Frame schema's; the block is registered on first use.
    ///
    /// Raises
    /// ------
    /// ValueError
    ///     If ``block`` has typed methods (``bonds``, ``angles``,
    ///     ``dihedrals``, ``impropers``, ``ports``), its arity is unknown or
    ///     differs from ``len(atoms)``, an atom belongs to another graph, or a
    ///     value does not fit its column; nothing is written then.
    #[pyo3(signature = (block, atoms, /, **props))]
    fn add_term<'py>(
        slf: &Bound<'py, Self>,
        block: &str,
        atoms: &Bound<'py, PyAny>,
        props: Option<&Bound<'py, PyDict>>,
    ) -> PyResult<Bound<'py, PyAny>> {
        let leaf = Leaf::of(slf.as_any())?;
        let ids = leaf.own_nodes(atoms)?;
        let props = py_props(props)?;
        let id = slf
            .try_borrow_mut()?
            .inner
            .add_term(block, &ids, props)
            .map_err(molrs_error_to_pyerr)?;
        leaf.relation(block, u64::from(id))
    }

    /// Delete the term ``term``; its view goes stale.
    fn del_term(slf: &Bound<'_, Self>, term: &Bound<'_, PyAny>) -> PyResult<()> {
        let leaf = Leaf::of(slf.as_any())?;
        let view = term
            .cast::<PyRelationRef>()
            .map_err(|_| PyTypeError::new_err("expected a term view of this graph"))?;
        let block = view.get().kind_name().to_owned();
        let id = leaf.own_relation(view, &block)?;
        slf.try_borrow_mut()?
            .inner
            .del_term(&block, id)
            .map(drop)
            .map_err(molrs_error_to_pyerr)
    }

    /// Delete every term of ``terms`` — views of ONE term block. Atomic: on
    /// an error nothing is deleted.
    fn del_terms(slf: &Bound<'_, Self>, terms: &Bound<'_, PyAny>) -> PyResult<()> {
        let leaf = Leaf::of(slf.as_any())?;
        let Some((block, ids)) = own_terms(&leaf, terms)? else {
            return Ok(());
        };
        slf.try_borrow_mut()?
            .inner
            .del_terms(&block, &ids)
            .map(drop)
            .map_err(molrs_error_to_pyerr)
    }

    /// Every row of the term block ``block``, in row order; empty for a
    /// canonical block this graph holds no row of.
    ///
    /// Raises
    /// ------
    /// ValueError
    ///     If ``block`` has typed methods or is no term block.
    fn terms(slf: &Bound<'_, Self>, block: &str) -> PyResult<PyRefs> {
        let handles: Vec<u64> = {
            let this = slf.try_borrow()?;
            this.inner
                .term_ids(block)
                .map_err(molrs_error_to_pyerr)?
                .map(u64::from)
                .collect()
        };
        Ok(PyRefs::of_relations(slf.as_any(), block, handles))
    }

    /// Number of rows of the term block ``block``.
    fn n_terms(&self, block: &str) -> PyResult<usize> {
        self.inner.n_terms(block).map_err(molrs_error_to_pyerr)
    }

    /// The term blocks this graph holds, in registration order.
    fn term_blocks(&self) -> Vec<String> {
        self.inner.term_blocks().map(str::to_owned).collect()
    }

    // ---- composition the leaves do not share ----

    /// Merge ``other`` into this graph and return its atoms, now in this
    /// graph, in ``other``'s former row order. ``other`` is left empty; its
    /// views go stale.
    ///
    /// Raises
    /// ------
    /// ValueError
    ///     If ``other`` is this graph, or a property of ``other`` contradicts
    ///     the type this graph holds for that key (a string ``tag`` into an
    ///     int ``tag`` column); this graph is then unchanged.
    fn merge(slf: &Bound<'_, Self>, other: &Bound<'_, Self>) -> PyResult<PyRefs> {
        if slf.is(other) {
            return Err(PyValueError::new_err("a graph cannot merge itself"));
        }
        let taken = {
            let mut other = other.try_borrow_mut()?;
            other.views.clear();
            std::mem::take(&mut other.inner)
        };
        let order: Vec<NodeId> = taken.atom_ids().collect();
        let map = slf
            .try_borrow_mut()?
            .inner
            .merge(taken)
            .map_err(molrs_error_to_pyerr)?;
        let added = order.iter().map(|old| u64::from(map[old])).collect();
        Ok(Leaf::of(slf.as_any())?.node_refs(added))
    }

    /// Radius ball around ``centers`` (atom views or a ``Refs``) over the
    /// bond graph.
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
        slf: &Bound<'_, Self>,
        centers: &Bound<'_, PyAny>,
        radius: i64,
        regenerate_topology: bool,
        max_ring_size: Option<usize>,
    ) -> PyResult<PyExtractedSubgraph> {
        let ids = Leaf::of(slf.as_any())?.own_nodes(centers)?;
        let ext = {
            let this = slf.try_borrow()?;
            let groups = match max_ring_size {
                Some(max_ring_size) => {
                    molrs::perceive::small_ring_closure(&this.inner, &ids, radius, max_ring_size)
                }
                None => Vec::new(),
            };
            this.inner
                .extract_subgraph(&ids, radius, regenerate_topology, &groups)
                .map_err(molrs_error_to_pyerr)?
        };
        PyExtractedSubgraph::from_atomistic(slf, ext)
    }

    /// Build an `Atomistic` from a :class:`~molrs.core.Frame`: one atom per
    /// ``atoms`` row (each needs ``atomic_number``; convert symbols first
    /// with ``Element.fill_atomic_numbers``), then the relation blocks.
    #[staticmethod]
    fn from_frame(py: Python<'_>, frame: &PyFrame) -> PyResult<Py<PyAtomistic>> {
        let core = frame.clone_core_frame()?;
        let inner = Atomistic::from_frame(&core).map_err(molrs_error_to_pyerr)?;
        PyAtomistic::from_core(py, inner)
    }

    // ---- pickling ----

    fn __reduce__<'py>(slf: &Bound<'py, Self>) -> PyResult<Bound<'py, PyTuple>> {
        let py = slf.py();
        let this = slf.try_borrow()?;
        let mut kinds: Vec<String> = ["bonds", "angles", "dihedrals", "impropers"]
            .into_iter()
            .map(str::to_owned)
            .collect();
        kinds.extend(this.inner.term_blocks().map(str::to_owned));
        kinds.push("ports".to_owned());
        let content = dump_leaf(py, &this.inner, &kinds)?;
        let state = (this.props.bind(py).clone(), content).into_pyobject(py)?;
        crate::pickle::reduce_with_state(slf.as_any(), PyTuple::empty(py), state.into_any())
    }

    fn __setstate__(&mut self, state: (Bound<'_, PyDict>, Bound<'_, PyAny>)) -> PyResult<()> {
        let (props, content) = state;
        self.inner = load_atomistic(&content)?;
        self.props = props.unbind();
        self.views.clear();
        Ok(())
    }

    fn __traverse__(&self, visit: PyVisit<'_>) -> Result<(), PyTraverseError> {
        visit.call(&self.props)
    }

    fn __clear__(&mut self) {
        self.props = Python::attach(|py| PyDict::new(py).unbind());
    }
}

leaf_impl!(PyAtomistic, "Atomistic", "atom");

impl PyAtomistic {
    /// A new `Atomistic` holding `inner`, with no props: the graph-out door
    /// of every API that builds a graph from something that is not one
    /// (a reader, a SMILES string, a frame).
    pub(crate) fn from_core(py: Python<'_>, inner: Atomistic) -> PyResult<Py<PyAtomistic>> {
        Py::new(
            py,
            Self {
                inner,
                props: PyDict::new(py).unbind(),
                views: ViewCache::default(),
            },
        )
    }

    /// A new `Atomistic` holding `inner`, derived from this one: it carries
    /// a copy of this graph's :attr:`props`. The graph-out door of every API
    /// that maps a graph to a graph (copy, perception, typing, conformers,
    /// subgraphs).
    pub(crate) fn derive(&self, py: Python<'_>, inner: Atomistic) -> PyResult<Py<PyAtomistic>> {
        Py::new(
            py,
            Self {
                inner,
                props: self.props.bind(py).copy()?.unbind(),
                views: ViewCache::default(),
            },
        )
    }

    /// Borrow the held core [`Atomistic`] (for domain consumers like the
    /// conformer / force-field typifier that operate on atomistic chemistry).
    pub(crate) fn core(&self) -> &Atomistic {
        &self.inner
    }

    /// Mutably borrow the held core [`Atomistic`], for an in-place edit
    /// through its typed API (perception, charges). Handles that survive the
    /// edit keep their views.
    pub(crate) fn core_mut(&mut self) -> &mut Atomistic {
        &mut self.inner
    }
}

// ---------------------------------------------------------------------------
// PyCoarseGrain — coarse-grained leaf (holds a core CoarseGrain)
// ---------------------------------------------------------------------------

/// Coarse-grained molecular graph, exposed to Python as `molrs.core.CoarseGrain`.
///
/// ``CoarseGrain(**props)``: the keywords are the graph's :attr:`props`.
/// A bead's membership (:meth:`set_bead_members`) names nodes of one source
/// graph — the *member world*, fixed by the first membership recorded.
#[pyclass(module = "molrs.core", name = "CoarseGrain", subclass)]
pub struct PyCoarseGrain {
    inner: CoarseGrain,
    props: Py<PyDict>,
    pub(super) views: ViewCache,
    /// The graph whose nodes the bead memberships name, once one is given.
    member_world: Option<Py<PyAny>>,
}

/// Why memberships read from a frame cannot be answered as views.
const NO_MEMBER_WORLD: &str = "this CoarseGrain's memberships name raw handles of no graph (they \
     were read from a frame); record them from the source graph's views with set_bead_members";

impl PyCoarseGrain {
    /// Whether any bead records a membership.
    fn has_members(&self) -> bool {
        self.inner
            .bead_ids()
            .any(|bead| !self.inner.bead_members(bead).is_empty())
    }

    /// The node handles of `nodes` (views of one graph), checked against —
    /// or fixing — the member world.
    fn member_handles(slf: &Bound<'_, Self>, nodes: &Bound<'_, PyAny>) -> PyResult<Vec<u64>> {
        let py = slf.py();
        let mut handles = Vec::new();
        for node in nodes.try_iter()? {
            let node = node?;
            let node = node
                .cast::<PyNodeRef>()
                .map_err(|_| PyTypeError::new_err("bead members must be node views (Atom / Bead)"))?
                .get();
            let mut this = slf.try_borrow_mut()?;
            match &this.member_world {
                Some(world) if world.bind(py).is(node.world(py)) => {}
                Some(_) => {
                    return Err(PyValueError::new_err(
                        "bead members must all come from the same source graph",
                    ));
                }
                None if this.has_members() => {
                    return Err(PyValueError::new_err(NO_MEMBER_WORLD));
                }
                None => this.member_world = Some(node.world(py).clone().unbind()),
            }
            handles.push(node.handle());
        }
        Ok(handles)
    }
}

#[pymethods]
impl PyCoarseGrain {
    #[new]
    #[pyo3(signature = (**props))]
    fn new(py: Python<'_>, props: Option<Bound<'_, PyDict>>) -> Self {
        Self {
            inner: CoarseGrain::new(),
            props: props.unwrap_or_else(|| PyDict::new(py)).unbind(),
            views: ViewCache::default(),
            member_world: None,
        }
    }

    // ---- beads ----

    /// Every bead, in row order.
    #[getter]
    fn beads(slf: &Bound<'_, Self>) -> PyResult<PyRefs> {
        Leaf::of(slf.as_any())?.nodes()
    }

    /// Number of beads.
    #[getter]
    fn n_beads(&self) -> usize {
        self.inner.n_beads()
    }

    /// Add a bead carrying ``props`` and return its view.
    ///
    /// Parameters
    /// ----------
    /// **props : bool | int | float | str
    ///     The bead's fields. A string ``bead_type`` is required.
    ///
    /// Raises
    /// ------
    /// TypeError
    ///     If a value is not a bool, int, float or str.
    /// ValueError
    ///     If ``props`` has no string ``bead_type``, or a value contradicts
    ///     the type an existing column holds; the graph is then unchanged.
    #[pyo3(signature = (**props))]
    fn add_bead<'py>(
        slf: &Bound<'py, Self>,
        props: Option<&Bound<'py, PyDict>>,
    ) -> PyResult<Bound<'py, PyAny>> {
        let props = py_props(props)?;
        Leaf::of(slf.as_any())?.create_node(props)
    }

    /// Delete ``bead``, its bonds and its membership; their views go stale.
    fn del_bead(slf: &Bound<'_, Self>, bead: &Bound<'_, PyAny>) -> PyResult<()> {
        let leaf = Leaf::of(slf.as_any())?;
        let id = own_node(&leaf, bead)?;
        slf.try_borrow_mut()?
            .inner
            .del_bead(id)
            .map(drop)
            .map_err(molrs_error_to_pyerr)
    }

    /// Delete every bead of ``beads`` (views or a ``Refs``), their bonds and
    /// memberships. Atomic.
    fn del_beads(slf: &Bound<'_, Self>, beads: &Bound<'_, PyAny>) -> PyResult<()> {
        let leaf = Leaf::of(slf.as_any())?;
        let ids = leaf.own_nodes(beads)?;
        slf.try_borrow_mut()?
            .inner
            .del_beads(&ids)
            .map(drop)
            .map_err(molrs_error_to_pyerr)
    }

    // ---- membership ----

    /// Record the nodes ``bead`` groups (its membership), replacing any
    /// previous set; an empty iterable clears it.
    ///
    /// ``members`` are node views of one source graph — the member world,
    /// which the first membership fixes.
    ///
    /// Raises
    /// ------
    /// ValueError
    ///     If ``bead`` belongs to another graph, or the members do not all
    ///     come from the member world.
    fn set_bead_members(
        slf: &Bound<'_, Self>,
        bead: &Bound<'_, PyAny>,
        members: &Bound<'_, PyAny>,
    ) -> PyResult<()> {
        let leaf = Leaf::of(slf.as_any())?;
        let id = own_node(&leaf, bead)?;
        let handles = Self::member_handles(slf, members)?;
        slf.try_borrow_mut()?
            .inner
            .set_bead_members(id, handles)
            .map_err(molrs_error_to_pyerr)
    }

    /// The nodes ``bead`` groups, as views of the member world (empty when
    /// it groups none).
    ///
    /// Raises
    /// ------
    /// ValueError
    ///     If ``bead`` belongs to another graph, a member was deleted from
    ///     the member world, or the memberships were read from a frame (raw
    ///     handles of no graph).
    fn bead_members<'py>(
        slf: &Bound<'py, Self>,
        bead: &Bound<'py, PyAny>,
    ) -> PyResult<Bound<'py, PyTuple>> {
        let py = slf.py();
        let leaf = Leaf::of(slf.as_any())?;
        let id = own_node(&leaf, bead)?;
        let (members, world) = {
            let this = slf.try_borrow()?;
            (
                this.inner.bead_members(id).to_vec(),
                this.member_world.as_ref().map(|w| w.clone_ref(py)),
            )
        };
        if members.is_empty() {
            return Ok(PyTuple::empty(py));
        }
        let world = world.ok_or_else(|| PyValueError::new_err(NO_MEMBER_WORLD))?;
        let source = Leaf::of(world.bind(py))?;
        let views = members
            .into_iter()
            .map(|handle| source.node(handle))
            .collect::<PyResult<Vec<_>>>()?;
        PyTuple::new(py, views)
    }

    /// The beads whose membership includes ``node`` (a view of the member
    /// world), in bead-handle order.
    fn beads_of_atom<'py>(
        slf: &Bound<'py, Self>,
        node: &Bound<'py, PyAny>,
    ) -> PyResult<Vec<Bound<'py, PyAny>>> {
        let py = slf.py();
        let node = node
            .cast::<PyNodeRef>()
            .map_err(|_| PyTypeError::new_err("expected a node view of the member world"))?
            .get();
        let beads = {
            let this = slf.try_borrow()?;
            match &this.member_world {
                Some(world) if world.bind(py).is(node.world(py)) => {
                    this.inner.beads_of_atom(node.handle())
                }
                Some(_) => {
                    return Err(PyValueError::new_err(
                        "the node belongs to another graph than this CoarseGrain's members",
                    ));
                }
                None if this.has_members() => return Err(PyValueError::new_err(NO_MEMBER_WORLD)),
                None => Vec::new(),
            }
        };
        node_views(&Leaf::of(slf.as_any())?, beads)
    }

    // ---- per-bead reads ----

    /// The positions of ``beads`` (views or a ``Refs``), one row per listed
    /// bead, in the listed order.
    ///
    /// Returns
    /// -------
    /// numpy.ndarray, shape (k, 3), float64
    ///     ``x`` / ``y`` / ``z`` as stored (Å).
    ///
    /// Raises
    /// ------
    /// ValueError
    ///     If a bead belongs to another graph, or lacks a finite ``x`` /
    ///     ``y`` / ``z``; the message names its handle.
    fn positions<'py>(
        slf: &Bound<'py, Self>,
        beads: &Bound<'py, PyAny>,
    ) -> PyResult<Bound<'py, PyArray2<f64>>> {
        let ids = Leaf::of(slf.as_any())?.own_nodes(beads)?;
        let points = slf
            .try_borrow()?
            .inner
            .positions(&ids)
            .map_err(molrs_error_to_pyerr)?;
        Ok(
            Array2::from_shape_fn((points.len(), 3), |(row, axis)| points[row][axis])
                .into_pyarray(slf.py()),
        )
    }

    /// The site axes of ``beads``, one row per listed bead, in the listed
    /// order. A site from ``Coarsener.coarsen`` carries the vector from the
    /// first node of its group to the site; a one-node site's axis is zero.
    ///
    /// Returns
    /// -------
    /// numpy.ndarray, shape (k, 3), float64
    ///     ``axis_x`` / ``axis_y`` / ``axis_z`` as stored (Å).
    ///
    /// Raises
    /// ------
    /// ValueError
    ///     If a bead belongs to another graph, or lacks a finite axis; the
    ///     message names its handle.
    fn axes<'py>(
        slf: &Bound<'py, Self>,
        beads: &Bound<'py, PyAny>,
    ) -> PyResult<Bound<'py, PyArray2<f64>>> {
        let ids = Leaf::of(slf.as_any())?.own_nodes(beads)?;
        let axes = slf
            .try_borrow()?
            .inner
            .axes(&ids)
            .map_err(molrs_error_to_pyerr)?;
        Ok(Array2::from_shape_fn((axes.len(), 3), |(row, c)| axes[row][c]).into_pyarray(slf.py()))
    }

    /// The ``bead_type`` of each of ``beads``, in the listed order.
    ///
    /// Raises
    /// ------
    /// ValueError
    ///     If a bead belongs to another graph.
    fn bead_types(slf: &Bound<'_, Self>, beads: &Bound<'_, PyAny>) -> PyResult<Vec<String>> {
        let ids = Leaf::of(slf.as_any())?.own_nodes(beads)?;
        slf.try_borrow()?
            .inner
            .bead_types(&ids)
            .map_err(molrs_error_to_pyerr)
    }

    // ---- composition the leaves do not share ----

    /// Merge ``other`` into this graph and return its beads, now in this
    /// graph, in ``other``'s former row order. ``other`` is left empty; its
    /// views go stale. Memberships come along: both graphs' members must
    /// name one member world.
    ///
    /// Raises
    /// ------
    /// ValueError
    ///     If ``other`` is this graph, the two graphs' memberships name
    ///     different source graphs, or a property of ``other`` contradicts
    ///     the type this graph holds for that key; this graph is then
    ///     unchanged.
    fn merge(slf: &Bound<'_, Self>, other: &Bound<'_, Self>) -> PyResult<PyRefs> {
        let py = slf.py();
        if slf.is(other) {
            return Err(PyValueError::new_err("a graph cannot merge itself"));
        }
        let world = {
            let this = slf.try_borrow()?;
            let that = other.try_borrow()?;
            if !that.has_members() {
                this.member_world.as_ref().map(|w| w.clone_ref(py))
            } else {
                match (&this.member_world, &that.member_world) {
                    (Some(a), Some(b)) if a.bind(py).is(b.bind(py)) => Some(a.clone_ref(py)),
                    (None, Some(b)) if !this.has_members() => Some(b.clone_ref(py)),
                    // Raw memberships on both sides (read from frames).
                    (None, None) => None,
                    _ => {
                        return Err(PyValueError::new_err(
                            "the two graphs' bead memberships name different source graphs",
                        ));
                    }
                }
            }
        };
        let taken = {
            let mut that = other.try_borrow_mut()?;
            that.views.clear();
            std::mem::take(&mut that.inner)
        };
        let order: Vec<NodeId> = taken.bead_ids().collect();
        let map = {
            let mut this = slf.try_borrow_mut()?;
            let map = this.inner.merge(taken).map_err(molrs_error_to_pyerr)?;
            this.member_world = world;
            map
        };
        other.try_borrow_mut()?.member_world = None;
        let added = order.iter().map(|old| u64::from(map[old])).collect();
        Ok(Leaf::of(slf.as_any())?.node_refs(added))
    }

    /// Radius ball around bead ``centers`` (views or a ``Refs``) over CG
    /// bonds.
    fn extract_subgraph(
        slf: &Bound<'_, Self>,
        centers: &Bound<'_, PyAny>,
        radius: i64,
    ) -> PyResult<PyExtractedSubgraph> {
        let ids = Leaf::of(slf.as_any())?.own_nodes(centers)?;
        let ext = slf
            .try_borrow()?
            .inner
            .extract_subgraph(&ids, radius)
            .map_err(molrs_error_to_pyerr)?;
        PyExtractedSubgraph::from_coarsegrain(slf, ext)
    }

    /// Build a `CoarseGrain` from a frame holding **one molecule**.
    ///
    /// Every ``atoms`` row becomes a bead and every ``bonds`` row a CG bond
    /// (``atomi`` / ``atomj`` are 0-based ``atoms`` rows); a ``members``
    /// block, when present, is read back as bead membership (raw handles of
    /// no graph: :meth:`bead_members` refuses them). ``bead_type`` comes from
    /// the first column present: ``bead_type``, then ``type``, then
    /// ``type_id`` (rendered in decimal). Other blocks are ignored.
    ///
    /// Raises
    /// ------
    /// ValueError
    ///     If the frame has no ``atoms`` rows; if ``mol_id`` holds a null row,
    ///     is not an unsigned-int column, or holds more than one distinct
    ///     value (the message points at ``Frame.subset``); if none of
    ///     ``bead_type`` / ``type`` / ``type_id`` is present, or the chosen one
    ///     has a null row or the wrong dtype; if a bond row names no bead
    ///     row; or if the ``members`` block lacks an unsigned-int ``ibead`` or
    ///     ``atom`` column, names no bead row, or repeats an
    ///     ``(ibead, atom)`` pair.
    #[staticmethod]
    fn from_frame(py: Python<'_>, frame: &PyFrame) -> PyResult<Py<PyCoarseGrain>> {
        let core = frame.clone_core_frame()?;
        let inner = CoarseGrain::from_frame(&core).map_err(molrs_error_to_pyerr)?;
        PyCoarseGrain::from_core(py, inner)
    }

    // ---- pickling ----

    fn __reduce__<'py>(slf: &Bound<'py, Self>) -> PyResult<Bound<'py, PyTuple>> {
        let py = slf.py();
        let this = slf.try_borrow()?;
        let world = this.member_world.as_ref().map(|w| w.bind(py).clone());
        // Memberships by bead row; each member by its row in the member
        // world, or as the raw handle when there is none.
        let member_rows = match &world {
            Some(world) => Some(
                Leaf::of(world)?
                    .node_ids()?
                    .into_iter()
                    .enumerate()
                    .map(|(row, id)| (id, row as u64))
                    .collect::<HashMap<u64, u64>>(),
            ),
            None => None,
        };
        let mut memberships: Vec<(usize, Vec<u64>)> = Vec::new();
        for (row, bead) in this.inner.bead_ids().enumerate() {
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
                                "bead member {atom} is not a node of the member world"
                            ))
                        })
                    })
                    .collect::<PyResult<Vec<u64>>>()?,
                None => members.to_vec(),
            };
            memberships.push((row, members));
        }
        let kinds = ["bonds".to_owned(), "ports".to_owned()];
        let state = (
            this.props.bind(py).clone(),
            dump_leaf(py, &this.inner, &kinds)?,
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
            Some(world) => Some(Leaf::of(world)?.node_ids()?),
            None => None,
        };
        let mut inner = load_coarsegrain(&content)?;
        let beads: Vec<NodeId> = inner.bead_ids().collect();
        for (row, members) in memberships {
            let bead = *beads
                .get(row)
                .ok_or_else(|| PyValueError::new_err(format!("no bead at row {row}")))?;
            let members = match &world_handles {
                Some(handles) => members
                    .iter()
                    .map(|&member| {
                        handles.get(member as usize).copied().ok_or_else(|| {
                            PyValueError::new_err(format!("no member-world node at row {member}"))
                        })
                    })
                    .collect::<PyResult<Vec<u64>>>()?,
                None => members,
            };
            inner
                .set_bead_members(bead, members)
                .map_err(molrs_error_to_pyerr)?;
        }
        let mut this = slf.try_borrow_mut()?;
        this.inner = inner;
        this.props = props.unbind();
        this.member_world = world.map(Bound::unbind);
        this.views.clear();
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
}

leaf_impl!(PyCoarseGrain, "CoarseGrain", "bead");

impl PyCoarseGrain {
    /// A new `CoarseGrain` holding `inner`, with no props and no member
    /// world (see [`PyAtomistic::from_core`]).
    pub(crate) fn from_core(py: Python<'_>, inner: CoarseGrain) -> PyResult<Py<PyCoarseGrain>> {
        Self::with_member_world(py, inner, None)
    }

    /// A new `CoarseGrain` holding `inner`, whose memberships name nodes of
    /// `world` (the graph a coarsening read).
    pub(crate) fn with_member_world(
        py: Python<'_>,
        inner: CoarseGrain,
        world: Option<Py<PyAny>>,
    ) -> PyResult<Py<PyCoarseGrain>> {
        Py::new(
            py,
            Self {
                inner,
                props: PyDict::new(py).unbind(),
                views: ViewCache::default(),
                member_world: world,
            },
        )
    }

    /// A new `CoarseGrain` holding `inner`, derived from this one: a copy of
    /// its props and the same member world (see [`PyAtomistic::derive`]).
    pub(crate) fn derive(&self, py: Python<'_>, inner: CoarseGrain) -> PyResult<Py<PyCoarseGrain>> {
        Py::new(
            py,
            Self {
                inner,
                props: self.props.bind(py).copy()?.unbind(),
                views: ViewCache::default(),
                member_world: self.member_world.as_ref().map(|w| w.clone_ref(py)),
            },
        )
    }

    /// Borrow the held core [`CoarseGrain`] (for the bead-pattern matcher,
    /// which reads the pattern and the target graph).
    pub(crate) fn core(&self) -> &CoarseGrain {
        &self.inner
    }

    /// Mutably borrow the held core [`CoarseGrain`], for an in-place edit
    /// through its typed API.
    pub(crate) fn core_mut(&mut self) -> &mut CoarseGrain {
        &mut self.inner
    }
}
