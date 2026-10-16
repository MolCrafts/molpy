//! Live views over the leaf graphs (`molrs.core.Atomistic`, `molrs.core.CoarseGrain`).
//!
//! A view is a `(world, handle)` pair and nothing else: every read and write
//! goes through the owning graph's typed core API ([`LeafGraph`]), which stays
//! the only holder of data and the only judge of its invariants. The graph
//! interns its views weakly ([`ViewCache`]), so while a view is alive the same
//! handle always answers with the same object (`graph.atoms[0] is atom`).
//!
//! - [`PyNodeRef`] (`molrs.core.NodeRef`) and its classes `Atom`, `VirtualSite`,
//!   `DrudeParticle`, `MasslessSite`, `Bead`: one node. The class follows the
//!   graph type and, for an atom, its stored ``vsite``.
//! - [`PyRelationRef`] (`molrs.core.RelationRef`) and `Bond`, `Angle`, `Dihedral`,
//!   `Improper`, `Port`, `CgBond`: one relation, with its interned endpoints.
//!   The class follows the graph type and the relation kind; a term block
//!   (`cmaps`, `exclusions`, …) has plain `RelationRef` views.
//! - [`PyRefs`] (`molrs.core.Refs`): an ordered handle list of one kind, read as a
//!   sequence of views or, by field name, as a column.
//!
//! A view's mapping protocol is the one door to an entity's fields: read
//! ``view[key]``, write ``view[key] = value``, clear ``del view[key]``.

use std::collections::HashMap;

use pyo3::exceptions::{PyIndexError, PyKeyError, PyTypeError, PyValueError};
use pyo3::prelude::*;
use pyo3::types::{
    PyDict, PyList, PySlice, PyString, PyTuple, PyType, PyWeakrefMethods, PyWeakrefReference,
};
use pyo3::{PyTraverseError, PyVisit, intern};

use molrs::core::keys::{ATOMIC_NUMBER, VSITE};
use molrs::core::{
    Atomistic, CoarseGrain, Element, EntityCell, EntityColumn, EntityTable, MolRsError, NodeId,
    Port, PropValue, Props, RelationId,
};

use super::molgraph::{PyAtomistic, PyCoarseGrain, cell_to_py, prop_to_py, py_to_prop};
use crate::core::schema::extract_column_key;
use crate::error::molrs_error_to_pyerr;

// ---------------------------------------------------------------------------
// The typed core API a view reads and writes through
// ---------------------------------------------------------------------------

/// A relation kind of a leaf graph: the four typed topology blocks, the ports,
/// or (on an `Atomistic`) a term block by name.
#[derive(Clone, Debug)]
pub(crate) enum RelKind {
    Bonds,
    Angles,
    Dihedrals,
    Impropers,
    Ports,
    Term(String),
}

impl RelKind {
    /// The kind `name` of an `Atomistic`: a typed block, or a term block whose
    /// validity the core judges on use.
    fn atomistic(name: &str) -> Self {
        match name {
            "bonds" => Self::Bonds,
            "angles" => Self::Angles,
            "dihedrals" => Self::Dihedrals,
            "impropers" => Self::Impropers,
            "ports" => Self::Ports,
            block => Self::Term(block.to_owned()),
        }
    }

    /// The kind `name` of a `CoarseGrain`: its bonds or its ports.
    fn coarsegrain(name: &str) -> PyResult<Self> {
        match name {
            "bonds" => Ok(Self::Bonds),
            "ports" => Ok(Self::Ports),
            other => Err(PyValueError::new_err(format!(
                "a CoarseGrain has no relation kind '{other}': its relations are 'bonds' and 'ports'"
            ))),
        }
    }
}

/// The fields a port reads back as: its descriptor.
fn port_fields(port: &Port) -> Props {
    Props::new()
        .with("port_kind", port.kind.as_str())
        .with("port_label", port.label.as_str())
        .with("port_order", PropValue::Int(port.order.code() as i32))
}

/// A port's descriptor is fixed when it is added.
fn port_is_fixed() -> MolRsError {
    MolRsError::validation(
        "a port's descriptor (port_kind, port_label, port_order) is fixed when it is added: \
         del_port it and add_port the descriptor you want",
    )
}

/// The node- and relation-level operations a view needs, each through the
/// leaf's own typed core API — so every invariant (an atom's valid
/// ``atomic_number``, a bead's string ``bead_type``, a bond's class set only
/// by ``set_bond_class``) is the core's, never re-implemented here.
pub(crate) trait LeafGraph {
    /// The read-only node table (atoms or beads), in row order.
    fn table(&self) -> &EntityTable<NodeId>;
    /// Add a node carrying `props`.
    fn add_node(&mut self, props: Props) -> Result<NodeId, MolRsError>;
    /// Set one field of a node.
    fn set_node(&mut self, id: NodeId, key: &str, value: PropValue) -> Result<(), MolRsError>;
    /// Clear one field of a node.
    fn clear_node(&mut self, id: NodeId, key: &str) -> Result<(), MolRsError>;
    /// The kind `name` names on this leaf.
    fn kind(&self, name: &str) -> PyResult<RelKind>;
    /// Endpoints and fields of one relation.
    fn get_relation(
        &self,
        kind: &RelKind,
        id: RelationId,
    ) -> Result<(Vec<NodeId>, Props), MolRsError>;
    /// Set one field of a relation.
    fn set_relation(
        &mut self,
        kind: &RelKind,
        id: RelationId,
        key: &str,
        value: PropValue,
    ) -> Result<(), MolRsError>;
    /// Clear one field of a relation.
    fn clear_relation(
        &mut self,
        kind: &RelKind,
        id: RelationId,
        key: &str,
    ) -> Result<(), MolRsError>;
    /// Live relation handles of `kind`, in row order.
    fn relation_ids(&self, kind: &RelKind) -> Result<Vec<RelationId>, MolRsError>;
}

impl LeafGraph for Atomistic {
    fn table(&self) -> &EntityTable<NodeId> {
        self.atom_table()
    }
    fn add_node(&mut self, props: Props) -> Result<NodeId, MolRsError> {
        self.add_atom(props)
    }
    fn set_node(&mut self, id: NodeId, key: &str, value: PropValue) -> Result<(), MolRsError> {
        self.set_atom(id, key, value)
    }
    fn clear_node(&mut self, id: NodeId, key: &str) -> Result<(), MolRsError> {
        self.clear_atom(id, key)
    }
    fn kind(&self, name: &str) -> PyResult<RelKind> {
        Ok(RelKind::atomistic(name))
    }
    fn get_relation(
        &self,
        kind: &RelKind,
        id: RelationId,
    ) -> Result<(Vec<NodeId>, Props), MolRsError> {
        let relation = match kind {
            RelKind::Bonds => self.get_bond(id)?,
            RelKind::Angles => self.get_angle(id)?,
            RelKind::Dihedrals => self.get_dihedral(id)?,
            RelKind::Impropers => self.get_improper(id)?,
            RelKind::Term(block) => self.get_term(block, id)?,
            RelKind::Ports => {
                let port = self.get_port(id)?;
                return Ok((vec![port.anchor, port.handle], port_fields(&port)));
            }
        };
        Ok((relation.nodes.to_vec(), relation.props))
    }
    fn set_relation(
        &mut self,
        kind: &RelKind,
        id: RelationId,
        key: &str,
        value: PropValue,
    ) -> Result<(), MolRsError> {
        match kind {
            RelKind::Bonds => self.set_bond(id, key, value),
            RelKind::Angles => self.set_angle(id, key, value),
            RelKind::Dihedrals => self.set_dihedral(id, key, value),
            RelKind::Impropers => self.set_improper(id, key, value),
            RelKind::Term(block) => self.set_term(block, id, key, value),
            RelKind::Ports => Err(port_is_fixed()),
        }
    }
    fn clear_relation(
        &mut self,
        kind: &RelKind,
        id: RelationId,
        key: &str,
    ) -> Result<(), MolRsError> {
        match kind {
            RelKind::Bonds => self.clear_bond(id, key),
            RelKind::Angles => self.clear_angle(id, key),
            RelKind::Dihedrals => self.clear_dihedral(id, key),
            RelKind::Impropers => self.clear_improper(id, key),
            RelKind::Term(block) => self.clear_term(block, id, key),
            RelKind::Ports => Err(port_is_fixed()),
        }
    }
    fn relation_ids(&self, kind: &RelKind) -> Result<Vec<RelationId>, MolRsError> {
        Ok(match kind {
            RelKind::Bonds => self.bond_ids().collect(),
            RelKind::Angles => self.angle_ids().collect(),
            RelKind::Dihedrals => self.dihedral_ids().collect(),
            RelKind::Impropers => self.improper_ids().collect(),
            RelKind::Ports => self.port_ids().collect(),
            RelKind::Term(block) => self.term_ids(block)?.collect(),
        })
    }
}

impl LeafGraph for CoarseGrain {
    fn table(&self) -> &EntityTable<NodeId> {
        self.bead_table()
    }
    fn add_node(&mut self, props: Props) -> Result<NodeId, MolRsError> {
        self.add_bead(props)
    }
    fn set_node(&mut self, id: NodeId, key: &str, value: PropValue) -> Result<(), MolRsError> {
        self.set_bead(id, key, value)
    }
    fn clear_node(&mut self, id: NodeId, key: &str) -> Result<(), MolRsError> {
        self.clear_bead(id, key)
    }
    fn kind(&self, name: &str) -> PyResult<RelKind> {
        RelKind::coarsegrain(name)
    }
    fn get_relation(
        &self,
        kind: &RelKind,
        id: RelationId,
    ) -> Result<(Vec<NodeId>, Props), MolRsError> {
        match kind {
            RelKind::Ports => {
                let port = self.get_port(id)?;
                Ok((vec![port.anchor, port.handle], port_fields(&port)))
            }
            _ => {
                let bond = self.get_bond(id)?;
                Ok((bond.nodes.to_vec(), bond.props))
            }
        }
    }
    fn set_relation(
        &mut self,
        kind: &RelKind,
        id: RelationId,
        key: &str,
        value: PropValue,
    ) -> Result<(), MolRsError> {
        match kind {
            RelKind::Ports => Err(port_is_fixed()),
            _ => self.set_bond(id, key, value),
        }
    }
    fn clear_relation(
        &mut self,
        kind: &RelKind,
        id: RelationId,
        key: &str,
    ) -> Result<(), MolRsError> {
        match kind {
            RelKind::Ports => Err(port_is_fixed()),
            _ => self.clear_bond(id, key),
        }
    }
    fn relation_ids(&self, kind: &RelKind) -> Result<Vec<RelationId>, MolRsError> {
        Ok(match kind {
            RelKind::Ports => self.port_ids().collect(),
            _ => self.bond_ids().collect(),
        })
    }
}

// ---------------------------------------------------------------------------
// Interning
// ---------------------------------------------------------------------------

/// A graph's weak index of its live views, keyed by handle.
///
/// Weak, so a view nobody holds is freed; a dead entry is pruned once the
/// table has doubled since the last sweep.
#[derive(Default)]
pub(crate) struct ViewCache {
    nodes: HashMap<u64, Py<PyWeakrefReference>>,
    relations: HashMap<(String, u64), Py<PyWeakrefReference>>,
    watermark: usize,
}

impl ViewCache {
    /// Entries below which no sweep runs.
    const MIN_SWEEP: usize = 1024;

    fn node<'py>(&self, py: Python<'py>, handle: u64) -> Option<Bound<'py, PyAny>> {
        self.nodes.get(&handle)?.bind(py).upgrade()
    }

    fn relation<'py>(&self, py: Python<'py>, kind: &str, handle: u64) -> Option<Bound<'py, PyAny>> {
        self.relations
            .get(&(kind.to_owned(), handle))?
            .bind(py)
            .upgrade()
    }

    fn keep_node(&mut self, py: Python<'_>, handle: u64, view: &Bound<'_, PyAny>) -> PyResult<()> {
        self.sweep(py);
        self.nodes
            .insert(handle, PyWeakrefReference::new(view)?.unbind());
        Ok(())
    }

    fn keep_relation(
        &mut self,
        py: Python<'_>,
        kind: &str,
        handle: u64,
        view: &Bound<'_, PyAny>,
    ) -> PyResult<()> {
        self.sweep(py);
        self.relations.insert(
            (kind.to_owned(), handle),
            PyWeakrefReference::new(view)?.unbind(),
        );
        Ok(())
    }

    fn sweep(&mut self, py: Python<'_>) {
        let entries = self.nodes.len() + self.relations.len();
        if entries < self.watermark.max(Self::MIN_SWEEP) {
            return;
        }
        self.nodes
            .retain(|_, view| view.bind(py).upgrade().is_some());
        self.relations
            .retain(|_, view| view.bind(py).upgrade().is_some());
        self.watermark = 2 * (self.nodes.len() + self.relations.len());
    }

    /// Forget every view: the graph's storage was replaced, so a cached
    /// handle may now name a different node or relation.
    pub(crate) fn clear(&mut self) {
        self.nodes.clear();
        self.relations.clear();
        self.watermark = 0;
    }
}

// ---------------------------------------------------------------------------
// The leaf a view belongs to
// ---------------------------------------------------------------------------

/// A graph that has views: one of the two leaves.
pub(crate) enum Leaf<'py> {
    Atomistic(Bound<'py, PyAtomistic>),
    CoarseGrain(Bound<'py, PyCoarseGrain>),
}

impl<'py> Leaf<'py> {
    /// The leaf `world` is.
    ///
    /// # Errors
    ///
    /// `TypeError` when `world` is neither an `Atomistic` nor a `CoarseGrain`.
    pub(crate) fn of(world: &Bound<'py, PyAny>) -> PyResult<Self> {
        if let Ok(graph) = world.cast::<PyAtomistic>() {
            return Ok(Self::Atomistic(graph.clone()));
        }
        if let Ok(graph) = world.cast::<PyCoarseGrain>() {
            return Ok(Self::CoarseGrain(graph.clone()));
        }
        Err(PyTypeError::new_err(format!(
            "views live on an Atomistic or a CoarseGrain, not {}",
            world.get_type().name()?
        )))
    }

    fn py(&self) -> Python<'py> {
        self.object().py()
    }

    pub(crate) fn object(&self) -> &Bound<'py, PyAny> {
        match self {
            Self::Atomistic(graph) => graph.as_any(),
            Self::CoarseGrain(graph) => graph.as_any(),
        }
    }

    /// Read the leaf's core graph.
    pub(crate) fn read<R>(&self, f: impl FnOnce(&dyn LeafGraph) -> R) -> PyResult<R> {
        Ok(match self {
            Self::Atomistic(graph) => f(graph.try_borrow()?.core()),
            Self::CoarseGrain(graph) => f(graph.try_borrow()?.core()),
        })
    }

    /// Write the leaf's core graph, through its typed API.
    pub(crate) fn write<R>(&self, f: impl FnOnce(&mut dyn LeafGraph) -> R) -> PyResult<R> {
        Ok(match self {
            Self::Atomistic(graph) => f(graph.try_borrow_mut()?.core_mut()),
            Self::CoarseGrain(graph) => f(graph.try_borrow_mut()?.core_mut()),
        })
    }

    fn cache<R>(&self, f: impl FnOnce(&mut ViewCache) -> R) -> PyResult<R> {
        Ok(match self {
            Self::Atomistic(graph) => f(&mut graph.try_borrow_mut()?.views),
            Self::CoarseGrain(graph) => f(&mut graph.try_borrow_mut()?.views),
        })
    }

    /// The relation kind `name` names on this leaf.
    pub(crate) fn kind(&self, name: &str) -> PyResult<RelKind> {
        self.read(|graph| graph.kind(name))?
    }

    /// Every node handle, in row order.
    pub(crate) fn node_ids(&self) -> PyResult<Vec<u64>> {
        self.read(|graph| graph.table().handles().map(u64::from).collect())
    }

    /// Every handle of relation kind `name`, in row order.
    pub(crate) fn relation_ids(&self, name: &str) -> PyResult<Vec<u64>> {
        let kind = self.kind(name)?;
        self.read(|graph| graph.relation_ids(&kind))?
            .map(|ids| ids.into_iter().map(u64::from).collect())
            .map_err(molrs_error_to_pyerr)
    }

    /// The interned view of node `handle`.
    ///
    /// # Errors
    ///
    /// `ValueError` when `handle` is not a live node of this graph.
    pub(crate) fn node(&self, handle: u64) -> PyResult<Bound<'py, PyAny>> {
        let py = self.py();
        if let Some(view) = self.cache(|cache| cache.node(py, handle))? {
            return Ok(view);
        }
        let class = self.read(|graph| NodeClass::of(self, graph.table(), handle))??;
        let base = PyNodeRef {
            world: self.object().clone().unbind(),
            handle,
        };
        let view = class.instantiate(py, base)?.into_bound(py);
        self.cache(|cache| cache.keep_node(py, handle, &view))??;
        Ok(view)
    }

    /// The interned view of relation `handle` of kind `name`.
    ///
    /// # Errors
    ///
    /// `ValueError` when `name` is no kind of this leaf or `handle` is not a
    /// live relation of it.
    pub(crate) fn relation(&self, name: &str, handle: u64) -> PyResult<Bound<'py, PyAny>> {
        let py = self.py();
        if let Some(view) = self.cache(|cache| cache.relation(py, name, handle))? {
            return Ok(view);
        }
        let kind = self.kind(name)?;
        let (ends, _) = self
            .read(|graph| graph.get_relation(&kind, RelationId::from(handle)))?
            .map_err(molrs_error_to_pyerr)?;
        let endpoints = ends
            .into_iter()
            .map(|end| self.node(u64::from(end)))
            .collect::<PyResult<Vec<_>>>()?;
        let base = PyRelationRef {
            world: self.object().clone().unbind(),
            kind: name.to_owned(),
            handle,
            endpoints: PyTuple::new(py, endpoints)?.unbind(),
        };
        let view = RelationClass::of(self, name)
            .instantiate(py, base)?
            .into_bound(py);
        self.cache(|cache| cache.keep_relation(py, name, handle, &view))??;
        Ok(view)
    }

    /// Every node, in row order.
    pub(crate) fn nodes(&self) -> PyResult<PyRefs> {
        Ok(self.node_refs(self.node_ids()?))
    }

    /// The nodes `handles`, in that order.
    pub(crate) fn node_refs(&self, handles: Vec<u64>) -> PyRefs {
        PyRefs {
            world: self.object().clone().unbind(),
            kind: None,
            handles,
        }
    }

    /// Every relation of kind `name`, in row order.
    pub(crate) fn relations(&self, name: &str) -> PyResult<PyRefs> {
        let handles = self.relation_ids(name)?;
        Ok(PyRefs {
            world: self.object().clone().unbind(),
            kind: Some(name.to_owned()),
            handles,
        })
    }

    /// Add a node carrying `props` through the leaf's checked add (an atom
    /// needs a valid ``atomic_number``, a bead a string ``bead_type``) and
    /// return its view. A refused add leaves the graph unchanged.
    pub(crate) fn create_node(&self, props: Props) -> PyResult<Bound<'py, PyAny>> {
        let id = self
            .write(|graph| graph.add_node(props))?
            .map_err(molrs_error_to_pyerr)?;
        self.node(u64::from(id))
    }

    /// The node of `node`, checked to be a node of this graph.
    ///
    /// A slotmap handle from a foreign graph can alias a live one rather than
    /// fail, so a native writer would happily join the wrong nodes: the
    /// world is compared, not the handle.
    pub(crate) fn own_node(&self, node: &Bound<'py, PyNodeRef>) -> PyResult<NodeId> {
        let node = node.get();
        if !node.world.bind(self.py()).is(self.object()) {
            return Err(PyValueError::new_err("the node belongs to another graph"));
        }
        Ok(NodeId::from(node.handle))
    }

    /// The nodes of `nodes` — a `Refs` of this graph's nodes or any iterable
    /// of its node views — in order.
    pub(crate) fn own_nodes(&self, nodes: &Bound<'py, PyAny>) -> PyResult<Vec<NodeId>> {
        if let Ok(refs) = nodes.cast::<PyRefs>() {
            let refs = refs.get();
            if refs.kind.is_none() && refs.world.bind(self.py()).is(self.object()) {
                return Ok(refs.handles.iter().copied().map(NodeId::from).collect());
            }
        }
        nodes
            .try_iter()?
            .map(|node| {
                let node = node?;
                let node = node.cast::<PyNodeRef>().map_err(|_| {
                    PyTypeError::new_err("expected node views (Atom / Bead) of this graph")
                })?;
                self.own_node(node)
            })
            .collect()
    }

    /// The relation of `relation`, checked to be a `name` relation of this
    /// graph.
    pub(crate) fn own_relation(
        &self,
        relation: &Bound<'py, PyRelationRef>,
        name: &str,
    ) -> PyResult<RelationId> {
        let relation = relation.get();
        if !relation.world.bind(self.py()).is(self.object()) {
            return Err(PyValueError::new_err(
                "the relation belongs to another graph",
            ));
        }
        if relation.kind != name {
            return Err(PyValueError::new_err(format!(
                "expected a '{name}' relation, got a '{}' one",
                relation.kind
            )));
        }
        Ok(RelationId::from(relation.handle))
    }

    /// The relations of `relations` — a `Refs` of this graph's `name`
    /// relations or any iterable of its relation views of that kind — in
    /// order.
    pub(crate) fn own_relations(
        &self,
        relations: &Bound<'py, PyAny>,
        name: &str,
    ) -> PyResult<Vec<RelationId>> {
        if let Ok(refs) = relations.cast::<PyRefs>() {
            let refs = refs.get();
            if refs.kind.as_deref() == Some(name) && refs.world.bind(self.py()).is(self.object()) {
                return Ok(refs.handles.iter().copied().map(RelationId::from).collect());
            }
        }
        relations
            .try_iter()?
            .map(|relation| {
                let relation = relation?;
                let relation = relation.cast::<PyRelationRef>().map_err(|_| {
                    PyTypeError::new_err(format!("expected '{name}' relation views of this graph"))
                })?;
                self.own_relation(relation, name)
            })
            .collect()
    }
}

// ---------------------------------------------------------------------------
// Which class a view is
// ---------------------------------------------------------------------------

/// The class of a node view.
enum NodeClass {
    Atom,
    VirtualSite,
    DrudeParticle,
    MasslessSite,
    Bead,
}

impl NodeClass {
    /// An atom's class follows its stored ``vsite``; a bead is a `Bead`.
    fn of(leaf: &Leaf<'_>, table: &EntityTable<NodeId>, handle: u64) -> PyResult<Self> {
        let id = NodeId::from(handle);
        if !table.contains(id) {
            return Err(PyValueError::new_err(format!(
                "cannot bind a node view to stale handle {handle}"
            )));
        }
        Ok(match leaf {
            Leaf::CoarseGrain(_) => Self::Bead,
            Leaf::Atomistic(_) => match table.value(id, VSITE) {
                None => Self::Atom,
                Some(EntityCell::Str("drude")) => Self::DrudeParticle,
                Some(EntityCell::Str("massless")) => Self::MasslessSite,
                Some(_) => Self::VirtualSite,
            },
        })
    }

    fn instantiate(self, py: Python<'_>, base: PyNodeRef) -> PyResult<Py<PyAny>> {
        let init = PyClassInitializer::from(base);
        Ok(match self {
            Self::Bead => Py::new(py, init.add_subclass(PyBead {}))?.into_any(),
            Self::Atom => Py::new(py, init.add_subclass(PyAtom {}))?.into_any(),
            Self::VirtualSite => {
                let init = init.add_subclass(PyAtom {}).add_subclass(PyVirtualSite {});
                Py::new(py, init)?.into_any()
            }
            Self::DrudeParticle => {
                let init = init
                    .add_subclass(PyAtom {})
                    .add_subclass(PyVirtualSite {})
                    .add_subclass(PyDrudeParticle {});
                Py::new(py, init)?.into_any()
            }
            Self::MasslessSite => {
                let init = init
                    .add_subclass(PyAtom {})
                    .add_subclass(PyVirtualSite {})
                    .add_subclass(PyMasslessSite {});
                Py::new(py, init)?.into_any()
            }
        })
    }
}

/// The ``vsite`` an ``add_atom(kind=cls)`` stamps.
///
/// # Errors
///
/// `TypeError` when `cls` is not `VirtualSite`, `DrudeParticle` or
/// `MasslessSite`.
pub(crate) fn vsite_of(cls: &Bound<'_, PyType>) -> PyResult<&'static str> {
    let py = cls.py();
    if cls.is(py.get_type::<PyDrudeParticle>()) {
        Ok("drude")
    } else if cls.is(py.get_type::<PyMasslessSite>()) {
        Ok("massless")
    } else if cls.is(py.get_type::<PyVirtualSite>()) {
        Ok("virtual")
    } else {
        Err(PyTypeError::new_err(format!(
            "kind must be VirtualSite, DrudeParticle or MasslessSite, not {}",
            cls.name()?
        )))
    }
}

/// The class of a relation view.
pub(crate) enum RelationClass {
    /// A term block: a plain `RelationRef`.
    RelationRef,
    Bond,
    Angle,
    Dihedral,
    Improper,
    Port,
    CgBond,
}

impl RelationClass {
    fn of(leaf: &Leaf<'_>, kind: &str) -> Self {
        match (leaf, kind) {
            (_, "ports") => Self::Port,
            (Leaf::Atomistic(_), "bonds") => Self::Bond,
            (Leaf::Atomistic(_), "angles") => Self::Angle,
            (Leaf::Atomistic(_), "dihedrals") => Self::Dihedral,
            (Leaf::Atomistic(_), "impropers") => Self::Improper,
            (Leaf::CoarseGrain(_), "bonds") => Self::CgBond,
            _ => Self::RelationRef,
        }
    }

    /// The `Atomistic` relation kind whose view class is exactly `cls`
    /// (``Bond`` → ``"bonds"``, …), if any.
    pub(crate) fn atomistic_kind(cls: &Bound<'_, PyAny>) -> Option<&'static str> {
        let py = cls.py();
        [
            (py.get_type::<PyBond>(), "bonds"),
            (py.get_type::<PyAngle>(), "angles"),
            (py.get_type::<PyDihedral>(), "dihedrals"),
            (py.get_type::<PyImproper>(), "impropers"),
            (py.get_type::<PyPort>(), "ports"),
        ]
        .into_iter()
        .find(|(class, _)| class.is(cls))
        .map(|(_, kind)| kind)
    }

    fn instantiate(self, py: Python<'_>, base: PyRelationRef) -> PyResult<Py<PyAny>> {
        let init = PyClassInitializer::from(base);
        Ok(match self {
            Self::RelationRef => Py::new(py, init)?.into_any(),
            Self::Bond => Py::new(py, init.add_subclass(PyBond {}))?.into_any(),
            Self::Angle => Py::new(py, init.add_subclass(PyAngle {}))?.into_any(),
            Self::Dihedral => Py::new(py, init.add_subclass(PyDihedral {}))?.into_any(),
            Self::Improper => {
                let init = init.add_subclass(PyDihedral {}).add_subclass(PyImproper {});
                Py::new(py, init)?.into_any()
            }
            Self::Port => Py::new(py, init.add_subclass(PyPort {}))?.into_any(),
            Self::CgBond => Py::new(py, init.add_subclass(PyCgBond {}))?.into_any(),
        })
    }
}

// ---------------------------------------------------------------------------
// Field access: the mapping protocol both view kinds share
// ---------------------------------------------------------------------------

/// The fields of one node or one relation.
trait Fields {
    fn keys(&self) -> PyResult<Vec<String>>;
    fn get(&self, key: &str) -> PyResult<Option<Py<PyAny>>>;
    fn set(&self, key: &str, value: PropValue) -> PyResult<()>;
    fn clear(&self, key: &str) -> PyResult<()>;
}

struct NodeFields<'a, 'py> {
    leaf: &'a Leaf<'py>,
    handle: u64,
}

impl Fields for NodeFields<'_, '_> {
    fn keys(&self) -> PyResult<Vec<String>> {
        let id = NodeId::from(self.handle);
        self.leaf.read(|graph| -> PyResult<Vec<String>> {
            let table = graph.table();
            if !table.contains(id) {
                return Err(stale_node(self.handle));
            }
            Ok(table.row_cells(id).map(|(key, _)| key.to_owned()).collect())
        })?
    }

    fn get(&self, key: &str) -> PyResult<Option<Py<PyAny>>> {
        let py = self.leaf.py();
        let id = NodeId::from(self.handle);
        self.leaf.read(|graph| -> PyResult<Option<Py<PyAny>>> {
            let table = graph.table();
            if !table.contains(id) {
                return Err(stale_node(self.handle));
            }
            table
                .value(id, key)
                .map(|cell| cell_to_py(py, cell))
                .transpose()
        })?
    }

    fn set(&self, key: &str, value: PropValue) -> PyResult<()> {
        self.leaf
            .write(|graph| graph.set_node(NodeId::from(self.handle), key, value))?
            .map_err(molrs_error_to_pyerr)
    }

    fn clear(&self, key: &str) -> PyResult<()> {
        self.leaf
            .write(|graph| graph.clear_node(NodeId::from(self.handle), key))?
            .map_err(molrs_error_to_pyerr)
    }
}

fn stale_node(handle: u64) -> PyErr {
    PyValueError::new_err(format!("node {handle} was deleted: its view is stale"))
}

struct RelationFields<'a, 'py> {
    leaf: &'a Leaf<'py>,
    kind: &'a str,
    handle: u64,
}

impl RelationFields<'_, '_> {
    fn props(&self) -> PyResult<Props> {
        let kind = self.leaf.kind(self.kind)?;
        let (_, props) = self
            .leaf
            .read(|graph| graph.get_relation(&kind, RelationId::from(self.handle)))?
            .map_err(molrs_error_to_pyerr)?;
        Ok(props)
    }
}

impl Fields for RelationFields<'_, '_> {
    fn keys(&self) -> PyResult<Vec<String>> {
        Ok(self.props()?.keys().map(str::to_owned).collect())
    }

    fn get(&self, key: &str) -> PyResult<Option<Py<PyAny>>> {
        let py = self.leaf.py();
        self.props()?
            .get(key)
            .map(|value| prop_to_py(py, value))
            .transpose()
    }

    fn set(&self, key: &str, value: PropValue) -> PyResult<()> {
        let kind = self.leaf.kind(self.kind)?;
        self.leaf
            .write(|graph| graph.set_relation(&kind, RelationId::from(self.handle), key, value))?
            .map_err(molrs_error_to_pyerr)
    }

    fn clear(&self, key: &str) -> PyResult<()> {
        let kind = self.leaf.kind(self.kind)?;
        self.leaf
            .write(|graph| graph.clear_relation(&kind, RelationId::from(self.handle), key))?
            .map_err(molrs_error_to_pyerr)
    }
}

/// A field value: ``bool`` / ``int`` / ``float`` / ``str``. ``None`` is not a
/// value, and its refusal names the one door that clears a field.
fn field_value(key: &str, value: &Bound<'_, PyAny>) -> PyResult<PropValue> {
    if value.is_none() {
        return Err(PyTypeError::new_err(format!(
            "None is not a field value; clear '{key}' with `del view[\"{key}\"]`"
        )));
    }
    py_to_prop(value)
}

/// The mapping protocol of a view, over its [`Fields`]: a field name (or a
/// `molrs.core.keys.Key`) reads, writes and deletes one field; a tuple of names
/// reads or writes several at once (``atom["x", "y", "z"]``).
macro_rules! field_mapping_impl {
    ($ty:ty) => {
        #[pymethods]
        impl $ty {
            fn __getitem__(slf: &Bound<'_, Self>, key: &Bound<'_, PyAny>) -> PyResult<Py<PyAny>> {
                let py = slf.py();
                let leaf = slf.get().leaf(py)?;
                let fields = slf.get().fields(&leaf);
                if let Ok(names) = key.cast::<PyTuple>() {
                    let values = names
                        .iter()
                        .map(|name| field_or_key_error(&fields, &name))
                        .collect::<PyResult<Vec<_>>>()?;
                    return Ok(PyList::new(py, values)?.into_any().unbind());
                }
                field_or_key_error(&fields, key)
            }

            fn __setitem__(
                slf: &Bound<'_, Self>,
                key: &Bound<'_, PyAny>,
                value: &Bound<'_, PyAny>,
            ) -> PyResult<()> {
                let leaf = slf.get().leaf(slf.py())?;
                let fields = slf.get().fields(&leaf);
                if let Ok(names) = key.cast::<PyTuple>() {
                    let values: Vec<Bound<'_, PyAny>> =
                        value.try_iter()?.collect::<PyResult<_>>()?;
                    if values.len() != names.len() {
                        return Err(PyValueError::new_err(format!(
                            "assigning to {} needs {} values, got {}",
                            names.repr()?,
                            names.len(),
                            values.len()
                        )));
                    }
                    let mut pairs = Vec::with_capacity(values.len());
                    for (name, value) in names.iter().zip(&values) {
                        let name = extract_column_key(&name)?;
                        let value = field_value(&name, value)?;
                        pairs.push((name, value));
                    }
                    for (name, value) in pairs {
                        fields.set(&name, value)?;
                    }
                    return Ok(());
                }
                let name = extract_column_key(key)?;
                let value = field_value(&name, value)?;
                fields.set(&name, value)
            }

            fn __delitem__(slf: &Bound<'_, Self>, key: &Bound<'_, PyAny>) -> PyResult<()> {
                let leaf = slf.get().leaf(slf.py())?;
                let fields = slf.get().fields(&leaf);
                let name = extract_column_key(key)?;
                if fields.get(&name)?.is_none() {
                    return Err(PyKeyError::new_err(name));
                }
                fields.clear(&name)
            }

            fn __contains__(slf: &Bound<'_, Self>, key: &Bound<'_, PyAny>) -> PyResult<bool> {
                let leaf = slf.get().leaf(slf.py())?;
                let fields = slf.get().fields(&leaf);
                if let Ok(names) = key.cast::<PyTuple>() {
                    for name in names.iter() {
                        let Ok(name) = extract_column_key(&name) else {
                            return Ok(false);
                        };
                        if fields.get(&name)?.is_none() {
                            return Ok(false);
                        }
                    }
                    return Ok(true);
                }
                match extract_column_key(key) {
                    Ok(name) => Ok(fields.get(&name)?.is_some()),
                    Err(_) => Ok(false),
                }
            }

            fn __iter__(slf: &Bound<'_, Self>) -> PyResult<Py<PyAny>> {
                let py = slf.py();
                let leaf = slf.get().leaf(py)?;
                let keys = slf.get().fields(&leaf).keys()?;
                Ok(PyList::new(py, keys)?.try_iter()?.into_any().unbind())
            }

            fn __len__(slf: &Bound<'_, Self>) -> PyResult<usize> {
                let leaf = slf.get().leaf(slf.py())?;
                Ok(slf.get().fields(&leaf).keys()?.len())
            }

            /// The field names, in column order.
            fn keys(slf: &Bound<'_, Self>) -> PyResult<Vec<String>> {
                let leaf = slf.get().leaf(slf.py())?;
                slf.get().fields(&leaf).keys()
            }

            /// The field values, in :meth:`keys` order.
            fn values(slf: &Bound<'_, Self>) -> PyResult<Vec<Py<PyAny>>> {
                Ok(Self::items(slf)?
                    .into_iter()
                    .map(|(_, value)| value)
                    .collect())
            }

            /// ``(name, value)`` pairs, in :meth:`keys` order.
            fn items(slf: &Bound<'_, Self>) -> PyResult<Vec<(String, Py<PyAny>)>> {
                let py = slf.py();
                let leaf = slf.get().leaf(py)?;
                let fields = slf.get().fields(&leaf);
                fields
                    .keys()?
                    .into_iter()
                    .map(|key| {
                        let value = fields
                            .get(&key)?
                            .ok_or_else(|| PyKeyError::new_err(key.clone()))?;
                        Ok((key, value))
                    })
                    .collect()
            }

            /// The field `key`, or `default` when it is not set.
            #[pyo3(signature = (key, default = None))]
            fn get(
                slf: &Bound<'_, Self>,
                key: &Bound<'_, PyAny>,
                default: Option<Py<PyAny>>,
            ) -> PyResult<Py<PyAny>> {
                let py = slf.py();
                let leaf = slf.get().leaf(py)?;
                let value = match extract_column_key(key) {
                    Ok(name) => slf.get().fields(&leaf).get(&name)?,
                    Err(_) => None,
                };
                Ok(value.or(default).unwrap_or_else(|| py.None()))
            }

            /// Write several fields, as ``dict.update`` takes them.
            ///
            /// Every value is checked before any is written.
            #[pyo3(signature = (*args, **kwargs))]
            fn update(
                slf: &Bound<'_, Self>,
                args: &Bound<'_, PyTuple>,
                kwargs: Option<&Bound<'_, PyDict>>,
            ) -> PyResult<()> {
                let py = slf.py();
                let pairs = py.get_type::<PyDict>().call(args, kwargs)?;
                let pairs = pairs.cast::<PyDict>()?;
                let leaf = slf.get().leaf(py)?;
                let fields = slf.get().fields(&leaf);
                let mut checked = Vec::with_capacity(pairs.len());
                for (key, value) in pairs.iter() {
                    let name = extract_column_key(&key)?;
                    let value = field_value(&name, &value)?;
                    checked.push((name, value));
                }
                for (name, value) in checked {
                    fields.set(&name, value)?;
                }
                Ok(())
            }
        }
    };
}

fn field_or_key_error(fields: &impl Fields, key: &Bound<'_, PyAny>) -> PyResult<Py<PyAny>> {
    let name = extract_column_key(key)?;
    fields.get(&name)?.ok_or_else(|| PyKeyError::new_err(name))
}

// ---------------------------------------------------------------------------
// Node views
// ---------------------------------------------------------------------------

/// A live view of one node of a graph.
///
/// A view is made by the graph (``graph.add_atom(...)``, ``graph.atoms[i]``),
/// never constructed directly. It reads and writes the node's fields as a
/// mapping: ``atom["x"]``, ``atom["x", "y", "z"] = (0.0, 0.0, 1.0)``,
/// ``del atom["charge"]``, ``"charge" in atom``. Deleting the node leaves the
/// view stale: its reads raise.
#[pyclass(module = "molrs.core", name = "NodeRef", frozen, subclass, weakref)]
pub struct PyNodeRef {
    world: Py<PyAny>,
    handle: u64,
}

impl PyNodeRef {
    fn leaf<'py>(&self, py: Python<'py>) -> PyResult<Leaf<'py>> {
        Leaf::of(self.world.bind(py))
    }

    fn fields<'a, 'py>(&self, leaf: &'a Leaf<'py>) -> NodeFields<'a, 'py> {
        NodeFields {
            leaf,
            handle: self.handle,
        }
    }

    pub(crate) fn world<'py>(&self, py: Python<'py>) -> &Bound<'py, PyAny> {
        self.world.bind(py)
    }

    pub(crate) fn handle(&self) -> u64 {
        self.handle
    }
}

#[pymethods]
impl PyNodeRef {
    /// The graph this node belongs to.
    #[getter(world)]
    fn world_getter(&self, py: Python<'_>) -> Py<PyAny> {
        self.world.clone_ref(py)
    }

    /// The node's stable handle in :attr:`world`.
    #[getter(handle)]
    fn handle_getter(&self) -> u64 {
        self.handle
    }

    fn __repr__(slf: &Bound<'_, Self>) -> PyResult<String> {
        let py = slf.py();
        let class = slf.get_type().name()?;
        let this = slf.get();
        let leaf = this.leaf(py)?;
        let id = NodeId::from(this.handle);
        let label = match leaf {
            // An atom is labelled by its species, the symbol its atomic
            // number names (`*` for the wildcard 0).
            Leaf::Atomistic(_) => {
                leaf.read(|graph| match graph.table().value(id, ATOMIC_NUMBER) {
                    Some(EntityCell::I32(0)) => Some("*".to_owned()),
                    Some(EntityCell::I32(z)) => u8::try_from(z)
                        .ok()
                        .and_then(Element::by_number)
                        .map(|element| element.symbol().to_owned()),
                    _ => None,
                })?
            }
            Leaf::CoarseGrain(_) => leaf.read(|graph| {
                ["bead_type", "name"].into_iter().find_map(|key| {
                    match graph.table().value(id, key) {
                        Some(EntityCell::Str(s)) => Some(s.to_owned()),
                        _ => None,
                    }
                })
            })?,
        };
        let label = label.unwrap_or_else(|| "?".to_owned());
        Ok(format!("<{class} {}: {label}>", this.handle))
    }

    fn __reduce__<'py>(
        slf: &Bound<'py, Self>,
    ) -> PyResult<(Bound<'py, PyAny>, Bound<'py, PyTuple>)> {
        let py = slf.py();
        let this = slf.get();
        let row = this
            .leaf(py)?
            .read(|graph| graph.table().row(NodeId::from(this.handle)))?
            .ok_or_else(|| PyValueError::new_err("cannot pickle a view of a deleted node"))?;
        let restore = py.get_type::<Self>().getattr(intern!(py, "_restore"))?;
        Ok((
            restore,
            PyTuple::new(
                py,
                [
                    this.world.bind(py).clone(),
                    row.into_pyobject(py)?.into_any(),
                ],
            )?,
        ))
    }

    /// Unpickle: the view of the node at `row` of `world`.
    #[classmethod]
    fn _restore<'py>(
        _cls: &Bound<'py, PyType>,
        world: &Bound<'py, PyAny>,
        row: usize,
    ) -> PyResult<Bound<'py, PyAny>> {
        let leaf = Leaf::of(world)?;
        let handle = leaf
            .node_ids()?
            .get(row)
            .copied()
            .ok_or_else(|| PyValueError::new_err(format!("no node at row {row}")))?;
        leaf.node(handle)
    }

    fn __traverse__(&self, visit: PyVisit<'_>) -> Result<(), PyTraverseError> {
        visit.call(&self.world)
    }
}
field_mapping_impl!(PyNodeRef);

/// A node of an :class:`Atomistic`.
#[pyclass(module = "molrs.core", name = "Atom", extends = PyNodeRef, frozen, subclass)]
pub struct PyAtom {}

/// An atom carrying a ``vsite`` field: a site that is not a nucleus.
#[pyclass(module = "molrs.core", name = "VirtualSite", extends = PyAtom, frozen, subclass)]
pub struct PyVirtualSite {}

/// A Drude shell (``vsite == "drude"``).
#[pyclass(module = "molrs.core", name = "DrudeParticle", extends = PyVirtualSite, frozen, subclass)]
pub struct PyDrudeParticle {}

/// A massless site (``vsite == "massless"``), e.g. the TIP4P M site.
#[pyclass(module = "molrs.core", name = "MasslessSite", extends = PyVirtualSite, frozen, subclass)]
pub struct PyMasslessSite {}

/// A node of a :class:`CoarseGrain`.
#[pyclass(module = "molrs.core", name = "Bead", extends = PyNodeRef, frozen, subclass)]
pub struct PyBead {}

// ---------------------------------------------------------------------------
// Relation views
// ---------------------------------------------------------------------------

/// A live view of one relation (a bond, an angle, a port, a term, …) of a
/// graph.
///
/// Made by the graph, never constructed directly. ``endpoints`` are the
/// interned node views, in order; the relation's own fields read and write
/// as a mapping (``bond["order"]``).
#[pyclass(module = "molrs.core", name = "RelationRef", frozen, subclass, weakref)]
pub struct PyRelationRef {
    world: Py<PyAny>,
    kind: String,
    handle: u64,
    endpoints: Py<PyTuple>,
}

impl PyRelationRef {
    fn leaf<'py>(&self, py: Python<'py>) -> PyResult<Leaf<'py>> {
        Leaf::of(self.world.bind(py))
    }

    fn fields<'a, 'py>(&'a self, leaf: &'a Leaf<'py>) -> RelationFields<'a, 'py> {
        RelationFields {
            leaf,
            kind: &self.kind,
            handle: self.handle,
        }
    }

    fn endpoint<'py>(&self, py: Python<'py>, index: usize) -> PyResult<Bound<'py, PyAny>> {
        self.endpoints.bind(py).get_item(index)
    }

    /// The relation kind (``"bonds"``, a term block name, …).
    pub(crate) fn kind_name(&self) -> &str {
        &self.kind
    }
}

#[pymethods]
impl PyRelationRef {
    /// The graph this relation belongs to.
    #[getter]
    fn world(&self, py: Python<'_>) -> Py<PyAny> {
        self.world.clone_ref(py)
    }

    /// The relation kind (``"bonds"``, ``"angles"``, a term block, …).
    #[getter]
    fn kind(&self) -> &str {
        &self.kind
    }

    /// The relation's stable handle within its kind.
    #[getter]
    fn handle(&self) -> u64 {
        self.handle
    }

    /// The endpoint node views, in order.
    #[getter]
    fn endpoints(&self, py: Python<'_>) -> Py<PyTuple> {
        self.endpoints.clone_ref(py)
    }

    fn __repr__(slf: &Bound<'_, Self>) -> PyResult<String> {
        let this = slf.get();
        Ok(format!(
            "<{} {}: {}>",
            slf.get_type().name()?,
            this.handle,
            this.endpoints.bind(slf.py()).repr()?
        ))
    }

    fn __reduce__<'py>(
        slf: &Bound<'py, Self>,
    ) -> PyResult<(Bound<'py, PyAny>, Bound<'py, PyTuple>)> {
        let py = slf.py();
        let this = slf.get();
        let row = this
            .leaf(py)?
            .relation_ids(&this.kind)?
            .iter()
            .position(|&h| h == this.handle)
            .ok_or_else(|| PyValueError::new_err("cannot pickle a view of a deleted relation"))?;
        let restore = py.get_type::<Self>().getattr(intern!(py, "_restore"))?;
        let args = (this.world.bind(py).clone(), this.kind.clone(), row).into_pyobject(py)?;
        Ok((restore, args))
    }

    /// Unpickle: the view of the relation at `row` of `kind` in `world`.
    #[classmethod]
    fn _restore<'py>(
        _cls: &Bound<'py, PyType>,
        world: &Bound<'py, PyAny>,
        kind: &str,
        row: usize,
    ) -> PyResult<Bound<'py, PyAny>> {
        let leaf = Leaf::of(world)?;
        let handle =
            leaf.relation_ids(kind)?.get(row).copied().ok_or_else(|| {
                PyValueError::new_err(format!("no '{kind}' relation at row {row}"))
            })?;
        leaf.relation(kind, handle)
    }

    fn __traverse__(&self, visit: PyVisit<'_>) -> Result<(), PyTraverseError> {
        visit.call(&self.world)?;
        visit.call(&self.endpoints)
    }
}
field_mapping_impl!(PyRelationRef);

/// A bond of an :class:`Atomistic`; ``itom`` / ``jtom`` are its atoms.
#[pyclass(module = "molrs.core", name = "Bond", extends = PyRelationRef, frozen, subclass)]
pub struct PyBond {}

#[pymethods]
impl PyBond {
    /// The first atom.
    #[getter]
    fn itom<'py>(slf: &Bound<'py, Self>) -> PyResult<Bound<'py, PyAny>> {
        slf.as_super().get().endpoint(slf.py(), 0)
    }

    /// The second atom.
    #[getter]
    fn jtom<'py>(slf: &Bound<'py, Self>) -> PyResult<Bound<'py, PyAny>> {
        slf.as_super().get().endpoint(slf.py(), 1)
    }
}

/// An angle ``i–j–k`` of an :class:`Atomistic` (``j`` the vertex).
#[pyclass(module = "molrs.core", name = "Angle", extends = PyRelationRef, frozen, subclass)]
pub struct PyAngle {}

/// A proper dihedral ``i–j–k–l`` of an :class:`Atomistic`.
#[pyclass(module = "molrs.core", name = "Dihedral", extends = PyRelationRef, frozen, subclass)]
pub struct PyDihedral {}

/// An improper of an :class:`Atomistic`, in its style's slot order.
#[pyclass(module = "molrs.core", name = "Improper", extends = PyDihedral, frozen, subclass)]
pub struct PyImproper {}

/// A bond of a :class:`CoarseGrain`.
#[pyclass(module = "molrs.core", name = "CgBond", extends = PyRelationRef, frozen, subclass)]
pub struct PyCgBond {}

/// One unsatisfied valence (a port) of an :class:`Atomistic` or a
/// :class:`CoarseGrain`.
///
/// A port is the ordered pair ``(anchor, handle_atom)``: the **anchor** keeps
/// its place in the product molecule and the **handle atom** is the capping
/// node bonded to it that a paired descriptor's bond replaces. The descriptor
/// reads back as the fields ``port_kind`` (``$``, ``<``, ``>`` or ``!``),
/// ``port_label`` (``""`` when unnamed) and ``port_order``; it is fixed when
/// the port is added (writing a field raises).
///
/// The second endpoint is ``handle_atom``, not ``handle``: ``handle`` is the
/// port's own relation handle. The two answer different questions.
#[pyclass(module = "molrs.core", name = "Port", extends = PyRelationRef, frozen, subclass)]
pub struct PyPort {}

#[pymethods]
impl PyPort {
    /// The node that keeps its place in the product molecule.
    #[getter]
    fn anchor<'py>(slf: &Bound<'py, Self>) -> PyResult<Bound<'py, PyAny>> {
        slf.as_super().get().endpoint(slf.py(), 0)
    }

    /// The node the descriptor sits on: the root of the leaving group.
    #[getter]
    fn handle_atom<'py>(slf: &Bound<'py, Self>) -> PyResult<Bound<'py, PyAny>> {
        slf.as_super().get().endpoint(slf.py(), 1)
    }
}

// ---------------------------------------------------------------------------
// Refs
// ---------------------------------------------------------------------------

/// An ordered collection of views of one kind of one graph.
///
/// ``refs[i]`` is a view, ``refs[a:b]`` a smaller collection; ``refs["x"]``
/// reads a field of every item as a numpy array (``None`` where unset) and
/// ``refs["x", "y", "z"]`` stacks several side by side. Views are made on
/// demand, so ``len(graph.atoms)`` and ``graph.atoms["x"]`` make none. Every
/// door that takes several nodes or relations of a graph takes a ``Refs`` of
/// it as well as any iterable of its views.
#[pyclass(module = "molrs.core", name = "Refs", frozen, subclass)]
pub struct PyRefs {
    world: Py<PyAny>,
    /// `None` for nodes, the relation kind otherwise.
    kind: Option<String>,
    handles: Vec<u64>,
}

impl PyRefs {
    fn view<'py>(&self, leaf: &Leaf<'py>, handle: u64) -> PyResult<Bound<'py, PyAny>> {
        match &self.kind {
            None => leaf.node(handle),
            Some(kind) => leaf.relation(kind, handle),
        }
    }

    fn column<'py>(&self, py: Python<'py>, key: &Bound<'py, PyAny>) -> PyResult<Bound<'py, PyAny>> {
        let key = extract_column_key(key)?;
        let leaf = Leaf::of(self.world.bind(py))?;
        let numpy = py.import(intern!(py, "numpy"))?;
        if self.handles.is_empty() {
            return numpy.call_method1(intern!(py, "array"), (PyList::empty(py),));
        }
        let values: Vec<Py<PyAny>> = match &self.kind {
            None => {
                // The whole node table, fully set: one copy of the column.
                let dense = leaf.read(|graph| -> PyResult<Option<Bound<'py, PyAny>>> {
                    let table = graph.table();
                    let whole = table
                        .handles()
                        .map(u64::from)
                        .eq(self.handles.iter().copied());
                    let Some(column) = table.entity_column(&key).filter(|_| whole) else {
                        return Ok(None);
                    };
                    if !column.validity().as_slice().iter().all(|v| *v) {
                        return Ok(None);
                    }
                    Ok(Some(match column {
                        EntityColumn::F64(data, _) => {
                            numpy::PyArray1::from_slice(py, data).into_any()
                        }
                        EntityColumn::I32(data, _) => {
                            numpy::PyArray1::from_slice(py, data).into_any()
                        }
                        EntityColumn::Bool(data, _) => {
                            numpy::PyArray1::from_slice(py, data).into_any()
                        }
                        EntityColumn::Str(data, _) => {
                            let list = PyList::new(py, data.iter().map(String::as_str))?;
                            numpy.call_method1(intern!(py, "array"), (list,))?
                        }
                    }))
                })??;
                if let Some(dense) = dense {
                    return Ok(dense);
                }
                leaf.read(|graph| -> PyResult<Vec<Py<PyAny>>> {
                    let table = graph.table();
                    self.handles
                        .iter()
                        .map(|&h| {
                            let id = NodeId::from(h);
                            if !table.contains(id) {
                                return Err(stale_node(h));
                            }
                            match table.value(id, &key) {
                                Some(cell) => cell_to_py(py, cell),
                                None => Ok(py.None()),
                            }
                        })
                        .collect()
                })??
            }
            Some(kind) => {
                let kind = leaf.kind(kind)?;
                leaf.read(|graph| -> PyResult<Vec<Py<PyAny>>> {
                    self.handles
                        .iter()
                        .map(|&h| {
                            let (_, props) = graph
                                .get_relation(&kind, RelationId::from(h))
                                .map_err(molrs_error_to_pyerr)?;
                            match props.get(&key) {
                                Some(value) => prop_to_py(py, value),
                                None => Ok(py.None()),
                            }
                        })
                        .collect()
                })??
            }
        };
        let list = PyList::new(py, values)?;
        match numpy.call_method1(intern!(py, "array"), (&list,)) {
            Ok(array) => Ok(array),
            Err(_) => {
                let kwargs = PyDict::new(py);
                kwargs.set_item(intern!(py, "dtype"), intern!(py, "object"))?;
                numpy.call_method(intern!(py, "array"), (&list,), Some(&kwargs))
            }
        }
    }

    fn rows(&self, leaf: &Leaf<'_>) -> PyResult<Vec<usize>> {
        let all = match &self.kind {
            None => leaf.node_ids()?,
            Some(kind) => leaf.relation_ids(kind)?,
        };
        let row_of: HashMap<u64, usize> =
            all.into_iter().enumerate().map(|(r, h)| (h, r)).collect();
        self.handles
            .iter()
            .map(|h| {
                row_of.get(h).copied().ok_or_else(|| {
                    PyValueError::new_err("cannot pickle a view of a deleted node or relation")
                })
            })
            .collect()
    }
}

impl PyRefs {
    /// The relations `handles` of kind `kind` of `world`, in that order.
    pub(crate) fn of_relations(world: &Bound<'_, PyAny>, kind: &str, handles: Vec<u64>) -> Self {
        Self {
            world: world.clone().unbind(),
            kind: Some(kind.to_owned()),
            handles,
        }
    }
}

#[pymethods]
impl PyRefs {
    fn __len__(&self) -> usize {
        self.handles.len()
    }

    fn __getitem__<'py>(
        slf: &Bound<'py, Self>,
        key: &Bound<'py, PyAny>,
    ) -> PyResult<Bound<'py, PyAny>> {
        let py = slf.py();
        let this = slf.get();
        if key.is_instance_of::<PyString>()
            || key
                .extract::<PyRef<'_, crate::core::schema::PyKey>>()
                .is_ok()
        {
            return this.column(py, key);
        }
        if let Ok(names) = key.cast::<PyTuple>() {
            let columns = names
                .iter()
                .map(|name| this.column(py, &name))
                .collect::<PyResult<Vec<_>>>()?;
            let numpy = py.import(intern!(py, "numpy"))?;
            return numpy.call_method1(intern!(py, "column_stack"), (PyList::new(py, columns)?,));
        }
        if let Ok(slice) = key.cast::<PySlice>() {
            let indices = slice.indices(this.handles.len() as isize)?;
            let mut handles = Vec::with_capacity(indices.slicelength);
            let mut at = indices.start;
            for _ in 0..indices.slicelength {
                handles.push(this.handles[at as usize]);
                at += indices.step;
            }
            let sliced = Self {
                world: this.world.clone_ref(py),
                kind: this.kind.clone(),
                handles,
            };
            return Ok(Bound::new(py, sliced)?.into_any());
        }
        let index: isize = key.extract()?;
        let len = this.handles.len() as isize;
        let at = if index < 0 { index + len } else { index };
        if !(0..len).contains(&at) {
            return Err(PyIndexError::new_err("Refs index out of range"));
        }
        this.view(&Leaf::of(this.world.bind(py))?, this.handles[at as usize])
    }

    fn __iter__<'py>(slf: &Bound<'py, Self>) -> PyResult<Bound<'py, PyAny>> {
        let py = slf.py();
        let this = slf.get();
        let leaf = Leaf::of(this.world.bind(py))?;
        let views = this
            .handles
            .iter()
            .map(|&h| this.view(&leaf, h))
            .collect::<PyResult<Vec<_>>>()?;
        Ok(PyList::new(py, views)?.try_iter()?.into_any())
    }

    fn __contains__(&self, py: Python<'_>, item: &Bound<'_, PyAny>) -> bool {
        let (world, kind, handle) = if let Ok(node) = item.cast::<PyNodeRef>() {
            let node = node.get();
            (&node.world, None, node.handle)
        } else if let Ok(relation) = item.cast::<PyRelationRef>() {
            let relation = relation.get();
            (
                &relation.world,
                Some(relation.kind.as_str()),
                relation.handle,
            )
        } else {
            return false;
        };
        world.bind(py).is(self.world.bind(py))
            && kind == self.kind.as_deref()
            && self.handles.contains(&handle)
    }

    fn __repr__(&self) -> String {
        format!(
            "<Refs of {}: {}>",
            self.kind.as_deref().unwrap_or("nodes"),
            self.handles.len()
        )
    }

    fn __reduce__<'py>(
        slf: &Bound<'py, Self>,
    ) -> PyResult<(Bound<'py, PyAny>, Bound<'py, PyTuple>)> {
        let py = slf.py();
        let this = slf.get();
        let rows = this.rows(&Leaf::of(this.world.bind(py))?)?;
        let restore = py.get_type::<Self>().getattr(intern!(py, "_restore"))?;
        let args = (this.world.bind(py).clone(), this.kind.clone(), rows).into_pyobject(py)?;
        Ok((restore, args))
    }

    /// Unpickle: the items at `rows` of `kind` (``None``: nodes) in `world`.
    #[classmethod]
    fn _restore(
        _cls: &Bound<'_, PyType>,
        world: &Bound<'_, PyAny>,
        kind: Option<String>,
        rows: Vec<usize>,
    ) -> PyResult<Self> {
        let leaf = Leaf::of(world)?;
        let all = match &kind {
            None => leaf.node_ids()?,
            Some(kind) => leaf.relation_ids(kind)?,
        };
        let handles = rows
            .iter()
            .map(|&row| {
                all.get(row)
                    .copied()
                    .ok_or_else(|| PyValueError::new_err(format!("no item at row {row}")))
            })
            .collect::<PyResult<Vec<u64>>>()?;
        Ok(Self {
            world: world.clone().unbind(),
            kind,
            handles,
        })
    }

    fn __traverse__(&self, visit: PyVisit<'_>) -> Result<(), PyTraverseError> {
        visit.call(&self.world)
    }
}
