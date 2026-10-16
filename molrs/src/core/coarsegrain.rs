//! Coarse-grained molecular graph: [`CoarseGrain`].

use std::collections::{BTreeSet, HashMap};

use ndarray::Array1;
use slotmap::Key;

use crate::core::Block;
use crate::core::Frame;
use crate::core::MolRsError;
use crate::core::keys;

use crate::core::graph::{Graph, sealed};
use crate::core::molgraph::NodeGuard;
use crate::core::{BondNumber, CenterError, EntityTable, LinkError, Port, PortKind};
use crate::core::{GraphAccess, KindId, MolGraph, NodeId, PropValue, Props, Relation, RelationId};
use crate::op::Idx;

/// Result of [`CoarseGrain::extract_subgraph`].
#[derive(Debug, Clone)]
pub struct ExtractedCoarseGrain {
    /// The extracted bead graph (membership copied for selected beads).
    pub graph: CoarseGrain,
    /// Selected parent beads with a CG-bond neighbour outside the ball.
    pub boundary: Vec<NodeId>,
    /// New bead id → parent bead id.
    pub parent_of: HashMap<NodeId, NodeId>,
    /// Parent bead id → hops (CG bonds) from the nearest center.
    pub hops: HashMap<NodeId, i64>,
    /// Parent bead id → new bead id.
    pub node_map: HashMap<NodeId, NodeId>,
}

/// Coarse-grained molecular graph.
///
/// A *coarse-grained* (CG) model describes a molecule with fewer particles
/// than it has atoms: each particle, a **bead**, stands for a group of atoms
/// (a monomer, a few CH₂ units, a water cluster) and sits at a representative
/// position of that group. A **CG bond** says two beads are connected; it has
/// no chemical bond order.
///
/// `CoarseGrain` wraps the domain-agnostic [`MolGraph`] where every node is a
/// bead. It registers its own `bonds` kind and exposes one typed method per
/// operation on beads (`add_bead`, `del_bead` / `del_beads`, `get_bead`,
/// `set_bead`, `clear_bead`, `bead_ids`, `beads`, `n_beads`, `bead_table`),
/// CG bonds (the same operations), membership, ports, geometry, composition
/// and hashing; `MolGraph` itself stays chemistry-agnostic.
///
/// Invariant: every node has a string [`keys::BEAD_TYPE`] property. The
/// graph carries it as a guard, so every node write enforces it —
/// [`add_bead`](Self::add_bead), [`set_bead`](Self::set_bead),
/// [`clear_bead`](Self::clear_bead), [`merge`](Self::merge) and
/// [`replicate`](Self::replicate).
///
/// A bead additionally owns a **membership**: the set of underlying atom handles
/// it groups. Membership is variable-size directed ownership across worlds (the
/// atoms live in a separate all-atom world), so it is **not** a fixed-arity peer
/// relation and is stored here as opaque atom handles keyed by bead, not as a
/// scalar component. Resolving a handle back to an atom view is the caller's job
/// (it owns the source world); this layer owns only the handle topology.
///
/// # Examples
///
/// ```
/// use molrs::core::CoarseGrain;
/// use molrs::props;
/// use molrs::core::keys;
///
/// let mut cg = CoarseGrain::new();
/// let b1 = cg.add_bead(props! {
///     keys::BEAD_TYPE => "W",
///     keys::X => 0.0,
///     keys::Y => 0.0,
///     keys::Z => 0.0,
/// }).unwrap();
/// let b2 = cg.add_bead(props! {
///     keys::BEAD_TYPE => "W",
///     keys::X => 3.0,
///     keys::Y => 0.0,
///     keys::Z => 0.0,
/// }).unwrap();
/// cg.add_bond(b1, b2, props! {}).unwrap();
///
/// assert_eq!(cg.n_beads(), 2);
/// assert_eq!(cg.n_bonds(), 1);
/// ```
#[derive(Debug, Clone)]
pub struct CoarseGrain {
    graph: MolGraph,
    bond: KindId,
    members: HashMap<NodeId, Vec<u64>>,
}

impl Default for CoarseGrain {
    fn default() -> Self {
        Self::new()
    }
}

impl CoarseGrain {
    /// Create an empty coarse-grained molecular graph with the CG `bonds` kind
    /// registered.
    pub fn new() -> Self {
        let mut graph = MolGraph::new();
        graph.set_node_guard(Some(BEAD_TYPE_GUARD));
        let bond = graph.register_kind("bonds", 2);
        Self {
            graph,
            bond,
            members: HashMap::new(),
        }
    }

    /// Add a bead carrying the property bag `props` (its
    /// [`keys::BEAD_TYPE`], coordinates, …).
    ///
    /// # Errors
    ///
    /// [`MolRsError::Validation`] when `props` has no string
    /// [`keys::BEAD_TYPE`] (the invariant; nothing is written), or when a
    /// value contradicts the element type an existing bead column holds for
    /// that key (a string `x` where `x` is an `f64` column).
    pub fn add_bead(&mut self, props: impl Into<Props>) -> Result<NodeId, MolRsError> {
        self.graph.add_node(props)
    }

    /// Delete one bead, its CG bonds, ports and membership; returns the
    /// bead's property bag.
    ///
    /// # Errors
    ///
    /// [`MolRsError::NotFound`] for a dead handle; the graph is unchanged.
    pub fn del_bead(&mut self, id: NodeId) -> Result<Props, MolRsError> {
        let props = self.graph.del_node(id)?;
        self.members.remove(&id);
        Ok(props)
    }

    /// Delete several beads, with their CG bonds, ports and membership, in
    /// one pass; returns their property bags in `ids` order.
    ///
    /// # Errors
    ///
    /// [`MolRsError::Validation`] when `ids` names a bead twice and
    /// [`MolRsError::NotFound`] when one is dead. **Atomic**: on a refusal no
    /// bead, and no membership, is deleted.
    pub fn del_beads(&mut self, ids: &[NodeId]) -> Result<Vec<Props>, MolRsError> {
        let props = self.graph.del_nodes(ids)?;
        for id in ids {
            self.members.remove(id);
        }
        Ok(props)
    }

    // ---- bead → atom membership (opaque foreign atom handles) ----

    /// Set the atom handles a bead groups (replaces any existing membership).
    /// An empty slice clears the membership.
    ///
    /// # Errors
    ///
    /// [`MolRsError::NotFound`] when `bead` names no live bead; no membership
    /// is recorded then.
    pub fn set_bead_members(&mut self, bead: NodeId, atoms: Vec<u64>) -> Result<(), MolRsError> {
        if !self.graph.node_table().contains(bead) {
            return Err(MolRsError::not_found("bead", format!("bead {bead:?}")));
        }
        if atoms.is_empty() {
            self.members.remove(&bead);
        } else {
            self.members.insert(bead, atoms);
        }
        Ok(())
    }

    /// The atom handles a bead groups (empty if none recorded).
    pub fn bead_members(&self, bead: NodeId) -> &[u64] {
        self.members.get(&bead).map_or(&[], Vec::as_slice)
    }

    /// Beads whose membership includes `atom`, in bead-handle order.
    pub fn beads_of_atom(&self, atom: u64) -> Vec<NodeId> {
        let mut out: Vec<NodeId> = self
            .members
            .iter()
            .filter(|(_, atoms)| atoms.contains(&atom))
            .map(|(&bead, _)| bead)
            .collect();
        out.sort_by_key(|b| b.data().as_ffi());
        out
    }

    /// Materialize a bead's property bag (owned copy of its set components).
    pub fn get_bead(&self, id: NodeId) -> Result<Props, MolRsError> {
        self.graph.get_node(id)
    }

    /// Set a single component on a bead.
    ///
    /// # Errors
    ///
    /// [`MolRsError::Validation`] when `key` is [`keys::BEAD_TYPE`] and `val`
    /// is not a string (the invariant; the bead is unchanged), or when `val`
    /// does not fit the column; [`MolRsError::NotFound`] for a dead handle.
    pub fn set_bead(
        &mut self,
        id: NodeId,
        key: &str,
        val: impl Into<PropValue>,
    ) -> Result<(), MolRsError> {
        self.graph.set_node(id, key, val)
    }

    /// Clear a single component on a bead (no-op if absent).
    ///
    /// # Errors
    ///
    /// [`MolRsError::Validation`] when `key` is [`keys::BEAD_TYPE`]: a bead
    /// cannot lose its type. [`MolRsError::NotFound`] for a dead handle.
    pub fn clear_bead(&mut self, id: NodeId, key: &str) -> Result<(), MolRsError> {
        self.graph.clear_node(id, key)
    }

    /// Live bead handles, in row order (the order
    /// [`to_frame`](Self::to_frame) writes).
    pub fn bead_ids(&self) -> impl Iterator<Item = NodeId> + '_ {
        self.graph.node_ids()
    }

    /// Iterate over all `(NodeId, Props)` pairs (each property bag materialized).
    pub fn beads(&self) -> impl Iterator<Item = (NodeId, Props)> + '_ {
        self.graph.nodes()
    }

    /// Number of beads.
    pub fn n_beads(&self) -> usize {
        self.graph.n_nodes()
    }

    /// The beads' property columns, read-only and aligned to
    /// [`bead_ids`](Self::bead_ids) order. Writes go through
    /// [`set_bead`](Self::set_bead).
    pub fn bead_table(&self) -> &EntityTable<NodeId> {
        self.graph.node_table()
    }

    /// The positions `[x, y, z]` of `beads`, in Å as stored, in the order of
    /// `beads` (a bead listed twice appears twice). O(k) for k listed beads;
    /// each coordinate column is looked up once.
    ///
    /// # Errors
    ///
    /// Reporting the first offender in slice order:
    ///
    /// - [`MolRsError::NotFound`] when a handle is stale or belongs to
    ///   another graph;
    /// - [`MolRsError::Validation`] when a bead's `x`, `y` or `z` is missing or
    ///   not finite, naming the bead by its integer handle.
    ///
    /// # Examples
    ///
    /// ```
    /// use molrs::core::CoarseGrain;
    /// use molrs::props;
    /// use molrs::core::keys;
    ///
    /// let mut cg = CoarseGrain::new();
    /// let a = cg.add_bead(props! {
    ///     keys::BEAD_TYPE => "W",
    ///     keys::X => 0.0,
    ///     keys::Y => 1.0,
    ///     keys::Z => 2.0,
    /// }).unwrap();
    /// let b = cg.add_bead(props! {
    ///     keys::BEAD_TYPE => "P1",
    ///     keys::X => 3.0,
    ///     keys::Y => 4.0,
    ///     keys::Z => 5.0,
    /// }).unwrap();
    /// assert_eq!(cg.positions(&[b, a])?, vec![[3.0, 4.0, 5.0], [0.0, 1.0, 2.0]]);
    /// # Ok::<(), molrs::core::MolRsError>(())
    /// ```
    pub fn positions(&self, beads: &[NodeId]) -> Result<Vec<[f64; 3]>, MolRsError> {
        self.vectors(beads, keys::COORDS, "coordinate")
    }

    /// The site axes `[axis_x, axis_y, axis_z]` of `beads`, in Å as stored,
    /// in the order of `beads`. A site made by
    /// `builder::Coarsener::coarsen` carries
    /// the vector from the first member of its group to the site, which
    /// fixes the site's direction; a one-member site's axis is zero. O(k) for
    /// k listed beads.
    ///
    /// # Errors
    ///
    /// As [`positions`](Self::positions), for the axis columns.
    ///
    /// # Examples
    ///
    /// ```
    /// use molrs::core::keys;
    /// use molrs::core::CoarseGrain;
    /// use molrs::props;
    ///
    /// let mut cg = CoarseGrain::new();
    /// let a = cg.add_bead(props! {
    ///     keys::BEAD_TYPE => "S",
    ///     keys::X => 0.0,
    ///     keys::Y => 0.0,
    ///     keys::Z => 0.0,
    /// }).unwrap();
    /// for (key, v) in keys::AXIS.into_iter().zip([1.0, 2.0, 3.0]) {
    ///     cg.set_bead(a, key, v)?;
    /// }
    /// assert_eq!(cg.axes(&[a])?, vec![[1.0, 2.0, 3.0]]);
    /// # Ok::<(), molrs::core::MolRsError>(())
    /// ```
    pub fn axes(&self, beads: &[NodeId]) -> Result<Vec<[f64; 3]>, MolRsError> {
        self.vectors(beads, keys::AXIS, "axis")
    }

    /// The three f64 columns `columns` of `beads`, each finite.
    fn vectors(
        &self,
        beads: &[NodeId],
        columns: [&str; 3],
        what: &str,
    ) -> Result<Vec<[f64; 3]>, MolRsError> {
        let table = self.graph.node_table();
        // An absent (or non-f64) column reads as "missing" for every bead.
        let columns = columns.map(|key| (key, table.column::<f64>(key).ok()));
        beads
            .iter()
            .map(|&bead| {
                let row = Self::bead_row(table, bead)?;
                let mut point = [0.0; 3];
                for (value, (key, column)) in point.iter_mut().zip(&columns) {
                    *value = match column {
                        Some((data, valid)) if valid.get(row) && data[row].is_finite() => data[row],
                        _ => {
                            return Err(MolRsError::validation(format!(
                                "bead {} has no finite '{key}' {what}",
                                bead.data().as_ffi()
                            )));
                        }
                    };
                }
                Ok(point)
            })
            .collect()
    }

    /// The `bead_type` of each of `beads`, in the order of `beads` (a bead
    /// listed twice appears twice). O(k) for k listed beads; the column is
    /// looked up once.
    ///
    /// # Errors
    ///
    /// Reporting the first offender in slice order:
    ///
    /// - [`MolRsError::NotFound`] when a handle is stale or belongs to
    ///   another graph;
    /// - [`MolRsError::Validation`] when a bead's `bead_type` was cleared
    ///   through the inner graph, naming the bead by its integer handle.
    ///
    /// # Examples
    ///
    /// ```
    /// use molrs::core::CoarseGrain;
    /// use molrs::props;
    /// use molrs::core::keys;
    ///
    /// let mut cg = CoarseGrain::new();
    /// let a = cg.add_bead(props! {
    ///     keys::BEAD_TYPE => "W",
    ///     keys::X => 0.0,
    ///     keys::Y => 0.0,
    ///     keys::Z => 0.0,
    /// }).unwrap();
    /// let b = cg.add_bead(props! {
    ///     keys::BEAD_TYPE => "P1",
    ///     keys::X => 1.0,
    ///     keys::Y => 0.0,
    ///     keys::Z => 0.0,
    /// }).unwrap();
    /// assert_eq!(cg.bead_types(&[b, a, b])?, ["P1", "W", "P1"]);
    /// # Ok::<(), molrs::core::MolRsError>(())
    /// ```
    pub fn bead_types(&self, beads: &[NodeId]) -> Result<Vec<String>, MolRsError> {
        let table = self.graph.node_table();
        let column = table.column::<String>(keys::BEAD_TYPE).ok();
        beads
            .iter()
            .map(|&bead| {
                let row = Self::bead_row(table, bead)?;
                match column {
                    Some((data, valid)) if valid.get(row) => Ok(data[row].clone()),
                    _ => Err(MolRsError::validation(format!(
                        "bead {} carries no '{}'",
                        bead.data().as_ffi(),
                        keys::BEAD_TYPE
                    ))),
                }
            })
            .collect()
    }

    /// The row of `bead` in the node table, or `NotFound` naming its integer
    /// handle.
    fn bead_row(table: &EntityTable<NodeId>, bead: NodeId) -> Result<usize, MolRsError> {
        table
            .row(bead)
            .ok_or_else(|| MolRsError::not_found("bead", format!("bead {}", bead.data().as_ffi())))
    }

    /// Add a CG bond between two existing beads.
    pub fn add_bond(
        &mut self,
        a: NodeId,
        b: NodeId,
        props: impl Into<Props>,
    ) -> Result<RelationId, MolRsError> {
        self.graph.add_relation(self.bond, &[a, b], props)
    }

    /// Delete one CG bond (its beads stay); returns it as stored.
    ///
    /// # Errors
    ///
    /// [`MolRsError::NotFound`] when `id` names no live CG bond.
    pub fn del_bond(&mut self, id: RelationId) -> Result<Relation, MolRsError> {
        self.graph.del_relation(self.bond, id)
    }

    /// Delete several CG bonds; returns them as stored, in `ids` order.
    ///
    /// # Errors
    ///
    /// [`MolRsError::Validation`] when `ids` names a bond twice and
    /// [`MolRsError::NotFound`] when one names no live CG bond. **Atomic**:
    /// on a refusal no bond is deleted.
    pub fn del_bonds(&mut self, ids: &[RelationId]) -> Result<Vec<Relation>, MolRsError> {
        self.graph.del_relations(self.bond, ids)
    }

    /// Materialize a CG bond (endpoints + properties).
    pub fn get_bond(&self, id: RelationId) -> Result<Relation, MolRsError> {
        self.graph.get_relation(self.bond, id)
    }

    /// Set a single property on a CG bond.
    ///
    /// # Errors
    ///
    /// [`MolRsError::Validation`] when `val` does not fit the column;
    /// [`MolRsError::NotFound`] when `id` names no live CG bond.
    pub fn set_bond(
        &mut self,
        id: RelationId,
        key: &str,
        val: impl Into<PropValue>,
    ) -> Result<(), MolRsError> {
        self.graph.set_relation_prop(self.bond, id, key, val)
    }

    /// Clear a single property on a CG bond (no-op if absent).
    ///
    /// # Errors
    ///
    /// [`MolRsError::NotFound`] when `id` names no live CG bond.
    pub fn clear_bond(&mut self, id: RelationId, key: &str) -> Result<(), MolRsError> {
        self.graph.clear_relation_prop(self.bond, id, key)
    }

    /// Live CG bond handles, in row order.
    pub fn bond_ids(&self) -> impl Iterator<Item = RelationId> + '_ {
        self.graph.relation_ids(self.bond)
    }

    /// Iterate over all `(RelationId, Relation)` pairs (each materialized).
    pub fn bonds(&self) -> impl Iterator<Item = (RelationId, Relation)> + '_ {
        self.graph.relations(self.bond)
    }

    /// Number of CG bonds.
    pub fn n_bonds(&self) -> usize {
        self.graph.n_relations(self.bond)
    }

    /// Iterate over `(neighbor_id, bond_id)` for a given bead — its CG bonds
    /// only, never a port.
    pub fn neighbor_bonds(&self, id: NodeId) -> impl Iterator<Item = (NodeId, RelationId)> + '_ {
        self.graph
            .neighbor_relations(id)
            .filter(move |(kind, _, _)| *kind == self.bond)
            .map(move |(_, rid, other)| (other, rid))
    }

    // ---- ports ----

    /// Record a descriptor on the `(anchor, handle)` valence; see [`Port`]
    /// for the vocabulary.
    ///
    /// # Errors
    ///
    /// As [`MolGraph::add_port`]: a dead bead, a handle not bonded to the
    /// anchor, an `Unknown` order, or a valence that already carries a port.
    pub fn add_port(
        &mut self,
        anchor: NodeId,
        handle: NodeId,
        kind: PortKind,
        label: &str,
        order: BondNumber,
    ) -> Result<RelationId, MolRsError> {
        self.graph.add_port(anchor, handle, kind, label, order)
    }

    /// Delete one port (its bond stays); returns the port relation as stored.
    ///
    /// # Errors
    ///
    /// [`MolRsError::NotFound`] when `id` names no live port.
    pub fn del_port(&mut self, id: RelationId) -> Result<Relation, MolRsError> {
        self.graph.del_port(id)
    }

    /// Delete several ports (their bonds stay); returns the port relations as
    /// stored, in `ids` order.
    ///
    /// # Errors
    ///
    /// As [`MolGraph::del_ports`]. **Atomic**.
    pub fn del_ports(&mut self, ids: &[RelationId]) -> Result<Vec<Relation>, MolRsError> {
        self.graph.del_ports(ids)
    }

    /// Read a port back, validating its descriptor.
    ///
    /// # Errors
    ///
    /// As [`MolGraph::get_port`].
    pub fn get_port(&self, id: RelationId) -> Result<Port, MolRsError> {
        self.graph.get_port(id)
    }

    /// Live port handles, in row order.
    pub fn port_ids(&self) -> impl Iterator<Item = RelationId> + '_ {
        self.graph.port_ids()
    }

    /// Number of ports.
    pub fn n_ports(&self) -> usize {
        self.graph.n_ports()
    }

    /// The port on the `(anchor, handle)` valence, if any.
    pub fn port_on(&self, anchor: NodeId, handle: NodeId) -> Option<RelationId> {
        self.graph.port_on(anchor, handle)
    }

    /// The leaving group of `port`: its handle's connected component over CG
    /// bonds once the anchor–handle bond is cut.
    ///
    /// # Errors
    ///
    /// As [`MolGraph::leaving_group`]: the port is stale.
    pub fn leaving_group(&self, port: &Port) -> Result<BTreeSet<NodeId>, MolRsError> {
        self.graph.leaving_group(port)
    }

    /// Join port `a` to port `b` and return the new anchor–anchor CG bond;
    /// see [`MolGraph::link`].
    ///
    /// # Errors
    ///
    /// The [`LinkError`] of the first refusal; the graph is unchanged then.
    pub fn link(&mut self, a: RelationId, b: RelationId) -> Result<RelationId, LinkError> {
        self.graph.link(a, b)
    }

    /// Propagate each [`keys::FRAG_ID`] to the unlabelled degree-1 beads
    /// bonded to a labelled one; returns how many beads were labelled. See
    /// [`MolGraph::inherit_frag_ids`].
    pub fn inherit_frag_ids(&mut self) -> usize {
        self.graph.inherit_frag_ids()
    }

    // ---- geometry ----

    /// Translate every bead that has coordinates by `delta` (Å).
    pub fn translate(&mut self, delta: [f64; 3]) {
        self.graph.translate(delta);
    }

    /// Rotate every bead that has coordinates around `axis` by `angle`
    /// radians, optionally about `about` (Å; the origin by default).
    ///
    /// # Errors
    ///
    /// As [`MolGraph::rotate`]: an axis with no direction or a non-finite
    /// angle; nothing is written then.
    pub fn rotate(
        &mut self,
        axis: [f64; 3],
        angle: f64,
        about: Option<[f64; 3]>,
    ) -> Result<(), MolRsError> {
        self.graph.rotate(axis, angle, about)
    }

    /// Scale every bead that has coordinates by the per-axis `factor` about
    /// `about` (Å; the origin by default).
    pub fn scale(&mut self, factor: [f64; 3], about: Option<[f64; 3]>) {
        self.graph.scale(factor, about);
    }

    /// The mass-weighted centre (Å) of `beads`.
    ///
    /// # Errors
    ///
    /// The [`CenterError`] of [`MolGraph::center`].
    pub fn center(&self, beads: &[NodeId]) -> Result<[f64; 3], CenterError> {
        self.graph.center(beads)
    }

    /// Export to a tabular [`Frame`] in the canonical vocabulary every graph
    /// type shares: one `atoms` row per bead, one `bonds` row per CG bond
    /// (`atomi` / `atomj` = the endpoint beads' `atoms` rows), plus a
    /// `members` block for bead membership. The frame validates canonically.
    ///
    /// `members` has one row per (bead, atom) pair: `ibead` (UInt, the bead's
    /// row in `atoms`) and `atom` (UInt, the opaque atom handle), grouped by
    /// bead in `atoms` row order and keeping each bead's member order. The
    /// block is absent when no bead has members.
    ///
    /// # Known limit
    ///
    /// Every property a bead carries is written, `mol_id` included, while
    /// [`Self::from_frame`] reads exactly one molecule. So
    /// `from_frame(to_frame(cg))` refuses a CoarseGrain whose beads carry two
    /// or more distinct `mol_id` values; select one molecule with
    /// [`Frame::subset`] first.
    ///
    /// # Errors
    ///
    /// [`MolRsError::Validation`] when a bead or bond property contradicts the
    /// dtype the Frame schema declares for its key; the message names the
    /// refused column. The inner [`MolGraph`] accepts any value under a key it
    /// has no column for, so a string written under `"x"` is legal in the
    /// graph and only refused here.
    pub fn to_frame(&self) -> Result<Frame, MolRsError> {
        let mut frame = self.graph.to_frame()?;

        // `node_ids` is the row order the graph just wrote `atoms` in.
        let mut ibead: Vec<Idx> = Vec::new();
        let mut atom: Vec<Idx> = Vec::new();
        for (row, id) in self.graph.node_ids().enumerate() {
            for &handle in self.bead_members(id) {
                ibead.push(row as Idx);
                atom.push(handle);
            }
        }
        if !ibead.is_empty() {
            let mut members = Block::new();
            members
                .insert("ibead", Array1::from_vec(ibead).into_dyn(), None)
                .and_then(|()| members.insert("atom", Array1::from_vec(atom).into_dyn(), None))
                // `ibead` indexes this frame's bead rows; stated, not only
                // conventional. `atom` is a handle into the all-atom side,
                // which only the caller can name as a target.
                .and_then(|()| members.set_target("ibead", "atoms"))
                .map_err(|e| MolRsError::validation(format!("Frame 'members' block: {e}")))?;
            frame.insert("members", members);
        }
        Ok(frame)
    }

    /// Build from a [`Frame`] that holds **one molecule** in the canonical
    /// `atoms` / `bonds` vocabulary — the frame [`Self::to_frame`] writes,
    /// or one molecule of a LAMMPS data frame.
    ///
    /// Every `atoms` row becomes one bead and every `bonds` row one CG bond
    /// (`atomi` / `atomj` are 0-based `atoms` rows). Every `atoms` column is
    /// copied onto its bead and every `bonds` column onto its bond.
    ///
    /// `bead_type` comes from the first of these `atoms` columns present:
    ///
    /// 1. `bead_type` (Str), used as is;
    /// 2. `type` (Str), copied;
    /// 3. `type_id` (UInt), rendered in decimal.
    ///
    /// The source column also stays on each bead as a plain property. A null
    /// row in the chosen column is an error; it never falls through to the
    /// next source.
    ///
    /// A frame of many molecules is refused: when `atoms` carries `mol_id`,
    /// every row must hold the same value. Select one molecule with
    /// [`Frame::subset`] first. A frame without `mol_id` is one molecule.
    ///
    /// Only `atoms`, `bonds` and `members` are read. `angles`, `dihedrals`
    /// and every other block are ignored, so a CG relation kind other than
    /// `bonds` does not survive a round trip. A `members` block (see
    /// [`Self::to_frame`]) is read after the graph: row `ibead` of `atoms`
    /// gains atom handle `atom`, in block row order. The caller's frame is
    /// never mutated.
    ///
    /// # Errors
    ///
    /// [`MolRsError::Validation`] when the frame has no `atoms` rows; when
    /// `mol_id` has a null row, is not UInt, or holds two distinct values
    /// (naming both and `Frame::subset`); when none of `bead_type` / `type` /
    /// `type_id` is present, or the chosen one has a null row or the wrong
    /// dtype; when the `members` block lacks a UInt `ibead` or `atom` column,
    /// an `ibead` names no bead row, or an `(ibead, atom)` row repeats; when a
    /// column of `atoms` or `bonds` is not 1-D (an `(N, 3)` coordinate column,
    /// say), naming the block and the column — plus every other error of the
    /// graph read itself (e.g. a bond endpoint past the atom rows).
    pub fn from_frame(frame: &Frame) -> Result<Self, MolRsError> {
        let atoms = frame
            .get("atoms")
            .filter(|a| a.n_rows().unwrap_or(0) > 0)
            .ok_or_else(|| {
                MolRsError::validation("Frame has no 'atoms' rows to read beads from")
            })?;

        if atoms.contains_key(keys::MOL_ID) {
            if atoms.validity(keys::MOL_ID).is_some() {
                return Err(MolRsError::validation(
                    "Frame 'atoms' column 'mol_id' has a null row, so a bead belongs to no \
                     molecule",
                ));
            }
            let mol_id = atoms
                .get(keys::MOL_ID)
                .and_then(|c| c.get::<u64>())
                .ok_or_else(|| {
                    MolRsError::validation("Frame 'atoms' column 'mol_id' is not UInt")
                })?;
            let mut ids = mol_id.iter();
            if let Some(&first) = ids.next()
                && let Some(&other) = ids.find(|&&id| id != first)
            {
                return Err(MolRsError::validation(format!(
                    "Frame 'atoms' holds more than one molecule (mol_id {first} and \
                     {other}); select one molecule with `Frame::subset` first"
                )));
            }
        }

        let sources = [keys::BEAD_TYPE, keys::TYPE, keys::TYPE_ID];
        let Some(source) = sources.into_iter().find(|k| atoms.contains_key(k)) else {
            return Err(MolRsError::validation(format!(
                "Frame 'atoms' block has none of the columns {} to take bead_type from",
                sources.map(|k| format!("'{k}'")).join(", ")
            )));
        };
        if atoms.validity(source).is_some() {
            return Err(MolRsError::validation(format!(
                "Frame 'atoms' column '{source}' has a null row, so a bead would have no \
                 bead_type"
            )));
        }
        let wrong_dtype = |expected: &str| {
            MolRsError::validation(format!(
                "Frame 'atoms' column '{source}' is {}, not {expected}",
                atoms.dtype(source).map_or("unknown", |d| d.name())
            ))
        };
        // `None`: the source already is a Str `bead_type`.
        let bead_type: Option<Vec<String>> = if source == keys::BEAD_TYPE {
            atoms
                .get(source)
                .and_then(|c| c.get::<String>())
                .ok_or_else(|| wrong_dtype("Str"))?;
            None
        } else if source == keys::TYPE {
            let col = atoms
                .get(source)
                .and_then(|c| c.get::<String>())
                .ok_or_else(|| wrong_dtype("Str"))?;
            Some(col.iter().cloned().collect())
        } else {
            let col = atoms
                .get(source)
                .and_then(|c| c.get::<u64>())
                .ok_or_else(|| wrong_dtype("UInt"))?;
            Some(col.iter().map(ToString::to_string).collect())
        };

        let mut beads = atoms.clone();
        if let Some(bead_type) = bead_type {
            beads
                .insert(
                    keys::BEAD_TYPE,
                    Array1::from_vec(bead_type).into_dyn(),
                    None,
                )
                .map_err(|e| {
                    MolRsError::validation(format!("Frame 'atoms' column '{source}': {e}"))
                })?;
        }
        let mut canonical = Frame::new();
        canonical.insert("atoms", beads);
        if let Some(bonds) = frame.get("bonds") {
            canonical.insert("bonds", bonds.clone());
        }
        let mut cg = Self::from_canonical_frame(&canonical)?;
        if let Some(members) = frame.get("members") {
            cg.attach_members(members)?;
        }
        Ok(cg)
    }

    /// Read a frame already in [`MolGraph`]'s canonical vocabulary (`atoms` /
    /// `bonds`) into a validated `CoarseGrain`: every bead must carry
    /// `bead_type`, as the promotion from a [`MolGraph`] requires.
    fn from_canonical_frame(canonical: &Frame) -> Result<Self, MolRsError> {
        let mut graph = MolGraph::new();
        graph.register_kind("bonds", 2);
        graph.extend_from_frame(canonical)?;
        Self::promote(graph)
    }

    /// Attach the membership a `members` block states: row `r` gives bead row
    /// `ibead[r]` the atom handle `atom[r]`. Bead rows are the graph's row
    /// order, which a freshly read graph shares with its `atoms` block. A
    /// repeated `(ibead, atom)` row is refused, since it would list the atom
    /// twice in its bead.
    fn attach_members(&mut self, members: &Block) -> Result<(), MolRsError> {
        let column = |key: &str| {
            members
                .get(key)
                .and_then(|c| c.get::<u64>())
                .ok_or_else(|| {
                    MolRsError::validation(format!(
                        "Frame 'members' block has no UInt '{key}' column"
                    ))
                })
        };
        let ibead = column("ibead")?;
        let atom = column("atom")?;
        let bead_ids: Vec<NodeId> = self.graph.node_ids().collect();
        for (row, (&bead_row, &handle)) in ibead.iter().zip(atom.iter()).enumerate() {
            let Some(&bead) = bead_ids.get(bead_row as usize) else {
                return Err(MolRsError::validation(format!(
                    "Frame 'members' row {row} names bead {bead_row}, past the {} beads",
                    bead_ids.len()
                )));
            };
            let listed = self.members.entry(bead).or_default();
            if listed.contains(&handle) {
                return Err(MolRsError::validation(format!(
                    "Frame 'members' row {row} repeats atom {handle} of bead {bead_row}"
                )));
            }
            listed.push(handle);
        }
        Ok(())
    }

    /// Promote from a [`MolGraph`], validating all nodes have `"bead_type"`.
    ///
    /// The CG `bonds` kind is re-registered **by name**: an existing `bonds`
    /// kind of arity 2 keeps its id, and a missing one is registered fresh.
    /// Resolving by dense id instead would report a foreign kind's relations as
    /// this system's CG bonds whenever the graph registered something else
    /// first, and the next `add_bond` would write into it.
    ///
    /// # Errors
    ///
    /// Returns [`MolRsError::Validation`] when a node carries no `"bead_type"`,
    /// and when the graph already spells `bonds` at an arity other than 2
    /// (naming the kind and both arities).
    fn promote(mut mol: MolGraph) -> Result<Self, MolRsError> {
        for (id, bead) in mol.nodes() {
            require_bead_type(&bead, &format!("node {id:?}"))?;
        }
        let bond = mol.try_register_kind("bonds", 2)?;
        mol.set_node_guard(Some(BEAD_TYPE_GUARD));
        Ok(Self {
            graph: mol,
            bond,
            members: HashMap::new(),
        })
    }

    /// Place `transforms.len()` rigid copies of `template`, copy `c` moved by
    /// `transforms[c]` (rotation, then a translation in Å) and stamped
    /// `frag_id = frag_ids[c]`; returns the new beads copy-major. Column-wise
    /// and atomic — see [`MolGraph::replicate`].
    ///
    /// **No bead membership is copied.** Membership names foreign atoms of a
    /// separate all-atom world, and the copies do not own them; every
    /// replicated bead starts with an empty membership.
    ///
    /// # Errors
    ///
    /// The errors of [`MolGraph::replicate`]; `self` is unchanged then.
    pub fn replicate(
        &mut self,
        template: &CoarseGrain,
        transforms: &[crate::op::Rigid],
        frag_ids: &[crate::op::I],
    ) -> Result<Vec<NodeId>, MolRsError> {
        self.graph.replicate(&template.graph, transforms, frag_ids)
    }

    // ---- subgraph extraction / composition ----

    /// Induced subgraph on an explicit bead set. Stale handles fail-fast.
    pub fn induced_subgraph(
        &self,
        beads: &[NodeId],
    ) -> Result<(CoarseGrain, HashMap<NodeId, NodeId>), MolRsError> {
        let induced = self.graph.induced_subgraph(beads)?;
        let mut cg = CoarseGrain::promote(induced.graph)?;
        for (&old, &new) in &induced.node_map {
            let mem = self.bead_members(old);
            if !mem.is_empty() {
                cg.set_bead_members(new, mem.to_vec())?;
            }
        }
        Ok((cg, induced.node_map))
    }

    /// Radius ball around `centers` over CG bonds. Membership (opaque foreign
    /// atom handles) is copied for selected beads unchanged.
    pub fn extract_subgraph(
        &self,
        centers: &[NodeId],
        radius: i64,
    ) -> Result<ExtractedCoarseGrain, MolRsError> {
        let ball = self.graph.extract_ball(
            centers,
            radius,
            self.bond,
            /* copy_higher_order */ true,
            /* whole_groups */ &[],
        )?;
        let mut cg = CoarseGrain::promote(ball.graph)?;
        for (&old, &new) in &ball.node_map {
            let mem = self.bead_members(old);
            if !mem.is_empty() {
                cg.set_bead_members(new, mem.to_vec())?;
            }
        }
        Ok(ExtractedCoarseGrain {
            graph: cg,
            boundary: ball.boundary,
            parent_of: ball.parent_of,
            hops: ball.hops,
            node_map: ball.node_map,
        })
    }

    /// Structural merge. Returns `handle in other → handle in self`. Remaps bead
    /// membership keys; foreign atom handles inside membership stay as-is.
    ///
    /// # Errors
    ///
    /// [`MolRsError::Validation`] when a bead or relation property of `other`
    /// contradicts the element type `self` holds for that key — see
    /// [`MolGraph::merge`](crate::core::MolGraph::merge), whose
    /// partial-write contract this inherits.
    pub fn merge(&mut self, other: CoarseGrain) -> Result<HashMap<NodeId, NodeId>, MolRsError> {
        let node_map = self.graph.merge(other.graph)?;
        for (old_bead, members) in other.members {
            if let Some(&new_bead) = node_map.get(&old_bead)
                && !members.is_empty()
            {
                self.members.insert(new_bead, members);
            }
        }
        Ok(node_map)
    }

    // ---- structural graph hash (see [`crate::core::graph_hash`]) ----

    /// Isomorphism-invariant Weisfeiler–Lehman structural hash of the bead graph
    /// (bead-type node labels, bond-order edge labels). Shares the same
    /// [`MolGraph`] primitive that serves the all-atom case.
    pub fn structural_hash(&self) -> u64 {
        self.graph.structural_hash()
    }

    /// Deterministic canonical bead ordering from the WL refinement (see
    /// [`MolGraph::canonical_order`]).
    pub fn canonical_order(&self) -> Vec<NodeId> {
        self.graph.canonical_order()
    }

    /// Whether `self` and `other` are isomorphic as labeled bead graphs.
    pub fn is_isomorphic(&self, other: &CoarseGrain) -> bool {
        self.graph.is_isomorphic(&other.graph)
    }
}

/// Promote a [`MolGraph`] to a `CoarseGrain`, with no membership.
///
/// The CG `bonds` kind is matched **by name**: an existing `bonds` kind of
/// arity 2 keeps its id and a missing one is registered fresh; every other
/// kind (ports, …) is kept.
///
/// # Errors
///
/// [`MolRsError::Validation`] when a node carries no string
/// [`keys::BEAD_TYPE`], and when the graph spells `bonds` at an arity other
/// than 2.
impl TryFrom<MolGraph> for CoarseGrain {
    type Error = MolRsError;

    fn try_from(graph: MolGraph) -> Result<Self, MolRsError> {
        CoarseGrain::from_graph(graph)
    }
}

impl sealed::Sealed for CoarseGrain {
    fn inner_graph(&self, _: sealed::Token) -> &MolGraph {
        &self.graph
    }

    fn inner_graph_mut(&mut self, _: sealed::Token) -> &mut MolGraph {
        &mut self.graph
    }

    fn from_inner_graph(graph: MolGraph, _: sealed::Token) -> Result<Self, MolRsError> {
        CoarseGrain::promote(graph)
    }
}

impl Graph for CoarseGrain {}

/// The guard a `CoarseGrain` installs on its graph: every node is a bead
/// with a string type.
const BEAD_TYPE_GUARD: NodeGuard = NodeGuard {
    key: keys::BEAD_TYPE,
    check: require_bead_type,
};

/// The bead invariant: `bead` carries a string [`keys::BEAD_TYPE`].
///
/// # Errors
///
/// [`MolRsError::Validation`] naming `what`.
fn require_bead_type(bead: &Props, what: &str) -> Result<(), MolRsError> {
    match bead.get(keys::BEAD_TYPE) {
        Some(PropValue::Str(_)) => Ok(()),
        Some(other) => Err(MolRsError::validation(format!(
            "{what}: '{}' must be a string, got {other:?}",
            keys::BEAD_TYPE
        ))),
        None => Err(MolRsError::validation(format!(
            "{what} has no string '{}'",
            keys::BEAD_TYPE
        ))),
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::core::keys;
    use crate::props;

    #[test]
    fn test_new_and_add_bead() {
        let mut cg = CoarseGrain::new();
        let b1 = cg
            .add_bead(
                props! { keys::BEAD_TYPE => "W", keys::X => 0.0, keys::Y => 0.0, keys::Z => 0.0 },
            )
            .unwrap();
        let b2 = cg
            .add_bead(
                props! { keys::BEAD_TYPE => "P1", keys::X => 3.0, keys::Y => 0.0, keys::Z => 0.0 },
            )
            .unwrap();
        cg.add_bond(b1, b2, props! {}).unwrap();
        assert_eq!(cg.n_beads(), 2);
        assert_eq!(cg.n_bonds(), 1);
    }

    #[test]
    fn test_bead_has_type() {
        let mut cg = CoarseGrain::new();
        let b = cg
            .add_bead(
                props! { keys::BEAD_TYPE => "W", keys::X => 1.0, keys::Y => 2.0, keys::Z => 3.0 },
            )
            .unwrap();
        let bead = cg.get_bead(b).unwrap();
        assert_eq!(bead.get_str("bead_type"), Some("W"));
        assert_eq!(bead.get_f64("x"), Some(1.0));
    }

    #[test]
    fn to_frame_writes_atoms_and_bonds_that_validate() {
        let mut cg = CoarseGrain::new();
        let a = cg
            .add_bead(
                props! { keys::BEAD_TYPE => "W", keys::X => 0.0, keys::Y => 0.0, keys::Z => 0.0 },
            )
            .unwrap();
        let b = cg
            .add_bead(
                props! { keys::BEAD_TYPE => "P1", keys::X => 3.0, keys::Y => 0.0, keys::Z => 0.0 },
            )
            .unwrap();
        cg.add_bond(a, b, props! {}).unwrap();

        let frame = cg.to_frame().expect("a schema-conforming graph converts");
        let mut names: Vec<&str> = frame.keys().collect();
        names.sort_unstable();
        assert_eq!(names, vec!["atoms", "bonds"]);
        let bead_type: Vec<String> = frame["atoms"]
            .get("bead_type")
            .and_then(|c| c.get::<String>())
            .expect("atoms carries a Str bead_type")
            .iter()
            .cloned()
            .collect();
        assert_eq!(bead_type, vec!["W", "P1"]);
        let endpoint = |col: &str| -> Vec<Idx> {
            frame["bonds"]
                .get(col)
                .and_then(|c| c.get::<u64>())
                .expect("bonds carries UInt endpoints")
                .iter()
                .copied()
                .collect()
        };
        assert_eq!(endpoint("atomi"), vec![0]);
        assert_eq!(endpoint("atomj"), vec![1]);
        frame.validate().expect("the CG frame is a canonical frame");

        let restored = CoarseGrain::from_frame(&frame).expect("CG frame round-trip");
        assert_eq!(restored.n_beads(), 2);
        assert_eq!(restored.n_bonds(), 1);
    }

    /// `set_bead` carries the graph's schema opinion: a string under the
    /// float key `charge` never becomes a bead property, so no bead column
    /// can contradict the Frame schema in the first place.
    #[test]
    fn set_bead_refuses_a_str_under_a_schema_float_key() {
        use crate::core::DType;
        let mut cg = CoarseGrain::new();
        let b = cg
            .add_bead(
                props! { keys::BEAD_TYPE => "W", keys::X => 0.0, keys::Y => 0.0, keys::Z => 0.0 },
            )
            .unwrap();

        let err = cg
            .set_bead(b, "charge", "negative")
            .expect_err("a str cannot be stored at a schema-float key");
        assert!(matches!(err, MolRsError::Validation { .. }), "{err:?}");
        let msg = err.to_string();
        assert!(
            msg.contains("'charge'"),
            "the error names the key, got {msg}"
        );
        assert!(
            msg.contains(DType::Float.name()) && msg.contains(DType::String.name()),
            "the error names both dtypes, got {msg}"
        );
    }

    #[test]
    fn try_from_refuses_a_node_without_bead_type() {
        let mut g = MolGraph::new();
        g.register_kind("bonds", 2);
        g.add_node(Props::new()).expect("fixture node");
        let err = CoarseGrain::try_from(g).expect_err("no bead_type");
        assert!(err.to_string().contains(keys::BEAD_TYPE), "{err}");
    }

    /// A graph whose *first* registered kind is not `bonds` — what a fragment
    /// graph looks like — must not have that foreign kind's relations reported
    /// as its CG bonds, nor be written into by the next `add_bond`.
    #[test]
    fn try_from_resolves_bonds_by_name_not_kind_zero() {
        let mut graph = MolGraph::new();
        let ports = graph.register_kind("ports", 2);
        let mut bead = Props::new();
        bead.set("bead_type", "W");
        let b1 = graph.add_node(bead.clone()).expect("fixture node");
        let b2 = graph.add_node(bead).expect("fixture node");
        graph.add_relation(ports, &[b1, b2], props! {}).unwrap();

        let mut cg = CoarseGrain::try_from(graph).expect("bead types are present");
        assert_eq!(cg.n_ports(), 1, "the foreign kind survives");
        assert_eq!(cg.n_bonds(), 0, "a port relation is not a CG bond");

        cg.add_bond(b1, b2, props! {})
            .expect("a bond can still be added");
        assert_eq!(cg.n_bonds(), 1);
        assert_eq!(
            cg.n_ports(),
            1,
            "add_bond must not write into the foreign kind"
        );
    }

    /// A caller-supplied graph that spells `bonds` at another arity is a data
    /// condition, so the promotion returns an error instead of aborting the
    /// process inside `register_kind`.
    #[test]
    fn try_from_rejects_conflicting_arity() {
        let mut graph = MolGraph::new();
        graph.register_kind("bonds", 3);
        let err = CoarseGrain::try_from(graph)
            .expect_err("a 3-ary 'bonds' kind conflicts with the CG bond kind");
        assert!(matches!(err, MolRsError::Validation { .. }), "{err:?}");
    }

    #[test]
    fn scale_multiplies_offsets_from_the_centre_per_axis() {
        // p' = (p - c) * f + c. Point (1, 2, 3), factor (2, 3, 0.5):
        //   about c = (1, 1, 1) -> (1, 4, 2);  about the origin -> (2, 6, 1.5).
        let factor = [2.0, 3.0, 0.5];
        let cases: [(Option<[f64; 3]>, [f64; 3]); 2] = [
            (Some([1.0, 1.0, 1.0]), [1.0, 4.0, 2.0]),
            (None, [2.0, 6.0, 1.5]),
        ];
        for (about, expected) in cases {
            let mut sys = CoarseGrain::new();
            let id = sys
                .add_bead(props! {
                    keys::BEAD_TYPE => "W",
                    keys::X => 1.0,
                    keys::Y => 2.0,
                    keys::Z => 3.0,
                })
                .unwrap();
            let fixed = sys
                .add_bead(props! {
                    keys::BEAD_TYPE => "W",
                    keys::X => 1.0,
                    keys::Y => 1.0,
                    keys::Z => 1.0,
                })
                .unwrap();
            sys.scale(factor, about);
            let moved = sys.get_bead(id).expect("live handle");
            for (key, want) in ["x", "y", "z"].into_iter().zip(expected) {
                let got = moved.get_f64(key).expect("coordinate kept");
                assert!(
                    (got - want).abs() < 1e-12,
                    "{about:?} {key}: {got} != {want}"
                );
            }
            if about.is_some() {
                // The centre itself is a fixed point of the map.
                let centre = sys.get_bead(fixed).expect("live handle");
                for key in ["x", "y", "z"] {
                    let got = centre.get_f64(key).expect("coordinate kept");
                    assert!((got - 1.0).abs() < 1e-12, "centre {key} moved to {got}");
                }
            }
        }
    }

    // ---- from_frame: atoms/bonds vocabulary, validation, membership ----

    use crate::core::Block;
    use ndarray::Array1;

    fn float_col(values: &[f64]) -> ndarray::ArrayD<f64> {
        Array1::from_vec(values.to_vec()).into_dyn()
    }

    fn uint_col(values: &[u64]) -> ndarray::ArrayD<u64> {
        Array1::from_vec(values.to_vec()).into_dyn()
    }

    fn str_col(values: &[&str]) -> ndarray::ArrayD<String> {
        Array1::from_vec(values.iter().map(|s| (*s).to_owned()).collect()).into_dyn()
    }

    /// An `atoms` block of `n` rows with x/y/z only; the caller adds the
    /// bead-type source under test.
    fn xyz_atoms(x: &[f64], y: &[f64], z: &[f64]) -> Block {
        let mut atoms = Block::new();
        atoms.insert("x", float_col(x), None).unwrap();
        atoms.insert("y", float_col(y), None).unwrap();
        atoms.insert("z", float_col(z), None).unwrap();
        atoms
    }

    fn frame_of(atoms: Block) -> Frame {
        let mut frame = Frame::new();
        frame.insert("atoms", atoms);
        frame
    }

    fn bead_types(cg: &CoarseGrain) -> Vec<String> {
        cg.beads()
            .map(|(_, bead)| {
                bead.get_str("bead_type")
                    .expect("every bead carries bead_type")
                    .to_owned()
            })
            .collect()
    }

    #[test]
    fn from_frame_takes_bead_type_before_type() {
        let mut atoms = xyz_atoms(&[0.0, 1.0], &[0.0; 2], &[0.0; 2]);
        atoms
            .insert("bead_type", str_col(&["W", "W"]), None)
            .unwrap();
        atoms.insert("type", str_col(&["CT", "OH"]), None).unwrap();

        let cg = CoarseGrain::from_frame(&frame_of(atoms)).expect("a bead_type column");
        assert_eq!(bead_types(&cg), vec!["W", "W"]);
    }

    #[test]
    fn from_frame_takes_type_before_type_id() {
        let mut atoms = xyz_atoms(&[0.0, 1.0], &[0.0; 2], &[0.0; 2]);
        atoms.insert("type", str_col(&["CT", "OH"]), None).unwrap();
        atoms.insert("type_id", uint_col(&[1, 2]), None).unwrap();

        let cg = CoarseGrain::from_frame(&frame_of(atoms)).expect("a type column");
        assert_eq!(bead_types(&cg), vec!["CT", "OH"]);
    }

    #[test]
    fn from_frame_renders_a_uint_type_id_in_decimal() {
        let mut atoms = xyz_atoms(&[0.0], &[0.0], &[0.0]);
        atoms.insert("type_id", uint_col(&[2]), None).unwrap();

        let cg = CoarseGrain::from_frame(&frame_of(atoms)).expect("a UInt type_id column");
        assert_eq!(bead_types(&cg), vec!["2"]);
    }

    #[test]
    fn from_frame_copies_a_type_column_and_every_other_atoms_column() {
        let mut atoms = xyz_atoms(&[0.0, 1.0], &[0.5, 2.0], &[-1.0, 3.0]);
        atoms.insert("type", str_col(&["CT", "OH"]), None).unwrap();
        atoms
            .insert("charge", float_col(&[0.25, -0.25]), None)
            .unwrap();

        let cg = CoarseGrain::from_frame(&frame_of(atoms)).expect("a Str type column");
        assert_eq!(cg.n_beads(), 2);
        let beads: Vec<Props> = cg.beads().map(|(_, bead)| bead).collect();
        let expected = [("CT", 0.0, 0.5, -1.0, 0.25), ("OH", 1.0, 2.0, 3.0, -0.25)];
        for (bead, (ty, x, y, z, q)) in beads.iter().zip(expected) {
            assert_eq!(bead.get_str("bead_type"), Some(ty));
            assert_eq!(bead.get_str("type"), Some(ty), "the source stays a prop");
            assert_eq!(bead.get_f64("x"), Some(x));
            assert_eq!(bead.get_f64("y"), Some(y));
            assert_eq!(bead.get_f64("z"), Some(z));
            assert_eq!(bead.get_f64("charge"), Some(q));
        }
    }

    #[test]
    fn from_frame_turns_bond_rows_into_cg_bonds() {
        let mut atoms = xyz_atoms(&[0.0, 1.0, 2.0], &[0.0; 3], &[0.0; 3]);
        atoms
            .insert("type", str_col(&["A", "B", "C"]), None)
            .unwrap();
        // Bond endpoints address 0-based atoms rows.
        let mut bonds = Block::new();
        bonds.insert("atomi", uint_col(&[0, 1]), None).unwrap();
        bonds.insert("atomj", uint_col(&[2, 2]), None).unwrap();
        bonds.insert("type_id", uint_col(&[1, 1]), None).unwrap();
        let mut frame = frame_of(atoms);
        frame.insert("bonds", bonds);

        let cg = CoarseGrain::from_frame(&frame).expect("atoms + bonds");
        assert_eq!(cg.n_bonds(), 2);
        let ids: Vec<NodeId> = cg.bead_ids().collect();
        let mut endpoints: Vec<[NodeId; 2]> = cg
            .bonds()
            .map(|(_, bond)| [bond.nodes[0], bond.nodes[1]])
            .collect();
        endpoints.sort_by_key(|[a, _]| ids.iter().position(|id| id == a));
        assert_eq!(endpoints, vec![[ids[0], ids[2]], [ids[1], ids[2]]]);
    }

    #[test]
    fn from_frame_ignores_an_angles_block() {
        let mut atoms = xyz_atoms(&[0.0, 1.0, 2.0], &[0.0; 3], &[0.0; 3]);
        atoms
            .insert("type", str_col(&["A", "B", "C"]), None)
            .unwrap();
        let mut bonds = Block::new();
        bonds.insert("atomi", uint_col(&[0, 1]), None).unwrap();
        bonds.insert("atomj", uint_col(&[1, 2]), None).unwrap();
        let mut angles = Block::new();
        angles.insert("atomi", uint_col(&[0]), None).unwrap();
        angles.insert("atomj", uint_col(&[1]), None).unwrap();
        angles.insert("atomk", uint_col(&[2]), None).unwrap();
        let mut frame = frame_of(atoms);
        frame.insert("bonds", bonds);
        frame.insert("angles", angles);

        let cg = CoarseGrain::from_frame(&frame).expect("an angles block is ignored");
        assert_eq!(cg.n_beads(), 3);
        assert_eq!(cg.n_bonds(), 2);
    }

    #[test]
    fn from_frame_accepts_a_single_mol_id() {
        let mut atoms = xyz_atoms(&[0.0, 1.0], &[0.0; 2], &[0.0; 2]);
        atoms.insert("type", str_col(&["A", "B"]), None).unwrap();
        atoms.insert("mol_id", uint_col(&[7, 7]), None).unwrap();

        let cg = CoarseGrain::from_frame(&frame_of(atoms)).expect("one molecule, id 7");
        assert_eq!(cg.n_beads(), 2);
    }

    #[test]
    fn from_frame_refuses_a_frame_without_atoms() {
        let err =
            CoarseGrain::from_frame(&Frame::new()).expect_err("no atoms block to read beads from");
        assert!(matches!(err, MolRsError::Validation { .. }), "{err:?}");
    }

    #[test]
    fn from_frame_refuses_an_atoms_block_with_no_rows() {
        let mut atoms = Block::new();
        atoms.insert("bead_type", str_col(&[]), None).unwrap();

        let err = CoarseGrain::from_frame(&frame_of(atoms))
            .expect_err("a CoarseGrain needs at least one bead row");
        assert!(matches!(err, MolRsError::Validation { .. }), "{err:?}");
    }

    #[test]
    fn from_frame_refuses_more_than_one_mol_id() {
        let mut atoms = xyz_atoms(&[0.0, 1.0, 2.0], &[0.0; 3], &[0.0; 3]);
        atoms
            .insert("type", str_col(&["A", "B", "C"]), None)
            .unwrap();
        atoms.insert("mol_id", uint_col(&[0, 0, 1]), None).unwrap();

        let err = CoarseGrain::from_frame(&frame_of(atoms))
            .expect_err("molecules 0 and 1 are two molecules");
        assert!(matches!(err, MolRsError::Validation { .. }), "{err:?}");
        let msg = err.to_string();
        assert!(msg.contains("mol_id"), "the error names mol_id, got {msg}");
        assert!(
            msg.contains("Frame::subset"),
            "the error points at Frame::subset, got {msg}"
        );
    }

    #[test]
    fn from_frame_refuses_a_null_mol_id() {
        let mut atoms = xyz_atoms(&[0.0, 1.0], &[0.0; 2], &[0.0; 2]);
        atoms.insert("type", str_col(&["A", "B"]), None).unwrap();
        atoms
            .insert("mol_id", uint_col(&[7, 0]), Some(vec![true, false]))
            .unwrap();

        let err =
            CoarseGrain::from_frame(&frame_of(atoms)).expect_err("bead 1 belongs to no molecule");
        assert!(matches!(err, MolRsError::Validation { .. }), "{err:?}");
    }

    #[test]
    fn from_frame_refuses_atoms_without_a_bead_type_source() {
        let atoms = xyz_atoms(&[0.0, 1.0], &[0.0, 0.0], &[0.0, 0.0]);

        let err = CoarseGrain::from_frame(&frame_of(atoms))
            .expect_err("none of bead_type, type, type_id is present");
        assert!(matches!(err, MolRsError::Validation { .. }), "{err:?}");
        let msg = err.to_string();
        for source in ["bead_type", "type", "type_id"] {
            assert!(msg.contains(source), "the error names {source}, got {msg}");
        }
    }

    #[test]
    fn from_frame_refuses_a_null_type_row() {
        // A present `type_id` is not a fallback: the first present source
        // decides, and its null row is an error.
        let mut atoms = xyz_atoms(&[0.0, 1.0], &[0.0; 2], &[0.0; 2]);
        atoms
            .insert("type", str_col(&["CT", ""]), Some(vec![true, false]))
            .unwrap();
        atoms.insert("type_id", uint_col(&[1, 2]), None).unwrap();

        let err = CoarseGrain::from_frame(&frame_of(atoms))
            .expect_err("bead 1 has a null type and no fall-through");
        assert!(matches!(err, MolRsError::Validation { .. }), "{err:?}");
    }

    #[test]
    fn from_frame_refuses_a_2d_atoms_column() {
        let mut atoms = Block::new();
        atoms.insert("type", str_col(&["A", "B"]), None).unwrap();
        atoms
            .insert(
                "xyz",
                ndarray::Array2::from_shape_vec((2, 3), vec![0.0, 1.0, 2.0, 3.0, 4.0, 5.0])
                    .unwrap()
                    .into_dyn(),
                None,
            )
            .unwrap();

        let err = CoarseGrain::from_frame(&frame_of(atoms))
            .expect_err("a (2, 3) column is not a per-bead property");
        assert!(matches!(err, MolRsError::Validation { .. }), "{err:?}");
        assert!(err.to_string().contains("'xyz'"), "{err}");
    }

    #[test]
    fn bead_membership_survives_to_frame_then_from_frame() {
        let mut cg = CoarseGrain::new();
        let a = cg
            .add_bead(
                props! { keys::BEAD_TYPE => "W", keys::X => 0.0, keys::Y => 0.0, keys::Z => 0.0 },
            )
            .unwrap();
        let b = cg
            .add_bead(
                props! { keys::BEAD_TYPE => "P1", keys::X => 1.0, keys::Y => 0.0, keys::Z => 0.0 },
            )
            .unwrap();
        cg.add_bead(
            props! { keys::BEAD_TYPE => "P2", keys::X => 2.0, keys::Y => 0.0, keys::Z => 0.0 },
        )
        .unwrap();
        cg.set_bead_members(a, vec![10, 11]).unwrap();
        cg.set_bead_members(b, vec![12]).unwrap();

        let frame = cg.to_frame().expect("a schema-conforming graph converts");
        let restored = CoarseGrain::from_frame(&frame).expect("CG frame round-trip");
        let ids: Vec<NodeId> = restored.bead_ids().collect();
        assert_eq!(ids.len(), 3);
        assert_eq!(restored.bead_members(ids[0]), &[10, 11]);
        assert_eq!(restored.bead_members(ids[1]), &[12]);
        assert!(restored.bead_members(ids[2]).is_empty());
    }

    #[test]
    fn from_frame_refuses_a_members_row_past_the_bead_count() {
        let mut atoms = xyz_atoms(&[0.0, 1.0], &[0.0, 0.0], &[0.0, 0.0]);
        atoms
            .insert("bead_type", str_col(&["W", "W"]), None)
            .unwrap();
        let mut members = Block::new();
        members.insert("ibead", uint_col(&[0, 2]), None).unwrap();
        members.insert("atom", uint_col(&[10, 11]), None).unwrap();
        let mut frame = Frame::new();
        frame.insert("atoms", atoms);
        frame.insert("members", members);

        let err =
            CoarseGrain::from_frame(&frame).expect_err("ibead 2 names no bead of a two-bead frame");
        assert!(matches!(err, MolRsError::Validation { .. }), "{err:?}");
    }

    #[test]
    fn from_frame_refuses_a_repeated_members_row() {
        let mut atoms = xyz_atoms(&[0.0, 1.0], &[0.0, 0.0], &[0.0, 0.0]);
        atoms
            .insert("bead_type", str_col(&["W", "W"]), None)
            .unwrap();
        let mut members = Block::new();
        members.insert("ibead", uint_col(&[0, 0]), None).unwrap();
        members.insert("atom", uint_col(&[10, 10]), None).unwrap();
        let mut frame = Frame::new();
        frame.insert("atoms", atoms);
        frame.insert("members", members);

        let err =
            CoarseGrain::from_frame(&frame).expect_err("the (ibead 0, atom 10) row appears twice");
        assert!(matches!(err, MolRsError::Validation { .. }), "{err:?}");
    }

    // ---- replicate ----

    /// Membership names foreign atoms the copies do not own, so a replicated
    /// bead carries none even when its template bead does.
    #[test]
    fn replicate_copies_no_bead_membership() {
        use crate::op::Rigid;

        let mut template = CoarseGrain::new();
        let w = template
            .add_bead(
                props! { keys::BEAD_TYPE => "W", keys::X => 0.0, keys::Y => 0.0, keys::Z => 0.0 },
            )
            .unwrap();
        let p = template
            .add_bead(
                props! { keys::BEAD_TYPE => "P1", keys::X => 1.0, keys::Y => 0.0, keys::Z => 0.0 },
            )
            .unwrap();
        template.add_bond(w, p, props! {}).unwrap();
        template.set_bead_members(w, vec![10, 11]).unwrap();
        template.set_bead_members(p, vec![12]).unwrap();

        let shifted = Rigid {
            rotation: Rigid::IDENTITY.rotation,
            translation: [5.0, 0.0, 0.0],
        };
        let mut out = CoarseGrain::new();
        let beads = out
            .replicate(&template, &[Rigid::IDENTITY, shifted], &[0, 1])
            .expect("two copies of a two-bead template replicate");

        assert_eq!(beads.len(), 4);
        assert_eq!(out.n_beads(), 4);
        assert_eq!(out.n_bonds(), 2);
        for &bead in &beads {
            assert!(
                out.bead_members(bead).is_empty(),
                "bead {bead:?} was given membership it does not own"
            );
        }
        assert_eq!(template.bead_members(w), &[10, 11], "template untouched");
    }

    // ---- center ----

    /// The group restricts the node set: an unlisted mass-100 bead at
    /// (50,0,0) leaves masses 1 and 3 at x = 0 and 4 centred at (3,0,0).
    #[test]
    fn center_of_a_group_ignores_beads_outside_it() {
        let mut cg = CoarseGrain::new();
        let light = cg
            .add_bead(
                props! { keys::BEAD_TYPE => "W", keys::X => 0.0, keys::Y => 0.0, keys::Z => 0.0 },
            )
            .unwrap();
        let outside = cg
            .add_bead(
                props! { keys::BEAD_TYPE => "W", keys::X => 50.0, keys::Y => 0.0, keys::Z => 0.0 },
            )
            .unwrap();
        let heavy = cg
            .add_bead(
                props! { keys::BEAD_TYPE => "P1", keys::X => 4.0, keys::Y => 0.0, keys::Z => 0.0 },
            )
            .unwrap();
        for (bead, mass) in [(light, 1.0), (outside, 100.0), (heavy, 3.0)] {
            cg.set_bead(bead, crate::core::keys::MASS, mass).unwrap();
        }
        assert_eq!(cg.center(&[light, heavy]), Ok([3.0, 0.0, 0.0]));
    }

    // ---- positions / bead_types ----

    #[test]
    fn positions_come_back_in_the_order_asked() {
        let mut cg = CoarseGrain::new();
        let b0 = cg
            .add_bead(
                props! { keys::BEAD_TYPE => "W", keys::X => 0.5, keys::Y => -1.25, keys::Z => 2.0 },
            )
            .unwrap();
        cg.add_bead(
            props! { keys::BEAD_TYPE => "P1", keys::X => 3.0, keys::Y => 4.0, keys::Z => 5.0 },
        )
        .unwrap();
        let b2 = cg
            .add_bead(props! {
                keys::BEAD_TYPE => "P2",
                keys::X => -0.75,
                keys::Y => 8.5,
                keys::Z => 0.125,
            })
            .unwrap();

        let positions = cg.positions(&[b2, b0]).expect("both beads are placed");

        assert_eq!(positions, vec![[-0.75, 8.5, 0.125], [0.5, -1.25, 2.0]]);
    }

    #[test]
    fn bead_types_repeat_a_repeated_bead() {
        let mut cg = CoarseGrain::new();
        cg.add_bead(
            props! { keys::BEAD_TYPE => "W", keys::X => 0.0, keys::Y => 0.0, keys::Z => 0.0 },
        )
        .unwrap();
        let b1 = cg
            .add_bead(
                props! { keys::BEAD_TYPE => "P1", keys::X => 1.0, keys::Y => 0.0, keys::Z => 0.0 },
            )
            .unwrap();

        let types = cg.bead_types(&[b1, b1]).expect("b1 is live");

        assert_eq!(types, vec!["P1".to_owned(), "P1".to_owned()]);
    }

    #[test]
    fn a_stale_bead_is_not_found_by_both_accessors() {
        let mut cg = CoarseGrain::new();
        let kept = cg
            .add_bead(
                props! { keys::BEAD_TYPE => "W", keys::X => 0.0, keys::Y => 0.0, keys::Z => 0.0 },
            )
            .unwrap();
        let gone = cg
            .add_bead(
                props! { keys::BEAD_TYPE => "P1", keys::X => 1.0, keys::Y => 0.0, keys::Z => 0.0 },
            )
            .unwrap();
        cg.del_bead(gone).expect("fixture removal");

        let err = cg.positions(&[kept, gone]).expect_err("gone is stale");
        assert!(matches!(err, MolRsError::NotFound { .. }), "{err:?}");
        let err = cg.bead_types(&[kept, gone]).expect_err("gone is stale");
        assert!(matches!(err, MolRsError::NotFound { .. }), "{err:?}");
    }

    #[test]
    fn a_bead_without_z_is_refused_by_its_integer_handle() {
        let mut cg = CoarseGrain::new();
        cg.add_bead(
            props! { keys::BEAD_TYPE => "W", keys::X => 0.0, keys::Y => 0.0, keys::Z => 0.0 },
        )
        .unwrap();
        let flat = cg.add_bead(props! { keys::BEAD_TYPE => "P1" }).unwrap();
        for key in [keys::X, keys::Y] {
            cg.set_bead(flat, key, 1.0).unwrap();
        }

        let err = cg.positions(&[flat]).expect_err("flat has no z");

        assert!(matches!(err, MolRsError::Validation { .. }), "{err:?}");
        let msg = err.to_string();
        let handle = flat.data().as_ffi().to_string();
        assert!(msg.contains(&handle), "names handle {handle}: {msg}");
        assert!(!msg.contains("NodeId("), "no debug id: {msg}");
    }

    #[test]
    fn a_non_finite_coordinate_is_refused() {
        let mut cg = CoarseGrain::new();
        let bad = cg
            .add_bead(props! {
                keys::BEAD_TYPE => "W",
                keys::X => 0.0,
                keys::Y => f64::NAN,
                keys::Z => 0.0,
            })
            .unwrap();

        let err = cg.positions(&[bad]).expect_err("y is NaN");

        assert!(matches!(err, MolRsError::Validation { .. }), "{err:?}");
    }

    // ---- the bead-type invariant ----

    #[test]
    fn add_bead_requires_a_string_bead_type_and_writes_nothing_without_one() {
        let mut cg = CoarseGrain::new();
        cg.add_bead(props! { keys::BEAD_TYPE => "W" }).unwrap();
        for bag in [
            props! { keys::X => 1.0 },
            props! { keys::BEAD_TYPE => 3, keys::X => 1.0 },
            props! { keys::ATOMIC_NUMBER => 6u8, keys::X => 1.0 },
        ] {
            let err = cg.add_bead(bag).unwrap_err();
            assert!(matches!(err, MolRsError::Validation { .. }), "{err:?}");
        }
        assert_eq!(cg.n_beads(), 1);
        let frame = cg.to_frame().unwrap();
        assert!(frame.get("atoms").unwrap().get(keys::X).is_none());
    }

    #[test]
    fn set_bead_refuses_a_non_string_bead_type_and_clear_bead_refuses_it() {
        let mut cg = CoarseGrain::new();
        let b = cg
            .add_bead(props! { keys::BEAD_TYPE => "W", keys::MASS => 72.0 })
            .unwrap();
        assert!(cg.set_bead(b, keys::BEAD_TYPE, 7).is_err());
        assert!(cg.clear_bead(b, keys::BEAD_TYPE).is_err());
        assert_eq!(cg.get_bead(b).unwrap().get_str(keys::BEAD_TYPE), Some("W"));

        cg.set_bead(b, keys::BEAD_TYPE, "P1").unwrap();
        assert_eq!(cg.get_bead(b).unwrap().get_str(keys::BEAD_TYPE), Some("P1"));
        cg.clear_bead(b, keys::MASS).unwrap();
        assert!(cg.get_bead(b).unwrap().get(keys::MASS).is_none());
    }

    #[test]
    fn the_bead_type_guard_holds_on_the_generic_node_mutators() {
        let mut cg = CoarseGrain::new();
        let b = cg.add_bead(props! { keys::BEAD_TYPE => "W" }).unwrap();
        let raw = cg.graph_mut();
        assert!(raw.add_node(props! { keys::X => 0.0 }).is_err());
        assert!(raw.set_node(b, keys::BEAD_TYPE, 1).is_err());
        assert!(raw.clear_node(b, keys::BEAD_TYPE).is_err());
        assert_eq!(cg.n_beads(), 1);
        assert_eq!(cg.get_bead(b).unwrap().get_str(keys::BEAD_TYPE), Some("W"));

        let mut bare = cg.graph().clone();
        bare.set_node_guard(None);
        bare.add_node(props! {}).unwrap();
        assert!(CoarseGrain::try_from(bare).is_err());
    }

    // ---- the typed bead / CG-bond / port surface ----

    /// Three beads A-B-C with members, bonded in a row.
    fn trimer() -> (CoarseGrain, Vec<NodeId>, Vec<RelationId>) {
        let mut cg = CoarseGrain::new();
        let ids: Vec<NodeId> = ["A", "B", "C"]
            .into_iter()
            .enumerate()
            .map(|(i, t)| {
                cg.add_bead(props! {
                    keys::BEAD_TYPE => t,
                    keys::X => i as f64,
                    keys::Y => 0.0,
                    keys::Z => 0.0,
                })
                .unwrap()
            })
            .collect();
        for (k, &b) in ids.iter().enumerate() {
            cg.set_bead_members(b, vec![10 * k as u64, 10 * k as u64 + 1])
                .unwrap();
        }
        let bonds = vec![
            cg.add_bond(ids[0], ids[1], props! {}).unwrap(),
            cg.add_bond(ids[1], ids[2], props! {}).unwrap(),
        ];
        (cg, ids, bonds)
    }

    #[test]
    fn del_bead_drops_its_bonds_and_membership() {
        let (mut cg, ids, _) = trimer();
        let bag = cg.del_bead(ids[1]).unwrap();
        assert_eq!(bag.get_str(keys::BEAD_TYPE), Some("B"));
        assert_eq!((cg.n_beads(), cg.n_bonds()), (2, 0));
        assert!(cg.bead_members(ids[1]).is_empty());
        assert!(cg.beads_of_atom(10).is_empty());
        assert_eq!(cg.bead_members(ids[0]), &[0, 1]);
        assert!(matches!(
            cg.del_bead(ids[1]),
            Err(MolRsError::NotFound { .. })
        ));
    }

    /// A batch with a dead or repeated bead is refused whole: no bead, bond
    /// or membership of it is deleted.
    #[test]
    fn del_beads_is_atomic() {
        let (mut cg, ids, _) = trimer();
        let dead = cg.add_bead(props! { keys::BEAD_TYPE => "D" }).unwrap();
        cg.del_bead(dead).unwrap();
        assert!(cg.del_beads(&[ids[0], dead]).is_err());
        assert!(cg.del_beads(&[ids[0], ids[0]]).is_err());
        assert_eq!((cg.n_beads(), cg.n_bonds()), (3, 2));
        assert_eq!(cg.bead_members(ids[0]), &[0, 1], "membership kept");

        assert_eq!(cg.del_beads(&[ids[2], ids[0]]).unwrap().len(), 2);
        assert_eq!(cg.bead_ids().collect::<Vec<_>>(), [ids[1]]);
        assert!(cg.bead_members(ids[0]).is_empty() && cg.bead_members(ids[2]).is_empty());
        assert_eq!(cg.bead_members(ids[1]), &[10, 11]);
    }

    #[test]
    fn bead_ids_beads_and_bead_table_share_row_order() {
        let (cg, ids, _) = trimer();
        assert_eq!(cg.bead_ids().collect::<Vec<_>>(), ids);
        assert_eq!(cg.beads().map(|(id, _)| id).collect::<Vec<_>>(), ids);
        let (types, _) = cg.bead_table().column::<String>(keys::BEAD_TYPE).unwrap();
        assert_eq!(types, ["A", "B", "C"]);
    }

    #[test]
    fn cg_bond_operations_single_and_batch() {
        let (mut cg, ids, bonds) = trimer();
        assert_eq!(cg.bond_ids().collect::<Vec<_>>(), bonds);
        cg.set_bond(bonds[0], "r0", 4.7).unwrap();
        assert_eq!(
            cg.get_bond(bonds[0]).unwrap().props.get_f64("r0"),
            Some(4.7)
        );
        cg.clear_bond(bonds[0], "r0").unwrap();
        assert!(cg.get_bond(bonds[0]).unwrap().props.is_empty());

        let gone = cg.del_bond(bonds[0]).unwrap();
        assert_eq!(gone.nodes.as_slice(), &[ids[0], ids[1]]);
        assert!(cg.del_bonds(&[bonds[1], bonds[0]]).is_err());
        assert_eq!(cg.n_bonds(), 1, "a refused batch deletes nothing");
        assert_eq!(cg.del_bonds(&[bonds[1]]).unwrap().len(), 1);
        assert_eq!((cg.n_beads(), cg.n_bonds()), (3, 0));
        assert!(cg.set_bond(bonds[0], "r0", 1.0).is_err());
        assert!(cg.clear_bond(bonds[0], "r0").is_err());
    }

    #[test]
    fn neighbor_bonds_are_cg_bonds_only() {
        let (mut cg, ids, bonds) = trimer();
        cg.add_port(ids[1], ids[2], PortKind::Left, "", BondNumber::Single)
            .unwrap();
        let mut nb: Vec<_> = cg.neighbor_bonds(ids[1]).collect();
        nb.sort();
        let mut want = vec![(ids[0], bonds[0]), (ids[2], bonds[1])];
        want.sort();
        assert_eq!(nb, want);
        assert_eq!(
            cg.graph().neighbors(ids[1]).count(),
            3,
            "the port is there too"
        );
    }

    #[test]
    fn cg_ports_link_and_geometry() {
        let (mut cg, ids, _) = trimer();
        let extra = cg
            .add_bead(
                props! { keys::BEAD_TYPE => "E", keys::X => 5.0, keys::Y => 0.0, keys::Z => 0.0 },
            )
            .unwrap();
        let cap = cg
            .add_bead(
                props! { keys::BEAD_TYPE => "H", keys::X => 6.0, keys::Y => 0.0, keys::Z => 0.0 },
            )
            .unwrap();
        cg.add_bond(extra, cap, props! {}).unwrap();
        let p0 = cg
            .add_port(ids[2], ids[1], PortKind::Left, "x", BondNumber::Single)
            .unwrap();
        let p1 = cg
            .add_port(extra, cap, PortKind::Right, "x", BondNumber::Single)
            .unwrap();
        assert_eq!(cg.port_ids().collect::<Vec<_>>(), [p0, p1]);
        assert_eq!(cg.port_on(extra, cap), Some(p1));
        let port = cg.get_port(p1).unwrap();
        assert_eq!(cg.leaving_group(&port).unwrap(), BTreeSet::from([cap]));
        cg.del_port(p0).unwrap();
        assert_eq!(cg.n_ports(), 1);
        assert!(cg.del_ports(&[p0]).is_err());

        cg.translate([1.0, 0.0, 0.0]);
        assert_eq!(cg.positions(&[ids[0]]).unwrap(), vec![[1.0, 0.0, 0.0]]);
        cg.rotate([0.0, 0.0, 1.0], std::f64::consts::PI, None)
            .unwrap();
        let [x, y, _] = cg.positions(&[ids[0]]).unwrap()[0];
        assert!((x + 1.0).abs() < 1e-12 && y.abs() < 1e-12);
    }

    #[test]
    fn cg_hashing_through_the_methods() {
        let (a, _, _) = trimer();
        let (b, _, _) = trimer();
        assert_eq!(a.structural_hash(), b.structural_hash());
        assert!(a.is_isomorphic(&b));
        assert_eq!(a.canonical_order().len(), 3);
    }
}
