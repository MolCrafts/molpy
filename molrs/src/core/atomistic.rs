//! All-atom molecular graph: [`Atomistic`].

use std::collections::HashMap;
use std::ops::{Deref, DerefMut};

use crate::core::Frame;
use crate::core::MolRsError;
use crate::core::bond_order::stamp_bond_class;
use crate::core::keys;
use crate::core::{Atom, KindId, MolGraph, NodeId, PropValue, Relation, RelationId};
use crate::core::{BondNumber, BondOrder};

/// Result of [`Atomistic::extract_subgraph`].
#[derive(Debug, Clone)]
pub struct ExtractedAtomistic {
    /// The extracted all-atom graph.
    pub graph: Atomistic,
    /// Selected parent atoms with a bond-neighbor outside the ball.
    pub boundary: Vec<NodeId>,
    /// New atom id → parent atom id.
    pub parent_of: HashMap<NodeId, NodeId>,
    /// Parent atom id → hops from nearest center.
    pub hops: HashMap<NodeId, i64>,
    /// Parent atom id → new atom id.
    pub node_map: HashMap<NodeId, NodeId>,
}

/// All-atom molecular graph with element-level chemistry semantics.
///
/// `Atomistic` wraps the domain-agnostic [`MolGraph`] and owns **all** atom /
/// bond / angle / dihedral / improper vocabulary: it registers those relation
/// kinds at construction (caching their [`KindId`]s) and exposes the typed
/// convenience API (`add_bond`, `bonds`, `get_bond`, …). `MolGraph` itself knows
/// nothing of bonds — the chemistry lives here.
///
/// Generic graph methods (`nodes`, `neighbors`, `add_relation`, …) remain
/// available via `Deref`/`DerefMut`; the coordinate transforms are
/// [`MolGraph::translate`], [`MolGraph::rotate`], [`MolGraph::scale`] and
/// [`MolGraph::center`].
///
/// Invariant: every atom carries the canonical [`keys::ELEMENT`] property.
///
/// # Examples
///
/// ```
/// use molrs::core::Atomistic;
///
/// let mut mol = Atomistic::new();
/// let c = mol.add_atom_bare("C");
/// let h = mol.add_atom_bare("H");
/// mol.add_bond(c, h).unwrap();
///
/// assert_eq!(mol.n_atoms(), 2);
/// assert_eq!(mol.n_bonds(), 1);
/// ```
#[derive(Debug, Clone)]
pub struct Atomistic {
    graph: MolGraph,
    bond: KindId,
    angle: KindId,
    dihedral: KindId,
    improper: KindId,
}

impl Deref for Atomistic {
    type Target = MolGraph;
    fn deref(&self) -> &MolGraph {
        &self.graph
    }
}

impl DerefMut for Atomistic {
    fn deref_mut(&mut self) -> &mut MolGraph {
        &mut self.graph
    }
}

impl Default for Atomistic {
    fn default() -> Self {
        Self::new()
    }
}

impl Atomistic {
    /// Create an empty all-atom molecular graph with the bond / angle /
    /// dihedral / improper kinds registered.
    pub fn new() -> Self {
        let mut graph = MolGraph::new();
        let bond = graph.register_kind("bonds", 2);
        let angle = graph.register_kind("angles", 3);
        let dihedral = graph.register_kind("dihedrals", 4);
        let improper = graph.register_kind("impropers", 4);
        Self {
            graph,
            bond,
            angle,
            dihedral,
            improper,
        }
    }

    // ---- atoms (nodes) ----

    /// Add an atom carrying a property bag.
    ///
    /// # Panics
    ///
    /// Panics when a value of `atom` contradicts the element type an existing
    /// atom column holds for that key (a string `charge` into an `f64`
    /// `charge` column). The caller of this constructor *built* the bag, so
    /// that is a defect in the caller and not a data condition; callers
    /// holding a **foreign** bag reach
    /// [`MolGraph::add_node_with`](crate::core::MolGraph::add_node_with)
    /// through [`as_molgraph_mut`](Self::as_molgraph_mut), which returns the
    /// conflict instead.
    pub fn add_atom(&mut self, atom: Atom) -> NodeId {
        self.graph
            .add_node_with(atom)
            .expect("caller-built atom bag contradicts an existing atom column")
    }

    /// Add an atom with element symbol and 3D coordinates.
    ///
    /// # Panics
    ///
    /// Panics when the `element` / `x` / `y` / `z` columns already hold a
    /// different element type — see [`add_atom`](Self::add_atom).
    pub fn add_atom_xyz(&mut self, symbol: &str, x: f64, y: f64, z: f64) -> NodeId {
        self.add_atom(Atom::xyz(symbol, x, y, z))
    }

    /// Add an atom with element symbol only (no coordinates).
    ///
    /// Writes the chemical identity under the canonical [`keys::ELEMENT`] field
    /// (not a format alias such as `"symbol"`).
    ///
    /// # Panics
    ///
    /// Panics when the `element` column already holds a different element
    /// type — see [`add_atom`](Self::add_atom).
    pub fn add_atom_bare(&mut self, symbol: &str) -> NodeId {
        let mut a = Atom::new();
        a.set(keys::ELEMENT, symbol);
        self.add_atom(a)
    }

    /// Remove an atom and all incident bonds / angles / dihedrals / impropers.
    pub fn remove_atom(&mut self, id: NodeId) -> Result<Atom, MolRsError> {
        self.graph.remove_node(id)
    }

    /// Materialize an atom's property bag (owned copy of its set components).
    pub fn get_atom(&self, id: NodeId) -> Result<Atom, MolRsError> {
        self.graph.get_node(id)
    }

    /// Set a single component on an atom.
    pub fn set_atom(
        &mut self,
        id: NodeId,
        key: &str,
        val: impl Into<PropValue>,
    ) -> Result<(), MolRsError> {
        self.graph.set_node(id, key, val)
    }

    /// Clear a single component on an atom (no-op if absent).
    pub fn clear_atom(&mut self, id: NodeId, key: &str) -> Result<(), MolRsError> {
        self.graph.clear_node(id, key)
    }

    /// Iterate over all `(NodeId, Atom)` pairs (each property bag materialized).
    pub fn atoms(&self) -> impl Iterator<Item = (NodeId, Atom)> + '_ {
        self.graph.nodes()
    }

    /// Number of atoms.
    pub fn n_atoms(&self) -> usize {
        self.graph.n_nodes()
    }

    // ---- bonds ----

    /// Add a bond between two existing atoms (default order 1.0).
    pub fn add_bond(&mut self, a: NodeId, b: NodeId) -> Result<RelationId, MolRsError> {
        let bid = self.graph.add_relation(self.bond, &[a, b])?;
        self.set_bond_type(bid, BondOrder::Single)?;
        Ok(bid)
    }

    /// The bond's chemical class; [`BondOrder::Unknown`] if it has none.
    pub fn bond_type(&self, id: RelationId) -> BondOrder {
        match self.get_bond(id) {
            Ok(b) => BondOrder::from_prop(b.props.get(keys::BOND_TYPE)),
            Err(_) => BondOrder::Unknown,
        }
    }

    /// The bond's localized (Kekulé) bond number; [`BondNumber::Unknown`] if it
    /// has none — an aromatic bond before kekulization, for instance.
    pub fn bond_number(&self, id: RelationId) -> BondNumber {
        match self.get_bond(id) {
            Ok(b) => BondNumber::from_prop(b.props.get(keys::BOND_NUMBER)),
            Err(_) => BondNumber::Unknown,
        }
    }

    /// Set both facts about a bond at once.
    ///
    /// They are set together because they are only meaningful together: a class
    /// without a number leaves the bond un-standardized, and a number without a
    /// class leaves a renderer no way to tell aromatic from double.
    ///
    /// # Errors
    ///
    /// Returns [`MolRsError::NotFound`] when `id` names no live bond.
    pub fn set_bond_class(
        &mut self,
        id: RelationId,
        bond_type: BondOrder,
        bond_number: BondNumber,
    ) -> Result<(), MolRsError> {
        stamp_bond_class(&mut self.graph, self.bond, id, bond_type, bond_number)
    }

    /// Set a plain (non-aromatic) bond, whose class implies its number.
    ///
    /// `Aromatic` has no implied number, so it must go through
    /// [`set_bond_class`](Self::set_bond_class) with the number a Kekulé
    /// assignment decided.
    pub fn set_bond_type(
        &mut self,
        id: RelationId,
        bond_type: BondOrder,
    ) -> Result<(), MolRsError> {
        let number = bond_type.implied_number().unwrap_or_default();
        self.set_bond_class(id, bond_type, number)
    }

    /// Remove a bond.
    pub fn remove_bond(&mut self, id: RelationId) -> Result<Relation, MolRsError> {
        self.graph.remove_relation(self.bond, id)
    }

    /// Materialize a bond (endpoints + properties).
    pub fn get_bond(&self, id: RelationId) -> Result<Relation, MolRsError> {
        self.graph.get_relation(self.bond, id)
    }

    /// Set a single property on a bond.
    pub fn set_bond_prop(
        &mut self,
        id: RelationId,
        key: &str,
        val: impl Into<PropValue>,
    ) -> Result<(), MolRsError> {
        self.graph.set_relation_prop(self.bond, id, key, val)
    }

    /// Iterate over all `(RelationId, Relation)` pairs (each materialized).
    pub fn bonds(&self) -> impl Iterator<Item = (RelationId, Relation)> + '_ {
        self.graph.relations(self.bond)
    }

    /// Number of bonds.
    pub fn n_bonds(&self) -> usize {
        self.graph.n_relations(self.bond)
    }

    /// Iterate over `(neighbor_id, bond_id)` for a given atom.
    ///
    /// Yields the bond handle rather than a number: a caller that wants the
    /// localized count asks [`bond_number`](Self::bond_number), and one that
    /// wants to know whether the bond is aromatic asks
    /// [`bond_type`](Self::bond_type). Handing back a single float is what let
    /// those two questions be answered by the same value.
    pub fn neighbor_bonds(&self, id: NodeId) -> impl Iterator<Item = (NodeId, RelationId)> + '_ {
        self.graph
            .neighbor_relations(id)
            .filter(move |(kind, _, _)| *kind == self.bond)
            .map(move |(_, rid, other)| (other, rid))
    }

    /// Endpoints `(a, b)` of a bond by id, without materializing its property
    /// map (reads only the relation's endpoint list).
    pub fn bond_endpoints(&self, id: RelationId) -> Option<(NodeId, NodeId)> {
        self.graph
            .relation_nodes(self.bond, id)
            .ok()
            .map(|eps| (eps[0], eps[1]))
    }

    /// Iterate `(RelationId, neighbor_id)` incident to `id` via the adjacency index
    /// (O(degree)), without materializing each bond's property map. The bond
    /// order, if needed, is looked up separately by the caller.
    pub fn incident_bond_ids(&self, id: NodeId) -> impl Iterator<Item = (RelationId, NodeId)> + '_ {
        let bond = self.bond;
        self.graph
            .neighbor_relations(id)
            .filter_map(move |(kind, rid, other)| (kind == bond).then_some((rid, other)))
    }

    // ---- angles ----

    /// Add an angle (i-j-k, j central).
    pub fn add_angle(&mut self, i: NodeId, j: NodeId, k: NodeId) -> Result<RelationId, MolRsError> {
        self.graph.add_relation(self.angle, &[i, j, k])
    }

    /// Remove an angle.
    pub fn remove_angle(&mut self, id: RelationId) -> Result<Relation, MolRsError> {
        self.graph.remove_relation(self.angle, id)
    }

    /// Materialize an angle (endpoints + properties).
    pub fn get_angle(&self, id: RelationId) -> Result<Relation, MolRsError> {
        self.graph.get_relation(self.angle, id)
    }

    /// Set a single property on an angle.
    pub fn set_angle_prop(
        &mut self,
        id: RelationId,
        key: &str,
        val: impl Into<PropValue>,
    ) -> Result<(), MolRsError> {
        self.graph.set_relation_prop(self.angle, id, key, val)
    }

    /// Iterate over all `(RelationId, Relation)` pairs (each materialized).
    pub fn angles(&self) -> impl Iterator<Item = (RelationId, Relation)> + '_ {
        self.graph.relations(self.angle)
    }

    /// Number of angles.
    pub fn n_angles(&self) -> usize {
        self.graph.n_relations(self.angle)
    }

    // ---- dihedrals ----

    /// Add a dihedral (i-j-k-l).
    pub fn add_dihedral(
        &mut self,
        i: NodeId,
        j: NodeId,
        k: NodeId,
        l: NodeId,
    ) -> Result<RelationId, MolRsError> {
        self.graph.add_relation(self.dihedral, &[i, j, k, l])
    }

    /// Remove a dihedral.
    pub fn remove_dihedral(&mut self, id: RelationId) -> Result<Relation, MolRsError> {
        self.graph.remove_relation(self.dihedral, id)
    }

    /// Materialize a dihedral (endpoints + properties).
    pub fn get_dihedral(&self, id: RelationId) -> Result<Relation, MolRsError> {
        self.graph.get_relation(self.dihedral, id)
    }

    /// Set a single property on a dihedral.
    pub fn set_dihedral_prop(
        &mut self,
        id: RelationId,
        key: &str,
        val: impl Into<PropValue>,
    ) -> Result<(), MolRsError> {
        self.graph.set_relation_prop(self.dihedral, id, key, val)
    }

    /// Iterate over all `(RelationId, Relation)` pairs (each materialized).
    pub fn dihedrals(&self) -> impl Iterator<Item = (RelationId, Relation)> + '_ {
        self.graph.relations(self.dihedral)
    }

    /// Number of dihedrals.
    pub fn n_dihedrals(&self) -> usize {
        self.graph.n_relations(self.dihedral)
    }

    /// Perceive angle and dihedral relations from the bond graph.
    ///
    /// Builds a [`Topology`](crate::core::Topology) (a native
    /// adjacency snapshot) from the current bonds and reuses its
    /// `angles()` / `dihedrals()` enumeration — angles are 2-edge paths
    /// `i-j-k` (deduplicated `i < k`), proper dihedrals are 3-edge paths
    /// `i-j-k-l` (each central edge once). `Atomistic` is just the domain leaf
    /// that names the graph-theoretic result.
    ///
    /// Impropers follow the molecular-mechanics reading —
    /// [`Topology::trivalent_impropers`](crate::core::Topology::trivalent_impropers),
    /// one `[centre, i, j, k]` quartet per
    /// atom with exactly three neighbours — not the geometric enumeration of
    /// every 3-combination. Whether such a centre is planar enough to deserve
    /// the term is force-field data, not a graph property, so every trivalent
    /// centre is emitted and the selection belongs to the layer with the table.
    ///
    /// Idempotent: an angle/dihedral/improper already present (by canonical
    /// endpoints) is not duplicated. With `clear_existing`, all existing
    /// relations of the requested kinds are removed first. Returns
    /// `(n_angles_added, n_dihedrals_added, n_impropers_added)`.
    pub fn generate_topology(
        &mut self,
        gen_angle: bool,
        gen_dihedral: bool,
        gen_improper: bool,
        clear_existing: bool,
    ) -> Result<(usize, usize, usize), MolRsError> {
        use crate::core::Topology;

        if clear_existing {
            if gen_angle {
                let ids: Vec<_> = self.graph.relation_ids(self.angle).collect();
                for id in ids {
                    self.graph.remove_relation(self.angle, id)?;
                }
            }
            if gen_dihedral {
                let ids: Vec<_> = self.graph.relation_ids(self.dihedral).collect();
                for id in ids {
                    self.graph.remove_relation(self.dihedral, id)?;
                }
            }
            if gen_improper {
                let ids: Vec<_> = self.graph.relation_ids(self.improper).collect();
                for id in ids {
                    self.graph.remove_relation(self.improper, id)?;
                }
            }
        }

        // Build the native topology snapshot from bonds (atoms in node-id order).
        let atoms: Vec<NodeId> = self.graph.node_ids().collect();
        let pos: std::collections::HashMap<NodeId, usize> =
            atoms.iter().enumerate().map(|(i, &a)| (a, i)).collect();
        let mut edges: Vec<[usize; 2]> = Vec::new();
        for id in self.graph.relation_ids(self.bond) {
            let n = self.graph.relation_nodes(self.bond, id)?;
            if n.len() == 2 {
                edges.push([pos[&n[0]], pos[&n[1]]]);
            }
        }
        let mut topo = Topology::with_atoms(atoms.len());
        topo.add_bonds(&edges);

        let mut n_ang = 0usize;
        let mut n_dih = 0usize;
        let mut n_imp = 0usize;

        if gen_angle {
            let mut seen: std::collections::HashSet<Vec<NodeId>> = std::collections::HashSet::new();
            for id in self.graph.relation_ids(self.angle) {
                seen.insert(canonical_path(&self.graph.relation_nodes(self.angle, id)?));
            }
            for a in topo.angles() {
                let nodes = [atoms[a[0]], atoms[a[1]], atoms[a[2]]];
                if seen.insert(canonical_path(&nodes)) {
                    self.add_angle(nodes[0], nodes[1], nodes[2])?;
                    n_ang += 1;
                }
            }
        }

        if gen_dihedral {
            let mut seen: std::collections::HashSet<Vec<NodeId>> = std::collections::HashSet::new();
            for id in self.graph.relation_ids(self.dihedral) {
                seen.insert(canonical_path(
                    &self.graph.relation_nodes(self.dihedral, id)?,
                ));
            }
            for d in topo.dihedrals() {
                let nodes = [atoms[d[0]], atoms[d[1]], atoms[d[2]], atoms[d[3]]];
                if seen.insert(canonical_path(&nodes)) {
                    self.add_dihedral(nodes[0], nodes[1], nodes[2], nodes[3])?;
                    n_dih += 1;
                }
            }
        }

        if gen_improper {
            let mut seen: std::collections::HashSet<Vec<NodeId>> = std::collections::HashSet::new();
            for id in self.graph.relation_ids(self.improper) {
                seen.insert(canonical_improper(
                    &self.graph.relation_nodes(self.improper, id)?,
                ));
            }
            for q in topo.trivalent_impropers() {
                let nodes = [atoms[q[0]], atoms[q[1]], atoms[q[2]], atoms[q[3]]];
                if seen.insert(canonical_improper(&nodes)) {
                    self.add_improper(nodes[0], nodes[1], nodes[2], nodes[3])?;
                    n_imp += 1;
                }
            }
        }

        Ok((n_ang, n_dih, n_imp))
    }

    /// BFS shortest-path distances over the bond graph from `source`, as
    /// `(atom_id, hops)` pairs for every atom reachable from `source`
    /// (including `source` at distance 0). Unreachable atoms are omitted; an
    /// unknown `source` yields an empty vector.
    ///
    /// The BFS runs directly on the native `MolGraph` adjacency (relations
    /// filtered to the bond kind), keyed by a `SecondaryMap` (O(1),
    /// slot-indexed) — no `Topology` materialization and no
    /// NodeId→contiguous-index remap.
    pub fn topo_distances(&self, source: NodeId, max_hops: Option<i64>) -> Vec<(NodeId, i64)> {
        use slotmap::SecondaryMap;
        use std::collections::VecDeque;

        if self.graph.get_node(source).is_err() {
            return Vec::new();
        }
        let mut dist: SecondaryMap<NodeId, i64> = SecondaryMap::new();
        dist.insert(source, 0);
        let mut queue = VecDeque::new();
        queue.push_back(source);
        while let Some(cur) = queue.pop_front() {
            let d = dist[cur];
            if max_hops.is_some_and(|limit| d >= limit) {
                continue; // atom is at the boundary; do not expand past the radius
            }
            for (kind, _rid, other) in self.graph.neighbor_relations(cur) {
                if kind == self.bond && !dist.contains_key(other) {
                    dist.insert(other, d + 1);
                    queue.push_back(other);
                }
            }
        }
        dist.into_iter().collect()
    }

    // ---- impropers ----

    /// Add an improper dihedral `i-j-k-l`, in the atom order its style prices
    /// (molrs's convention, LAMMPS's): LAMMPS's own order for `harmonic`,
    /// `cvff` and the out-of-plane styles — the centre first — and AMBER's for
    /// `improper periodic` — the centre third. The topology perception
    /// ([`generate_topology`](Self::generate_topology)) lists the centre first.
    pub fn add_improper(
        &mut self,
        i: NodeId,
        j: NodeId,
        k: NodeId,
        l: NodeId,
    ) -> Result<RelationId, MolRsError> {
        self.graph.add_relation(self.improper, &[i, j, k, l])
    }

    /// Remove an improper.
    pub fn remove_improper(&mut self, id: RelationId) -> Result<Relation, MolRsError> {
        self.graph.remove_relation(self.improper, id)
    }

    /// Materialize an improper (endpoints + properties).
    pub fn get_improper(&self, id: RelationId) -> Result<Relation, MolRsError> {
        self.graph.get_relation(self.improper, id)
    }

    /// Set a single property on an improper.
    pub fn set_improper_prop(
        &mut self,
        id: RelationId,
        key: &str,
        val: impl Into<PropValue>,
    ) -> Result<(), MolRsError> {
        self.graph.set_relation_prop(self.improper, id, key, val)
    }

    /// Iterate over all `(RelationId, Relation)` pairs (each materialized).
    pub fn impropers(&self) -> impl Iterator<Item = (RelationId, Relation)> + '_ {
        self.graph.relations(self.improper)
    }

    /// Number of impropers.
    pub fn n_impropers(&self) -> usize {
        self.graph.n_relations(self.improper)
    }

    // ---- kind handles (for callers that go through the generic API) ----

    /// The bond relation kind.
    pub fn bond_kind(&self) -> KindId {
        self.bond
    }

    // ---- frame conversion ----

    /// Export to a tabular [`Frame`] (atoms / bonds / angles / dihedrals /
    /// impropers blocks, and one block per other non-empty relation kind,
    /// `drudes` or `virtual_sites` say, named by the kind).
    ///
    /// # Errors
    ///
    /// [`MolRsError::Validation`] when an atom or relation property
    /// contradicts the dtype the Frame schema declares for its key; the
    /// message names the refused column. [`set_atom`](Self::set_atom) accepts
    /// any value under a key the graph has no column for, so a string written
    /// under `"x"` is legal in the graph and only refused here.
    pub fn to_frame(&self) -> Result<Frame, MolRsError> {
        self.graph.to_frame()
    }

    /// Build from a [`Frame`], registering the standard kinds first so bond /
    /// angle / dihedral / improper blocks are read back.
    ///
    /// Every other relation block of the schema vocabulary whose endpoints
    /// are `atoms` rows ([`schema::block`](crate::core::schema::block):
    /// `drudes`, `virtual_sites`, `constraints`, `cmaps`, `pairs`,
    /// `exclusions`) is registered as a kind of its name and arity when the
    /// frame carries it, so a frame → graph → frame round trip keeps every
    /// relation [`to_frame`](Self::to_frame) writes. A block outside the
    /// vocabulary is not the graph's and is ignored.
    ///
    /// # Bond classes are read, never inferred
    ///
    /// A `bonds` block states its classes per bond in a [`keys::BOND_TYPE`]
    /// column, and this reader reproduces exactly what is there. Where the
    /// column is **absent** — a PDB `CONECT` list, a GROMACS `.top`, an XYZ
    /// `Connct` extension, all of which carry connectivity and no orders —
    /// every bond stays [`BondOrder::Unknown`]. That is the honest reading:
    /// the file did not say, so neither does the graph.
    ///
    /// Consumers that need a class where none was stated apply their own
    /// fallback. A conformer search may reasonably treat an unclassed bond as
    /// rotatable; an aromaticity perceiver must not. That decision belongs to
    /// them, not here — inferring `Single` at read time would hand every
    /// consumer a guess indistinguishable from a fact.
    pub fn from_frame(frame: &Frame) -> Result<Self, MolRsError> {
        let mut mol = Self::new();
        for name in frame.keys() {
            if let Some(arity) = atom_relation_arity(name) {
                mol.graph.try_register_kind(name, arity)?;
            }
        }
        mol.graph.extend_from_frame(frame)?;
        Ok(mol)
    }

    // ---- conversions ----

    /// Promote from a [`MolGraph`], validating all atoms have [`keys::ELEMENT`].
    ///
    /// The graph's relation kinds are re-registered to the standard set **by
    /// name**: an existing `bonds` / `angles` / `dihedrals` / `impropers` kind
    /// of the right arity keeps its id, and a missing one is registered fresh.
    /// Resolving by dense id instead would report a foreign kind's relations as
    /// this molecule's bonds whenever the graph registered something else first
    /// (a `ports` kind, say), and the next `add_bond` would write into it.
    ///
    /// # Errors
    ///
    /// Returns [`MolRsError::Validation`] when an atom carries no
    /// [`keys::ELEMENT`], and when the graph already spells one of the standard
    /// kind names at a different arity (naming the kind and both arities).
    pub fn try_from_molgraph(mut mol: MolGraph) -> Result<Self, MolRsError> {
        for (id, atom) in mol.nodes() {
            if atom.get_str(keys::ELEMENT).is_none() {
                return Err(MolRsError::validation(format!(
                    "node {:?} missing '{}' property",
                    id,
                    keys::ELEMENT
                )));
            }
        }
        let bond = mol.try_register_kind("bonds", 2)?;
        let angle = mol.try_register_kind("angles", 3)?;
        let dihedral = mol.try_register_kind("dihedrals", 4)?;
        let improper = mol.try_register_kind("impropers", 4)?;
        Ok(Self {
            graph: mol,
            bond,
            angle,
            dihedral,
            improper,
        })
    }

    /// Unwrap to the inner [`MolGraph`] (zero cost).
    pub fn into_inner(self) -> MolGraph {
        self.graph
    }

    /// Borrow the inner [`MolGraph`].
    pub fn as_molgraph(&self) -> &MolGraph {
        &self.graph
    }

    /// Mutably borrow the inner [`MolGraph`].
    pub fn as_molgraph_mut(&mut self) -> &mut MolGraph {
        &mut self.graph
    }

    /// Place `transforms.len()` rigid copies of `template`, copy `c` moved by
    /// `transforms[c]` and stamped `frag_id = frag_ids[c]`; returns the new
    /// atoms copy-major. Column-wise and atomic — see
    /// [`MolGraph::replicate`](crate::core::MolGraph::replicate).
    ///
    /// # Errors
    ///
    /// The errors of [`MolGraph::replicate`](crate::core::MolGraph::replicate);
    /// `self` is unchanged then.
    pub fn replicate(
        &mut self,
        template: &Atomistic,
        transforms: &[crate::op::Rigid],
        frag_ids: &[crate::op::I],
    ) -> Result<Vec<NodeId>, MolRsError> {
        self.graph.replicate(&template.graph, transforms, frag_ids)
    }

    // ---- subgraph extraction (see [`crate::core::extract`]) ----

    /// Induced subgraph on an explicit atom set. Stale handles fail-fast.
    /// Returns `(subgraph, parent→new handle map)`.
    ///
    /// Re-wraps via [`Self::try_from_molgraph`] so the `"element"` invariant is
    /// enforced — chemical identity is the canonical ELEMENT field, never a
    /// format alias such as `"symbol"`.
    pub fn induced_subgraph(
        &self,
        atoms: &[NodeId],
    ) -> Result<(Atomistic, HashMap<NodeId, NodeId>), MolRsError> {
        let induced = self.graph.induced_subgraph(atoms)?;
        let atomistic = Atomistic::try_from_molgraph(induced.graph)?;
        Ok((atomistic, induced.node_map))
    }

    /// Radius ball around `centers` over the bond graph.
    ///
    /// When `regenerate_topology` is true, only bonds are copied from the parent
    /// and angles/dihedrals are perceived on the ball (O(ball)). When false,
    /// higher-order terms fully contained in the ball are copied (may scan the
    /// parent's higher-order tables — small-graph / verbatim path).
    pub fn extract_subgraph(
        &self,
        centers: &[NodeId],
        radius: i64,
        regenerate_topology: bool,
        whole_groups: &[Vec<NodeId>],
    ) -> Result<ExtractedAtomistic, MolRsError> {
        let ball = self.graph.extract_ball(
            centers,
            radius,
            self.bond,
            /* copy_higher_order */ !regenerate_topology,
            whole_groups,
        )?;
        let mut atomistic = Atomistic::try_from_molgraph(ball.graph)?;
        if regenerate_topology {
            atomistic.generate_topology(true, true, false, false)?;
        }
        Ok(ExtractedAtomistic {
            graph: atomistic,
            boundary: ball.boundary,
            parent_of: ball.parent_of,
            hops: ball.hops,
            node_map: ball.node_map,
        })
    }

    /// Structural merge of `other` into `self`. Returns `handle in other → handle
    /// in self`. Handles are remapped (not identity-preserving).
    ///
    /// # Errors
    ///
    /// [`MolRsError::Validation`] when an atom or bond property of `other`
    /// contradicts the element type `self` holds for that key — see
    /// [`MolGraph::merge`](crate::core::MolGraph::merge), whose
    /// partial-write contract this inherits.
    pub fn merge(&mut self, other: Atomistic) -> Result<HashMap<NodeId, NodeId>, MolRsError> {
        self.graph.merge(other.graph)
    }

    /// Independent deep copy. **Handles are preserved** (same generational keys).
    pub fn copy(&self) -> Self {
        self.clone()
    }

    // ---- structural graph hash (see [`crate::core::graph_hash`]) ----

    /// Isomorphism-invariant Weisfeiler–Lehman structural hash of the molecule
    /// (element / charge / aromatic node labels, bond-order edge labels). A
    /// stable, reproducible dedup key — identical for a node-permuted copy.
    pub fn structural_hash(&self) -> u64 {
        crate::core::structural_hash(&self.graph)
    }

    /// Deterministic canonical atom ordering from the WL refinement, so two
    /// isomorphic molecules line up node-by-node (see
    /// [`crate::core::canonical_order`]).
    pub fn canonical_order(&self) -> Vec<NodeId> {
        crate::core::canonical_order(&self.graph)
    }

    /// Whether `self` and `other` are isomorphic as labeled molecular graphs.
    pub fn is_isomorphic(&self, other: &Atomistic) -> bool {
        crate::core::is_isomorphic(&self.graph, &other.graph)
    }

    // Aromaticity perception belongs to `crate::perceive::assign_aromaticity`
    // (`assign_aromaticity`). No algorithm method here.
}

/// Canonical (orientation-independent) key for an angle/dihedral endpoint
/// sequence: the lexicographically smaller of the sequence and its reverse.
/// Matches the canonicalization in
/// [`MolGraph::paths_of_length`](crate::core::MolGraph::paths_of_length).
/// Canonical key of an improper: the centre stays first, the peripherals are a
/// set. An improper is symmetric under permuting its outer legs, not under
/// reversal, so [`canonical_path`] is the wrong key for one.
fn canonical_improper(nodes: &[NodeId]) -> Vec<NodeId> {
    if nodes.is_empty() {
        return Vec::new();
    }
    let mut out = vec![nodes[0]];
    let mut rest = nodes[1..].to_vec();
    rest.sort_unstable();
    out.extend(rest);
    out
}

fn canonical_path(nodes: &[NodeId]) -> Vec<NodeId> {
    let fwd = nodes.to_vec();
    let mut rev = fwd.clone();
    rev.reverse();
    if fwd <= rev { fwd } else { rev }
}

impl crate::core::FromMolGraph for Atomistic {
    fn from_molgraph(graph: MolGraph) -> Result<Self, MolRsError> {
        Atomistic::try_from_molgraph(graph)
    }
}

/// The arity of `name` when the schema declares it a relation block whose
/// endpoints are all `atoms` rows (`bonds`, `drudes`, `virtual_sites`, …);
/// `None` for any other block, `members` (its `atom` endpoint is declared per
/// block) among them.
fn atom_relation_arity(name: &str) -> Option<usize> {
    use crate::core::schema::{EndpointTarget, RowKind, block, block_names};
    let spec = block(name)?;
    let RowKind::Relation { arity } = spec.row_kind else {
        return None;
    };
    let endpoints = spec.endpoints?;
    endpoints
        .columns
        .iter()
        .all(|(_, target)| matches!(target, EndpointTarget::Block(b) if *b == block_names::ATOMS))
        .then_some(arity)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::core::Atom;
    use crate::core::Block;
    use crate::op::{I, Idx};
    use ndarray::Array1;
    use std::collections::HashSet;

    #[test]
    fn merge_returns_complete_node_map() {
        let mut a = Atomistic::new();
        let c1 = a.add_atom_bare("C");
        let c2 = a.add_atom_bare("C");
        a.add_bond(c1, c2).unwrap();

        let mut b = Atomistic::new();
        let o = b.add_atom_bare("O");
        let h = b.add_atom_bare("H");
        b.add_bond(o, h).unwrap();

        let map = a.merge(b).expect("merge succeeds on compatible graphs");
        assert_eq!(map.len(), 2);
        assert_eq!(a.n_atoms(), 4);
        assert_eq!(a.n_bonds(), 2);
        let new_o = map[&o];
        let new_h = map[&h];
        assert!(a.get_atom(new_o).unwrap().get_str("element") == Some("O"));
        assert!(a.get_atom(new_h).unwrap().get_str("element") == Some("H"));
        // remapped bond endpoints resolve
        let mut ends = HashSet::new();
        for (_, bond) in a.bonds() {
            ends.extend(bond.nodes.iter().copied());
        }
        assert!(ends.contains(&new_o) && ends.contains(&new_h));
    }

    #[test]
    fn copy_preserves_handles() {
        let mut mol = Atomistic::new();
        let c = mol.add_atom_bare("C");
        let h = mol.add_atom_bare("H");
        let bid = mol.add_bond(c, h).unwrap();
        let cloned = mol.copy();
        assert!(cloned.get_atom(c).is_ok());
        assert!(cloned.get_atom(h).is_ok());
        assert!(cloned.get_bond(bid).is_ok());
    }

    #[test]
    fn test_new_and_add() {
        let mut mol = Atomistic::new();
        let c = mol.add_atom_bare("C");
        let h = mol.add_atom_bare("H");
        mol.add_bond(c, h).unwrap();
        assert_eq!(mol.n_atoms(), 2);
        assert_eq!(mol.n_bonds(), 1);
        assert_eq!(mol.get_atom(c).unwrap().get_str("element"), Some("C"));
    }

    #[test]
    fn test_add_atom_xyz() {
        let mut mol = Atomistic::new();
        let o = mol.add_atom_xyz("O", 0.0, 0.0, 0.0);
        let atom = mol.get_atom(o).unwrap();
        assert_eq!(atom.get_str("element"), Some("O"));
        assert_eq!(atom.get_f64("x"), Some(0.0));
    }

    /// Ethane (C2H6): C0-C1 plus 3 H on each carbon. 7 bonds.
    fn ethane() -> Atomistic {
        let mut mol = Atomistic::new();
        let atoms: Vec<NodeId> = ["C", "C", "H", "H", "H", "H", "H", "H"]
            .iter()
            .map(|e| mol.add_atom_bare(e))
            .collect();
        for (i, j) in [(0, 1), (0, 2), (0, 3), (0, 4), (1, 5), (1, 6), (1, 7)] {
            mol.add_bond(atoms[i], atoms[j]).unwrap();
        }
        mol
    }

    #[test]
    fn generate_topology_ethane_counts() {
        let mut mol = ethane();
        let (n_ang, n_dih, _) = mol.generate_topology(true, true, false, false).unwrap();
        // Angles: C0 centre C(1,2,3) -> C(4,2,3)... 2 C-centres each with 4
        // neighbours -> 2*C(4,2)=12. Dihedrals across the C0-C1 bond: 3*3 = 9.
        assert_eq!(n_ang, 12, "ethane angles");
        assert_eq!(n_dih, 9, "ethane dihedrals");
        assert_eq!(mol.n_angles(), 12);
        assert_eq!(mol.n_dihedrals(), 9);
    }

    #[test]
    fn generate_topology_is_idempotent() {
        let mut mol = ethane();
        mol.generate_topology(true, true, false, false).unwrap();
        // Second call adds nothing (already present).
        let (n_ang, n_dih, _) = mol.generate_topology(true, true, false, false).unwrap();
        assert_eq!((n_ang, n_dih), (0, 0));
        assert_eq!(mol.n_angles(), 12);
        assert_eq!(mol.n_dihedrals(), 9);
    }

    #[test]
    fn generate_topology_clear_existing_regenerates() {
        let mut mol = ethane();
        mol.generate_topology(true, true, false, false).unwrap();
        let (n_ang, n_dih, _) = mol.generate_topology(true, true, false, true).unwrap();
        // clear_existing wipes then regenerates the identical set.
        assert_eq!((n_ang, n_dih), (12, 9));
        assert_eq!(mol.n_angles(), 12);
        assert_eq!(mol.n_dihedrals(), 9);
    }

    /// A trivalent centre: C bonded to O, H, H.
    fn formaldehyde() -> Atomistic {
        let mut mol = Atomistic::new();
        let atoms: Vec<NodeId> = ["C", "O", "H", "H"]
            .iter()
            .map(|e| mol.add_atom_bare(e))
            .collect();
        for (i, j) in [(0, 1), (0, 2), (0, 3)] {
            mol.add_bond(atoms[i], atoms[j]).unwrap();
        }
        mol
    }

    #[test]
    fn generate_topology_improper_at_a_trivalent_centre() {
        let mut mol = formaldehyde();
        let (_, _, n_imp) = mol.generate_topology(false, false, true, false).unwrap();
        assert_eq!(n_imp, 1);
        assert_eq!(mol.n_impropers(), 1);
    }

    #[test]
    fn generate_topology_emits_no_improper_at_an_sp3_centre() {
        // Ethane's carbons have four neighbours each. The geometric
        // enumeration would give C(4,3) = 4 quartets per carbon; a force field
        // wants none.
        let mut mol = ethane();
        let (_, _, n_imp) = mol.generate_topology(false, false, true, false).unwrap();
        assert_eq!(n_imp, 0);
        assert_eq!(mol.n_impropers(), 0);
    }

    #[test]
    fn generate_topology_improper_is_idempotent() {
        let mut mol = formaldehyde();
        mol.generate_topology(false, false, true, false).unwrap();
        let (_, _, n_imp) = mol.generate_topology(false, false, true, false).unwrap();
        assert_eq!(n_imp, 0, "an improper already present is not duplicated");
        assert_eq!(mol.n_impropers(), 1);
    }

    #[test]
    fn generate_topology_improper_dedup_ignores_leg_order() {
        // An improper is symmetric under permuting its outer legs, so one
        // written with the legs in another order is the same relation.
        let mut mol = formaldehyde();
        let ids: Vec<NodeId> = mol.atoms().map(|(id, _)| id).collect();
        mol.add_improper(ids[0], ids[3], ids[1], ids[2]).unwrap();
        let (_, _, n_imp) = mol.generate_topology(false, false, true, false).unwrap();
        assert_eq!(n_imp, 0);
        assert_eq!(mol.n_impropers(), 1);
    }

    #[test]
    fn generate_topology_selective() {
        let mut mol = ethane();
        let (n_ang, n_dih, _) = mol.generate_topology(true, false, false, false).unwrap();
        assert_eq!(n_ang, 12);
        assert_eq!(n_dih, 0);
        assert_eq!(mol.n_dihedrals(), 0);
    }

    #[test]
    fn topo_distances_parity() {
        // Native BFS over the bond graph. Asserts the (atom, hops) set directly
        // against the expected hop multisets.
        let hops = |mut v: Vec<(NodeId, i64)>| -> Vec<i64> {
            v.sort();
            v.into_iter().map(|(_, d)| d).collect::<Vec<i64>>()
        };
        let sorted_hops = |v: Vec<(NodeId, i64)>| {
            let mut d: Vec<i64> = v.into_iter().map(|(_, d)| d).collect();
            d.sort_unstable();
            d
        };

        let eth = ethane();
        let atoms: Vec<NodeId> = eth.node_ids().collect();
        // Ethane: C0(0)-C1(1) with H2,H3,H4 on C0 and H5,H6,H7 on C1.
        // From C0: self 0; C1 1; its own H's 1; far H's 2.
        let expected_per_source: [Vec<i64>; 8] = [
            // C0
            vec![0, 1, 1, 1, 1, 2, 2, 2],
            // C1
            vec![1, 0, 2, 2, 2, 1, 1, 1],
            // H2 (on C0)
            vec![1, 2, 0, 2, 2, 3, 3, 3],
            // H3 (on C0)
            vec![1, 2, 2, 0, 2, 3, 3, 3],
            // H4 (on C0)
            vec![1, 2, 2, 2, 0, 3, 3, 3],
            // H5 (on C1)
            vec![2, 1, 3, 3, 3, 0, 2, 2],
            // H6 (on C1)
            vec![2, 1, 3, 3, 3, 2, 0, 2],
            // H7 (on C1)
            vec![2, 1, 3, 3, 3, 2, 2, 0],
        ];
        for (i, &src) in atoms.iter().enumerate() {
            assert_eq!(
                sorted_hops(eth.topo_distances(src, None)),
                sorted_hops(expected_per_source[i].iter().map(|&d| (src, d)).collect()),
                "ethane source {src:?}"
            );
        }

        // 12-atom linear chain: an endpoint sees every hop count 0..=11 once.
        let mut chain = Atomistic::new();
        let ids: Vec<_> = (0..12).map(|_| chain.add_atom_bare("C")).collect();
        for k in 0..ids.len() - 1 {
            chain.add_bond(ids[k], ids[k + 1]).unwrap();
        }
        assert_eq!(
            sorted_hops(chain.topo_distances(ids[0], None)),
            (0..12).collect::<Vec<_>>(),
        );
        // The mid-chain atom 5 sees a symmetric profile.
        assert_eq!(
            hops(chain.topo_distances(ids[0], None)),
            (0..12).collect::<Vec<_>>(),
            "chain endpoint ordered by atom id"
        );

        // Unknown source → empty vec (error path). Use a genuinely-stale id
        // from the same arena (add then remove bumps the slot generation so
        // the old id no longer resolves) — a foreign arena's id can collide on
        // slot+generation and is not a reliable "absent" probe.
        let mut solo = Atomistic::new();
        let stale = solo.add_atom_bare("C");
        solo.remove_atom(stale).unwrap();
        assert!(solo.topo_distances(stale, None).is_empty());
    }

    #[test]
    fn test_full_topology_and_cascade() {
        let mut mol = Atomistic::new();
        let a = mol.add_atom_bare("C");
        let b = mol.add_atom_bare("C");
        let c = mol.add_atom_bare("C");
        let d = mol.add_atom_bare("C");
        mol.add_bond(a, b).unwrap();
        mol.add_bond(a, c).unwrap();
        mol.add_bond(a, d).unwrap();
        mol.add_angle(b, a, c).unwrap();
        mol.add_dihedral(b, a, c, d).unwrap();
        mol.add_improper(b, a, c, d).unwrap();
        assert_eq!(mol.n_bonds(), 3);
        assert_eq!(mol.n_angles(), 1);
        assert_eq!(mol.n_dihedrals(), 1);
        assert_eq!(mol.n_impropers(), 1);
        // typed RelationId usable as a HashSet key
        let ids: std::collections::HashSet<RelationId> = mol.bonds().map(|(id, _)| id).collect();
        assert_eq!(ids.len(), 3);
        // cascade on central atom
        mol.remove_atom(a).unwrap();
        assert_eq!(mol.n_bonds(), 0);
        assert_eq!(mol.n_angles(), 0);
        assert_eq!(mol.n_dihedrals(), 0);
        assert_eq!(mol.n_impropers(), 0);
    }

    #[test]
    fn test_neighbors_and_neighbor_bonds() {
        let mut mol = Atomistic::new();
        let o = mol.add_atom_bare("O");
        let h1 = mol.add_atom_bare("H");
        let h2 = mol.add_atom_bare("H");
        mol.add_bond(o, h1).unwrap();
        mol.add_bond(o, h2).unwrap();
        assert_eq!(mol.neighbors(o).count(), 2);
        let nb: Vec<(NodeId, RelationId)> = mol.neighbor_bonds(o).collect();
        assert_eq!(nb.len(), 2);
        assert!(
            nb.iter()
                .all(|(_, bid)| mol.bond_number(*bid) == BondNumber::Single)
        );
    }

    #[test]
    fn test_frame_roundtrip() {
        let mut mol = Atomistic::new();
        let a = mol.add_atom_xyz("C", 0.0, 0.0, 0.0);
        let b = mol.add_atom_xyz("C", 1.0, 0.0, 0.0);
        let c = mol.add_atom_xyz("C", 0.0, 1.0, 0.0);
        let d = mol.add_atom_xyz("C", 0.0, 0.0, 1.0);
        mol.add_bond(a, b).unwrap();
        mol.add_angle(a, b, c).unwrap();
        mol.add_improper(a, b, c, d).unwrap();
        let frame = mol.to_frame().expect("a schema-conforming graph converts");
        let mol2 = Atomistic::from_frame(&frame).unwrap();
        assert_eq!(mol2.n_atoms(), 4);
        assert_eq!(mol2.n_bonds(), 1);
        assert_eq!(mol2.n_angles(), 1);
        assert_eq!(mol2.n_impropers(), 1);
    }

    /// A hand-built frame carrying only `atomi` / `atomj` has no class column;
    /// every bond must come back `Single`, matching `add_bond`'s default, or
    /// rotatable-bond perception silently sees nothing.
    /// Connectivity without stated orders — a PDB `CONECT` list, a GROMACS
    /// `.top` — must read back `Unknown`, not a guessed `Single`. Consumers
    /// that need a class apply their own fallback; the reader stays faithful
    /// to the file.
    #[test]
    fn from_frame_leaves_unstated_bond_type_unknown() {
        use crate::core::Block;
        use ndarray::Array1;

        let mut atoms = Block::new();
        for k in ["x", "y", "z"] {
            atoms
                .insert(k, Array1::from_vec(vec![0.0_f64, 1.0, 2.0]).into_dyn())
                .unwrap();
        }
        let mut bonds = Block::new();
        bonds
            .insert("atomi", Array1::from_vec(vec![0u64, 1]).into_dyn())
            .unwrap();
        bonds
            .insert("atomj", Array1::from_vec(vec![1u64, 2]).into_dyn())
            .unwrap();
        let mut frame = Frame::new();
        frame.insert("atoms", atoms);
        frame.insert("bonds", bonds);

        let mol = Atomistic::from_frame(&frame).unwrap();
        assert_eq!(mol.n_bonds(), 2, "connectivity is read");
        for (id, _) in mol.bonds() {
            assert_eq!(
                mol.bond_type(id),
                BondOrder::Unknown,
                "an unstated class must not be inferred"
            );
        }
    }

    /// A stated class is read back per bond, exactly as written — including an
    /// explicit `0`, which means "the input said it does not know".
    #[test]
    fn from_frame_keeps_stated_bond_type_per_bond() {
        use crate::core::Block;
        use ndarray::Array1;

        let mut atoms = Block::new();
        for k in ["x", "y", "z"] {
            atoms
                .insert(k, Array1::from_vec(vec![0.0_f64, 1.0, 2.0, 3.0]).into_dyn())
                .unwrap();
        }
        let mut bonds = Block::new();
        bonds
            .insert("atomi", Array1::from_vec(vec![0u64, 1, 2]).into_dyn())
            .unwrap();
        bonds
            .insert("atomj", Array1::from_vec(vec![1u64, 2, 3]).into_dyn())
            .unwrap();
        bonds
            .insert("bond_type", Array1::from_vec(vec![2u64, 4, 0]).into_dyn())
            .unwrap();
        let mut frame = Frame::new();
        frame.insert("atoms", atoms);
        frame.insert("bonds", bonds);

        let mol = Atomistic::from_frame(&frame).unwrap();
        let got: Vec<BondOrder> = mol.bonds().map(|(id, _)| mol.bond_type(id)).collect();
        assert_eq!(
            got,
            vec![BondOrder::Double, BondOrder::Aromatic, BondOrder::Unknown]
        );
    }

    /// A frame is open: a relation block of a kind `Atomistic` does not
    /// register is not its business and is ignored, bonds still read.
    #[test]
    fn from_frame_ignores_a_relation_block_of_an_unregistered_kind() {
        use crate::core::Block;
        use ndarray::Array1;

        let mut mol = Atomistic::new();
        let a = mol.add_atom_xyz("C", 0.0, 0.0, 0.0);
        let b = mol.add_atom_xyz("C", 1.54, 0.0, 0.0);
        mol.add_bond(a, b).unwrap();
        let mut frame = mol.to_frame().expect("a schema-conforming graph converts");
        let mut widgets = Block::new();
        widgets
            .insert("atomi", Array1::from_vec(vec![0_u64]).into_dyn())
            .unwrap();
        widgets
            .insert("atomj", Array1::from_vec(vec![1_u64]).into_dyn())
            .unwrap();
        frame.insert("widgets", widgets);

        let back = Atomistic::from_frame(&frame).expect("an unknown block is ignored");
        assert_eq!(back.n_atoms(), 2);
        assert_eq!(back.n_bonds(), 1);
    }

    /// Every relation block `to_frame` writes comes back through
    /// `from_frame`: a Drude oscillator (`drudes`), a TIP4P site
    /// (`virtual_sites`), a constraint — each with its props.
    #[test]
    fn from_frame_reads_back_every_relation_kind_to_frame_writes() {
        let mut mol = Atomistic::new();
        let o = mol.add_atom_xyz("O", 0.0, 0.0, 0.0);
        let h1 = mol.add_atom_xyz("H", 0.96, 0.0, 0.0);
        let h2 = mol.add_atom_xyz("H", -0.24, 0.93, 0.0);
        let m = mol.add_atom_bare("M");
        let shell = mol.add_atom_bare("D");
        mol.add_bond(o, h1).unwrap();
        mol.add_bond(o, h2).unwrap();
        let drudes = mol.register_kind("drudes", 2);
        let rid = mol.add_relation(drudes, &[o, shell]).unwrap();
        mol.set_relation_prop(drudes, rid, "type", "OW-DOW")
            .unwrap();
        mol.set_relation_prop(drudes, rid, "k", 500.0).unwrap();
        let sites = mol.register_kind("virtual_sites", 4);
        let rid = mol.add_relation(sites, &[m, o, h1, h2]).unwrap();
        mol.set_relation_prop(sites, rid, "style", "average3")
            .unwrap();
        mol.set_relation_prop(sites, rid, "w1", 0.8).unwrap();
        let constraints = mol.register_kind("constraints", 2);
        mol.add_relation(constraints, &[h1, h2]).unwrap();

        let frame = mol.to_frame().unwrap();
        for block in ["drudes", "virtual_sites", "constraints"] {
            assert!(frame.get(block).is_some(), "to_frame writes {block}");
        }
        let back = Atomistic::from_frame(&frame).unwrap();
        let again = back.to_frame().unwrap();
        for block in ["atoms", "bonds", "drudes", "virtual_sites", "constraints"] {
            let (a, b) = (&frame[block], &again[block]);
            let keys: Vec<&str> = a.keys().collect();
            assert_eq!(keys, b.keys().collect::<Vec<_>>(), "{block} columns");
            for key in keys {
                assert_eq!(
                    a.get(key).map(|c| format!("{c:?}")),
                    b.get(key).map(|c| format!("{c:?}")),
                    "{block}.{key}"
                );
            }
        }
    }

    /// A graph relation names every endpoint: a row with a null endpoint (a
    /// virtual site built from two atoms in the four-endpoint block) is
    /// refused rather than read as a relation to atom 0.
    #[test]
    fn from_frame_refuses_a_relation_row_with_a_null_endpoint() {
        use crate::core::Block;
        use ndarray::Array1;

        let mut mol = Atomistic::new();
        for _ in 0..3 {
            mol.add_atom_bare("C");
        }
        let mut frame = mol.to_frame().unwrap();
        let mut sites = Block::new();
        for (key, v) in [("atomi", 2_u64), ("atomj", 0), ("atomk", 1), ("atoml", 0)] {
            sites
                .insert(key, Array1::from_vec(vec![v]).into_dyn())
                .unwrap();
        }
        sites.set_validity("atoml", vec![false]).unwrap();
        frame.insert("virtual_sites", sites);
        let err = Atomistic::from_frame(&frame).expect_err("a null endpoint");
        assert!(err.to_string().contains("atoml"), "{err}");
    }

    #[test]
    fn test_try_from_molgraph_missing_element() {
        let mut g = MolGraph::new();
        g.register_kind("bonds", 2);
        g.add_node_with(Atom::new()).expect("fixture node"); // no element
        assert!(Atomistic::try_from_molgraph(g).is_err());
    }

    #[test]
    fn test_deref_generic_methods() {
        let mut mol = Atomistic::new();
        let c1 = mol.add_atom_bare("C");
        let c2 = mol.add_atom_bare("C");
        mol.add_bond(c1, c2).unwrap();
        // generic MolGraph methods via Deref
        assert_eq!(mol.n_nodes(), 2);
        assert_eq!(mol.neighbors(c1).count(), 1);
    }

    /// A graph whose *first* registered kind is not `bonds` — what a fragment
    /// graph looks like — must not have that foreign kind's relations reported
    /// as its bonds, nor be written into by the next `add_bond`.
    #[test]
    fn try_from_molgraph_resolves_bonds_by_name_not_kind_zero() {
        let mut graph = MolGraph::new();
        let ports = graph.register_kind("ports", 2);
        let c = graph
            .add_node_with(Atom::xyz("C", 0.0, 0.0, 0.0))
            .expect("fixture node");
        let h = graph
            .add_node_with(Atom::xyz("H", 1.09, 0.0, 0.0))
            .expect("fixture node");
        graph.add_relation(ports, &[c, h]).unwrap();

        let mut mol = Atomistic::try_from_molgraph(graph).expect("elements are present");
        let ports = mol.kind_id("ports").expect("the foreign kind survives");
        assert_eq!(mol.n_bonds(), 0, "a port relation is not a bond");

        mol.add_bond(c, h).expect("a bond can still be added");
        assert_eq!(mol.n_bonds(), 1);
        assert_eq!(
            mol.n_relations(ports),
            1,
            "add_bond must not write into the foreign kind"
        );
    }

    // ---- nullable columns: a partially set component keeps its mask ----

    /// Three atoms, one of them labelled: the column is emitted with the
    /// mask the entity table holds, so the two unlabelled atoms read as
    /// "no value" rather than as fragment instance zero.
    fn partly_labelled() -> (Atomistic, NodeId) {
        let mut mol = Atomistic::new();
        let a0 = mol.add_atom_bare("C");
        mol.add_atom_bare("C");
        mol.add_atom_bare("C");
        (mol, a0)
    }

    #[test]
    fn to_frame_masks_a_partially_set_int_column() {
        let (mut mol, a0) = partly_labelled();
        mol.set_atom(a0, "frag_id", PropValue::Int(7 as I)).unwrap();
        let frame = mol.to_frame().expect("a schema-conforming graph converts");
        let atoms = frame.get("atoms").expect("atoms block");
        assert_eq!(atoms.validity("frag_id"), Some(&[true, false, false][..]));
    }

    #[test]
    fn to_frame_masks_a_partially_set_float_column() {
        let (mut mol, a0) = partly_labelled();
        mol.set_atom(a0, "charge", -0.5_f64).unwrap();
        let frame = mol.to_frame().expect("a schema-conforming graph converts");
        let atoms = frame.get("atoms").expect("atoms block");
        assert_eq!(atoms.validity("charge"), Some(&[true, false, false][..]));
    }

    #[test]
    fn to_frame_masks_a_partially_set_string_column() {
        let (mut mol, a0) = partly_labelled();
        mol.set_atom(a0, "name", "CA").unwrap();
        let frame = mol.to_frame().expect("a schema-conforming graph converts");
        let atoms = frame.get("atoms").expect("atoms block");
        assert_eq!(atoms.validity("name"), Some(&[true, false, false][..]));
    }

    /// `set_atom` is the door an `Atomistic` caller reaches, and it carries
    /// the same schema opinion as the graph underneath: a string under the
    /// float key `x` never becomes an atom property.
    #[test]
    fn set_atom_refuses_a_str_under_a_schema_float_key() {
        use crate::core::DType;
        let mut mol = Atomistic::new();
        let a = mol.add_atom_bare("C");

        let err = mol
            .set_atom(a, "x", "left")
            .expect_err("a str cannot be stored at a schema-float key");
        assert!(matches!(err, MolRsError::Validation { .. }), "{err:?}");
        let msg = err.to_string();
        assert!(msg.contains("'x'"), "the error names the key, got {msg}");
        assert!(
            msg.contains(DType::Float.name()) && msg.contains(DType::String.name()),
            "the error names both dtypes, got {msg}"
        );
    }

    /// Storing a str `x` and then adding an atom carrying a real float `x`
    /// cannot be assembled: the sequence ends at its first step, so
    /// `add_atom_xyz` meets a float `x` column and the frame it produces is
    /// the schema-conforming one.
    #[test]
    fn set_atom_refusal_leaves_add_atom_xyz_a_float_x_column() {
        let mut mol = Atomistic::new();
        let a = mol.add_atom_bare("C");

        mol.set_atom(a, "x", "left")
            .expect_err("step one of the panicking sequence is refused");

        let b = mol.add_atom_xyz("O", 1.0, 2.0, 3.0);
        assert_eq!(
            mol.get_atom(b).expect("atom exists").get_f64("x"),
            Some(1.0)
        );
        let frame = mol
            .to_frame()
            .expect("no str 'x' was ever stored, so the frame converts");
        assert!(
            frame
                .get("atoms")
                .expect("atoms block")
                .get("x")
                .and_then(|c| c.as_float())
                .is_some()
        );
    }

    #[test]
    fn to_frame_leaves_a_fully_populated_column_unmasked() {
        let (mol, _a0) = partly_labelled();
        let frame = mol.to_frame().expect("a schema-conforming graph converts");
        let atoms = frame.get("atoms").expect("atoms block");
        assert_eq!(atoms.validity("element"), None);
    }

    #[test]
    fn from_frame_leaves_a_masked_int_cell_unset() {
        let (mut mol, a0) = partly_labelled();
        mol.set_atom(a0, "frag_id", PropValue::Int(7 as I)).unwrap();
        let back =
            Atomistic::from_frame(&mol.to_frame().expect("a schema-conforming graph converts"))
                .expect("an atomistic frame reads back");
        let read: Vec<Option<I>> = back.atoms().map(|(_, a)| a.get_int("frag_id")).collect();
        assert_eq!(read, vec![Some(7 as I), None, None]);
    }

    #[test]
    fn from_frame_leaves_a_masked_float_cell_unset() {
        let (mut mol, a0) = partly_labelled();
        mol.set_atom(a0, "charge", -0.5_f64).unwrap();
        let back =
            Atomistic::from_frame(&mol.to_frame().expect("a schema-conforming graph converts"))
                .expect("an atomistic frame reads back");
        let read: Vec<Option<f64>> = back.atoms().map(|(_, a)| a.get_f64("charge")).collect();
        assert_eq!(read, vec![Some(-0.5), None, None]);
    }

    #[test]
    fn from_frame_leaves_a_masked_string_cell_unset() {
        let (mut mol, a0) = partly_labelled();
        mol.set_atom(a0, "name", "CA").unwrap();
        let back =
            Atomistic::from_frame(&mol.to_frame().expect("a schema-conforming graph converts"))
                .expect("an atomistic frame reads back");
        let read: Vec<Option<String>> = back
            .atoms()
            .map(|(_, a)| a.get_str("name").map(str::to_owned))
            .collect();
        assert_eq!(read, vec![Some("CA".to_owned()), None, None]);
    }

    // ---- Contract B: a relation block that cannot be read is refused ----

    /// An `angles` block carrying only two of the kind's three endpoint
    /// columns is unreadable, and skipping it hands back a molecule whose
    /// angle the frame plainly stated. The error names the block and the
    /// column that is missing.
    #[test]
    fn from_frame_rejects_a_relation_block_missing_an_endpoint_column() {
        let mut atoms = Block::new();
        atoms
            .insert(
                "element",
                Array1::from_vec(vec!["C".to_owned(), "C".to_owned(), "C".to_owned()]).into_dyn(),
            )
            .unwrap();
        let mut angles = Block::new();
        angles
            .insert("atomi", Array1::from_vec(vec![0 as Idx]).into_dyn())
            .unwrap();
        angles
            .insert("atomj", Array1::from_vec(vec![1 as Idx]).into_dyn())
            .unwrap();
        let mut frame = Frame::new();
        frame.insert("atoms", atoms);
        frame.insert("angles", angles);

        let err = Atomistic::from_frame(&frame)
            .expect_err("an angles block without 'atomk' cannot be read, so it is not skipped");
        assert!(matches!(err, MolRsError::Validation { .. }), "{err:?}");
        let msg = err.to_string();
        assert!(msg.contains("angles"), "{msg}");
        assert!(msg.contains("atomk"), "{msg}");
    }

    /// A `bonds` row whose endpoint addresses an atom past the end of the
    /// `atoms` block states a bond over an atom the frame never gave — a
    /// truncated file, or a 1-based index written into a 0-based column. The
    /// row cannot be read, and dropping it hands back a molecule missing a
    /// bond the frame plainly stated, so the read is refused. The error names
    /// the block and the offending index.
    #[test]
    fn from_frame_rejects_a_relation_row_addressing_a_missing_atom() {
        let mut atoms = Block::new();
        atoms
            .insert(
                "element",
                Array1::from_vec(vec!["C".to_owned(), "C".to_owned()]).into_dyn(),
            )
            .unwrap();
        let mut bonds = Block::new();
        bonds
            .insert("atomi", Array1::from_vec(vec![0 as Idx]).into_dyn())
            .unwrap();
        bonds
            .insert("atomj", Array1::from_vec(vec![5 as Idx]).into_dyn())
            .unwrap();
        let mut frame = Frame::new();
        frame.insert("atoms", atoms);
        frame.insert("bonds", bonds);

        let err = Atomistic::from_frame(&frame).expect_err(
            "a bond onto atom 5 of a 2-atom frame cannot be read, so it is not skipped",
        );
        assert!(matches!(err, MolRsError::Validation { .. }), "{err:?}");
        let msg = err.to_string();
        assert!(
            msg.contains("bonds"),
            "the refusal must name the block: {msg}"
        );
        assert!(
            msg.contains('5'),
            "the refusal must name the endpoint index it could not resolve: {msg}"
        );
    }

    /// A caller-supplied graph that spells `bonds` at another arity is a data
    /// condition, so the promotion returns an error instead of aborting the
    /// process inside `register_kind`.
    #[test]
    fn try_from_molgraph_rejects_conflicting_arity() {
        let mut graph = MolGraph::new();
        graph.register_kind("bonds", 3);
        let err = Atomistic::try_from_molgraph(graph)
            .expect_err("a 3-ary 'bonds' kind conflicts with the standard set");
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
            let mut sys = Atomistic::new();
            let id = sys.add_atom_xyz("C", 1.0, 2.0, 3.0);
            let fixed = sys.add_atom_xyz("C", 1.0, 1.0, 1.0);
            sys.as_molgraph_mut().scale(factor, about);
            let moved = sys.get_atom(id).expect("live handle");
            for (key, want) in ["x", "y", "z"].into_iter().zip(expected) {
                let got = moved.get_f64(key).expect("coordinate kept");
                assert!(
                    (got - want).abs() < 1e-12,
                    "{about:?} {key}: {got} != {want}"
                );
            }
            if about.is_some() {
                // The centre itself is a fixed point of the map.
                let centre = sys.get_atom(fixed).expect("live handle");
                for key in ["x", "y", "z"] {
                    let got = centre.get_f64(key).expect("coordinate kept");
                    assert!((got - 1.0).abs() < 1e-12, "centre {key} moved to {got}");
                }
            }
        }
    }

    // ---- center ----

    /// Every atom enters: masses 1 and 3 at x = 0 and 4 give (3,0,0); an
    /// atomless molecule has no centre.
    #[test]
    fn center_covers_every_atom_and_an_empty_molecule_is_empty() {
        let mut mol = Atomistic::new();
        let light = mol.add_atom_xyz("H", 0.0, 0.0, 0.0);
        let heavy = mol.add_atom_xyz("Li", 4.0, 0.0, 0.0);
        mol.set_atom(light, crate::core::keys::MASS, 1.0).unwrap();
        mol.set_atom(heavy, crate::core::keys::MASS, 3.0).unwrap();
        let all: Vec<_> = mol.node_ids().collect();
        assert_eq!(mol.as_molgraph().center(&all), Ok([3.0, 0.0, 0.0]));

        assert_eq!(
            Atomistic::new().as_molgraph().center(&[]),
            Err(crate::core::CenterError::Empty)
        );
    }
}
