//! All-atom molecular graph: [`Atomistic`].

use std::collections::{BTreeSet, HashMap, HashSet};

use crate::core::Frame;
use crate::core::MolRsError;
use crate::core::bond_order::stamp_bond_class;
use crate::core::graph::{Graph, sealed};
use crate::core::keys;
use crate::core::molgraph::NodeGuard;
use crate::core::schema::{self, RowKind};
use crate::core::{BondNumber, BondOrder};
use crate::core::{CenterError, EntityTable, LinkError, Port, PortKind};
use crate::core::{GraphAccess, KindId, MolGraph, NodeId, PropValue, Props, Relation, RelationId};

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
/// kinds at construction and exposes one typed method per operation on each
/// entity — atoms (`add_atom`, `del_atom` / `del_atoms`, `get_atom`,
/// `set_atom`, `clear_atom`, `atom_ids`, `atoms`, `n_atoms`, `atom_table`),
/// bonds, angles, dihedrals and impropers (the same eight operations each),
/// the further relation blocks a force field states (`cmaps`, `exclusions`, …
/// — the `*_term` family), ports, geometry ([`translate`](Self::translate),
/// [`rotate`](Self::rotate), [`scale`](Self::scale),
/// [`center`](Self::center)), composition and hashing. A single entity and a
/// batch are distinct operations: [`del_atom`](Self::del_atom) deletes one
/// atom, [`del_atoms`](Self::del_atoms) a set of atoms atomically.
///
/// Invariant: every atom carries a valid [`keys::ATOMIC_NUMBER`] (0..=118,
/// `0` the wildcard `*`) — the one species field molrs reads. [`keys::ELEMENT`]
/// is ordinary optional data: nothing consults it, and nothing converts
/// between the two implicitly. The graph carries the invariant as a guard,
/// so every node write enforces it — [`add_atom`](Self::add_atom),
/// [`set_atom`](Self::set_atom), [`clear_atom`](Self::clear_atom),
/// [`from_frame`](Self::from_frame), [`merge`](Self::merge) and
/// [`replicate`](Self::replicate). A graph promoted with
/// `Atomistic::try_from(graph)` is checked node by node first.
///
/// # Examples
///
/// ```
/// use molrs::core::{Atomistic, Element, keys};
/// use molrs::props;
///
/// let mut mol = Atomistic::new();
/// let c = mol.add_atom(props! { keys::ATOMIC_NUMBER => Element::C }).unwrap();
/// let h = mol.add_atom(props! { keys::ATOMIC_NUMBER => Element::H }).unwrap();
/// mol.add_bond(c, h, props! {}).unwrap();
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

impl Default for Atomistic {
    fn default() -> Self {
        Self::new()
    }
}

/// The relation blocks `Atomistic` has typed methods for; the `*_term`
/// family refuses them.
const TYPED_BLOCKS: [(&str, &str); 5] = [
    (
        schema::block_names::BONDS,
        "add_bond / del_bond / get_bond / set_bond",
    ),
    (
        schema::block_names::ANGLES,
        "add_angle / del_angle / get_angle / set_angle",
    ),
    (
        schema::block_names::DIHEDRALS,
        "add_dihedral / del_dihedral / get_dihedral / set_dihedral",
    ),
    (
        schema::block_names::IMPROPERS,
        "add_improper / del_improper / get_improper / set_improper",
    ),
    (keys::PORTS, "add_port / del_port / get_port"),
];

impl Atomistic {
    /// Create an empty all-atom molecular graph with the bond / angle /
    /// dihedral / improper kinds registered.
    pub fn new() -> Self {
        let mut graph = MolGraph::new();
        graph.set_node_guard(Some(SPECIES_GUARD));
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

    /// Add an atom carrying the property bag `props`.
    ///
    /// `props` must hold a valid [`keys::ATOMIC_NUMBER`]; everything else is
    /// free-form key:value data.
    ///
    /// # Errors
    ///
    /// [`MolRsError::Validation`] when `props` has no valid atomic number, or
    /// when a value contradicts the element type an existing atom column
    /// holds for that key (a string `charge` into an `f64` `charge` column).
    /// The insert is atomic: on any refusal the molecule is unchanged.
    pub fn add_atom(&mut self, props: impl Into<Props>) -> Result<NodeId, MolRsError> {
        self.graph.add_node(props)
    }

    /// Delete one atom and every bond, angle, dihedral, improper, term and
    /// port that references it; returns the atom's property bag.
    ///
    /// # Errors
    ///
    /// [`MolRsError::NotFound`] for a dead handle; the molecule is unchanged.
    pub fn del_atom(&mut self, id: NodeId) -> Result<Props, MolRsError> {
        self.graph.del_node(id)
    }

    /// Delete several atoms, with everything that references them, in one
    /// pass; returns their property bags in `ids` order.
    ///
    /// # Errors
    ///
    /// [`MolRsError::Validation`] when `ids` names an atom twice and
    /// [`MolRsError::NotFound`] when one is dead. **Atomic**: every handle is
    /// checked first, so on a refusal no atom is deleted.
    pub fn del_atoms(&mut self, ids: &[NodeId]) -> Result<Vec<Props>, MolRsError> {
        self.graph.del_nodes(ids)
    }

    /// Materialize an atom's property bag (owned copy of its set components).
    ///
    /// # Errors
    ///
    /// [`MolRsError::NotFound`] for a dead handle.
    pub fn get_atom(&self, id: NodeId) -> Result<Props, MolRsError> {
        self.graph.get_node(id)
    }

    /// Set a single component on an atom.
    ///
    /// # Errors
    ///
    /// [`MolRsError::Validation`] when `key` is [`keys::ATOMIC_NUMBER`] and
    /// `val` is not a valid atomic number (the invariant), or when `val`
    /// does not fit the column; [`MolRsError::NotFound`] for a dead handle.
    pub fn set_atom(
        &mut self,
        id: NodeId,
        key: &str,
        val: impl Into<PropValue>,
    ) -> Result<(), MolRsError> {
        self.graph.set_node(id, key, val)
    }

    /// Clear a single component on an atom (no-op if absent).
    ///
    /// # Errors
    ///
    /// [`MolRsError::Validation`] when `key` is [`keys::ATOMIC_NUMBER`]: an
    /// atom cannot lose its species. [`MolRsError::NotFound`] for a dead
    /// handle.
    pub fn clear_atom(&mut self, id: NodeId, key: &str) -> Result<(), MolRsError> {
        self.graph.clear_node(id, key)
    }

    /// Live atom handles, in row order (the order [`to_frame`](Self::to_frame)
    /// writes).
    pub fn atom_ids(&self) -> impl Iterator<Item = NodeId> + '_ {
        self.graph.node_ids()
    }

    /// Iterate over all `(NodeId, Props)` pairs (each property bag materialized).
    pub fn atoms(&self) -> impl Iterator<Item = (NodeId, Props)> + '_ {
        self.graph.nodes()
    }

    /// Number of atoms.
    pub fn n_atoms(&self) -> usize {
        self.graph.n_nodes()
    }

    /// The atoms' property columns, read-only and aligned to
    /// [`atom_ids`](Self::atom_ids) order — the zero-copy view for a caller
    /// that reads one property of every atom. Writes go through
    /// [`set_atom`](Self::set_atom).
    pub fn atom_table(&self) -> &EntityTable<NodeId> {
        self.graph.node_table()
    }

    // ---- bonds ----

    /// Add a bond between two existing atoms, carrying `props`.
    ///
    /// A bond whose `props` state no [`keys::BOND_TYPE`] is a single bond
    /// (class and number both stamped); one that states a class keeps
    /// exactly what `props` say.
    pub fn add_bond(
        &mut self,
        a: NodeId,
        b: NodeId,
        props: impl Into<Props>,
    ) -> Result<RelationId, MolRsError> {
        let props: Props = props.into();
        let stated = props.get(keys::BOND_TYPE).is_some();
        let bid = self.graph.add_relation(self.bond, &[a, b], props)?;
        if !stated {
            self.set_bond_type(bid, BondOrder::Single)?;
        }
        Ok(bid)
    }

    /// Delete one bond (its atoms stay); returns the bond as stored.
    ///
    /// # Errors
    ///
    /// [`MolRsError::NotFound`] when `id` names no live bond.
    pub fn del_bond(&mut self, id: RelationId) -> Result<Relation, MolRsError> {
        self.graph.del_relation(self.bond, id)
    }

    /// Delete several bonds; returns them as stored, in `ids` order.
    ///
    /// # Errors
    ///
    /// [`MolRsError::Validation`] when `ids` names a bond twice and
    /// [`MolRsError::NotFound`] when one names no live bond. **Atomic**: on a
    /// refusal no bond is deleted.
    pub fn del_bonds(&mut self, ids: &[RelationId]) -> Result<Vec<Relation>, MolRsError> {
        self.graph.del_relations(self.bond, ids)
    }

    /// Materialize a bond (endpoints + properties).
    ///
    /// # Errors
    ///
    /// [`MolRsError::NotFound`] when `id` names no live bond.
    pub fn get_bond(&self, id: RelationId) -> Result<Relation, MolRsError> {
        self.graph.get_relation(self.bond, id)
    }

    /// Set a single property on a bond.
    ///
    /// # Errors
    ///
    /// [`MolRsError::Validation`] when `key` is [`keys::BOND_TYPE`] or
    /// [`keys::BOND_NUMBER`] — the class and its number are set together by
    /// [`set_bond_class`](Self::set_bond_class) — or when `val` does not fit
    /// the column; [`MolRsError::NotFound`] when `id` names no live bond.
    pub fn set_bond(
        &mut self,
        id: RelationId,
        key: &str,
        val: impl Into<PropValue>,
    ) -> Result<(), MolRsError> {
        refuse_bond_class_key(key, "set")?;
        self.graph.set_relation_prop(self.bond, id, key, val)
    }

    /// Clear a single property on a bond (no-op if absent).
    ///
    /// # Errors
    ///
    /// [`MolRsError::Validation`] when `key` is [`keys::BOND_TYPE`] or
    /// [`keys::BOND_NUMBER`] (a bond whose class is not known says so through
    /// [`set_bond_class`](Self::set_bond_class) with the `Unknown`s);
    /// [`MolRsError::NotFound`] when `id` names no live bond.
    pub fn clear_bond(&mut self, id: RelationId, key: &str) -> Result<(), MolRsError> {
        refuse_bond_class_key(key, "cleared")?;
        self.graph.clear_relation_prop(self.bond, id, key)
    }

    /// Live bond handles, in row order.
    pub fn bond_ids(&self) -> impl Iterator<Item = RelationId> + '_ {
        self.graph.relation_ids(self.bond)
    }

    /// Iterate over all `(RelationId, Relation)` pairs (each materialized).
    pub fn bonds(&self) -> impl Iterator<Item = (RelationId, Relation)> + '_ {
        self.graph.relations(self.bond)
    }

    /// Number of bonds.
    pub fn n_bonds(&self) -> usize {
        self.graph.n_relations(self.bond)
    }

    /// The bond's chemical class; [`BondOrder::Unknown`] if it has none.
    ///
    /// # Errors
    ///
    /// [`MolRsError::NotFound`] when `id` names no live bond.
    pub fn bond_type(&self, id: RelationId) -> Result<BondOrder, MolRsError> {
        let b = self.get_bond(id)?;
        Ok(BondOrder::from_prop(b.props.get(keys::BOND_TYPE)))
    }

    /// The bond's localized (Kekulé) bond number; [`BondNumber::Unknown`] if it
    /// has none — an aromatic bond before kekulization, for instance.
    ///
    /// # Errors
    ///
    /// [`MolRsError::NotFound`] when `id` names no live bond.
    pub fn bond_number(&self, id: RelationId) -> Result<BondNumber, MolRsError> {
        let b = self.get_bond(id)?;
        Ok(BondNumber::from_prop(b.props.get(keys::BOND_NUMBER)))
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

    /// Stamp a plain (non-aromatic) class with the number it implies — the
    /// default `add_bond` applies. The public setter is
    /// [`set_bond_class`](Self::set_bond_class).
    fn set_bond_type(&mut self, id: RelationId, bond_type: BondOrder) -> Result<(), MolRsError> {
        let number = bond_type.implied_number().unwrap_or_default();
        self.set_bond_class(id, bond_type, number)
    }

    /// The bond joining atoms `a` and `b`, if any. A port, a constraint or
    /// any other two-atom term between them is not a bond.
    pub fn bond_between(&self, a: NodeId, b: NodeId) -> Option<RelationId> {
        self.neighbors(a, crate::core::NeighborKinds::BONDS)
            .find(|&(other, _)| other == b)
            .map(|(_, bid)| bid)
    }

    /// Arity-2 relations incident to `id`.
    ///
    /// [`NeighborKinds::BONDS`](crate::core::NeighborKinds::BONDS) yields
    /// `(other_atom, bond_id)`.
    /// [`NeighborKinds::with_ports`](crate::core::NeighborKinds::with_ports) and
    /// [`NeighborKinds::with_terms`](crate::core::NeighborKinds::with_terms)
    /// also yield ports and the other arity-2
    /// terms (constraints, pairs, exclusions, drudes, virtual sites), and
    /// each item is an [`Adjacent`](crate::core::Adjacent) so the handle
    /// stays tied to its table. Angles and dihedrals have more than two
    /// endpoints and are never neighbors.
    ///
    /// The bond item is a handle, not a bond order. A caller that wants the
    /// localized count asks [`bond_number`](Self::bond_number), and one that
    /// wants to know whether the bond is aromatic asks
    /// [`bond_type`](Self::bond_type).
    pub fn neighbors<'a, K: crate::core::adjacency::AdjacencyKind + 'a>(
        &'a self,
        id: NodeId,
        which: crate::core::NeighborKinds<K>,
    ) -> impl Iterator<Item = K::Item<'a>> + 'a {
        self.graph.adjacent(id, self.bond, which)
    }

    // ---- angles ----

    /// Add an angle (i-j-k, j central), carrying `props`.
    pub fn add_angle(
        &mut self,
        i: NodeId,
        j: NodeId,
        k: NodeId,
        props: impl Into<Props>,
    ) -> Result<RelationId, MolRsError> {
        self.graph.add_relation(self.angle, &[i, j, k], props)
    }

    /// Delete one angle; returns it as stored.
    ///
    /// # Errors
    ///
    /// [`MolRsError::NotFound`] when `id` names no live angle.
    pub fn del_angle(&mut self, id: RelationId) -> Result<Relation, MolRsError> {
        self.graph.del_relation(self.angle, id)
    }

    /// Delete several angles; returns them as stored, in `ids` order.
    ///
    /// # Errors
    ///
    /// As [`del_bonds`](Self::del_bonds), for angles. **Atomic**.
    pub fn del_angles(&mut self, ids: &[RelationId]) -> Result<Vec<Relation>, MolRsError> {
        self.graph.del_relations(self.angle, ids)
    }

    /// Materialize an angle (endpoints + properties).
    pub fn get_angle(&self, id: RelationId) -> Result<Relation, MolRsError> {
        self.graph.get_relation(self.angle, id)
    }

    /// Set a single property on an angle.
    ///
    /// # Errors
    ///
    /// [`MolRsError::Validation`] when `val` does not fit the column;
    /// [`MolRsError::NotFound`] when `id` names no live angle.
    pub fn set_angle(
        &mut self,
        id: RelationId,
        key: &str,
        val: impl Into<PropValue>,
    ) -> Result<(), MolRsError> {
        self.graph.set_relation_prop(self.angle, id, key, val)
    }

    /// Clear a single property on an angle (no-op if absent).
    ///
    /// # Errors
    ///
    /// [`MolRsError::NotFound`] when `id` names no live angle.
    pub fn clear_angle(&mut self, id: RelationId, key: &str) -> Result<(), MolRsError> {
        self.graph.clear_relation_prop(self.angle, id, key)
    }

    /// Live angle handles, in row order.
    pub fn angle_ids(&self) -> impl Iterator<Item = RelationId> + '_ {
        self.graph.relation_ids(self.angle)
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

    /// Add a dihedral (i-j-k-l), carrying `props`.
    pub fn add_dihedral(
        &mut self,
        i: NodeId,
        j: NodeId,
        k: NodeId,
        l: NodeId,
        props: impl Into<Props>,
    ) -> Result<RelationId, MolRsError> {
        self.graph.add_relation(self.dihedral, &[i, j, k, l], props)
    }

    /// Delete one dihedral; returns it as stored.
    ///
    /// # Errors
    ///
    /// [`MolRsError::NotFound`] when `id` names no live dihedral.
    pub fn del_dihedral(&mut self, id: RelationId) -> Result<Relation, MolRsError> {
        self.graph.del_relation(self.dihedral, id)
    }

    /// Delete several dihedrals; returns them as stored, in `ids` order.
    ///
    /// # Errors
    ///
    /// As [`del_bonds`](Self::del_bonds), for dihedrals. **Atomic**.
    pub fn del_dihedrals(&mut self, ids: &[RelationId]) -> Result<Vec<Relation>, MolRsError> {
        self.graph.del_relations(self.dihedral, ids)
    }

    /// Materialize a dihedral (endpoints + properties).
    pub fn get_dihedral(&self, id: RelationId) -> Result<Relation, MolRsError> {
        self.graph.get_relation(self.dihedral, id)
    }

    /// Set a single property on a dihedral.
    ///
    /// # Errors
    ///
    /// [`MolRsError::Validation`] when `val` does not fit the column;
    /// [`MolRsError::NotFound`] when `id` names no live dihedral.
    pub fn set_dihedral(
        &mut self,
        id: RelationId,
        key: &str,
        val: impl Into<PropValue>,
    ) -> Result<(), MolRsError> {
        self.graph.set_relation_prop(self.dihedral, id, key, val)
    }

    /// Clear a single property on a dihedral (no-op if absent).
    ///
    /// # Errors
    ///
    /// [`MolRsError::NotFound`] when `id` names no live dihedral.
    pub fn clear_dihedral(&mut self, id: RelationId, key: &str) -> Result<(), MolRsError> {
        self.graph.clear_relation_prop(self.dihedral, id, key)
    }

    /// Live dihedral handles, in row order.
    pub fn dihedral_ids(&self) -> impl Iterator<Item = RelationId> + '_ {
        self.graph.relation_ids(self.dihedral)
    }

    /// Iterate over all `(RelationId, Relation)` pairs (each materialized).
    pub fn dihedrals(&self) -> impl Iterator<Item = (RelationId, Relation)> + '_ {
        self.graph.relations(self.dihedral)
    }

    /// Number of dihedrals.
    pub fn n_dihedrals(&self) -> usize {
        self.graph.n_relations(self.dihedral)
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
        props: impl Into<Props>,
    ) -> Result<RelationId, MolRsError> {
        self.graph.add_relation(self.improper, &[i, j, k, l], props)
    }

    /// Delete one improper; returns it as stored.
    ///
    /// # Errors
    ///
    /// [`MolRsError::NotFound`] when `id` names no live improper.
    pub fn del_improper(&mut self, id: RelationId) -> Result<Relation, MolRsError> {
        self.graph.del_relation(self.improper, id)
    }

    /// Delete several impropers; returns them as stored, in `ids` order.
    ///
    /// # Errors
    ///
    /// As [`del_bonds`](Self::del_bonds), for impropers. **Atomic**.
    pub fn del_impropers(&mut self, ids: &[RelationId]) -> Result<Vec<Relation>, MolRsError> {
        self.graph.del_relations(self.improper, ids)
    }

    /// Materialize an improper (endpoints + properties).
    pub fn get_improper(&self, id: RelationId) -> Result<Relation, MolRsError> {
        self.graph.get_relation(self.improper, id)
    }

    /// Set a single property on an improper.
    ///
    /// # Errors
    ///
    /// [`MolRsError::Validation`] when `val` does not fit the column;
    /// [`MolRsError::NotFound`] when `id` names no live improper.
    pub fn set_improper(
        &mut self,
        id: RelationId,
        key: &str,
        val: impl Into<PropValue>,
    ) -> Result<(), MolRsError> {
        self.graph.set_relation_prop(self.improper, id, key, val)
    }

    /// Clear a single property on an improper (no-op if absent).
    ///
    /// # Errors
    ///
    /// [`MolRsError::NotFound`] when `id` names no live improper.
    pub fn clear_improper(&mut self, id: RelationId, key: &str) -> Result<(), MolRsError> {
        self.graph.clear_relation_prop(self.improper, id, key)
    }

    /// Live improper handles, in row order.
    pub fn improper_ids(&self) -> impl Iterator<Item = RelationId> + '_ {
        self.graph.relation_ids(self.improper)
    }

    /// Iterate over all `(RelationId, Relation)` pairs (each materialized).
    pub fn impropers(&self) -> impl Iterator<Item = (RelationId, Relation)> + '_ {
        self.graph.relations(self.improper)
    }

    /// Number of impropers.
    pub fn n_impropers(&self) -> usize {
        self.graph.n_relations(self.improper)
    }

    // ---- topology perception ----

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
    /// [`Topology::impropers`](crate::core::Topology::impropers),
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
        if clear_existing {
            for (wanted, kind) in [
                (gen_angle, self.angle),
                (gen_dihedral, self.dihedral),
                (gen_improper, self.improper),
            ] {
                if wanted {
                    let ids: Vec<_> = self.graph.relation_ids(kind).collect();
                    self.graph.del_relations(kind, &ids)?;
                }
            }
        }

        let atoms: Vec<NodeId> = self.atom_ids().collect();
        let topo = self.topology();

        let mut n_ang = 0usize;
        let mut n_dih = 0usize;
        let mut n_imp = 0usize;

        if gen_angle {
            let mut seen: HashSet<Vec<NodeId>> = HashSet::new();
            for id in self.graph.relation_ids(self.angle) {
                seen.insert(canonical_path(&self.graph.relation_nodes(self.angle, id)?));
            }
            for a in topo.angles() {
                let nodes = [atoms[a[0]], atoms[a[1]], atoms[a[2]]];
                if seen.insert(canonical_path(&nodes)) {
                    self.add_angle(nodes[0], nodes[1], nodes[2], Props::new())?;
                    n_ang += 1;
                }
            }
        }

        if gen_dihedral {
            let mut seen: HashSet<Vec<NodeId>> = HashSet::new();
            for id in self.graph.relation_ids(self.dihedral) {
                seen.insert(canonical_path(
                    &self.graph.relation_nodes(self.dihedral, id)?,
                ));
            }
            for d in topo.dihedrals() {
                let nodes = [atoms[d[0]], atoms[d[1]], atoms[d[2]], atoms[d[3]]];
                if seen.insert(canonical_path(&nodes)) {
                    self.add_dihedral(nodes[0], nodes[1], nodes[2], nodes[3], Props::new())?;
                    n_dih += 1;
                }
            }
        }

        if gen_improper {
            let mut seen: HashSet<Vec<NodeId>> = HashSet::new();
            for id in self.graph.relation_ids(self.improper) {
                seen.insert(canonical_improper(
                    &self.graph.relation_nodes(self.improper, id)?,
                ));
            }
            for q in topo.impropers() {
                let nodes = [atoms[q[0]], atoms[q[1]], atoms[q[2]], atoms[q[3]]];
                if seen.insert(canonical_improper(&nodes)) {
                    self.add_improper(nodes[0], nodes[1], nodes[2], nodes[3], Props::new())?;
                    n_imp += 1;
                }
            }
        }

        Ok((n_ang, n_dih, n_imp))
    }

    /// The bond graph as a [`Topology`](crate::core::Topology) over atom
    /// rows: atom `i` is the `i`-th of [`atom_ids`](Self::atom_ids) (its
    /// [`atom_table`](Self::atom_table) row), and every bond is one edge.
    ///
    /// The snapshot index-space algorithms (rings, rotatable bonds, generated
    /// angles and dihedrals) run on; it does not follow later edits.
    pub fn topology(&self) -> crate::core::Topology {
        const LIVE: &str = "a bond's endpoints are live atoms";
        let rows = self.atom_table();
        let edges: Vec<[usize; 2]> = self
            .graph
            .relation_ids(self.bond)
            .map(|id| {
                let ends = self.graph.relation_nodes(self.bond, id).expect(LIVE);
                [
                    rows.row(ends[0]).expect(LIVE),
                    rows.row(ends[1]).expect(LIVE),
                ]
            })
            .collect();
        let mut topo = crate::core::Topology::with_atoms(self.n_atoms());
        topo.add_bonds(&edges);
        topo
    }

    /// BFS shortest-path distances over the bond graph from `sources`, as
    /// `(atom_id, hops)` pairs for every atom reachable from one of them
    /// (each source at distance 0; every other atom at its distance to the
    /// nearest source), in discovery order, expanding no further than
    /// `max_hops`. Unreachable atoms are omitted, and an unknown source seeds
    /// nothing. The cost is the atoms reached, not the molecule.
    pub fn topo_distances(&self, sources: &[NodeId], max_hops: Option<i64>) -> Vec<(NodeId, i64)> {
        self.graph.hops_from(sources, self.bond, max_hops, None)
    }

    // ---- further relation blocks (the `*_term` family) ----

    /// Add a row to the relation block `block` — `cmaps`, `constraints`,
    /// `drudes`, `exclusions`, `pairs`, `virtual_sites` or any block this
    /// molecule already holds — over `atoms`, carrying `props`.
    ///
    /// The block's arity is the one this molecule registered for it, else the
    /// Frame schema's ([`schema::block`]); the block is registered on first
    /// use. Returns the term's handle, which addresses it within `block`.
    ///
    /// # Errors
    ///
    /// [`MolRsError::Validation`] when `block` has typed methods (`bonds`,
    /// `angles`, `dihedrals`, `impropers`, `ports` — the message names them),
    /// when its arity is unknown (neither registered nor a canonical relation
    /// block over `atoms` rows), when `atoms.len()` differs from that arity,
    /// or when a value of `props` does not fit its column;
    /// [`MolRsError::NotFound`] when an atom is dead. Nothing is written on a
    /// refusal.
    pub fn add_term(
        &mut self,
        block: &str,
        atoms: &[NodeId],
        props: impl Into<Props>,
    ) -> Result<RelationId, MolRsError> {
        let arity = match self.term_kind(block)? {
            Some(kind) => self.graph.arity(kind),
            None => schema_term_arity(block)?,
        };
        self.graph.add_relation_named(block, arity, atoms, props)
    }

    /// Declare the term block `block` with `arity` atoms per row, so that
    /// [`add_term`](Self::add_term) accepts a block the Frame schema does not
    /// define — a force-field term such as `urey_bradleys`. Declaring a block
    /// the molecule already holds at the same arity is a no-op.
    ///
    /// # Errors
    ///
    /// [`MolRsError::Validation`] when `block` has typed methods (`bonds`,
    /// `angles`, `dihedrals`, `impropers`, `ports`), when `arity` is 0, when
    /// the Frame schema defines `block` at another arity or not over `atoms`
    /// rows, or when the molecule already holds `block` at another arity.
    pub fn declare_term(&mut self, block: &str, arity: usize) -> Result<(), MolRsError> {
        self.term_kind(block)?;
        if arity == 0 {
            return Err(MolRsError::validation(format!(
                "term block '{block}' needs at least one atom per row"
            )));
        }
        if schema::block(block).is_some() {
            let canonical = schema_term_arity(block)?;
            if canonical != arity {
                return Err(MolRsError::validation(format!(
                    "term block '{block}' has arity {canonical} in the Frame schema, not {arity}"
                )));
            }
        }
        self.graph.try_register_kind(block, arity).map(|_| ())
    }

    /// Delete one row of the term block `block`; returns it as stored.
    ///
    /// # Errors
    ///
    /// The block refusals of [`add_term`](Self::add_term), and
    /// [`MolRsError::NotFound`] when `id` names no live term of `block`.
    pub fn del_term(&mut self, block: &str, id: RelationId) -> Result<Relation, MolRsError> {
        Ok(self
            .del_terms(block, &[id])?
            .pop()
            .expect("one validated term produces one payload"))
    }

    /// Delete several rows of the term block `block`; returns them as
    /// stored, in `ids` order.
    ///
    /// # Errors
    ///
    /// As [`del_term`](Self::del_term), plus [`MolRsError::Validation`] when
    /// `ids` names a term twice. **Atomic**: on a refusal no term is deleted.
    pub fn del_terms(
        &mut self,
        block: &str,
        ids: &[RelationId],
    ) -> Result<Vec<Relation>, MolRsError> {
        let Some(&first) = ids.first() else {
            self.term_kind_or_schema(block)?;
            return Ok(Vec::new());
        };
        let kind = self.named_term_kind(block, first)?;
        self.graph.del_relations(kind, ids)
    }

    /// Materialize one row of the term block `block` (endpoints +
    /// properties).
    ///
    /// # Errors
    ///
    /// As [`del_term`](Self::del_term).
    pub fn get_term(&self, block: &str, id: RelationId) -> Result<Relation, MolRsError> {
        let kind = self.named_term_kind(block, id)?;
        self.graph.get_relation(kind, id)
    }

    /// Set a single property on one row of the term block `block`.
    ///
    /// # Errors
    ///
    /// As [`del_term`](Self::del_term), plus [`MolRsError::Validation`] when
    /// `val` does not fit the column.
    pub fn set_term(
        &mut self,
        block: &str,
        id: RelationId,
        key: &str,
        val: impl Into<PropValue>,
    ) -> Result<(), MolRsError> {
        let kind = self.named_term_kind(block, id)?;
        self.graph.set_relation_prop(kind, id, key, val)
    }

    /// Clear a single property on one row of the term block `block` (no-op
    /// if absent).
    ///
    /// # Errors
    ///
    /// As [`del_term`](Self::del_term).
    pub fn clear_term(&mut self, block: &str, id: RelationId, key: &str) -> Result<(), MolRsError> {
        let kind = self.named_term_kind(block, id)?;
        self.graph.clear_relation_prop(kind, id, key)
    }

    /// Live handles of the term block `block`, in row order; empty for a
    /// canonical block this molecule holds no row of.
    ///
    /// # Errors
    ///
    /// The block refusals of [`add_term`](Self::add_term).
    pub fn term_ids(
        &self,
        block: &str,
    ) -> Result<impl Iterator<Item = RelationId> + '_, MolRsError> {
        let kind = self.term_kind_or_schema(block)?;
        Ok(kind
            .into_iter()
            .flat_map(move |kind| self.graph.relation_ids(kind)))
    }

    /// Number of rows of the term block `block`; `0` for a canonical block
    /// this molecule holds no row of.
    ///
    /// # Errors
    ///
    /// The block refusals of [`add_term`](Self::add_term).
    pub fn n_terms(&self, block: &str) -> Result<usize, MolRsError> {
        Ok(self
            .term_kind_or_schema(block)?
            .map_or(0, |kind| self.graph.n_relations(kind)))
    }

    /// The declared (or canonical) arity of a term block.
    pub fn term_arity(&self, block: &str) -> Result<usize, MolRsError> {
        match self.term_kind(block)? {
            Some(kind) => Ok(self.graph.arity(kind)),
            None => schema_term_arity(block),
        }
    }

    /// The term blocks this molecule holds, in registration order — every
    /// registered relation block without typed methods.
    pub fn term_blocks(&self) -> impl Iterator<Item = &str> + '_ {
        self.graph
            .kind_ids()
            .map(|kind| self.graph.kind_name(kind))
            .filter(|name| typed_block(name).is_none())
    }

    /// The registered kind of term block `block`, `None` when it is not
    /// registered; refuses a typed block.
    fn term_kind(&self, block: &str) -> Result<Option<KindId>, MolRsError> {
        if let Some(methods) = typed_block(block) {
            return Err(MolRsError::validation(format!(
                "'{block}' is not a term block: Atomistic has typed methods for it ({methods})"
            )));
        }
        Ok(self.graph.kind_id(block))
    }

    /// [`term_kind`](Self::term_kind), additionally refusing an unregistered
    /// block that is no canonical term block.
    fn term_kind_or_schema(&self, block: &str) -> Result<Option<KindId>, MolRsError> {
        let kind = self.term_kind(block)?;
        if kind.is_none() {
            schema_term_arity(block)?;
        }
        Ok(kind)
    }

    /// The kind a handle of term block `block` lives in: `NotFound` when
    /// the molecule holds no row of a canonical term block.
    fn named_term_kind(&self, block: &str, id: RelationId) -> Result<KindId, MolRsError> {
        self.term_kind_or_schema(block)?.ok_or_else(|| {
            MolRsError::not_found(
                "term",
                format!(
                    "term {} of block '{block}' (the molecule holds no '{block}' row)",
                    u64::from(id)
                ),
            )
        })
    }

    // ---- ports ----

    /// Record a descriptor on the `(anchor, handle)` valence; see
    /// [`Port`] for the vocabulary.
    ///
    /// # Errors
    ///
    /// As [`MolGraph::add_port`]: a dead atom, a handle not bonded to the
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

    /// The leaving group of `port`: its handle's connected component over
    /// bonds once the anchor–handle bond is cut.
    ///
    /// # Errors
    ///
    /// As [`MolGraph::leaving_group`]: the port is stale.
    pub fn leaving_group(&self, port: &Port) -> Result<BTreeSet<NodeId>, MolRsError> {
        self.graph.leaving_group(port)
    }

    /// Join port `a` to port `b` and return the new anchor–anchor bond; see
    /// [`MolGraph::link`] for the charge fold and the refusals.
    ///
    /// # Errors
    ///
    /// The [`LinkError`] of the first refusal; the molecule is unchanged then.
    pub fn link(&mut self, a: RelationId, b: RelationId) -> Result<RelationId, LinkError> {
        self.graph.link(a, b)
    }

    /// Propagate each [`keys::FRAG_ID`] to the unlabelled degree-1 atoms
    /// bonded to a labelled one; returns how many atoms were labelled. See
    /// [`MolGraph::inherit_frag_ids`].
    pub fn inherit_frag_ids(&mut self) -> usize {
        self.graph.inherit_frag_ids()
    }

    // ---- geometry ----

    /// Translate every atom that has coordinates by `delta` (Å).
    pub fn translate(&mut self, delta: [f64; 3]) {
        self.graph.translate(delta);
    }

    /// Rotate every atom that has coordinates around `axis` by `angle`
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

    /// Scale every atom that has coordinates by the per-axis `factor` about
    /// `about` (Å; the origin by default).
    pub fn scale(&mut self, factor: [f64; 3], about: Option<[f64; 3]>) {
        self.graph.scale(factor, about);
    }

    /// The mass-weighted centre (Å) of `atoms`.
    ///
    /// # Errors
    ///
    /// The [`CenterError`] of [`MolGraph::center`]: no atoms, a dead atom, a
    /// missing coordinate or mass, or a zero total mass.
    pub fn center(&self, atoms: &[NodeId]) -> Result<[f64; 3], CenterError> {
        self.graph.center(atoms)
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
    ///
    /// # Species
    ///
    /// The `atoms` block must carry [`keys::ATOMIC_NUMBER`]. A frame read
    /// from a symbol-spelling format (XYZ, PDB, SDF, …) carries
    /// [`keys::ELEMENT`] instead; convert it explicitly first with
    /// [`Element::fill_atomic_numbers`](crate::core::Element::fill_atomic_numbers).
    ///
    /// # Term blocks
    ///
    /// Every canonical term block over atoms the frame carries (`cmaps`,
    /// `constraints`, `drudes`, `exclusions`, `pairs`, `virtual_sites`) is
    /// read back as that block's terms, so a molecule that
    /// [`to_frame`](Self::to_frame) wrote after
    /// [`add_term`](Self::add_term) round-trips. Other blocks (a box, a grid,
    /// a coefficient table) are not relations of the molecule and are
    /// ignored.
    ///
    /// # Errors
    ///
    /// [`MolRsError::Validation`] when an atom has no valid atomic number,
    /// or when a term block lacks one of the `atomi`/`atomj`/… endpoint
    /// columns its schema arity names, besides the errors of reading the
    /// blocks.
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

    /// Promote from a [`MolGraph`], validating all atoms have a valid
    /// [`keys::ATOMIC_NUMBER`].
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
    /// Returns [`MolRsError::Validation`] when an atom carries no valid
    /// [`keys::ATOMIC_NUMBER`], and when the graph already spells one of the standard
    /// kind names at a different arity (naming the kind and both arities).
    fn promote(mut mol: MolGraph) -> Result<Self, MolRsError> {
        for (id, atom) in mol.nodes() {
            require_atomic_number(&atom, &format!("node {id:?}"))?;
        }
        let bond = mol.try_register_kind("bonds", 2)?;
        let angle = mol.try_register_kind("angles", 3)?;
        let dihedral = mol.try_register_kind("dihedrals", 4)?;
        let improper = mol.try_register_kind("impropers", 4)?;
        mol.set_node_guard(Some(SPECIES_GUARD));
        Ok(Self {
            graph: mol,
            bond,
            angle,
            dihedral,
            improper,
        })
    }

    // ---- composition ----

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

    /// Induced subgraph on an explicit atom set. Stale handles fail-fast.
    /// Returns `(subgraph, parent→new handle map)`; the subgraph carries
    /// every relation (bonds, terms, ports, …) whose atoms all lie in the
    /// set.
    pub fn induced_subgraph(
        &self,
        atoms: &[NodeId],
    ) -> Result<(Atomistic, HashMap<NodeId, NodeId>), MolRsError> {
        let induced = self.graph.induced_subgraph(atoms)?;
        let atomistic = Atomistic::promote(induced.graph)?;
        Ok((atomistic, induced.node_map))
    }

    /// Radius ball around `centers` over the bond graph.
    ///
    /// When `regenerate_topology` is true, only bonds are copied from the parent
    /// and angles/dihedrals are perceived on the ball (O(ball)). When false,
    /// higher-order terms fully contained in the ball are copied (may scan the
    /// parent's higher-order tables — small-graph / verbatim path).
    /// `whole_groups` are atom sets the ball may not split (ring systems,
    /// say); see [`MolGraph::extract_ball`].
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
        let mut atomistic = Atomistic::promote(ball.graph)?;
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

    // ---- structural graph hash (see [`MolGraph::structural_hash`]) ----

    /// Isomorphism-invariant Weisfeiler–Lehman structural hash of the molecule
    /// (element / formal charge / aromatic node labels, bond-order edge labels). A
    /// stable, reproducible dedup key — identical for a node-permuted copy.
    pub fn structural_hash(&self) -> u64 {
        self.graph.structural_hash()
    }

    /// Deterministic canonical atom ordering from individualization/refinement, so two
    /// isomorphic molecules line up node-by-node (see
    /// [`MolGraph::canonical_order`]).
    pub fn canonical_order(&self) -> Vec<NodeId> {
        self.graph.canonical_order()
    }

    /// Whether `self` and `other` are isomorphic as labeled molecular graphs.
    pub fn is_isomorphic(&self, other: &Atomistic) -> bool {
        self.graph.is_isomorphic(&other.graph)
    }

    // Aromaticity perception belongs to `crate::perceive::assign_aromaticity`
    // (`assign_aromaticity`). No algorithm method here.
}

/// The typed methods `block` has on `Atomistic`, if it is one of the typed
/// relation blocks.
fn typed_block(block: &str) -> Option<&'static str> {
    TYPED_BLOCKS
        .iter()
        .find(|(name, _)| *name == block)
        .map(|&(_, methods)| methods)
}

/// The arity the Frame schema declares for the term block `block`.
///
/// # Errors
///
/// [`MolRsError::Validation`] when `block` is no canonical relation block
/// whose endpoints are `atoms` rows named `atomi`, `atomj`, … (`atoms` and
/// `members` are not).
fn schema_term_arity(block: &str) -> Result<usize, MolRsError> {
    let unknown = |why: &str| {
        MolRsError::validation(format!(
            "'{block}' is not a term block of this molecule: {why}; a term block is a \
             registered relation block or a canonical one over atoms (cmaps, constraints, \
             drudes, exclusions, pairs, virtual_sites)"
        ))
    };
    let spec = schema::block(block).ok_or_else(|| unknown("no row of it is held"))?;
    let RowKind::Relation { arity } = spec.row_kind else {
        return Err(unknown(&format!("its rows are {}s", spec.row_kind)));
    };
    let over_atoms = spec.endpoints.is_some_and(|e| {
        e.columns
            .iter()
            .enumerate()
            .all(|(pos, &(column, target))| {
                column == crate::core::molgraph::rel_col_name(pos)
                    && target == schema::EndpointTarget::Block(schema::block_names::ATOMS)
            })
    });
    if !over_atoms {
        return Err(unknown("its endpoints are not atoms rows"));
    }
    Ok(arity)
}

/// Refuse a direct write of a bond's class keys.
fn refuse_bond_class_key(key: &str, verb: &str) -> Result<(), MolRsError> {
    if key == keys::BOND_TYPE || key == keys::BOND_NUMBER {
        return Err(MolRsError::validation(format!(
            "'{key}' is not {verb} on its own: a bond's '{}' and '{}' are set together \
             with set_bond_class(id, BondOrder, BondNumber)",
            keys::BOND_TYPE,
            keys::BOND_NUMBER
        )));
    }
    Ok(())
}

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

/// Canonical (orientation-independent) key for an angle/dihedral endpoint
/// sequence: the lexicographically smaller of the sequence and its reverse.
fn canonical_path(nodes: &[NodeId]) -> Vec<NodeId> {
    let fwd = nodes.to_vec();
    let mut rev = fwd.clone();
    rev.reverse();
    if fwd <= rev { fwd } else { rev }
}

/// Promote a [`MolGraph`] to an `Atomistic`.
///
/// The graph's relation kinds are matched to the standard set **by name**:
/// an existing `bonds` / `angles` / `dihedrals` / `impropers` kind of the
/// right arity keeps its id, a missing one is registered fresh, and every
/// other kind (ports, terms) is kept.
///
/// # Errors
///
/// [`MolRsError::Validation`] when a node carries no valid
/// [`keys::ATOMIC_NUMBER`], and when the graph spells one of the standard kind
/// names at another arity (naming the kind and both arities).
impl TryFrom<MolGraph> for Atomistic {
    type Error = MolRsError;

    fn try_from(graph: MolGraph) -> Result<Self, MolRsError> {
        Atomistic::from_graph(graph)
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

impl sealed::Sealed for Atomistic {
    fn inner_graph(&self, _: sealed::Token) -> &MolGraph {
        &self.graph
    }

    fn inner_graph_mut(&mut self, _: sealed::Token) -> &mut MolGraph {
        &mut self.graph
    }

    fn from_inner_graph(graph: MolGraph, _: sealed::Token) -> Result<Self, MolRsError> {
        Atomistic::promote(graph)
    }
}

impl Graph for Atomistic {}

/// The guard an `Atomistic` installs on its graph: every node is an atom
/// with a valid atomic number.
const SPECIES_GUARD: NodeGuard = NodeGuard {
    key: keys::ATOMIC_NUMBER,
    check: require_atomic_number,
};

/// The species invariant: `atom` carries a valid [`keys::ATOMIC_NUMBER`].
///
/// # Errors
///
/// [`MolRsError::Validation`] naming `what`, with the explicit conversion to
/// use when the atom spells its species only as an [`keys::ELEMENT`] symbol.
fn require_atomic_number(atom: &Props, what: &str) -> Result<(), MolRsError> {
    if atom.atomic_number().is_some() {
        return Ok(());
    }
    let hint = if atom.get_str(keys::ELEMENT).is_some() {
        format!(
            "; it carries an '{}' symbol, which molrs does not read — convert \
             explicitly (Element::fill_atomic_numbers on the frame, or \
             Element::by_symbol on the value)",
            keys::ELEMENT
        )
    } else {
        String::new()
    };
    Err(MolRsError::validation(format!(
        "{what} has no valid '{}' (0..=118, 0 = wildcard){hint}",
        keys::ATOMIC_NUMBER
    )))
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::core::Adjacent;
    use crate::core::Block;
    use crate::core::Element;
    use crate::core::NeighborKinds;
    use crate::core::Props;
    use crate::core::keys;
    use crate::op::{I, Idx};
    use crate::props;
    use ndarray::Array1;
    use std::collections::HashSet;

    #[test]
    fn merge_returns_complete_node_map() {
        let mut a = Atomistic::new();
        let c1 = a
            .add_atom(props! { keys::ATOMIC_NUMBER => Element::C })
            .unwrap();
        let c2 = a
            .add_atom(props! { keys::ATOMIC_NUMBER => Element::C })
            .unwrap();
        a.add_bond(c1, c2, props! {}).unwrap();

        let mut b = Atomistic::new();
        let o = b
            .add_atom(props! { keys::ATOMIC_NUMBER => Element::O })
            .unwrap();
        let h = b
            .add_atom(props! { keys::ATOMIC_NUMBER => Element::H })
            .unwrap();
        b.add_bond(o, h, props! {}).unwrap();

        let map = a.merge(b).expect("merge succeeds on compatible graphs");
        assert_eq!(map.len(), 2);
        assert_eq!(a.n_atoms(), 4);
        assert_eq!(a.n_bonds(), 2);
        let new_o = map[&o];
        let new_h = map[&h];
        assert_eq!(a.get_atom(new_o).unwrap().atomic_number(), Some(8));
        assert_eq!(a.get_atom(new_h).unwrap().atomic_number(), Some(1));
        // remapped bond endpoints resolve
        let mut ends = HashSet::new();
        for (_, bond) in a.bonds() {
            ends.extend(bond.nodes.iter().copied());
        }
        assert!(ends.contains(&new_o) && ends.contains(&new_h));
    }

    #[test]
    fn clone_preserves_handles() {
        let mut mol = Atomistic::new();
        let c = mol
            .add_atom(props! { keys::ATOMIC_NUMBER => Element::C })
            .unwrap();
        let h = mol
            .add_atom(props! { keys::ATOMIC_NUMBER => Element::H })
            .unwrap();
        let bid = mol.add_bond(c, h, props! {}).unwrap();
        let cloned = mol.clone();
        assert!(cloned.get_atom(c).is_ok());
        assert!(cloned.get_atom(h).is_ok());
        assert!(cloned.get_bond(bid).is_ok());
    }

    #[test]
    fn test_new_and_add() {
        let mut mol = Atomistic::new();
        let c = mol
            .add_atom(props! { keys::ATOMIC_NUMBER => Element::C })
            .unwrap();
        let h = mol
            .add_atom(props! { keys::ATOMIC_NUMBER => Element::H })
            .unwrap();
        mol.add_bond(c, h, props! {}).unwrap();
        assert_eq!(mol.n_atoms(), 2);
        assert_eq!(mol.n_bonds(), 1);
        assert_eq!(mol.get_atom(c).unwrap().atomic_number(), Some(6));
    }

    #[test]
    fn add_atom_keeps_species_and_coordinates() {
        let mut mol = Atomistic::new();
        let o = mol
            .add_atom(props! {
                keys::ATOMIC_NUMBER => Element::O,
                keys::X => 0.0,
                keys::Y => 0.0,
                keys::Z => 0.0,
            })
            .unwrap();
        let atom = mol.get_atom(o).unwrap();
        assert_eq!(atom.atomic_number(), Some(8));
        assert_eq!(atom.get_f64("x"), Some(0.0));
    }

    /// Ethane (C2H6): C0-C1 plus 3 H on each carbon. 7 bonds.
    fn ethane() -> Atomistic {
        let mut mol = Atomistic::new();
        let atoms: Vec<NodeId> = [
            Element::C,
            Element::C,
            Element::H,
            Element::H,
            Element::H,
            Element::H,
            Element::H,
            Element::H,
        ]
        .iter()
        .map(|&e| mol.add_atom(props! { keys::ATOMIC_NUMBER => e }).unwrap())
        .collect();
        for (i, j) in [(0, 1), (0, 2), (0, 3), (0, 4), (1, 5), (1, 6), (1, 7)] {
            mol.add_bond(atoms[i], atoms[j], props! {}).unwrap();
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
        let atoms: Vec<NodeId> = [Element::C, Element::O, Element::H, Element::H]
            .iter()
            .map(|&e| mol.add_atom(props! { keys::ATOMIC_NUMBER => e }).unwrap())
            .collect();
        for (i, j) in [(0, 1), (0, 2), (0, 3)] {
            mol.add_bond(atoms[i], atoms[j], props! {}).unwrap();
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
        mol.add_improper(ids[0], ids[3], ids[1], ids[2], props! {})
            .unwrap();
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
        let atoms: Vec<NodeId> = eth.atom_ids().collect();
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
                sorted_hops(eth.topo_distances(&[src], None)),
                sorted_hops(expected_per_source[i].iter().map(|&d| (src, d)).collect()),
                "ethane source {src:?}"
            );
        }

        // 12-atom linear chain: an endpoint sees every hop count 0..=11 once.
        let mut chain = Atomistic::new();
        let ids: Vec<_> = (0..12)
            .map(|_| {
                chain
                    .add_atom(props! { keys::ATOMIC_NUMBER => Element::C })
                    .unwrap()
            })
            .collect();
        for k in 0..ids.len() - 1 {
            chain.add_bond(ids[k], ids[k + 1], props! {}).unwrap();
        }
        assert_eq!(
            sorted_hops(chain.topo_distances(&[ids[0]], None)),
            (0..12).collect::<Vec<_>>(),
        );
        // The mid-chain atom 5 sees a symmetric profile.
        assert_eq!(
            hops(chain.topo_distances(&[ids[0]], None)),
            (0..12).collect::<Vec<_>>(),
            "chain endpoint ordered by atom id"
        );

        // Unknown source → empty vec (error path). Use a genuinely-stale id
        // from the same arena (add then remove bumps the slot generation so
        // the old id no longer resolves) — a foreign arena's id can collide on
        // slot+generation and is not a reliable "absent" probe.
        let mut solo = Atomistic::new();
        let stale = solo
            .add_atom(props! { keys::ATOMIC_NUMBER => Element::C })
            .unwrap();
        solo.del_atom(stale).unwrap();
        assert!(solo.topo_distances(&[stale], None).is_empty());
    }

    #[test]
    fn test_full_topology_and_cascade() {
        let mut mol = Atomistic::new();
        let a = mol
            .add_atom(props! { keys::ATOMIC_NUMBER => Element::C })
            .unwrap();
        let b = mol
            .add_atom(props! { keys::ATOMIC_NUMBER => Element::C })
            .unwrap();
        let c = mol
            .add_atom(props! { keys::ATOMIC_NUMBER => Element::C })
            .unwrap();
        let d = mol
            .add_atom(props! { keys::ATOMIC_NUMBER => Element::C })
            .unwrap();
        mol.add_bond(a, b, props! {}).unwrap();
        mol.add_bond(a, c, props! {}).unwrap();
        mol.add_bond(a, d, props! {}).unwrap();
        mol.add_angle(b, a, c, props! {}).unwrap();
        mol.add_dihedral(b, a, c, d, props! {}).unwrap();
        mol.add_improper(b, a, c, d, props! {}).unwrap();
        assert_eq!(mol.n_bonds(), 3);
        assert_eq!(mol.n_angles(), 1);
        assert_eq!(mol.n_dihedrals(), 1);
        assert_eq!(mol.n_impropers(), 1);
        // typed RelationId usable as a HashSet key
        let ids: std::collections::HashSet<RelationId> = mol.bonds().map(|(id, _)| id).collect();
        assert_eq!(ids.len(), 3);
        // cascade on central atom
        mol.del_atom(a).unwrap();
        assert_eq!(mol.n_bonds(), 0);
        assert_eq!(mol.n_angles(), 0);
        assert_eq!(mol.n_dihedrals(), 0);
        assert_eq!(mol.n_impropers(), 0);
    }

    #[test]
    fn test_neighbors() {
        let mut mol = Atomistic::new();
        let o = mol
            .add_atom(props! { keys::ATOMIC_NUMBER => Element::O })
            .unwrap();
        let h1 = mol
            .add_atom(props! { keys::ATOMIC_NUMBER => Element::H })
            .unwrap();
        let h2 = mol
            .add_atom(props! { keys::ATOMIC_NUMBER => Element::H })
            .unwrap();
        mol.add_bond(o, h1, props! {}).unwrap();
        mol.add_bond(o, h2, props! {}).unwrap();
        let nb: Vec<(NodeId, RelationId)> = mol.neighbors(o, NeighborKinds::BONDS).collect();
        assert_eq!(nb.len(), 2);
        assert!(
            nb.iter()
                .all(|(_, bid)| mol.bond_number(*bid).unwrap() == BondNumber::Single)
        );
    }

    /// `neighbors` with [`NeighborKinds::BONDS`] answers "which atoms is this
    /// one bonded to". The generic `MolGraph::neighbors` walks every two-node
    /// relation, so a port and a constraint on the same atoms show up there
    /// as extra neighbours.
    #[test]
    fn neighbors_are_bonds_until_ports_and_terms_are_requested() {
        let mut mol = Atomistic::new();
        let o = mol
            .add_atom(props! { keys::ATOMIC_NUMBER => Element::O })
            .unwrap();
        let h = mol
            .add_atom(props! { keys::ATOMIC_NUMBER => Element::H })
            .unwrap();
        let bond = mol.add_bond(o, h, props! {}).unwrap();
        mol.add_port(o, h, PortKind::Symmetric, "", BondNumber::Single)
            .unwrap();
        mol.add_term("constraints", &[o, h], props! {}).unwrap();

        assert_eq!(
            mol.neighbors(o, NeighborKinds::BONDS).collect::<Vec<_>>(),
            vec![(h, bond)]
        );
        let with_ports: Vec<_> = mol
            .neighbors(o, NeighborKinds::BONDS.with_ports())
            .collect();
        assert_eq!(with_ports.len(), 2);
        assert!(with_ports.iter().any(|item| matches!(
            item,
            Adjacent::Bond { other, id } if *other == h && *id == bond
        )));
        assert!(with_ports.iter().any(|item| matches!(
            item,
            Adjacent::Port { other, .. } if *other == h
        )));
        let with_terms: Vec<_> = mol
            .neighbors(o, NeighborKinds::BONDS.with_ports().with_terms())
            .collect();
        assert_eq!(with_terms.len(), 3);
        assert!(with_terms.iter().any(|item| matches!(
            item,
            Adjacent::Term { block: "constraints", other, .. } if *other == h
        )));
        assert_eq!(mol.bond_between(o, h), Some(bond));
        assert_eq!(mol.bond_between(h, o), Some(bond));
        assert_eq!(
            mol.graph.neighbors(o).count(),
            3,
            "the generic walk sees the bond, the port and the constraint"
        );
    }

    #[test]
    fn test_frame_roundtrip() {
        let mut mol = Atomistic::new();
        let a = mol
            .add_atom(props! {
                keys::ATOMIC_NUMBER => Element::C,
                keys::X => 0.0,
                keys::Y => 0.0,
                keys::Z => 0.0,
            })
            .unwrap();
        let b = mol
            .add_atom(props! {
                keys::ATOMIC_NUMBER => Element::C,
                keys::X => 1.0,
                keys::Y => 0.0,
                keys::Z => 0.0,
            })
            .unwrap();
        let c = mol
            .add_atom(props! {
                keys::ATOMIC_NUMBER => Element::C,
                keys::X => 0.0,
                keys::Y => 1.0,
                keys::Z => 0.0,
            })
            .unwrap();
        let d = mol
            .add_atom(props! {
                keys::ATOMIC_NUMBER => Element::C,
                keys::X => 0.0,
                keys::Y => 0.0,
                keys::Z => 1.0,
            })
            .unwrap();
        mol.add_bond(a, b, props! {}).unwrap();
        mol.add_angle(a, b, c, props! {}).unwrap();
        mol.add_improper(a, b, c, d, props! {}).unwrap();
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
                .insert(
                    k,
                    Array1::from_vec(vec![0.0_f64, 1.0, 2.0]).into_dyn(),
                    None,
                )
                .unwrap();
        }
        atoms
            .insert(
                keys::ATOMIC_NUMBER,
                Array1::from_vec(vec![6 as Idx; 3]).into_dyn(),
                None,
            )
            .unwrap();
        let mut bonds = Block::new();
        bonds
            .insert("atomi", Array1::from_vec(vec![0u64, 1]).into_dyn(), None)
            .unwrap();
        bonds
            .insert("atomj", Array1::from_vec(vec![1u64, 2]).into_dyn(), None)
            .unwrap();
        let mut frame = Frame::new();
        frame.insert("atoms", atoms);
        frame.insert("bonds", bonds);

        let mol = Atomistic::from_frame(&frame).unwrap();
        assert_eq!(mol.n_bonds(), 2, "connectivity is read");
        for (id, _) in mol.bonds() {
            assert_eq!(
                mol.bond_type(id).unwrap(),
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
                .insert(
                    k,
                    Array1::from_vec(vec![0.0_f64, 1.0, 2.0, 3.0]).into_dyn(),
                    None,
                )
                .unwrap();
        }
        atoms
            .insert(
                keys::ATOMIC_NUMBER,
                Array1::from_vec(vec![6 as Idx; 4]).into_dyn(),
                None,
            )
            .unwrap();
        let mut bonds = Block::new();
        bonds
            .insert("atomi", Array1::from_vec(vec![0u64, 1, 2]).into_dyn(), None)
            .unwrap();
        bonds
            .insert("atomj", Array1::from_vec(vec![1u64, 2, 3]).into_dyn(), None)
            .unwrap();
        bonds
            .insert(
                "bond_type",
                Array1::from_vec(vec![2u64, 4, 0]).into_dyn(),
                None,
            )
            .unwrap();
        let mut frame = Frame::new();
        frame.insert("atoms", atoms);
        frame.insert("bonds", bonds);

        let mol = Atomistic::from_frame(&frame).unwrap();
        let got: Vec<BondOrder> = mol
            .bonds()
            .map(|(id, _)| mol.bond_type(id).unwrap())
            .collect();
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
        let a = mol
            .add_atom(props! {
                keys::ATOMIC_NUMBER => Element::C,
                keys::X => 0.0,
                keys::Y => 0.0,
                keys::Z => 0.0,
            })
            .unwrap();
        let b = mol
            .add_atom(props! {
                keys::ATOMIC_NUMBER => Element::C,
                keys::X => 1.54,
                keys::Y => 0.0,
                keys::Z => 0.0,
            })
            .unwrap();
        mol.add_bond(a, b, props! {}).unwrap();
        let mut frame = mol.to_frame().expect("a schema-conforming graph converts");
        let mut widgets = Block::new();
        widgets
            .insert("atomi", Array1::from_vec(vec![0_u64]).into_dyn(), None)
            .unwrap();
        widgets
            .insert("atomj", Array1::from_vec(vec![1_u64]).into_dyn(), None)
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
        let o = mol
            .add_atom(crate::props! { "atomic_number" => 8_u8, "x" => 0.0, "y" => 0.0, "z" => 0.0 })
            .unwrap();
        let h1 = mol
            .add_atom(
                crate::props! { "atomic_number" => 1_u8, "x" => 0.96, "y" => 0.0, "z" => 0.0 },
            )
            .unwrap();
        let h2 = mol
            .add_atom(
                crate::props! { "atomic_number" => 1_u8, "x" => -0.24, "y" => 0.93, "z" => 0.0 },
            )
            .unwrap();
        let m = mol
            .add_atom(crate::props! { "atomic_number" => 0_u8 })
            .unwrap();
        let shell = mol
            .add_atom(crate::props! { "atomic_number" => 0_u8 })
            .unwrap();
        mol.add_bond(o, h1, Props::new()).unwrap();
        mol.add_bond(o, h2, Props::new()).unwrap();
        mol.add_term(
            "drudes",
            &[o, shell],
            crate::props! { "type" => "OW-DOW", "k" => 500.0 },
        )
        .unwrap();
        mol.add_term(
            "virtual_sites",
            &[m, o, h1, h2],
            crate::props! { "style" => "average3", "w1" => 0.8 },
        )
        .unwrap();
        mol.add_term("constraints", &[h1, h2], Props::new())
            .unwrap();

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
            mol.add_atom(crate::props! { "atomic_number" => 6_u8 })
                .unwrap();
        }
        let mut frame = mol.to_frame().unwrap();
        let mut sites = Block::new();
        for (key, v) in [("atomi", 2_u64), ("atomj", 0), ("atomk", 1), ("atoml", 0)] {
            sites
                .insert(key, Array1::from_vec(vec![v]).into_dyn(), None)
                .unwrap();
        }
        sites.set_validity("atoml", vec![false]).unwrap();
        frame.insert("virtual_sites", sites);
        let err = Atomistic::from_frame(&frame).expect_err("a null endpoint");
        assert!(err.to_string().contains("atoml"), "{err}");
    }

    #[test]
    fn try_from_refuses_a_node_without_atomic_number() {
        let mut g = MolGraph::new();
        g.register_kind("bonds", 2);
        g.add_node(Props::new()).expect("fixture node"); // no atomic_number
        let msg = validation_message(Atomistic::try_from(g).unwrap_err());
        assert!(msg.contains(keys::ATOMIC_NUMBER), "{msg}");
    }

    /// The ids, the iterator and the read-only table agree on one row order.
    #[test]
    fn atom_ids_atoms_and_atom_table_share_row_order() {
        let mut mol = Atomistic::new();
        let ids: Vec<NodeId> = [1.0, 2.0, 3.0]
            .into_iter()
            .map(|q| {
                mol.add_atom(props! { keys::ATOMIC_NUMBER => Element::C, keys::CHARGE => q })
                    .unwrap()
            })
            .collect();
        assert_eq!(mol.atom_ids().collect::<Vec<_>>(), ids);
        assert_eq!(mol.atoms().map(|(id, _)| id).collect::<Vec<_>>(), ids);
        let (charges, _) = mol.atom_table().column::<f64>(keys::CHARGE).unwrap();
        assert_eq!(charges, &[1.0, 2.0, 3.0]);
    }

    /// A graph whose *first* registered kind is not `bonds` — what a fragment
    /// graph looks like — must not have that foreign kind's relations reported
    /// as its bonds, nor be written into by the next `add_bond`.
    #[test]
    fn try_from_resolves_bonds_by_name_not_kind_zero() {
        let mut graph = MolGraph::new();
        let ports = graph.register_kind("ports", 2);
        let c = graph
            .add_node(props! {
                keys::ATOMIC_NUMBER => Element::C,
                keys::X => 0.0,
                keys::Y => 0.0,
                keys::Z => 0.0,
            })
            .expect("fixture node");
        let h = graph
            .add_node(props! {
                keys::ATOMIC_NUMBER => Element::H,
                keys::X => 1.09,
                keys::Y => 0.0,
                keys::Z => 0.0,
            })
            .expect("fixture node");
        graph.add_relation(ports, &[c, h], props! {}).unwrap();

        let mut mol = Atomistic::try_from(graph).expect("elements are present");
        assert_eq!(mol.n_ports(), 1, "the foreign kind survives");
        assert_eq!(mol.n_bonds(), 0, "a port relation is not a bond");

        mol.add_bond(c, h, props! {})
            .expect("a bond can still be added");
        assert_eq!(mol.n_bonds(), 1);
        assert_eq!(
            mol.n_ports(),
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
        let a0 = mol
            .add_atom(props! { keys::ATOMIC_NUMBER => Element::C })
            .unwrap();
        mol.add_atom(props! { keys::ATOMIC_NUMBER => Element::C })
            .unwrap();
        mol.add_atom(props! { keys::ATOMIC_NUMBER => Element::C })
            .unwrap();
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
        let a = mol
            .add_atom(props! { keys::ATOMIC_NUMBER => Element::C })
            .unwrap();

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
    /// `add_atom` meets a float `x` column and the frame it produces is the
    /// schema-conforming one.
    #[test]
    fn set_atom_refusal_leaves_add_atom_a_float_x_column() {
        let mut mol = Atomistic::new();
        let a = mol
            .add_atom(props! { keys::ATOMIC_NUMBER => Element::C })
            .unwrap();

        mol.set_atom(a, "x", "left")
            .expect_err("step one of the panicking sequence is refused");

        let b = mol
            .add_atom(props! {
                keys::ATOMIC_NUMBER => Element::O,
                keys::X => 1.0,
                keys::Y => 2.0,
                keys::Z => 3.0,
            })
            .unwrap();
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
                .and_then(|c| c.get::<f64>())
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
                keys::ATOMIC_NUMBER,
                Array1::from_vec(vec![6 as Idx; 3]).into_dyn(),
                None,
            )
            .unwrap();
        let mut angles = Block::new();
        angles
            .insert("atomi", Array1::from_vec(vec![0 as Idx]).into_dyn(), None)
            .unwrap();
        angles
            .insert("atomj", Array1::from_vec(vec![1 as Idx]).into_dyn(), None)
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
                keys::ATOMIC_NUMBER,
                Array1::from_vec(vec![6 as Idx; 2]).into_dyn(),
                None,
            )
            .unwrap();
        let mut bonds = Block::new();
        bonds
            .insert("atomi", Array1::from_vec(vec![0 as Idx]).into_dyn(), None)
            .unwrap();
        bonds
            .insert("atomj", Array1::from_vec(vec![5 as Idx]).into_dyn(), None)
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
    fn try_from_rejects_conflicting_arity() {
        let mut graph = MolGraph::new();
        graph.register_kind("bonds", 3);
        let err = Atomistic::try_from(graph)
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
            let id = sys
                .add_atom(props! {
                    keys::ATOMIC_NUMBER => Element::C,
                    keys::X => 1.0,
                    keys::Y => 2.0,
                    keys::Z => 3.0,
                })
                .unwrap();
            let fixed = sys
                .add_atom(props! {
                    keys::ATOMIC_NUMBER => Element::C,
                    keys::X => 1.0,
                    keys::Y => 1.0,
                    keys::Z => 1.0,
                })
                .unwrap();
            sys.scale(factor, about);
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
        let light = mol
            .add_atom(props! {
                keys::ATOMIC_NUMBER => Element::H,
                keys::X => 0.0,
                keys::Y => 0.0,
                keys::Z => 0.0,
            })
            .unwrap();
        let heavy = mol
            .add_atom(props! {
                keys::ATOMIC_NUMBER => Element::Li,
                keys::X => 4.0,
                keys::Y => 0.0,
                keys::Z => 0.0,
            })
            .unwrap();
        mol.set_atom(light, crate::core::keys::MASS, 1.0).unwrap();
        mol.set_atom(heavy, crate::core::keys::MASS, 3.0).unwrap();
        let all: Vec<_> = mol.atom_ids().collect();
        assert_eq!(mol.center(&all), Ok([3.0, 0.0, 0.0]));

        assert_eq!(
            Atomistic::new().center(&[]),
            Err(crate::core::CenterError::Empty)
        );
    }

    // ---- the species invariant ----

    fn validation_message(err: MolRsError) -> String {
        match err {
            MolRsError::Validation { message } => message,
            other => panic!("expected a validation error, got {other:?}"),
        }
    }

    /// A bag without a valid atomic number is refused before anything is
    /// written: no node, and no column the refused bag would have started.
    #[test]
    fn add_atom_refuses_a_bag_without_atomic_number_and_writes_nothing() {
        let mut mol = Atomistic::new();
        mol.add_atom(props! { keys::ATOMIC_NUMBER => Element::C })
            .unwrap();

        for bag in [
            props! { keys::X => 1.0, keys::CHARGE => 0.5 },
            props! { keys::ATOMIC_NUMBER => 119u8, keys::X => 1.0 },
            props! { keys::ATOMIC_NUMBER => -6, keys::X => 1.0 },
            props! { keys::ATOMIC_NUMBER => "C", keys::X => 1.0 },
        ] {
            let msg = validation_message(mol.add_atom(bag).unwrap_err());
            assert!(msg.contains(keys::ATOMIC_NUMBER), "{msg}");
        }
        assert_eq!(mol.n_atoms(), 1);
        let frame = mol.to_frame().unwrap();
        let atoms = frame.get("atoms").unwrap();
        assert!(atoms.get(keys::X).is_none() && atoms.get(keys::CHARGE).is_none());
    }

    /// A bag that spells its species only as a symbol is refused with the
    /// explicit conversion to use — the symbol is never read as a species.
    #[test]
    fn add_atom_refuses_a_symbol_only_bag_and_names_the_conversion() {
        let mut mol = Atomistic::new();
        let msg = validation_message(mol.add_atom(props! { keys::ELEMENT => "C" }).unwrap_err());
        assert!(
            msg.contains("fill_atomic_numbers") && msg.contains("by_symbol"),
            "{msg}"
        );
        assert_eq!(mol.n_atoms(), 0);
    }

    #[test]
    fn add_atom_accepts_the_wildcard() {
        let mut mol = Atomistic::new();
        let star = mol.add_atom(props! { keys::ATOMIC_NUMBER => 0u8 }).unwrap();
        assert_eq!(mol.get_atom(star).unwrap().atomic_number(), Some(0));
    }

    #[test]
    fn set_atom_refuses_an_invalid_atomic_number_and_keeps_the_old_one() {
        let mut mol = Atomistic::new();
        let a = mol
            .add_atom(props! { keys::ATOMIC_NUMBER => Element::C })
            .unwrap();
        for bad in [
            PropValue::Int(119),
            PropValue::Int(-1),
            PropValue::from("O"),
        ] {
            let msg = validation_message(mol.set_atom(a, keys::ATOMIC_NUMBER, bad).unwrap_err());
            assert!(msg.contains(keys::ATOMIC_NUMBER), "{msg}");
            assert_eq!(mol.get_atom(a).unwrap().atomic_number(), Some(6));
        }
        mol.set_atom(a, keys::ATOMIC_NUMBER, Element::O).unwrap();
        assert_eq!(mol.get_atom(a).unwrap().atomic_number(), Some(8));
    }

    #[test]
    fn clear_atom_refuses_the_atomic_number_and_clears_anything_else() {
        let mut mol = Atomistic::new();
        let a = mol
            .add_atom(props! { keys::ATOMIC_NUMBER => Element::N, keys::CHARGE => -0.5 })
            .unwrap();
        let msg = validation_message(mol.clear_atom(a, keys::ATOMIC_NUMBER).unwrap_err());
        assert!(msg.contains(keys::ATOMIC_NUMBER), "{msg}");
        assert_eq!(mol.get_atom(a).unwrap().atomic_number(), Some(7));

        mol.clear_atom(a, keys::CHARGE).unwrap();
        assert!(mol.get_atom(a).unwrap().get(keys::CHARGE).is_none());
    }

    /// `from_frame` reads species only from `atomic_number`: a frame that
    /// spells symbols is refused with the conversion to run, and reads once
    /// that conversion ran.
    #[test]
    fn from_frame_requires_atomic_numbers_and_hints_fill_atomic_numbers() {
        let mut atoms = Block::new();
        atoms
            .insert(
                keys::ELEMENT,
                Array1::from_vec(vec!["O".to_string(), "H".to_string()]).into_dyn(),
                None,
            )
            .unwrap();
        let mut frame = Frame::new();
        frame.insert("atoms", atoms);

        let msg = validation_message(Atomistic::from_frame(&frame).unwrap_err());
        assert!(msg.contains("fill_atomic_numbers"), "{msg}");

        Element::fill_atomic_numbers(&mut frame).unwrap();
        let mol = Atomistic::from_frame(&frame).unwrap();
        let mut species: Vec<u8> = mol
            .atoms()
            .map(|(_, a)| a.atomic_number().unwrap())
            .collect();
        species.sort_unstable();
        assert_eq!(species, [1, 8]);
    }

    #[test]
    fn from_frame_refuses_atoms_with_no_species_at_all() {
        let mut atoms = Block::new();
        atoms
            .insert(keys::X, Array1::from_vec(vec![0.0_f64]).into_dyn(), None)
            .unwrap();
        let mut frame = Frame::new();
        frame.insert("atoms", atoms);
        let msg = validation_message(Atomistic::from_frame(&frame).unwrap_err());
        assert!(msg.contains(keys::ATOMIC_NUMBER), "{msg}");
    }

    /// The species invariant lives on the graph, so the generic node
    /// mutators the crate reaches through `graph_mut` hold it too.
    #[test]
    fn the_species_guard_holds_on_the_generic_node_mutators() {
        let mut mol = Atomistic::new();
        let c = mol
            .add_atom(props! { keys::ATOMIC_NUMBER => Element::C })
            .unwrap();

        let raw = mol.graph_mut();
        assert!(raw.add_node(props! { keys::X => 0.0 }).is_err());
        assert!(raw.set_node(c, keys::ATOMIC_NUMBER, 119u8).is_err());
        assert!(raw.clear_node(c, keys::ATOMIC_NUMBER).is_err());
        // Any other key is free data.
        raw.set_node(c, keys::CHARGE, 0.25).unwrap();
        raw.clear_node(c, keys::CHARGE).unwrap();
        assert_eq!(mol.n_atoms(), 1);
        assert_eq!(mol.get_atom(c).unwrap().atomic_number(), Some(6));
    }

    /// Merging or replicating a bare graph into a molecule checks every
    /// incoming node first, and refuses before writing anything.
    #[test]
    fn the_species_guard_refuses_a_bare_merge_and_replicate_up_front() {
        let mut bare = MolGraph::new();
        bare.add_node(props! { keys::ATOMIC_NUMBER => Element::O })
            .unwrap();
        bare.add_node(props! { keys::X => 1.0 }).unwrap();

        let mut mol = Atomistic::new();
        mol.add_atom(props! { keys::ATOMIC_NUMBER => Element::C })
            .unwrap();
        assert!(mol.graph_mut().merge(bare.clone()).is_err());
        assert_eq!(mol.n_atoms(), 1, "no incoming node was written");
        assert!(
            mol.graph_mut()
                .replicate(&bare, &[crate::op::Rigid::IDENTITY], &[1])
                .is_err()
        );
        assert_eq!(mol.n_atoms(), 1);
    }

    /// Promotion re-checks every node, and the promoted molecule carries the
    /// guard again.
    #[test]
    fn try_from_checks_every_node_and_reinstalls_the_guard() {
        let mut bare = MolGraph::new();
        bare.add_node(props! { keys::ATOMIC_NUMBER => Element::C })
            .unwrap();
        let x = bare.add_node(props! {}).unwrap();
        assert!(Atomistic::try_from(bare.clone()).is_err());

        bare.set_node(x, keys::ATOMIC_NUMBER, Element::H).unwrap();
        let mut again = Atomistic::try_from(bare).unwrap();
        assert_eq!(again.n_atoms(), 2);
        assert!(
            again.graph_mut().add_node(props! {}).is_err(),
            "the guard is back"
        );
    }

    // ---- deletion: one entity vs a batch ----

    /// A chain C0-C1-C2-C3 with one angle and one dihedral over it.
    fn chain4() -> (Atomistic, Vec<NodeId>, Vec<RelationId>) {
        let mut mol = Atomistic::new();
        let ids: Vec<NodeId> = (0..4)
            .map(|_| {
                mol.add_atom(props! { keys::ATOMIC_NUMBER => Element::C })
                    .unwrap()
            })
            .collect();
        let bonds = (0..3)
            .map(|k| mol.add_bond(ids[k], ids[k + 1], props! {}).unwrap())
            .collect();
        mol.add_angle(ids[0], ids[1], ids[2], props! {}).unwrap();
        mol.add_dihedral(ids[0], ids[1], ids[2], ids[3], props! {})
            .unwrap();
        (mol, ids, bonds)
    }

    #[test]
    fn del_atom_returns_the_bag_and_cascades() {
        let (mut mol, ids, _) = chain4();
        mol.set_atom(ids[1], keys::CHARGE, -0.5).unwrap();
        let bag = mol.del_atom(ids[1]).unwrap();
        assert_eq!(bag.get_f64(keys::CHARGE), Some(-0.5));
        assert_eq!(mol.n_atoms(), 3);
        assert_eq!(mol.n_bonds(), 1, "only C2-C3 survives");
        assert_eq!((mol.n_angles(), mol.n_dihedrals()), (0, 0));
        assert!(matches!(
            mol.del_atom(ids[1]),
            Err(MolRsError::NotFound { .. })
        ));
    }

    #[test]
    fn del_atoms_deletes_a_set_in_order() {
        let (mut mol, ids, _) = chain4();
        let bags = mol.del_atoms(&[ids[3], ids[0]]).unwrap();
        assert_eq!(bags.len(), 2);
        assert_eq!(
            mol.atom_ids().collect::<HashSet<_>>(),
            HashSet::from([ids[1], ids[2]])
        );
        assert_eq!(mol.n_bonds(), 1);
        assert!(mol.del_atoms(&[]).unwrap().is_empty());
    }

    /// A batch with a dead or a repeated handle is refused whole: nothing of
    /// it is deleted, not even the live atoms listed before the bad one.
    #[test]
    fn del_atoms_is_atomic() {
        let (mut mol, ids, _) = chain4();
        let dead = mol
            .add_atom(props! { keys::ATOMIC_NUMBER => Element::H })
            .unwrap();
        mol.del_atom(dead).unwrap();

        assert!(matches!(
            mol.del_atoms(&[ids[0], dead]),
            Err(MolRsError::NotFound { .. })
        ));
        assert!(matches!(
            mol.del_atoms(&[ids[0], ids[0]]),
            Err(MolRsError::Validation { .. })
        ));
        assert_eq!(mol.n_atoms(), 4);
        assert_eq!(mol.n_bonds(), 3);
        assert_eq!(mol.n_dihedrals(), 1);
    }

    #[test]
    fn del_bond_and_del_bonds_leave_the_atoms() {
        let (mut mol, ids, bonds) = chain4();
        let gone = mol.del_bond(bonds[0]).unwrap();
        assert_eq!(gone.nodes.as_slice(), &[ids[0], ids[1]]);
        assert!(matches!(
            mol.del_bond(bonds[0]),
            Err(MolRsError::NotFound { .. })
        ));
        // A batch naming the already-deleted bond deletes nothing.
        assert!(mol.del_bonds(&[bonds[1], bonds[0]]).is_err());
        assert_eq!(mol.n_bonds(), 2);
        assert!(mol.del_bonds(&[bonds[1], bonds[1]]).is_err());
        assert_eq!(mol.n_bonds(), 2);

        let gone = mol.del_bonds(&[bonds[2], bonds[1]]).unwrap();
        assert_eq!(gone[0].nodes.as_slice(), &[ids[2], ids[3]]);
        assert_eq!(mol.n_bonds(), 0);
        assert_eq!(mol.n_atoms(), 4);
        assert_eq!(mol.bond_ids().count(), 0);
    }

    /// Angles, dihedrals and impropers share the eight operations; one pass
    /// over each checks set / get / clear / ids / del and the batch refusal.
    #[test]
    fn angle_dihedral_and_improper_operations() {
        let (mut mol, ids, _) = chain4();
        let imp = mol
            .add_improper(ids[1], ids[0], ids[2], ids[3], props! {})
            .unwrap();
        let ang = mol.angle_ids().next().unwrap();
        let dih = mol.dihedral_ids().next().unwrap();

        mol.set_angle(ang, "theta0", 109.5).unwrap();
        mol.set_dihedral(dih, keys::TYPE, "CT-CT-CT-CT").unwrap();
        mol.set_improper(imp, "k", 2.0).unwrap();
        assert_eq!(
            mol.get_angle(ang).unwrap().props.get_f64("theta0"),
            Some(109.5)
        );
        assert_eq!(
            mol.get_dihedral(dih).unwrap().props.get_str(keys::TYPE),
            Some("CT-CT-CT-CT")
        );
        assert_eq!(mol.get_improper(imp).unwrap().props.get_f64("k"), Some(2.0));

        mol.clear_angle(ang, "theta0").unwrap();
        mol.clear_dihedral(dih, keys::TYPE).unwrap();
        mol.clear_improper(imp, "k").unwrap();
        assert!(mol.get_angle(ang).unwrap().props.is_empty());
        assert!(mol.get_dihedral(dih).unwrap().props.is_empty());
        assert!(mol.get_improper(imp).unwrap().props.is_empty());
        assert_eq!(mol.improper_ids().collect::<Vec<_>>(), vec![imp]);

        assert!(mol.del_angles(&[ang, ang]).is_err());
        assert_eq!(mol.n_angles(), 1);
        mol.del_angle(ang).unwrap();
        mol.del_dihedrals(&[dih]).unwrap();
        mol.del_improper(imp).unwrap();
        assert_eq!(
            (mol.n_angles(), mol.n_dihedrals(), mol.n_impropers()),
            (0, 0, 0)
        );
        assert!(mol.set_angle(ang, "theta0", 1.0).is_err());
        assert!(mol.clear_dihedral(dih, keys::TYPE).is_err());
        assert!(mol.del_impropers(&[imp]).is_err());
        assert_eq!(mol.n_bonds(), 3, "deleting terms leaves the bonds");
    }

    // ---- bond properties and class ----

    #[test]
    fn set_bond_refuses_the_class_keys_and_writes_any_other() {
        let (mut mol, _, bonds) = chain4();
        let b = bonds[0];
        for key in [keys::BOND_TYPE, keys::BOND_NUMBER] {
            let msg = validation_message(mol.set_bond(b, key, 2).unwrap_err());
            assert!(msg.contains("set_bond_class"), "{msg}");
            let msg = validation_message(mol.clear_bond(b, key).unwrap_err());
            assert!(msg.contains("set_bond_class"), "{msg}");
        }
        assert_eq!(
            mol.bond_type(b).unwrap(),
            BondOrder::Single,
            "the class is untouched"
        );
        assert_eq!(mol.bond_number(b).unwrap(), BondNumber::Single);

        mol.set_bond(b, keys::TYPE, "CT-CT").unwrap();
        assert_eq!(
            mol.get_bond(b).unwrap().props.get_str(keys::TYPE),
            Some("CT-CT")
        );
        mol.clear_bond(b, keys::TYPE).unwrap();
        assert_eq!(mol.get_bond(b).unwrap().props.get(keys::TYPE), None);

        mol.set_bond_class(b, BondOrder::Aromatic, BondNumber::Double)
            .unwrap();
        assert_eq!(mol.bond_type(b).unwrap(), BondOrder::Aromatic);
        assert_eq!(mol.bond_number(b).unwrap(), BondNumber::Double);

        mol.del_bond(b).unwrap();
        assert!(matches!(
            mol.set_bond(b, keys::TYPE, "CT-CT"),
            Err(MolRsError::NotFound { .. })
        ));
        assert!(matches!(
            mol.clear_bond(b, keys::TYPE),
            Err(MolRsError::NotFound { .. })
        ));
    }

    #[test]
    fn bond_between_finds_the_bond_either_way_round() {
        let (mol, ids, bonds) = chain4();
        assert_eq!(mol.bond_between(ids[1], ids[2]), Some(bonds[1]));
        assert_eq!(mol.bond_between(ids[2], ids[1]), Some(bonds[1]));
        assert_eq!(mol.bond_between(ids[0], ids[3]), None);
    }

    // ---- the `*_term` family ----

    /// Five atoms for a CMAP term.
    fn cmap_chain() -> (Atomistic, Vec<NodeId>) {
        let (mut mol, mut ids, _) = chain4();
        let tail = mol
            .add_atom(props! { keys::ATOMIC_NUMBER => Element::N })
            .unwrap();
        mol.add_bond(ids[3], tail, props! {}).unwrap();
        ids.push(tail);
        (mol, ids)
    }

    #[test]
    fn add_term_takes_its_arity_from_the_schema_and_round_trips() {
        let (mut mol, ids) = cmap_chain();
        assert_eq!(mol.term_blocks().count(), 0);
        assert_eq!(
            mol.n_terms("cmaps").unwrap(),
            0,
            "a canonical block with no row"
        );

        let cmap = mol
            .add_term("cmaps", &ids, props! { keys::TYPE => "C1" })
            .unwrap();
        let excl = mol.add_term("exclusions", &ids[..2], props! {}).unwrap();
        assert_eq!(
            mol.term_blocks().collect::<Vec<_>>(),
            ["cmaps", "exclusions"]
        );
        assert_eq!(mol.n_terms("cmaps").unwrap(), 1);
        assert_eq!(
            mol.term_ids("exclusions").unwrap().collect::<Vec<_>>(),
            [excl]
        );
        let read = mol.get_term("cmaps", cmap).unwrap();
        assert_eq!(read.nodes.as_slice(), ids.as_slice());
        assert_eq!(read.props.get_str(keys::TYPE), Some("C1"));

        let frame = mol.to_frame().unwrap();
        let block = frame.get("cmaps").expect("the term block is written");
        assert_eq!(block.n_rows(), Some(1));
        assert!(block.get("atomm").is_some());

        let back = Atomistic::from_frame(&frame).unwrap();
        assert_eq!(
            back.term_blocks().collect::<Vec<_>>(),
            ["cmaps", "exclusions"],
            "from_frame reads back every term block to_frame wrote"
        );
        assert_eq!(back.n_terms("cmaps").unwrap(), 1);
        assert_eq!(back.n_terms("exclusions").unwrap(), 1);
        let row = |m: &Atomistic, id: NodeId| m.atom_ids().position(|a| a == id).unwrap();
        let back_ids: Vec<NodeId> = back.atom_ids().collect();
        let cmap_back = back.term_ids("cmaps").unwrap().next().unwrap();
        let read_back = back.get_term("cmaps", cmap_back).unwrap();
        let rows_back: Vec<usize> = read_back.nodes.iter().map(|&n| row(&back, n)).collect();
        let rows: Vec<usize> = ids.iter().map(|&n| row(&mol, n)).collect();
        assert_eq!(rows_back, rows, "the cmap's atoms round-trip in order");
        assert_eq!(read_back.props.get_str(keys::TYPE), Some("C1"));
        let excl_back = back.term_ids("exclusions").unwrap().next().unwrap();
        let excl_nodes = back.get_term("exclusions", excl_back).unwrap().nodes;
        assert_eq!(excl_nodes.as_slice(), &back_ids[..2]);
        assert_eq!(
            back.to_frame().unwrap().get("cmaps").unwrap().n_rows(),
            Some(1)
        );
    }

    #[test]
    fn add_term_refuses_a_wrong_arity_and_registers_nothing() {
        let (mut mol, ids) = cmap_chain();
        let msg = validation_message(mol.add_term("cmaps", &ids[..4], props! {}).unwrap_err());
        assert!(msg.contains("arity 5"), "{msg}");
        assert_eq!(mol.term_blocks().count(), 0, "no kind was registered");

        mol.add_term("cmaps", &ids, props! {}).unwrap();
        // Once held, the block keeps its own arity.
        assert!(mol.add_term("cmaps", &ids[..3], props! {}).is_err());
        assert_eq!(mol.n_terms("cmaps").unwrap(), 1);
    }

    /// A block the schema does not define is usable once declared; the
    /// declaration is checked against the schema, the typed blocks and an
    /// earlier declaration.
    #[test]
    fn declare_term_admits_a_custom_block_at_its_declared_arity() {
        let (mut mol, ids) = cmap_chain();
        assert!(mol.add_term("urey_bradleys", &ids[..3], props! {}).is_err());

        mol.declare_term("urey_bradleys", 3).unwrap();
        mol.declare_term("urey_bradleys", 3).unwrap(); // same arity: no-op
        let ub = mol
            .add_term("urey_bradleys", &ids[..3], props! { "k" => 1.5 })
            .unwrap();
        assert_eq!(
            mol.get_term("urey_bradleys", ub).unwrap().nodes.as_slice(),
            &ids[..3]
        );
        assert!(mol.add_term("urey_bradleys", &ids[..2], props! {}).is_err());

        let other = validation_message(mol.declare_term("urey_bradleys", 2).unwrap_err());
        assert!(other.contains("arity 3"), "{other}");
        let schema = validation_message(mol.declare_term("cmaps", 4).unwrap_err());
        assert!(schema.contains("arity 5"), "{schema}");
        mol.declare_term("cmaps", 5).unwrap();
        assert!(mol.declare_term("bonds", 2).is_err());
        assert!(mol.declare_term("empty", 0).is_err());
    }

    #[test]
    fn add_term_refuses_a_dead_atom_and_registers_nothing() {
        let (mut mol, ids) = cmap_chain();
        mol.del_atom(ids[4]).unwrap();
        assert!(matches!(
            mol.add_term("cmaps", &ids, props! {}),
            Err(MolRsError::NotFound { .. })
        ));
        assert_eq!(mol.term_blocks().count(), 0);
    }

    /// The typed blocks have their own methods, named in the refusal; a block
    /// with no arity to read, or whose rows are not atom tuples, is refused.
    #[test]
    fn the_term_family_refuses_typed_unknown_and_non_atom_blocks() {
        let (mut mol, ids) = cmap_chain();
        for (block, method) in [
            ("bonds", "add_bond"),
            ("angles", "add_angle"),
            ("dihedrals", "add_dihedral"),
            ("impropers", "add_improper"),
            ("ports", "add_port"),
        ] {
            let msg = validation_message(mol.add_term(block, &ids[..2], props! {}).unwrap_err());
            assert!(msg.contains(method), "{block}: {msg}");
            assert!(mol.n_terms(block).is_err(), "{block}");
            assert!(mol.term_ids(block).is_err(), "{block}");
        }
        for block in ["cmapz", "atoms", "members"] {
            let msg = validation_message(mol.add_term(block, &ids[..2], props! {}).unwrap_err());
            assert!(msg.contains(block), "{msg}");
            assert!(mol.n_terms(block).is_err(), "{block}");
        }
        assert_eq!(mol.term_blocks().count(), 0);
        assert_eq!(mol.n_bonds(), 4);
    }

    #[test]
    fn term_rows_are_set_cleared_and_deleted_singly_or_atomically() {
        let (mut mol, ids) = cmap_chain();
        let a = mol.add_term("constraints", &ids[..2], props! {}).unwrap();
        let b = mol.add_term("constraints", &ids[1..3], props! {}).unwrap();
        let c = mol.add_term("constraints", &ids[2..4], props! {}).unwrap();

        mol.set_term("constraints", a, "r0", 1.53).unwrap();
        assert_eq!(
            mol.get_term("constraints", a).unwrap().props.get_f64("r0"),
            Some(1.53)
        );
        mol.clear_term("constraints", a, "r0").unwrap();
        assert!(mol.get_term("constraints", a).unwrap().props.is_empty());

        let gone = mol.del_term("constraints", a).unwrap();
        assert_eq!(gone.nodes.as_slice(), &ids[..2]);
        assert!(matches!(
            mol.get_term("constraints", a),
            Err(MolRsError::NotFound { .. })
        ));
        // `a` is dead: the batch is refused whole, `b` and `c` stay.
        assert!(mol.del_terms("constraints", &[b, a]).is_err());
        assert!(mol.del_terms("constraints", &[b, b]).is_err());
        assert_eq!(mol.n_terms("constraints").unwrap(), 2);
        assert_eq!(mol.del_terms("constraints", &[c, b]).unwrap().len(), 2);
        assert_eq!(mol.n_terms("constraints").unwrap(), 0);
        assert_eq!(mol.n_bonds(), 4, "deleting terms leaves the bonds");

        // A term handle is addressed within its block.
        assert!(matches!(
            mol.get_term("drudes", b),
            Err(MolRsError::NotFound { .. })
        ));
        assert!(mol.del_terms("drudes", &[]).unwrap().is_empty());
    }

    #[test]
    fn deleting_an_atom_deletes_its_terms() {
        let (mut mol, ids) = cmap_chain();
        mol.add_term("cmaps", &ids, props! {}).unwrap();
        mol.add_term("exclusions", &[ids[0], ids[4]], props! {})
            .unwrap();
        mol.del_atom(ids[4]).unwrap();
        assert_eq!(mol.n_terms("cmaps").unwrap(), 0);
        assert_eq!(mol.n_terms("exclusions").unwrap(), 0);
    }

    // ---- ports ----

    #[test]
    fn port_operations_single_and_batch() {
        let (mut mol, ids, _) = chain4();
        let mut hydrogens = Vec::new();
        for &c in &[ids[0], ids[3]] {
            let h = mol
                .add_atom(props! { keys::ATOMIC_NUMBER => Element::H })
                .unwrap();
            mol.add_bond(c, h, props! {}).unwrap();
            hydrogens.push(h);
        }
        let p0 = mol
            .add_port(
                ids[0],
                hydrogens[0],
                PortKind::Left,
                "a",
                BondNumber::Single,
            )
            .unwrap();
        let p1 = mol
            .add_port(
                ids[3],
                hydrogens[1],
                PortKind::Right,
                "a",
                BondNumber::Single,
            )
            .unwrap();
        assert_eq!(mol.port_ids().collect::<Vec<_>>(), [p0, p1]);
        assert_eq!(mol.port_on(ids[0], hydrogens[0]), Some(p0));
        let port = mol.get_port(p0).unwrap();
        assert_eq!((port.anchor, port.handle), (ids[0], hydrogens[0]));
        assert_eq!(
            mol.leaving_group(&port).unwrap(),
            BTreeSet::from([hydrogens[0]])
        );

        let gone = mol.del_port(p0).unwrap();
        assert_eq!(gone.nodes.as_slice(), &[ids[0], hydrogens[0]]);
        assert_eq!(mol.n_ports(), 1);
        assert_eq!(mol.n_bonds(), 5, "the port's bond stays");
        assert!(mol.del_ports(&[p1, p0]).is_err());
        assert_eq!(mol.n_ports(), 1, "a refused batch deletes nothing");
        assert_eq!(mol.del_ports(&[p1]).unwrap().len(), 1);
        assert_eq!(mol.n_ports(), 0);
        assert!(matches!(mol.get_port(p1), Err(MolRsError::NotFound { .. })));
    }

    #[test]
    fn del_ports_on_a_molecule_without_ports() {
        let (mut mol, _, bonds) = chain4();
        assert!(mol.del_ports(&[]).unwrap().is_empty());
        assert!(matches!(
            mol.del_port(bonds[0]),
            Err(MolRsError::NotFound { .. })
        ));
    }

    #[test]
    fn link_joins_two_ports_through_the_wrapper() {
        let mut mol = Atomistic::new();
        let mut unit = |kind: PortKind| {
            let c = mol
                .add_atom(props! { keys::ATOMIC_NUMBER => Element::C })
                .unwrap();
            let h = mol
                .add_atom(props! { keys::ATOMIC_NUMBER => Element::H })
                .unwrap();
            mol.add_bond(c, h, props! {}).unwrap();
            mol.set_atom(c, keys::FRAG_ID, PropValue::Int(1)).unwrap();
            (c, mol.add_port(c, h, kind, "", BondNumber::Single).unwrap())
        };
        let (c0, p0) = unit(PortKind::Left);
        let (c1, p1) = unit(PortKind::Right);
        let bond = mol.link(p0, p1).unwrap();
        assert_eq!(mol.bond_between(c0, c1), Some(bond));
        assert_eq!((mol.n_atoms(), mol.n_ports()), (2, 0));
    }

    #[test]
    fn inherit_frag_ids_labels_a_new_hydrogen() {
        let (mut mol, ids, _) = chain4();
        for &c in &ids {
            mol.set_atom(c, keys::FRAG_ID, PropValue::Int(4)).unwrap();
        }
        let h = mol
            .add_atom(props! { keys::ATOMIC_NUMBER => Element::H })
            .unwrap();
        mol.add_bond(ids[0], h, props! {}).unwrap();
        assert_eq!(mol.inherit_frag_ids(), 1);
        assert_eq!(mol.get_atom(h).unwrap().get_int(keys::FRAG_ID), Some(4));
    }

    // ---- geometry ----

    #[test]
    fn translate_and_rotate_move_every_placed_atom() {
        let mut mol = Atomistic::new();
        let a = mol
            .add_atom(props! {
                keys::ATOMIC_NUMBER => Element::C,
                keys::X => 1.0,
                keys::Y => 0.0,
                keys::Z => 0.0,
            })
            .unwrap();
        mol.translate([0.0, 0.0, 2.0]);
        assert_eq!(mol.get_atom(a).unwrap().position(), Some([1.0, 0.0, 2.0]));
        mol.rotate([0.0, 0.0, 1.0], std::f64::consts::FRAC_PI_2, None)
            .unwrap();
        let [x, y, z] = mol.get_atom(a).unwrap().position().unwrap();
        assert!(x.abs() < 1e-12 && (y - 1.0).abs() < 1e-12 && (z - 2.0).abs() < 1e-12);
        assert!(mol.rotate([0.0; 3], 1.0, None).is_err());
    }

    // ---- composition and hashing through the wrapper ----

    #[test]
    fn induced_subgraph_keeps_terms_and_ports_inside_the_set() {
        let (mut mol, ids) = cmap_chain();
        mol.add_term("exclusions", &ids[..2], props! {}).unwrap();
        mol.add_term("exclusions", &[ids[0], ids[4]], props! {})
            .unwrap();
        let (sub, map) = mol.induced_subgraph(&ids[..3]).unwrap();
        assert_eq!((sub.n_atoms(), sub.n_bonds()), (3, 2));
        assert_eq!(sub.n_terms("exclusions").unwrap(), 1);
        assert_eq!(sub.n_angles(), 1);
        assert_eq!(map.len(), 3);
    }

    #[test]
    fn hashing_through_the_methods() {
        let (a, _, _) = chain4();
        let (b, _, _) = chain4();
        assert_eq!(a.structural_hash(), b.structural_hash());
        assert!(a.is_isomorphic(&b));
        assert_eq!(a.canonical_order().len(), 4);
    }
}
