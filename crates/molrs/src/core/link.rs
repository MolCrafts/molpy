use std::collections::hash_map::Entry;
use std::collections::{BTreeSet, HashMap};
use std::fmt;

use slotmap::Key;

use crate::core::MolGraph;
use crate::core::MolRsError;
use crate::core::Port;
use crate::core::keys;
use crate::core::{NodeId, RelationId};

/// Why [`MolGraph::link`] refused to join two ports.
#[derive(Debug)]
pub enum LinkError {
    /// A port did not read back (stale id or malformed descriptor).
    Port(MolRsError),
    /// The port reads back, but its anchor–handle bond no longer exists: the
    /// bond was deleted after [`MolGraph::add_port`] checked it, so the port
    /// has no leaving group.
    StalePort {
        /// The stale port.
        port: RelationId,
    },
    /// [`Port::accepts`] refuses the pair: the kinds are not complements, or
    /// the labels or orders differ.
    Incompatible {
        /// The first port.
        a: RelationId,
        /// The second port.
        b: RelationId,
    },
    /// Both ports sit on one anchor, and a bond cannot join an atom to itself.
    SameAnchor {
        /// The first port.
        a: RelationId,
        /// The second port.
        b: RelationId,
    },
    /// The two anchors are already bonded; linking them would add a duplicate
    /// bond.
    AlreadyBonded {
        /// The first port's anchor.
        a: NodeId,
        /// The second port's anchor.
        b: NodeId,
    },
    /// The port's branch — its handle's component once the anchor–handle bond
    /// is cut — contains the port's own anchor: the handle sits on a ring, so
    /// deleting the branch would delete the anchor.
    BranchReachesAnchor {
        /// The offending port.
        port: RelationId,
    },
    /// The two branches share an atom, or one port's anchor lies in the other
    /// port's branch.
    BranchesOverlap,
    /// Charge is present on only part of `{anchor} ∪ branch`, so folding it
    /// would invent or lose charge.
    OneSidedCharge {
        /// The anchor of the offending port.
        anchor: NodeId,
    },
    /// The graph refused a read during validation.
    Graph(MolRsError),
}

impl fmt::Display for LinkError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Port(e) => write!(f, "port does not read back: {e}"),
            Self::StalePort { port } => write!(
                f,
                "port {port:?} is stale: its anchor–handle bond no longer exists"
            ),
            Self::Incompatible { a, b } => {
                write!(f, "ports {a:?} and {b:?} are not compatible")
            }
            Self::SameAnchor { a, b } => write!(
                f,
                "ports {a:?} and {b:?} share one anchor; a bond cannot join an atom to itself"
            ),
            Self::AlreadyBonded { a, b } => {
                write!(f, "anchors {a:?} and {b:?} are already bonded")
            }
            Self::BranchReachesAnchor { port } => {
                write!(f, "port {port:?}'s handle branch reaches its own anchor")
            }
            Self::BranchesOverlap => write!(f, "the two ports' handle branches overlap"),
            Self::OneSidedCharge { anchor } => write!(
                f,
                "charge is present on only part of anchor {anchor:?} and its handle branch"
            ),
            Self::Graph(e) => write!(f, "graph refused a read: {e}"),
        }
    }
}

impl std::error::Error for LinkError {
    fn source(&self) -> Option<&(dyn std::error::Error + 'static)> {
        match self {
            Self::Port(e) | Self::Graph(e) => Some(e),
            _ => None,
        }
    }
}

/// Why the crate-internal batch join `MolGraph::link_many` refused a batch
/// of port pairs.
///
/// Pairs are named by their 0-based index in the slice passed. In the
/// two-pair variants `first < second`.
#[derive(Debug)]
pub enum LinkManyError {
    /// Pair `pair`, checked alone, is refused as [`MolGraph::link`] would
    /// refuse it.
    Pair {
        /// Index of the refused pair.
        pair: usize,
        /// The refusal of that pair alone.
        source: LinkError,
    },
    /// One port appears in two pairs.
    PortReused {
        /// The reused port.
        port: RelationId,
        /// Index of the first pair naming it.
        first: usize,
        /// Index of the second pair naming it.
        second: usize,
    },
    /// Two pairs join the same two anchors, so the batch would add a
    /// duplicate bond.
    DuplicateBond {
        /// Index of the first pair on the anchor couple.
        first: usize,
        /// Index of the second pair on the anchor couple.
        second: usize,
    },
    /// A leaving group of one pair shares an atom with a leaving group of
    /// another pair, or holds another pair's anchor.
    BranchesOverlap {
        /// Index of the lower pair.
        first: usize,
        /// Index of the higher pair.
        second: usize,
    },
}

impl fmt::Display for LinkManyError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Pair { pair, source } => write!(f, "pair {pair} is refused: {source}"),
            Self::PortReused {
                port,
                first,
                second,
            } => write!(
                f,
                "port {} appears in pairs {first} and {second}",
                port.data().as_ffi()
            ),
            Self::DuplicateBond { first, second } => {
                write!(f, "pairs {first} and {second} join the same two anchors")
            }
            Self::BranchesOverlap { first, second } => write!(
                f,
                "the leaving groups of pair {first} and pair {second} overlap, or one \
                 holds the other pair's anchor"
            ),
        }
    }
}

impl std::error::Error for LinkManyError {
    fn source(&self) -> Option<&(dyn std::error::Error + 'static)> {
        match self {
            Self::Pair { source, .. } => Some(source),
            _ => None,
        }
    }
}

/// Everything one validated pair writes: both ports, both leaving groups
/// and the charge each leaving group folds onto its anchor (`None` when that
/// side carries no charge).
struct LinkPlan {
    a: Port,
    b: Port,
    branch_a: BTreeSet<NodeId>,
    branch_b: BTreeSet<NodeId>,
    fold_a: Option<f64>,
    fold_b: Option<f64>,
}

impl LinkPlan {
    /// The two `(anchor, folded charge)` sides, `a` first.
    fn folds(&self) -> [(NodeId, Option<f64>); 2] {
        [(self.a.anchor, self.fold_a), (self.b.anchor, self.fold_b)]
    }
}

impl MolGraph {
    /// Join port `a` to port `b` and return the new anchor–anchor bond.
    ///
    /// Both ports are consumed: on success the two anchors are bonded and
    /// neither port remains. `link` validates both ports before it writes
    /// anything, so a refusal leaves `self` unchanged. It refuses, in this
    /// order:
    ///
    /// 1. a port that does not read back → [`LinkError::Port`];
    /// 2. a port whose anchor–handle bond is gone → [`LinkError::StalePort`];
    /// 3. `!a.accepts(&b)` → [`LinkError::Incompatible`];
    /// 4. two ports on one anchor → [`LinkError::SameAnchor`];
    /// 5. anchors already bonded → [`LinkError::AlreadyBonded`];
    /// 6. a branch containing its own anchor → [`LinkError::BranchReachesAnchor`];
    /// 7. overlapping branches, or an anchor inside the other branch →
    ///    [`LinkError::BranchesOverlap`];
    /// 8. charge on only part of `{anchor} ∪ branch` →
    ///    [`LinkError::OneSidedCharge`] (an atom without `charge` counts as
    ///    absent; a side with no charge at all is fine and writes nothing).
    ///
    /// A port's **branch** is its [`leaving_group`](Self::leaving_group). The
    /// write folds each branch's charge (e) onto its anchor,
    /// `q_a' = q_a + Σ_{i∈D} q_i` with `D` the branch, removes both branches
    /// (their ports go with them) and bonds the two anchors, classed from the
    /// port order through
    /// [`BondNumber::implied_type`](crate::core::BondNumber::implied_type)
    /// and stamped on the new bond. Total charge
    /// is conserved; per-`frag_id` charge only under the labelling condition
    /// below.
    ///
    /// Several pairs are joined in one pass by the crate-internal batch
    /// `MolGraph::link_many`, which applies these checks to every pair.
    ///
    /// # Errors
    ///
    /// The [`LinkError`] of the first refusal above.
    ///
    /// # Panics
    ///
    /// A write that fails after validation passed is a broken invariant of this
    /// type (every atom written is live, the removed set is duplicate-free and
    /// leaves both anchors, and the new bond is fresh), so it panics rather than
    /// returning a half-applied edit as an error.
    ///
    /// # Examples
    ///
    /// Two one-H units joined through a `<` / `>` pair: each H's charge folds
    /// onto its own carbon.
    ///
    /// ```
    /// use molrs::core::keys;
    /// use molrs::core::BondNumber;
    /// use molrs::core::Atomistic;
    /// use molrs::core::PortKind;
    /// use molrs::props;
    /// use molrs::core::Element;
    ///
    /// let mut world = Atomistic::new();
    /// let mut unit = |q_c: f64, kind: PortKind| -> Result<_, molrs::core::MolRsError> {
    ///     let c = world.add_atom(props! { keys::ATOMIC_NUMBER => Element::C }).unwrap();
    ///     let h = world.add_atom(props! { keys::ATOMIC_NUMBER => Element::H }).unwrap();
    ///     world.set_atom(c, keys::CHARGE, q_c)?;
    ///     world.set_atom(h, keys::CHARGE, 0.125)?;
    ///     world.add_bond(c, h, props! {})?;
    ///     let port = world.add_port(c, h, kind, "p", BondNumber::Single)?;
    ///     Ok((c, port))
    /// };
    /// let (c0, p0) = unit(-0.25, PortKind::Left)?;
    /// let (c1, p1) = unit(-0.125, PortKind::Right)?;
    ///
    /// world.link(p0, p1)?;
    ///
    /// let q = |atom| world.get_atom(atom).map(|a| a.get_f64(keys::CHARGE));
    /// assert_eq!(q(c0)?, Some(-0.125));
    /// assert_eq!(q(c1)?, Some(0.0));
    /// assert_eq!(world.n_atoms(), 2);
    /// assert_eq!(world.n_bonds(), 1);
    /// assert_eq!(world.n_ports(), 0);
    /// # Ok::<(), Box<dyn std::error::Error>>(())
    /// ```
    ///
    /// Port joining: [`MolGraph::link`] joins two ports of one world graph into a
    /// bond.
    ///
    /// The vocabulary is defined in full on [`crate::core::Port`]. In short: a
    /// **port** is a marked, not-yet-used bonding site on a graph. It
    /// names two bonded atoms: the **anchor** `a`, which gains the new bond, and
    /// the **handle** `h`, a real atom (usually a capping hydrogen) standing where
    /// the partner will go. The **world** is the one fragment that holds every unit
    /// being joined; a caller first [`merge`](MolGraph::merge)s each unit into it.
    /// An atom may carry a partial charge `q` (the `charge` property, in units of
    /// the elementary charge e) and a `frag_id`, the index of the unit it came
    /// from.
    ///
    /// A port `p = (a, h)` joins its anchor `a` to a partner's anchor. Its leaving
    /// group `D_p` is the handle's connected component over bonds once the `a`–`h`
    /// bond is cut ([`MolGraph::leaving_group`]). Linking `p` (anchor `a`) with `r`
    /// (anchor `b`) pairs the two only through
    /// [`Port::accepts`](crate::core::Port::accepts), removes
    /// `D_p ∪ D_r` (their ports go with them), folds each leaving group's partial
    /// charge (e) onto its own anchor,
    ///
    /// ```text
    /// q_a' = q_a + Σ_{i ∈ D_p} q_i        q_b' = q_b + Σ_{i ∈ D_r} q_i
    /// ```
    ///
    /// and bonds the two anchors with the port order.
    ///
    /// **Conservation.** The refusals guarantee `D_p ∩ D_r = ∅` and that both
    /// anchors lie outside both branches, so the total charge `Σ_{i ∈ V} q_i` over
    /// the world's atom set `V` is conserved: every removed atom's charge reappears
    /// on exactly one surviving anchor (exactly in real arithmetic; to rounding in
    /// `f64`). The sum within one
    /// `frag_id` unit is conserved when every atom of `D_p` carries `frag_id(a)`
    /// and every atom of `D_r` carries `frag_id(b)` (a sufficient condition);
    /// `link` does not check that labelling.
    ///
    /// Putting the whole leaving-group charge on the one anchor is the practice of
    /// pysimm's `random_walk` polymer builder (Fortunato & Colina, *SoftwareX* **6**,
    /// 7 (2017), doi:10.1016/j.softx.2016.12.002); AMBER's `prepgen` spreads it
    /// instead, which molrs does not offer.
    pub fn link(&mut self, a: RelationId, b: RelationId) -> Result<RelationId, LinkError> {
        let plan = self.plan_link(a, b)?;

        const VALIDATED: &str = "validated before the first write";
        for (anchor, fold) in plan.folds() {
            self.fold_charge(anchor, fold);
        }
        let doomed: Vec<NodeId> = plan
            .branch_a
            .iter()
            .chain(&plan.branch_b)
            .copied()
            .collect();
        self.del_nodes(&doomed).expect(VALIDATED);
        Ok(self.bond_anchors(&plan))
    }

    /// Join every pair of `pairs` in one pass and return the new anchor–anchor
    /// bonds in pair order.
    ///
    /// Each pair `(p, r)` is joined by the rule of [`link`](Self::link): the
    /// two leaving groups are removed, each one's charge (e) folds onto its
    /// own anchor, and the anchors are bonded, classed from the port order.
    /// An anchor named by several pairs receives the folds in pair order,
    /// `q_a' = q_a + Σ_k Σ_{i∈D_k} q_i`.
    ///
    /// # Refusals
    ///
    /// Every check runs before the first write, so a refusal leaves `self`
    /// unchanged. In this order:
    ///
    /// 1. each pair alone, in pair order, by every check of
    ///    [`link`](Self::link) → [`LinkManyError::Pair`];
    /// 2. a port named by two pairs → [`LinkManyError::PortReused`];
    /// 3. two pairs on one unordered anchor couple →
    ///    [`LinkManyError::DuplicateBond`];
    /// 4. a leaving group of one pair meeting a leaving group or an anchor of
    ///    another pair → [`LinkManyError::BranchesOverlap`].
    ///
    /// An empty slice is `Ok(vec![])`.
    ///
    /// # Equivalence with `link`
    ///
    /// If `link_many` succeeds, looping [`link`](Self::link) over the same
    /// pairs in pair order gives the same world: the same atoms, bonds, bond
    /// classes, bitwise charges and remaining ports. The converse does not
    /// hold: check 4 refuses some batches a loop would accept, e.g. when pair
    /// `k`'s leaving group holds pair `j`'s anchor (`j < k`), which the loop
    /// bonds first and then removes.
    ///
    /// # Cost
    ///
    /// O(E + Σ|D| + pairs), with E the relation count and Σ|D| the total
    /// leaving-group size: checks 2–4 are hash lookups and the removal is one
    /// [`del_nodes`](crate::core::MolGraph::del_nodes) call,
    /// which scans every relation once. A loop over `link` costs
    /// O(pairs · E).
    ///
    /// # Row order
    ///
    /// `del_nodes` swap-removes rows, so the surviving atoms' row order
    /// (the order [`to_frame`](Self::to_frame) writes) is not their order
    /// before the call, and not the order a loop over `link` leaves either.
    /// Handles are unaffected.
    ///
    /// # Panics
    ///
    /// As [`link`](Self::link): a write that fails after validation is a
    /// broken invariant of this type.
    #[cfg_attr(
        not(any(test, feature = "builder")),
        expect(
            dead_code,
            reason = "its only caller, builder::Assembler, compiles with the builder feature"
        )
    )]
    pub(crate) fn link_many(
        &mut self,
        pairs: &[(RelationId, RelationId)],
    ) -> Result<Vec<RelationId>, LinkManyError> {
        // ---- 1. each pair alone ----
        let plans = pairs
            .iter()
            .enumerate()
            .map(|(pair, &(a, b))| {
                self.plan_link(a, b)
                    .map_err(|source| LinkManyError::Pair { pair, source })
            })
            .collect::<Result<Vec<LinkPlan>, LinkManyError>>()?;

        // ---- 2. a port in two pairs ----
        let mut port_owner: HashMap<RelationId, usize> = HashMap::with_capacity(2 * pairs.len());
        for (second, &(a, b)) in pairs.iter().enumerate() {
            for port in [a, b] {
                if let Some(&first) = port_owner.get(&port) {
                    return Err(LinkManyError::PortReused {
                        port,
                        first,
                        second,
                    });
                }
                port_owner.insert(port, second);
            }
        }

        // ---- 3. two pairs on one anchor couple ----
        let mut couple_owner: HashMap<(NodeId, NodeId), usize> =
            HashMap::with_capacity(plans.len());
        for (second, plan) in plans.iter().enumerate() {
            let (x, y) = (plan.a.anchor, plan.b.anchor);
            match couple_owner.entry((x.min(y), x.max(y))) {
                Entry::Occupied(owner) => {
                    return Err(LinkManyError::DuplicateBond {
                        first: *owner.get(),
                        second,
                    });
                }
                Entry::Vacant(slot) => {
                    slot.insert(second);
                }
            }
        }

        // ---- 4. leaving groups against other pairs' leaving groups and anchors ----
        let overlap = |j: usize, k: usize| LinkManyError::BranchesOverlap {
            first: j.min(k),
            second: j.max(k),
        };
        // `doomed` lists the removal set in pair order, so the rows
        // `del_nodes` swaps are the same on every run.
        let mut doomed: Vec<NodeId> = Vec::new();
        let mut doomed_owner: HashMap<NodeId, usize> = HashMap::new();
        for (k, plan) in plans.iter().enumerate() {
            for &atom in plan.branch_a.iter().chain(&plan.branch_b) {
                // `plan_link` keeps a pair's own two groups disjoint, so an
                // owner is always another pair.
                if let Some(&j) = doomed_owner.get(&atom) {
                    return Err(overlap(j, k));
                }
                doomed_owner.insert(atom, k);
                doomed.push(atom);
            }
        }
        for (k, plan) in plans.iter().enumerate() {
            for anchor in [plan.a.anchor, plan.b.anchor] {
                if let Some(&j) = doomed_owner.get(&anchor) {
                    return Err(overlap(j, k));
                }
            }
        }

        // ---- write ----
        const VALIDATED: &str = "validated before the first write";
        for plan in &plans {
            for (anchor, fold) in plan.folds() {
                self.fold_charge(anchor, fold);
            }
        }
        self.del_nodes(&doomed).expect(VALIDATED);
        Ok(plans.iter().map(|plan| self.bond_anchors(plan)).collect())
    }

    /// Validate joining port `a` to port `b` without writing anything, in the
    /// refusal order [`link`](Self::link) documents.
    fn plan_link(&self, a: RelationId, b: RelationId) -> Result<LinkPlan, LinkError> {
        let pa = self.get_port(a).map_err(LinkError::Port)?;
        let pb = self.get_port(b).map_err(LinkError::Port)?;
        // `leaving_group`'s only error is a missing anchor–handle bond.
        let branch_a = self
            .leaving_group(&pa)
            .map_err(|_| LinkError::StalePort { port: a })?;
        let branch_b = self
            .leaving_group(&pb)
            .map_err(|_| LinkError::StalePort { port: b })?;
        if !pa.accepts(&pb) {
            return Err(LinkError::Incompatible { a, b });
        }
        if pa.anchor == pb.anchor {
            return Err(LinkError::SameAnchor { a, b });
        }
        if self.is_bonded(pa.anchor, pb.anchor) {
            return Err(LinkError::AlreadyBonded {
                a: pa.anchor,
                b: pb.anchor,
            });
        }
        if branch_a.contains(&pa.anchor) {
            return Err(LinkError::BranchReachesAnchor { port: a });
        }
        if branch_b.contains(&pb.anchor) {
            return Err(LinkError::BranchReachesAnchor { port: b });
        }
        if !branch_a.is_disjoint(&branch_b)
            || branch_a.contains(&pb.anchor)
            || branch_b.contains(&pa.anchor)
        {
            return Err(LinkError::BranchesOverlap);
        }
        let fold_a = self.folded_charge(&pa, &branch_a)?;
        let fold_b = self.folded_charge(&pb, &branch_b)?;
        Ok(LinkPlan {
            a: pa,
            b: pb,
            branch_a,
            branch_b,
            fold_a,
            fold_b,
        })
    }

    /// The charge `branch` folds onto the port's anchor, `Σ_{i∈branch} q_i`
    /// (e): `None` when no atom of `{anchor} ∪ branch` carries a charge,
    /// `Some` when every atom does.
    fn folded_charge(
        &self,
        port: &Port,
        branch: &BTreeSet<NodeId>,
    ) -> Result<Option<f64>, LinkError> {
        let charge = |atom: NodeId| -> Result<Option<f64>, LinkError> {
            self.get_node(atom)
                .map(|props| props.get_f64(keys::CHARGE))
                .map_err(LinkError::Graph)
        };
        let anchor_q = charge(port.anchor)?;
        let mut present = usize::from(anchor_q.is_some());
        let mut branch_q = 0.0;
        for &atom in branch {
            if let Some(q) = charge(atom)? {
                present += 1;
                branch_q += q;
            }
        }
        match anchor_q {
            _ if present == 0 => Ok(None),
            Some(_) if present == branch.len() + 1 => Ok(Some(branch_q)),
            _ => Err(LinkError::OneSidedCharge {
                anchor: port.anchor,
            }),
        }
    }

    /// Add a validated fold to `anchor`'s current charge, `q_a' = q_a + fold`;
    /// `None` writes nothing.
    fn fold_charge(&mut self, anchor: NodeId, fold: Option<f64>) {
        const VALIDATED: &str = "validated before the first write";
        if let Some(fold) = fold {
            let q = self
                .node_table()
                .get::<f64>(anchor, keys::CHARGE)
                .expect(VALIDATED);
            self.set_node(anchor, keys::CHARGE, q + fold)
                .expect(VALIDATED);
        }
    }

    /// Bond a validated plan's two anchors, classed from the port order.
    fn bond_anchors(&mut self, plan: &LinkPlan) -> RelationId {
        const VALIDATED: &str = "validated before the first write";
        self.add_classed_bond(
            plan.a.anchor,
            plan.b.anchor,
            plan.a.order.implied_type(),
            plan.a.order,
        )
        .expect(VALIDATED)
    }
}

#[cfg(test)]
mod tests {
    use crate::core::Element;
    use crate::props;
    use std::collections::BTreeMap;

    use super::{LinkError, LinkManyError};
    use crate::core::Atomistic;
    use crate::core::BondNumber;
    use crate::core::GraphAccess;
    use crate::core::MolRsError;
    use crate::core::NodeId;
    use crate::core::PropValue;
    use crate::core::RelationId;
    use crate::core::keys;
    use crate::core::{Port, PortKind};

    // ---- fixtures ----------------------------------------------------------
    //
    // Every fixture is one hand-built world `Atomistic` of 2–5 atoms per unit.
    // Charges are dyadic rationals (k / 2^n), so every sum below is exact in
    // binary floating point and the goldens are compared with `==`.

    /// Add an atom of `element` carrying `charge` and (when given) `frag_id`.
    fn atom(world: &mut Atomistic, element: Element, charge: Option<f64>, frag: i32) -> NodeId {
        let a = world
            .add_atom(props! { keys::ATOMIC_NUMBER => element })
            .expect("fixture atom has a species");
        if let Some(q) = charge {
            world
                .set_atom(a, keys::CHARGE, q)
                .expect("fixture atom takes a charge");
        }
        world
            .set_atom(a, keys::FRAG_ID, PropValue::Int(frag))
            .expect("fixture frag_id fits");
        a
    }

    /// One unit: an `anchor` atom bonded to an H handle, with a port
    /// `(anchor, H)` of `kind`, label `"p"`, order `Single`.
    /// Returns `(anchor, handle, port)`.
    fn h_unit(
        world: &mut Atomistic,
        anchor_element: Element,
        anchor_q: Option<f64>,
        handle_q: Option<f64>,
        kind: PortKind,
        frag: i32,
    ) -> (NodeId, NodeId, RelationId) {
        let anchor = atom(world, anchor_element, anchor_q, frag);
        let handle = atom(world, Element::H, handle_q, frag);
        world
            .add_bond(anchor, handle, props! {})
            .expect("fixture bond");
        let port = world
            .add_port(anchor, handle, kind, "p", BondNumber::Single)
            .expect("a bonded H handle is a legal port");
        (anchor, handle, port)
    }

    /// The `frag_id` of atom `a`, if labelled.
    fn frag_of(world: &Atomistic, a: NodeId) -> Option<i32> {
        world.get_atom(a).expect("live atom").get_int(keys::FRAG_ID)
    }

    fn charge(world: &Atomistic, a: NodeId) -> f64 {
        world
            .get_atom(a)
            .expect("live atom")
            .get_f64(keys::CHARGE)
            .expect("atom carries a charge")
    }

    fn total_charge(world: &Atomistic) -> f64 {
        world
            .atoms()
            .map(|(_, atom)| atom.get_f64(keys::CHARGE).unwrap_or(0.0))
            .sum()
    }

    /// Per-`frag_id` charge sums (atoms without a charge count as 0).
    fn frag_sums(world: &Atomistic) -> BTreeMap<i32, f64> {
        let mut sums = BTreeMap::new();
        for id in world.atom_ids().collect::<Vec<_>>() {
            let frag = frag_of(world, id).expect("fixture atoms carry frag_id");
            let q = world
                .get_atom(id)
                .expect("live atom")
                .get_f64(keys::CHARGE)
                .unwrap_or(0.0);
            *sums.entry(frag).or_insert(0.0) += q;
        }
        sums
    }

    fn frag_count(world: &Atomistic, frag: i32) -> usize {
        world
            .atom_ids()
            .filter(|&id| frag_of(world, id) == Some(frag))
            .count()
    }

    fn is_bonded(world: &Atomistic, a: NodeId, b: NodeId) -> bool {
        world.bonds().any(|(_, bond)| {
            let n = bond.nodes.as_slice();
            (n[0] == a && n[1] == b) || (n[0] == b && n[1] == a)
        })
    }

    /// Everything a refused link must leave untouched.
    #[derive(Debug, PartialEq)]
    struct Snapshot {
        atoms: Vec<(NodeId, Option<f64>)>,
        bonds: Vec<Vec<NodeId>>,
        /// A port that does not read back is recorded as `None`.
        ports: Vec<(RelationId, Option<Port>)>,
    }

    fn snapshot(world: &Atomistic) -> Snapshot {
        Snapshot {
            atoms: world
                .atoms()
                .map(|(id, atom)| (id, atom.get_f64(keys::CHARGE)))
                .collect(),
            bonds: world.bonds().map(|(_, bond)| bond.nodes.to_vec()).collect(),
            ports: world
                .port_ids()
                .map(|id| (id, world.get_port(id).ok()))
                .collect(),
        }
    }

    /// Run a link that must be refused, assert the world is unchanged, and
    /// return the refusal.
    fn refuse(world: &mut Atomistic, a: RelationId, b: RelationId) -> LinkError {
        let before = snapshot(world);
        let err = world.link(a, b).expect_err("this pair must be refused");
        assert_eq!(
            snapshot(world),
            before,
            "a refusal leaves atoms, bonds, charges and ports unchanged ({err:?})"
        );
        err
    }

    // ---- charge-folding goldens C1–C5 ---------------------------------------

    #[test]
    fn c1_single_h_handle_folds_onto_anchor() {
        let mut world = Atomistic::new();
        let (a0, h0, p0) = h_unit(
            &mut world,
            Element::C,
            Some(-0.25),
            Some(0.125),
            PortKind::Left,
            0,
        );
        let (a1, h1, p1) = h_unit(
            &mut world,
            Element::C,
            Some(-0.125),
            Some(0.125),
            PortKind::Right,
            1,
        );

        world.link(p0, p1).expect("complementary ports link");

        assert_eq!(charge(&world, a0), -0.125, "C1: -0.25 + 0.125");
        assert_eq!(charge(&world, a1), 0.0, "partner: -0.125 + 0.125");
        assert!(world.get_atom(h0).is_err(), "the handle H is removed");
        assert!(
            world.get_atom(h1).is_err(),
            "the partner handle H is removed"
        );
        assert_eq!(world.n_atoms(), 2);
        assert_eq!(world.n_ports(), 0, "both ports go with their handles");
        assert_eq!(world.n_bonds(), 1);
        assert!(is_bonded(&world, a0, a1), "the anchors are bonded");
    }

    /// Regression example (spec assembly-06, golden C2; hard-coded golden:
    /// C +0.5, O −0.625, H +0.375 → C +0.25 with the 2-atom OH branch
    /// removed). The handle is the heavy O, the branch is {O, H}.
    #[test]
    fn c2_oh_branch_folds_onto_carbon() {
        let mut world = Atomistic::new();
        let c = atom(&mut world, Element::C, Some(0.5), 0);
        let o = atom(&mut world, Element::O, Some(-0.625), 0);
        let h = atom(&mut world, Element::H, Some(0.375), 0);
        world.add_bond(c, o, props! {}).unwrap();
        world.add_bond(o, h, props! {}).unwrap();
        let p0 = world
            .add_port(c, o, PortKind::Left, "p", BondNumber::Single)
            .expect("a bonded O leaving group is a legal handle");
        let (n, nh, p1) = h_unit(
            &mut world,
            Element::N,
            Some(-0.125),
            Some(0.125),
            PortKind::Right,
            1,
        );

        world.link(p0, p1).expect("complementary ports link");

        assert_eq!(charge(&world, c), 0.25, "C2: 0.5 - 0.625 + 0.375");
        assert_eq!(
            frag_count(&world, 0),
            1,
            "the OH branch's 2 atoms are removed"
        );
        assert!(world.get_atom(o).is_err(), "O is removed");
        assert!(world.get_atom(h).is_err(), "the hydroxyl H is removed");
        assert!(world.get_atom(nh).is_err(), "the partner handle is removed");
        assert!(is_bonded(&world, c, n), "C is bonded to the partner anchor");
    }

    #[test]
    fn c3_two_h_handles_both_linked() {
        let mut world = Atomistic::new();
        let a = atom(&mut world, Element::C, Some(-0.5), 0);
        let h1 = atom(&mut world, Element::H, Some(0.125), 0);
        let h2 = atom(&mut world, Element::H, Some(0.125), 0);
        world.add_bond(a, h1, props! {}).unwrap();
        world.add_bond(a, h2, props! {}).unwrap();
        let pa1 = world
            .add_port(a, h1, PortKind::Left, "p", BondNumber::Single)
            .unwrap();
        let pa2 = world
            .add_port(a, h2, PortKind::Left, "p", BondNumber::Single)
            .unwrap();
        let (b1, _, pb1) = h_unit(
            &mut world,
            Element::C,
            Some(-0.125),
            Some(0.125),
            PortKind::Right,
            1,
        );
        let (b2, _, pb2) = h_unit(
            &mut world,
            Element::C,
            Some(-0.125),
            Some(0.125),
            PortKind::Right,
            2,
        );

        let first = world.link(pa1, pb1).expect("the first pair links");
        let second = world
            .link(pa2, pb2)
            .expect("the second port on the same anchor still links");

        assert_ne!(first, second, "one new bond per link");
        assert_eq!(charge(&world, a), -0.25, "C3: -0.5 + 0.125 + 0.125");
        assert_eq!(world.n_bonds(), 2);
        assert!(is_bonded(&world, a, b1));
        assert!(is_bonded(&world, a, b2));
        assert_eq!(world.n_ports(), 0);
        assert_eq!(world.n_atoms(), 3);
    }

    #[test]
    fn c4_neutral_units_stay_neutral_per_frag_id() {
        let mut world = Atomistic::new();
        // unit 0: C(-0.25) – C anchor(+0.125) – H handle(+0.125); sum 0
        let c0 = atom(&mut world, Element::C, Some(-0.25), 0);
        let (a0, _, p0) = h_unit(
            &mut world,
            Element::C,
            Some(0.125),
            Some(0.125),
            PortKind::Left,
            0,
        );
        world.add_bond(c0, a0, props! {}).unwrap();
        // unit 1: O anchor(-0.5) – H handle(+0.25), plus H(+0.25) on O; sum 0
        let (a1, _, p1) = h_unit(
            &mut world,
            Element::O,
            Some(-0.5),
            Some(0.25),
            PortKind::Right,
            1,
        );
        let h = atom(&mut world, Element::H, Some(0.25), 1);
        world.add_bond(a1, h, props! {}).unwrap();

        world.link(p0, p1).expect("link");

        assert_eq!(total_charge(&world), 0.0, "C4: total stays 0");
        let sums = frag_sums(&world);
        assert_eq!(sums.get(&0), Some(&0.0), "frag 0 sums to 0");
        assert_eq!(sums.get(&1), Some(&0.0), "frag 1 sums to 0");
    }

    #[test]
    fn c5_integer_charged_units_keep_minus_one_per_frag_id() {
        let mut world = Atomistic::new();
        // unit 0: O anchor(-1.25) + H handle(+0.25); sum -1
        let (_, _, p0) = h_unit(
            &mut world,
            Element::O,
            Some(-1.25),
            Some(0.25),
            PortKind::Left,
            0,
        );
        // unit 1: C anchor(-0.5) + O(-0.625) + H handle(+0.125); sum -1
        let (a1, _, p1) = h_unit(
            &mut world,
            Element::C,
            Some(-0.5),
            Some(0.125),
            PortKind::Right,
            1,
        );
        let o = atom(&mut world, Element::O, Some(-0.625), 1);
        world.add_bond(a1, o, props! {}).unwrap();

        world.link(p0, p1).expect("link");

        let sums = frag_sums(&world);
        assert_eq!(sums.get(&0), Some(&-1.0), "C5: frag 0 sums to -1");
        assert_eq!(sums.get(&1), Some(&-1.0), "C5: frag 1 sums to -1");
        assert_eq!(total_charge(&world), -2.0);
    }

    // ---- the new bond --------------------------------------------------------

    #[test]
    fn link_returns_the_anchor_bond_classed_from_the_port_order() {
        let mut world = Atomistic::new();
        let a = atom(&mut world, Element::C, None, 0);
        let ha = atom(&mut world, Element::H, None, 0);
        let b = atom(&mut world, Element::C, None, 1);
        let hb = atom(&mut world, Element::H, None, 1);
        world.add_bond(a, ha, props! {}).unwrap();
        world.add_bond(b, hb, props! {}).unwrap();
        let pa = world
            .add_port(a, ha, PortKind::Left, "p", BondNumber::Double)
            .unwrap();
        let pb = world
            .add_port(b, hb, PortKind::Right, "p", BondNumber::Double)
            .unwrap();

        let bid = world.link(pa, pb).expect("link");

        let bond = world.get_bond(bid).expect("the returned bond is live");
        let mut ends = bond.nodes.to_vec();
        ends.sort();
        let mut expected = vec![a, b];
        expected.sort();
        assert_eq!(ends, expected, "the returned bond joins the two anchors");
        assert_eq!(
            BondNumber::from_prop(bond.props.get(keys::BOND_NUMBER)),
            BondNumber::Double,
            "the bond order comes from the port order"
        );
    }

    // ---- refusals (each leaves the world unchanged) ---------------------------

    #[test]
    fn same_kind_ports_are_incompatible() {
        let mut world = Atomistic::new();
        let (_, _, p0) = h_unit(
            &mut world,
            Element::C,
            Some(-0.25),
            Some(0.25),
            PortKind::Left,
            0,
        );
        let (_, _, p1) = h_unit(
            &mut world,
            Element::C,
            Some(-0.25),
            Some(0.25),
            PortKind::Left,
            1,
        );
        assert!(
            !world
                .get_port(p0)
                .unwrap()
                .accepts(&world.get_port(p1).unwrap()),
            "fixture: Port::accepts refuses `<`/`<`"
        );

        let err = refuse(&mut world, p0, p1);

        assert!(
            matches!(err, LinkError::Incompatible { a, b } if a == p0 && b == p1),
            "{err:?}"
        );
    }

    #[test]
    fn label_mismatch_is_incompatible_through_port_accepts() {
        let mut world = Atomistic::new();
        let (_, _, p0) = h_unit(&mut world, Element::C, None, None, PortKind::Left, 0);
        let b = atom(&mut world, Element::C, None, 1);
        let hb = atom(&mut world, Element::H, None, 1);
        world.add_bond(b, hb, props! {}).unwrap();
        let p1 = world
            .add_port(b, hb, PortKind::Right, "q", BondNumber::Single)
            .unwrap();
        assert!(
            !world
                .get_port(p0)
                .unwrap()
                .accepts(&world.get_port(p1).unwrap()),
            "fixture: Port::accepts refuses unequal labels"
        );

        let err = refuse(&mut world, p0, p1);

        assert!(matches!(err, LinkError::Incompatible { .. }), "{err:?}");
    }

    #[test]
    fn ring_bonded_handle_reaches_its_anchor() {
        let mut world = Atomistic::new();
        // Triangle C0–C1–C2–C0; the port handle C1 sits on the ring, so its
        // branch after cutting C0–C1 still reaches C0.
        let c0 = atom(&mut world, Element::C, None, 0);
        let c1 = atom(&mut world, Element::C, None, 0);
        let c2 = atom(&mut world, Element::C, None, 0);
        world.add_bond(c0, c1, props! {}).unwrap();
        world.add_bond(c1, c2, props! {}).unwrap();
        world.add_bond(c2, c0, props! {}).unwrap();
        let p0 = world
            .add_port(c0, c1, PortKind::Left, "p", BondNumber::Single)
            .expect("a bonded heavy handle is a legal port");
        let (_, _, p1) = h_unit(&mut world, Element::C, None, None, PortKind::Right, 1);

        let err = refuse(&mut world, p0, p1);

        assert!(
            matches!(err, LinkError::BranchReachesAnchor { port } if port == p0),
            "{err:?}"
        );
    }

    #[test]
    fn shared_handle_branches_overlap() {
        let mut world = Atomistic::new();
        // A – X – B, with X the handle of both ports.
        let a = atom(&mut world, Element::C, None, 0);
        let x = atom(&mut world, Element::O, None, 0);
        let b = atom(&mut world, Element::C, None, 1);
        world.add_bond(a, x, props! {}).unwrap();
        world.add_bond(x, b, props! {}).unwrap();
        let pa = world
            .add_port(a, x, PortKind::Left, "p", BondNumber::Single)
            .unwrap();
        let pb = world
            .add_port(b, x, PortKind::Right, "p", BondNumber::Single)
            .unwrap();

        let err = refuse(&mut world, pa, pb);

        assert!(matches!(err, LinkError::BranchesOverlap), "{err:?}");
    }

    #[test]
    fn anchor_inside_the_other_branch_is_overlap() {
        let mut world = Atomistic::new();
        // H1 – A – Y – B: port a = (A, H1), port b = (B, Y). b's branch
        // {Y, A, H1} holds anchor A.
        let h1 = atom(&mut world, Element::H, None, 0);
        let a = atom(&mut world, Element::C, None, 0);
        let y = atom(&mut world, Element::C, None, 1);
        let b = atom(&mut world, Element::C, None, 1);
        world.add_bond(h1, a, props! {}).unwrap();
        world.add_bond(a, y, props! {}).unwrap();
        world.add_bond(y, b, props! {}).unwrap();
        let pa = world
            .add_port(a, h1, PortKind::Left, "p", BondNumber::Single)
            .unwrap();
        let pb = world
            .add_port(b, y, PortKind::Right, "p", BondNumber::Single)
            .unwrap();

        let err = refuse(&mut world, pa, pb);

        assert!(matches!(err, LinkError::BranchesOverlap), "{err:?}");
    }

    #[test]
    fn charged_anchor_with_uncharged_handle_is_one_sided() {
        let mut world = Atomistic::new();
        let (a0, _, p0) = h_unit(&mut world, Element::C, Some(-0.25), None, PortKind::Left, 0);
        let (_, _, p1) = h_unit(
            &mut world,
            Element::C,
            Some(-0.125),
            Some(0.125),
            PortKind::Right,
            1,
        );

        let err = refuse(&mut world, p0, p1);

        assert!(
            matches!(err, LinkError::OneSidedCharge { anchor } if anchor == a0),
            "{err:?}"
        );
    }

    #[test]
    fn charged_handle_with_uncharged_anchor_is_one_sided() {
        let mut world = Atomistic::new();
        let (_, _, p0) = h_unit(
            &mut world,
            Element::C,
            Some(-0.125),
            Some(0.125),
            PortKind::Left,
            0,
        );
        let (a1, _, p1) = h_unit(
            &mut world,
            Element::C,
            None,
            Some(0.125),
            PortKind::Right,
            1,
        );

        let err = refuse(&mut world, p0, p1);

        assert!(
            matches!(err, LinkError::OneSidedCharge { anchor } if anchor == a1),
            "{err:?}"
        );
    }

    /// Two compatible ports whose anchors are already
    /// bonded would otherwise gain a duplicate anchor–anchor bond.
    #[test]
    fn link_refuses_anchors_that_are_already_bonded() {
        let mut world = Atomistic::new();
        let (a0, _, p0) = h_unit(&mut world, Element::C, None, None, PortKind::Left, 0);
        let (a1, _, p1) = h_unit(&mut world, Element::C, None, None, PortKind::Right, 1);
        world
            .add_bond(a0, a1, props! {})
            .expect("fixture: the anchors are bonded");
        assert!(
            world
                .get_port(p0)
                .unwrap()
                .accepts(&world.get_port(p1).unwrap()),
            "fixture: the ports themselves are compatible"
        );

        let err = refuse(&mut world, p0, p1);

        assert!(
            matches!(err, LinkError::AlreadyBonded { a, b } if a == a0 && b == a1),
            "{err:?}"
        );
    }

    /// Two compatible ports on one anchor are their own
    /// refusal, not a `Graph(Validation)`.
    #[test]
    fn link_refuses_two_ports_on_one_anchor() {
        let mut world = Atomistic::new();
        let c = atom(&mut world, Element::C, None, 0);
        let h1 = atom(&mut world, Element::H, None, 0);
        let h2 = atom(&mut world, Element::H, None, 0);
        world.add_bond(c, h1, props! {}).unwrap();
        world.add_bond(c, h2, props! {}).unwrap();
        let p0 = world
            .add_port(c, h1, PortKind::Left, "p", BondNumber::Single)
            .unwrap();
        let p1 = world
            .add_port(c, h2, PortKind::Right, "p", BondNumber::Single)
            .unwrap();
        assert!(
            world
                .get_port(p0)
                .unwrap()
                .accepts(&world.get_port(p1).unwrap()),
            "fixture: the ports themselves are compatible"
        );

        let err = refuse(&mut world, p0, p1);

        assert!(
            matches!(err, LinkError::SameAnchor { a, b } if a == p0 && b == p1),
            "{err:?}"
        );
    }

    /// The anchor–handle bond is checked only at
    /// `add_port`; deleting it later leaves a port that
    /// still reads back but is stale.
    #[test]
    fn link_refuses_a_port_whose_handle_bond_is_gone() {
        let mut world = Atomistic::new();
        let (a0, h0, p0) = h_unit(&mut world, Element::C, None, None, PortKind::Left, 0);
        let (_, _, p1) = h_unit(&mut world, Element::C, None, None, PortKind::Right, 1);
        let cut = world
            .bond_between(a0, h0)
            .expect("fixture: the anchor–handle bond exists");
        world.del_bond(cut).expect("the bond is deleted");
        assert!(
            world.get_port(p0).is_ok(),
            "fixture: the stale port still reads back"
        );

        let err = refuse(&mut world, p0, p1);

        assert!(
            matches!(err, LinkError::StalePort { port } if port == p0),
            "{err:?}"
        );
    }

    /// Refusal step 1: a port whose kind glyph was overwritten through
    /// the raw graph does not read back through `MolGraph::get_port`.
    #[test]
    fn link_refuses_a_port_that_does_not_read_back() {
        let mut world = Atomistic::new();
        let (_, _, p0) = h_unit(&mut world, Element::C, None, None, PortKind::Left, 0);
        let (_, _, p1) = h_unit(&mut world, Element::C, None, None, PortKind::Right, 1);
        let ports = world.graph().kind_id("ports").expect("'ports' registered");
        world
            .graph_mut()
            .set_relation_prop(ports, p0, "port_kind", "Z")
            .expect("the inner graph overwrites the glyph");
        assert!(
            world.get_port(p0).is_err(),
            "fixture: the corrupted port does not read back"
        );

        let err = refuse(&mut world, p0, p1);

        assert!(
            matches!(err, LinkError::Port(MolRsError::Validation { .. })),
            "{err:?}"
        );
    }

    // ---- link_many -----------------------------------------------------------

    /// One chain unit: a C anchor (q −0.25) carrying a `<` port on one H
    /// handle and a `>` port on another (q +0.125 each).
    struct ChainUnit {
        anchor: NodeId,
        left_h: NodeId,
        left: RelationId,
        right_h: NodeId,
        right: RelationId,
    }

    fn chain(world: &mut Atomistic, n: i32) -> Vec<ChainUnit> {
        (0..n)
            .map(|frag| {
                let anchor = atom(world, Element::C, Some(-0.25), frag);
                let left_h = atom(world, Element::H, Some(0.125), frag);
                let right_h = atom(world, Element::H, Some(0.125), frag);
                world.add_bond(anchor, left_h, props! {}).unwrap();
                world.add_bond(anchor, right_h, props! {}).unwrap();
                let left = world
                    .add_port(anchor, left_h, PortKind::Left, "p", BondNumber::Single)
                    .unwrap();
                let right = world
                    .add_port(anchor, right_h, PortKind::Right, "p", BondNumber::Single)
                    .unwrap();
                ChainUnit {
                    anchor,
                    left_h,
                    left,
                    right_h,
                    right,
                }
            })
            .collect()
    }

    /// Head-to-tail pairs: unit `i`'s `>` port with unit `i + 1`'s `<` port.
    fn chain_pairs(units: &[ChainUnit]) -> Vec<(RelationId, RelationId)> {
        units.windows(2).map(|w| (w[0].right, w[1].left)).collect()
    }

    /// Row-order-free world state: atoms by handle with their charge bits,
    /// bonds as sorted endpoint pairs with their property bag, ports by
    /// handle. Two worlds reached by different edit orders compare equal
    /// here even though `del_nodes` swap-removes rows.
    /// A bond as sorted endpoints plus its sorted property bag.
    type BondState = (NodeId, NodeId, Vec<(String, PropValue)>);

    #[derive(Debug, PartialEq)]
    struct WorldState {
        atoms: Vec<(NodeId, Option<u64>)>,
        bonds: Vec<BondState>,
        ports: Vec<(RelationId, Option<Port>)>,
    }

    fn world_state(world: &Atomistic) -> WorldState {
        let mut atoms: Vec<_> = world
            .atoms()
            .map(|(id, atom)| (id, atom.get_f64(keys::CHARGE).map(f64::to_bits)))
            .collect();
        atoms.sort_by_key(|&(id, _)| id);
        let mut bonds: Vec<_> = world
            .bonds()
            .map(|(_, bond)| {
                let (x, y) = (bond.nodes[0], bond.nodes[1]);
                let mut props: Vec<(String, PropValue)> = bond
                    .props
                    .iter()
                    .map(|(k, v)| (k.to_owned(), v.clone()))
                    .collect();
                props.sort_by(|p, q| p.0.cmp(&q.0));
                (x.min(y), x.max(y), props)
            })
            .collect();
        bonds.sort_by_key(|&(x, y, _)| (x, y));
        let mut ports: Vec<_> = world
            .port_ids()
            .map(|id| (id, world.get_port(id).ok()))
            .collect();
        ports.sort_by_key(|&(id, _)| id);
        WorldState {
            atoms,
            bonds,
            ports,
        }
    }

    /// Run a batch that must be refused, assert the world is unchanged (row
    /// order included), and return the refusal.
    fn refuse_many(world: &mut Atomistic, pairs: &[(RelationId, RelationId)]) -> LinkManyError {
        let before = snapshot(world);
        let err = world
            .graph_mut()
            .link_many(pairs)
            .expect_err("this batch must be refused");
        assert_eq!(
            snapshot(world),
            before,
            "a refusal leaves atoms, bonds, charges and ports unchanged ({err:?})"
        );
        err
    }

    #[test]
    fn link_many_equals_a_loop_of_link_on_a_chain() {
        let mut world = Atomistic::new();
        let units = chain(&mut world, 4);
        let pairs = chain_pairs(&units);
        let mut looped = world.clone();

        let bonds = world
            .graph_mut()
            .link_many(&pairs)
            .expect("three chain pairs link");
        for &(a, b) in &pairs {
            looped.link(a, b).expect("each chain pair links in turn");
        }

        assert_eq!(world_state(&world), world_state(&looped));
        assert_eq!(bonds.len(), 3, "one bond per pair");
        for (k, &bid) in bonds.iter().enumerate() {
            let bond = world.get_bond(bid).expect("a returned bond is live");
            let mut ends = bond.nodes.to_vec();
            ends.sort();
            let mut expected = vec![units[k].anchor, units[k + 1].anchor];
            expected.sort();
            assert_eq!(ends, expected, "bond {k} joins pair {k}'s anchors");
        }
    }

    #[test]
    fn link_many_of_an_empty_slice_changes_nothing() {
        let mut world = Atomistic::new();
        chain(&mut world, 2);
        let before = snapshot(&world);

        let bonds = world
            .graph_mut()
            .link_many(&[])
            .expect("an empty batch succeeds");

        assert!(bonds.is_empty());
        assert_eq!(snapshot(&world), before);
    }

    /// Golden C3 as one batch: −0.5 + 0.125 + 0.125 on the shared anchor.
    #[test]
    fn link_many_folds_a_shared_anchor_in_pair_order() {
        let mut world = Atomistic::new();
        let a = atom(&mut world, Element::C, Some(-0.5), 0);
        let h1 = atom(&mut world, Element::H, Some(0.125), 0);
        let h2 = atom(&mut world, Element::H, Some(0.125), 0);
        world.add_bond(a, h1, props! {}).unwrap();
        world.add_bond(a, h2, props! {}).unwrap();
        let pa1 = world
            .add_port(a, h1, PortKind::Left, "p", BondNumber::Single)
            .unwrap();
        let pa2 = world
            .add_port(a, h2, PortKind::Left, "p", BondNumber::Single)
            .unwrap();
        let (b1, _, pb1) = h_unit(
            &mut world,
            Element::C,
            Some(-0.125),
            Some(0.125),
            PortKind::Right,
            1,
        );
        let (b2, _, pb2) = h_unit(
            &mut world,
            Element::C,
            Some(-0.125),
            Some(0.125),
            PortKind::Right,
            2,
        );

        world
            .graph_mut()
            .link_many(&[(pa1, pb1), (pa2, pb2)])
            .expect("two pairs on one anchor link in one batch");

        assert_eq!(charge(&world, a), -0.25, "C3: -0.5 + 0.125 + 0.125");
        assert_eq!(charge(&world, b1), 0.0);
        assert_eq!(charge(&world, b2), 0.0);
        assert_eq!(world.n_atoms(), 3);
        assert_eq!(world.n_bonds(), 2);
        assert_eq!(world.n_ports(), 0);
    }

    #[test]
    fn link_many_names_the_pair_link_refuses() {
        let mut world = Atomistic::new();
        let units = chain(&mut world, 2);
        let pairs = [
            (units[0].right, units[1].left),
            (units[0].left, units[1].left),
        ];

        let err = refuse_many(&mut world, &pairs);

        assert!(
            matches!(
                err,
                LinkManyError::Pair {
                    pair: 1,
                    source: LinkError::Incompatible { a, b },
                } if a == units[0].left && b == units[1].left
            ),
            "{err:?}"
        );
    }

    #[test]
    fn link_many_refuses_a_port_used_twice() {
        let mut world = Atomistic::new();
        let units = chain(&mut world, 3);
        let pairs = [
            (units[0].right, units[1].left),
            (units[0].right, units[2].left),
        ];

        let err = refuse_many(&mut world, &pairs);

        assert!(
            matches!(
                err,
                LinkManyError::PortReused {
                    port,
                    first: 0,
                    second: 1,
                } if port == units[0].right
            ),
            "{err:?}"
        );
    }

    #[test]
    fn link_many_refuses_two_pairs_on_one_anchor_couple() {
        let mut world = Atomistic::new();
        let a = atom(&mut world, Element::C, None, 0);
        let b = atom(&mut world, Element::C, None, 1);
        let mut ports = Vec::new();
        for (anchor, kind, frag) in [
            (a, PortKind::Left, 0),
            (a, PortKind::Left, 0),
            (b, PortKind::Right, 1),
            (b, PortKind::Right, 1),
        ] {
            let h = atom(&mut world, Element::H, None, frag);
            world.add_bond(anchor, h, props! {}).unwrap();
            ports.push(
                world
                    .add_port(anchor, h, kind, "p", BondNumber::Single)
                    .unwrap(),
            );
        }
        // Reversed port order in the second pair: the couple is unordered.
        let pairs = [(ports[0], ports[2]), (ports[3], ports[1])];

        let err = refuse_many(&mut world, &pairs);

        assert!(
            matches!(
                err,
                LinkManyError::DuplicateBond {
                    first: 0,
                    second: 1
                }
            ),
            "{err:?}"
        );
    }

    /// A – X – C with X the handle of a port on A and of a port on C: each
    /// pair alone passes, but the two leaving groups share X.
    #[test]
    fn link_many_refuses_leaving_groups_that_share_an_atom() {
        let mut world = Atomistic::new();
        let a = atom(&mut world, Element::C, None, 0);
        let x = atom(&mut world, Element::O, None, 0);
        let c = atom(&mut world, Element::C, None, 0);
        world.add_bond(a, x, props! {}).unwrap();
        world.add_bond(x, c, props! {}).unwrap();
        let pa = world
            .add_port(a, x, PortKind::Left, "p", BondNumber::Single)
            .unwrap();
        let pc = world
            .add_port(c, x, PortKind::Left, "p", BondNumber::Single)
            .unwrap();
        let (_, _, pb) = h_unit(&mut world, Element::C, None, None, PortKind::Right, 1);
        let (_, _, pd) = h_unit(&mut world, Element::C, None, None, PortKind::Right, 2);

        let err = refuse_many(&mut world, &[(pa, pb), (pc, pd)]);

        assert!(
            matches!(
                err,
                LinkManyError::BranchesOverlap {
                    first: 0,
                    second: 1
                }
            ),
            "{err:?}"
        );
    }

    /// C – O – N – H_N: pair 1's leaving group {O, N, H_N} holds pair 0's
    /// anchor N. The batch is refused, yet a loop of `link` accepts the same
    /// pairs (N is bonded to M, then removed with O) — the equivalence runs
    /// one way only.
    #[test]
    fn link_many_refuses_an_anchor_inside_another_leaving_group_a_loop_accepts() {
        let mut world = Atomistic::new();
        let c = atom(&mut world, Element::C, None, 0);
        let o = atom(&mut world, Element::O, None, 0);
        world.add_bond(c, o, props! {}).unwrap();
        let pk = world
            .add_port(c, o, PortKind::Left, "p", BondNumber::Single)
            .unwrap();
        let (n, _, pj) = h_unit(&mut world, Element::N, None, None, PortKind::Left, 0);
        world.add_bond(o, n, props! {}).unwrap();
        let (_, _, pm) = h_unit(&mut world, Element::C, None, None, PortKind::Right, 1);
        let (_, _, pp) = h_unit(&mut world, Element::C, None, None, PortKind::Right, 2);
        let pairs = [(pj, pm), (pk, pp)];
        let mut looped = world.clone();

        let err = refuse_many(&mut world, &pairs);

        assert!(
            matches!(
                err,
                LinkManyError::BranchesOverlap {
                    first: 0,
                    second: 1
                }
            ),
            "{err:?}"
        );
        for &(a, b) in &pairs {
            looped.link(a, b).expect("the sequential loop accepts");
        }
    }

    #[test]
    fn link_many_leaves_unlinked_end_ports_untouched() {
        let mut world = Atomistic::new();
        let units = chain(&mut world, 3);
        let head = world.get_port(units[0].left).expect("head port");
        let tail = world.get_port(units[2].right).expect("tail port");

        world
            .graph_mut()
            .link_many(&chain_pairs(&units))
            .expect("two chain pairs link");

        assert_eq!(world.n_ports(), 2, "only the two end ports remain");
        assert_eq!(world.get_port(units[0].left).ok(), Some(head));
        assert_eq!(world.get_port(units[2].right).ok(), Some(tail));
        assert!(world.get_atom(units[0].left_h).is_ok(), "head H survives");
        assert!(world.get_atom(units[2].right_h).is_ok(), "tail H survives");
        for inner in [
            units[0].right_h,
            units[1].left_h,
            units[1].right_h,
            units[2].left_h,
        ] {
            assert!(world.get_atom(inner).is_err(), "inner H is removed");
        }
    }
}
