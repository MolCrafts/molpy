//! Induced subgraph and radius-ball extraction on [`MolGraph`].
//!
//! These are pure structure-fact kernels: which nodes lie within *R* hops of a
//! set of centers, and what is the induced edge set. Region / force-field policy
//! lives in consumers (e.g. molpy), not here.

use std::collections::{HashMap, HashSet, VecDeque};

use crate::core::MolRsError;
use crate::core::{KindId, MolGraph, NodeId, RelationId};

/// Result of [`MolGraph::induced_subgraph`].
#[derive(Debug, Clone)]
pub struct InducedSubgraph {
    pub graph: MolGraph,
    /// Parent [`NodeId`] → node id in [`Self::graph`].
    pub node_map: HashMap<NodeId, NodeId>,
}

/// Result of [`MolGraph::extract_ball`].
///
/// Boundary / hops keys are **parent** handles (before remap). `node_map` maps
/// parent → new; `parent_of` is the inverse over new handles.
///
/// With `whole_groups`, `hops` is the distance from the nearest center
/// *measured inside the extracted node set*. For every node the radius alone
/// would have selected this is the true graph distance — a shortest path to a
/// node `radius` hops out has every intermediate node closer still, so the
/// whole path is selected. Only nodes pulled in by a group can carry a
/// longer-than-true distance, and they are by construction beyond `radius`.
#[derive(Debug, Clone)]
pub struct ExtractedBall {
    pub graph: MolGraph,
    /// Selected parent nodes that have a `kind`-neighbor outside the ball.
    pub boundary: Vec<NodeId>,
    /// New node id → parent node id.
    pub parent_of: HashMap<NodeId, NodeId>,
    /// Parent node id → hops from the nearest center.
    pub hops: HashMap<NodeId, i64>,
    /// Parent node id → new node id.
    pub node_map: HashMap<NodeId, NodeId>,
}

impl MolGraph {
    /// Induced subgraph on an explicit node set.
    ///
    /// Every handle in `nodes` must be live (fail-fast). All registered relation
    /// kinds are copied when every endpoint lies in the set. Empty `nodes`
    /// yields an empty graph (kinds still mirrored from `self` so leaf
    /// re-wrapping stays consistent).
    pub fn induced_subgraph(&self, nodes: &[NodeId]) -> Result<InducedSubgraph, MolRsError> {
        for &n in nodes {
            if !self.node_table().contains(n) {
                return Err(MolRsError::not_found("node", format!("NodeId {n:?}")));
            }
        }
        // Dedup while preserving first-seen order for deterministic maps.
        let mut seen = HashSet::with_capacity(nodes.len());
        let mut ordered: Vec<NodeId> = Vec::with_capacity(nodes.len());
        for &n in nodes {
            if seen.insert(n) {
                ordered.push(n);
            }
        }
        Ok(self.materialize_induced(&ordered, /* only_kind */ None))
    }

    /// Multi-source BFS ball over a 2-ary relation `kind`, then materialise.
    ///
    /// - `radius < 0` → validation error.
    /// - `kind` must be registered and arity 2.
    /// - Stale centers fail-fast.
    /// - `copy_higher_order = false` copies only `kind` relations (O(ball×degree)
    ///   via the adjacency index). `true` also copies every other kind by
    ///   scanning its relation table (small-graph / verbatim path).
    /// - `whole_groups` are node sets the ball may not split: a group with any
    ///   member inside the ball is selected in full. Closure is iterated to a
    ///   fixpoint, so overlapping groups are handled; groups nothing selects
    ///   cost nothing. Pass `&[]` for a plain ball.
    ///
    /// Which groups to pass is the caller's policy — this kernel only knows
    /// "do not select part of one". The chemistry case is ring systems: a cut
    /// ring is not a smaller molecule, it is a different one, and ring
    /// perception downstream reads it as such. Build them with
    /// [`small_ring_closure`](crate::perceive::small_ring_closure),
    /// which bounds ring size and costs the ball rather than the parent;
    /// [`RingSet::ring_systems`](crate::perceive::RingSet::ring_systems)
    /// answers the same question globally and without a bound, so it closes on
    /// macrocycles too and can hand back the entire molecule.
    pub fn extract_ball(
        &self,
        centers: &[NodeId],
        radius: i64,
        kind: KindId,
        copy_higher_order: bool,
        whole_groups: &[Vec<NodeId>],
    ) -> Result<ExtractedBall, MolRsError> {
        if radius < 0 {
            return Err(MolRsError::validation(format!(
                "extract_ball radius must be >= 0, got {radius}"
            )));
        }
        let kidx = kind.0 as usize;
        if kidx >= self.kind_ids().count() {
            return Err(MolRsError::not_found("kind", format!("KindId {kind:?}")));
        }
        if self.arity(kind) != 2 {
            return Err(MolRsError::validation(format!(
                "extract_ball requires arity-2 kind, '{}' has arity {}",
                self.kind_name(kind),
                self.arity(kind)
            )));
        }
        for &c in centers {
            if !self.node_table().contains(c) {
                return Err(MolRsError::not_found("node", format!("NodeId {c:?}")));
            }
        }

        if centers.is_empty() {
            let induced = self.materialize_induced(&[], only_kind(copy_higher_order, kind));
            return Ok(ExtractedBall {
                graph: induced.graph,
                boundary: Vec::new(),
                parent_of: HashMap::new(),
                hops: HashMap::new(),
                node_map: induced.node_map,
            });
        }

        // Multi-source BFS: hops[v] = min distance to any center.
        let mut hops: HashMap<NodeId, i64> = HashMap::new();
        let mut queue = VecDeque::new();
        for &c in centers {
            if hops.insert(c, 0).is_none() {
                queue.push_back(c);
            }
        }
        while let Some(cur) = queue.pop_front() {
            let d = hops[&cur];
            if d >= radius {
                continue;
            }
            for (k, _rid, other) in self.neighbor_relations(cur) {
                if k != kind {
                    continue;
                }
                let nd = d + 1;
                if nd > radius {
                    continue;
                }
                match hops.get(&other) {
                    Some(&old) if old <= nd => {}
                    _ => {
                        hops.insert(other, nd);
                        queue.push_back(other);
                    }
                }
            }
        }

        let mut selected: HashSet<NodeId> = hops.keys().copied().collect();

        // Close the ball under `whole_groups`, then re-measure hops inside the
        // grown set — the added nodes have no BFS distance yet, and a node the
        // radius already reached keeps the same one (see `ExtractedBall`).
        if self.close_under_groups(&mut selected, whole_groups) {
            hops = self.hops_within(centers, &selected, kind);
        }

        let mut boundary: Vec<NodeId> = selected
            .iter()
            .copied()
            .filter(|&n| {
                self.neighbor_relations(n)
                    .any(|(k, _, other)| k == kind && !selected.contains(&other))
            })
            .collect();
        boundary.sort_by_key(|n| n_as_sort_key(*n));

        let mut ordered: Vec<NodeId> = selected.iter().copied().collect();
        ordered.sort_by_key(|n| n_as_sort_key(*n));

        let induced = self.materialize_induced(&ordered, only_kind(copy_higher_order, kind));
        let parent_of: HashMap<NodeId, NodeId> = induced
            .node_map
            .iter()
            .map(|(&parent, &new)| (new, parent))
            .collect();

        Ok(ExtractedBall {
            graph: induced.graph,
            boundary,
            parent_of,
            hops,
            node_map: induced.node_map,
        })
    }

    /// Grow `selected` until no group in `whole_groups` is only partly in it.
    /// Returns whether anything was added.
    ///
    /// One pass suffices for disjoint groups; the loop is what makes the kernel
    /// total for overlapping ones (one group's completion can touch another).
    fn close_under_groups(
        &self,
        selected: &mut HashSet<NodeId>,
        whole_groups: &[Vec<NodeId>],
    ) -> bool {
        let mut grew = false;
        loop {
            let mut grew_this_pass = false;
            for group in whole_groups {
                if !group.iter().any(|n| selected.contains(n)) {
                    continue;
                }
                for &n in group {
                    // A group may name a node that is no longer live; selecting
                    // it would poison materialisation, so skip it.
                    if self.node_table().contains(n) && selected.insert(n) {
                        grew_this_pass = true;
                    }
                }
            }
            grew |= grew_this_pass;
            if !grew_this_pass {
                return grew;
            }
        }
    }

    /// Multi-source BFS from `centers` restricted to `selected`, unbounded.
    fn hops_within(
        &self,
        centers: &[NodeId],
        selected: &HashSet<NodeId>,
        kind: KindId,
    ) -> HashMap<NodeId, i64> {
        let mut hops: HashMap<NodeId, i64> = HashMap::with_capacity(selected.len());
        let mut queue = VecDeque::new();
        for &c in centers {
            if selected.contains(&c) && hops.insert(c, 0).is_none() {
                queue.push_back(c);
            }
        }
        while let Some(cur) = queue.pop_front() {
            let d = hops[&cur] + 1;
            for (k, _rid, other) in self.neighbor_relations(cur) {
                if k != kind || !selected.contains(&other) {
                    continue;
                }
                if let std::collections::hash_map::Entry::Vacant(slot) = hops.entry(other) {
                    slot.insert(d);
                    queue.push_back(other);
                }
            }
        }
        hops
    }

    /// Build a new graph with mirrored kind registry, remapped nodes in
    /// `ordered`, and relations either of every kind or only `only_kind`.
    fn materialize_induced(
        &self,
        ordered: &[NodeId],
        only_kind: Option<KindId>,
    ) -> InducedSubgraph {
        let mut graph = MolGraph::new();
        // Mirror every kind so Atomistic/CoarseGrain re-wrap sees stable names.
        for kid in self.kind_ids() {
            graph.register_kind(self.kind_name(kid), self.arity(kid));
        }

        let mut node_map: HashMap<NodeId, NodeId> = HashMap::with_capacity(ordered.len());
        for &old in ordered {
            let payload = self.materialize_atom(old);
            // The payload was just read out of `self`'s own node columns into
            // a graph that mirrors `self`'s kinds, and an `EntityTable` holds
            // one element type per key, so every key arrives at the type it
            // already has there: a conflict would mean the subgraph stopped
            // mirroring the parent mid-copy.
            let new_id = graph
                .add_node(payload)
                .expect("induced subgraph mirrors the parent's component types");
            node_map.insert(old, new_id);
        }

        let selected: HashSet<NodeId> = ordered.iter().copied().collect();

        let copy_relation = |graph: &mut MolGraph, kid: KindId, rid: RelationId| {
            let rel = self.materialize_relation(kid, rid);
            if !rel.nodes.iter().all(|n| selected.contains(n)) {
                return;
            }
            let self_kind = graph
                .kind_id(self.kind_name(kid))
                .expect("kind mirrored at start of materialize_induced");
            let mapped: smallvec::SmallVec<[NodeId; 4]> =
                rel.nodes.iter().map(|n| node_map[n]).collect();
            // The props were just read out of `self`'s own columns into a
            // graph whose kinds mirror `self`'s, so every key arrives at the
            // one element type it has there: a conflict would mean the
            // subgraph stopped mirroring the parent mid-copy.
            match graph.add_relation(self_kind, &mapped, rel.props) {
                Ok(_) => {}
                Err(e @ MolRsError::Validation { .. }) => {
                    panic!("induced subgraph mirrors the parent's component types: {e}")
                }
                Err(_) => {}
            }
        };

        match only_kind {
            // Single kind: reach its relations through the adjacency index of
            // the selected nodes — O(ball × degree). Scanning the kind's whole
            // relation table instead would make every ball cost the *parent*,
            // which is the complexity this path exists to avoid: a 9-atom ball
            // on a 64 000-atom chain was walking all 64 000 bonds.
            Some(only) => {
                let mut copied: HashSet<RelationId> = HashSet::new();
                for &node in ordered {
                    for (kid, rid, _other) in self.neighbor_relations(node) {
                        // Each relation is reachable from every endpoint.
                        if kid == only && copied.insert(rid) {
                            copy_relation(&mut graph, kid, rid);
                        }
                    }
                }
            }
            // Every kind, including higher-order terms the adjacency index does
            // not key by endpoint: scan the tables (the verbatim path).
            None => {
                for kid in self.kind_ids() {
                    for rid in self.relation_ids(kid) {
                        copy_relation(&mut graph, kid, rid);
                    }
                }
            }
        }

        InducedSubgraph { graph, node_map }
    }
}

fn only_kind(copy_higher_order: bool, kind: KindId) -> Option<KindId> {
    if copy_higher_order { None } else { Some(kind) }
}

/// Stable sort key for NodeId without depending on slotmap Key internals in call sites.
fn n_as_sort_key(n: NodeId) -> u64 {
    use slotmap::Key;
    n.data().as_ffi()
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::core::Atomistic;
    use crate::core::CoarseGrain;
    use crate::core::Element;
    use crate::core::GraphAccess;
    use crate::core::Props;
    use crate::core::keys;
    use crate::props;

    /// The bond kind of `mol`'s graph, for the generic ball kernel.
    fn bonds(mol: &Atomistic) -> KindId {
        mol.graph()
            .bonds_kind()
            .expect("an Atomistic registers bonds")
    }

    fn linear_chain(n: usize) -> (Atomistic, Vec<crate::core::NodeId>) {
        let mut mol = Atomistic::new();
        let mut ids = Vec::with_capacity(n);
        for i in 0..n {
            ids.push(
                mol.add_atom(props! {
                    keys::ATOMIC_NUMBER => Element::C,
                    keys::X => i as f64,
                    keys::Y => 0.0,
                    keys::Z => 0.0,
                })
                .unwrap(),
            );
        }
        for w in ids.windows(2) {
            mol.add_bond(w[0], w[1], props! {}).unwrap();
        }
        (mol, ids)
    }

    #[test]
    fn induced_subgraph_path4_middle_two() {
        let (mut mol, ids) = linear_chain(4);
        mol.generate_topology(true, false, false, false).unwrap();
        assert!(mol.n_angles() >= 2);

        let sub = mol.graph().induced_subgraph(&[ids[1], ids[2]]).unwrap();
        assert_eq!(sub.graph.n_nodes(), 2);
        assert_eq!(sub.node_map.len(), 2);
        let bond = sub.graph.kind_id("bonds").unwrap();
        assert_eq!(sub.graph.n_relations(bond), 1);
        let angle = sub.graph.kind_id("angles").unwrap();
        assert_eq!(
            sub.graph.n_relations(angle),
            0,
            "angles need 3 endpoints; incomplete ones must not copy"
        );
    }

    #[test]
    fn induced_subgraph_stale_fails() {
        let mut mol = Atomistic::new();
        let a = mol
            .add_atom(props! { keys::ATOMIC_NUMBER => Element::C })
            .unwrap();
        mol.del_atom(a).unwrap();
        let err = mol.graph().induced_subgraph(&[a]);
        assert!(err.is_err());
    }

    // -- whole_groups: the ball may not split a group ------------------------

    #[test]
    fn extract_ball_whole_group_pulls_in_the_rest() {
        let (mol, ids) = linear_chain(10);
        let group = vec![ids[5], ids[6], ids[7], ids[8]];
        let ball = mol
            .graph()
            .extract_ball(&[ids[4]], 1, bonds(&mol), false, &[group])
            .unwrap();
        // radius alone: {3,4,5}. The group has 5 inside, so 6/7/8 come along.
        let selected: HashSet<_> = ball.node_map.keys().copied().collect();
        assert_eq!(selected.len(), 6);
        for i in [3, 4, 5, 6, 7, 8] {
            assert!(selected.contains(&ids[i]), "missing {i}");
        }
        // Distances stay honest: 3 and 5 are one hop out, 8 is four.
        assert_eq!(ball.hops[&ids[4]], 0);
        assert_eq!(ball.hops[&ids[5]], 1);
        assert_eq!(ball.hops[&ids[8]], 4);
    }

    #[test]
    fn extract_ball_untouched_group_costs_nothing() {
        let (mol, ids) = linear_chain(10);
        let far = vec![ids[8], ids[9]];
        let ball = mol
            .graph()
            .extract_ball(&[ids[1]], 1, bonds(&mol), false, &[far])
            .unwrap();
        assert_eq!(ball.node_map.len(), 3);
    }

    #[test]
    fn extract_ball_closes_overlapping_groups() {
        // {2,3} ∩ {3,4,5} ≠ ∅: completing the first exposes the second, so a
        // single pass would stop short of the fixpoint.
        let (mol, ids) = linear_chain(10);
        let groups = vec![vec![ids[2], ids[3]], vec![ids[3], ids[4], ids[5]]];
        let ball = mol
            .graph()
            .extract_ball(&[ids[1]], 1, bonds(&mol), false, &groups)
            .unwrap();
        let selected: HashSet<_> = ball.node_map.keys().copied().collect();
        for i in [0, 1, 2, 3, 4, 5] {
            assert!(selected.contains(&ids[i]), "missing {i}");
        }
        assert_eq!(selected.len(), 6);
    }

    #[test]
    fn extract_ball_boundary_follows_the_closed_set() {
        let (mol, ids) = linear_chain(10);
        let group = vec![ids[2], ids[3], ids[4]];
        let ball = mol
            .graph()
            .extract_ball(&[ids[2]], 0, bonds(&mol), false, &[group])
            .unwrap();
        // Selected {2,3,4}: only 2 and 4 still have a neighbour outside.
        let bset: HashSet<_> = ball.boundary.iter().copied().collect();
        assert_eq!(bset.len(), 2);
        assert!(bset.contains(&ids[2]) && bset.contains(&ids[4]));
    }

    #[test]
    fn extract_ball_linear_multi_center() {
        let (mol, ids) = linear_chain(10);
        // centers={2,7}, radius=1 → {1,2,3,6,7,8}
        let ball = mol
            .graph()
            .extract_ball(&[ids[2], ids[7]], 1, bonds(&mol), false, &[])
            .unwrap();
        assert_eq!(ball.hops.len(), 6);
        assert_eq!(ball.hops[&ids[2]], 0);
        assert_eq!(ball.hops[&ids[1]], 1);
        assert_eq!(ball.hops[&ids[3]], 1);
        assert_eq!(ball.hops[&ids[7]], 0);
        let bset: HashSet<_> = ball.boundary.iter().copied().collect();
        for edge in [ids[1], ids[3], ids[6], ids[8]] {
            assert!(bset.contains(&edge), "missing boundary {edge:?}");
        }
        assert_eq!(ball.graph.n_nodes(), 6);
    }

    #[test]
    fn extract_ball_radius_zero_singleton() {
        let (mol, ids) = linear_chain(5);
        let ball = mol
            .graph()
            .extract_ball(&[ids[2]], 0, bonds(&mol), false, &[])
            .unwrap();
        assert_eq!(ball.graph.n_nodes(), 1);
        assert_eq!(ball.hops[&ids[2]], 0);
        // center has neighbors outside → boundary
        assert_eq!(ball.boundary, vec![ids[2]]);
    }

    #[test]
    fn extract_ball_negative_radius_errors() {
        let (mol, ids) = linear_chain(2);
        let err = mol
            .graph()
            .extract_ball(&[ids[0]], -1, bonds(&mol), false, &[]);
        assert!(err.is_err());
    }

    #[test]
    fn extract_ball_stale_center_errors() {
        let mut mol = Atomistic::new();
        let a = mol
            .add_atom(props! { keys::ATOMIC_NUMBER => Element::C })
            .unwrap();
        mol.del_atom(a).unwrap();
        let err = mol.graph().extract_ball(&[a], 1, bonds(&mol), false, &[]);
        assert!(err.is_err());
    }

    #[test]
    fn atomistic_extract_regenerate_topology() {
        let (mol, ids) = linear_chain(5);
        // parent has no angles
        assert_eq!(mol.n_angles(), 0);
        let ext = mol
            .extract_subgraph(&[ids[2]], 2, /* regenerate_topology */ true, &[])
            .unwrap();
        // ball = atoms 0..4 all five; has 2-edge paths → angles
        assert_eq!(ext.graph.n_atoms(), 5);
        assert!(
            ext.graph.n_angles() > 0,
            "regenerate must perceive angles from ball bonds alone"
        );
    }

    #[test]
    fn atomistic_extract_copy_higher_order_angles() {
        let (mut mol, ids) = linear_chain(4);
        mol.generate_topology(true, false, false, false).unwrap();
        let parent_angles = mol.n_angles();
        assert!(parent_angles > 0);
        let ext = mol
            .extract_subgraph(&[ids[1]], 2, /* regenerate_topology */ false, &[])
            .unwrap();
        // full chain is in the ball of radius 2 from index 1
        assert_eq!(ext.graph.n_atoms(), 4);
        assert_eq!(ext.graph.n_angles(), parent_angles);
    }

    #[test]
    fn coarsegrain_extract_preserves_membership() {
        let mut cg = CoarseGrain::new();
        let b0 = cg
            .add_bead(
                props! { keys::BEAD_TYPE => "A", keys::X => 0.0, keys::Y => 0.0, keys::Z => 0.0 },
            )
            .unwrap();
        let b1 = cg
            .add_bead(
                props! { keys::BEAD_TYPE => "B", keys::X => 1.0, keys::Y => 0.0, keys::Z => 0.0 },
            )
            .unwrap();
        cg.add_bond(b0, b1, props! {}).unwrap();
        cg.set_bead_members(b0, vec![10, 20]).unwrap();
        let ext = cg.extract_subgraph(&[b0], 0).unwrap();
        assert_eq!(ext.graph.n_beads(), 1);
        let new_b = *ext.node_map.get(&b0).unwrap();
        assert_eq!(ext.graph.bead_members(new_b), &[10, 20]);
    }

    #[test]
    fn empty_centers_empty_ball() {
        let (mol, _) = linear_chain(3);
        let ball = mol
            .graph()
            .extract_ball(&[], 1, bonds(&mol), false, &[])
            .unwrap();
        assert_eq!(ball.graph.n_nodes(), 0);
        assert!(ball.hops.is_empty());
    }

    #[test]
    fn empty_induced() {
        let g = MolGraph::new();
        let sub = g.induced_subgraph(&[]).unwrap();
        assert_eq!(sub.graph.n_nodes(), 0);
        assert!(sub.node_map.is_empty());
    }

    #[test]
    fn materialize_does_not_need_atom_payload_import() {
        let mut g = MolGraph::new();
        let n = g.add_node(Props::new()).expect("fixture node");
        let _ = g.induced_subgraph(&[n]).unwrap();
    }

    #[test]
    fn single_kind_ball_copies_every_bond_inside_it() {
        // The adjacency-indexed path must not miss a bond that closes a cycle:
        // such a bond is not on any BFS tree, so a copy driven by the search
        // order rather than by adjacency would drop it and silently hand back
        // an acyclic slice of a ring.
        let mut mol = Atomistic::new();
        let ids: Vec<_> = (0..6)
            .map(|i| {
                mol.add_atom(props! {
                    keys::ATOMIC_NUMBER => Element::C,
                    keys::X => i as f64,
                    keys::Y => 0.0,
                    keys::Z => 0.0,
                })
                .unwrap()
            })
            .collect();
        for i in 0..6 {
            mol.add_bond(ids[i], ids[(i + 1) % 6], props! {}).unwrap();
        }

        let bond = bonds(&mol);
        // Radius 3 from one atom of a 6-ring selects all six.
        let ball = mol
            .graph()
            .extract_ball(&[ids[0]], 3, bond, false, &[])
            .unwrap();
        assert_eq!(ball.graph.n_nodes(), 6);
        assert_eq!(
            ball.graph.n_relations(ball.graph.kind_id("bonds").unwrap()),
            6,
            "the ring-closing bond was dropped"
        );
    }

    #[test]
    fn single_kind_and_verbatim_paths_agree_on_the_bond_set() {
        // `copy_higher_order` switches between the adjacency-indexed copy and
        // the full table scan. They may differ in which *higher-order* terms
        // they carry — never in the bonds.
        let (mut mol, ids) = linear_chain(8);
        mol.generate_topology(true, true, false, false).unwrap();
        let bond = bonds(&mol);

        let indexed = mol
            .graph()
            .extract_ball(&[ids[3]], 2, bond, false, &[])
            .unwrap();
        let verbatim = mol
            .graph()
            .extract_ball(&[ids[3]], 2, bond, true, &[])
            .unwrap();

        assert_eq!(indexed.graph.n_nodes(), verbatim.graph.n_nodes());
        let count = |g: &MolGraph| g.n_relations(g.kind_id("bonds").unwrap());
        assert_eq!(count(&indexed.graph), count(&verbatim.graph));
        assert_eq!(count(&indexed.graph), 4, "5 atoms in a row share 4 bonds");
    }
}
