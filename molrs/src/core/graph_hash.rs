//! Isomorphism-invariant structural graph hash, canonical ordering, and whole-graph isomorphism over a [`MolGraph`].

use std::collections::HashMap;

use crate::core::keys;
use crate::core::{Graph, GraphAccess, MolGraph, NodeId, PropValue, Props};
use crate::op::I;

// ---------------------------------------------------------------------------
// Deterministic hashing primitives (fixed-seed FNV-1a, 64-bit)
// ---------------------------------------------------------------------------

const FNV_OFFSET: u64 = 0xcbf2_9ce4_8422_2325;
const FNV_PRIME: u64 = 0x0000_0100_0000_01b3;

/// Fold a raw byte slice into an FNV-1a accumulator.
#[inline]
fn hash_bytes(mut h: u64, bytes: &[u8]) -> u64 {
    for &b in bytes {
        h ^= b as u64;
        h = h.wrapping_mul(FNV_PRIME);
    }
    h
}

/// Fold one `u64` (as little-endian bytes, so the digest is architecture
/// independent) into an FNV-1a accumulator.
#[inline]
fn hash_u64(h: u64, x: u64) -> u64 {
    hash_bytes(h, &x.to_le_bytes())
}

/// Fold a slice of `u64`s in order into an FNV-1a accumulator.
#[inline]
fn hash_u64_slice(mut h: u64, xs: &[u64]) -> u64 {
    for &x in xs {
        h = hash_u64(h, x);
    }
    h
}

// ---------------------------------------------------------------------------
// Node label helpers
// ---------------------------------------------------------------------------

/// The species label of a node — the first ingredient of its initial WL color.
///
/// The rule, applied in order:
///
/// 1. a node carrying [`keys::ATOMIC_NUMBER`] is an **atom**, labelled by that
///    integer as stored (the wildcard `*` atom is `0`, an ordinary label);
/// 2. otherwise a node carrying a string [`keys::BEAD_TYPE`] is a **bead**,
///    labelled by that name;
/// 3. a node with neither — a bare [`MolGraph`] node — is
///    [`Unlabelled`](NodeLabel::Unlabelled): every such node shares that one
///    label, and it equals no atom and no bead.
///
/// [`keys::ELEMENT`] is never read: a symbol is raw data and reaches the
/// species field only through
/// [`Element::fill_atomic_numbers`](crate::core::Element::fill_atomic_numbers).
/// Each variant is hashed behind its own tag byte, so an atom, a bead and an
/// unlabelled node can never share a color by payload coincidence.
#[derive(Debug, Clone, PartialEq, Eq)]
enum NodeLabel {
    Atom(I),
    Bead(String),
    Unlabelled,
}

impl NodeLabel {
    fn of(node: &Props) -> Self {
        if let Some(z) = node.get_int(keys::ATOMIC_NUMBER) {
            NodeLabel::Atom(z)
        } else if let Some(bead) = node.get_str(keys::BEAD_TYPE) {
            NodeLabel::Bead(bead.to_owned())
        } else {
            NodeLabel::Unlabelled
        }
    }

    /// Fold the label, tag byte first, into an FNV-1a accumulator.
    fn hash_into(&self, h: u64) -> u64 {
        match self {
            NodeLabel::Unlabelled => hash_bytes(h, &[0]),
            NodeLabel::Atom(z) => hash_u64(hash_bytes(h, &[1]), *z as i64 as u64),
            NodeLabel::Bead(name) => {
                let h = hash_u64(hash_bytes(h, &[2]), name.len() as u64);
                hash_bytes(h, name.as_bytes())
            }
        }
    }
}

/// Read a prop as an `f64` flag / number, accepting int or float storage.
#[inline]
fn prop_as_f64(v: Option<&PropValue>) -> Option<f64> {
    v.and_then(PropValue::as_f64)
}

/// The `bond_type` code of an aromatic bond. Also used to infer a node's
/// aromatic flag when no explicit `is_aromatic` prop is set.
const AROMATIC_BOND_TYPE: u64 = 4;

// ---------------------------------------------------------------------------
// GraphView — a dense, precomputed snapshot for the WL kernel
// ---------------------------------------------------------------------------

/// A dense, immutable snapshot of `g`: contiguous node indices, per-node initial
/// WL color, and a labeled adjacency list built once so the refinement loop and
/// the matcher never re-materialize relations.
///
/// Shared with the subgraph matcher (`perceive::SubgraphMatcher`), which reads `nodes`
/// and the neighbours in `adj` (it zeroes their edge labels); the WL colors are
/// this module's alone.
pub(crate) struct GraphView {
    /// Node handles in dense-index order (row order of the node table).
    pub(crate) nodes: Vec<NodeId>,
    /// Initial WL color per dense node index.
    init_colors: Vec<u64>,
    /// Per node: the `(neighbor dense index, edge-label bits)` of each incident
    /// arity-2 relation.
    pub(crate) adj: Vec<Vec<(usize, u64)>>,
    /// Total number of (undirected) edges.
    n_edges: usize,
}

impl GraphView {
    /// Build the dense snapshot: assign contiguous indices, read edge labels
    /// (bond order bits) from each incident relation, then compute initial
    /// colors from element/bead label + degree + charge + aromatic flag.
    pub(crate) fn build(g: &MolGraph) -> Self {
        let nodes: Vec<NodeId> = g.node_ids().collect();
        let index: HashMap<NodeId, usize> =
            nodes.iter().enumerate().map(|(i, &id)| (id, i)).collect();

        let mut adj: Vec<Vec<(usize, u64)>> = vec![Vec::new(); nodes.len()];
        let mut edge_count = 0usize;
        for (i, &id) in nodes.iter().enumerate() {
            for (kind, rid, other) in g.neighbor_relations(id) {
                let Some(&j) = index.get(&other) else {
                    continue;
                };
                // Both facts enter the hash: two molecules differing only in
                // whether a bond is aromatic are different molecules, and so are
                // two differing only in their Kekulé phase.
                let edge = g
                    .get_relation(kind, rid)
                    .ok()
                    .map(|r| {
                        let t = prop_as_f64(r.props.get(keys::BOND_TYPE)).unwrap_or(0.0) as u64;
                        let n = prop_as_f64(r.props.get(keys::BOND_NUMBER)).unwrap_or(0.0) as u64;
                        (t << 32) | n
                    })
                    .unwrap_or(0);
                adj[i].push((j, edge));
            }
            // Each undirected edge is seen once from each endpoint.
            edge_count += adj[i].len();
        }
        let n_edges = edge_count / 2;

        let init_colors = nodes
            .iter()
            .enumerate()
            .map(|(i, &id)| Self::initial_color(g, id, &adj[i]))
            .collect();

        Self {
            nodes,
            init_colors,
            adj,
            n_edges,
        }
    }

    /// Initial color of a node from its label, degree, charge and aromatic flag.
    fn initial_color(g: &MolGraph, id: NodeId, incident: &[(usize, u64)]) -> u64 {
        let atom = g.get_node(id).ok();
        let label = atom.as_ref().map_or(NodeLabel::Unlabelled, NodeLabel::of);
        let degree = incident.len() as u64;

        let charge = atom
            .as_ref()
            .and_then(|a| prop_as_f64(a.get(keys::CHARGE)))
            .unwrap_or(0.0);

        // Aromatic: explicit `is_aromatic` prop wins; else infer from any
        // incident bond whose class is aromatic.
        let explicit = match atom.as_ref().and_then(|a| a.get(keys::IS_AROMATIC)) {
            Some(PropValue::Bool(v)) => Some(*v),
            _ => None,
        };
        let aromatic = explicit.unwrap_or_else(|| {
            // The edge label packs `(bond_type << 32) | bond_number`, so the
            // class is the high half. Reinterpreting it as a float order of
            // 1.5 can never match anything.
            incident
                .iter()
                .any(|&(_, label)| (label >> 32) == AROMATIC_BOND_TYPE)
        });

        let mut h = FNV_OFFSET;
        h = label.hash_into(h);
        h = hash_u64(h, degree);
        h = hash_u64(h, charge.to_bits());
        h = hash_u64(h, aromatic as u64);
        h
    }
}

/// Run WL color refinement to convergence, returning the final color per dense
/// node index.
fn wl_colors(view: &GraphView) -> Vec<u64> {
    let n = view.nodes.len();
    let mut colors = view.init_colors.clone();
    if n == 0 {
        return colors;
    }

    let mut distinct = distinct_count(&colors);
    // WL stabilizes within `n` rounds; break early once the partition (distinct
    // color count, monotone non-decreasing) stops refining.
    for _ in 0..n {
        let mut next = vec![0u64; n];
        for i in 0..n {
            // Sorted multiset of (edge label, neighbor color).
            let mut env: Vec<(u64, u64)> = view.adj[i]
                .iter()
                .map(|&(j, label)| (label, colors[j]))
                .collect();
            env.sort_unstable();

            let mut h = hash_u64(FNV_OFFSET, colors[i]);
            for (label, neigh) in env {
                h = hash_u64(h, label);
                h = hash_u64(h, neigh);
            }
            next[i] = h;
        }
        let next_distinct = distinct_count(&next);
        colors = next;
        if next_distinct == distinct {
            break;
        }
        distinct = next_distinct;
    }
    colors
}

/// Number of distinct values in a color vector.
fn distinct_count(colors: &[u64]) -> usize {
    let mut sorted = colors.to_vec();
    sorted.sort_unstable();
    sorted.dedup();
    sorted.len()
}

// ---------------------------------------------------------------------------
// Public API
// ---------------------------------------------------------------------------

/// `g.structural_hash()`; see [`MolGraph::structural_hash`].
pub(crate) fn structural_hash(g: &impl Graph) -> u64 {
    let view = GraphView::build(g.graph());
    let colors = wl_colors(&view);
    let mut sorted = colors;
    sorted.sort_unstable();

    let mut h = FNV_OFFSET;
    h = hash_u64(h, view.nodes.len() as u64);
    h = hash_u64(h, view.n_edges as u64);
    hash_u64_slice(h, &sorted)
}

/// `g.canonical_order()`; see [`MolGraph::canonical_order`].
pub(crate) fn canonical_order(g: &impl Graph) -> Vec<NodeId> {
    let view = GraphView::build(g.graph());
    let colors = wl_colors(&view);

    let mut order: Vec<usize> = (0..view.nodes.len()).collect();
    order.sort_by(|&a, &b| {
        colors[a]
            .cmp(&colors[b])
            .then(view.init_colors[a].cmp(&view.init_colors[b]))
            .then_with(|| node_ffi(view.nodes[a]).cmp(&node_ffi(view.nodes[b])))
    });
    order.into_iter().map(|i| view.nodes[i]).collect()
}

/// `a.is_isomorphic(b)`; see [`MolGraph::is_isomorphic`].
pub(crate) fn is_isomorphic(a: &impl Graph, b: &impl Graph) -> bool {
    let (a, b) = (a.graph(), b.graph());
    if a.n_nodes() != b.n_nodes() {
        return false;
    }
    let va = GraphView::build(a);
    let vb = GraphView::build(b);
    if va.n_edges != vb.n_edges {
        return false;
    }

    let ca = wl_colors(&va);
    let cb = wl_colors(&vb);
    {
        let mut sa = ca.clone();
        let mut sb = cb.clone();
        sa.sort_unstable();
        sb.sort_unstable();
        if sa != sb {
            return false;
        }
    }

    let n = va.nodes.len();
    if n == 0 {
        return true;
    }

    // Candidate B nodes by final color.
    let mut by_color: HashMap<u64, Vec<usize>> = HashMap::new();
    for (j, &c) in cb.iter().enumerate() {
        by_color.entry(c).or_default().push(j);
    }
    // Order A nodes rarest-color-first for tighter pruning.
    let mut freq: HashMap<u64, usize> = HashMap::new();
    for &c in &ca {
        *freq.entry(c).or_insert(0) += 1;
    }
    let mut order: Vec<usize> = (0..n).collect();
    order.sort_by_key(|&i| (*freq.get(&ca[i]).unwrap_or(&0), i));

    let adj_a = adjacency_map(&va);
    let adj_b = adjacency_map(&vb);

    let mut map_ab = vec![usize::MAX; n];
    let mut map_ba = vec![usize::MAX; n];
    backtrack(
        0,
        &order,
        &ca,
        &cb,
        &by_color,
        &adj_a,
        &adj_b,
        &mut map_ab,
        &mut map_ba,
    )
}

impl MolGraph {
    /// Isomorphism-invariant 64-bit structural hash of the graph.
    ///
    /// Identical for a node-permuted copy; sensitive to element/bead-type, degree,
    /// charge, aromatic flag, bond order, and connectivity.
    ///
    /// This, [`canonical_order`](Self::canonical_order) and
    /// [`is_isomorphic`](Self::is_isomorphic) are built on the generic
    /// [`MolGraph`] adjacency (arity-2 relations are the graph edges), so they serve
    /// both [`Atomistic`](crate::core::Atomistic) and
    /// [`CoarseGrain`](crate::core::CoarseGrain) unchanged — the leaf only supplies
    /// its node vocabulary (`atomic_number` / `bead_type`).
    ///
    /// # Algorithm — Weisfeiler–Lehman color refinement
    ///
    /// - **Initial node color** = a deterministic hash of the node's *label*: the
    ///   atomic number (atoms), the bead type (CG), or the one shared
    ///   "unlabelled" label of a node carrying neither + degree (incident edge count) + formal
    ///   charge (the `charge` component, `0` when absent) + an aromatic flag (the
    ///   `is_aromatic` component when set, else inferred from any incident bond of
    ///   order `1.5`, matching the project aromaticity convention).
    /// - **Refinement round**: a node's new color = hash of its current color plus the
    ///   *sorted multiset* of `(edge label = bond order, neighbor's current color)`.
    ///   Rounds repeat until the color partition stops refining (the count of
    ///   distinct colors — which is monotone non-decreasing under WL — stabilizes),
    ///   bounded by the node count.
    /// - `structural_hash` = a hash of the *sorted multiset of final colors*, so it
    ///   is invariant under any node permutation.
    /// - [`canonical_order`](Self::canonical_order) = nodes sorted by `(final color, initial color, stable
    ///   handle tiebreak)` — deterministic within a graph, and consistent across two
    ///   isomorphic graphs (tied nodes share a color, hence share every label
    ///   attribute).
    /// - [`is_isomorphic`](Self::is_isomorphic) = a quick reject on `structural_hash` / edge-count
    ///   mismatch, then a WL-color-pruned backtracking bijective match.
    ///
    /// The hasher is a fixed-constant FNV-1a over little-endian bytes — **not** the
    /// std `DefaultHasher` (random-seeded) — so a hash is reproducible across runs
    /// and processes and can be trusted as a persistent cache key.
    pub fn structural_hash(&self) -> u64 {
        structural_hash(self)
    }

    /// Deterministic canonical node ordering from the WL refinement.
    ///
    /// Nodes are sorted by `(final color, initial color, stable handle)`. Two
    /// isomorphic graphs induce a consistent node bijection by pairing their
    /// `canonical_order` lists position-by-position.
    ///
    /// The refinement is described at [`structural_hash`](Self::structural_hash).
    pub fn canonical_order(&self) -> Vec<NodeId> {
        canonical_order(self)
    }

    /// Whether `self` and `other` are isomorphic as labeled graphs
    /// (atomic-number/bead-type + charge + aromatic node labels and
    /// bond-order edge labels all preserved).
    ///
    /// Quick-rejects on node-count, edge-count, or `structural_hash` mismatch,
    /// then runs a WL-color-pruned backtracking bijective match to resolve the
    /// rare hash collision before equality is trusted.
    ///
    /// The refinement is described at [`structural_hash`](Self::structural_hash).
    pub fn is_isomorphic(&self, other: &MolGraph) -> bool {
        is_isomorphic(self, other)
    }
}

// ---------------------------------------------------------------------------
// Isomorphism backtracking internals
// ---------------------------------------------------------------------------

/// Neighbor → edge-label map per dense node index (O(1) edge lookup).
pub(crate) fn adjacency_map(view: &GraphView) -> Vec<HashMap<usize, u64>> {
    view.adj
        .iter()
        .map(|nbrs| nbrs.iter().map(|&(j, label)| (j, label)).collect())
        .collect()
}

/// Recursive WL-pruned bijective match. `order` drives the A-node visitation
/// order; `map_ab` / `map_ba` are the partial forward / inverse maps.
#[allow(clippy::too_many_arguments)]
fn backtrack(
    depth: usize,
    order: &[usize],
    ca: &[u64],
    cb: &[u64],
    by_color: &HashMap<u64, Vec<usize>>,
    adj_a: &[HashMap<usize, u64>],
    adj_b: &[HashMap<usize, u64>],
    map_ab: &mut [usize],
    map_ba: &mut [usize],
) -> bool {
    if depth == order.len() {
        return true;
    }
    let u = order[depth];
    let Some(candidates) = by_color.get(&ca[u]) else {
        return false;
    };
    for &v in candidates {
        if map_ba[v] != usize::MAX || cb[v] != ca[u] {
            continue;
        }
        if feasible(u, v, adj_a, adj_b, map_ab, map_ba) {
            map_ab[u] = v;
            map_ba[v] = u;
            if backtrack(
                depth + 1,
                order,
                ca,
                cb,
                by_color,
                adj_a,
                adj_b,
                map_ab,
                map_ba,
            ) {
                return true;
            }
            map_ab[u] = usize::MAX;
            map_ba[v] = usize::MAX;
        }
    }
    false
}

/// Whether mapping `u -> v` is consistent with the edges among already-mapped
/// nodes, in both directions (so no A-edge is dropped and no extra B-edge is
/// introduced).
///
/// Map slots hold a dense index of the other graph, or `usize::MAX` when
/// unmapped. The check is purely structural, so it serves both the bijective
/// [`is_isomorphic`] search and the injective, induced subgraph search of
/// `perceive::SubgraphMatcher` (where `map_ba` is sized to the larger graph and its
/// unmapped slots are skipped); node-label compatibility is each caller's own
/// test.
pub(crate) fn feasible(
    u: usize,
    v: usize,
    adj_a: &[HashMap<usize, u64>],
    adj_b: &[HashMap<usize, u64>],
    map_ab: &[usize],
    map_ba: &[usize],
) -> bool {
    // Every already-mapped A-neighbor of u must be a B-neighbor of v, same label.
    for (&w, &label) in &adj_a[u] {
        let mw = map_ab[w];
        if mw != usize::MAX && adj_b[v].get(&mw) != Some(&label) {
            return false;
        }
    }
    // Every already-mapped B-neighbor of v must be an A-neighbor of u, same label.
    for (&x, &label) in &adj_b[v] {
        let mx = map_ba[x];
        if mx != usize::MAX && adj_a[u].get(&mx) != Some(&label) {
            return false;
        }
    }
    true
}

/// Stable per-graph tiebreak key for a node handle.
#[inline]
fn node_ffi(id: NodeId) -> u64 {
    u64::from(id)
}

// =========================================================================
// Tests
// =========================================================================

#[cfg(test)]
mod tests {
    use super::*;
    use crate::core::Atomistic;
    use crate::core::CoarseGrain;
    use crate::core::Element;
    use crate::core::NodeId;
    use crate::core::keys;
    use crate::props;

    /// Ethanol skeleton C-C-O with explicit H (9 atoms), returned with its
    /// atom handles in build order.
    fn ethanol() -> (Atomistic, Vec<NodeId>) {
        let mut mol = Atomistic::new();
        let ids: Vec<NodeId> = [
            Element::C,
            Element::C,
            Element::O,
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
        // C0-C1, C1-O2, O2-H8; H3,H4,H5 on C0; H6,H7 on C1
        for (i, j) in [
            (0, 1),
            (1, 2),
            (2, 8),
            (0, 3),
            (0, 4),
            (0, 5),
            (1, 6),
            (1, 7),
        ] {
            mol.add_bond(ids[i], ids[j], props! {}).unwrap();
        }
        (mol, ids)
    }

    /// Rebuild the same molecule with a permuted node-insertion order.
    fn ethanol_permuted() -> Atomistic {
        let mut mol = Atomistic::new();
        // Insert atoms in a shuffled order, remember by original index.
        let perm = [8usize, 2, 1, 0, 7, 6, 5, 4, 3];
        let elems = [
            Element::C,
            Element::C,
            Element::O,
            Element::H,
            Element::H,
            Element::H,
            Element::H,
            Element::H,
            Element::H,
        ];
        let mut by_orig = std::collections::HashMap::new();
        for &orig in &perm {
            let id = mol
                .add_atom(props! { keys::ATOMIC_NUMBER => elems[orig] })
                .unwrap();
            by_orig.insert(orig, id);
        }
        for (i, j) in [
            (0, 1),
            (1, 2),
            (2, 8),
            (0, 3),
            (0, 4),
            (0, 5),
            (1, 6),
            (1, 7),
        ] {
            mol.add_bond(by_orig[&i], by_orig[&j], props! {}).unwrap();
        }
        mol
    }

    #[test]
    fn structural_hash_is_permutation_invariant() {
        let (a, _) = ethanol();
        let b = ethanol_permuted();
        assert_eq!(
            a.structural_hash(),
            b.structural_hash(),
            "hash must be invariant under node permutation"
        );
        assert!(a.is_isomorphic(&b));
    }

    #[test]
    fn identical_junctions_hash_equal() {
        let (a, _) = ethanol();
        let (b, _) = ethanol();
        assert_eq!(a.structural_hash(), b.structural_hash());
        assert!(a.is_isomorphic(&b));
    }

    #[test]
    fn element_change_changes_hash() {
        let (a, _) = ethanol();
        // Swap the O for an S — a different local environment.
        let mut mol = Atomistic::new();
        let ids: Vec<NodeId> = [
            Element::C,
            Element::C,
            Element::S,
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
        for (i, j) in [
            (0, 1),
            (1, 2),
            (2, 8),
            (0, 3),
            (0, 4),
            (0, 5),
            (1, 6),
            (1, 7),
        ] {
            mol.add_bond(ids[i], ids[j], props! {}).unwrap();
        }
        assert_ne!(a.structural_hash(), mol.structural_hash());
        assert!(!a.is_isomorphic(&mol));
    }

    #[test]
    fn charge_change_changes_hash() {
        let (a, ids) = ethanol();
        let mut b = a.clone();
        b.set_atom(ids[2], keys::CHARGE, -1.0).unwrap();
        assert_ne!(
            a.structural_hash(),
            b.structural_hash(),
            "charge must affect the hash"
        );
    }

    #[test]
    fn aromatic_class_alone_changes_the_hash() {
        // The fallback path: no atom carries `is_aromatic`, so aromaticity is
        // inferred from the incident bond's *class*. This had no test, and the
        // inference was silently dead — it reinterpreted the packed integer
        // edge label as a float and compared it to 1.5, which never matches.
        let (a, _) = ethanol();
        let mut b = a.clone();
        let (bid, _) = b.bonds().next().unwrap();
        b.set_bond_class(
            bid,
            crate::core::BondOrder::Aromatic,
            crate::core::BondNumber::Single,
        )
        .unwrap();

        assert!(
            b.atoms().all(|(_, at)| at.get(keys::IS_AROMATIC).is_none()),
            "the fallback only runs when no atom states its own flag"
        );
        assert_ne!(
            a.structural_hash(),
            b.structural_hash(),
            "an aromatic bond class must reach the hash through the fallback"
        );
    }

    #[test]
    fn aromatic_flag_changes_hash() {
        let (a, ids) = ethanol();
        let mut b = a.clone();
        b.set_atom(ids[0], keys::IS_AROMATIC, true).unwrap();
        assert_ne!(
            a.structural_hash(),
            b.structural_hash(),
            "aromatic flag must affect the hash"
        );
    }

    #[test]
    fn bond_order_change_changes_hash() {
        let (a, _) = ethanol();
        let mut b = a.clone();
        let (bid, _) = b.bonds().next().unwrap();
        b.set_bond_class(
            bid,
            crate::core::BondOrder::Double,
            crate::core::BondNumber::Double,
        )
        .unwrap();
        assert_ne!(
            a.structural_hash(),
            b.structural_hash(),
            "bond order must affect the hash"
        );
    }

    #[test]
    fn non_isomorphic_graphs_rejected() {
        // Ethanol (9 atoms) vs a 9-atom linear chain — same size, different topo.
        let (a, _) = ethanol();
        let mut chain = Atomistic::new();
        let ids: Vec<NodeId> = (0..9)
            .map(|_| {
                chain
                    .add_atom(props! { keys::ATOMIC_NUMBER => Element::C })
                    .unwrap()
            })
            .collect();
        for k in 0..ids.len() - 1 {
            chain.add_bond(ids[k], ids[k + 1], props! {}).unwrap();
        }
        assert!(!a.is_isomorphic(&chain));
    }

    #[test]
    fn canonical_order_bijection_is_consistent() {
        let (a, _) = ethanol();
        let b = ethanol_permuted();
        let oa = a.canonical_order();
        let ob = b.canonical_order();
        assert_eq!(oa.len(), ob.len());
        // Paired nodes must agree on atomic number + degree.
        for (ia, ib) in oa.iter().zip(ob.iter()) {
            let ea = a.get_atom(*ia).unwrap();
            let eb = b.get_atom(*ib).unwrap();
            assert_eq!(ea.atomic_number(), eb.atomic_number());
            assert_eq!(a.neighbor_bonds(*ia).count(), b.neighbor_bonds(*ib).count());
        }
    }

    #[test]
    fn coarsegrain_hashes_by_bead_type() {
        // Two identical 3-bead angular CG molecules hash equal; changing a bead
        // type breaks both the hash and the isomorphism.
        let mut cg1 = CoarseGrain::new();
        let a = cg1.add_bead(props! { keys::BEAD_TYPE => "W" }).unwrap();
        let b = cg1.add_bead(props! { keys::BEAD_TYPE => "P" }).unwrap();
        let c = cg1.add_bead(props! { keys::BEAD_TYPE => "W" }).unwrap();
        cg1.add_bond(a, b, props! {}).unwrap();
        cg1.add_bond(b, c, props! {}).unwrap();

        let mut cg2 = CoarseGrain::new();
        let d = cg2.add_bead(props! { keys::BEAD_TYPE => "W" }).unwrap();
        let e = cg2.add_bead(props! { keys::BEAD_TYPE => "P" }).unwrap();
        let f = cg2.add_bead(props! { keys::BEAD_TYPE => "W" }).unwrap();
        cg2.add_bond(e, f, props! {}).unwrap();
        cg2.add_bond(d, e, props! {}).unwrap();

        assert_eq!(cg1.structural_hash(), cg2.structural_hash());
        assert!(cg1.is_isomorphic(&cg2));

        let mut cg3 = CoarseGrain::new();
        let g = cg3.add_bead(props! { keys::BEAD_TYPE => "W" }).unwrap();
        let h = cg3.add_bead(props! { keys::BEAD_TYPE => "Q" }).unwrap(); // different centre type
        let i = cg3.add_bead(props! { keys::BEAD_TYPE => "W" }).unwrap();
        cg3.add_bond(g, h, props! {}).unwrap();
        cg3.add_bond(h, i, props! {}).unwrap();
        assert_ne!(cg1.structural_hash(), cg3.structural_hash());
        assert!(!cg1.is_isomorphic(&cg3));
    }

    #[test]
    fn empty_graph_is_stable() {
        let a = Atomistic::new();
        let b = Atomistic::new();
        assert_eq!(a.structural_hash(), b.structural_hash());
        assert!(a.is_isomorphic(&b));
        assert!(a.canonical_order().is_empty());
    }

    // ---- the node-label rule ----

    /// One node per graph: its structural hash is a function of its label.
    fn one_node(props: crate::core::Props) -> MolGraph {
        let mut g = MolGraph::new();
        g.add_node(props).unwrap();
        g
    }

    #[test]
    fn an_atom_is_labelled_by_its_atomic_number_not_its_symbol() {
        let carbon = one_node(props! { keys::ATOMIC_NUMBER => Element::C });
        let carbon_spelled = one_node(props! {
            keys::ATOMIC_NUMBER => Element::C,
            keys::ELEMENT => "X",
        });
        let oxygen = one_node(props! { keys::ATOMIC_NUMBER => Element::O });
        assert_eq!(carbon.structural_hash(), carbon_spelled.structural_hash());
        assert!(carbon.is_isomorphic(&carbon_spelled));
        assert_ne!(carbon.structural_hash(), oxygen.structural_hash());
        assert!(!carbon.is_isomorphic(&oxygen));
    }

    #[test]
    fn a_symbol_alone_is_not_a_species() {
        let c = one_node(props! { keys::ELEMENT => "C" });
        let o = one_node(props! { keys::ELEMENT => "O" });
        let bare = one_node(props! {});
        assert_eq!(c.structural_hash(), o.structural_hash());
        assert_eq!(c.structural_hash(), bare.structural_hash());
    }

    /// Atoms, beads and unlabelled nodes never share a label: not the
    /// wildcard atom (Z = 0) and an unlabelled node, nor an atom and a bead
    /// whose name spells its number.
    #[test]
    fn atoms_beads_and_unlabelled_nodes_hash_apart() {
        let graphs = [
            one_node(props! {}),
            one_node(props! { keys::ATOMIC_NUMBER => 0u8 }),
            one_node(props! { keys::ATOMIC_NUMBER => Element::C }),
            one_node(props! { keys::BEAD_TYPE => "6" }),
            one_node(props! { keys::BEAD_TYPE => "" }),
        ];
        for (i, a) in graphs.iter().enumerate() {
            for b in &graphs[i + 1..] {
                assert_ne!(a.structural_hash(), b.structural_hash());
                assert!(!a.is_isomorphic(b));
            }
        }
    }

    /// A node carrying both fields is an atom: the atomic number decides.
    #[test]
    fn atomic_number_takes_precedence_over_bead_type() {
        let both = one_node(props! {
            keys::ATOMIC_NUMBER => Element::C,
            keys::BEAD_TYPE => "W",
        });
        let atom = one_node(props! { keys::ATOMIC_NUMBER => Element::C });
        assert_eq!(both.structural_hash(), atom.structural_hash());
    }

    /// The crate-internal functions take any graph type through the sealed
    /// `Graph` bound and agree with each type's own methods.
    #[test]
    fn the_generic_functions_take_every_graph_type() {
        let (mol, _) = ethanol();
        assert_eq!(structural_hash(&mol), mol.structural_hash());
        assert_eq!(structural_hash(mol.graph()), mol.structural_hash());
        assert_eq!(canonical_order(&mol), mol.canonical_order());
        assert!(is_isomorphic(&mol, mol.graph()));

        let mut cg = CoarseGrain::new();
        cg.add_bead(props! { keys::BEAD_TYPE => "W" }).unwrap();
        assert_eq!(structural_hash(&cg), cg.structural_hash());
    }
}
