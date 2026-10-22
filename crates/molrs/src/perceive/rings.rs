//! Ring detection for molecular graphs — the **handle-keyed chemistry
//! decoration** over the core graph primitive.

use std::collections::{HashMap, HashSet, VecDeque};

use crate::core::Atomistic;
use crate::core::MolRsError;

use crate::core::Topology;
use crate::core::{NodeId, RelationId};

// ---------------------------------------------------------------------------
// Public types
// ---------------------------------------------------------------------------

/// The ring set of an [`Atomistic`], produced by [`perceive_rings`]: the SSSR
/// rings and per-atom / per-bond ring membership, addressed by [`NodeId`] /
/// [`RelationId`]. RDKit calls the same object `Chem.RingInfo`.
#[derive(Debug, Clone)]
pub struct RingSet {
    /// Each ring is an ordered list of `NodeId`s forming a closed path.
    rings: Vec<Vec<NodeId>>,
    /// atom → indices of rings that contain it.
    atom_rings: HashMap<NodeId, Vec<usize>>,
    /// bond → indices of rings that contain it.
    bond_rings: HashMap<RelationId, Vec<usize>>,
}

impl RingSet {
    /// Whether the atom belongs to any ring.
    pub fn is_atom_in_ring(&self, id: NodeId) -> bool {
        self.atom_rings.get(&id).is_some_and(|v| !v.is_empty())
    }

    /// Whether the bond belongs to any ring.
    pub fn is_bond_in_ring(&self, id: RelationId) -> bool {
        self.bond_rings.get(&id).is_some_and(|v| !v.is_empty())
    }

    /// Number of rings containing this atom.
    pub fn n_atom_rings(&self, id: NodeId) -> usize {
        self.atom_rings.get(&id).map_or(0, Vec::len)
    }

    /// Number of rings containing this bond.
    pub fn n_bond_rings(&self, id: RelationId) -> usize {
        self.bond_rings.get(&id).map_or(0, Vec::len)
    }

    /// Size (atom count) of every ring, in ascending order.
    pub fn ring_sizes(&self) -> Vec<usize> {
        self.rings.iter().map(Vec::len).collect()
    }

    /// All rings of exactly `n` atoms.
    pub fn rings_of_size(&self, n: usize) -> Vec<&Vec<NodeId>> {
        self.rings.iter().filter(|r| r.len() == n).collect()
    }

    /// Size of the smallest ring containing `id`, if any.
    pub fn smallest_ring_containing_atom(&self, id: NodeId) -> Option<usize> {
        self.atom_rings
            .get(&id)?
            .iter()
            .map(|&ri| self.rings[ri].len())
            .min()
    }

    /// Total number of rings detected.
    pub fn n_rings(&self) -> usize {
        self.rings.len()
    }

    /// All rings as slices of `NodeId`.
    pub fn rings(&self) -> &[Vec<NodeId>] {
        &self.rings
    }

    /// The **ring systems**: rings that share at least one atom, unioned into
    /// one atom list each. Systems are disjoint and cover every ring atom.
    ///
    /// Benzene → one system of 6; naphthalene → one of 10 (not two of 6);
    /// biphenyl → two of 6; a spiro pair → one system (they share their spiro
    /// atom). Acyclic molecules → empty.
    ///
    /// This is RDKit's `GetRingSystems` recipe (RDKit Cookbook, "Count Ring
    /// Systems"): walk the rings, merging any that intersect one already
    /// collected. The recipe's `includeSpiro` switch decides whether a
    /// single shared atom is enough to merge (`nInCommon > 1` when off); this
    /// method always merges on one, i.e. `includeSpiro=True`. The switch only
    /// changes how systems are *counted* — for any caller that asks "is this
    /// system wholly present?", a spiro atom belongs to both rings, so both
    /// are answered together either way.
    ///
    /// Which ring set feeds this does not matter: the true SSSR and RDKit's
    /// symmetrized `GetSymmSSSR` differ in *which* small rings they list, not
    /// in which atoms are cyclic, and the union over shared atoms collapses
    /// that difference.
    pub fn ring_systems(&self) -> Vec<Vec<NodeId>> {
        let n = self.rings.len();
        if n == 0 {
            return Vec::new();
        }
        // Union-find over rings: share an atom ⇒ same component.
        let mut parent: Vec<usize> = (0..n).collect();
        fn find(parent: &mut [usize], mut i: usize) -> usize {
            while parent[i] != i {
                parent[i] = parent[parent[i]];
                i = parent[i];
            }
            i
        }
        fn unite(parent: &mut [usize], a: usize, b: usize) {
            let ra = find(parent, a);
            let rb = find(parent, b);
            if ra != rb {
                parent[rb] = ra;
            }
        }
        // atom → first ring index seen
        let mut atom_owner: HashMap<NodeId, usize> = HashMap::new();
        for (ri, ring) in self.rings.iter().enumerate() {
            for &atom in ring {
                if let Some(&prev) = atom_owner.get(&atom) {
                    unite(&mut parent, prev, ri);
                } else {
                    atom_owner.insert(atom, ri);
                }
            }
        }
        // Component → atoms, in first-seen order so the result is deterministic
        // without leaking a handle ordering.
        let mut order: Vec<usize> = Vec::new();
        let mut systems: HashMap<usize, Vec<NodeId>> = HashMap::new();
        let mut seen: HashSet<NodeId> = HashSet::new();
        for (ri, ring) in self.rings.iter().enumerate() {
            let root = find(&mut parent, ri);
            let atoms = systems.entry(root).or_insert_with(|| {
                order.push(root);
                Vec::new()
            });
            for &atom in ring {
                if seen.insert(atom) {
                    atoms.push(atom);
                }
            }
        }
        order
            .into_iter()
            .map(|root| systems.remove(&root).expect("root inserted above"))
            .collect()
    }

    /// Size (atom count) of the largest [`ring system`](Self::ring_systems).
    ///
    /// Benzene → 6; naphthalene → 10 (not 6). Acyclic molecules → 0.
    pub fn max_ring_system_size(&self) -> usize {
        self.ring_systems().iter().map(Vec::len).max().unwrap_or(0)
    }
}

// ---------------------------------------------------------------------------
// Public entry point
// ---------------------------------------------------------------------------

/// The ring systems a `radius`-ball around `centers` must not cut, counting
/// only rings of at most `max_ring_size` atoms.
///
/// Feed the result to
/// [`Atomistic::extract_subgraph`](crate::core::Atomistic::extract_subgraph)
/// as its `whole_groups`.
///
/// # Why a size bound, and why local
///
/// "A cut ring is a different molecule" is a claim about **small** rings:
/// aromaticity, ring strain, SMARTS `r3`–`r8`, and MMFF's 3-/4-/6-membered ring
/// parameters all read a local cycle. It is false for a large one. A 5000-atom
/// macrocycle is locally indistinguishable from a chain — no typifier in this
/// tree can tell them apart — yet closing on it drags all 5000 atoms into a
/// ball that wanted 9, which is exactly what happens the moment a crosslink
/// joins two chains into a loop. Unbounded closure does not merely cost time:
/// it defeats region typing, whose whole premise is bounded work per edit.
///
/// With the bound, the question stops needing a global ring decomposition.
/// A bond lies on a ring of at most `max_ring_size` atoms **iff** its endpoints
/// stay connected within `max_ring_size - 1` hops after it is removed — a
/// bounded local search. Closure is then the flood from the ball along exactly
/// those bonds, so the cost follows the ball and its chemical neighbourhood,
/// not the size of the parent. Fused and spiro systems still arrive whole: the
/// flood crosses every small-ring bond it meets, which is what fuses them.
///
/// `max_ring_size < 3` disables closure (no ring has fewer than three atoms).
/// Groups come back one per ring system touched; atoms on no small ring are
/// absent, and an acyclic molecule yields nothing at all.
pub fn small_ring_closure(
    mol: &Atomistic,
    centers: &[NodeId],
    radius: i64,
    max_ring_size: usize,
) -> Vec<Vec<NodeId>> {
    if max_ring_size < 3 || centers.is_empty() {
        return Vec::new();
    }

    // Seed atoms: the plain radius ball. `extract_ball` re-derives the same set;
    // recomputing it here keeps this function self-contained.
    //
    // Deliberately not `Atomistic::topo_distances`: that keys its BFS by a
    // slot-indexed `SecondaryMap`, so it allocates and drains one slot per atom
    // of the *parent* even when `max_hops` stops the search after nine. A
    // hash-keyed BFS costs the ball, which is what this function promises.
    let mut ball: Vec<NodeId> = Vec::new();
    let mut hops: HashMap<NodeId, i64> = HashMap::new();
    let mut queue: VecDeque<NodeId> = VecDeque::new();
    for &center in centers {
        if hops.insert(center, 0).is_none() {
            ball.push(center);
            queue.push_back(center);
        }
    }
    while let Some(atom) = queue.pop_front() {
        let hop = hops[&atom];
        if hop >= radius {
            continue;
        }
        let neighbors: Vec<NodeId> = mol.neighbor_bonds(atom).map(|(other, _)| other).collect();
        for other in neighbors {
            if let std::collections::hash_map::Entry::Vacant(slot) = hops.entry(other) {
                slot.insert(hop + 1);
                ball.push(other);
                queue.push_back(other);
            }
        }
    }

    // One flood per ring system, so an atom is emitted in exactly one group.
    let mut assigned: HashSet<NodeId> = HashSet::new();
    let mut on_small_ring: HashMap<(NodeId, NodeId), bool> = HashMap::new();
    let mut groups: Vec<Vec<NodeId>> = Vec::new();

    for &start in &ball {
        if !assigned.insert(start) {
            continue;
        }
        let mut system = vec![start];
        let mut frontier = vec![start];
        let mut in_system: HashSet<NodeId> = HashSet::from([start]);
        while let Some(atom) = frontier.pop() {
            let neighbors: Vec<NodeId> = mol.neighbor_bonds(atom).map(|(other, _)| other).collect();
            for other in neighbors {
                if in_system.contains(&other) {
                    continue;
                }
                let key = if atom < other {
                    (atom, other)
                } else {
                    (other, atom)
                };
                let cyclic = match on_small_ring.get(&key) {
                    Some(&known) => known,
                    None => {
                        let known = bond_on_small_ring(mol, atom, other, max_ring_size);
                        on_small_ring.insert(key, known);
                        known
                    }
                };
                if cyclic {
                    in_system.insert(other);
                    assigned.insert(other);
                    system.push(other);
                    frontier.push(other);
                }
            }
        }
        // A lone atom is not a ring system — it is an atom on no small ring.
        if system.len() > 1 {
            groups.push(system);
        }
    }
    groups
}

/// Whether the bond between `a` and `b` lies on a ring of at most
/// `max_ring_size` atoms.
///
/// Bounded BFS for an alternative `a`→`b` route: a route of `L` bonds closes a
/// ring of `L + 1` atoms, so the search stops at `max_ring_size - 1` hops. Cost
/// is the neighbourhood within that many bonds, never the whole molecule.
fn bond_on_small_ring(mol: &Atomistic, a: NodeId, b: NodeId, max_ring_size: usize) -> bool {
    let max_hops = max_ring_size - 1;
    let mut visited: HashSet<NodeId> = HashSet::from([a]);
    let mut queue: VecDeque<(NodeId, usize)> = VecDeque::from([(a, 0usize)]);
    while let Some((atom, hops)) = queue.pop_front() {
        if hops >= max_hops {
            continue;
        }
        for (other, _) in mol.neighbor_bonds(atom) {
            // The bond under test is not a route around itself.
            if atom == a && other == b {
                continue;
            }
            if other == b {
                return true;
            }
            if visited.insert(other) {
                queue.push_back((other, hops + 1));
            }
        }
    }
    false
}

/// Compute the ring information (SSSR / minimum cycle basis) for `mol`.
///
/// This is the one owner of ring perception in the tree. Its SSSR search
/// computes the **Smallest Set of Smallest Rings** (equivalently the minimum
/// cycle basis) over the contiguous `usize` vertex/edge indices of a
/// [`Topology`]; this function:
///
/// 1. projects an [`Atomistic`] onto that index space — atom index `i` is the
///    `i`-th atom of [`Atomistic::atoms`], edge index `i` the `i`-th bond of
///    [`Atomistic::bonds`],
/// 2. runs the SSSR search,
/// 3. lifts the resulting `usize` indices back onto the [`NodeId`] / [`RelationId`]
///    handles chemistry code (aromaticity, SMARTS, MMFF, AM1-BCC, the conformer
///    pipeline) actually holds, as a [`RingSet`].
///
/// Rings come back smallest-first.
pub fn perceive_rings(mol: &Atomistic) -> RingSet {
    // ---- 1. Project onto the core index space ------------------------------
    // Atom index `i` == the `i`-th atom of `mol.atoms()`; ring edges are lifted
    // back onto bonds by their endpoint pair (`bond_map`), never by position.
    let atom_vec: Vec<NodeId> = mol.atoms().map(|(id, _)| id).collect();
    let atom_to_idx: HashMap<NodeId, usize> = atom_vec
        .iter()
        .enumerate()
        .map(|(i, &id)| (id, i))
        .collect();

    let bond_vec: Vec<RelationId> = mol.bonds().map(|(id, _)| id).collect();

    let mut edges: Vec<[usize; 2]> = Vec::with_capacity(bond_vec.len());
    // Fast (NodeId, NodeId) → RelationId lookup, used to lift ring edges back.
    let mut bond_map: HashMap<(NodeId, NodeId), RelationId> = HashMap::new();
    for &bid in &bond_vec {
        let ends = mol.get_bond(bid).expect("bond must exist").nodes;
        let (n0, n1) = (ends[0], ends[1]);
        edges.push([atom_to_idx[&n0], atom_to_idx[&n1]]);
        bond_map.insert((n0, n1), bid);
        bond_map.insert((n1, n0), bid);
    }

    // ---- 2. The one and only SSSR in the tree -------------------------------
    let mut topo = Topology::with_atoms(atom_vec.len());
    topo.add_bonds(&edges);
    let cycles = sssr(&topo);

    // ---- 3. Lift node-index cycles → NodeId rings ---------------------------
    let rings: Vec<Vec<NodeId>> = cycles
        .rings
        .iter()
        .map(|cycle| cycle.iter().map(|&ni| atom_vec[ni]).collect())
        .collect();

    // ---- 4. Build the handle-keyed reverse-lookup maps ----------------------
    // Each ring is a closed path, so its cyclically consecutive atom pairs are
    // exactly its bonds.
    let mut atom_rings: HashMap<NodeId, Vec<usize>> = HashMap::new();
    let mut bond_rings: HashMap<RelationId, Vec<usize>> = HashMap::new();

    for (ri, ring) in rings.iter().enumerate() {
        let n = ring.len();
        for i in 0..n {
            let a = ring[i];
            let b = ring[(i + 1) % n];
            atom_rings.entry(a).or_default().push(ri);
            if let Some(&bid) = bond_map.get(&(a, b)) {
                bond_rings.entry(bid).or_default().push(ri);
            }
        }
    }

    RingSet {
        rings,
        atom_rings,
        bond_rings,
    }
}

/// Atom / bond prop written by [`assign_rings`]: `1` when the atom / bond lies
/// on at least one SSSR ring.
const IS_IN_RING: &str = "is_in_ring";
/// Atom / bond prop written by [`assign_rings`]: the number of SSSR rings the
/// atom / bond belongs to.
const N_RINGS: &str = "n_rings";

/// Perceive rings (SSSR) and write them onto `mol`, in place.
///
/// The [`perceive_rings`] side table, projected as props: every atom and every
/// bond receives `is_in_ring` (`0` / `1`) and `n_rings` — including the acyclic
/// ones, which are explicitly flagged `0` rather than left unset.
///
/// # Errors
///
/// [`MolRsError`] when a flag cannot be written (a column of `mol` holding the
/// key under a foreign dtype); `mol` may then be partly annotated.
///
/// # Examples
///
/// ```
/// use molrs::core::{Atomistic, Element, keys};
/// use molrs::perceive::assign_rings;
/// use molrs::props;
///
/// let mut mol = Atomistic::new();
/// let c = mol.add_atom(props! { keys::ATOMIC_NUMBER => Element::C }).unwrap();
/// assign_rings(&mut mol).unwrap();
/// let atom = mol.get_atom(c).unwrap();
/// assert_eq!(atom.get_int("is_in_ring"), Some(0));
/// ```
pub fn assign_rings(mol: &mut Atomistic) -> Result<(), MolRsError> {
    let info = perceive_rings(mol);
    let atom_ids: Vec<NodeId> = mol.atom_ids().collect();
    for id in atom_ids {
        mol.set_atom(id, IS_IN_RING, i32::from(info.is_atom_in_ring(id)))?;
        mol.set_atom(id, N_RINGS, saturating_i32(info.n_atom_rings(id)))?;
    }
    let bond_ids: Vec<RelationId> = mol.bond_ids().collect();
    for bid in bond_ids {
        mol.set_bond(bid, IS_IN_RING, i32::from(info.is_bond_in_ring(bid)))?;
        mol.set_bond(bid, N_RINGS, saturating_i32(info.n_bond_rings(bid)))?;
    }
    Ok(())
}

/// Narrow a count to the `i32` that [`crate::core::PropValue`] stores,
/// saturating instead of wrapping (counts never approach the bound in
/// practice).
pub(super) fn saturating_i32(n: usize) -> i32 {
    i32::try_from(n).unwrap_or(i32::MAX)
}

// ---------------------------------------------------------------------------
// SSSR over a Topology
// ---------------------------------------------------------------------------

/// Compute the Smallest Set of Smallest Rings (SSSR).
///
/// Returns the rings as atom-index cycles with per-atom and per-bond
/// (edge-index) membership tables.
///
/// # Known limitation: candidate generation, not the basis reduction
///
/// Candidates are the shortest cycle through each *edge* — `E` of them.
/// Horton's algorithm generates one per *(vertex, edge)* pair, `V · E`, and
/// `E` of them do not always span the cycle space: on a dense graph the
/// result can hold fewer than `E - V + C` rings, leaving a genuinely cyclic
/// bond with `is_bond_in_ring` false.
///
/// It does not bite on molecular graphs. Surveyed: cubane, adamantane,
/// norbornane, bicyclo[2.2.2]octane, dodecahedrane, coronene, the porphyrin
/// core, the Petersen graph, the triangular prism and a 3×3 grid all return
/// the full rank (pinned by `sssr_returns_a_full_cycle_basis`).
/// Under-selection needed random graphs far denser than any chemistry —
/// K5, and 61 % of random degree-≤4 graphs, which are nothing like a
/// molecule. Widening to true Horton candidates costs a BFS per
/// (vertex, edge) pair; do it when a real structure fails, not before.
/// Per-edge ring membership of `topo`: `true` for every edge index on an
/// SSSR ring. The index-space query of the same search [`perceive_rings`] runs.
pub(crate) fn ring_bond_mask(topo: &Topology) -> Vec<bool> {
    sssr(topo).bond_ring_mask(topo.n_bonds())
}

fn sssr(topo: &Topology) -> IndexRings {
    let edges = topo.bonds();
    let n_nodes = topo.n_atoms();
    let n_edges = edges.len();
    if n_nodes == 0 || n_edges == 0 {
        return IndexRings::empty();
    }

    // Expected cycle basis size = E - V + C
    let n_comp = topo.n_components();
    let cycle_rank = n_edges as isize - n_nodes as isize + n_comp as isize;
    if cycle_rank <= 0 {
        return IndexRings::empty();
    }
    let cycle_rank = cycle_rank as usize;

    // Generate candidate cycles (Horton-style)
    let adjacency: Vec<Vec<usize>> = (0..n_nodes).map(|i| topo.neighbors(i)).collect();
    let mut candidates: Vec<Vec<usize>> = Vec::new();
    for edge in &edges {
        let (u, v) = (edge[0], edge[1]);
        let skip = if u < v { (u, v) } else { (v, u) };
        if let Some(path) = bfs_skip_edge(&adjacency, u, v, skip) {
            candidates.push(path);
        }
    }
    candidates.sort_by_key(|c| c.len());

    // Edge lookup for cycle-vector construction
    let mut edge_lookup: HashMap<(usize, usize), usize> = HashMap::new();
    for (ei, edge) in edges.iter().enumerate() {
        let (a, b) = (edge[0], edge[1]);
        let key = if a < b { (a, b) } else { (b, a) };
        edge_lookup.insert(key, ei);
    }

    // Select linearly independent cycles via GF(2) Gaussian elimination.
    let mut basis = CycleBasis::over_edges(n_edges);
    let mut selected: Vec<Vec<usize>> = Vec::new();

    for cycle in &candidates {
        if selected.len() >= cycle_rank {
            break;
        }
        let mut support: Vec<usize> = Vec::with_capacity(cycle.len());
        let n = cycle.len();
        for i in 0..n {
            let a = cycle[i];
            let b = cycle[(i + 1) % n];
            let key = if a < b { (a, b) } else { (b, a) };
            if let Some(&ei) = edge_lookup.get(&key) {
                support.push(ei);
            }
        }
        support.sort_unstable();
        if basis.admit(support) {
            selected.push(cycle.clone());
        }
    }

    selected.sort_by_key(Vec::len);

    // Build the per-edge reverse lookup.
    let mut bond_rings: HashMap<usize, Vec<usize>> = HashMap::new();

    for (ri, ring) in selected.iter().enumerate() {
        let n = ring.len();
        for i in 0..n {
            let a = ring[i];
            let b = ring[(i + 1) % n];
            let key = if a < b { (a, b) } else { (b, a) };
            if let Some(&ei) = edge_lookup.get(&key) {
                bond_rings.entry(ei).or_default().push(ri);
            }
        }
    }

    IndexRings {
        rings: selected,
        bond_rings,
    }
}

/// BFS from `start` to `goal`, skipping one specific edge identified by its
/// endpoint pair `skip = (min, max)`.
fn bfs_skip_edge(
    adjacency: &[Vec<usize>],
    start: usize,
    goal: usize,
    skip: (usize, usize),
) -> Option<Vec<usize>> {
    let mut visited = vec![false; adjacency.len()];
    let mut parent: Vec<i64> = vec![-1; adjacency.len()];
    let mut queue = VecDeque::new();

    visited[start] = true;
    queue.push_back(start);

    while let Some(current) = queue.pop_front() {
        if current == goal {
            let mut path = vec![goal];
            let mut node = goal;
            while node != start {
                node = parent[node] as usize;
                path.push(node);
            }
            path.reverse();
            return Some(path);
        }
        for &neighbor in &adjacency[current] {
            let key = if current < neighbor {
                (current, neighbor)
            } else {
                (neighbor, current)
            };
            if key == skip {
                continue;
            }
            if !visited[neighbor] {
                visited[neighbor] = true;
                parent[neighbor] = current as i64;
                queue.push_back(neighbor);
            }
        }
    }
    None
}

// ---------------------------------------------------------------------------
// Index-space rings
// ---------------------------------------------------------------------------

/// The SSSR of a [`Topology`] in its own index space: rings as ordered lists
/// of atom indices forming closed paths, with membership by edge index.
/// [`perceive_rings`] lifts it onto handles as a [`RingSet`].
#[derive(Debug, Clone)]
struct IndexRings {
    rings: Vec<Vec<usize>>,
    bond_rings: HashMap<usize, Vec<usize>>,
}

impl IndexRings {
    fn empty() -> Self {
        Self {
            rings: Vec::new(),
            bond_rings: HashMap::new(),
        }
    }

    /// Per-bond boolean mask: true if the bond is in any ring.
    fn bond_ring_mask(&self, n_bonds: usize) -> Vec<bool> {
        let mut mask = vec![false; n_bonds];
        for &idx in self.bond_rings.keys() {
            if idx < n_bonds {
                mask[idx] = true;
            }
        }
        mask
    }

    /// Whether the atom belongs to any ring.
    #[cfg(test)]
    fn is_atom_in_ring(&self, idx: usize) -> bool {
        self.rings.iter().any(|r| r.contains(&idx))
    }

    /// Whether the bond belongs to any ring.
    #[cfg(test)]
    fn is_bond_in_ring(&self, idx: usize) -> bool {
        self.bond_rings.get(&idx).is_some_and(|v| !v.is_empty())
    }

    /// Number of rings containing this bond.
    #[cfg(test)]
    fn n_bond_rings(&self, idx: usize) -> usize {
        self.bond_rings.get(&idx).map_or(0, Vec::len)
    }

    /// Size (atom count) of every ring, smallest first.
    #[cfg(test)]
    fn ring_sizes(&self) -> Vec<usize> {
        self.rings.iter().map(Vec::len).collect()
    }

    /// Total number of rings detected.
    #[cfg(test)]
    fn n_rings(&self) -> usize {
        self.rings.len()
    }
}

// ---------------------------------------------------------------------------
// GF(2) cycle basis
// ---------------------------------------------------------------------------

/// Row-reduced GF(2) basis of the cycle space, in the edge-indexed coordinates
/// used by the SSSR search.
///
/// A cycle is a **sparse** ascending list of the edge indices it covers; adding
/// two cycles is the symmetric difference of those lists. Reduction is indexed
/// by pivot edge — [`Self::admit`] follows a candidate's own leading edge down
/// through the pivots it actually collides with — so a candidate costs the XORs
/// it really performs.
///
/// A dense basis XOR-scanned whole per candidate costs `O(R · E/64)` each and
/// `O(R² · E/64)` overall. On ring-rich systems that is cubic in the atom
/// count and dominates everything downstream of ring perception (aromaticity,
/// SMARTS, MMFF/UFF/ATD typing): 4000 disjoint benzenes take 4.3 s that way.
/// Here disjoint rings reduce in one step each.
struct CycleBasis {
    /// Reduced basis vectors, each an ascending list of edge indices.
    vectors: Vec<Vec<usize>>,
    /// `pivot_of_edge[e]` = index into [`Self::vectors`] of the vector whose
    /// leading (largest) edge is `e`.
    pivot_of_edge: Vec<Option<usize>>,
}

impl CycleBasis {
    /// An empty basis over an edge space of `n_edges` coordinates.
    fn over_edges(n_edges: usize) -> Self {
        Self {
            vectors: Vec::new(),
            pivot_of_edge: vec![None; n_edges],
        }
    }

    /// Reduce `support` against the basis; extend the basis and return `true`
    /// iff it was linearly independent.
    ///
    /// `support` must be ascending and duplicate-free.
    fn admit(&mut self, mut support: Vec<usize>) -> bool {
        while let Some(&lead) = support.last() {
            match self.pivot_of_edge[lead] {
                Some(vi) => support = symmetric_difference(&support, &self.vectors[vi]),
                None => {
                    self.pivot_of_edge[lead] = Some(self.vectors.len());
                    self.vectors.push(support);
                    return true;
                }
            }
        }
        false
    }
}

/// Symmetric difference of two ascending, duplicate-free index lists — GF(2)
/// addition of the sparse vectors they represent.
fn symmetric_difference(a: &[usize], b: &[usize]) -> Vec<usize> {
    let mut out = Vec::with_capacity(a.len() + b.len());
    let (mut i, mut j) = (0, 0);
    while i < a.len() && j < b.len() {
        match a[i].cmp(&b[j]) {
            std::cmp::Ordering::Less => {
                out.push(a[i]);
                i += 1;
            }
            std::cmp::Ordering::Greater => {
                out.push(b[j]);
                j += 1;
            }
            // Present in both ⇒ cancels over GF(2).
            std::cmp::Ordering::Equal => {
                i += 1;
                j += 1;
            }
        }
    }
    out.extend_from_slice(&a[i..]);
    out.extend_from_slice(&b[j..]);
    out
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    /// `n` atoms joined by `edges`.
    fn bonded(n: usize, edges: &[[usize; 2]]) -> Topology {
        let mut topo = Topology::with_atoms(n);
        topo.add_bonds(edges);
        topo
    }

    #[test]
    fn sssr_linear() {
        let topo = bonded(4, &[[0, 1], [1, 2], [2, 3]]);
        assert_eq!(sssr(&topo).n_rings(), 0);
    }

    #[test]
    fn sssr_single_6ring() {
        // Hexagon: 0-1-2-3-4-5-0
        let topo = bonded(6, &[[0, 1], [1, 2], [2, 3], [3, 4], [4, 5], [5, 0]]);
        let ri = sssr(&topo);
        assert_eq!(ri.n_rings(), 1);
        assert_eq!(ri.ring_sizes(), vec![6]);
        for i in 0..6 {
            assert!(ri.is_atom_in_ring(i));
        }
        for i in 0..topo.n_bonds() {
            assert!(ri.is_bond_in_ring(i));
            assert_eq!(ri.n_bond_rings(i), 1);
        }
        assert_eq!(
            ri.bond_ring_mask(topo.n_bonds()),
            vec![true; topo.n_bonds()]
        );
    }

    #[test]
    fn sssr_naphthalene() {
        // Two fused 6-membered rings
        let topo = bonded(
            10,
            &[
                [0, 1],
                [1, 2],
                [2, 3],
                [3, 4],
                [4, 5],
                [5, 0], // ring A
                [2, 6],
                [6, 7],
                [7, 8],
                [8, 9],
                [9, 3], // ring B
            ],
        );
        let ri = sssr(&topo);
        assert_eq!(ri.n_rings(), 2);
        let mut sizes = ri.ring_sizes();
        sizes.sort();
        assert_eq!(sizes, vec![6, 6]);
        assert!(ri.bond_ring_mask(topo.n_bonds()).into_iter().all(|x| x));
        assert_eq!(ri.n_bond_rings(2), 2);
    }

    #[test]
    fn sssr_empty() {
        let topo = Topology::with_atoms(0);
        assert_eq!(sssr(&topo).n_rings(), 0);
    }

    /// Bridgeless polycyclic graphs, each with its own independent cycle count.
    ///
    /// Every one is 2-edge-connected, so *every* edge lies on a cycle — which
    /// makes both invariants below unconditional, with nothing for a subset
    /// assertion to hide behind. Listing them in one table is what stops the
    /// suite from silently covering only the easy fused-aromatic case.
    fn bridgeless_polycycles() -> Vec<(&'static str, usize, Vec<[usize; 2]>)> {
        vec![
            (
                "K4",
                4,
                vec![[0, 1], [0, 2], [0, 3], [1, 2], [1, 3], [2, 3]],
            ),
            (
                "triangular prism",
                6,
                vec![
                    [0, 1],
                    [1, 2],
                    [2, 0],
                    [3, 4],
                    [4, 5],
                    [5, 3],
                    [0, 3],
                    [1, 4],
                    [2, 5],
                ],
            ),
            (
                "cubane",
                8,
                vec![
                    [0, 1],
                    [1, 2],
                    [2, 3],
                    [3, 0],
                    [4, 5],
                    [5, 6],
                    [6, 7],
                    [7, 4],
                    [0, 4],
                    [1, 5],
                    [2, 6],
                    [3, 7],
                ],
            ),
            (
                "3x3 grid",
                9,
                vec![
                    [0, 1],
                    [1, 2],
                    [3, 4],
                    [4, 5],
                    [6, 7],
                    [7, 8],
                    [0, 3],
                    [3, 6],
                    [1, 4],
                    [4, 7],
                    [2, 5],
                    [5, 8],
                ],
            ),
            (
                "Petersen",
                10,
                vec![
                    [0, 1],
                    [1, 2],
                    [2, 3],
                    [3, 4],
                    [4, 0],
                    [5, 7],
                    [7, 9],
                    [9, 6],
                    [6, 8],
                    [8, 5],
                    [0, 5],
                    [1, 6],
                    [2, 7],
                    [3, 8],
                    [4, 9],
                ],
            ),
            (
                "adamantane",
                10,
                vec![
                    [0, 1],
                    [1, 2],
                    [2, 3],
                    [3, 4],
                    [4, 5],
                    [5, 0],
                    [0, 6],
                    [2, 7],
                    [4, 8],
                    [6, 9],
                    [7, 9],
                    [8, 9],
                ],
            ),
            (
                "two disjoint hexagons",
                12,
                vec![
                    [0, 1],
                    [1, 2],
                    [2, 3],
                    [3, 4],
                    [4, 5],
                    [5, 0],
                    [6, 7],
                    [7, 8],
                    [8, 9],
                    [9, 10],
                    [10, 11],
                    [11, 6],
                ],
            ),
        ]
    }

    #[test]
    fn sssr_returns_a_full_cycle_basis() {
        // SSSR must return exactly `E - V + C` rings. Fewer means a linearly
        // *dependent* candidate was accepted as independent and displaced a
        // real ring from a basis capped at the cycle rank.
        for (name, n_atoms, edges) in bridgeless_polycycles() {
            let topo = bonded(n_atoms, &edges);
            let rank = edges.len() - n_atoms + topo.n_components();
            assert_eq!(
                sssr(&topo).n_rings(),
                rank,
                "{name}: ring count is not the cycle rank"
            );
        }
    }

    #[test]
    fn sssr_covers_every_edge_of_a_bridgeless_graph() {
        // A cycle basis spans the cycle space, so an edge missing from every
        // basis ring would have to lie on no cycle at all — impossible here.
        for (name, n_atoms, edges) in bridgeless_polycycles() {
            let topo = bonded(n_atoms, &edges);
            let mask = sssr(&topo).bond_ring_mask(topo.n_bonds());
            assert_eq!(
                mask.iter().filter(|&&covered| !covered).count(),
                0,
                "{name}: bridgeless graph has an edge in no ring: {mask:?}"
            );
        }
    }

    use crate::core::{Element, Props, keys};
    use crate::props;

    /// A carbon atom: ring perception reads only connectivity, but every
    /// atom of an `Atomistic` carries a species.
    fn carbon() -> Props {
        props! { keys::ATOMIC_NUMBER => Element::C }
    }

    fn cycle(n: usize) -> Atomistic {
        let mut g = Atomistic::new();
        let ids: Vec<NodeId> = (0..n).map(|_| g.add_atom(carbon()).unwrap()).collect();
        for i in 0..n {
            g.add_bond(ids[i], ids[(i + 1) % n], props! {})
                .expect("add cycle bond");
        }
        g
    }

    #[test]
    fn test_single_6ring() {
        let g = cycle(6);
        let ri = perceive_rings(&g);
        assert_eq!(ri.n_rings(), 1);
        assert_eq!(ri.ring_sizes(), vec![6]);
    }

    #[test]
    fn test_linear_no_rings() {
        let mut g = Atomistic::new();
        let ids: Vec<NodeId> = (0..6).map(|_| g.add_atom(carbon()).unwrap()).collect();
        for i in 0..5 {
            g.add_bond(ids[i], ids[i + 1], props! {})
                .expect("add chain bond");
        }
        assert_eq!(perceive_rings(&g).n_rings(), 0);
    }

    #[test]
    fn test_all_atoms_in_6ring() {
        let g = cycle(6);
        let ri = perceive_rings(&g);
        for (id, _) in g.atoms() {
            assert!(ri.is_atom_in_ring(id));
        }
    }

    #[test]
    fn test_all_bonds_in_6ring() {
        let g = cycle(6);
        let ri = perceive_rings(&g);
        for (bid, _) in g.bonds() {
            assert!(ri.is_bond_in_ring(bid));
        }
    }

    #[test]
    fn test_empty_mol() {
        assert_eq!(perceive_rings(&Atomistic::new()).n_rings(), 0);
    }

    #[test]
    fn test_naphthalene() {
        let mut g = Atomistic::new();
        let ids: Vec<NodeId> = (0..10).map(|_| g.add_atom(carbon()).unwrap()).collect();
        // Ring A: 0-1-2-3-4-5-0
        for i in 0..5 {
            g.add_bond(ids[i], ids[i + 1], props! {}).expect("bond");
        }
        g.add_bond(ids[5], ids[0], props! {}).expect("bond");
        // Ring B: 2-3-6-7-8-9-2
        g.add_bond(ids[3], ids[6], props! {}).expect("bond");
        g.add_bond(ids[6], ids[7], props! {}).expect("bond");
        g.add_bond(ids[7], ids[8], props! {}).expect("bond");
        g.add_bond(ids[8], ids[9], props! {}).expect("bond");
        g.add_bond(ids[9], ids[2], props! {}).expect("bond");

        let ri = perceive_rings(&g);
        assert_eq!(ri.n_rings(), 2);
        let mut sizes = ri.ring_sizes();
        sizes.sort_unstable();
        assert_eq!(sizes, vec![6, 6]);
        // Fused system: 10 carbons, not a single 6-ring.
        assert_eq!(ri.max_ring_system_size(), 10);
    }

    #[test]
    fn test_max_ring_system_size_benzene_and_acyclic() {
        assert_eq!(perceive_rings(&cycle(6)).max_ring_system_size(), 6);
        let mut chain = Atomistic::new();
        let ids: Vec<NodeId> = (0..3).map(|_| chain.add_atom(carbon()).unwrap()).collect();
        chain.add_bond(ids[0], ids[1], props! {}).unwrap();
        chain.add_bond(ids[1], ids[2], props! {}).unwrap();
        assert_eq!(perceive_rings(&chain).max_ring_system_size(), 0);
    }

    // -- ring systems -------------------------------------------------------

    /// Two `n`-cycles joined at `shared` consecutive atoms: `shared = 2` is
    /// ortho-fused (naphthalene-like), `shared = 1` is spiro.
    fn joined_cycles(n: usize, shared: usize) -> (Atomistic, Vec<NodeId>) {
        let mut g = Atomistic::new();
        let total = 2 * n - shared;
        let ids: Vec<NodeId> = (0..total).map(|_| g.add_atom(carbon()).unwrap()).collect();
        for i in 0..n {
            g.add_bond(ids[i], ids[(i + 1) % n], props! {})
                .expect("ring A bond");
        }
        // Ring B reuses ids[0..shared] and continues through the fresh atoms.
        let mut ring_b: Vec<NodeId> = (0..shared).map(|i| ids[i]).collect();
        ring_b.extend(ids[n..].iter().copied());
        for i in 0..ring_b.len() {
            g.add_bond(ring_b[i], ring_b[(i + 1) % ring_b.len()], props! {})
                .expect("ring B bond");
        }
        (g, ids)
    }

    #[test]
    fn test_ring_systems_benzene_is_one_system_of_six() {
        let systems = perceive_rings(&cycle(6)).ring_systems();
        assert_eq!(systems.len(), 1);
        assert_eq!(systems[0].len(), 6);
    }

    #[test]
    fn test_ring_systems_fused_pair_is_one_system() {
        let (g, _) = joined_cycles(6, 2);
        let systems = perceive_rings(&g).ring_systems();
        assert_eq!(systems.len(), 1, "naphthalene is one system, not two");
        assert_eq!(systems[0].len(), 10);
    }

    #[test]
    fn test_ring_systems_spiro_pair_is_one_system() {
        // RDKit's GetRingSystems would count these as two with the default
        // includeSpiro=False; for closure they are one, because the shared atom
        // sits in both rings and cannot be present without completing both.
        let (g, _) = joined_cycles(6, 1);
        let systems = perceive_rings(&g).ring_systems();
        assert_eq!(systems.len(), 1);
        assert_eq!(systems[0].len(), 11);
    }

    #[test]
    fn test_ring_systems_disconnected_rings_stay_separate() {
        let mut g = cycle(6);
        let ids: Vec<NodeId> = (0..6).map(|_| g.add_atom(carbon()).unwrap()).collect();
        for i in 0..6 {
            g.add_bond(ids[i], ids[(i + 1) % 6], props! {})
                .expect("second ring");
        }
        let systems = perceive_rings(&g).ring_systems();
        assert_eq!(systems.len(), 2);
        assert!(systems.iter().all(|s| s.len() == 6));
    }

    #[test]
    fn test_ring_systems_acyclic_is_empty() {
        let mut chain = Atomistic::new();
        let ids: Vec<NodeId> = (0..3).map(|_| chain.add_atom(carbon()).unwrap()).collect();
        chain.add_bond(ids[0], ids[1], props! {}).unwrap();
        chain.add_bond(ids[1], ids[2], props! {}).unwrap();
        assert!(perceive_rings(&chain).ring_systems().is_empty());
    }

    #[test]
    fn test_ring_systems_partition_every_ring_atom_exactly_once() {
        let (g, _) = joined_cycles(6, 2);
        let ri = perceive_rings(&g);
        let mut seen: HashSet<NodeId> = HashSet::new();
        for system in ri.ring_systems() {
            for atom in system {
                assert!(seen.insert(atom), "atom in two systems");
            }
        }
        for (id, _) in g.atoms() {
            assert_eq!(ri.is_atom_in_ring(id), seen.contains(&id));
        }
    }

    // -- the reason ring systems exist: extraction must not cut one ---------

    #[test]
    fn test_extraction_closed_on_ring_systems_never_cuts_a_ring() {
        // A methyl hanging off a 6-ring. A radius-2 ball from the methyl carbon
        // reaches three ring atoms — half a ring, which perceives as a different
        // molecule than the one it was cut from.
        // Extraction re-wraps as an Atomistic, so these atoms need an element.
        let mut g = Atomistic::new();
        let ring_ids: Vec<NodeId> = (0..6).map(|_| g.add_atom(carbon()).unwrap()).collect();
        for i in 0..6 {
            g.add_bond(ring_ids[i], ring_ids[(i + 1) % 6], props! {})
                .expect("ring bond");
        }
        let methyl = g.add_atom(carbon()).unwrap();
        g.add_bond(ring_ids[0], methyl, props! {})
            .expect("methyl bond");

        let ri = perceive_rings(&g);
        let plain = g.extract_subgraph(&[methyl], 2, false, &[]).unwrap();
        assert_eq!(plain.graph.n_atoms(), 4, "half the ring, plus the methyl");

        let closed = g
            .extract_subgraph(&[methyl], 2, false, &ri.ring_systems())
            .unwrap();
        assert_eq!(closed.graph.n_atoms(), 7, "whole ring, plus the methyl");
        let selected: HashSet<NodeId> = closed.node_map.keys().copied().collect();
        for system in ri.ring_systems() {
            let inside = system.iter().filter(|a| selected.contains(a)).count();
            assert!(
                inside == 0 || inside == system.len(),
                "ring system split: {inside} of {} atoms",
                system.len()
            );
        }
        // The atoms the radius alone would have reached keep their true distance.
        assert_eq!(closed.hops[&methyl], 0);
        assert_eq!(closed.hops[&ring_ids[0]], 1);
        assert_eq!(closed.hops[&ring_ids[1]], 2);
        // The far side of the ring came in through the closure, beyond the radius.
        assert_eq!(closed.hops[&ring_ids[3]], 4);
    }

    // -----------------------------------------------------------------------
    // small_ring_closure — bounded, local
    // -----------------------------------------------------------------------

    /// A ring of `n` carbons with a methyl on atom 0. Returns `(graph, methyl)`.
    fn methyl_on_ring(n: usize) -> (Atomistic, Vec<NodeId>, NodeId) {
        let mut g = Atomistic::new();
        let ring: Vec<NodeId> = (0..n).map(|_| g.add_atom(carbon()).unwrap()).collect();
        for i in 0..n {
            g.add_bond(ring[i], ring[(i + 1) % n], props! {})
                .expect("ring bond");
        }
        let methyl = g.add_atom(carbon()).unwrap();
        g.add_bond(ring[0], methyl, props! {}).expect("methyl bond");
        (g, ring, methyl)
    }

    #[test]
    fn test_small_ring_closure_returns_the_whole_benzene_from_its_methyl() {
        let (g, ring, methyl) = methyl_on_ring(6);
        let groups = small_ring_closure(&g, &[methyl], 1, 8);
        assert_eq!(groups.len(), 1, "one ring system");
        let closed: HashSet<NodeId> = groups[0].iter().copied().collect();
        assert_eq!(closed, ring.iter().copied().collect::<HashSet<_>>());
        // The exocyclic bond is on no ring, so the methyl is not part of it.
        assert!(!closed.contains(&methyl));
    }

    #[test]
    fn test_small_ring_closure_fuses_naphthalene_into_one_group() {
        let (g, ids) = joined_cycles(6, 2);
        let groups = small_ring_closure(&g, &[ids[0]], 0, 8);
        assert_eq!(groups.len(), 1, "fused rings are one system");
        assert_eq!(groups[0].len(), 10);
    }

    #[test]
    fn test_small_ring_closure_fuses_a_spiro_pair_into_one_group() {
        let (g, ids) = joined_cycles(6, 1);
        let groups = small_ring_closure(&g, &[ids[0]], 0, 8);
        assert_eq!(groups.len(), 1, "a shared atom fuses the two rings");
        assert_eq!(groups[0].len(), 11);
    }

    #[test]
    fn test_small_ring_closure_ignores_a_macrocycle() {
        // The whole point of the bound. A 50-membered ring is topologically a
        // ring and locally a chain; before the bound existed this pulled all 50
        // atoms in to type the handful the ball actually wanted.
        let (g, _ring, methyl) = methyl_on_ring(50);
        assert!(
            small_ring_closure(&g, &[methyl], 2, 8).is_empty(),
            "a 50-ring is not a small ring"
        );
        // Raise the bound past the ring and it comes back — the bound is the
        // only thing deciding this, not some other property of the graph.
        let groups = small_ring_closure(&g, &[methyl], 2, 51);
        assert_eq!(groups.len(), 1);
        assert_eq!(groups[0].len(), 50);
    }

    #[test]
    fn test_small_ring_closure_stops_at_the_bond_between_two_rings() {
        // Biphenyl: the inter-ring bond lies on no small ring, so a ball on one
        // ring must not drag in the other.
        let mut g = Atomistic::new();
        let ring_of = |g: &mut Atomistic| -> Vec<NodeId> {
            let r: Vec<NodeId> = (0..6).map(|_| g.add_atom(carbon()).unwrap()).collect();
            for i in 0..6 {
                g.add_bond(r[i], r[(i + 1) % 6], props! {})
                    .expect("ring bond");
            }
            r
        };
        let left = ring_of(&mut g);
        let right = ring_of(&mut g);
        g.add_bond(left[0], right[0], props! {})
            .expect("biaryl bond");

        let groups = small_ring_closure(&g, &[left[3]], 1, 8);
        assert_eq!(groups.len(), 1, "only the ring the ball is on");
        assert_eq!(groups[0].len(), 6);
    }

    #[test]
    fn test_small_ring_closure_is_empty_for_an_acyclic_molecule() {
        let mut g = Atomistic::new();
        let ids: Vec<NodeId> = (0..8).map(|_| g.add_atom(carbon()).unwrap()).collect();
        for i in 0..7 {
            g.add_bond(ids[i], ids[i + 1], props! {})
                .expect("chain bond");
        }
        assert!(small_ring_closure(&g, &[ids[3]], 3, 8).is_empty());
    }

    #[test]
    fn test_small_ring_closure_is_disabled_below_three_atoms() {
        let (g, _ring, methyl) = methyl_on_ring(6);
        assert!(small_ring_closure(&g, &[methyl], 2, 2).is_empty());
    }

    #[test]
    fn test_small_ring_closure_agrees_with_sssr_ring_systems_on_small_rings() {
        // Where both are defined — every ring within the bound — the local
        // answer must be the global one. Otherwise the bound is not the only
        // difference between them.
        for (name, g, seed) in [
            ("benzene + methyl", methyl_on_ring(6).0, 0usize),
            ("naphthalene", joined_cycles(6, 2).0, 0),
            ("spiro pair", joined_cycles(6, 1).0, 0),
            ("cyclopropane", cycle(3), 0),
        ] {
            let atoms: Vec<NodeId> = g.atoms().map(|(id, _)| id).collect();
            let local = small_ring_closure(&g, &[atoms[seed]], 0, 8);
            let global: Vec<Vec<NodeId>> = perceive_rings(&g)
                .ring_systems()
                .into_iter()
                .filter(|system| system.contains(&atoms[seed]))
                .collect();
            let as_set = |groups: &[Vec<NodeId>]| -> HashSet<NodeId> {
                groups.iter().flatten().copied().collect()
            };
            assert_eq!(
                as_set(&local),
                as_set(&global),
                "{name}: local closure disagrees with the SSSR ring system"
            );
        }
    }

    #[test]
    fn test_extraction_with_small_ring_closure_keeps_the_ball_bounded() {
        // The end-to-end claim: on a small ring the ball still arrives whole,
        // and on a macrocycle it stays the size the radius asked for.
        let (benzene, _, methyl) = methyl_on_ring(6);
        let groups = small_ring_closure(&benzene, &[methyl], 2, 8);
        let closed = benzene
            .extract_subgraph(&[methyl], 2, false, &groups)
            .unwrap();
        assert_eq!(closed.graph.n_atoms(), 7, "whole ring, plus the methyl");

        let (macro_ring, _, methyl) = methyl_on_ring(500);
        let groups = small_ring_closure(&macro_ring, &[methyl], 2, 8);
        let closed = macro_ring
            .extract_subgraph(&[methyl], 2, false, &groups)
            .unwrap();
        assert_eq!(
            closed.graph.n_atoms(),
            4,
            "a macrocycle must not pull its whole loop into the ball"
        );
    }
}
