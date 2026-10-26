//! Graph-based molecular topology.
//!
//! Provides graph-based representation of molecular connectivity with
//! automated detection of angles, dihedrals, and impropers via neighbor
//! traversal.

use std::collections::{HashMap, HashSet, VecDeque};

use crate::core::MolRsError;
use crate::core::schema::block_names::{ATOMS, BONDS};
use crate::core::{Block, Frame, keys};
use crate::op::Idx;

/// Why [`Topology::from_frame`] could not read a frame's bond graph.
///
/// Each case names the block and, where there is one, the row at fault, so a
/// caller can branch on the reason instead of matching message text.
/// Converts into [`MolRsError`] (`MissingBlock` as
/// [`MolRsError::NotFound`], the rest as [`MolRsError::Validation`]) for
/// callers that propagate the crate-wide error.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum TopologyError {
    /// The frame has no `block` block (`atoms`: there is nothing to count).
    MissingBlock {
        /// The absent block.
        block: &'static str,
    },
    /// The `block` block has no row count: it holds no column and was never
    /// sized, so the number of atoms is unknown.
    NoRows {
        /// The block without a row count.
        block: &'static str,
    },
    /// A non-empty relation block lacks one of its endpoint columns (or
    /// carries it as anything but `UInt`).
    MissingEndpoint {
        /// The relation block (`bonds`).
        block: &'static str,
        /// The absent endpoint column (`atomi` / `atomj`).
        column: &'static str,
    },
    /// A relation endpoint is null rather than an atom row.
    NullEndpoint {
        block: &'static str,
        column: &'static str,
        row: usize,
    },
    /// Row `row` of `block` names atom `atom`, outside `0..n_atoms`.
    EndpointOutOfRange {
        /// The relation block (`bonds`).
        block: &'static str,
        /// The offending row of `block`.
        row: usize,
        /// The out-of-range endpoint value.
        atom: usize,
        /// Row count of the `atoms` block.
        n_atoms: usize,
    },
}

impl std::fmt::Display for TopologyError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            TopologyError::MissingBlock { block } => write!(f, "frame has no '{block}' block"),
            TopologyError::NoRows { block } => {
                write!(f, "'{block}' block has no nrows (empty block, no columns)")
            }
            TopologyError::MissingEndpoint { block, column } => {
                write!(f, "'{block}' block is missing 1-D uint column '{column}'")
            }
            TopologyError::NullEndpoint { block, column, row } => {
                write!(f, "'{block}' row {row} has a null '{column}' endpoint")
            }
            TopologyError::EndpointOutOfRange {
                block,
                row,
                atom,
                n_atoms,
            } => write!(
                f,
                "'{block}' row {row} references atom {atom}, outside the frame \
                 (n_atoms = {n_atoms})"
            ),
        }
    }
}

impl std::error::Error for TopologyError {}

impl From<TopologyError> for MolRsError {
    fn from(err: TopologyError) -> Self {
        match err {
            TopologyError::MissingBlock { block } => MolRsError::not_found(block, err.to_string()),
            other => MolRsError::validation(other.to_string()),
        }
    }
}

/// Graph-based molecular topology.
///
/// Holds a native adjacency snapshot where vertices are contiguous atom
/// indices `0..n` and edges are bonds. Angles, dihedrals, and impropers are
/// detected automatically from bond connectivity using neighbor traversal.
#[derive(Debug, Clone)]
pub struct Topology {
    /// Node count.
    n: usize,
    /// `adj[node]` = neighbor node indices, in insertion order.
    adj: Vec<Vec<usize>>,
    /// Edges in insertion order, `[i, j]` as added.
    edges: Vec<[usize; 2]>,
}

impl Topology {
    /// Create a topology with `n` atoms (indices `0..n`) and no bonds — the
    /// one constructor; `with_atoms(0)` is the empty graph. Grow it with
    /// [`add_atoms`](Self::add_atoms) and connect it with
    /// [`add_bonds`](Self::add_bonds).
    pub fn with_atoms(n: usize) -> Self {
        Self {
            n,
            adj: vec![Vec::new(); n],
            edges: Vec::new(),
        }
    }

    /// Read connectivity from a [`Frame`]'s `atoms` / `bonds` blocks.
    ///
    /// Atom count comes from `atoms.n_rows()`. Edges come from uint
    /// [`keys::ATOMI`] / [`keys::ATOMJ`] columns and are added through
    /// [`add_bonds`](Self::add_bonds), so neighbour slices follow bonds-block
    /// insertion order and are **never sorted**. Sorting them would reshape a
    /// consumer's growth tree (molpack picks the first neighbour). A bond
    /// listed twice is one bond.
    ///
    /// Coordinates are not read. A fixture with only an `id` column is valid.
    /// Binders must not invent an `i`/`j` column vocabulary; this reader
    /// recognizes `atomi` / `atomj` only.
    ///
    /// In-range self-loops `(a, a)` are dropped before `add_bonds` (a
    /// `from_frame`-only policy). Endpoints are range-checked first: `(n, n)`
    /// on an `n`-atom frame is a validation error, not a dropped loop.
    ///
    /// A missing or empty `bonds` block is `Ok` with zero edges. A present
    /// non-empty bonds block that lacks `atomi`/`atomj` is a named error, not
    /// a silent empty graph.
    ///
    /// # Errors
    ///
    /// - no `atoms` block → [`TopologyError::MissingBlock`]
    /// - `atoms` with `nrows() == None` → [`TopologyError::NoRows`]
    /// - bond endpoint outside the frame, including `(n, n)` →
    ///   [`TopologyError::EndpointOutOfRange`] naming the `bonds` row
    /// - non-empty `bonds` missing `atomi` or `atomj` →
    ///   [`TopologyError::MissingEndpoint`]
    pub fn from_frame(frame: &Frame) -> Result<Self, TopologyError> {
        let atoms = frame
            .get(ATOMS)
            .ok_or(TopologyError::MissingBlock { block: ATOMS })?;
        let n = atoms
            .n_rows()
            .ok_or(TopologyError::NoRows { block: ATOMS })?;
        let Some(bonds) = frame.get(BONDS) else {
            return Ok(Self::with_atoms(n));
        };
        match bonds.n_rows() {
            None | Some(0) => return Ok(Self::with_atoms(n)),
            Some(_) => {}
        }
        let endpoint = |column: &'static str| {
            if let Some(row) = bonds
                .validity(column)
                .and_then(|mask| mask.iter().position(|valid| !valid))
            {
                return Err(TopologyError::NullEndpoint {
                    block: BONDS,
                    column,
                    row,
                });
            }
            bonds
                .get(column)
                .and_then(|c| c.get::<u64>())
                .filter(|c| c.ndim() == 1)
                .ok_or(TopologyError::MissingEndpoint {
                    block: BONDS,
                    column,
                })
        };
        let (atomi, atomj) = (endpoint(keys::ATOMI)?, endpoint(keys::ATOMJ)?);
        let mut edges = Vec::with_capacity(atomi.len());
        for (row, (&a, &b)) in atomi.iter().zip(atomj.iter()).enumerate() {
            let (a, b) = (a as usize, b as usize);
            if let Some(atom) = [a, b].into_iter().find(|&x| x >= n) {
                return Err(TopologyError::EndpointOutOfRange {
                    block: BONDS,
                    row,
                    atom,
                    n_atoms: n,
                });
            }
            if a == b {
                continue;
            }
            edges.push([a, b]);
        }
        let mut topology = Self::with_atoms(n);
        topology.add_bonds(&edges);
        Ok(topology)
    }

    // -----------------------------------------------------------------------
    // Count accessors
    // -----------------------------------------------------------------------

    /// Number of atoms (vertices).
    pub fn n_atoms(&self) -> usize {
        self.n
    }

    /// Number of bonds (edges).
    pub fn n_bonds(&self) -> usize {
        self.edges.len()
    }

    /// Number of unique angles (i-j-k triplets).
    pub fn n_angles(&self) -> usize {
        self.angles().len()
    }

    /// Number of unique proper dihedrals (i-j-k-l quartets).
    pub fn n_dihedrals(&self) -> usize {
        self.dihedrals().len()
    }

    // -----------------------------------------------------------------------
    // List accessors
    // -----------------------------------------------------------------------

    /// All atom indices.
    pub fn atoms(&self) -> Vec<usize> {
        (0..self.n).collect()
    }

    /// All bond pairs as `[i, j]`.
    pub fn bonds(&self) -> Vec<[usize; 2]> {
        self.edges.clone()
    }

    /// All unique angle triplets `[i, j, k]`, deduplicated (i < k).
    pub fn angles(&self) -> Vec<[usize; 3]> {
        let mut result = Vec::new();
        for j in 0..self.n {
            let neighbors = &self.adj[j];
            for a in 0..neighbors.len() {
                for b in (a + 1)..neighbors.len() {
                    let i = neighbors[a];
                    let k = neighbors[b];
                    if i < k {
                        result.push([i, j, k]);
                    } else {
                        result.push([k, j, i]);
                    }
                }
            }
        }
        result
    }

    /// All unique proper dihedral quartets `[i, j, k, l]`, deduplicated (j < k).
    pub fn dihedrals(&self) -> Vec<[usize; 4]> {
        let mut result = Vec::new();
        for edge in &self.edges {
            let (a, b) = (edge[0], edge[1]);
            // Canonical ordering: j < k
            let (j, k) = if a < b { (a, b) } else { (b, a) };

            let j_neighbors: Vec<usize> = self.adj[j].iter().copied().filter(|&n| n != k).collect();
            let k_neighbors: Vec<usize> = self.adj[k].iter().copied().filter(|&n| n != j).collect();

            for &i in &j_neighbors {
                for &l in &k_neighbors {
                    if i != l {
                        result.push([i, j, k, l]);
                    }
                }
            }
        }
        result
    }

    /// One improper quartet `[centre, i, j, k]` per atom with **exactly three**
    /// neighbours, the peripherals sorted.
    ///
    /// This is the molecular-mechanics reading of an improper: a trivalent
    /// centre carries one out-of-plane term, and a four-neighbour (sp3)
    /// centre carries none.
    ///
    /// Planarity is **not** judged here. Whether a trivalent centre actually
    /// carries an improper is force-field data (GAFF reads the `improper_flag`
    /// column of PARMCHK.DAT), not a property of the graph, so this returns
    /// every trivalent centre and leaves the selection to the layer that has
    /// the table.
    ///
    /// The centre is first — LAMMPS's symmetry atom for its improper styles.
    /// AMBER's slot order, with the centre third (the order whose dihedral is
    /// AMBER's improper angle, and the one `improper periodic` is stored in),
    /// is a re-ordering performed by the force field that wants it.
    pub fn impropers(&self) -> Vec<[usize; 4]> {
        let mut result = Vec::new();
        for center in 0..self.n {
            let mut neighbors: Vec<usize> = self.adj[center].clone();
            if neighbors.len() != 3 {
                continue;
            }
            neighbors.sort_unstable();
            result.push([center, neighbors[0], neighbors[1], neighbors[2]]);
        }
        result
    }

    // -----------------------------------------------------------------------
    // Query accessors
    // -----------------------------------------------------------------------

    /// Neighbor atom indices of atom `idx`.
    pub fn neighbors(&self, idx: usize) -> Vec<usize> {
        self.adj[idx].clone()
    }

    /// Degree (number of bonds) of atom `idx`.
    pub fn degree(&self, idx: usize) -> usize {
        self.adj[idx].len()
    }

    /// Whether atoms `i` and `j` are directly bonded.
    pub fn are_bonded(&self, i: usize, j: usize) -> bool {
        self.adj[i].contains(&j)
    }

    // -----------------------------------------------------------------------
    // Connected components (cluster by bond topology)
    // -----------------------------------------------------------------------

    /// Per-atom connected component labels.
    ///
    /// Returns a `Vec<i64>` of length `n_atoms`, where each element is the
    /// component ID (0-based, contiguous). Isolated atoms each form their own
    /// component.
    pub fn connected_components(&self) -> Vec<i64> {
        let n = self.n;
        let mut labels = vec![-1i64; n];
        let mut label = 0i64;

        for start in 0..n {
            if labels[start] >= 0 {
                continue;
            }
            let mut queue = VecDeque::new();
            queue.push_back(start);
            labels[start] = label;

            while let Some(current) = queue.pop_front() {
                for &ni in &self.adj[current] {
                    if labels[ni] < 0 {
                        labels[ni] = label;
                        queue.push_back(ni);
                    }
                }
            }
            label += 1;
        }
        labels
    }

    /// Single-source shortest-path distances (BFS over the unweighted bond
    /// graph).
    ///
    /// Returns a `Vec<i64>` of length `n_atoms`: the hop count from `source` to
    /// each atom, or `-1` for atoms unreachable from `source` (a different
    /// connected component). `source` itself has distance 0. An out-of-range
    /// `source` yields an all-`-1` vector.
    pub fn distances(&self, source: usize) -> Vec<i64> {
        let mut dist = vec![-1i64; self.n];
        for (atom, hops) in self.hops_within(source, None) {
            dist[atom] = hops as i64;
        }
        dist
    }

    /// Every atom within `cap` bonds of `source` (`None`: its whole connected
    /// component) with its hop count, in BFS order, `source` first at 0; empty
    /// for an out-of-range `source`.
    ///
    /// The one breadth-first walk of index space — [`distances`](Self::distances)
    /// and [`close_pairs`](Self::close_pairs) read it — and a bounded walk
    /// costs its ball, not the molecule.
    pub(crate) fn hops_within(&self, source: usize, cap: Option<usize>) -> Vec<(usize, usize)> {
        if source >= self.n {
            return Vec::new();
        }
        let mut seen: HashSet<usize> = HashSet::from([source]);
        let mut order = vec![(source, 0)];
        let mut next = 0;
        while let Some(&(current, d)) = order.get(next) {
            next += 1;
            if cap.is_some_and(|c| d >= c) {
                continue;
            }
            for &ni in &self.adj[current] {
                if seen.insert(ni) {
                    order.push((ni, d + 1));
                }
            }
        }
        order
    }

    /// The `pairs` block of this bond graph: every 1-2, 1-3 and 1-4 pair
    /// ([`close_pairs`](Self::close_pairs)), `atomi < atomj` in order, with
    /// its `n_bonds` — the non-bonded exceptions a force field's
    /// `special_bonds` weigh, which
    /// [`Atomistic::generate_topology`](crate::core::Atomistic::generate_topology)
    /// writes beside the angles and dihedrals, and a reader of a format that
    /// states none (a LAMMPS data file: LAMMPS derives them the same way)
    /// writes from its bonds.
    pub fn pairs_block(&self) -> Block {
        let mut close: Vec<((usize, usize), usize)> = self.close_pairs(3).into_iter().collect();
        close.sort_unstable();
        let (mut ai, mut aj, mut hops) = (Vec::new(), Vec::new(), Vec::new());
        for ((a, b), d) in close {
            ai.push(a as Idx);
            aj.push(b as Idx);
            hops.push(d as Idx);
        }
        let mut block = Block::new();
        for (key, column) in [(keys::ATOMI, ai), (keys::ATOMJ, aj), (keys::N_BONDS, hops)] {
            block
                .insert(key, ndarray::Array1::from_vec(column).into_dyn(), None)
                .expect("a fresh block takes columns of one length");
        }
        block
    }

    /// Every pair `(i, j)`, `i < j`, at most `max_hops` bonds apart, with its
    /// bond distance — the **shortest** path, so the diagonal of a four-ring
    /// is a 1-3 pair, not a 1-4 one.
    ///
    /// The one answer to "is this pair 1-2, 1-3 or 1-4": the special pairs
    /// ([`pairs_block`](Self::pairs_block), `Atomistic::generate_topology`)
    /// and the readers' rows read bond-graph distance from here, so no two of
    /// them can classify a pair differently.
    pub fn close_pairs(&self, max_hops: usize) -> HashMap<(usize, usize), usize> {
        let mut out = HashMap::new();
        for i in 0..self.n {
            for (j, d) in self.hops_within(i, Some(max_hops)) {
                if j > i {
                    out.insert((i, j), d);
                }
            }
        }
        out
    }

    /// Number of connected components.
    pub fn n_components(&self) -> usize {
        self.connected_components()
            .iter()
            .max()
            .map_or(0, |&m| (m + 1) as usize)
    }

    // -----------------------------------------------------------------------
    // Atom operations
    // -----------------------------------------------------------------------

    /// Append `n` unbonded atoms, indexed after the existing ones.
    pub fn add_atoms(&mut self, n: usize) {
        self.adj.resize_with(self.adj.len() + n, Vec::new);
        self.n += n;
    }

    /// Delete atom `idx` and every bond incident to it.
    ///
    /// Swap-remove semantics: the last atom moves into slot `idx`, so the
    /// index of that one atom changes; every other index is kept. To delete
    /// several atoms at once, use [`del_atoms`](Self::del_atoms) — deleting
    /// them one by one would make the later indices refer to relabelled atoms.
    ///
    /// # Panics
    ///
    /// Panics when `idx` is not an atom index (`>= n_atoms()`).
    pub fn del_atom(&mut self, idx: usize) {
        assert!(
            idx < self.n,
            "Topology::del_atom: atom {idx} out of range (n_atoms = {})",
            self.n
        );
        self.swap_remove_atom(idx);
    }

    /// Delete every atom of `idxs` (indices of the topology *before* the
    /// call) and every bond incident to any of them. Duplicates name one atom.
    ///
    /// The atoms are swap-removed in descending index order, so each index
    /// still names the atom it named on entry when its turn comes; the
    /// surviving atoms that moved are the trailing ones, each into the slot of
    /// a deleted atom. Nothing is changed when an index is out of range.
    ///
    /// # Panics
    ///
    /// Panics when any index is not an atom index (`>= n_atoms()`).
    pub fn del_atoms(&mut self, idxs: &[usize]) {
        let mut order = idxs.to_vec();
        order.sort_unstable_by(|a, b| b.cmp(a));
        order.dedup();
        if let Some(&max) = order.first() {
            assert!(
                max < self.n,
                "Topology::del_atoms: atom {max} out of range (n_atoms = {})",
                self.n
            );
        }
        for idx in order {
            self.swap_remove_atom(idx);
        }
    }

    /// Remove in-range atom `idx`: drop its bonds, move the last atom into
    /// its slot.
    fn swap_remove_atom(&mut self, idx: usize) {
        let last = self.n - 1;

        // Remove edges incident to `idx` and drop `idx` from every adj list.
        self.edges.retain(|e| e[0] != idx && e[1] != idx);
        for list in &mut self.adj {
            list.retain(|&x| x != idx);
        }

        // Relabel the last node into `idx` (swap-remove semantics).
        if idx != last {
            for e in &mut self.edges {
                if e[0] == last {
                    e[0] = idx;
                }
                if e[1] == last {
                    e[1] = idx;
                }
            }
            for list in &mut self.adj {
                for x in list.iter_mut() {
                    if *x == last {
                        *x = idx;
                    }
                }
            }
            self.adj.swap(idx, last);
        }

        self.adj.pop();
        self.n -= 1;
    }

    // -----------------------------------------------------------------------
    // Bond operations
    // -----------------------------------------------------------------------

    /// Add every bond `[i, j]` of `pairs`, in order, skipping any already
    /// present (either orientation) — one `O(existing + pairs)` pass. Edge
    /// indices follow first occurrence, so a duplicate-free list keeps its
    /// positions. One bond is `add_bonds(&[[i, j]])`.
    ///
    /// # Panics
    ///
    /// Panics when an endpoint is not an atom index (`>= n_atoms()`).
    pub fn add_bonds(&mut self, pairs: &[[usize; 2]]) {
        let key = |a: usize, b: usize| (a.min(b), a.max(b));
        let mut present: HashSet<(usize, usize)> =
            self.edges.iter().map(|&[a, b]| key(a, b)).collect();
        for &[i, j] in pairs {
            if present.insert(key(i, j)) {
                self.push_bond(i, j);
            }
        }
    }

    /// Record the bond `i–j` (no duplicate check).
    fn push_bond(&mut self, i: usize, j: usize) {
        self.edges.push([i, j]);
        self.adj[i].push(j);
        self.adj[j].push(i);
    }

    /// Delete bond `idx` (an index into [`bonds`](Self::bonds)).
    ///
    /// Swap-remove semantics: the last bond moves into slot `idx`. To delete
    /// several bonds at once, use [`del_bonds`](Self::del_bonds).
    ///
    /// # Panics
    ///
    /// Panics when `idx` is not a bond index (`>= n_bonds()`).
    pub fn del_bond(&mut self, idx: usize) {
        assert!(
            idx < self.edges.len(),
            "Topology::del_bond: bond {idx} out of range (n_bonds = {})",
            self.edges.len()
        );
        self.swap_remove_bond(idx);
    }

    /// Delete every bond of `idxs` (indices into [`bonds`](Self::bonds)
    /// *before* the call). Duplicates name one bond.
    ///
    /// The bonds are swap-removed in descending index order, so each index
    /// still names the bond it named on entry when its turn comes. Nothing is
    /// changed when an index is out of range.
    ///
    /// # Panics
    ///
    /// Panics when any index is not a bond index (`>= n_bonds()`).
    pub fn del_bonds(&mut self, idxs: &[usize]) {
        let mut order = idxs.to_vec();
        order.sort_unstable_by(|a, b| b.cmp(a));
        order.dedup();
        if let Some(&max) = order.first() {
            assert!(
                max < self.edges.len(),
                "Topology::del_bonds: bond {max} out of range (n_bonds = {})",
                self.edges.len()
            );
        }
        for idx in order {
            self.swap_remove_bond(idx);
        }
    }

    /// Remove in-range bond `idx`, moving the last bond into its slot.
    fn swap_remove_bond(&mut self, idx: usize) {
        let [a, b] = self.edges[idx];
        // Remove one instance of the bond from each endpoint's adjacency list.
        if let Some(pos) = self.adj[a].iter().position(|&x| x == b) {
            self.adj[a].remove(pos);
        }
        if let Some(pos) = self.adj[b].iter().position(|&x| x == a) {
            self.adj[b].remove(pos);
        }
        self.edges.swap_remove(idx);
    }
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    /// `n` atoms joined by `edges`.
    fn bonded(n: usize, edges: &[[usize; 2]]) -> Topology {
        let mut topo = Topology::with_atoms(n);
        topo.add_bonds(edges);
        topo
    }

    /// A four-ring with a two-atom tail: every pair within three bonds, in
    /// order, at its shortest distance — the ring's diagonals are 1-3, and
    /// the tail's far end is four bonds from the ring atom across.
    #[test]
    fn pairs_block_is_every_close_pair_at_its_shortest_distance() {
        let topo = bonded(6, &[[0, 1], [1, 2], [2, 3], [3, 0], [3, 4], [4, 5]]);
        let block = topo.pairs_block();
        let col = |k: &str| block.get(k).unwrap().get::<Idx>().unwrap().to_owned();
        let (i, j, n) = (col("atomi"), col("atomj"), col("n_bonds"));
        let rows: Vec<(Idx, Idx, Idx)> = (0..i.len()).map(|r| (i[r], j[r], n[r])).collect();
        assert_eq!(
            rows,
            [
                (0, 1, 1),
                (0, 2, 2),
                (0, 3, 1),
                (0, 4, 2),
                (0, 5, 3),
                (1, 2, 1),
                (1, 3, 2),
                (1, 4, 3),
                (2, 3, 1),
                (2, 4, 2),
                (2, 5, 3),
                (3, 4, 1),
                (3, 5, 2),
                (4, 5, 1),
            ]
        );
    }

    use super::*;

    #[test]
    fn test_empty_topology() {
        let topo = Topology::with_atoms(0);
        assert_eq!(topo.n_atoms(), 0);
        assert_eq!(topo.n_bonds(), 0);
    }

    #[test]
    fn test_distances_path_and_disconnected() {
        // Path 0-1-2-3 plus an isolated atom 4.
        let topo = bonded(5, &[[0, 1], [1, 2], [2, 3]]);
        assert_eq!(topo.distances(0), vec![0, 1, 2, 3, -1]);
        assert_eq!(topo.distances(3), vec![3, 2, 1, 0, -1]);
        // Isolated atom: only itself reachable.
        assert_eq!(topo.distances(4), vec![-1, -1, -1, -1, 0]);
        // Out-of-range source -> all unreachable.
        assert_eq!(topo.distances(9), vec![-1; 5]);
    }

    #[test]
    fn test_with_atoms() {
        let topo = Topology::with_atoms(5);
        assert_eq!(topo.n_atoms(), 5);
        assert_eq!(topo.n_bonds(), 0);
    }

    #[test]
    fn add_atoms_appends_unbonded_atoms_after_the_existing_ones() {
        let mut topo = bonded(2, &[[0, 1]]);
        topo.add_atoms(1);
        assert_eq!(topo.n_atoms(), 3);
        topo.add_atoms(3);
        assert_eq!(topo.n_atoms(), 6);
        topo.add_atoms(0);
        assert_eq!(topo.n_atoms(), 6);
        assert_eq!(topo.n_bonds(), 1);
        // The new atoms are addressable and bond like the original ones.
        topo.add_bonds(&[[1, 5]]);
        assert_eq!(topo.neighbors(5), vec![1]);
        assert_eq!(topo.degree(2), 0);
    }

    #[test]
    fn test_del_atom() {
        let mut topo = Topology::with_atoms(3);
        topo.del_atom(1);
        assert_eq!(topo.n_atoms(), 2);
    }

    #[test]
    fn add_bonds_adds_a_repeated_bond_once() {
        let mut topo = Topology::with_atoms(3);
        topo.add_bonds(&[[0, 1]]);
        assert_eq!(topo.n_bonds(), 1);

        // Adding same bond again should not create duplicate
        topo.add_bonds(&[[0, 1]]);
        assert_eq!(topo.n_bonds(), 1);
    }

    #[test]
    fn test_add_bonds() {
        let mut topo = Topology::with_atoms(4);
        topo.add_bonds(&[[0, 1], [1, 2], [2, 3]]);
        assert_eq!(topo.n_bonds(), 3);
    }

    /// A bond listed twice — in either orientation, in one batch or against
    /// one already present — is one bond; first occurrences keep their order.
    #[test]
    fn add_bonds_skips_duplicates() {
        let mut topo = Topology::with_atoms(4);
        topo.add_bonds(&[[2, 3]]);
        topo.add_bonds(&[[0, 1], [1, 0], [1, 2], [3, 2], [0, 1]]);
        assert_eq!(topo.bonds(), &[[2, 3], [0, 1], [1, 2]]);
        assert_eq!(topo.degree(1), 2);
        assert_eq!(topo.neighbors(2), vec![3, 1]);
    }

    #[test]
    fn test_del_bond() {
        let mut topo = Topology::with_atoms(3);
        topo.add_bonds(&[[0, 1], [1, 2]]);
        assert_eq!(topo.n_bonds(), 2);
        topo.del_bond(0);
        assert_eq!(topo.n_bonds(), 1);
    }

    #[test]
    fn test_bonds_list() {
        let mut topo = Topology::with_atoms(3);
        topo.add_bonds(&[[0, 1], [1, 2]]);
        let bonds = topo.bonds();
        assert_eq!(bonds.len(), 2);
    }

    #[test]
    fn test_angles_3atom_chain() {
        // 0 - 1 - 2
        let mut topo = Topology::with_atoms(3);
        topo.add_bonds(&[[0, 1], [1, 2]]);
        assert_eq!(topo.n_angles(), 1);
        let angles = topo.angles();
        assert_eq!(angles.len(), 1);
        assert_eq!(angles[0], [0, 1, 2]);
    }

    #[test]
    fn test_dihedrals_4atom_chain() {
        // 0 - 1 - 2 - 3
        let mut topo = Topology::with_atoms(4);
        topo.add_bonds(&[[0, 1], [1, 2], [2, 3]]);
        assert_eq!(topo.n_dihedrals(), 1);
        let dihedrals = topo.dihedrals();
        assert_eq!(dihedrals.len(), 1);
        assert_eq!(dihedrals[0], [0, 1, 2, 3]);
    }

    #[test]
    fn test_impropers_star() {
        // Central atom 0 bonded to 1, 2, 3
        let mut topo = Topology::with_atoms(4);
        topo.add_bonds(&[[0, 1], [0, 2], [0, 3]]);
        let impropers = topo.impropers();
        // One unique improper with center 0
        assert_eq!(impropers.len(), 1);
        assert_eq!(impropers[0][0], 0);
    }

    #[test]
    fn test_methane_ch4() {
        let mut topo = Topology::with_atoms(5);
        topo.add_bonds(&[[0, 1], [0, 2], [0, 3], [0, 4]]);

        assert_eq!(topo.n_atoms(), 5);
        assert_eq!(topo.n_bonds(), 4);
        assert_eq!(topo.n_angles(), 6);
        assert_eq!(topo.n_dihedrals(), 0);
        assert!(topo.impropers().is_empty(), "an sp3 centre has no improper");
    }

    #[test]
    fn impropers_is_one_per_three_neighbour_centre() {
        // Formaldehyde-shaped: centre 0 bonded to 1, 2, 3.
        let topo = bonded(4, &[[0, 1], [0, 2], [0, 3]]);
        assert_eq!(topo.impropers(), vec![[0, 1, 2, 3]]);
    }

    #[test]
    fn impropers_skips_a_four_neighbour_centre() {
        // Methane-shaped: an sp3 centre carries no out-of-plane term.
        let topo = bonded(5, &[[0, 1], [0, 2], [0, 3], [0, 4]]);
        assert!(topo.impropers().is_empty());
    }

    #[test]
    fn impropers_puts_the_centre_first_and_sorts_the_legs() {
        let topo = bonded(4, &[[0, 3], [0, 1], [0, 2]]);
        let quartets = topo.impropers();
        assert_eq!(quartets.len(), 1);
        assert_eq!(quartets[0][0], 0, "centre first");
        assert_eq!(&quartets[0][1..], &[1, 2, 3], "legs sorted");
    }

    #[test]
    fn test_ethane_c2h6() {
        let mut topo = Topology::with_atoms(8);
        topo.add_bonds(&[[0, 1], [0, 2], [0, 3], [0, 4], [1, 5], [1, 6], [1, 7]]);

        assert_eq!(topo.n_atoms(), 8);
        assert_eq!(topo.n_bonds(), 7);
        assert_eq!(topo.n_angles(), 12);
        assert_eq!(topo.n_dihedrals(), 9);
    }

    #[test]
    fn test_add_bonds_builds_a_chain() {
        let topo = bonded(4, &[[0, 1], [1, 2], [2, 3]]);
        assert_eq!(topo.n_atoms(), 4);
        assert_eq!(topo.n_bonds(), 3);
        assert_eq!(topo.n_angles(), 2);
        assert_eq!(topo.n_dihedrals(), 1);
    }

    #[test]
    fn test_neighbors() {
        let topo = bonded(4, &[[0, 1], [1, 2], [2, 3]]);
        let mut n = topo.neighbors(1);
        n.sort();
        assert_eq!(n, vec![0, 2]);
    }

    #[test]
    fn test_degree() {
        let topo = bonded(4, &[[0, 1], [1, 2], [2, 3]]);
        assert_eq!(topo.degree(0), 1);
        assert_eq!(topo.degree(1), 2);
    }

    #[test]
    fn test_are_bonded() {
        let topo = bonded(3, &[[0, 1], [1, 2]]);
        assert!(topo.are_bonded(0, 1));
        assert!(!topo.are_bonded(0, 2));
    }

    #[test]
    fn test_connected_components_single() {
        let topo = bonded(3, &[[0, 1], [1, 2]]);
        let cc = topo.connected_components();
        assert_eq!(cc.len(), 3);
        assert_eq!(cc[0], cc[1]);
        assert_eq!(cc[1], cc[2]);
        assert_eq!(topo.n_components(), 1);
    }

    #[test]
    fn test_connected_components_two() {
        let topo = bonded(4, &[[0, 1], [2, 3]]);
        let cc = topo.connected_components();
        assert_eq!(cc[0], cc[1]);
        assert_eq!(cc[2], cc[3]);
        assert_ne!(cc[0], cc[2]);
        assert_eq!(topo.n_components(), 2);
    }

    #[test]
    fn test_connected_components_isolated() {
        let topo = Topology::with_atoms(3); // no bonds
        assert_eq!(topo.n_components(), 3);
        let cc = topo.connected_components();
        assert_ne!(cc[0], cc[1]);
        assert_ne!(cc[1], cc[2]);
    }

    // -----------------------------------------------------------------------
    // Edge-case parity tests (native adjacency rewrite)
    // -----------------------------------------------------------------------

    #[test]
    fn test_empty_graph_enumerations() {
        // Empty graph: no angles/dihedrals/impropers, no panic.
        let topo = Topology::with_atoms(0);
        assert!(topo.angles().is_empty());
        assert!(topo.dihedrals().is_empty());
        assert!(topo.impropers().is_empty());
        assert_eq!(topo.n_components(), 0);
    }

    #[test]
    fn test_single_edge_graph() {
        // A single bond 0-1: one angle path is impossible (degree 1 each), no
        // dihedral, no improper; two connected atoms.
        let topo = bonded(2, &[[0, 1]]);
        assert_eq!(topo.n_atoms(), 2);
        assert_eq!(topo.n_bonds(), 1);
        assert!(topo.angles().is_empty());
        assert!(topo.dihedrals().is_empty());
        assert!(topo.impropers().is_empty());
        assert_eq!(topo.n_components(), 1);
        assert_eq!(topo.distances(0), vec![0, 1]);
        assert_eq!(topo.distances(1), vec![1, 0]);
    }

    #[test]
    fn test_disconnected_distances_multiple_sources() {
        // Two components: {0-1-2} and {3-4}.
        let topo = bonded(5, &[[0, 1], [1, 2], [3, 4]]);
        assert_eq!(topo.n_components(), 2);
        // Source in first component never reaches second.
        assert_eq!(topo.distances(0), vec![0, 1, 2, -1, -1]);
        assert_eq!(topo.distances(2), vec![2, 1, 0, -1, -1]);
        // Source in second component never reaches first.
        assert_eq!(topo.distances(3), vec![-1, -1, -1, 0, 1]);
        assert_eq!(topo.distances(4), vec![-1, -1, -1, 1, 0]);
    }

    #[test]
    fn test_del_atom_swap_remove_relabels() {
        // 0-1-2-3 chain; deleting atom 1 swaps node 3 into slot 1.
        let mut topo = bonded(4, &[[0, 1], [1, 2], [2, 3]]);
        topo.del_atom(1);
        assert_eq!(topo.n_atoms(), 3);
        // Edge [0,1] and [1,2] are removed; edge [2,3] survives but relabeled
        // (old node 3 -> slot 1). So node 2 and new node 1 stay bonded.
        assert_eq!(topo.n_bonds(), 1);
        assert!(topo.are_bonded(2, 1));
    }

    #[test]
    fn test_del_bond_swap_remove() {
        let mut topo = Topology::with_atoms(4);
        topo.add_bonds(&[[0, 1], [1, 2], [2, 3]]);
        assert_eq!(topo.n_bonds(), 3);
        topo.del_bond(0); // remove edge [0,1]; [2,3] swaps into slot 0
        assert_eq!(topo.n_bonds(), 2);
        assert!(!topo.are_bonded(0, 1));
        assert!(topo.are_bonded(1, 2));
        assert!(topo.are_bonded(2, 3));
    }

    /// The batch deletes exactly the atoms named on entry, whatever order
    /// they are listed in — one-by-one `del_atom` would not (its swap moves
    /// a later-listed atom).
    #[test]
    fn del_atoms_removes_the_atoms_named_on_entry() {
        let mut topo = bonded(5, &[[0, 1], [1, 2], [2, 3], [3, 4]]);
        topo.del_atoms(&[1, 4, 1]);
        // Survivors: old 0, 2, 3 → only bond 2-3 survives.
        assert_eq!(topo.n_atoms(), 3);
        assert_eq!(topo.n_bonds(), 1);
        // Descending swap-remove: delete 4 (last, no move), then 1 (old 3
        // moves into slot 1). So old 2 is slot 2, old 3 is slot 1.
        assert!(topo.are_bonded(1, 2));
        assert_eq!(topo.degree(0), 0);
    }

    #[test]
    #[should_panic(expected = "out of range")]
    fn del_atoms_refuses_an_out_of_range_index() {
        let mut topo = Topology::with_atoms(3);
        topo.del_atoms(&[0, 3]);
    }

    #[test]
    #[should_panic(expected = "out of range")]
    fn del_atom_refuses_an_out_of_range_index() {
        let mut topo = Topology::with_atoms(3);
        topo.del_atom(3);
    }

    #[test]
    fn del_bonds_removes_the_bonds_named_on_entry() {
        let mut topo = Topology::with_atoms(4);
        topo.add_bonds(&[[0, 1], [1, 2], [2, 3]]);
        topo.del_bonds(&[2, 0]);
        assert_eq!(topo.bonds(), vec![[1, 2]]);
        assert_eq!(topo.neighbors(1), vec![2]);
        assert_eq!(topo.degree(0), 0);
        assert_eq!(topo.degree(3), 0);
    }

    #[test]
    #[should_panic(expected = "out of range")]
    fn del_bonds_refuses_an_out_of_range_index() {
        let mut topo = Topology::with_atoms(3);
        topo.add_bonds(&[[0, 1]]);
        topo.del_bonds(&[1]);
    }

    use crate::core::MolRsError;
    use crate::core::{Block, Frame};
    use ndarray::Array1;

    fn atoms_id_only(n: usize) -> Block {
        let mut atoms = Block::new();
        atoms
            .insert(
                "id",
                Array1::from_vec((0u64..n as u64).collect::<Vec<_>>()).into_dyn(),
                None,
            )
            .unwrap();
        atoms
    }

    fn bonds_pairs(pairs: &[[u64; 2]]) -> Block {
        let mut bonds = Block::new();
        bonds
            .insert(
                "atomi",
                Array1::from_vec(pairs.iter().map(|p| p[0]).collect()).into_dyn(),
                None,
            )
            .unwrap();
        bonds
            .insert(
                "atomj",
                Array1::from_vec(pairs.iter().map(|p| p[1]).collect()).into_dyn(),
                None,
            )
            .unwrap();
        bonds
    }

    fn frame_from_parts(atoms: Block, bonds: Option<Block>) -> Frame {
        let mut frame = Frame::new();
        frame.insert("atoms", atoms);
        if let Some(b) = bonds {
            frame.insert("bonds", b);
        }
        frame
    }

    fn c12_chain_pairs() -> Vec<[u64; 2]> {
        (0..11).map(|i| [i, i + 1]).collect()
    }

    #[test]
    fn from_frame_c12_chain_counts_atoms_and_bonds_without_coordinates() {
        let frame = frame_from_parts(atoms_id_only(12), Some(bonds_pairs(&c12_chain_pairs())));
        let topo = Topology::from_frame(&frame).unwrap();
        assert_eq!(topo.n_atoms(), 12);
        assert_eq!(topo.n_bonds(), 11);
    }

    #[test]
    fn from_frame_neighbor_order_follows_bonds_block_insertion() {
        let mut pairs = vec![[5u64, 6], [4, 5]];
        for i in 0..11u64 {
            if !matches!((i, i + 1), (5, 6) | (4, 5)) {
                pairs.push([i, i + 1]);
            }
        }
        let topo = Topology::from_frame(&frame_from_parts(
            atoms_id_only(12),
            Some(bonds_pairs(&pairs)),
        ))
        .unwrap();
        assert_eq!(topo.neighbors(5), vec![6, 4]);
        let topo = Topology::from_frame(&frame_from_parts(
            atoms_id_only(12),
            Some(bonds_pairs(&c12_chain_pairs())),
        ))
        .unwrap();
        assert_eq!(topo.neighbors(5), vec![4, 6]);
    }

    #[test]
    fn from_frame_missing_atoms_block_is_not_found() {
        let mut frame = Frame::new();
        frame.insert("bonds", bonds_pairs(&[[0, 1]]));
        assert_eq!(
            Topology::from_frame(&frame).unwrap_err(),
            TopologyError::MissingBlock { block: "atoms" }
        );
        // Propagated as the crate-wide error it stays NotFound.
        let err: MolRsError = Topology::from_frame(&frame).unwrap_err().into();
        assert!(matches!(
            err,
            MolRsError::NotFound {
                entity: "atoms",
                ..
            }
        ));
    }

    #[test]
    fn from_frame_atoms_without_nrows_is_validation() {
        let mut frame = Frame::new();
        frame.insert("atoms", Block::new());
        assert_eq!(
            Topology::from_frame(&frame).unwrap_err(),
            TopologyError::NoRows { block: "atoms" }
        );
    }

    #[test]
    fn from_frame_out_of_range_bond_is_validation() {
        assert_eq!(
            Topology::from_frame(&frame_from_parts(
                atoms_id_only(12),
                Some(bonds_pairs(&[[0, 1], [0, 12]])),
            ))
            .unwrap_err(),
            TopologyError::EndpointOutOfRange {
                block: "bonds",
                row: 1,
                atom: 12,
                n_atoms: 12,
            }
        );
    }

    #[test]
    fn from_frame_self_loop_at_n_is_validation() {
        assert!(matches!(
            Topology::from_frame(&frame_from_parts(
                atoms_id_only(12),
                Some(bonds_pairs(&[[12, 12]]))
            )),
            Err(TopologyError::EndpointOutOfRange {
                row: 0,
                atom: 12,
                ..
            })
        ));
    }

    #[test]
    fn from_frame_in_range_self_loop_is_dropped() {
        let topo = Topology::from_frame(&frame_from_parts(
            atoms_id_only(3),
            Some(bonds_pairs(&[[1, 1], [0, 1]])),
        ))
        .unwrap();
        assert_eq!(topo.n_bonds(), 1);
    }

    #[test]
    fn from_frame_missing_bonds_block_is_ok_zero_edges() {
        let topo = Topology::from_frame(&frame_from_parts(atoms_id_only(3), None)).unwrap();
        assert_eq!((topo.n_atoms(), topo.n_bonds()), (3, 0));
    }

    #[test]
    fn from_frame_empty_bonds_block_is_ok_zero_edges() {
        let mut frame = Frame::new();
        frame.insert("atoms", atoms_id_only(3));
        frame.insert("bonds", Block::new());
        assert_eq!(Topology::from_frame(&frame).unwrap().n_bonds(), 0);
    }

    #[test]
    fn from_frame_bonds_with_i_j_columns_is_validation() {
        let mut bonds = Block::new();
        bonds
            .insert("i", Array1::from_vec(vec![0u64]).into_dyn(), None)
            .unwrap();
        bonds
            .insert("j", Array1::from_vec(vec![1u64]).into_dyn(), None)
            .unwrap();
        assert_eq!(
            Topology::from_frame(&frame_from_parts(atoms_id_only(2), Some(bonds))).unwrap_err(),
            TopologyError::MissingEndpoint {
                block: "bonds",
                column: "atomi",
            }
        );
    }

    #[test]
    fn from_frame_isolated_atom_is_own_component() {
        let topo = Topology::from_frame(&frame_from_parts(
            atoms_id_only(3),
            Some(bonds_pairs(&[[0, 1]])),
        ))
        .unwrap();
        assert_eq!(topo.n_components(), 2);
    }

    #[test]
    fn from_frame_branched_star() {
        let topo = Topology::from_frame(&frame_from_parts(
            atoms_id_only(4),
            Some(bonds_pairs(&[[0, 1], [0, 2], [0, 3]])),
        ))
        .unwrap();
        assert_eq!(topo.neighbors(0), vec![1, 2, 3]);
    }

    #[test]
    fn from_frame_six_ring_plus_tail() {
        let pairs = [[0u64, 1], [1, 2], [2, 3], [3, 4], [4, 5], [5, 0], [5, 6]];
        let topo = Topology::from_frame(&frame_from_parts(
            atoms_id_only(7),
            Some(bonds_pairs(&pairs)),
        ))
        .unwrap();
        assert_eq!(
            (topo.n_atoms(), topo.n_bonds(), topo.n_components()),
            (7, 7, 1)
        );
    }
}
