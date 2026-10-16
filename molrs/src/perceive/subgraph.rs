use std::collections::{HashMap, HashSet};

use crate::core::CoarseGrain;
use crate::core::graph_hash::{GraphView, adjacency_map, feasible};
use crate::core::keys;
use crate::core::{Graph, GraphAccess, NodeId};

/// Empty slot in a partial map.
const UNMAPPED: usize = usize::MAX;

/// A labelled snapshot of one graph, as a pattern or as a target: nodes in
/// row order, `bead_type` labels and unlabelled adjacency.
#[derive(Debug, Clone)]
struct MatchGraph {
    /// Node handles in row order.
    nodes: Vec<NodeId>,
    /// `bead_type` per node row (`""` when absent).
    labels: Vec<String>,
    /// Neighbour row → edge label (always `0`), per node row.
    adj: Vec<HashMap<usize, u64>>,
}

impl MatchGraph {
    /// Snapshot `g`'s nodes, `bead_type` labels and adjacency, with every
    /// edge label zeroed.
    fn new(g: &impl Graph) -> Self {
        let g = g.graph();
        let view = GraphView::build(g);
        let mut adj = adjacency_map(&view);
        for label in adj.iter_mut().flat_map(HashMap::values_mut) {
            *label = 0;
        }
        let labels = view
            .nodes
            .iter()
            .map(|&id| {
                g.get_node(id)
                    .ok()
                    .and_then(|bead| bead.get_str(keys::BEAD_TYPE).map(str::to_owned))
                    .unwrap_or_default()
            })
            .collect();
        Self {
            nodes: view.nodes,
            labels,
            adj,
        }
    }
}

/// An induced, `bead_type`-labelled subgraph matcher over one bead pattern.
///
/// Built once per pattern and run with [`find`](Self::find) against any
/// number of targets. The full contract (labels, induced edges, one group per
/// bead set, output order, the no-partition limit) is below.
///
/// # Example
///
/// ```
/// use molrs::perceive::SubgraphMatcher;
/// use molrs::core::CoarseGrain;
/// use molrs::core::NodeId;
/// use molrs::core::keys;
/// use molrs::props;
///
/// let chain = || {
///     let mut g = CoarseGrain::new();
///     let a = g.add_bead(props! { keys::BEAD_TYPE => "1" }).unwrap();
///     let b = g.add_bead(props! { keys::BEAD_TYPE => "4" }).unwrap();
///     let c = g.add_bead(props! { keys::BEAD_TYPE => "1" }).unwrap();
///     g.add_bond(a, b, props! {}).unwrap();
///     g.add_bond(b, c, props! {}).unwrap();
///     g
/// };
/// let pattern = chain();
/// let target = chain();
///
/// let groups = SubgraphMatcher::new(&pattern).find(&target);
/// // The mirror map `[c, b, a]` names the same bead set and is dropped.
/// let beads: Vec<NodeId> = target.bead_ids().collect();
/// assert_eq!(groups, vec![beads]);
/// ```
///
/// Induced, labelled subgraph matching of coarse-grained bead graphs: find
/// every group of beads in a target [`CoarseGrain`] that forms one occurrence
/// of a bead pattern (one monomer, one molecule).
///
/// In a coarse-grained (CG) graph each node is a **bead** standing for a group
/// of atoms, labelled by its `bead_type` string, and each CG bond says two
/// beads are connected. Backmapping — replacing each bead group by the
/// all-atom molecule it stands for — first has to find those groups, which is
/// what this module does.
///
/// # Contract of [`SubgraphMatcher::find`]
///
/// A **match** of a pattern `P` in a target `T` is an injective map `m` from
/// pattern beads to target beads (no two pattern beads share an image) such
/// that
///
/// - **Labels.** `bead_type(p) == bead_type(m(p))` for every pattern bead
///   `p`, by string equality. A bead's label is its `bead_type` alone (every
///   bead of a [`CoarseGrain`] carries one).
/// - **Edges.** Every arity-2 relation is an unlabelled edge: `bond_type` /
///   `bond_number` on a bond never block a match (a CG bond has no order).
/// - **Induced.** Two pattern beads are bonded iff their images are, so a
///   3-bead path does not match inside a triangle.
///
/// Degree is a pruning bound (`deg T ≥ deg P`), never part of a label, so a
/// pattern end bead of degree 1 matches a chain-interior target bead.
///
/// The result:
///
/// - **Group shape.** `group[i]` is the target bead matched to pattern row
///   `i`, the `i`-th bead of `pattern.bead_ids()`.
/// - **One group per bead set.** All maps onto one bead set differ by an
///   automorphism of the pattern — a relabelling of its beads that maps the
///   pattern onto itself, such as reading the symmetric `1-4-1` from the
///   other end — and describe the same group; the kept map is
///   the lexicographically smallest by target row (position in
///   `target.bead_ids()`).
/// - **Order.** Groups are sorted lexicographically by the target rows of
///   their kept map.
/// - **Degenerate inputs.** An empty pattern, an empty target, a pattern bead
///   type the target lacks, or no induced embedding all give `[]`.
/// - **Never fails.** `find` returns no `Result`.
/// - **Disconnected patterns** are accepted. The search restarts at each
///   pattern component, so the number of groups can grow as `O(Nᵏ)` for `k`
///   components.
///
/// # Known limit: `find` does not partition
///
/// `find` enumerates occurrences; it does not choose a non-overlapping cover.
/// On the head-to-tail chain `1-1-1-4-1-1-1-4` the pattern `1-1-1-4` occurs
/// three times, as rows `[0,1,2,3]`, `[4,5,6,7]` and `[6,5,4,3]`; the third
/// overlaps both others, and all three are returned. Picking a cover is the
/// caller's composition step.
///
/// # Algorithm
///
/// State-space backtracking in the VF2 lineage: the search grows a partial map
/// one (pattern bead, target bead) pair at a time, checks after each step that
/// the pairs placed so far still agree on adjacency (the *feasibility* test),
/// and undoes the last pair when no extension is feasible (L. P. Cordella,
/// P. Foggia, C. Sansone and M. Vento, "A (sub)graph isomorphism algorithm for
/// matching large graphs", *IEEE TPAMI* **26**, 1367–1372 (2004),
/// doi:10.1109/TPAMI.2004.75). It is the same lineage as
/// [`CoarseGrain::is_isomorphic`], whose
/// structural feasibility check (`feasible`) this search shares. The worst
/// case is exponential in the pattern size.
///
/// - **Candidates.** Per pattern bead, the target beads with an equal label
///   and a large enough degree.
/// - **Order.** Fewest candidates first; each later bead is the
///   fewest-candidate pattern bead adjacent to one already placed
///   (neighbourhood extension, as in the SMARTS substructure matcher of
///   [`crate::perceive::smarts`]), so its images come
///   from its anchor's image neighbourhood rather than the whole target. A
///   disconnected pattern restarts from the fewest-candidate unplaced bead.
#[derive(Debug, Clone)]
pub struct SubgraphMatcher {
    pattern: MatchGraph,
}

impl SubgraphMatcher {
    /// A matcher over the bead pattern `pattern`, snapshotted once.
    pub fn new(pattern: &CoarseGrain) -> Self {
        Self {
            pattern: MatchGraph::new(pattern),
        }
    }

    /// Every induced occurrence of the pattern in `target`, one group per
    /// distinct bead set.
    ///
    /// `group[i]` is the target bead of pattern row `i`; each group is the
    /// lexicographically smallest map (by target row) onto its bead set, and
    /// the groups are sorted by those rows. Overlapping groups are all
    /// returned. No occurrence (including an empty pattern or target) gives
    /// an empty `Vec`.
    pub fn find(&self, target: &CoarseGrain) -> Vec<Vec<NodeId>> {
        let n = self.pattern.labels.len();
        if n == 0 {
            return Vec::new();
        }
        let target = MatchGraph::new(target);
        let p_adj = &self.pattern.adj;

        let is_cand: Vec<Vec<bool>> = (0..n)
            .map(|u| {
                target
                    .labels
                    .iter()
                    .zip(&target.adj)
                    .map(|(t_label, nbrs)| {
                        nbrs.len() >= p_adj[u].len() && self.pattern.labels[u] == *t_label
                    })
                    .collect()
            })
            .collect();
        let counts: Vec<usize> = is_cand
            .iter()
            .map(|row| row.iter().filter(|&&c| c).count())
            .collect();
        if counts.contains(&0) {
            return Vec::new();
        }

        let order = self.search_order(&counts);
        let mut search = Search {
            order: &order,
            p_adj,
            t_adj: &target.adj,
            is_cand: &is_cand,
            map_pt: vec![UNMAPPED; n],
            map_tp: vec![UNMAPPED; target.nodes.len()],
            found: Vec::new(),
        };
        search.extend(0);

        let mut found = search.found;
        found.sort_unstable();
        // Sorted input: the first map seen for a bead set is its smallest,
        // and keeping first maps in input order keeps the output sorted.
        let mut seen: HashSet<Vec<usize>> = HashSet::new();
        found
            .into_iter()
            .filter(|m| {
                let mut set = m.clone();
                set.sort_unstable();
                seen.insert(set)
            })
            .map(|m| m.into_iter().map(|row| target.nodes[row]).collect())
            .collect()
    }

    /// Placement order as `(pattern row, anchor row)`: fewest candidates
    /// first (ties by row), then always the fewest-candidate unplaced node
    /// adjacent to a placed one, anchored on its earliest-placed neighbour.
    /// `None` anchors a component root.
    fn search_order(&self, counts: &[usize]) -> Vec<(usize, Option<usize>)> {
        let adj = &self.pattern.adj;
        let n = counts.len();
        let mut placed_at = vec![UNMAPPED; n];
        let mut order = Vec::with_capacity(n);
        for step in 0..n {
            let unplaced = || (0..n).filter(|&u| placed_at[u] == UNMAPPED);
            let frontier = unplaced()
                .filter(|&u| adj[u].keys().any(|&w| placed_at[w] != UNMAPPED))
                .min_by_key(|&u| (counts[u], u));
            let next = frontier.or_else(|| unplaced().min_by_key(|&u| (counts[u], u)));
            let Some(u) = next else {
                break;
            };
            let anchor = adj[u]
                .keys()
                .copied()
                .filter(|&w| placed_at[w] != UNMAPPED)
                .min_by_key(|&w| placed_at[w]);
            placed_at[u] = step;
            order.push((u, anchor));
        }
        order
    }
}

/// Mutable state of one `find` backtracking run.
struct Search<'a> {
    /// `(pattern row, anchor row)` per depth.
    order: &'a [(usize, Option<usize>)],
    p_adj: &'a [HashMap<usize, u64>],
    t_adj: &'a [HashMap<usize, u64>],
    /// `is_cand[p][t]`: label accepted and degree sufficient.
    is_cand: &'a [Vec<bool>],
    /// Pattern row → target row.
    map_pt: Vec<usize>,
    /// Target row → pattern row.
    map_tp: Vec<usize>,
    /// Complete matches as target rows, indexed by pattern row.
    found: Vec<Vec<usize>>,
}

impl Search<'_> {
    /// Place the pattern node at `depth`, recursing to the leaves.
    fn extend(&mut self, depth: usize) {
        if depth == self.order.len() {
            self.found.push(self.map_pt.clone());
            return;
        }
        let (u, anchor) = self.order[depth];
        match anchor {
            Some(a) => {
                let t_adj = self.t_adj;
                for &v in t_adj[self.map_pt[a]].keys() {
                    self.try_place(depth, u, v);
                }
            }
            None => {
                for v in 0..self.map_tp.len() {
                    self.try_place(depth, u, v);
                }
            }
        }
    }

    /// Map pattern `u` to target `v` if free, a candidate and structurally
    /// feasible, then recurse.
    fn try_place(&mut self, depth: usize, u: usize, v: usize) {
        if self.map_tp[v] != UNMAPPED || !self.is_cand[u][v] {
            return;
        }
        if !feasible(u, v, self.p_adj, self.t_adj, &self.map_pt, &self.map_tp) {
            return;
        }
        self.map_pt[u] = v;
        self.map_tp[v] = u;
        self.extend(depth + 1);
        self.map_pt[u] = UNMAPPED;
        self.map_tp[v] = UNMAPPED;
    }
}

#[cfg(test)]
mod tests {
    use super::SubgraphMatcher;
    use crate::core::BondNumber;
    use crate::core::CoarseGrain;
    use crate::core::NodeId;
    use crate::core::keys;
    use crate::props;

    /// A coarse-grained graph with one bead per entry of `types` (row = index)
    /// and one CG bond per `(row, row)` pair.
    fn cg(types: &[&str], bonds: &[(usize, usize)]) -> CoarseGrain {
        let mut g = CoarseGrain::new();
        let ids: Vec<NodeId> = types
            .iter()
            .map(|t| {
                g.add_bead(props! { keys::BEAD_TYPE => *t })
                    .expect("fixture bead")
            })
            .collect();
        for &(a, b) in bonds {
            g.add_bond(ids[a], ids[b], props! {}).expect("fixture bond");
        }
        g
    }

    /// Run the matcher and translate every group into target rows (position
    /// in `target.bead_ids()`).
    ///
    /// Contract under test: `group[i]` is the target bead of pattern row `i`,
    /// one group per bead set (the lexicographically smallest map kept), and
    /// the groups sorted lexicographically by those target rows, so the
    /// expected vectors below are exact.
    fn rows(pattern: &CoarseGrain, target: &CoarseGrain) -> Vec<Vec<usize>> {
        let order: Vec<NodeId> = target.bead_ids().collect();
        let row = |id: NodeId| {
            order
                .iter()
                .position(|&n| n == id)
                .expect("a group names live target beads")
        };
        SubgraphMatcher::new(pattern)
            .find(target)
            .into_iter()
            .map(|group| group.into_iter().map(row).collect())
            .collect()
    }

    /// The monomer pattern `1-1-1-4` of the chain goldens.
    fn monomer() -> CoarseGrain {
        cg(&["1", "1", "1", "4"], &[(0, 1), (1, 2), (2, 3)])
    }

    /// Path bonds `(0,1), (1,2), …` over `n` beads.
    fn path(n: usize) -> Vec<(usize, usize)> {
        (1..n).map(|i| (i - 1, i)).collect()
    }

    #[test]
    fn finds_both_edges_of_a_path() {
        let pattern = cg(&["A", "B"], &[(0, 1)]);
        let target = cg(&["A", "B", "A"], &[(0, 1), (1, 2)]);
        assert_eq!(rows(&pattern, &target), vec![vec![0, 1], vec![2, 1]]);
    }

    #[test]
    fn edge_matches_every_arm_of_a_star() {
        let pattern = cg(&["A", "B"], &[(0, 1)]);
        let star = cg(&["B", "A", "A", "A"], &[(0, 1), (0, 2), (0, 3)]);
        assert_eq!(
            rows(&pattern, &star),
            vec![vec![1, 0], vec![2, 0], vec![3, 0]]
        );
    }

    #[test]
    fn match_is_induced_so_a_path_misses_a_triangle() {
        let path = cg(&["A", "B", "C"], &[(0, 1), (1, 2)]);
        let triangle = cg(&["A", "B", "C"], &[(0, 1), (1, 2), (2, 0)]);
        assert_eq!(
            rows(&path, &triangle),
            Vec::<Vec<usize>>::new(),
            "target A–C is bonded but pattern A–C is not"
        );
    }

    #[test]
    fn triangle_matches_triangle() {
        let triangle = cg(&["A", "B", "C"], &[(0, 1), (1, 2), (2, 0)]);
        let groups = rows(&triangle, &triangle);
        assert!(!groups.is_empty());
        assert_eq!(groups, vec![vec![0, 1, 2]], "distinct labels fix the map");
    }

    #[test]
    fn head_to_head_chain_yields_two_disjoint_groups() {
        let target = cg(&["4", "1", "1", "1", "1", "1", "1", "4"], &path(8));
        assert_eq!(
            rows(&monomer(), &target),
            vec![vec![3, 2, 1, 0], vec![4, 5, 6, 7]]
        );
    }

    #[test]
    fn head_to_tail_chain_yields_all_three_overlapping_groups() {
        // Known limit: `find` enumerates occurrences, it does not partition
        // the target; the middle group overlaps both monomers.
        let target = cg(&["1", "1", "1", "4", "1", "1", "1", "4"], &path(8));
        assert_eq!(
            rows(&monomer(), &target),
            vec![vec![0, 1, 2, 3], vec![4, 5, 6, 7], vec![6, 5, 4, 3]]
        );
    }

    #[test]
    fn symmetric_pattern_yields_one_group_per_bead_set() {
        let pattern = cg(&["1", "4", "1"], &[(0, 1), (1, 2)]);
        let target = cg(&["1", "4", "1"], &[(0, 1), (1, 2)]);
        assert_eq!(
            rows(&pattern, &target),
            vec![vec![0, 1, 2]],
            "the automorphic map [2, 1, 0] names the same bead set and is dropped"
        );
    }

    #[test]
    fn edge_labels_are_ignored() {
        let pattern = cg(&["1", "4"], &[(0, 1)]);
        let mut target = cg(&["1", "4"], &[(0, 1)]);
        let bond = target.bond_ids().next().expect("fixture bond");
        target
            .set_bond(bond, keys::BOND_NUMBER, BondNumber::Double)
            .expect("bond_number on the target bond");
        assert_eq!(rows(&pattern, &target), vec![vec![0, 1]]);
    }

    #[test]
    fn empty_pattern_has_no_groups() {
        let pattern = cg(&[], &[]);
        let target = cg(&["1", "4"], &[(0, 1)]);
        assert!(SubgraphMatcher::new(&pattern).find(&target).is_empty());
    }

    #[test]
    fn absent_bead_type_has_no_groups() {
        let pattern = cg(&["1", "9"], &[(0, 1)]);
        let target = cg(&["1", "4", "1"], &[(0, 1), (1, 2)]);
        assert!(SubgraphMatcher::new(&pattern).find(&target).is_empty());
    }
}
