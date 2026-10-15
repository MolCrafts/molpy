//! Rotatable bond detection and downstream atom BFS.

use std::collections::{HashMap, HashSet, VecDeque};

use crate::core::Atomistic;
use crate::core::BondOrder;
use crate::core::NodeId;
use crate::core::Topology;
use crate::core::keys;

/// How rotatable-bond detection reads a bond whose class is
/// [`BondOrder::Unknown`] — no `bond_type` written, as from a connectivity-only
/// reader (LAMMPS data, a PDB without `CONECT` orders) before perception ran.
///
/// Detection never guesses on its own: the caller who knows where the graph
/// came from says what an unclassified bond is.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq, Hash)]
pub enum UnknownBondPolicy {
    /// An unclassified bond is not rotatable. The default: only a bond that
    /// *is* single rotates.
    #[default]
    NotRotatable,
    /// An unclassified bond counts as single, then the degree and ring rules
    /// apply as usual. For all-single graphs whose orders were never written.
    AsSingle,
}

impl UnknownBondPolicy {
    /// Whether a bond of class `bond_type` passes the single-bond rule.
    ///
    /// An aromatic bond never does, whatever its Kekulé number: it is read as a
    /// class, not as the localized number.
    fn is_single(self, bond_type: BondOrder) -> bool {
        match bond_type {
            BondOrder::Single => true,
            BondOrder::Unknown => self == UnknownBondPolicy::AsSingle,
            BondOrder::Double | BondOrder::Triple | BondOrder::Aromatic => false,
        }
    }
}

/// A rotatable bond between atoms `j` and `k`, with the set of downstream
/// atom indices (0-based positional indices, not `NodeId`s) on the `k`-side.
#[derive(Debug, Clone)]
pub struct RotatableBond {
    /// Axis start atom (positional index).
    pub j: usize,
    /// Axis end atom (positional index).
    pub k: usize,
    /// Atoms on the k-side of the bond (positional indices), rotated during torsion.
    pub downstream: Vec<usize>,
}

/// Build the [`Topology`] snapshot for `graph`, indexing atoms by the order of
/// [`Atomistic::atoms`]; a bond listed twice in `graph` is one edge.
fn build_topology(graph: &Atomistic, id_to_idx: &HashMap<NodeId, usize>) -> Topology {
    let edges: Vec<[usize; 2]> = graph
        .bonds()
        .map(|(_, b)| [id_to_idx[&b.nodes[0]], id_to_idx[&b.nodes[1]]])
        .collect();
    let mut topo = Topology::with_atoms(id_to_idx.len());
    topo.add_bonds(&edges);
    topo
}

/// Shared detection core. Returns the positional `(j, k)` index pair of every
/// rotatable bond together with the `Topology` they were derived from (reused by
/// the caller for downstream BFS) and the reverse `idx -> NodeId` table.
///
/// A bond is rotatable when its class is [`BondOrder::Single`] (an
/// [`Unknown`](BondOrder::Unknown) class decided by `unknown`), both endpoints
/// are non-terminal (degree > 1), and it is acyclic (not part of any ring per
/// [`Topology::perceive_rings`]).
fn scan_rotatable(
    graph: &Atomistic,
    unknown: UnknownBondPolicy,
) -> (Vec<NodeId>, Topology, Vec<(usize, usize)>) {
    let ids: Vec<NodeId> = graph.atoms().map(|(id, _)| id).collect();
    let id_to_idx: HashMap<NodeId, usize> =
        ids.iter().enumerate().map(|(i, &id)| (id, i)).collect();

    let bonds: Vec<_> = graph.bonds().collect();
    let topo = build_topology(graph, &id_to_idx);
    // Ring membership by endpoint pair, so a bond listed twice in `graph`
    // (one topology edge) reads the same answer as its twin.
    let ring_pairs: HashSet<(usize, usize)> = topo
        .bonds()
        .into_iter()
        .zip(super::rings::ring_bond_mask(&topo))
        .filter(|&(_, in_ring)| in_ring)
        .map(|([a, b], _)| (a.min(b), a.max(b)))
        .collect();

    let rotatable = bonds
        .iter()
        .filter_map(|(_, bond)| {
            // A *single* bond, as a class. An aromatic bond whose Kekulé phase
            // happens to be single does not rotate, so this must not read the
            // localized number.
            if !unknown.is_single(BondOrder::from_prop(bond.props.get(keys::BOND_TYPE))) {
                return None;
            }

            let j = id_to_idx[&bond.nodes[0]];
            let k = id_to_idx[&bond.nodes[1]];

            // Non-terminal: both endpoints must have degree > 1.
            if topo.degree(j) <= 1 || topo.degree(k) <= 1 {
                return None;
            }

            // Acyclic: bond is not part of any ring.
            if ring_pairs.contains(&(j.min(k), j.max(k))) {
                return None;
            }

            Some((j, k))
        })
        .collect();

    (ids, topo, rotatable)
}

/// BFS over `topo` from `k`, never crossing back through `j`. Returns the k-side
/// atom indices (including `k`).
fn downstream_indices(topo: &Topology, j: usize, k: usize) -> Vec<usize> {
    let mut visited = HashSet::new();
    visited.insert(j); // block traversal back through j
    visited.insert(k);

    let mut queue = VecDeque::new();
    queue.push_back(k);

    let mut result = vec![k];

    while let Some(current) = queue.pop_front() {
        for neighbor in topo.neighbors(current) {
            if visited.insert(neighbor) {
                result.push(neighbor);
                queue.push_back(neighbor);
            }
        }
    }

    result
}

/// Detect rotatable bonds in a molecular graph.
///
/// Returns a list of `(NodeId, NodeId)` pairs. A bond is rotatable when its
/// class is single, both endpoints have degree > 1, and it lies on no ring;
/// `unknown` decides a bond with no class written.
///
/// Connectivity, degree, ring membership and traversal all come from the
/// canonical [`Topology`] snapshot built from the graph's bonds, so ring
/// membership is `Topology`'s SSSR (correct for fused and bridged rings).
pub fn perceive_rotatable_bonds(
    graph: &Atomistic,
    unknown: UnknownBondPolicy,
) -> Vec<(NodeId, NodeId)> {
    let (ids, _topo, rotatable) = scan_rotatable(graph, unknown);
    rotatable
        .into_iter()
        .map(|(j, k)| (ids[j], ids[k]))
        .collect()
}

/// BFS to find all atoms downstream of bond (j, k) on the k-side.
///
/// Starting from `k`, traverses all connected atoms without crossing back
/// through `j`. Returns positional indices (not `NodeId`s).
pub fn downstream_atoms(
    j: NodeId,
    k: NodeId,
    graph: &Atomistic,
    id_to_idx: &HashMap<NodeId, usize>,
) -> Vec<usize> {
    let topo = build_topology(graph, id_to_idx);
    downstream_indices(&topo, id_to_idx[&j], id_to_idx[&k])
}

/// Detect rotatable bonds and return them as [`RotatableBond`] structs with
/// positional indices and downstream atom sets. `unknown` as in
/// [`perceive_rotatable_bonds`].
pub fn perceive_rotatable_bonds_with_downstream(
    graph: &Atomistic,
    unknown: UnknownBondPolicy,
) -> Vec<RotatableBond> {
    let (_ids, topo, rotatable) = scan_rotatable(graph, unknown);
    rotatable
        .into_iter()
        .map(|(j, k)| RotatableBond {
            j,
            k,
            downstream: downstream_indices(&topo, j, k),
        })
        .collect()
}

/// Bond prop written by [`assign_rotatable_bonds`]: `1` when the bond is a
/// rotatable single bond.
const IS_ROTATABLE: &str = "is_rotatable";

/// Perceive rotatable bonds and write them onto a clone of `mol`.
///
/// The [`perceive_rotatable_bonds`] side table — atom **pairs** — resolved back
/// to bond ids: every bond receives `is_rotatable` (`0` / `1`), the
/// non-rotatable ones explicitly `0`. `unknown` decides a bond with no class
/// written. `mol` is left untouched.
pub fn assign_rotatable_bonds(mol: &Atomistic, unknown: UnknownBondPolicy) -> Atomistic {
    let unordered = |a: NodeId, b: NodeId| if a <= b { (a, b) } else { (b, a) };
    let rotatable: std::collections::HashSet<(NodeId, NodeId)> =
        perceive_rotatable_bonds(mol, unknown)
            .into_iter()
            .map(|(a, b)| unordered(a, b))
            .collect();
    let mut out = mol.clone();
    let bonds: Vec<(crate::core::RelationId, NodeId, NodeId)> = out
        .bonds()
        .map(|(bid, bond)| (bid, bond.nodes[0], bond.nodes[1]))
        .collect();
    for (bid, a, b) in bonds {
        let flag = i32::from(rotatable.contains(&unordered(a, b)));
        let _ = out.set_bond_prop(bid, IS_ROTATABLE, flag);
    }
    out
}

#[cfg(test)]
mod tests {
    /// Build a positional index map from `NodeId` to 0-based index, following the
    /// iteration order of [`Atomistic::atoms`].
    fn atom_id_to_index(graph: &Atomistic) -> HashMap<NodeId, usize> {
        graph
            .atoms()
            .enumerate()
            .map(|(idx, (id, _))| (id, idx))
            .collect()
    }

    use super::*;
    use crate::core::Atom;
    use crate::core::BondNumber;

    /// Build a chain graph (topology only, coords irrelevant for detection).
    fn chain(n: usize) -> Atomistic {
        let mut g = Atomistic::new();
        let mut ids = Vec::new();
        for _ in 0..n {
            ids.push(g.add_atom(Atom::new()));
        }
        for i in 0..n - 1 {
            g.add_bond(ids[i], ids[i + 1]).expect("add chain bond");
        }
        g
    }

    #[test]
    fn test_chain_rotatable_bonds() {
        let g = chain(5);
        let bonds = perceive_rotatable_bonds(&g, UnknownBondPolicy::NotRotatable);
        assert_eq!(bonds.len(), 2);
    }

    #[test]
    fn test_ring_no_rotatable_bonds() {
        let mut g = Atomistic::new();
        let a = g.add_atom(Atom::new());
        let b = g.add_atom(Atom::new());
        let c = g.add_atom(Atom::new());
        g.add_bond(a, b).expect("add bond");
        g.add_bond(b, c).expect("add bond");
        g.add_bond(c, a).expect("add bond");

        assert_eq!(
            perceive_rotatable_bonds(&g, UnknownBondPolicy::NotRotatable).len(),
            0
        );
    }

    #[test]
    fn test_fused_ring_no_rotatable_bonds() {
        // Naphthalene-like fused bicyclic: two six-membered rings sharing the
        // 0-1 edge. Every bond is a ring bond -> none rotatable. The previous
        // union-find heuristic could mis-mark shared/bridge bonds; SSSR cannot.
        let mut g = Atomistic::new();
        let v: Vec<_> = (0..10).map(|_| g.add_atom(Atom::new())).collect();
        // ring A: 0-1-2-3-4-5-0
        for (a, b) in [(0, 1), (1, 2), (2, 3), (3, 4), (4, 5), (5, 0)] {
            g.add_bond(v[a], v[b]).expect("add bond");
        }
        // ring B: 0-1 shared, plus 1-6-7-8-9-0
        for (a, b) in [(1, 6), (6, 7), (7, 8), (8, 9), (9, 0)] {
            g.add_bond(v[a], v[b]).expect("add bond");
        }
        assert_eq!(
            perceive_rotatable_bonds(&g, UnknownBondPolicy::NotRotatable).len(),
            0
        );
    }

    #[test]
    fn test_downstream_atoms_chain() {
        let g = chain(5);
        let id_to_idx = atom_id_to_index(&g);
        let ids: Vec<NodeId> = g.atoms().map(|(id, _)| id).collect();

        let ds = downstream_atoms(ids[1], ids[2], &g, &id_to_idx);
        assert_eq!(ds.len(), 3);
        assert!(ds.contains(&2));
        assert!(ds.contains(&3));
        assert!(ds.contains(&4));
    }

    #[test]
    fn test_detect_with_downstream() {
        let g = chain(5);
        let bonds = perceive_rotatable_bonds_with_downstream(&g, UnknownBondPolicy::NotRotatable);
        assert_eq!(bonds.len(), 2);

        for rb in &bonds {
            assert!(!rb.downstream.is_empty());
            assert!(rb.downstream.contains(&rb.k));
        }
    }

    /// A 4-chain whose bonds carry no class: the middle bond rotates only
    /// when the caller says an unclassified bond is single.
    fn unclassified_chain() -> Atomistic {
        let mut g = chain(4);
        let bonds: Vec<_> = g.bonds().map(|(id, _)| id).collect();
        for id in bonds {
            g.set_bond_class(id, BondOrder::Unknown, BondNumber::Unknown)
                .expect("clear bond class");
        }
        g
    }

    #[test]
    fn unknown_bond_is_not_rotatable_by_default() {
        let g = unclassified_chain();
        assert_eq!(
            UnknownBondPolicy::default(),
            UnknownBondPolicy::NotRotatable
        );
        assert!(perceive_rotatable_bonds(&g, UnknownBondPolicy::NotRotatable).is_empty());
    }

    #[test]
    fn unknown_bond_counts_as_single_when_asked() {
        let g = unclassified_chain();
        assert_eq!(
            perceive_rotatable_bonds(&g, UnknownBondPolicy::AsSingle).len(),
            1
        );
    }

    #[test]
    fn as_single_never_rotates_a_classified_multiple_bond() {
        let mut g = chain(4);
        let middle = g
            .bonds()
            .map(|(id, _)| id)
            .nth(1)
            .expect("chain(4) has 3 bonds");
        g.set_bond_type(middle, BondOrder::Double)
            .expect("set bond class");
        assert!(perceive_rotatable_bonds(&g, UnknownBondPolicy::AsSingle).is_empty());
    }

    #[test]
    fn test_two_atoms_no_rotatable() {
        assert_eq!(
            perceive_rotatable_bonds(&chain(2), UnknownBondPolicy::NotRotatable).len(),
            0
        );
    }

    #[test]
    fn test_branched_molecule() {
        //   B0' - B0 - C - B1 - B1'
        //              |
        //              B2 - B2'
        let mut g = Atomistic::new();
        let center = g.add_atom(Atom::new());
        let b0 = g.add_atom(Atom::new());
        let b0p = g.add_atom(Atom::new());
        let b1 = g.add_atom(Atom::new());
        let b1p = g.add_atom(Atom::new());
        let b2 = g.add_atom(Atom::new());
        let b2p = g.add_atom(Atom::new());

        g.add_bond(center, b0).expect("add bond");
        g.add_bond(b0, b0p).expect("add bond");
        g.add_bond(center, b1).expect("add bond");
        g.add_bond(b1, b1p).expect("add bond");
        g.add_bond(center, b2).expect("add bond");
        g.add_bond(b2, b2p).expect("add bond");

        assert_eq!(
            perceive_rotatable_bonds(&g, UnknownBondPolicy::NotRotatable).len(),
            3
        );
    }
}
