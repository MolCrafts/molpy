//! Rotatable bond detection and downstream atom BFS.

use std::collections::{HashSet, VecDeque};

use crate::core::Atomistic;
use crate::core::BondOrder;
use crate::core::MolRsError;
use crate::core::Topology;
use crate::core::keys;
use crate::core::{NodeId, RelationId};

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
    let ids: Vec<NodeId> = graph.atom_ids().collect();
    let rows = graph.atom_table();

    let bonds: Vec<_> = graph.bonds().collect();
    let topo = graph.topology();
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

            let j = rows
                .row(bond.nodes[0])
                .expect("a bond's endpoints are live atoms");
            let k = rows
                .row(bond.nodes[1])
                .expect("a bond's endpoints are live atoms");

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

/// BFS to find all atoms downstream of bond (j, k) on the k-side.
///
/// Starting from `k`, traverses all connected atoms without crossing back
/// through `j`. Returns atom rows (the order of [`Atomistic::atom_ids`]),
/// not `NodeId`s.
///
/// # Errors
///
/// [`MolRsError::NotFound`](crate::core::MolRsError) when `j` or `k` is not
/// a live atom of `graph`.
pub fn downstream_atoms(
    j: NodeId,
    k: NodeId,
    graph: &Atomistic,
) -> Result<Vec<usize>, crate::core::MolRsError> {
    let rows = graph.atom_table();
    let row = |atom: NodeId| {
        rows.row(atom)
            .ok_or_else(|| crate::core::MolRsError::not_found("atom", format!("{atom:?}")))
    };
    Ok(downstream_indices(&graph.topology(), row(j)?, row(k)?))
}

/// Detect the rotatable bonds of a molecular graph, each with the atoms that
/// rotate about it.
///
/// A bond is rotatable when its class is single, both endpoints have
/// degree > 1, and it lies on no ring; `unknown` decides a bond with no class
/// written. Each [`RotatableBond`] names its axis and its downstream atoms by
/// atom row (the order of [`Atomistic::atom_ids`]).
///
/// Connectivity, degree, ring membership and traversal all come from the
/// canonical [`Topology`] snapshot built from the graph's bonds, so ring
/// membership is `Topology`'s SSSR (correct for fused and bridged rings).
pub fn perceive_rotatable_bonds(
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

/// Perceive rotatable bonds and write them onto `mol`, in place.
///
/// The [`perceive_rotatable_bonds`] answer — atom **pairs** — resolved back to
/// bond ids: every bond receives `is_rotatable` (`0` / `1`), the non-rotatable
/// ones explicitly `0`. `unknown` decides a bond with no class written.
///
/// # Errors
///
/// [`MolRsError`] when a flag cannot be written (a bond column holding the key
/// under a foreign dtype); `mol` may then be partly annotated.
pub fn assign_rotatable_bonds(
    mol: &mut Atomistic,
    unknown: UnknownBondPolicy,
) -> Result<(), MolRsError> {
    let unordered = |a: NodeId, b: NodeId| if a <= b { (a, b) } else { (b, a) };
    let (ids, _topo, pairs) = scan_rotatable(mol, unknown);
    let rotatable: HashSet<(NodeId, NodeId)> = pairs
        .into_iter()
        .map(|(j, k)| unordered(ids[j], ids[k]))
        .collect();
    let bonds: Vec<(RelationId, NodeId, NodeId)> = mol
        .bonds()
        .map(|(bid, bond)| (bid, bond.nodes[0], bond.nodes[1]))
        .collect();
    for (bid, a, b) in bonds {
        let flag = i32::from(rotatable.contains(&unordered(a, b)));
        mol.set_bond(bid, IS_ROTATABLE, flag)?;
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    /// Build a positional index map from `NodeId` to 0-based index, following the
    /// iteration order of [`Atomistic::atoms`].
    use super::*;
    use crate::core::BondNumber;
    use crate::core::{Element, Props, keys};
    use crate::props;

    /// A carbon atom: detection reads only topology, but every atom of an
    /// `Atomistic` carries a species.
    fn carbon() -> Props {
        props! { keys::ATOMIC_NUMBER => Element::C }
    }

    /// Build a chain graph (topology only, coords irrelevant for detection).
    fn chain(n: usize) -> Atomistic {
        let mut g = Atomistic::new();
        let mut ids = Vec::new();
        for _ in 0..n {
            ids.push(g.add_atom(carbon()).unwrap());
        }
        for i in 0..n - 1 {
            g.add_bond(ids[i], ids[i + 1], props! {})
                .expect("add chain bond");
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
        let a = g.add_atom(carbon()).unwrap();
        let b = g.add_atom(carbon()).unwrap();
        let c = g.add_atom(carbon()).unwrap();
        g.add_bond(a, b, props! {}).expect("add bond");
        g.add_bond(b, c, props! {}).expect("add bond");
        g.add_bond(c, a, props! {}).expect("add bond");

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
        let v: Vec<_> = (0..10).map(|_| g.add_atom(carbon()).unwrap()).collect();
        // ring A: 0-1-2-3-4-5-0
        for (a, b) in [(0, 1), (1, 2), (2, 3), (3, 4), (4, 5), (5, 0)] {
            g.add_bond(v[a], v[b], props! {}).expect("add bond");
        }
        // ring B: 0-1 shared, plus 1-6-7-8-9-0
        for (a, b) in [(1, 6), (6, 7), (7, 8), (8, 9), (9, 0)] {
            g.add_bond(v[a], v[b], props! {}).expect("add bond");
        }
        assert_eq!(
            perceive_rotatable_bonds(&g, UnknownBondPolicy::NotRotatable).len(),
            0
        );
    }

    #[test]
    fn test_downstream_atoms_chain() {
        let g = chain(5);
        let ids: Vec<NodeId> = g.atom_ids().collect();

        let ds = downstream_atoms(ids[1], ids[2], &g).unwrap();
        assert_eq!(ds.len(), 3);
        assert!(ds.contains(&2));
        assert!(ds.contains(&3));
        assert!(ds.contains(&4));
    }

    #[test]
    fn test_detect_with_downstream() {
        let g = chain(5);
        let bonds = perceive_rotatable_bonds(&g, UnknownBondPolicy::NotRotatable);
        assert_eq!(bonds.len(), 2);

        for rb in &bonds {
            assert!(!rb.downstream.is_empty());
            assert!(rb.downstream.contains(&rb.k));
        }
    }

    #[test]
    fn assign_rotatable_bonds_flags_every_bond() {
        // C0-C1-C2-C3: only the middle bond joins two non-terminal atoms.
        let mut g = chain(4);
        assign_rotatable_bonds(&mut g, UnknownBondPolicy::NotRotatable).unwrap();
        let flags: Vec<Option<i32>> = g
            .bonds()
            .map(|(_, bond)| bond.props.get_int(IS_ROTATABLE))
            .collect();
        assert_eq!(flags, vec![Some(0), Some(1), Some(0)]);
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
        g.set_bond_class(middle, BondOrder::Double, BondNumber::Double)
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
        let center = g.add_atom(carbon()).unwrap();
        let b0 = g.add_atom(carbon()).unwrap();
        let b0p = g.add_atom(carbon()).unwrap();
        let b1 = g.add_atom(carbon()).unwrap();
        let b1p = g.add_atom(carbon()).unwrap();
        let b2 = g.add_atom(carbon()).unwrap();
        let b2p = g.add_atom(carbon()).unwrap();

        g.add_bond(center, b0, props! {}).expect("add bond");
        g.add_bond(b0, b0p, props! {}).expect("add bond");
        g.add_bond(center, b1, props! {}).expect("add bond");
        g.add_bond(b1, b1p, props! {}).expect("add bond");
        g.add_bond(center, b2, props! {}).expect("add bond");
        g.add_bond(b2, b2p, props! {}).expect("add bond");

        assert_eq!(
            perceive_rotatable_bonds(&g, UnknownBondPolicy::NotRotatable).len(),
            3
        );
    }
}
