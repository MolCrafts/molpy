//! Topological symmetry classes: which atoms the bare connectivity cannot tell apart.
//!
//! The partition is colour refinement on the element-labelled graph with bond
//! orders and formal charges erased, the partition RDKit's
//! `CanonicalRankAtoms(breakTies=False)` gives on a molecule whose bonds were
//! all set single and whose charges were all set to zero. The two oxygens of a
//! carboxylate, written `C(=O)[O-]`, fall in one class; so do the three
//! hydrogens of a methyl group. A force field that must give such atoms one
//! parameter averages over these classes.
//!
//! Each atom starts from its element, its degree, a declared implicit
//! hydrogen count and the number of its bonds that lie in a ring. Every seed is
//! invariant under the molecule's automorphisms, so no seed can split a true
//! symmetry class. The number of SSSR rings through an atom is *not* such an
//! invariant: on a bridged system the SSSR keeps some of a set of equivalent
//! rings and drops the rest (bicyclo[2.2.2]octane keeps two of its three six-rings), so
//! seeding with it splits equivalent bridges apart. Each round replaces an atom's colour by its colour and the sorted multiset
//! of its neighbours' colours, until the number of classes stops growing.
//! Colours are ranks of the signatures, never hashes, so two classes never
//! merge by collision.

use std::collections::BTreeMap;

use crate::core::{Atomistic, MolRsError, keys};

use super::rings::{perceive_rings, saturating_i32};
use super::valence::atomic_number;

/// Every atom's class, in atom-table order: the row of the class's first member.
///
/// Two atoms share a value iff they fall in one class, and the value is a row of
/// the atom table, so it is itself an atom of the class.
pub fn perceive_symmetry_classes(mol: &Atomistic) -> Vec<usize> {
    let rows = mol.atom_table();
    let ids: Vec<_> = mol.atom_ids().collect();
    let n = ids.len();
    let mut adjacency = vec![Vec::new(); n];
    for (_, bond) in mol.bonds() {
        if let (Some(i), Some(j)) = (rows.row(bond.nodes[0]), rows.row(bond.nodes[1])) {
            adjacency[i].push(j);
            adjacency[j].push(i);
        }
    }
    let rings = perceive_rings(mol);
    let mut ring_bonds = vec![0_usize; n];
    for (bond_id, bond) in mol.bonds() {
        if rings.is_bond_in_ring(bond_id) {
            for node in bond.nodes {
                if let Some(row) = rows.row(node) {
                    ring_bonds[row] += 1;
                }
            }
        }
    }
    let seeds: Vec<(u8, usize, i32, usize)> = mol
        .atoms()
        .enumerate()
        .map(|(row, (_, atom))| {
            let declared_h = atom.get_int(keys::H_COUNT).unwrap_or(0);
            (
                atomic_number(&atom),
                adjacency[row].len(),
                declared_h,
                ring_bonds[row],
            )
        })
        .collect();
    let mut colours = ranks(&seeds);
    let mut n_classes = count(&colours);
    loop {
        let signatures: Vec<(usize, Vec<usize>)> = (0..n)
            .map(|i| {
                let mut around: Vec<usize> = adjacency[i].iter().map(|&j| colours[j]).collect();
                around.sort_unstable();
                (colours[i], around)
            })
            .collect();
        let refined = ranks(&signatures);
        let refined_classes = count(&refined);
        colours = refined;
        if refined_classes == n_classes {
            break;
        }
        n_classes = refined_classes;
    }
    let mut first: BTreeMap<usize, usize> = BTreeMap::new();
    (0..n)
        .map(|i| *first.entry(colours[i]).or_insert(i))
        .collect()
}

/// Perceive symmetry classes and write them onto `mol`, in place: every atom
/// receives a [`keys::SYMMETRY_CLASS`], the atom-table row of its class's first
/// member.
pub fn assign_symmetry_classes(mol: &mut Atomistic) -> Result<(), MolRsError> {
    let classes = perceive_symmetry_classes(mol);
    let ids: Vec<_> = mol.atom_ids().collect();
    for (id, class) in ids.into_iter().zip(classes) {
        mol.set_atom(id, keys::SYMMETRY_CLASS, saturating_i32(class))?;
    }
    Ok(())
}

/// Dense ranks of `values`: equal values share a rank, ranks follow the order.
fn ranks<T: Ord + Clone>(values: &[T]) -> Vec<usize> {
    let mut order: BTreeMap<T, usize> = BTreeMap::new();
    for value in values {
        order.entry(value.clone()).or_insert(0);
    }
    for (rank, slot) in order.values_mut().enumerate() {
        *slot = rank;
    }
    values.iter().map(|value| order[value]).collect()
}

/// Number of distinct values.
fn count(colours: &[usize]) -> usize {
    let mut seen = colours.to_vec();
    seen.sort_unstable();
    seen.dedup();
    seen.len()
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::core::{Element, NodeId, Props};
    use crate::props;

    fn atom(el: Element) -> Props {
        props! { keys::ATOMIC_NUMBER => el }
    }

    /// A molecule from elements and bonds `(i, j, bond_number)`.
    fn molecule(elements: &[Element], bonds: &[(usize, usize, i32)]) -> (Atomistic, Vec<NodeId>) {
        let mut mol = Atomistic::new();
        let ids: Vec<NodeId> = elements
            .iter()
            .map(|&el| mol.add_atom(atom(el)).unwrap())
            .collect();
        for &(i, j, order) in bonds {
            mol.add_bond(ids[i], ids[j], props! { keys::BOND_NUMBER => order })
                .unwrap();
        }
        (mol, ids)
    }

    #[test]
    fn a_carboxylates_oxygens_share_a_class_whatever_the_bond_orders() {
        // Acetate, written CH3-C(=O)-O(-): C0 C1 O2 O3 H4 H5 H6.
        let (mut mol, ids) = molecule(
            &[
                Element::C,
                Element::C,
                Element::O,
                Element::O,
                Element::H,
                Element::H,
                Element::H,
            ],
            &[
                (0, 1, 1),
                (1, 2, 2),
                (1, 3, 1),
                (0, 4, 1),
                (0, 5, 1),
                (0, 6, 1),
            ],
        );
        mol.set_atom(ids[3], keys::FORMAL_CHARGE, -1_i32).unwrap();
        assert_eq!(perceive_symmetry_classes(&mol), vec![0, 1, 2, 2, 4, 4, 4]);
    }

    #[test]
    fn ring_skeletons_split_by_distance_from_the_junction() {
        // Decalin's carbon skeleton (two fused 6-rings) and bicyclopentyl's (two
        // 5-rings joined by a bond): junction, next to the junction, the rest.
        let decalin = [
            (0, 1),
            (1, 2),
            (2, 3),
            (3, 4),
            (4, 5),
            (5, 0),
            (5, 6),
            (6, 7),
            (7, 8),
            (8, 9),
            (9, 0),
        ];
        let joined = [
            (0, 1),
            (1, 2),
            (2, 3),
            (3, 4),
            (4, 0),
            (0, 5),
            (5, 6),
            (6, 7),
            (7, 8),
            (8, 9),
            (9, 5),
        ];
        let carbons = [Element::C; 10];
        let with = |bonds: &[(usize, usize)]| {
            let b: Vec<_> = bonds.iter().map(|&(i, j)| (i, j, 1)).collect();
            molecule(&carbons, &b).0
        };
        let fused = perceive_symmetry_classes(&with(&decalin));
        let bridged = perceive_symmetry_classes(&with(&joined));
        assert_eq!(fused, vec![0, 1, 2, 2, 1, 0, 1, 2, 2, 1]);
        assert_eq!(bridged, vec![0, 1, 2, 2, 1, 0, 1, 2, 2, 1]);
    }

    #[test]
    fn bicyclooctanes_three_bridges_share_a_class_although_the_sssr_drops_a_ring() {
        // Bicyclo[2.2.2]octane: bridgeheads C0 and C1; bridges C2-C3, C4-C5, C6-C7.
        // Its three six-rings are equivalent and its SSSR holds two of them, so
        // one bridge lies in two SSSR rings and the other two in one.
        let bonds = [
            (0, 2, 1),
            (2, 3, 1),
            (3, 1, 1),
            (0, 4, 1),
            (4, 5, 1),
            (5, 1, 1),
            (0, 6, 1),
            (6, 7, 1),
            (7, 1, 1),
        ];
        let (mol, _) = molecule(&[Element::C; 8], &bonds);
        assert_eq!(
            perceive_symmetry_classes(&mol),
            vec![0, 0, 2, 2, 2, 2, 2, 2]
        );
    }

    #[test]
    fn assign_writes_the_class_row_on_every_atom() {
        let (mut mol, ids) = molecule(
            &[Element::O, Element::H, Element::H],
            &[(0, 1, 1), (0, 2, 1)],
        );
        assign_symmetry_classes(&mut mol).unwrap();
        let class = |id| {
            mol.get_atom(id)
                .unwrap()
                .get_int(keys::SYMMETRY_CLASS)
                .unwrap()
        };
        assert_eq!((class(ids[0]), class(ids[1]), class(ids[2])), (0, 1, 1));
    }
}
