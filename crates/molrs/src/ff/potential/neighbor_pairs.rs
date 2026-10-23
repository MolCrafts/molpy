//! The intramolecular `pairs` block of
//! [`intramolecular_pairs`](super::intramolecular_pairs) over the pairs of a
//! neighbour table instead of every pair of the molecule.

use std::collections::HashSet;

use ndarray::Array1;

use crate::ff::ir::SpecialBonds;
use molrs::core::Block;
use molrs::core::Frame;
use molrs::core::Neighbors;
use molrs::core::keys::{ATOMI, ATOMJ, IS_14};
use molrs::core::schema::block_names::ATOMS;
use molrs::op::Idx;

use super::{bond_classes, carried_overrides};

/// Build the intramolecular non-bonded `pairs` block (`atomi`, `atomj`,
/// `is_14`) from the pairs of `neighbors` — a cutoff-bounded neighbour table —
/// by the rules of [`intramolecular_pairs`](super::intramolecular_pairs): a
/// 1-2 / 1-3 pair (by bond distance) is dropped unless `special` keeps its
/// class, and a pair three bonds apart is flagged 1-4.
///
/// Each unordered pair appears once, as `(lo, hi)`, in the order the table
/// first reports it, so a full (both-direction) table and a half-shell one
/// give the same block. Per-pair override cells of the frame's own `pairs`
/// rows move onto the new rows exactly as in `intramolecular_pairs`; an
/// override on a pair the new list leaves out is an [`Err`].
///
/// # Errors
///
/// When `special`'s weights cannot be expressed by inclusion
/// ([`SpecialBonds::compiled_inclusion`]), or an override would be dropped.
pub(super) fn over_neighbors(
    frame: &Frame,
    special: &SpecialBonds,
    neighbors: &Neighbors,
) -> Result<Block, String> {
    let [keep_12, keep_13] = special.compiled_inclusion()?;
    let n_atoms = frame.get(ATOMS).and_then(|b| b.n_rows()).unwrap_or(0);
    let classes = bond_classes(frame, n_atoms)?;

    let mut pi: Vec<Idx> = Vec::new();
    let mut pj: Vec<Idx> = Vec::new();
    let mut p14: Vec<bool> = Vec::new();
    let mut seen = HashSet::new();
    for (&a, &b) in neighbors
        .query_point_indices()
        .iter()
        .zip(neighbors.point_indices())
    {
        let (lo, hi) = (a.min(b) as usize, a.max(b) as usize);
        if lo == hi || !seen.insert((lo, hi)) {
            continue;
        }
        let class = classes.get(&(lo, hi)).copied();
        if (!keep_12 && class == Some(1)) || (!keep_13 && class == Some(2)) {
            continue;
        }
        pi.push(lo as Idx);
        pj.push(hi as Idx);
        p14.push(class == Some(3));
    }

    let mut pairs = Block::new();
    if pi.is_empty() {
        return Ok(pairs);
    }
    let overrides = carried_overrides(frame, &pi, &pj)?;
    pairs
        .insert(ATOMI, Array1::from_vec(pi).into_dyn(), None)
        .expect("fresh pairs block");
    pairs
        .insert(ATOMJ, Array1::from_vec(pj).into_dyn(), None)
        .expect("fresh pairs block");
    pairs
        .insert(IS_14, Array1::from_vec(p14).into_dyn(), None)
        .expect("fresh pairs block");
    for (key, values, valid) in overrides {
        pairs
            .insert(key, Array1::from_vec(values).into_dyn(), Some(valid))
            .map_err(|e| e.to_string())?;
    }
    Ok(pairs)
}

#[cfg(test)]
mod tests {
    use super::*;
    use molrs::core::keys::{ATOMK, ATOML};
    use molrs::core::schema::block_names::{ANGLES, BONDS, DIHEDRALS};
    use molrs::core::{NeighborColumns, NeighborPair, QueryMode};

    /// A four-atom chain 0-1-2-3 with its bonds, angles and dihedral.
    fn chain() -> Frame {
        let block = |cols: &[(&str, Vec<Idx>)]| {
            let mut b = Block::new();
            for (k, v) in cols {
                b.insert(*k, Array1::from_vec(v.clone()).into_dyn(), None)
                    .unwrap();
            }
            b
        };
        let mut frame = Frame::new();
        let mut atoms = Block::new();
        atoms
            .insert("x", Array1::from_vec(vec![0.0_f64; 4]).into_dyn(), None)
            .unwrap();
        frame.insert(molrs::core::schema::block_names::ATOMS, atoms);
        frame.insert(
            BONDS,
            block(&[(ATOMI, vec![0, 1, 2]), (ATOMJ, vec![1, 2, 3])]),
        );
        frame.insert(
            ANGLES,
            block(&[
                (ATOMI, vec![0, 1]),
                (ATOMJ, vec![1, 2]),
                (ATOMK, vec![2, 3]),
            ]),
        );
        frame.insert(
            DIHEDRALS,
            block(&[
                (ATOMI, vec![0]),
                (ATOMJ, vec![1]),
                (ATOMK, vec![2]),
                (ATOML, vec![3]),
            ]),
        );
        frame
    }

    /// A table over the four atoms: half-shell when every pair has `i < j`,
    /// otherwise a cross-query of the set against itself.
    fn table(pairs: &[(u32, u32)]) -> Neighbors {
        let rows = pairs.iter().map(|&(i, j)| NeighborPair {
            i,
            j,
            dist_sq: 1.0,
            disp: [1.0, 0.0, 0.0],
        });
        let mode = if pairs.iter().all(|&(i, j)| i < j) {
            QueryMode::SelfQuery { n_points: 4 }
        } else {
            QueryMode::CrossQuery {
                n_query_points: 4,
                n_points: 4,
            }
        };
        Neighbors::from_pairs(rows, NeighborColumns::FULL, mode)
    }

    #[test]
    fn excludes_bonded_and_angle_pairs_and_flags_the_dihedral_ends() {
        // Both directions of every pair, as a full table reports them.
        let all: Vec<(u32, u32)> = (0..4u32)
            .flat_map(|i| (0..4u32).filter(move |&j| j != i).map(move |j| (i, j)))
            .collect();
        let pairs = crate::ff::potential::intramolecular_pairs(
            &chain(),
            &SpecialBonds::EXCLUDE_12_13,
            Some(&table(&all)),
        )
        .unwrap();
        let i = pairs
            .get(ATOMI)
            .unwrap()
            .get::<u64>()
            .unwrap()
            .iter()
            .copied()
            .collect::<Vec<_>>();
        let j = pairs
            .get(ATOMJ)
            .unwrap()
            .get::<u64>()
            .unwrap()
            .iter()
            .copied()
            .collect::<Vec<_>>();
        let f = pairs
            .get(IS_14)
            .unwrap()
            .get::<bool>()
            .unwrap()
            .iter()
            .copied()
            .collect::<Vec<_>>();
        assert_eq!((i, j, f), (vec![0], vec![3], vec![true]));
    }

    #[test]
    fn agrees_with_the_full_list_on_a_complete_table() {
        let all: Vec<(u32, u32)> = (0..4u32)
            .flat_map(|i| ((i + 1)..4u32).map(move |j| (i, j)))
            .collect();
        let special = SpecialBonds::EXCLUDE_12_13;
        let from_table =
            crate::ff::potential::intramolecular_pairs(&chain(), &special, Some(&table(&all)))
                .unwrap();
        let full = crate::ff::potential::intramolecular_pairs(&chain(), &special, None).unwrap();
        for key in [ATOMI, ATOMJ] {
            assert_eq!(
                from_table.get(key).unwrap().get::<u64>().unwrap(),
                full.get(key).unwrap().get::<u64>().unwrap()
            );
        }
    }
}
