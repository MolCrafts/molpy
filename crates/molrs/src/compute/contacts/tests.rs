use super::*;
use molrs::core::{Cell, NeighborQuery, SimBox};
use ndarray::{Array2, array};

fn table(points: &Array2<F>, cutoff: F) -> Neighbors {
    let simbox = SimBox::new(Cell::Lengths([20.0; 3]), [0.0; 3], [true; 3]).unwrap();
    NeighborQuery::new(&simbox, points.view(), cutoff)
        .unwrap()
        .query_self()
}

/// A four-atom chain along x, 1 Å bonds, and a loose atom 0.5 Å off its end.
fn chain_and_stray() -> (Array2<F>, Vec<[usize; 2]>) {
    let points = array![
        [1.0, 1.0, 1.0],
        [2.0, 1.0, 1.0],
        [3.0, 1.0, 1.0],
        [4.0, 1.0, 1.0],
        [1.0, 1.5, 1.0],
    ];
    (points, vec![[0, 1], [1, 2], [2, 3]])
}

fn contacts(exclude_bonds: usize, cutoff: F) -> CloseContactsResult {
    let (points, bonds) = chain_and_stray();
    CloseContacts::new(exclude_bonds)
        .compute(CloseContactsInput {
            pairs: &table(&points, cutoff),
            bonds: &bonds,
        })
        .unwrap()
}

#[test]
fn bonded_pairs_are_not_contacts() {
    let out = contacts(1, 1.6);
    assert_eq!(out.pairs, array![[0, 4], [1, 4]]);
    assert!((out.distances[0] - 0.5).abs() < 1e-12);
    assert!((out.distances[1] - 1.25_f64.sqrt()).abs() < 1e-12);
}

#[test]
fn excluding_nothing_keeps_every_pair_of_the_table() {
    assert_eq!(contacts(0, 1.6).n_contacts(), 5);
}

#[test]
fn excluding_angle_ends_drops_the_one_three_pairs() {
    // At 2.05 Å the 1-3 pairs (0,2) and (1,3) are in the table; the 1-4
    // pair (0,3) at 3 Å is not, and the stray stays a contact of 0, 1, 2.
    let out = contacts(2, 2.05);
    assert_eq!(out.pairs, array![[0, 4], [1, 4]]);
    let out = contacts(1, 2.05);
    assert_eq!(out.pairs, array![[0, 2], [0, 4], [1, 3], [1, 4]]);
}

#[test]
fn a_bond_index_past_the_table_is_refused() {
    let (points, _) = chain_and_stray();
    let err = CloseContacts::new(1)
        .compute(CloseContactsInput {
            pairs: &table(&points, 1.6),
            bonds: &[[0, 9]],
        })
        .unwrap_err();
    assert!(matches!(err, ComputeError::OutOfRange { .. }), "{err}");
}

#[test]
fn a_cross_query_table_is_refused() {
    let (points, bonds) = chain_and_stray();
    let simbox = SimBox::new(Cell::Lengths([20.0; 3]), [0.0; 3], [true; 3]).unwrap();
    let cross = NeighborQuery::new(&simbox, points.view(), 1.6)
        .unwrap()
        .query(points.view());
    let err = CloseContacts::new(1)
        .compute(CloseContactsInput {
            pairs: &cross,
            bonds: &bonds,
        })
        .unwrap_err();
    assert!(matches!(err, ComputeError::BadShape { .. }), "{err}");
}
