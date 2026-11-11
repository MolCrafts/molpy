use super::*;
use molrs::core::Cell;

fn cell(edge: F) -> SimBox {
    SimBox::new(Cell::Lengths([edge; 3]), [0.0; 3], [true; 3]).unwrap()
}

/// A regular hexagon of radius 1.4 Å in the plane z = `z` about (`cx`, `cy`),
/// wrapped into a cubic cell of edge `edge`: atoms 0..6.
fn hexagon(cx: F, cy: F, z: F, edge: F) -> Vec<[F; 3]> {
    (0..6)
        .map(|k| {
            let a = std::f64::consts::PI / 3.0 * k as F;
            [
                (cx + 1.4 * a.cos()).rem_euclid(edge),
                (cy + 1.4 * a.sin()).rem_euclid(edge),
                z.rem_euclid(edge),
            ]
        })
        .collect()
}

fn piercings(
    fraction: F,
    edge: F,
    mut atoms: Vec<[F; 3]>,
    extra: &[[F; 3]],
    bonds: &[[usize; 2]],
) -> Vec<(usize, usize)> {
    atoms.extend_from_slice(extra);
    let positions = Array2::from_shape_fn((atoms.len(), 3), |(i, a)| atoms[i][a]);
    let ring: Vec<usize> = (0..6).collect();
    let out = RingPiercing::new(fraction)
        .unwrap()
        .compute(RingPiercingInput {
            positions: positions.view(),
            simbox: &cell(edge),
            bonds,
            rings: &[ring],
        })
        .unwrap();
    out.ring_idx
        .iter()
        .zip(out.bond_idx.iter())
        .map(|(&r, &b)| (r, b))
        .collect()
}

#[test]
fn a_bond_through_the_centre_pierces_and_one_beside_the_ring_does_not() {
    let ring = hexagon(5.0, 5.0, 5.0, 20.0);
    let extra = [
        [5.0, 5.0, 4.0],
        [5.0, 5.0, 6.0], // 6-7: through the centre
        [9.0, 5.0, 4.0],
        [9.0, 5.0, 6.0], // 8-9: 4 Å off, outside the ring
        [5.0, 5.0, 6.0],
        [5.0, 5.0, 7.0], // 10-11: above the plane, never crosses it
    ];
    let bonds = [[6, 7], [8, 9], [10, 11]];
    assert_eq!(piercings(1.0, 20.0, ring, &extra, &bonds), vec![(0, 0)]);
}

#[test]
fn the_radius_fraction_sets_the_disc() {
    // Crosses the plane 1.0 Å from the centre: inside the 1.4 Å mean radius,
    // outside half of it.
    let extra = [[6.0, 5.0, 4.0], [6.0, 5.0, 6.0]];
    let bonds = [[6, 7]];
    assert_eq!(
        piercings(1.0, 20.0, hexagon(5.0, 5.0, 5.0, 20.0), &extra, &bonds),
        vec![(0, 0)]
    );
    assert!(piercings(0.5, 20.0, hexagon(5.0, 5.0, 5.0, 20.0), &extra, &bonds).is_empty());
}

#[test]
fn a_bond_sharing_a_ring_atom_does_not_pierce_it() {
    // From ring atom 0 straight down through the plane inside the disc.
    let extra = [[5.0, 5.0, 4.0]];
    let bonds = [[0, 6]];
    assert!(piercings(10.0, 20.0, hexagon(5.0, 5.0, 5.0, 20.0), &extra, &bonds).is_empty());
}

#[test]
fn a_ring_and_a_bond_cut_by_the_cell_faces_are_read_whole() {
    // The ring is centred on a corner of the cell (x = y = 0) and lies in
    // z = 0; the bond runs from z = 9.5 to z = 0.5 across the z face.
    let extra = [[0.0, 0.0, 9.5], [0.0, 0.0, 0.5]];
    let bonds = [[6, 7]];
    assert_eq!(
        piercings(1.0, 10.0, hexagon(0.0, 0.0, 0.0, 10.0), &extra, &bonds),
        vec![(0, 0)]
    );
}

#[test]
fn a_fraction_that_is_not_positive_is_refused() {
    for bad in [0.0, -1.0, F::NAN] {
        assert!(RingPiercing::new(bad).is_err(), "{bad}");
    }
}

#[test]
fn a_ring_of_two_atoms_is_refused() {
    let positions = Array2::<F>::zeros((3, 3));
    let err = RingPiercing::new(1.0)
        .unwrap()
        .compute(RingPiercingInput {
            positions: positions.view(),
            simbox: &cell(10.0),
            bonds: &[[0, 2]],
            rings: &[vec![0, 1]],
        })
        .unwrap_err();
    assert!(matches!(err, ComputeError::OutOfRange { .. }), "{err}");
}
