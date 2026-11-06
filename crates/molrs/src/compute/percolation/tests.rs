use super::*;
use crate::core::Cell;
use ndarray::array;

fn cell(lengths: [F; 3], pbc: [bool; 3]) -> SimBox {
    SimBox::new(Cell::Lengths(lengths), [0.0; 3], pbc).unwrap()
}

fn run(positions: Array2<F>, simbox: &SimBox, bonds: &[[usize; 2]]) -> PercolationResult {
    Percolation::new()
        .compute(PercolationInput {
            positions: positions.view(),
            simbox,
            bonds,
        })
        .unwrap()
}

#[test]
fn a_finite_cluster_does_not_percolate() {
    let simbox = cell([10.0; 3], [true; 3]);
    let out = run(array![[1.0, 1.0, 1.0], [2.0, 1.0, 1.0]], &simbox, &[[0, 1]]);
    assert_eq!(
        (out.n_components, out.percolation_dims.clone()),
        (1, vec![0])
    );
}

#[test]
fn a_ring_inside_the_cell_does_not_percolate() {
    let simbox = cell([10.0; 3], [true; 3]);
    let square = array![
        [1.0, 1.0, 1.0],
        [2.0, 1.0, 1.0],
        [2.0, 2.0, 1.0],
        [1.0, 2.0, 1.0]
    ];
    let out = run(square, &simbox, &[[0, 1], [1, 2], [2, 3], [3, 0]]);
    assert_eq!(out.percolation_dims, vec![0]);
}

#[test]
fn a_chain_closed_across_x_percolates_in_one_dimension() {
    let simbox = cell([10.0; 3], [true; 3]);
    let chain = array![
        [0.5, 5.0, 5.0],
        [3.0, 5.0, 5.0],
        [5.5, 5.0, 5.0],
        [8.0, 5.0, 5.0]
    ];
    let out = run(chain, &simbox, &[[0, 1], [1, 2], [2, 3], [3, 0]]);
    assert_eq!(out.percolation_dims, vec![1]);
    assert_eq!(out.translations[0], vec![[1, 0, 0]]);
}

#[test]
fn a_diagonal_chain_percolates_along_its_diagonal() {
    let simbox = cell([10.0; 3], [true; 3]);
    let chain = array![
        [0.5, 0.5, 5.0],
        [3.0, 3.0, 5.0],
        [5.5, 5.5, 5.0],
        [8.0, 8.0, 5.0]
    ];
    let out = run(chain, &simbox, &[[0, 1], [1, 2], [2, 3], [3, 0]]);
    assert_eq!(out.percolation_dims, vec![1]);
    assert_eq!(out.translations[0], vec![[1, 1, 0]]);
}

/// A square net (3 × 3 atoms, spacing 2 Å, in a 6 × 6 Å periodic cell),
/// optionally stacked along z (3 layers).
fn grid(layers: usize) -> (Array2<F>, Vec<[usize; 2]>) {
    let index = |x: usize, y: usize, z: usize| (z * 3 + y) * 3 + x;
    let mut positions = Vec::new();
    for z in 0..layers {
        for y in 0..3 {
            for x in 0..3 {
                positions.push([0.5 + 2.0 * x as F, 0.5 + 2.0 * y as F, 0.5 + 2.0 * z as F]);
            }
        }
    }
    let mut bonds = Vec::new();
    for z in 0..layers {
        for y in 0..3 {
            for x in 0..3 {
                bonds.push([index(x, y, z), index((x + 1) % 3, y, z)]);
                bonds.push([index(x, y, z), index(x, (y + 1) % 3, z)]);
                if layers == 3 {
                    bonds.push([index(x, y, z), index(x, y, (z + 1) % 3)]);
                }
            }
        }
    }
    let flat: Vec<F> = positions.into_iter().flatten().collect();
    (
        Array2::from_shape_vec((flat.len() / 3, 3), flat).unwrap(),
        bonds,
    )
}

#[test]
fn a_periodic_sheet_percolates_in_two_dimensions() {
    let simbox = cell([6.0, 6.0, 20.0], [true; 3]);
    let (positions, bonds) = grid(1);
    let out = run(positions, &simbox, &bonds);
    assert_eq!(out.percolation_dims, vec![2]);
    assert_eq!(out.translations[0], vec![[1, 0, 0], [0, 1, 0]]);
}

#[test]
fn a_periodic_net_percolates_in_three_dimensions() {
    let simbox = cell([6.0; 3], [true; 3]);
    let (positions, bonds) = grid(3);
    let out = run(positions, &simbox, &bonds);
    assert_eq!((out.n_components, out.max_dim()), (1, 3));
}

#[test]
fn components_are_judged_separately() {
    let simbox = cell([10.0; 3], [true; 3]);
    let positions = array![
        [0.5, 5.0, 5.0],
        [3.0, 5.0, 5.0],
        [5.5, 5.0, 5.0],
        [8.0, 5.0, 5.0],
        [5.0, 1.0, 1.0],
        [6.0, 1.0, 1.0],
    ];
    let out = run(
        positions,
        &simbox,
        &[[0, 1], [1, 2], [2, 3], [3, 0], [4, 5]],
    );
    assert_eq!(out.n_components, 2);
    assert_eq!(out.percolation_dims, vec![1, 0]);
    assert_eq!(out.component_sizes, vec![4, 2]);
}

#[test]
fn a_bond_minimum_image_cannot_read_is_refused() {
    // 2 Å bonds in a 4 Å cell: half the cell, so which image is meant is ambiguous.
    let simbox = cell([4.0, 10.0, 10.0], [true; 3]);
    let positions = array![[0.0, 5.0, 5.0], [2.0, 5.0, 5.0]];
    let refused = Percolation::new().compute(PercolationInput {
        positions: positions.view(),
        simbox: &simbox,
        bonds: &[[0, 1]],
    });
    assert!(
        matches!(refused, Err(ComputeError::OutOfRange { .. })),
        "{refused:?}"
    );
}

#[test]
fn a_non_periodic_axis_has_no_images() {
    // z is not periodic: atoms at z = 0.5 and 9.5 are 9 Å apart, a finite pair.
    let simbox = cell([10.0, 10.0, 30.0], [true, true, false]);
    let chain = array![[5.0, 5.0, 0.5], [5.0, 5.0, 2.5]];
    let out = run(chain, &simbox, &[[0, 1]]);
    assert_eq!(out.percolation_dims, vec![0]);
}

#[test]
fn unwrap_images_make_a_finite_molecule_whole() {
    let simbox = cell([10.0; 3], [true; 3]);
    let split = array![[9.5, 5.0, 5.0], [0.5, 5.0, 5.0]];
    let out = run(split, &simbox, &[[0, 1]]);
    assert_eq!(out.percolation_dims, vec![0]);
    assert_eq!(out.unwrap_images.row(1).to_vec(), vec![1, 0, 0]);
}

#[test]
fn the_lattice_basis_is_exact() {
    let mut lattice = Lattice::default();
    lattice.insert([2, 0, 0]);
    lattice.insert([3, 0, 0]);
    assert_eq!(lattice.rows, vec![[1, 0, 0]]);
    lattice.insert([1, 2, 0]);
    lattice.insert([0, 4, 0]);
    assert_eq!(lattice.rows.len(), 2);
    assert_eq!(lattice.rows[1][1], 2);
}
