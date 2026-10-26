//! Shared `#[cfg(test)]` neighbor-list construction for the compute unit tests.

use molrs::core::{NeighborColumns, NeighborList, Neighbors};
use molrs::op::F;

/// Build a self-query [`Neighbors`] table over the `N × 3` `points` in
/// `simbox` at `cutoff`, every column materialized
/// ([`NeighborColumns::FULL`]). Row order is unspecified.
pub(crate) fn self_neighbors(
    points: ndarray::ArrayView2<'_, F>,
    simbox: &molrs::core::SimBox,
    cutoff: F,
) -> Neighbors {
    let mut nl = NeighborList::new(cutoff).unwrap();
    nl.build(points, simbox);
    nl.neighbors(NeighborColumns::FULL)
}

/// `n` points uniform in the cube `[0, box_len)³`, drawn x column first, then
/// y, then z, from a seeded RNG.
pub(crate) fn random_points(n: usize, box_len: F, seed: u64) -> ndarray::Array2<F> {
    use rand::RngExt;
    use rand::SeedableRng;
    use rand::rngs::StdRng;

    let mut rng = StdRng::seed_from_u64(seed);
    let mut points = ndarray::Array2::<F>::zeros((n, 3));
    for c in 0..3 {
        for i in 0..n {
            points[[i, c]] = rng.random::<F>() * box_len;
        }
    }
    points
}

/// Build a self-query [`Neighbors`] table over `positions` in `simbox` at
/// `cutoff`, every column materialized. Row order is unspecified.
pub(crate) fn neighbors_of(
    positions: &[[F; 3]],
    simbox: &molrs::core::SimBox,
    cutoff: F,
) -> Neighbors {
    let points = ndarray::Array2::from_shape_fn((positions.len(), 3), |(i, c)| positions[i][c]);
    self_neighbors(points.view(), simbox, cutoff)
}
