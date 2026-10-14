//! Shared `#[cfg(test)]` neighbor-list construction for the compute unit tests.
//!
//! The per-kernel test modules build their tables through the engine's column
//! entry point ([`get_positions_ref`] → [`NeighborList::build_columns`]),
//! reading the frame's `x`/`y`/`z` columns without an interleaved copy. It
//! yields the same pairs as the `N × 3` path (proven by
//! `engine_build_columns_matches_build`).

use molrs::core::FrameAccess;
use molrs::core::{NeighborColumns, NeighborList, Neighbors};
use molrs::op::F;

use super::positions::get_positions_ref;

/// Build a self-query [`Neighbors`] table for `frame` at `cutoff` through the
/// engine, using the frame's own simulation box.
///
/// Reads the `atoms` block's `x`/`y`/`z` columns with [`get_positions_ref`] (a
/// zero-copy borrow for contiguous columns) and hands them to
/// [`NeighborList::build_columns`], then materializes every column
/// ([`NeighborColumns::FULL`]) because the kernels under test read distances
/// and displacements alike.
///
/// Row order is unspecified — the cell-list backend materializes in parallel —
/// so a test that needs a fixed order sorts.
///
/// # Panics
/// Panics if the frame lacks `x`/`y`/`z` columns or has no simulation box.
pub(crate) fn nlist_from_frame<FA: FrameAccess>(frame: &FA, cutoff: F) -> Neighbors {
    let (xs, ys, zs) = get_positions_ref(frame).expect("frame must expose x/y/z columns");
    let simbox = frame
        .simbox_ref()
        .expect("frame must have a simulation box");
    let mut nl = NeighborList::new(cutoff);
    nl.build_columns(xs.slice(), ys.slice(), zs.slice(), simbox);
    nl.neighbors(NeighborColumns::FULL)
}

/// Build a self-query [`Neighbors`] table over the `N × 3` `points` in
/// `simbox` at `cutoff`, every column materialized
/// ([`NeighborColumns::FULL`]). Row order is unspecified.
pub(crate) fn self_neighbors(
    points: ndarray::ArrayView2<'_, F>,
    simbox: &molrs::core::SimBox,
    cutoff: F,
) -> Neighbors {
    let mut nl = NeighborList::new(cutoff);
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
