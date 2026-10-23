//! Ghost atoms: the periodic copies of a point set that lie within a reach of
//! the cell faces, each with its owner and integer shift.
//!
//! Periodicity is the [`SimBox`](crate::core::SimBox)'s: the neighbour search
//! ([`NeighborList`](crate::core::NeighborList)) holds `N` points and the
//! lattice re-enters only through the box's minimum-image displacement. A
//! ghost set is for the analyses that need the copies themselves — a radical
//! Voronoi tessellation of a periodic cell is one — and is built once by
//! [`GhostSet::borders`].

mod ghosts;
mod images;

pub use ghosts::GhostSet;
pub use images::{GhostError, ImageRange};
