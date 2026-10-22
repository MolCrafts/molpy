//! Periodic geometry for the MD path: ghost atoms.
//!
//! A periodic system has no edge, but a potential wants a finite neighbourhood.
//! There are two ways to give it one. The minimum-image convention keeps `N`
//! atoms and fixes up every displacement; the ghost convention materialises the
//! copies and hands the potential an ordinary, non-periodic cluster.
//!
//! molrs uses both, and the split is by **path**, not by module:
//!
//! * **analysis** (`compute`, the neighbour-search backends) takes its
//!   periodicity from [`SimBox`](crate::core::SimBox) — see
//!   [`neighbors`](crate::core::neighbors), where the index holds `N` points
//!   and the lattice re-enters only through the minimum-image displacement;
//! * **MD force evaluation** uses this module: owned atoms plus ghosts, so the
//!   potential sees local geometry and never learns that periodic boundaries
//!   exist.
//!
//! The second is not a convenience. A minimum-image displacement is a fix-up a
//! potential has to be told about, and every many-body or machine-learned model
//! that reads positions rather than edge vectors gets it wrong. Ghosts move the
//! periodicity out of the potential entirely.
//!
//! # LAMMPS names, on purpose
//!
//! The three operations are spelled as LAMMPS spells them, because a domain
//! decomposition would change what they *do* and not what they are:
//! [`GhostSet::borders`] decides which copies exist, [`GhostSet::forward_comm`]
//! moves them to follow their owners, [`GhostSet::reverse_comm`] sums their
//! forces back. Serial, those are a search, a translation and a scatter-add.
//! Under MPI they are messages. Nothing above this layer is written twice.
//!
//! ```text
//!   owned atoms ──borders──────►  which copies exist  (a search; at rebuild)
//!        │                               │
//!        └───────forward_comm───────►  where they are  (a translation; per step)
//!                                        │
//!                             potential ─┴─► forces over [owned | ghost]
//!                                        │
//!                          reverse_comm ─┴─► forces on owned atoms
//! ```

mod ghosts;
mod halo;
mod images;

pub use ghosts::GhostSet;
pub use halo::GhostHalo;
pub use images::{GhostError, ImageRange};
