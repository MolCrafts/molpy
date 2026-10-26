//! Hydrogen-bond analysis: geometric detection, and what the detected bonds
//! answer.
//!
//! Ported from the reference implementation (`src/hbond.cpp`,
//! `src/aggrtopo.cpp`):
//!
//! 1. [`HBonds`] — per-frame geometric D–H···A detection over donor `(D, H)`
//!    and acceptor selections, on the caller's core neighbour search
//!    (one cross-query per frame), gated by an [`HBondCriterion`]
//!    (Luzar–Chandler defaults) under the minimum image of the searched box.
//! 2. [`HBondsResult::components`] — the per-frame bond network's connected
//!    components, a label per atom (a [`Partition`](crate::compute::Partition)),
//!    via the native `core::Topology` (no petgraph).
//! 3. [`HBondsResult::presence`] — the per-bond presence series whose
//!    [`Survival`](crate::compute::Survival) correlations are the continuous
//!    `S_HB(t)` and intermittent `C_HB(t)` lifetime functions.
//!
//! Layer: `compute` → `core` only; WASM-clean; no new dependency.

mod criterion;
mod detect;

pub use criterion::{HBondCriterion, HBondDistanceKind};
pub use detect::{HBond, HBonds, HBondsInput, HBondsResult};
