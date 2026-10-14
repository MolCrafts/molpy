//! Space–time dynamics: the van Hove function and pair presence. Plain time
//! correlations are [`Correlator`](crate::compute::Correlator)'s; survival
//! correlations are [`Survival`](crate::compute::Survival) on a [`Presence`].
//!
//! | Method | Input | Output |
//! |--------|-------|--------|
//! | [`VanHove`] | unwrapped trajectory + boxes | [`VanHoveResult`] — self/distinct G(r, t) |
//! | [`PairPresence`] | one neighbour table per frame | [`Presence`] — 0/1 series per pair, for [`Survival`](crate::compute::Survival) |

mod presence;
mod van_hove;

pub use presence::{PairPresence, PairPresenceInput, Presence};
pub use van_hove::{VanHove, VanHoveInput, VanHoveResult};
