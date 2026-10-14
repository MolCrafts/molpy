//! Space–time correlation dynamics: the van Hove function and pair-persistence
//! survival correlators. Plain time correlations are
//! [`Correlator`](crate::compute::Correlator)'s.
//!
//! | Method | Input | Output |
//! |--------|-------|--------|
//! | [`VanHove`] | trajectory frames | [`VanHoveResult`] — self/distinct G(r, t) |
//! | [`pair_survival_tcf`] | per-frame pair presence | [`PairSurvivalResult`] — survival time-correlation |
//!
//! These methods keep the full space- and/or time-resolved correlation
//! structure.
//!
//! ```ignore
//! let gvh = VanHove::new(n_rbins, r_max, vec![0, 10, 100])?.compute(&frames, ())?;
//! ```

mod pair_survival;
mod van_hove;

pub use pair_survival::{PairSurvivalResult, SurvivalMethod, pair_survival_tcf};
pub use van_hove::{VanHove, VanHoveResult};
