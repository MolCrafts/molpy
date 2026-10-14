//! Space–time correlation dynamics: reorientation, the van Hove function and
//! pair-persistence survival correlators.
//!
//! | Method | Input | Output |
//! |--------|-------|--------|
//! | [`Oacf`] | a unit-vector series | [`TimeCorrelation`](crate::compute::TimeCorrelation) — `C_ℓ(t) = ⟨P_ℓ(û(0)·û(t))⟩` |
//! | [`VanHove`] | trajectory frames | [`VanHoveResult`] — self/distinct G(r, t) |
//! | [`pair_survival_tcf`] | per-frame pair presence | [`PairSurvivalResult`] — survival time-correlation |
//!
//! Complements [`transport`](crate::compute::transport) (which reduces motion
//! to scalar coefficients): these methods keep the full space- and/or
//! time-resolved correlation structure. The bare time-correlation engine —
//! [`Acf`](crate::compute::Acf) and the `autocorrelate` family — lives in
//! [`correlation`](crate::compute::correlation).
//!
//! ```ignore
//! let gvh = VanHove::new(n_rbins, r_max, vec![0, 10, 100])?.compute(&frames, ())?;
//! ```

mod oacf;
mod pair_survival;
mod van_hove;

pub use oacf::{Oacf, OacfArgs};
pub use pair_survival::{PairSurvivalResult, SurvivalMethod, pair_survival_tcf};
pub use van_hove::{VanHove, VanHoveResult};
