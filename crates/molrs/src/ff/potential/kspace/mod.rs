//! Reciprocal-space potential kernels (PME).
//!
//! Kept as a compilation-unit boundary so the FFT dependency can later be
//! gated out of the `ff` feature. This is not a ForceField category:
//! PME is registered as the pair style `coul/long/pme`.

pub(crate) mod pme;

pub use pme::{PairCoulLongPme, PairCoulLongPmeParams, pair_coul_long_pme_constructor};
