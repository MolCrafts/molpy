//! Reciprocal-space potential kernels (PME).
//!
//! Kept as a compilation-unit boundary so the FFT dependency can later be
//! gated out of the `ff` feature. This is not a ForceField category:
//! PME is registered as the pair style `coul/long/pme`. The real-space erfc
//! term is the pair form; this module is the reciprocal sum, the self
//! energy and the exclusion correction.

pub(crate) mod pme;

pub use pme::{PairCoulLongPme, PairCoulLongPmeParams, pair_coul_long_pme_constructor};
