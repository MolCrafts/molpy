//! Bond-orientational order parameters ported from `freud.order`.
//!
//! | Method | Measures |
//! |--------|----------|
//! | [`Steinhardt`] | per-particle `q_ℓ` / `w_ℓ` (+ averaged variant) and the [`QlmTable`] per ℓ |
//! | [`Hexatic`] | 2-D k-atic bond order `ψ_k` |
//! | [`Nematic`] | nematic director + scalar order parameter from particle directors |
//! | [`Cubatic`] | cubatic order parameter via simulated annealing |
//! | [`SolidLiquid`] | Frenkel–ten Wolde solid/liquid bond classification from a [`QlmTable`] |
//! | [`ContinuousCoordination`] | continuous coordination numbers from a [`QlmTable`] |
//!
//! Every bond-orientational parameter is one per-particle bond-harmonic
//! average (`bond_average`): the spherical `Y_ℓm` for `q_ℓm`, the planar
//! `e^{ikθ}` for `ψ_k`. Every method is a [`Compute`](crate::compute::Compute)
//! over one snapshot's data; many snapshots go through
//! [`Batch`](crate::compute::Batch).

mod bond_average;
mod continuous_coordination;
mod cubatic;
mod hexatic;
mod nematic;
mod solid_liquid;
mod steinhardt;

pub use bond_average::QlmTable;
pub(crate) use bond_average::{BondBasis, SphericalHarmonic};
pub use continuous_coordination::{
    ContinuousCoordination, ContinuousCoordinationInput, ContinuousCoordinationResult,
};
pub use cubatic::{Cubatic, CubaticInput, CubaticResult};
pub use hexatic::{Hexatic, HexaticInput, HexaticResult};
pub use nematic::{Nematic, NematicInput, NematicResult};
pub use solid_liquid::{SolidLiquid, SolidLiquidInput, SolidLiquidResult};
pub use steinhardt::{Steinhardt, SteinhardtInput, SteinhardtResult};
