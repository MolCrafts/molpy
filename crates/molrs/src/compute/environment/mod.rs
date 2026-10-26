//! Local-environment analyzers ported from `freud.environment`.
//!
//! | Method | Measures |
//! |--------|----------|
//! | [`AngularSeparationGlobal`] / [`AngularSeparationNeighbor`] | pairwise angular separation between orientation quaternions (all-vs-global-refs / per-neighbor) |
//! | [`BondOrientationalOrder`] | 2-D `(θ, φ)` histogram of neighbor bond directions on the unit sphere |
//! | [`LocalBondProjection`] | projection of neighbor bond vectors onto reference directions |
//! | [`LocalDescriptors`] | per-pair spherical-harmonic descriptors of the local neighborhood (a [`FeatureMatrix`](crate::compute::FeatureMatrix)) |
//! | [`EnvironmentMatch`] | environment classes by neighbor-vector geometry (a [`Partition`](crate::compute::Partition)) |
//!
//! Every method is a [`Compute`](crate::compute::Compute) over one snapshot's
//! data; many snapshots go through [`Batch`](crate::compute::Batch).

mod angular_separation;
mod bond_orientational_order;
mod environment_match;
mod local_bond_projection;
mod local_descriptors;

pub use angular_separation::{
    AngularSeparationGlobal, AngularSeparationGlobalInput, AngularSeparationGlobalResult,
    AngularSeparationNeighbor, AngularSeparationNeighborInput, AngularSeparationNeighborResult,
};
pub use bond_orientational_order::{
    BondOrientationalOrder, BondOrientationalOrderInput, BondOrientationalOrderResult,
};
pub use environment_match::{EnvironmentMatch, EnvironmentMatchInput, EnvironmentMatchResult};
pub use local_bond_projection::{
    LocalBondProjection, LocalBondProjectionInput, LocalBondProjectionResult,
};
pub use local_descriptors::{LocalDescriptors, LocalDescriptorsInput, LocalDescriptorsResult};
