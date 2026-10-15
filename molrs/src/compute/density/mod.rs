//! Density-related analyzers ported from `freud.density`.
//!
//! | Method | Measures |
//! |--------|----------|
//! | [`LocalDensity`] | per-particle number density in a sphere of radius `r_max` (Å) |
//! | [`GaussianDensity`] | grid-smeared density (a [`Gaussian`](crate::compute::Gaussian) spread on a [`RegularGrid`](crate::compute::RegularGrid)) |
//! | [`CorrelationFunction`] | distance-binned correlation `⟨A_i · B_j⟩(r)` of per-particle fields |
//! | [`SpatialDistribution`] | 3-D number density of a target species in a reference molecule's body-fixed frame |
//! | [`SphereVoxelization`] | boolean voxel rasterisation of particles as hard spheres |
//!
//! [`crate::compute::Rdf`] (the radial distribution function) lives in `crate::compute::rdf`
//! and is the prototypical member of this family; the others bin through the
//! same [`Axis`](crate::compute::Axis) and spread through the same
//! [`RegularGrid`](crate::compute::RegularGrid).

mod correlation_function;
mod gaussian_density;
mod local_density;
mod spatial_distribution;
mod sphere_voxelization;

pub use correlation_function::{
    CorrelationFunction, CorrelationFunctionInput, CorrelationFunctionResult,
};
pub use gaussian_density::{GaussianDensity, GaussianDensityInput, GaussianDensityResult};
pub use local_density::{LocalDensity, LocalDensityInput, LocalDensityResult};
pub use spatial_distribution::{
    SpatialDistribution, SpatialDistributionAccumulator, SpatialDistributionFrame,
    SpatialDistributionResult,
};
pub use sphere_voxelization::{
    SphereVoxelization, SphereVoxelizationInput, SphereVoxelizationResult,
};
