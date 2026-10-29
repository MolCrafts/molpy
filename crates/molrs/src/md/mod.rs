//! In-process molecular dynamics. `MD` walks the LAMMPS phase list and
//! launches the components bound to those phases. Energy kernels live in
//! [`crate::ff::potential`] and reach a step through `FfEvaluator`.
//!
//! A run is in one unit system, a [`UnitPreset`](crate::core::UnitPreset):
//! `dt` is in its time unit, and the step applies its
//! [`ftm2v`](crate::core::UnitPreset::ftm2v) and
//! [`mvv2e`](crate::core::UnitPreset::mvv2e) as LAMMPS does.
//!
//! `VelocityVerlet` takes only the timestep. Mass, positions, velocities and
//! forces are columns on the frame. `FixLangevin` adds its kick at
//! `post_force`. A pair kernel reads the pairs the run's one
//! [`NeighborList`](crate::core::NeighborList) publishes during `neighbor`
//! ([`MD::set_neighbor`](crate::md::MD::set_neighbor)). A [`BoundPotential`](crate::md::BoundPotential) is a
//! force provider of its own, mounted with
//! [`MD::set_bound_potential`](crate::md::MD::set_bound_potential).

mod barostat;
mod bound_potential;
mod columns;
mod component;
mod driver;
mod error;
mod evaluator;
mod fix_langevin;
mod images;
mod infra;
mod maxwell;
mod neighbor_list;
mod phase;
mod scratch;
mod verlet;

pub use barostat::BerendsenBarostat;
pub use bound_potential::{BoundPotential, PotentialError, PotentialResult, PotentialStep};
pub use component::{Barostat, Component, Integrator, Thermostat, TimestepProvider};
pub use driver::MD;
pub use error::MdError;
pub use evaluator::FfEvaluator;
pub use fix_langevin::FixLangevin;
pub use maxwell::MaxwellBoltzmann;
pub use phase::Phase;
pub use scratch::Scratch;
pub use verlet::VelocityVerlet;
