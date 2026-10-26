//! In-process molecular dynamics. `MD` walks the LAMMPS phase list and
//! launches the components bound to those phases. Energy kernels live in
//! [`crate::ff::potential`] and reach a step through `FfEvaluator`.
//!
//! `VelocityVerlet` takes only the timestep. Mass, positions, velocities and
//! forces are columns on the frame. `FixLangevin` adds its kick at
//! `post_force`. A pair kernel reads the pairs the run's one
//! [`NeighborList`](crate::core::NeighborList) publishes during `neighbor`
//! ([`MD::set_neighbor`](crate::md::MD::set_neighbor)).

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

pub use component::{Component, Integrator, Thermostat, TimestepProvider};
pub use driver::MD;
pub use error::MdError;
pub use evaluator::FfEvaluator;
pub use fix_langevin::FixLangevin;
pub use maxwell::MaxwellBoltzmann;
pub use phase::Phase;
pub use verlet::VelocityVerlet;
