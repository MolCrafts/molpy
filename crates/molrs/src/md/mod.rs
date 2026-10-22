//! In-process molecular dynamics. `MD` walks the LAMMPS phase list and
//! launches the components bound to those phases. Energy kernels live in
//! [`crate::ff::potential`] and reach a step through `FfEvaluator`.
//!
//! `VelocityVerlet` takes only the timestep. Mass, positions, velocities and
//! forces are columns on the frame. `FixLangevin` adds its kick at
//! `post_force`. A pair kernel reads the pairs `NeighborList` publishes
//! during `neighbor`.

mod columns;
mod component;
mod driver;
mod error;
mod evaluator;
mod fix_langevin;
mod forces;
mod ghost_topology;
mod infra;
mod integrators;
mod maxwell;
mod neighbor_list;
mod phase;
mod scratch;
mod state;
mod verlet;

// The phase driver is the public surface. `forces`, `integrators`, and
// `state` remain while their callers move onto components; they are not
// re-exported.
pub use component::{Barostat, Component, Evaluator, Integrator, Thermostat, TimestepProvider};
pub use driver::MD;
pub use error::MdError;
pub use evaluator::FfEvaluator;
pub use fix_langevin::FixLangevin;
pub use maxwell::MaxwellBoltzmann;
pub use neighbor_list::NeighborList;
pub use phase::Phase;
pub use verlet::VelocityVerlet;
