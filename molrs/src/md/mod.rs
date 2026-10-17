//! In-process molecular dynamics. [`MD`] walks the LAMMPS phase list and
//! launches components bound to those phases. Energy kernels live in
//! [`crate::ff::potential`] and reach the step through [`FfEvaluator`].
//!
//! Neighbour lists: [`crate::core::NeighborList`] (`NeighborList`,
//! `VerletSkin`). Science here:
//!
//! * [`crate::ff::potential::Potential`] produces energy and forces from flat
//!   coordinates. [`crate::ff::potential::ForceTerm`] is one term of a force
//!   evaluation with the part it plays already chosen — `Indexed` for a bonded
//!   term that takes an index table, `Pair` for one summed over a neighbour
//!   table, `Plain` for an external field. The providers below hold a
//!   `Vec<ForceTerm>` and match on it, so no step re-derives which is which.
//!   [`crate::ff::potential::Potentials`] sums them as one potential.
//! * Pair kernels implement [`crate::ff::potential::PairDriven`]; the force
//!   provider supplies current neighbour pairs via
//!   [`crate::ff::potential::PairDriven::evaluate_pairs`], so a potential
//!   neither owns nor updates the skin.
//! * [`crate::md::ForceProvider`] is the force-field seam: the potential, the
//!   neighbour bookkeeping and the periodic régime all live behind it, and the
//!   integrator sees only energy, forces and virial. [`crate::md::SelfPairedForces`],
//!   [`crate::md::MicPairs`] and [`crate::md::GhostPairs`] are the three that
//!   ship; a new way to make a force is a new implementor, not a new variant.
//! * [`crate::md::VelocityVerlet`] / [`crate::md::Langevin`] — constructed with
//!   `(dt, forces, mass, simbox)` (Langevin also `gamma`, `kbt`, `seed`). The
//!   provider owns the neighbour state, so the integrator neither holds a skin
//!   nor knows whether one exists.
//!
//! * The kinetic readings of a state (kinetic energy, temperature,
//!   centre-of-mass velocity) are analyses, and live in [`crate::compute`]
//!   ([`crate::compute::KineticEnergy`] and its siblings); `md` holds
//!   integrators and force providers only.
//!
//! # Periodic régimes
//!
//! Two periodic régimes reach the same physics by different routes, and a run
//! picks one by picking a [`ForceProvider`](crate::md::ForceProvider):
//!
//! * [`MicPairs`](crate::md::MicPairs) keeps `N` atoms and fixes up every
//!   displacement with the minimum-image convention. Cheapest, and correct for
//!   any potential that consumes edge vectors.
//! * [`GhostPairs`](crate::md::GhostPairs) materialises the periodic copies
//!   — [`GhostHalo`](crate::core::GhostHalo) (in `core`) owns their
//!   lifecycle — and hands the potential an ordinary, non-periodic cluster. It costs the copies, and it is the only route that
//!   is correct for a potential reading *positions*: a many-body or
//!   machine-learned model cannot be told about a minimum-image fix-up it does
//!   not know to apply.
//!
//! No `bind_*` façades. Compose required pieces in the constructor.

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
