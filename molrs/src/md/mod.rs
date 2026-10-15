//! In-process molecular-dynamics engine: integrators and force providers.
//! Energy kernels are not part of `md`; they live in
//! [`crate::ff::potential`] and reach the integrators through a
//! [`ForceProvider`](crate::md::ForceProvider).
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
//! * Pair kernels implement [`crate::ff::potential::pair::PairPotential`];
//!   the force provider supplies current neighbour pairs via
//!   [`crate::ff::potential::Potential::calc_energy_forces_with_pairs`], so a
//!   potential neither owns nor updates the skin.
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

mod error;
mod forces;
mod ghost_topology;
mod integrators;
mod maxwell;
mod state;

// No re-exports of `ff` or `core` types here. `PairLjCut`, `PairPotential`,
// `Potential`, `Potentials` and `Virial` are owned by the modules that define
// them, and a second public spelling is a second name to keep true — the
// module doc above says where each lives, which is the pointer a reader needs.
pub use error::MdError;
pub use forces::{ForceProvider, GhostPairs, MicPairs, NeighborStats, SelfPairedForces};
pub use ghost_topology::BondedLists;
pub use integrators::{Langevin, VelocityVerlet, uniform_masses};
pub use maxwell::MaxwellBoltzmann;
pub use state::{ForceOutput, MdState};
