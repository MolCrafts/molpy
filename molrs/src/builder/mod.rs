//! Structure builders — produce molecular graphs / paths from parameters.
//!
//! This is the inverse direction of `crate::compute` (Frame → analysis; a
//! separate feature, so not linked here).
//! Everything that *constructs* structure from a few scalars lives here:
//!
//! | Builder | Output |
//! |---------|--------|
//! | [`GrapheneBuilder`] | flat honeycomb sheet [`Frame`](crate::core::Frame) |
//! | [`CarbonTubeBuilder`] | rolled SWCNT [`Frame`](crate::core::Frame) (exact graphene quotient) |
//! | [`SelfAvoidingWalk`] | multi-chain [`Trace`](crate::core::Trace)s + [`SimBox`](crate::core::SimBox) (no chemistry) |
//! | [`SitePlacer`], [`GrowthPlacer`] (each a [`Placer`]) | one pose ([`Rigid`](crate::op::Rigid)) per site: centre of mass on the site, or grown onto the parent's port |
//! | [`AxisOrienter`] (an [`Orienter`]) | one rotation per site, about the template's centre of mass: chain units onto the site axis and bond line, branch units by port-direction fit |
//! | [`Assembler`] | one placed, linked world graph (of the library templates' graph type) from a site graph (`frag_id` per site, `mol_id` per connected component) |
//! | [`Coarsener`] | a coarse-grained graph from disjoint node groups of a source graph (any graph type): one site per group at its centre of mass, with its axis |
//!
//! The SARW path generator is a clean-room port of the kernel from the CAVS
//! LAMMPS tutorial `mc_gen.c` (Mark A. Tschopp & Don K. Ward), with chemistry
//! and file I/O stripped; FCC is one [`GrowthStrategy`] among others.

mod assemble;
mod carbon_tube;
mod coarsen;
mod graphene;
mod growth_strategy;
mod occupancy;
mod orient;
mod place;
mod self_avoiding_walk;

pub use assemble::{AssembleError, Assembler};
pub use carbon_tube::{CarbonTubeBuilder, CarbonTubeError, TubeExtent};
pub use coarsen::{CoarsenError, Coarsener};
pub use graphene::{GrapheneBuilder, GrapheneError};
pub use growth_strategy::{FccLattice, GrowthStrategy, OffLattice};
pub use occupancy::OccupancyMode;
pub use orient::{AxisOrienter, OrientError, Orienter, SiteLink, SiteView};
pub use place::{GrowthPlacer, ParentJoin, PlaceError, PlaceSite, Placer, SitePlacer};
pub use self_avoiding_walk::{SelfAvoidingWalk, WalkError, WalkOutput};
