//! The core data model: one public namespace, `molrs::core`.
//!
//! Every public name is flat on `molrs::core` — `molrs::core::Frame`,
//! `molrs::core::SimBox`, `molrs::core::Atomistic`, `molrs::core::Quantity`.
//! The implementation files are private. Three public submodules remain, each
//! a vocabulary whose names belong together:
//!
//! - [`keys`] — canonical column, block-group and frame-meta keys;
//! - [`schema`] — the column and block specifications those keys come from,
//!   and the validator that judges a frame against them;
//! - [`constants`] — physical and engine constants (CODATA 2018 / SI 2019,
//!   and the Coulomb and scaling constants each engine uses).
//!
//! What the flat surface holds:
//!
//! - columnar data: [`Block`], [`Frame`], [`Trajectory`] and their metadata;
//! - the molecular graph: [`MolGraph`], [`Atomistic`], [`CoarseGrain`],
//!   [`Port`], [`Topology`], [`Element`], [`BondOrder`];
//! - space: [`SimBox`], the regions ([`Region`], [`Sphere`], …), neighbour
//!   search ([`NeighborList`], [`VerletSkin`], …), [`GhostSet`],
//!   [`TriMesh`], [`Trace`];
//! - numerics and units: [`Complex`], the spherical harmonics and Wigner
//!   symbols, [`Virial`], [`Unit`], [`Quantity`], [`UnitRegistry`],
//!   [`UnitFactor`], [`UnitPreset`];
//! - the crate error, [`MolRsError`].
//!
//! # Units: one definition each
//!
//! Every unit conversion in molrs goes through the unit registry: a
//! [`UnitFactor`] written as its two units, each defined once in
//! [`unit_factors`] (`unit_factors::KCAL_TO_KJ`, resolved once),
//! [`UnitRegistry::factor`] or [`Quantity::to`]. No module writes a factor by hand (`* 4.184`,
//! `/ 10.0` for nm), and [`constants`] holds physical constants and the
//! constants engines define as data, never a conversion factor; the units
//! are built from those constants (`bohr` from [`constants::BOHR_RADIUS`],
//! `eV` from [`constants::ELEMENTARY_CHARGE`]). `module_boundaries` fails on
//! a conversion-factor constant or literal outside the units module.
//! Degrees ↔ radians is `to_radians` / `to_degrees` (the registry's `deg`
//! is the same π/180).
//!
//! Whole-graph transforms are methods of every graph type
//! ([`Atomistic::translate`], [`Atomistic::rotate`], [`Atomistic::scale`],
//! [`Atomistic::center`], and the same on [`CoarseGrain`] and [`MolGraph`]);
//! an algorithm that serves every graph type takes the sealed [`Graph`]
//! bound; ring
//! perception is `crate::perceive::perceive_rings`; the `*.mrec` record and
//! its force-field section are `crate::io::mrec`.
//!
//! # Examples
//!
//! ```
//! use molrs::core::Element;
//!
//! let hydrogen = Element::by_number(1).unwrap();
//! assert_eq!(hydrogen.symbol(), "H");
//! assert_eq!(Element::by_symbol("h").unwrap().name(), "Hydrogen");
//! ```

#![allow(missing_docs)]
#![warn(rustdoc::missing_crate_level_docs)]

// Vocabularies: the only public submodules.
pub mod constants;
pub mod keys;
pub mod schema;
pub mod unit_factors;

// Columnar data.
mod block;
mod frame;
mod frame_access;
mod frame_view;
mod metadata;
pub(crate) mod precision;
mod trajectory;
pub(crate) mod type_labels;
pub(crate) mod typed_json;

// The molecular graph.
mod atomistic;
pub(crate) mod bond_order;
mod bond_weights;
mod coarsegrain;
mod element;
mod entity_table;
mod extract;
mod graph;
pub(crate) mod graph_hash;
mod link;
mod molgraph;
mod molgraph_geometry;
mod port;
mod topology;

// Space.
pub(crate) mod bvh;
mod mesh;
pub(crate) mod neighbors;
mod periodic;
mod region;
mod simbox;
mod trace;

// Numerics, units, errors.
mod error;
mod math;
mod units;

#[cfg(all(test, feature = "rayon"))]
pub(crate) mod test_thread_pool;

pub use block::{
    Block, BlockAccess, BlockDtype, BlockError, BlockView, Column, ColumnArray, ColumnView, DType,
};
pub use frame::Frame;
pub use frame_access::FrameAccess;
pub use frame_view::FrameView;
pub use metadata::{MetaIter, MetaMap, MetaValue};
pub use precision::{
    PRECISION_MAX, PRECISION_MIN, check_precision, quantize, quantize_in_place, quantum,
};
pub use trajectory::{ObservableKind, ObservableRecord, ObservableValues, Trajectory};
pub use type_labels::{BlockTypeLabels, TypeLabels, TypeName};

pub use atomistic::{Atomistic, ExtractedAtomistic};
pub use bond_order::{BondNumber, BondOrder};
pub use bond_weights::BondDistanceWeights;
pub use coarsegrain::{CoarseGrain, ExtractedCoarseGrain};
pub use element::Element;
pub use entity_table::{EntityCell, EntityColumn, EntityDtype, EntityTable, Validity};
pub use extract::{ExtractedBall, InducedSubgraph};
pub use graph::Graph;
pub(crate) use graph::GraphAccess;
pub use link::{LinkError, LinkManyError};
pub use molgraph::{KindId, MolGraph, NodeId, PropValue, Props, Relation, RelationId};
pub use molgraph_geometry::CenterError;
pub use port::{Port, PortKind};
pub use topology::{Topology, TopologyError};

pub use mesh::{DEGENERATE_AREA2, TriMesh};
pub use neighbors::{
    AabbQuery, BruteForce, CellGrid, LinkCell, NeighborColumns, NeighborList, NeighborPair,
    NeighborPolicy, NeighborQuery, Neighbors, PairSource, QueryMode, SkinError, SkinPair,
    VerletSkin, filter_rad, filter_sann,
};
pub use periodic::{GhostError, GhostHalo, GhostSet, ImageRange};
pub use region::{
    AndRegion, Cuboid, Cylinder, Ellipsoid, HalfSpace, NotRegion, OrRegion, Parallelepiped,
    Polyhedron, PolyhedronError, Region, Sphere, SphereUnion, SphereUnionError,
};
pub use simbox::{BoxError, BoxKind, Cell, Mic, Points, SimBox};
pub use trace::Trace;

pub use error::MolRsError;
pub use math::{
    Complex, Virial, legendre_plm, wigner_3j, wigner_d_element, wigner_d_matrix, wigner_small_d,
    ylm_all, ylm_complex, ylm_normalization, ylm_real,
};
#[cfg(feature = "compute")]
pub(crate) use math::{FOUR_PI, FOUR_THIRDS_PI};
pub use units::{
    Dimension, PresetDim, Quantity, Unit, UnitDef, UnitFactor, UnitPreset, UnitPresetRegistry,
    UnitRegistry, UnitsError, lookup_unit_preset, register_unit_preset, replace_unit_preset,
    unit_preset_names,
};
