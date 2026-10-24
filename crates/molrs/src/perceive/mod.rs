//! Chemical perception algorithms operating on molecular graphs:
//! aromaticity, bond-type perception, hydrogen handling, ring detection,
//! stereochemistry, rotatable bonds, SMARTS matching, and subgraph matching
//! of coarse-grained bead graphs.
//!
//! *Perception* means deriving chemical facts that a connectivity graph
//! implies but does not state — which atoms lie on rings, which rings are
//! aromatic, which centres are chiral. SMARTS (SMILES Arbitrary Target
//! Specification) is the substructure query language matched here;
//! [`SubgraphMatcher`] is its coarse-grained counterpart, finding groups of
//! beads (one node per group of atoms) by bead type. Building a new
//! coarse-grained graph from such groups is construction, not perception: it
//! is `crate::builder::Coarsener`.
//!
//! Gasteiger charges are a *charge model*, not a perception: they sit with the
//! other charge models in `crate::ff::charge` (feature `ff`), reached through
//! the `ChargeModel` trait there.
//!
//! # Two verbs: `perceive_*` reports, `assign_*` writes
//!
//! Every perception is a free function, in one of two shapes — the split
//! RDKit makes between a query and an `Assign*` (`AssignStereochemistry`,
//! `AssignAtomChiralTagsFromStructure`), and the term the literature uses for
//! the computation itself (ring perception, bond-order perception):
//!
//! - **`perceive_<fact>(mol) -> table`** computes a fact and returns it as a
//!   side table, leaving the graph alone: [`perceive_rings`] → [`RingSet`],
//!   [`perceive_rotatable_bonds`], [`perceive_chiral_centers`],
//!   [`perceive_tetrahedral_stereo`], [`perceive_bond_stereo`],
//!   [`perceive_equivalence_classes`], [`perceive_bond_orders`],
//!   [`perceive_hybridizations`], [`perceive_conjugated_atoms`],
//!   [`perceive_ring_classes`].
//! - **`assign_<fact>(&mut mol) -> Result`** writes a perceived fact onto the
//!   graph itself, as atom / bond props: [`assign_rings`],
//!   [`assign_aromaticity`], [`assign_stereo`], [`assign_rotatable_bonds`],
//!   [`assign_bond_orders`], [`assign_kekule_bond_orders`],
//!   [`assign_bcc_bond_types`], [`assign_equivalence_classes`]. A caller that
//!   must keep its input clones it first.
//!
//! [`add_hydrogens`] / [`remove_hydrogens`] are edits of the graph, not
//! perceptions, and keep their verbs.
//! [`n_implicit_hydrogens`] is the per-atom count of the valence model they
//! and every other perception here share.
//!
//! | Function | Atom props | Bond props |
//! |---|---|---|
//! | [`assign_rings`] | `is_in_ring` (0/1), `n_rings` | `is_in_ring` (0/1), `n_rings` |
//! | [`assign_aromaticity`] | `is_aromatic` (0/1) | `bond_type` (aromatic), `bond_number` |
//! | [`assign_stereo`] | `stereo` (`"CW"` / `"CCW"`) | `stereo` (`"E"` / `"Z"` / `"either"`) |
//! | [`assign_rotatable_bonds`] | — | `is_rotatable` (0/1) |
//! | [`assign_bond_orders`] | — | `bond_number`, `bond_type` (antechamber's Kekulé structure) |
//! | [`assign_kekule_bond_orders`] | — | `bond_number` of every aromatic bond |
//! | [`assign_bcc_bond_types`] | — | `bcc_bond_type` (1/2/3/6/7/8/9), from the [`BondOrderSource`] named |
//! | [`assign_equivalence_classes`] | `equiv_class` (0-based class id) | — |
//!
//! The modules are private; every name is re-exported here, so the Rust path
//! `molrs::perceive::<name>` is the Python path `molpy.perceive.<name>`.

/// Executable specification for the Aromatic Bond Representation Standard.
/// Tests only; see the module docs for how to run the red line.
#[cfg(all(test, feature = "smiles"))]
mod aromatic_standard;
mod aromaticity;
mod bcc_bond_class;
mod bond_order;
mod equivalence;
mod hybridization;
mod hydrogens;
mod kekule;
#[cfg(feature = "ff")]
pub(crate) mod mmff_aromaticity;
mod ring_class;
mod rings;
mod rotatable;
// SMARTS matching compiles from the `io::smiles` parser, so it needs `smiles`.
#[cfg(feature = "smiles")]
pub mod smarts;
mod stereo;
mod subgraph;
mod valence;

pub use aromaticity::assign_aromaticity;
pub use bcc_bond_class::{BondOrderSource, assign_bcc_bond_types};
pub use bond_order::{assign_bond_orders, perceive_bond_orders};
pub use equivalence::{
    EquivalenceClasses, EquivalenceLevel, EquivalenceOptions, assign_equivalence_classes,
    perceive_equivalence_classes,
};
pub use hybridization::{Hybridization, perceive_conjugated_atoms, perceive_hybridizations};
pub use hydrogens::{add_hydrogens, remove_hydrogens};
pub use kekule::assign_kekule_bond_orders;
pub use ring_class::{
    AntechamberRingMembership, AntechamberRingSummary, RingClasses, perceive_ring_classes,
};
pub use rings::{RingSet, assign_rings, perceive_rings, small_ring_closure};
pub use rotatable::{
    RotatableBond, UnknownBondPolicy, assign_rotatable_bonds, downstream_atoms,
    perceive_rotatable_bonds,
};
pub use stereo::{
    BondStereo, TetrahedralStereo, assign_stereo, chiral_volume, perceive_bond_stereo,
    perceive_chiral_centers, perceive_tetrahedral_stereo,
};
pub(crate) use stereo::{
    TetrahedralSite, priority_substituent, stated_hydrogen_count, tetrahedral_reference,
};
pub use subgraph::SubgraphMatcher;
pub use valence::{n_hydrogens, n_implicit_hydrogens};
