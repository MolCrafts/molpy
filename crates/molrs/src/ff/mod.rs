//! Force fields: what a force field *is*, and everything that turns a molecule
//! into one priced by it.
//!
//! Each submodule has one job, and its path is the only path to its items —
//! nothing is re-exported at this root.
//!
//! | module | its one job |
//! |---|---|
//! | [`ir`] | the force-field IR (adopts the LAMMPS standard), as vocabulary: categories, style specs, dimensions, parameters, combining rules, special bonds, the expression language, form codecs, engine forms |
//! | [`potential`] | kernels: a force field's terms evaluated on coordinates |
//! | [`style_registry`] | which kernel prices each IR style: the registry, its conformance checks, the built-in kernel table |
//! | [`forcefield`] | the [`ForceField`](forcefield::ForceField) data model: styles, types, their parameters, and the force field's own declarations (mixing, 1-4 pairs, its record section) |
//! | [`compile`] | binding a force field's styles to their kernels on a frame, with its 1-4 exceptions |
//! | [`form_conversion`] | rewriting a force field between the styles of one form family, exactly or by a least-squares fit |
//! | [`typifier`] | typing: a molecular graph in, the force-field types (and per-instance parameters) it carries out |
//! | [`charge`] | charge models: a molecule (and QM charges) in, partial charges out |
//! | [`params`] | every parameter table molrs ships, as compile-time data |
//! | [`clpol_scaling`] | CL&Pol's SAPT-derived Lennard-Jones scaling of a force field |
//!
//! The first six are layers, each naming only the layers above it in this
//! list (`module_boundaries` checks it): the IR's vocabulary, the kernels
//! that speak it, the registry that binds a style to its kernel, the force
//! field that declares styles, the compiler that binds a force field to its
//! kernels, and the conversions that need all of them. `params` names no
//! other `ff` module; a charge model types its atoms with a typifier, never
//! the reverse.
//!
//! No file format is here. Every file reader and writer — structure,
//! trajectory, and force-field files alike — is [`crate::io`]'s
//! ([`crate::io`] maps force-field files to and from a
//! [`ForceField`](forcefield::ForceField)). `ff` never depends on `io`
//! outside its `#[cfg(test)]` checks, which read engine files to compare
//! against.

pub mod charge;
pub mod clpol_scaling;
pub mod compile;
#[cfg(test)]
mod completeness;
#[cfg(test)]
mod engine_codec_check;
#[cfg(test)]
pub(crate) mod equivalence_check;
pub mod forcefield;
pub mod form_conversion;
pub mod ir;
#[cfg(test)]
mod ir_invariance;
#[cfg(test)]
mod one_four_lammps_check;
#[cfg(test)]
mod openmm_check;
pub mod params;
pub mod potential;
pub mod style_registry;
pub mod typifier;
