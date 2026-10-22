//! The force-field IR as vocabulary: what a category, a style and its
//! parameters *are*, stated as data.
//!
//! The IR adopts the LAMMPS standard for every style LAMMPS has
//! (`docs/developer/forcefield-ir.md`); this module states that
//! standard as data, so a new style or a new category is a registration
//! ([`crate::ff::style_registry`]), not an edit:
//!
//! * a **category** is a [`CategorySpec`] — its arity, the Frame block it
//!   prices, the [`Coordinate`] its energy is a function of;
//! * a **style** is a [`StyleSpec`] — its ordered per-type parameters, each
//!   with a [`ParamDimension`], its style parameters, where its numbers come
//!   from ([`ParamSource`]) and which special-bonds weights scale it
//!   ([`SpecialClass`]);
//! * the numbers themselves are [`Params`]; the force field's declarations
//!   the kernels read are [`CombiningRule`], [`SpecialBonds`] and the 1-4
//!   semantics of `lj/charmm` ([`OneFour`]);
//! * [`expression`] is the Lepton expression language a style's energy may
//!   be written in, with exact derivatives;
//! * a style's **engine forms** ([`Engine`], [`EngineCodec`]): its
//!   [`LammpsForm`] (positional, derived from the spec with conversion per
//!   [`ParamDimension`], or a [`LammpsCodec`] of its own) drives the LAMMPS
//!   reader and writer; every engine that cannot hold a style refuses it with
//!   [`IrError::NoEngineForm`];
//! * a style of a **form family** has a [`FormCodec`] — its exact maps to
//!   and from the family's canonical style ([`torsion`] for the torsions).
//!
//! The IR is the lowest layer of `ff`: it names no kernel, no registry and
//! no force field. Which kernel prices a style is
//! [`crate::ff::style_registry`]'s; converting a force field between the
//! styles of a family is [`crate::ff::form_conversion`]'s.

mod category;
mod combining_rule;
mod engine_codec;
mod error;
pub mod expression;
pub(crate) mod form;
mod one_four;
mod param_dimension;
mod param_source;
mod params;
mod spec;
mod special_bonds;
mod special_class;
mod style_table;

pub use category::{
    Arity, CategorySpec, Coordinate, EndpointOrder, builtin_categories, category_arity,
};
pub(crate) use combining_rule::same_lj;
pub use combining_rule::{COMBINING_RULES, CombiningRule};
pub use engine_codec::{
    Engine, EngineCodec, LammpsCodec, LammpsCoeffs, LammpsForm, Token, UnitScale, positional,
};
pub use error::IrError;
pub use form::torsion;
pub use form::{FormCodec, FormFn, FormRefusal, TypeParams};
pub(crate) use one_four::has_own_one_four;
pub use one_four::{ONE_FOUR, ONE_FOUR_EPSILON14, ONE_FOUR_REGULAR, ONE_FOUR_VALUES, OneFour};
pub use param_dimension::ParamDimension;
pub use param_source::ParamSource;
pub use params::{Params, pair_key};
pub use spec::{
    ConformanceSample, ParamCombination, ParamKind, ParamSpec, ParamValue, StyleSpec,
    builtin_styles,
};
pub(crate) use special_bonds::DEFAULT_SPECIAL_BONDS;
pub use special_bonds::SpecialBonds;
pub use special_class::SpecialClass;
pub use style_table::{
    ANNOTATION_COLUMNS, CMAP_GRID, ENDPOINT_COLUMNS, PROVENANCE_COLUMNS, is_parameter_column,
    is_unpriced_column,
};

#[cfg(test)]
mod tests;
