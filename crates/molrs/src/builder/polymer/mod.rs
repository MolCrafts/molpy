//! Polymer ensembles: the generative-BigSMILES model of how chains of
//! random size and composition are drawn, as sites an
//! [`Assembler`](crate::builder::Assembler) builds.
//!
//! The model has four layers, each a type:
//!
//! | Type | What it is |
//! |---|---|
//! | [`Fragment`] | one ported template — a repeat unit, an end group, a fixed piece — with a [`SiteWeight`] per port |
//! | [`StochasticObject`] | repeat units and end groups grown to a size drawn from a [`ChainSize`] (a [`Distribution`] counted in a [`Measure`]) |
//! | [`Molecule`] | fixed fragments and stochastic objects joined end to end ([`Part`]) |
//! | [`System`] | molecule kinds sampled until each reaches its [`Amount`] of a total |
//!
//! [`Molecule::sample`] and [`System::sample`] give a [`CoarseGrain`] site
//! graph — one site per placed fragment, `bead_type` its fragment's name, a bond per
//! join carrying the two port ordinals — the same graph
//! [`CgSmilesIr::to_assembly_graph`](crate::io::cgsmiles::CgSmilesIr::to_assembly_graph)
//! gives, so one assembler serves both notations. The library it needs is
//! [`Molecule::library`] / [`System::library`].
//!
//! A generative-BigSMILES string reads into this model
//! ([`BigSmilesIr::to_system`](crate::io::bigsmiles::BigSmilesIr::to_system));
//! the types are also built directly, as the programmatic form of the same
//! model: an alternating copolymer is a list weight that pairs each fragment
//! only with the other, a block copolymer two objects in a row.
//!
//! # What is checked when, and what is not
//!
//! A model that can never be sampled is refused where it is built
//! ([`PolymerError::InvalidObject`], [`InvalidMolecule`](PolymerError::InvalidMolecule),
//! [`InvalidSystem`](PolymerError::InvalidSystem)): a terminal no repeat unit
//! accepts or that faces no part, a first object without an end group to
//! start from, a molecule that can measure zero. A port with nothing to join
//! is always an error, never a silently open end. What depends on the draw
//! is refused when sampled ([`PolymerError::Stuck`]): an open port no end
//! group accepts.
//!
//! Sampling itself takes the model as written: every object grows at least
//! one repeat unit, a system component stops at the first molecule reaching
//! its amount, and nothing is trimmed, retried or bounded.
//!
//! References: Schneider, L.; de Pablo, J. J. *Generative BigSMILES: an
//! extension for polymer informatics, computer simulations & ML/AI*, Digital
//! Discovery **3**, 51–61 (2024), DOI: 10.1039/D3DD00147D.
//!
//! [`CoarseGrain`]: crate::core::CoarseGrain

mod fragment;
mod molecule;
mod size;
mod stochastic;
mod system;
#[cfg(all(test, feature = "smiles"))]
mod tests;

use std::collections::HashMap;
use std::fmt;

pub use fragment::{Fragment, SiteWeight};
pub use molecule::{Molecule, Part};
pub use size::{ChainSize, Distribution, Measure};
pub use stochastic::{StochasticObject, Terminal};
pub use system::{Amount, MoleculeRecord, System, SystemSample};

/// Why a polymer model was refused or could not be sampled.
#[derive(Debug, Clone, PartialEq)]
pub enum PolymerError {
    /// A distribution parameter or weight is outside its range.
    InvalidParameter(String),
    /// A fragment was refused.
    InvalidFragment {
        /// The fragment's name.
        fragment: String,
        /// Why.
        reason: String,
    },
    /// A stochastic object was refused.
    InvalidObject(String),
    /// A molecule's parts do not join.
    InvalidMolecule(String),
    /// A system's components or amounts were refused.
    InvalidSystem(String),
    /// A stochastic object without a size distribution was sampled.
    Unsized,
    /// Growth could not place or cap a fragment its open ports require.
    Stuck(String),
    /// Writing the site graph failed.
    Sites(String),
}

impl fmt::Display for PolymerError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            PolymerError::InvalidParameter(m) => write!(f, "invalid parameter: {m}"),
            PolymerError::InvalidFragment { fragment, reason } => {
                write!(f, "invalid fragment '{fragment}': {reason}")
            }
            PolymerError::InvalidObject(m) => write!(f, "invalid stochastic object: {m}"),
            PolymerError::InvalidMolecule(m) => write!(f, "invalid molecule: {m}"),
            PolymerError::InvalidSystem(m) => write!(f, "invalid system: {m}"),
            PolymerError::Unsized => write!(
                f,
                "a stochastic object without a size distribution cannot be sampled"
            ),
            PolymerError::Stuck(m) => write!(f, "growth is stuck: {m}"),
            PolymerError::Sites(m) => write!(f, "writing the site graph failed: {m}"),
        }
    }
}

impl std::error::Error for PolymerError {}

/// Refuse two different fragments under one name among `fragments`; `invalid`
/// wraps the reason in the caller's error.
fn check_names<'a>(
    fragments: impl Iterator<Item = &'a Fragment>,
    invalid: fn(String) -> PolymerError,
) -> Result<(), PolymerError> {
    let mut seen: HashMap<&str, &Fragment> = HashMap::new();
    for fragment in fragments {
        if let Some(prior) = seen.insert(fragment.name(), fragment)
            && !prior.is_same(fragment)
        {
            return Err(invalid(format!(
                "two different fragments are named '{}' — build a fragment once and \
                 clone it wherever it is used",
                fragment.name()
            )));
        }
    }
    Ok(())
}
