//! Errors from the in-process MD engine.

use std::fmt;

use molrs::core::{GhostError, SkinError, UnitStyle, UnitsError};

/// Failures on the MD surface: construction, binding, and policy guards.
#[derive(Debug)]
pub enum MdError {
    /// A constructor or bind argument is out of domain.
    Invalid(String),
    /// A required binding (force provider, mass) is missing.
    Unbound(String),
    /// Neighbour-list completeness / unwrapped-position guard failed.
    Neighbor(String),
    /// A [`BoundPotential`](super::BoundPotential) failed at `step`. The run stops
    /// there; `source` is the potential's own error, unchanged.
    Potential {
        /// [`BoundPotential::name`](super::BoundPotential::name).
        name: String,
        /// The step index of the failing evaluation ([`MD::step`](super::MD::step)).
        step: u64,
        /// What the potential raised.
        source: super::PotentialError,
    },
    /// An input states a units style other than the run's. Nothing is
    /// converted between styles, so the run refuses it.
    UnitsMismatch {
        /// What declared the style (`potential Springs`, `the frame`, …).
        what: String,
        /// The style it declares, as it spells it.
        declared: String,
        /// The run's style.
        run: UnitStyle,
    },
}

impl fmt::Display for MdError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Invalid(msg) | Self::Unbound(msg) | Self::Neighbor(msg) => f.write_str(msg),
            Self::Potential { name, step, source } => {
                write!(f, "potential {name} failed at step {step}: {source}")
            }
            Self::UnitsMismatch {
                what,
                declared,
                run,
            } => write!(
                f,
                "{what} is in units '{declared}' and the run is in units '{run}'; MD converts \
                 nothing between units styles, so give it inputs in '{run}' or run in \
                 '{declared}'"
            ),
        }
    }
}

impl std::error::Error for MdError {
    fn source(&self) -> Option<&(dyn std::error::Error + 'static)> {
        match self {
            Self::Potential { source, .. } => Some(source.as_ref()),
            _ => None,
        }
    }
}

impl From<SkinError> for MdError {
    fn from(err: SkinError) -> Self {
        match err {
            SkinError::Invalid(msg) => Self::Invalid(msg),
            SkinError::Guard(msg) => Self::Neighbor(msg),
        }
    }
}

impl From<UnitsError> for MdError {
    fn from(err: UnitsError) -> Self {
        Self::Invalid(format!("units: {err}"))
    }
}

impl From<GhostError> for MdError {
    fn from(err: GhostError) -> Self {
        Self::Invalid(err.to_string())
    }
}
