//! Errors from the in-process MD engine.

use std::fmt;

use molrs::core::{GhostError, SkinError, UnitsError};

/// Failures on the MD surface: construction, binding, and policy guards.
#[derive(Debug)]
pub enum MdError {
    /// A constructor or bind argument is out of domain.
    Invalid(String),
    /// A required binding (force provider, mass) is missing.
    Unbound(String),
    /// Neighbour-list completeness / unwrapped-position guard failed.
    Neighbor(String),
    /// A [`Calculator`](super::Calculator) failed at `step`. The run stops
    /// there; `source` is the calculator's own error, unchanged.
    Calculator {
        /// [`Calculator::name`](super::Calculator::name).
        name: String,
        /// The step index of the failing evaluation ([`MD::step`](super::MD::step)).
        step: u64,
        /// What the calculator raised.
        source: super::CalculatorError,
    },
}

impl fmt::Display for MdError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Invalid(msg) | Self::Unbound(msg) | Self::Neighbor(msg) => f.write_str(msg),
            Self::Calculator { name, step, source } => {
                write!(f, "calculator {name} failed at step {step}: {source}")
            }
        }
    }
}

impl std::error::Error for MdError {
    fn source(&self) -> Option<&(dyn std::error::Error + 'static)> {
        match self {
            Self::Calculator { source, .. } => Some(source.as_ref()),
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
