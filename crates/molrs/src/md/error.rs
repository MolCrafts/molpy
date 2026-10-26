//! Errors from the in-process MD engine.

use std::fmt;

use molrs::core::{GhostError, SkinError};

/// Failures on the MD surface: construction, binding, and policy guards.
#[derive(Debug)]
pub enum MdError {
    /// A constructor or bind argument is out of domain.
    Invalid(String),
    /// A required binding (force provider, mass) is missing.
    Unbound(String),
    /// Neighbour-list completeness / unwrapped-position guard failed.
    Neighbor(String),
}

impl fmt::Display for MdError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Invalid(msg) | Self::Unbound(msg) | Self::Neighbor(msg) => f.write_str(msg),
        }
    }
}

impl std::error::Error for MdError {}

impl From<SkinError> for MdError {
    fn from(err: SkinError) -> Self {
        match err {
            SkinError::Invalid(msg) => Self::Invalid(msg),
            SkinError::Guard(msg) => Self::Neighbor(msg),
        }
    }
}

impl From<GhostError> for MdError {
    fn from(err: GhostError) -> Self {
        Self::Invalid(err.to_string())
    }
}
