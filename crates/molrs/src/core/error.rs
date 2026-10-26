//! Unified error types for the molrs library.

use std::fmt;
use std::io;

use crate::core::BlockError;
use crate::core::UnitsError;

/// Main error type for the molrs library.
#[derive(Debug)]
pub enum MolRsError {
    /// Error from Block operations
    Block(BlockError),

    /// Error from the units subsystem
    Units(UnitsError),

    /// IO error (file reading/writing)
    Io(io::Error),

    /// Parse error with context
    Parse {
        /// Line number where error occurred (if applicable)
        line: Option<usize>,
        /// Error message
        message: String,
    },

    /// Validation error
    Validation {
        /// Error message
        message: String,
    },

    /// Zarr I/O error
    Zarr {
        /// Error message
        message: String,
    },

    /// Entity not found (atom, bond, angle, dihedral)
    NotFound {
        /// Kind of entity ("atom", "bond", "angle", "dihedral")
        entity: &'static str,
        /// Human-readable message
        message: String,
    },

    /// Generic error with message
    Other(String),
}

impl fmt::Display for MolRsError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            MolRsError::Block(e) => write!(f, "Block error: {}", e),
            MolRsError::Units(e) => write!(f, "Units error: {}", e),
            MolRsError::Io(e) => write!(f, "IO error: {}", e),
            MolRsError::Parse {
                line: Some(line),
                message,
            } => {
                write!(f, "Parse error at line {}: {}", line, message)
            }
            MolRsError::Parse {
                line: None,
                message,
            } => {
                write!(f, "Parse error: {}", message)
            }
            MolRsError::Validation { message } => {
                write!(f, "Validation error: {}", message)
            }
            MolRsError::NotFound { entity, message } => {
                write!(f, "{} not found: {}", entity, message)
            }
            MolRsError::Zarr { message } => {
                write!(f, "Zarr error: {}", message)
            }
            MolRsError::Other(msg) => write!(f, "{}", msg),
        }
    }
}

impl std::error::Error for MolRsError {
    fn source(&self) -> Option<&(dyn std::error::Error + 'static)> {
        match self {
            MolRsError::Block(e) => Some(e),
            MolRsError::Units(e) => Some(e),
            MolRsError::Io(e) => Some(e),
            _ => None,
        }
    }
}

// Automatic conversions
impl From<BlockError> for MolRsError {
    fn from(err: BlockError) -> Self {
        MolRsError::Block(err)
    }
}

impl From<UnitsError> for MolRsError {
    fn from(err: UnitsError) -> Self {
        MolRsError::Units(err)
    }
}

impl From<io::Error> for MolRsError {
    fn from(err: io::Error) -> Self {
        MolRsError::Io(err)
    }
}

impl From<String> for MolRsError {
    fn from(msg: String) -> Self {
        MolRsError::Other(msg)
    }
}

impl From<&str> for MolRsError {
    fn from(msg: &str) -> Self {
        MolRsError::Other(msg.to_string())
    }
}

#[cfg(feature = "zarr")]
impl From<zarrs::group::GroupCreateError> for MolRsError {
    fn from(e: zarrs::group::GroupCreateError) -> Self {
        MolRsError::zarr(e.to_string())
    }
}

#[cfg(feature = "zarr")]
impl From<zarrs::storage::StorageError> for MolRsError {
    fn from(e: zarrs::storage::StorageError) -> Self {
        MolRsError::zarr(e.to_string())
    }
}

#[cfg(feature = "zarr")]
impl From<zarrs::array::ArrayCreateError> for MolRsError {
    fn from(e: zarrs::array::ArrayCreateError) -> Self {
        MolRsError::zarr(e.to_string())
    }
}

#[cfg(feature = "zarr")]
impl From<zarrs::array::ArrayError> for MolRsError {
    fn from(e: zarrs::array::ArrayError) -> Self {
        MolRsError::zarr(e.to_string())
    }
}

#[cfg(feature = "zarr")]
impl From<zarrs::node::NodeCreateError> for MolRsError {
    fn from(e: zarrs::node::NodeCreateError) -> Self {
        MolRsError::zarr(e.to_string())
    }
}

// Helper constructors
impl MolRsError {
    /// Create a parse error with line number
    pub fn parse_error(line: usize, message: impl Into<String>) -> Self {
        MolRsError::Parse {
            line: Some(line),
            message: message.into(),
        }
    }

    /// Create a parse error without line number
    pub fn parse(message: impl Into<String>) -> Self {
        MolRsError::Parse {
            line: None,
            message: message.into(),
        }
    }

    /// Create a validation error
    pub fn validation(message: impl Into<String>) -> Self {
        MolRsError::Validation {
            message: message.into(),
        }
    }

    /// Create a not-found error
    pub fn not_found(entity: &'static str, message: impl Into<String>) -> Self {
        MolRsError::NotFound {
            entity,
            message: message.into(),
        }
    }

    /// Create a Zarr I/O error
    pub fn zarr(message: impl Into<String>) -> Self {
        MolRsError::Zarr {
            message: message.into(),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::core::BlockError;

    #[test]
    fn test_error_display() {
        let err = MolRsError::parse_error(42, "unexpected token");
        assert_eq!(
            format!("{}", err),
            "Parse error at line 42: unexpected token"
        );

        let err = MolRsError::parse("invalid format");
        assert_eq!(format!("{}", err), "Parse error: invalid format");

        let err = MolRsError::validation("inconsistent dimensions");
        assert_eq!(
            format!("{}", err),
            "Validation error: inconsistent dimensions"
        );
    }

    #[test]
    fn test_from_block_error() {
        let block_err = BlockError::RankZero {
            key: "test".to_string(),
        };
        let err: MolRsError = block_err.into();
        assert!(matches!(err, MolRsError::Block(_)));
    }

    #[test]
    fn test_from_string() {
        let err: MolRsError = "test error".into();
        assert!(matches!(err, MolRsError::Other(_)));
        assert_eq!(format!("{}", err), "test error");
    }
}
