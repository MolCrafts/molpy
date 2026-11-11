//! Error types for the FFI layer.

use molrs::core::DType;
use std::fmt;

/// Errors that can occur in FFI operations.
#[derive(Debug, Clone)]
pub enum FfiError {
    /// Frame ID is invalid (frame was dropped or never existed)
    InvalidFrameId,

    /// Block handle is invalid (block was removed/replaced or frame gone)
    InvalidBlockHandle,

    /// Key not found in frame or block
    KeyNotFound { key: String },

    /// Array is not contiguous (zero-copy view not possible)
    NonContiguous { key: String },

    /// Column exists but has the wrong dtype for the requested access.
    /// Distinct from [`KeyNotFound`](Self::KeyNotFound): the caller asked the right key but
    /// mismatched its type expectation — usually a schema-level bug.
    DTypeMismatch {
        key: String,
        expected: DType,
        actual: DType,
    },

    /// The frame's store is held by an operation still running on it — an
    /// MD run reaching back into the frame it steps, from a callback, or
    /// native code on another thread. The access is refused instead of
    /// aliasing the store or waiting on it.
    FrameBusy,
}

impl fmt::Display for FfiError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            FfiError::InvalidFrameId => write!(f, "Invalid frame ID"),
            FfiError::InvalidBlockHandle => write!(f, "Invalid block handle"),
            FfiError::KeyNotFound { key } => write!(f, "Key '{}' not found", key),
            FfiError::NonContiguous { key } => {
                write!(f, "Column '{}' is not contiguous in memory", key)
            }
            FfiError::FrameBusy => write!(
                f,
                "the frame is in use by an operation still running on it (an MD run \
                 calling back into the frame it steps, or native code on another \
                 thread); access it after that call returns"
            ),
            FfiError::DTypeMismatch {
                key,
                expected,
                actual,
            } => write!(
                f,
                "Column '{}' has dtype {} but expected {}",
                key, actual, expected
            ),
        }
    }
}

impl std::error::Error for FfiError {}
