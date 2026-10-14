//! [`ComputeError`] — the single error enum shared by every `Compute` and
//! `Fit`; variants name the offending input (missing block/column/SimBox,
//! shape or range violations, non-finite values).

use std::fmt;

use molrs::core::MolRsError;

/// Error type for compute operations.
#[derive(Debug)]
pub enum ComputeError {
    /// Required block not found in Frame.
    MissingBlock { name: &'static str },

    /// Required column not found in a block.
    MissingColumn {
        block: &'static str,
        col: &'static str,
    },

    /// Frame has no SimBox but the compute requires one.
    MissingSimBox,

    /// Array dimensions do not match expectations.
    DimensionMismatch {
        expected: usize,
        got: usize,
        what: &'static str,
    },

    /// A composite value has the wrong shape (e.g. ragged matrix).
    BadShape { expected: String, got: String },

    /// Non-finite value (NaN / ±Inf) at the given flat index of the named field.
    NonFinite { where_: &'static str, index: usize },

    /// Scalar input out of its valid range.
    OutOfRange { field: &'static str, value: String },

    /// `frames` slice is empty but the compute needs at least one frame.
    EmptyInput,

    /// A name given for a named choice (kernel, window, mode, …) is not one of
    /// its canonical names.
    UnknownName {
        kind: &'static str,
        got: String,
        expected: &'static [&'static str],
    },

    /// A displacement analysis was given a trajectory that is wrapped into the
    /// box: between `frame - 1` and `frame`, `item` moved further than half
    /// the box.
    Wrapped { frame: usize, item: usize },

    /// An iterative solver stopped without meeting its tolerance.
    NotConverged {
        what: &'static str,
        iterations: usize,
    },

    /// Forwarded from `molrs::core`.
    MolRs(MolRsError),
}

impl fmt::Display for ComputeError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::MissingBlock { name } => write!(f, "missing block '{name}' in Frame"),
            Self::MissingColumn { block, col } => {
                write!(f, "missing column '{col}' in block '{block}'")
            }
            Self::MissingSimBox => write!(f, "Frame has no SimBox"),
            Self::DimensionMismatch {
                expected,
                got,
                what,
            } => write!(
                f,
                "{what} dimension mismatch: expected {expected}, got {got}"
            ),
            Self::BadShape { expected, got } => {
                write!(f, "bad shape: expected {expected}, got {got}")
            }
            Self::NonFinite { where_, index } => {
                write!(f, "non-finite value in {where_} at index {index}")
            }
            Self::OutOfRange { field, value } => {
                write!(f, "{field} out of range: {value}")
            }
            Self::EmptyInput => write!(f, "empty frames slice"),
            Self::UnknownName {
                kind,
                got,
                expected,
            } => write!(
                f,
                "unknown {kind} {got:?}; expected one of: {}",
                expected.join(", ")
            ),
            Self::Wrapped { frame, item } => write!(
                f,
                "item {item} moved more than half the box between frames {} and {frame}: \
                 the trajectory is wrapped; unwrap it before a displacement analysis",
                frame.saturating_sub(1)
            ),
            Self::NotConverged { what, iterations } => {
                write!(f, "{what} did not converge in {iterations} iterations")
            }
            Self::MolRs(e) => write!(f, "{e}"),
        }
    }
}

impl std::error::Error for ComputeError {
    fn source(&self) -> Option<&(dyn std::error::Error + 'static)> {
        match self {
            Self::MolRs(e) => Some(e),
            _ => None,
        }
    }
}

impl From<MolRsError> for ComputeError {
    fn from(err: MolRsError) -> Self {
        Self::MolRs(err)
    }
}
