//! [`ComputeError`] — the one error type of every [`Compute`](crate::compute::Compute),
//! [`Accumulate`](crate::compute::Accumulate) and configuration setter.
//!
//! Compute reads no frames, so no variant speaks of a missing block, column or
//! box: whatever an analysis needs is a typed field of its `Input`, and the
//! compiler has already checked that it is there. What is left to go wrong is
//! the data itself — a length that does not match, an array of the wrong
//! shape, a NaN, a value outside its physical range, nothing to analyse, a
//! name that is not one of a choice's canonical names, a trajectory still
//! wrapped into its box. Each variant names the offending input, so the
//! message points at the argument to fix. No analysis substitutes a default
//! for an out-of-contract input; it returns one of these instead.

use std::fmt;

/// Why an analysis refused its configuration or its data.
#[derive(Debug)]
pub enum ComputeError {
    /// Two lengths that must agree do not: `what` names the array whose length
    /// (or column count) is `got` where `expected` was required.
    DimensionMismatch {
        expected: usize,
        got: usize,
        what: &'static str,
    },

    /// A composite value has the wrong structure: a ragged or wrongly shaped
    /// array, a neighbour table without a column the kernel needs, a pairing
    /// mode the kernel cannot use, a box of an unsupported kind.
    BadShape { expected: String, got: String },

    /// Non-finite value (NaN / ±Inf) at the given flat index of the named field.
    NonFinite { where_: &'static str, index: usize },

    /// A scalar — a configuration value or a datum — outside its valid range.
    OutOfRange { field: &'static str, value: String },

    /// There is nothing to analyse: no samples, no items, no frames.
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
}

impl fmt::Display for ComputeError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
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
            Self::EmptyInput => write!(f, "empty input: nothing to analyse"),
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
        }
    }
}

impl std::error::Error for ComputeError {}
