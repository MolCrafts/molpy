//! Error types for Block operations.

use std::fmt;

/// Errors that can occur when manipulating a [`Block`](super::Block).
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum BlockError {
    /// Inserted array has rank 0 (no axis-0 length can be defined)
    RankZero {
        /// The key for which the error occurred
        key: String,
    },
    /// Inserted array's axis-0 length does not match the Block's `nrows`
    RaggedAxis0 {
        /// The key for which the mismatch was detected
        key: String,
        /// The axis-0 length the block expects
        expected: usize,
        /// The axis-0 length provided by the inserted array
        got: usize,
    },
    /// A canonical column key was written at a dtype the vocabulary does not
    /// allow.
    ///
    /// The vocabulary binds a key wherever it appears, so this fires without
    /// the block knowing its own name. A column's dtype is fixed by its first
    /// write and molrs refuses to coerce, so accepting the wrong dtype here
    /// means the *next* correct write silently fails to land.
    SchemaDtype {
        /// The canonical key.
        key: String,
        /// Dtype the vocabulary declares.
        expected: crate::core::DType,
        /// Dtype the caller supplied.
        got: crate::core::DType,
    },
    /// A canonical column key was written at a shape the vocabulary does not
    /// allow.
    SchemaShape {
        /// The canonical key.
        key: String,
        /// Shape the vocabulary declares.
        expected: crate::core::schema::ColShape,
        /// Shape the caller supplied.
        got: Vec<usize>,
    },
    /// A validity mask does not cover exactly the rows of the column it was
    /// given for.
    ///
    /// A mask is read positionally against the column, so a shorter or longer
    /// one does not describe "some rows are null" — it describes nothing at
    /// all, and accepting it would leave the block reporting nullability for
    /// rows that are not there.
    ValidityLength {
        /// The column the mask was given for.
        key: String,
        /// Number of rows the column has.
        expected: usize,
        /// Number of entries the mask carries.
        got: usize,
    },
    /// A column the operation reads is not in the block.
    MissingColumn {
        /// The column that is absent.
        key: String,
    },
    /// Two parts of a [`Block::stack`](super::Block::stack) carry one column
    /// under different dtypes, so the rows cannot share one column.
    ///
    /// molrs never coerces a column's dtype, so a stack does not pick a
    /// winner: `i32` beside `i64` is refused.
    StackDtype {
        /// The column both parts carry.
        key: String,
        /// Index of the refused part in the `parts` sequence.
        part: usize,
        /// Dtype of the first part that carries the column.
        expected: crate::core::DType,
        /// Dtype of the refused part's column.
        got: crate::core::DType,
    },
    /// Two parts of a [`Block::stack`](super::Block::stack) carry one column
    /// with different per-row shapes (e.g. `(n, 3)` beside `(n, 2)`).
    StackShape {
        /// The column both parts carry.
        key: String,
        /// Index of the refused part in the `parts` sequence.
        part: usize,
        /// Per-row shape (axes after axis 0) of the first part's column.
        expected: Vec<usize>,
        /// Per-row shape of the refused part's column.
        got: Vec<usize>,
    },
    /// General validation error
    Validation {
        /// Error message
        message: String,
    },
}

impl fmt::Display for BlockError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            BlockError::SchemaDtype { key, expected, got } => write!(
                f,
                "column '{key}' is declared '{expected}' by the Frame schema, got '{got}'"
            ),
            BlockError::SchemaShape { key, expected, got } => write!(
                f,
                "column '{key}' is declared {expected} by the Frame schema, got {got:?}"
            ),
            BlockError::RankZero { key } => {
                write!(
                    f,
                    "array for key '{}' has rank 0; expected at least 1D",
                    key
                )
            }
            BlockError::RaggedAxis0 { key, expected, got } => write!(
                f,
                "array for key '{}' has axis-0 length {} but block expects {}",
                key, got, expected
            ),
            BlockError::ValidityLength { key, expected, got } => write!(
                f,
                "validity mask for key '{}' has {} entr(ies) but the column has {} row(s)",
                key, got, expected
            ),
            BlockError::MissingColumn { key } => write!(f, "block has no column '{key}'"),
            BlockError::StackDtype {
                key,
                part,
                expected,
                got,
            } => write!(
                f,
                "cannot stack column '{key}': part {part} carries it as '{got}', \
                 an earlier part as '{expected}'"
            ),
            BlockError::StackShape {
                key,
                part,
                expected,
                got,
            } => write!(
                f,
                "cannot stack column '{key}': part {part} has per-row shape {got:?}, \
                 an earlier part {expected:?}"
            ),
            BlockError::Validation { message } => write!(f, "{}", message),
        }
    }
}

impl std::error::Error for BlockError {}

impl BlockError {
    /// Creates a validation error with the given message.
    pub fn validation(message: impl Into<String>) -> Self {
        BlockError::Validation {
            message: message.into(),
        }
    }
}
