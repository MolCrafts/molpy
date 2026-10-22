//! Typed schema violations with instance paths, and the whole-frame report.

use super::column::ColShape;
use crate::core::DType;

/// Where in a Frame a violation sits. Ordered coarse → fine so a report sorts
/// into a stable, diffable order.
#[derive(Debug, Clone, PartialEq, Eq, PartialOrd, Ord)]
pub enum InstancePath {
    /// The frame as a whole.
    Frame,
    /// A named block.
    Block {
        /// Block name.
        block: String,
    },
    /// A named column of a named block.
    Column {
        /// Block name.
        block: String,
        /// Column key.
        col: String,
    },
    /// A specific offending row of a column.
    Cell {
        /// Block name.
        block: String,
        /// Column key.
        col: String,
        /// 0-based row index.
        row: usize,
    },
}

impl InstancePath {
    /// RFC-6901-style pointer for tooling: `/bonds/atomi/7`.
    pub fn pointer(&self) -> String {
        match self {
            InstancePath::Frame => "/".to_string(),
            InstancePath::Block { block } => format!("/{block}"),
            InstancePath::Column { block, col } => format!("/{block}/{col}"),
            InstancePath::Cell { block, col, row } => format!("/{block}/{col}/{row}"),
        }
    }
}

impl std::fmt::Display for InstancePath {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            InstancePath::Frame => write!(f, "<frame>"),
            InstancePath::Block { block } => write!(f, "{block}"),
            InstancePath::Column { block, col } => write!(f, "{block}.{col}"),
            InstancePath::Cell { block, col, row } => write!(f, "{block}.{col}[{row}]"),
        }
    }
}

/// Why the schema was violated. Every variant carries expected-vs-found where
/// that is meaningful — a violation the caller cannot act on is not a report.
#[derive(Debug, Clone, PartialEq)]
pub enum ViolationKind {
    /// A column exists under a canonical key at the wrong dtype.
    WrongDtype {
        /// Dtype the vocabulary declares.
        expected: DType,
        /// Dtype actually stored.
        found: DType,
    },
    /// A column exists at the wrong shape.
    WrongShape {
        /// Shape the vocabulary declares.
        expected: ColShape,
        /// Shape actually stored.
        found: Vec<usize>,
    },
    /// A required block is absent.
    MissingBlock,
    /// A required column is absent from a present block.
    MissingColumn,
    /// A column key outside the vocabulary, in a block declared `open: false`.
    UnknownColumn,
    /// An endpoint value indexes past the end of its target node block.
    IndexOutOfRange {
        /// The offending index.
        value: u64,
        /// Block the endpoint indexes into.
        target: String,
        /// Row count of that block.
        target_nrows: usize,
    },
    /// A declared row-reference target names a block the frame lacks while
    /// the referencing block has rows.
    MissingTarget {
        /// The declared target.
        target: String,
    },
    /// Two node blocks that must align disagree on row count.
    RowCountMismatch {
        /// The other block.
        other: String,
        /// This block's row count.
        this_rows: usize,
        /// The other block's row count.
        other_rows: usize,
    },
    /// More offending cells than the per-column report cap; `extra` were elided.
    TruncatedCells {
        /// Number of offending rows not individually reported.
        extra: usize,
    },
}

impl std::fmt::Display for ViolationKind {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            ViolationKind::WrongDtype { expected, found } => {
                write!(f, "expected dtype '{expected}', found '{found}'")
            }
            ViolationKind::WrongShape { expected, found } => {
                write!(f, "expected shape {expected}, found {found:?}")
            }
            ViolationKind::MissingBlock => write!(f, "required block is absent"),
            ViolationKind::MissingColumn => write!(f, "required column is absent"),
            ViolationKind::UnknownColumn => {
                write!(f, "column is not in the vocabulary and the block is closed")
            }
            ViolationKind::IndexOutOfRange {
                value,
                target,
                target_nrows,
            } => write!(
                f,
                "index {value} out of range for '{target}' (nrows={target_nrows})"
            ),
            ViolationKind::MissingTarget { target } => {
                write!(f, "declared target '{target}' is not a block of this frame")
            }
            ViolationKind::RowCountMismatch {
                other,
                this_rows,
                other_rows,
            } => write!(
                f,
                "row count {this_rows} disagrees with '{other}' ({other_rows})"
            ),
            ViolationKind::TruncatedCells { extra } => {
                write!(f, "and {extra} more offending rows (report capped)")
            }
        }
    }
}

/// One schema violation: where, and why.
#[derive(Debug, Clone, PartialEq)]
pub struct Violation {
    /// Where in the frame.
    pub path: InstancePath,
    /// Why.
    pub kind: ViolationKind,
}

impl Violation {
    /// Build a violation at a column path.
    pub fn column(block: impl Into<String>, col: impl Into<String>, kind: ViolationKind) -> Self {
        Violation {
            path: InstancePath::Column {
                block: block.into(),
                col: col.into(),
            },
            kind,
        }
    }

    /// Build a violation at a block path.
    pub fn block(block: impl Into<String>, kind: ViolationKind) -> Self {
        Violation {
            path: InstancePath::Block {
                block: block.into(),
            },
            kind,
        }
    }
}

impl std::fmt::Display for Violation {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "{}: {}", self.path, self.kind)
    }
}

/// Per-column cap on individually reported offending rows.
///
/// A ten-million-row frame with one bad column must not produce a
/// ten-million-entry report; it must also not silently imply there were only a
/// handful, so the cap is followed by a [`ViolationKind::TruncatedCells`]
/// summary carrying the count that was elided.
pub const MAX_CELL_VIOLATIONS_PER_COLUMN: usize = 16;

/// Every violation found in one pass, sorted by [`InstancePath`].
///
/// Sorting makes two runs over the same frame produce byte-identical output, so
/// a report can be diffed in CI.
#[derive(Debug, Clone, Default, PartialEq)]
pub struct SchemaReport {
    violations: Vec<Violation>,
}

impl SchemaReport {
    /// An empty report.
    pub fn new() -> Self {
        Self::default()
    }

    /// Append one violation.
    pub fn push(&mut self, v: Violation) {
        self.violations.push(v);
    }

    /// Append every violation of another report.
    pub fn extend(&mut self, other: SchemaReport) {
        self.violations.extend(other.violations);
    }

    /// Sort into the stable, diffable order.
    pub fn sort(&mut self) {
        self.violations.sort_by(|a, b| a.path.cmp(&b.path));
    }

    /// Whether the frame conformed.
    pub fn is_empty(&self) -> bool {
        self.violations.is_empty()
    }

    /// Number of violations.
    pub fn len(&self) -> usize {
        self.violations.len()
    }

    /// Iterate the violations.
    pub fn iter(&self) -> impl Iterator<Item = &Violation> {
        self.violations.iter()
    }

    /// Violations under one block.
    pub fn at<'a>(&'a self, block: &'a str) -> impl Iterator<Item = &'a Violation> + 'a {
        self.violations.iter().filter(move |v| match &v.path {
            InstancePath::Block { block: b }
            | InstancePath::Column { block: b, .. }
            | InstancePath::Cell { block: b, .. } => b == block,
            InstancePath::Frame => false,
        })
    }

    /// `Ok(())` iff empty.
    pub fn into_result(self) -> Result<(), SchemaReport> {
        if self.is_empty() { Ok(()) } else { Err(self) }
    }
}

impl std::fmt::Display for SchemaReport {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        for (i, v) in self.violations.iter().enumerate() {
            if i > 0 {
                writeln!(f)?;
            }
            write!(f, "{v}")?;
        }
        Ok(())
    }
}

impl std::error::Error for SchemaReport {}

impl IntoIterator for SchemaReport {
    type Item = Violation;
    type IntoIter = std::vec::IntoIter<Violation>;
    fn into_iter(self) -> Self::IntoIter {
        self.violations.into_iter()
    }
}
