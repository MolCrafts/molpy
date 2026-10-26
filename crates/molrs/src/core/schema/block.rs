//! Canonical block vocabulary: what one row of a block means.

/// What one row of a block represents.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum RowKind {
    /// One row per entity; the block others index into (`atoms`).
    ///
    /// A coarse-grained frame's beads are `atoms` rows, too.
    Node,
    /// One row per k-tuple of nodes, addressed by `arity` endpoint columns.
    Relation {
        /// Number of endpoint columns (2 = bond, 3 = angle, 4 = dihedral).
        arity: usize,
    },
    /// One row per cell of an N-D lattice; `Block::shape()` carries the extents.
    Grid,
}

impl std::fmt::Display for RowKind {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            RowKind::Node => write!(f, "node"),
            RowKind::Relation { arity } => write!(f, "relation({arity})"),
            RowKind::Grid => write!(f, "grid"),
        }
    }
}

/// Which block one endpoint column of a relation indexes.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum EndpointTarget {
    /// The rows of this block of the same frame (`"atoms"` for every
    /// canonical endpoint but `members.atom`).
    Block(&'static str),
    /// Whatever the block's `targets` attribute declares for the column, or
    /// nothing: without a declaration the column is an opaque handle, not a
    /// row reference (`members.atom`, which points into an all-atom block of
    /// another section).
    Declared,
}

/// The endpoint columns of a relation block, each with the block it indexes.
///
/// A block's declared `targets` ([`Block::targets`](crate::core::Block::targets))
/// override these defaults; see
/// [`relation_endpoints`](super::relation_endpoints).
#[derive(Debug, Clone, Copy)]
pub struct EndpointSpec {
    /// `(column, target)` in position order.
    pub columns: &'static [(&'static str, EndpointTarget)],
}

/// One canonical block of the Frame vocabulary.
#[derive(Debug, Clone, Copy)]
pub struct BlockSpec {
    /// Canonical block name (`"atoms"`, `"bonds"`, `"pairs"`).
    pub name: &'static str,
    /// What one row means.
    pub row_kind: RowKind,
    /// `None` for [`RowKind::Node`] and [`RowKind::Grid`].
    pub endpoints: Option<EndpointSpec>,
    /// Columns that MUST be present for the block to be well-formed.
    pub required: &'static [&'static str],
    /// Conventional but optional columns. Documentation + completeness gate.
    pub optional: &'static [&'static str],
    /// Whether columns outside `required ∪ optional` are admissible.
    ///
    /// `true` everywhere today. It exists so openness is a visible per-block
    /// decision rather than an unstated assumption, and so a block whose row
    /// identity is defined by an exact column set can later be closed.
    pub open: bool,
    /// One-line meaning. Never empty (asserted by the vocabulary gate).
    pub doc: &'static str,
}

impl BlockSpec {
    /// Endpoint column keys in position order, empty for non-relation
    /// blocks.
    pub fn endpoint_columns(&self) -> Vec<&'static str> {
        self.endpoints
            .map(|e| e.columns.iter().map(|&(column, _)| column).collect())
            .unwrap_or_default()
    }
}
