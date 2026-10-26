use molrs::core::MolRsError;
use std::sync::Arc;
use zarrs::group::Group;
use zarrs::storage::{ReadableStorageTraits, ReadableWritableListableStorage};

use super::layout::{DENSE_UPDATES_ATTRIBUTE, STRUCTURAL_SHAPE_ATTRIBUTE, UNIFORM_ROWS_ATTRIBUTE};

// ---------------------------------------------------------------------------
// Block hints
// ---------------------------------------------------------------------------

/// Whether every update of a block so far carried the same row count.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(super) enum Uniform {
    /// No update yet.
    Unknown,
    /// Every update so far carried this many rows.
    Rows(u64),
    /// Two updates disagreed; the hint is gone for good.
    Broken,
}

/// The two block-section hints the writer maintains as group attributes.
///
/// `uniform_rows` says every update carried the same row count; `dense_updates`
/// says `step_index[j] == j` for every update. A reader holding both resolves
/// any frame arithmetically without decoding the index arrays. They are
/// **monotone**: once a hint breaks it is removed and never returns, so a
/// reader that sees one may trust it for every committed frame.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(super) struct BlockHints {
    pub(super) uniform: Uniform,
    dense: bool,
    /// What the group attributes currently say, so a rewrite happens only on
    /// a change.
    pub(super) stored: (Option<u64>, bool),
}

impl BlockHints {
    pub(super) fn fresh() -> Self {
        Self {
            uniform: Uniform::Unknown,
            dense: true,
            stored: (None, false),
        }
    }

    /// Rebuild from a section's index arrays — what a reopen does.
    pub(super) fn from_index(step_index: &[u64], offset: &[u64]) -> Self {
        let mut hints = Self::fresh();
        for (update, &ordinal) in step_index.iter().enumerate() {
            let rows = offset
                .get(update + 1)
                .zip(offset.get(update))
                .map_or(0, |(end, start)| end.saturating_sub(*start));
            hints.observe(update as u64, ordinal, rows);
        }
        hints
    }

    /// Fold one more update in.
    pub(super) fn observe(&mut self, update: u64, ordinal: u64, rows: u64) {
        self.dense &= ordinal == update;
        self.uniform = match self.uniform {
            Uniform::Unknown => Uniform::Rows(rows),
            Uniform::Rows(n) if n == rows => Uniform::Rows(n),
            _ => Uniform::Broken,
        };
    }

    /// Whether every update so far is regular: a fixed, non-zero row count
    /// at ordinals `0, 1, 2, …` — the state under which the CSR index is
    /// not written.
    pub(super) fn is_regular(&self) -> bool {
        matches!(self.uniform, Uniform::Rows(n) if n > 0) && self.dense
    }

    /// The attributes these hints spell: `(uniform_rows, dense_updates)`.
    pub(super) fn attributes(&self) -> (Option<u64>, bool) {
        match self.uniform {
            Uniform::Rows(n) if n > 0 => (Some(n), self.dense),
            Uniform::Rows(_) | Uniform::Unknown | Uniform::Broken => (None, false),
        }
    }
}

/// The group attributes of a block section: its mirrored structural shape
/// plus whichever hints currently hold.
pub(super) fn block_attributes(
    structural_shape: Option<&[usize]>,
    hints: (Option<u64>, bool),
) -> serde_json::Map<String, serde_json::Value> {
    let mut attributes = serde_json::Map::new();
    if let Some(shape) = structural_shape {
        attributes.insert(
            STRUCTURAL_SHAPE_ATTRIBUTE.to_string(),
            serde_json::Value::Array(
                shape
                    .iter()
                    .map(|axis| serde_json::Value::from(*axis as u64))
                    .collect(),
            ),
        );
    }
    if let Some(rows) = hints.0 {
        attributes.insert(UNIFORM_ROWS_ATTRIBUTE.to_string(), rows.into());
    }
    if hints.1 {
        attributes.insert(DENSE_UPDATES_ATTRIBUTE.to_string(), true.into());
    }
    attributes
}

/// Read a block section's hints back off its group attributes.
pub(super) fn stored_hints<S>(store: &Arc<S>, path: &str) -> Result<(Option<u64>, bool), MolRsError>
where
    S: ?Sized + ReadableStorageTraits + 'static,
{
    let group = Group::open(store.clone(), path)?;
    let attributes = group.attributes();
    Ok((
        attributes
            .get(UNIFORM_ROWS_ATTRIBUTE)
            .and_then(serde_json::Value::as_u64),
        attributes
            .get(DENSE_UPDATES_ATTRIBUTE)
            .and_then(serde_json::Value::as_bool)
            .unwrap_or(false),
    ))
}

/// The elision markers of a block section with no index arrays: `Some(N)`
/// for a well-formed `uniform_rows: N` / `dense_updates: true` pair, `None`
/// when it carries neither (a declared block never updated, absent at every
/// ordinal).
///
/// # Errors
///
/// A [`MolRsError::Zarr`] naming the section for one marker without the
/// other, a `uniform_rows` that is not a positive integer, a
/// `dense_updates` that is not `true`, or markers on a block that declares
/// no columns (it has no length to count its updates by).
pub(super) fn elision_markers<S>(
    store: &Arc<S>,
    path: &str,
    has_columns: bool,
) -> Result<Option<u64>, MolRsError>
where
    S: ?Sized + ReadableStorageTraits + 'static,
{
    let group = Group::open(store.clone(), path)?;
    let attributes = group.attributes();
    let rows = match (
        attributes.get(UNIFORM_ROWS_ATTRIBUTE),
        attributes.get(DENSE_UPDATES_ATTRIBUTE),
    ) {
        (None, None) => return Ok(None),
        (Some(rows), Some(dense)) => {
            if dense != &serde_json::Value::Bool(true) {
                return Err(MolRsError::zarr(format!(
                    "{path}: {DENSE_UPDATES_ATTRIBUTE} must be true, found {dense}"
                )));
            }
            match rows.as_u64() {
                Some(rows) if rows > 0 => rows,
                _ => {
                    return Err(MolRsError::zarr(format!(
                        "{path}: {UNIFORM_ROWS_ATTRIBUTE} must be a positive integer, found \
                         {rows}"
                    )));
                }
            }
        }
        _ => {
            return Err(MolRsError::zarr(format!(
                "{path}: {UNIFORM_ROWS_ATTRIBUTE} and {DENSE_UPDATES_ATTRIBUTE} are a pair; \
                 this section carries one without the other"
            )));
        }
    };
    if !has_columns {
        return Err(MolRsError::zarr(format!(
            "{path}: elision markers on a block with no columns, which has no length to \
             count its updates by"
        )));
    }
    Ok(Some(rows))
}

/// Rewrite a block section's group attributes.
pub(super) fn store_block_attributes(
    store: &ReadableWritableListableStorage,
    path: &str,
    structural_shape: Option<&[usize]>,
    hints: (Option<u64>, bool),
) -> Result<(), MolRsError> {
    let mut group = Group::open(store.clone(), path)?;
    *group.attributes_mut() = block_attributes(structural_shape, hints);
    group.store_metadata()?;
    Ok(())
}
