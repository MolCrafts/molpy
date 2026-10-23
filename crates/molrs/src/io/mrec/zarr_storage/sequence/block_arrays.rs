use crate::io::mrec::zarr_storage::frame_io::{GZIP_LEVEL, VALIDITY_GROUP, join_path};
use molrs::core::MolRsError;
use molrs::core::{Block, DType};
use std::collections::BTreeMap;
use zarrs::array::CodecOptions;
use zarrs::group::GroupBuilder;
use zarrs::storage::ReadableWritableListableStorage;

use super::block_hints::BlockHints;
use super::codec::{Compression, Extents, Pipeline, partial_encoding_options};
use super::growth::{GrowthArray, land_values};
use super::layout::{DENSE_CHUNKS_PER_SHARD, DENSE_ROWS_PER_CHUNK, OFFSET_ARRAY, STEP_INDEX_ARRAY};
use super::schema::BlockSchema;
use super::store_access::array_exists;

// ---------------------------------------------------------------------------
// Section arrays
// ---------------------------------------------------------------------------

/// The two test knobs that override the derived extents of every array.
#[derive(Debug, Clone, Copy, Default)]
pub(super) struct Knobs {
    pub(super) rows_per_chunk: Option<u64>,
    pub(super) chunks_per_shard: Option<u64>,
}

impl Knobs {
    /// Extents of a dense array under these knobs.
    pub(super) fn dense(self) -> Extents {
        Extents {
            rows_per_chunk: self.rows_per_chunk.unwrap_or(DENSE_ROWS_PER_CHUNK),
            chunks_per_shard: self.chunks_per_shard.unwrap_or(DENSE_CHUNKS_PER_SHARD),
        }
    }
}

/// The pipeline every dense array carries.
pub(super) const DENSE_PIPELINE: Pipeline = Pipeline::plain(Compression::Gzip(GZIP_LEVEL));

/// The arrays of one block section.
///
/// The CSR index — `offset` and `step_index` — is **not** written while the
/// block is regular (every update carries the same non-zero row count at
/// ordinals `0, 1, 2, …`): the block group's `uniform_rows` /
/// `dense_updates` hints let a reader resolve any frame arithmetically, and
/// the two arrays would only restate what the column shapes already say. The
/// first update that breaks either rule materializes both arrays, backfilling
/// the regular history, and withdraws the hints for good.
pub(super) struct BlockArrays {
    pub(super) columns: BTreeMap<String, GrowthArray>,
    /// The validity mask of each column pinned nullable: a `bool` array under
    /// the section's `_validity` subgroup, grown in lockstep with the columns
    /// so a mask row sits at the row index of the value it qualifies.
    pub(super) masks: BTreeMap<String, GrowthArray>,
    /// The CSR index, once the block stopped being regular.
    pub(super) index: Option<IndexArrays>,
    /// Rows landed across every update.
    pub(super) total_rows: u64,
    /// Updates landed.
    pub(super) updates: u64,
    /// Rows one inner chunk of this block's columns holds.
    pub(super) rows_per_chunk: u64,
    pub(super) hints: BlockHints,
}

impl BlockArrays {
    /// Land one flag per landed row into every mask array of this section.
    ///
    /// A block that carries no mask for a nullable column lands all-true: the
    /// mask array is dense over the section's rows, which is what keeps a
    /// frame's flags at exactly the row range its values occupy, and what
    /// lets a column be holed in one frame and whole in the next.
    pub(super) fn land_masks(
        &mut self,
        blocks: &[&Block],
        added: u64,
        options: &CodecOptions,
    ) -> Result<(), MolRsError> {
        for (column, array) in &mut self.masks {
            let mut flags: Vec<bool> = Vec::with_capacity(added as usize);
            for block in blocks {
                match block.validity(column) {
                    Some(mask) => flags.extend_from_slice(mask),
                    None => flags.extend(std::iter::repeat_n(true, block.n_rows().unwrap_or(0))),
                }
            }
            land_values!(array, added, flags, options);
        }
        Ok(())
    }
}

/// The names of the columns `declared` pins nullable.
fn nullable_columns(declared: &BlockSchema) -> Vec<&String> {
    declared
        .columns
        .iter()
        .filter(|(_, column)| column.nullable)
        .map(|(name, _)| name)
        .collect()
}

/// Build a section's reserved `_validity` subgroup and yield its path.
fn create_validity_group(
    store: &ReadableWritableListableStorage,
    path: &str,
) -> Result<String, MolRsError> {
    let prefix = join_path(path, VALIDITY_GROUP);
    GroupBuilder::new()
        .build(store.clone(), &prefix)?
        .store_metadata()?;
    Ok(prefix)
}

/// Create one column's mask array: `bool`, one flag per row, no trailing
/// axes, sharing its block's `rows_per_chunk` so a mask chunk covers the row
/// range its values' chunk does.
fn create_mask_array(
    store: &ReadableWritableListableStorage,
    prefix: &str,
    column: &str,
    rows_per_chunk: u64,
    knobs: Knobs,
) -> Result<GrowthArray, MolRsError> {
    GrowthArray::create(
        store,
        &join_path(prefix, column),
        DType::Bool,
        &[],
        Extents::for_column(rows_per_chunk, 1, knobs.chunks_per_shard),
        serde_json::Map::new(),
        DENSE_PIPELINE,
    )
}

/// Create the mask array of every column `declared` pins nullable.
///
/// A section with no nullable column gets no subgroup at all: its store is
/// what a molrs that never masked a column wrote, and what a reader that
/// knows nothing of masks still reads.
pub(super) fn create_mask_arrays(
    store: &ReadableWritableListableStorage,
    path: &str,
    declared: &BlockSchema,
    rows_per_chunk: u64,
    knobs: Knobs,
) -> Result<BTreeMap<String, GrowthArray>, MolRsError> {
    let mut masks = BTreeMap::new();
    let nullable = nullable_columns(declared);
    if nullable.is_empty() {
        return Ok(masks);
    }
    let prefix = create_validity_group(store, path)?;
    for column in nullable {
        masks.insert(
            column.clone(),
            create_mask_array(store, &prefix, column, rows_per_chunk, knobs)?,
        );
    }
    Ok(masks)
}

/// Reopen the mask array of every nullable column of `declared`, creating and
/// backfilling any the store does not carry.
///
/// A missing array means `rows` rows were landed while the column held no
/// nulls, so the backfill is all-true — that is what those rows are. It keeps
/// the invariant the reader leans on: a nullable column's mask covers exactly
/// the rows its values do.
pub(super) fn open_mask_arrays(
    store: &ReadableWritableListableStorage,
    path: &str,
    declared: &BlockSchema,
    rows: u64,
    rows_per_chunk: u64,
    knobs: Knobs,
) -> Result<BTreeMap<String, GrowthArray>, MolRsError> {
    let mut masks = BTreeMap::new();
    let mut missing: Vec<&String> = Vec::new();
    let prefix = join_path(path, VALIDITY_GROUP);
    for column in nullable_columns(declared) {
        let mask_path = join_path(&prefix, column);
        if array_exists(store, &mask_path)? {
            masks.insert(column.clone(), GrowthArray::open(store, &mask_path)?);
        } else {
            missing.push(column);
        }
    }
    if !missing.is_empty() {
        let prefix = create_validity_group(store, path)?;
        let options = partial_encoding_options();
        for column in missing {
            let mut array = create_mask_array(store, &prefix, column, rows_per_chunk, knobs)?;
            if rows > 0 {
                let flags = vec![true; rows as usize];
                land_values!(array, rows, flags, &options);
            }
            masks.insert(column.clone(), array);
        }
    }
    Ok(masks)
}

/// A block section's CSR index arrays.
pub(super) struct IndexArrays {
    pub(super) offset: GrowthArray,
    pub(super) step_index: GrowthArray,
}

impl IndexArrays {
    /// Create both arrays and backfill the `history` regular updates of
    /// `rows` rows each — the updates the hints describe.
    pub(super) fn materialize(
        store: &ReadableWritableListableStorage,
        path: &str,
        knobs: Knobs,
        history: u64,
        rows: u64,
        options: &CodecOptions,
    ) -> Result<Self, MolRsError> {
        let mut offset = GrowthArray::create(
            store,
            &join_path(path, OFFSET_ARRAY),
            DType::Uint,
            &[],
            knobs.dense(),
            serde_json::Map::new(),
            DENSE_PIPELINE,
        )?;
        let mut step_index = GrowthArray::create(
            store,
            &join_path(path, STEP_INDEX_ARRAY),
            DType::Uint,
            &[],
            knobs.dense(),
            serde_json::Map::new(),
            DENSE_PIPELINE,
        )?;
        // The CSR row pointer opens at zero; every regular update added
        // `rows` rows at the ordinal equal to its own index.
        let offsets: Vec<u64> = (0..=history).map(|update| update * rows).collect();
        let count = offsets.len() as u64;
        land_values!(offset, count, offsets, options);
        if history > 0 {
            let ordinals: Vec<u64> = (0..history).collect();
            land_values!(step_index, history, ordinals, options);
        }
        Ok(Self { offset, step_index })
    }
}
