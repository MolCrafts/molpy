mod append;

use crate::io::mrec::zarr_storage::frame_io::{
    insert_column_into_block, join_path, node_prefix, read_column_array,
};
use crate::io::mrec::zarr_storage::record_io::zerr;
use molrs::core::Block;
use molrs::core::MetaMap;
use molrs::core::MolRsError;
use molrs::core::SimBox;
use molrs::op::F;
use std::collections::BTreeMap;
use std::sync::Arc;
use zarrs::group::GroupBuilder;
use zarrs::storage::{ReadableListableStorage, ReadableWritableListableStorage, StorageHandle};

use super::block_arrays::Knobs;
use super::block_hints::Uniform;
use super::box_arrays::{CellStorage, cell_defined_of};
use super::codec::{Compression, rows_subset};
use super::growth::{apply_structural_shape, empty_block};
use super::layout::{
    FLUSH_TARGET_BYTES, MAX_FLUSH_EVERY, OFFSET_ARRAY, SCHEMA_ATTRIBUTE, STEP_ARRAY,
    TRAJECTORY_GROUP,
};
use super::reader::{BoxReader, check_aligned_not_in_system};
use super::schema::SequenceSchema;
use super::sequence_arrays::SequenceArrays;
use super::series::trajectory_attributes;
use super::store_access::{
    array_exists, ensure_root_and_meta, group_exists, read_whole, schema_of,
};

// ---------------------------------------------------------------------------
// Writer
// ---------------------------------------------------------------------------

/// One appended frame, buffered until a landing writes it.
struct PendingFrame {
    step: i64,
    time: Option<F>,
    /// Sections this frame changed. A block omitted by the frame is not here:
    /// it carries forward. A present block with zero rows is here, as the
    /// zero-row update that means present-and-empty.
    blocks: BTreeMap<String, Block>,
    /// The cell, when it changed.
    cell: Option<SimBox>,
    /// Every declared meta key, resolved to a value or its declared fill.
    meta: MetaMap,
}

/// The streaming producer of a frame sequence: one frame per
/// [`append`](Self::append); complete inner chunks land on their own,
/// [`flush`](Self::flush) commits the rest, `close(self)` ends the run.
///
/// One of the three access forms of a single object — the eager in-memory
/// carrier is `Trajectory`, the lazy store cursor is [`MrecReader`], and
/// this is the writer that produces what that cursor reads.
///
/// **No `Drop`.** Closing is [`close`](Self::close), which consumes the writer
/// and can therefore return the IO error a `Drop` would have had to swallow —
/// and a scientific writer that swallows an IO error is silent data loss.
///
/// **What lands when.** Every append is checked and buffered. When the buffer
/// reaches the landing cadence — derived from the frame size so that block
/// columns land whole inner chunks and roughly 4 MiB at a
/// time, or set by [`with_flush_every`](Self::with_flush_every) — the buffered
/// frames are committed. [`flush`](Self::flush) commits at any time and is
/// **durable** by default (the touched files are synced before the commit
/// marker moves); the automatic landings are not synced.
///
/// **Error vocabulary.** Every door on this type yields [`MolRsError`], and the
/// storage-backed ones arrive as the [`MolRsError::Zarr`] variant carrying a
/// message that names the block, column, metadata key or array path that
/// disagreed.
///
/// # Chunks, shards, and the commit
///
/// Every array of the sequence is a Zarr V3 `sharding_indexed` array whose
/// shard **index sits at the start** of the shard file. An **inner chunk** is
/// the unit the codec compresses; a **shard** is one file holding a fixed
/// number of consecutive inner chunks plus that index. With the index at the
/// start, appending a complete inner chunk is a write at the tail of the file
/// plus an in-place rewrite of the fixed-size index — nothing is re-encoded and
/// no dead bytes are left behind.
///
/// Block columns are **frame-aligned**: their inner chunk holds a whole number
/// of frames of the representative row count (see [`SequenceSchema`]), so a
/// frame decodes from exactly its own chunk and a commit at a chunk boundary
/// rewrites nothing. The writer lands complete chunks on its own cadence
/// ([`MrecWriter::with_flush_every`] overrides it); an explicit
/// [`flush`](MrecWriter::flush) may land a partially filled trailing
/// chunk, whose overwritten copy then stays in the shard as dead bytes — bounded
/// by one chunk per column per flush, and never cleaned up, because the
/// whole-shard rewrite that would clean it is the one write that can destroy
/// committed data on a crash.
///
/// **Commit protocol.** For every array: data is written, then the array's
/// `zarr.json` (its new shape) is replaced atomically. Last comes the
/// trajectory group's metadata — the `nstep` marker and the progression
/// attributes — in one atomic replace. A reader takes `nstep` from that
/// attribute; anything a crash left longer is invisible, and a reopened
/// writer rolls it back before appending. An explicit `flush` / `close` is
/// durable (the touched files are synced before `step` moves) unless
/// [`with_durable(false)`](MrecWriter::with_durable) says otherwise;
/// the automatic chunk-boundary landings are not synced.
///
/// [`MrecReader`]: super::MrecReader
pub struct MrecWriter {
    store: ReadableWritableListableStorage,
    /// The concrete positional-write store behind `store`, when this writer
    /// was opened by path — the only store that can be asked to sync.
    #[cfg(feature = "filesystem")]
    positional:
        Option<Arc<crate::io::mrec::zarr_storage::positional_write::PositionalWriteStorage>>,
    schema: SequenceSchema,
    knobs: Knobs,
    compression: Compression,
    /// Landing cadence in frames; `None` derives it at the first append.
    flush_every: Option<u64>,
    /// Whether an explicit `flush` / `close` syncs the touched files.
    durable: bool,
    /// `None` until the first append freezes the extents and creates them.
    arrays: Option<SequenceArrays>,
    /// The landing cadence in force, fixed at the first append.
    auto_flush_every: u64,
    /// Highest step number landed or buffered.
    last_step: Option<i64>,
    /// Whether this run carries times. Fixed by the first append.
    uses_time: Option<bool>,
    /// Frames committed.
    committed: u64,
    pending: Vec<PendingFrame>,
    /// Last content landed or buffered per section, for the change detection
    /// that keeps an unchanging section at one `step_index` entry.
    landed_blocks: BTreeMap<String, Block>,
    landed_cell: Option<SimBox>,
}

/// What a writer knows about the sequence it is attached to: nothing for a
/// fresh mint, the recovered state for a reopen.
struct Attached {
    arrays: Option<SequenceArrays>,
    last_step: Option<i64>,
    uses_time: Option<bool>,
    committed: u64,
    landed_blocks: BTreeMap<String, Block>,
    landed_cell: Option<SimBox>,
}

impl Attached {
    /// A sequence with no arrays yet: created, never appended.
    fn fresh() -> Self {
        Self {
            arrays: None,
            last_step: None,
            uses_time: None,
            committed: 0,
            landed_blocks: BTreeMap::new(),
            landed_cell: None,
        }
    }
}

impl MrecWriter {
    fn assemble(
        store: ReadableWritableListableStorage,
        #[cfg(feature = "filesystem")] positional: Option<
            Arc<crate::io::mrec::zarr_storage::positional_write::PositionalWriteStorage>,
        >,
        schema: SequenceSchema,
        attached: Attached,
    ) -> Self {
        let mut writer = Self {
            store,
            #[cfg(feature = "filesystem")]
            positional,
            schema,
            knobs: Knobs::default(),
            compression: Compression::None,
            flush_every: None,
            durable: true,
            arrays: attached.arrays,
            auto_flush_every: 1,
            last_step: attached.last_step,
            uses_time: attached.uses_time,
            committed: attached.committed,
            pending: Vec::new(),
            landed_blocks: attached.landed_blocks,
            landed_cell: attached.landed_cell,
        };
        if writer.arrays.is_some() {
            writer.auto_flush_every = writer.derive_flush_every();
        }
        writer
    }

    /// Mint a new sequence at `trajectory/` and pin `schema` to it.
    ///
    /// Writes the record root and its `meta/` group when the store has none,
    /// then the `trajectory/` group with its schema attribute — and nothing
    /// else: the arrays are created by the first [`append`](Self::append). A
    /// store that already holds a sequence is an `Err` naming it, never a
    /// silent overwrite.
    ///
    /// **`trajectory/` is cleared first.** Zarr stores are written key by key,
    /// so a mint that only wrote its own keys would inherit every leftover
    /// child of whatever stood at that node. Reopen with [`open`](Self::open)
    /// instead when the store may already hold data.
    ///
    /// # Errors
    ///
    /// A [`MolRsError::Zarr`] naming `trajectory/step` when that array already
    /// exists, so nothing is erased and nothing is written. Otherwise any
    /// storage error raised while writing the groups.
    pub fn create_in_storage(
        store: ReadableWritableListableStorage,
        schema: SequenceSchema,
    ) -> Result<Self, MolRsError> {
        Self::create_with(
            store,
            #[cfg(feature = "filesystem")]
            None,
            schema,
        )
    }

    fn create_with(
        store: ReadableWritableListableStorage,
        #[cfg(feature = "filesystem")] positional: Option<
            Arc<crate::io::mrec::zarr_storage::positional_write::PositionalWriteStorage>,
        >,
        schema: SequenceSchema,
    ) -> Result<Self, MolRsError> {
        if group_exists(&store, TRAJECTORY_GROUP)? {
            let attributes = trajectory_attributes(&store)?;
            if attributes.nstep.is_some()
                || array_exists(&store, &join_path(TRAJECTORY_GROUP, STEP_ARRAY))?
            {
                return Err(MolRsError::zarr(format!(
                    "a frame sequence already exists at {TRAJECTORY_GROUP:?}; refusing to \
                     overwrite it"
                )));
            }
        }
        schema.check_alignments()?;
        check_aligned_not_in_system(&store, &schema)?;
        ensure_root_and_meta(&store, None)?;
        // This writer's target node is `trajectory/`, so whatever it held
        // before is not part of the sequence being minted.
        store.erase_prefix(&node_prefix(TRAJECTORY_GROUP)?)?;
        let mut attributes = serde_json::Map::new();
        attributes.insert(
            SCHEMA_ATTRIBUTE.to_string(),
            serde_json::to_value(&schema).map_err(zerr)?,
        );
        GroupBuilder::new()
            .attributes(attributes)
            .build(store.clone(), TRAJECTORY_GROUP)?
            .store_metadata()?;
        Ok(Self::assemble(
            store,
            #[cfg(feature = "filesystem")]
            positional,
            schema,
            Attached::fresh(),
        ))
    }

    /// Mint a new sequence at `path` — the path-taking door over the fast
    /// positional-write store, which is also the store that makes
    /// [`flush`](Self::flush) durable.
    ///
    /// # Errors
    ///
    /// The store-root errors of the positional store, then every error
    /// [`create`](Self::create) can raise.
    #[cfg(feature = "filesystem")]
    pub fn create(
        path: impl AsRef<std::path::Path>,
        schema: SequenceSchema,
    ) -> Result<Self, MolRsError> {
        let positional = Arc::new(
            crate::io::mrec::zarr_storage::positional_write::PositionalWriteStorage::new(path)?,
        );
        let store: ReadableWritableListableStorage = positional.clone();
        Self::create_with(store, Some(positional), schema)
    }

    /// Reattach to the sequence at `path` and continue appending — the
    /// path-taking counterpart of [`open`](Self::open).
    ///
    /// # Errors
    ///
    /// The store-root errors of the positional store, then every error
    /// [`open`](Self::open) can raise.
    #[cfg(feature = "filesystem")]
    pub fn open(path: impl AsRef<std::path::Path>) -> Result<Self, MolRsError> {
        let positional = Arc::new(
            crate::io::mrec::zarr_storage::positional_write::PositionalWriteStorage::new(path)?,
        );
        let store: ReadableWritableListableStorage = positional.clone();
        Self::open_with(store, Some(positional))
    }

    /// Reattach to an existing sequence and continue appending to it.
    ///
    /// The schema comes off the group attributes and every array is checked
    /// against it; the extents come off the arrays themselves. Whatever a
    /// crash left longer than the commit marker is rolled back first, so the
    /// next append continues from the last committed frame.
    ///
    /// # Errors
    ///
    /// A [`MolRsError::Zarr`] when `trajectory/` carries no
    /// `sequence_schema` attribute (a foreign store can be *read* without the
    /// pin but not appended to); or when a reopened column array disagrees
    /// with the pinned schema on dtype or trailing shape.
    pub fn from_storage(store: ReadableWritableListableStorage) -> Result<Self, MolRsError> {
        Self::open_with(
            store,
            #[cfg(feature = "filesystem")]
            None,
        )
    }

    fn open_with(
        store: ReadableWritableListableStorage,
        #[cfg(feature = "filesystem")] positional: Option<
            Arc<crate::io::mrec::zarr_storage::positional_write::PositionalWriteStorage>,
        >,
    ) -> Result<Self, MolRsError> {
        let schema = schema_of(&store)?;
        schema.check_alignments()?;
        check_aligned_not_in_system(&store, &schema)?;
        if trajectory_attributes(&store)?.nstep.is_none()
            && !array_exists(&store, &join_path(TRAJECTORY_GROUP, STEP_ARRAY))?
        {
            // Created but never appended: nothing is frozen yet.
            return Ok(Self::assemble(
                store,
                #[cfg(feature = "filesystem")]
                positional,
                schema,
                Attached::fresh(),
            ));
        }

        let arrays = SequenceArrays::open(&store, &schema)?;
        let committed = arrays.nstep;
        let last_step = arrays
            .step
            .last(&store, &join_path(TRAJECTORY_GROUP, STEP_ARRAY))?;
        let uses_time = Some(arrays.time.is_some());

        // Recover the change-detection state, so a section that does not move
        // across the reopen still costs one `step_index` entry rather than two.
        let mut landed_blocks = BTreeMap::new();
        for (name, block) in &arrays.blocks {
            if block.updates == 0 {
                continue;
            }
            let path = join_path(TRAJECTORY_GROUP, name);
            let end = block.total_rows;
            let start = match &block.index {
                Some(_) => {
                    let offsets = read_whole::<u64, _>(&store, &join_path(&path, OFFSET_ARRAY))?;
                    offsets
                        .get(block.updates as usize - 1)
                        .copied()
                        .unwrap_or(end)
                }
                None => match block.hints.uniform {
                    Uniform::Rows(rows) => end.saturating_sub(rows),
                    _ => end,
                },
            };
            let declared = &schema.blocks[name];
            let landed = if end > start {
                let mut landed = Block::new();
                for (column, column_schema) in &declared.columns {
                    let subset = rows_subset(start, end - start, &column_schema.trailing)?;
                    let array = &block.columns[column].array;
                    insert_column_into_block(
                        &mut landed,
                        column,
                        read_column_array(array, &subset)?,
                    )?;
                }
                apply_structural_shape(&mut landed, declared, &path)?;
                for (column, target) in &declared.targets {
                    if landed.contains_key(column) {
                        landed.set_target(column, target)?;
                    }
                }
                landed
            } else {
                empty_block(declared)?
            };
            landed_blocks.insert(name.clone(), landed);
        }
        let landed_cell = match arrays.cell.as_ref() {
            Some(CellStorage::Attributes(cell)) => Some((**cell).clone()),
            Some(CellStorage::Arrays(section)) if section.vectors.rows > 0 => {
                let reader: ReadableListableStorage = Arc::new(StorageHandle::new(store.clone()));
                let mut boxes = BoxReader::open(&reader)?;
                Some(boxes.cell_at(section.vectors.rows - 1, cell_defined_of(&store)?)?)
            }
            _ => None,
        };

        Ok(Self::assemble(
            store,
            #[cfg(feature = "filesystem")]
            positional,
            schema,
            Attached {
                arrays: Some(arrays),
                last_step,
                uses_time,
                committed,
                landed_blocks,
                landed_cell,
            },
        ))
    }

    /// Rows one inner chunk of every growth array holds — a test knob.
    ///
    /// Legal only before the first append, which freezes the extents. Left
    /// unset, block columns are frame-aligned and dense arrays take
    /// [`DENSE_ROWS_PER_CHUNK`].
    #[cfg(all(test, feature = "filesystem"))]
    pub(crate) fn with_rows_per_chunk(mut self, rows: u64) -> Result<Self, MolRsError> {
        self.refuse_after_first_append("with_rows_per_chunk")?;
        if rows == 0 {
            return Err(MolRsError::zarr("rows_per_chunk must be at least 1"));
        }
        self.knobs.rows_per_chunk = Some(rows);
        Ok(self)
    }

    /// Inner chunks one shard file holds — a test knob.
    ///
    /// Legal only before the first append.
    #[cfg(all(test, feature = "filesystem"))]
    pub(crate) fn with_chunks_per_shard(mut self, chunks: u64) -> Result<Self, MolRsError> {
        self.refuse_after_first_append("with_chunks_per_shard")?;
        if chunks == 0 {
            return Err(MolRsError::zarr("chunks_per_shard must be at least 1"));
        }
        self.knobs.chunks_per_shard = Some(chunks);
        Ok(self)
    }

    /// How the floating-point columns are compressed. Legal only before the
    /// first append, which freezes every array's codecs.
    ///
    /// # Errors
    ///
    /// A [`MolRsError::Zarr`] when the first append has already run.
    pub fn with_compression(mut self, compression: Compression) -> Result<Self, MolRsError> {
        self.refuse_after_first_append("with_compression")?;
        self.compression = compression;
        Ok(self)
    }

    /// Land every `frames` appended frames, instead of the derived cadence.
    ///
    /// A cadence that is a whole multiple of the frames one inner chunk holds
    /// lands whole chunks; any other cadence re-encodes a partial trailing
    /// chunk per landing (bounded, but real). `1` lands every frame.
    ///
    /// # Errors
    ///
    /// A [`MolRsError::Zarr`] when `frames` is 0.
    pub fn with_flush_every(mut self, frames: u64) -> Result<Self, MolRsError> {
        if frames == 0 {
            return Err(MolRsError::zarr("flush_every must be at least 1"));
        }
        self.flush_every = Some(frames);
        if self.arrays.is_some() {
            self.auto_flush_every = frames;
        }
        Ok(self)
    }

    /// Whether an explicit [`flush`](Self::flush) / [`close`](Self::close)
    /// syncs the touched files before moving the commit marker. `true` by
    /// default; `false` trades power-loss safety for throughput on a store
    /// where every flush is a checkpoint anyway.
    pub fn with_durable(mut self, durable: bool) -> Self {
        self.durable = durable;
        self
    }

    /// Write `meta` as the record's identity document (`meta/` attributes).
    ///
    /// Replaces whatever the group held.
    ///
    /// # Errors
    ///
    /// Any storage error raised while writing the group.
    pub fn with_meta(
        self,
        meta: &serde_json::Map<String, serde_json::Value>,
    ) -> Result<Self, MolRsError> {
        ensure_root_and_meta(&self.store, Some(meta))?;
        Ok(self)
    }

    /// The landing cadence in force: frames per automatic commit.
    pub fn flush_every(&self) -> u64 {
        self.auto_flush_every
    }

    /// Frames committed so far.
    pub fn committed(&self) -> u64 {
        self.committed
    }

    fn refuse_after_first_append(&self, knob: &str) -> Result<(), MolRsError> {
        if self.arrays.is_some() {
            return Err(MolRsError::zarr(format!(
                "{knob} is legal only before the first append: the chunk and shard extents are \
                 frozen when the first append creates the arrays, and zarrs cannot re-plan a grid \
                 under live data"
            )));
        }
        Ok(())
    }

    /// The landing cadence: enough frames for 4 MiB of
    /// payload, rounded up to a whole number of the coarsest block's
    /// frames-per-chunk so every block column lands whole chunks.
    fn derive_flush_every(&self) -> u64 {
        if let Some(frames) = self.flush_every {
            return frames;
        }
        let frame_bytes = self.schema.frame_bytes().max(1);
        let base = (FLUSH_TARGET_BYTES / frame_bytes).clamp(1, MAX_FLUSH_EVERY);
        let coarsest = self
            .arrays
            .as_ref()
            .map(|arrays| {
                arrays
                    .blocks
                    .iter()
                    .filter_map(|(name, block)| {
                        let rows = self.schema.frame_rows(name);
                        (rows > 0).then(|| (block.rows_per_chunk / rows).max(1))
                    })
                    .max()
                    .unwrap_or(1)
            })
            .unwrap_or(1);
        base.div_ceil(coarsest) * coarsest
    }
}
