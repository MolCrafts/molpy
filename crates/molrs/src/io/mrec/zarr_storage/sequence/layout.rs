use crate::io::mrec::zarr_storage::frame_io::{BOX_GROUP, VALIDITY_GROUP};

// ---------------------------------------------------------------------------
// Layout vocabulary
// ---------------------------------------------------------------------------

/// The record root.
pub(super) const ROOT_GROUP: &str = "/";
/// The identity document's group.
pub(super) const META_ROOT_GROUP: &str = "/meta";
/// The record's `system` section, whose block names an aligned trajectory
/// block may not take.
pub(super) const SYSTEM_GROUP: &str = "/system";
/// The sequence root. Absolute, because every array path is built from it.
pub(super) const TRAJECTORY_GROUP: &str = "/trajectory";
/// The commit marker: `i64[nstep]`, extended last.
pub(super) const STEP_ARRAY: &str = "step";
/// Physical time per frame, `f64[nstep]`, only when the run supplies times.
pub(super) const TIME_ARRAY: &str = "time";
/// The per-step metadata group under `trajectory/`.
pub(super) const META_GROUP: &str = "meta";
/// A block section's CSR row pointer.
pub(super) const OFFSET_ARRAY: &str = "offset";
/// A section's sparse update index: the frame ordinals it changed at.
pub(super) const STEP_INDEX_ARRAY: &str = "step_index";
/// The cell matrices of the `box/` section.
pub(super) const VECTORS_ARRAY: &str = "vectors";
/// The cell origins of the `box/` section (omitted when all zero).
pub(super) const ORIGIN_ARRAY: &str = "origin";
/// The per-axis periodic flags of the `box/` section (omitted when all true).
pub(super) const BOUNDARY_ARRAY: &str = "boundary";
/// The `trajectory/` group attribute the pinned schema lives in.
pub(super) const SCHEMA_ATTRIBUTE: &str = "sequence_schema";
/// The attribute every per-step meta array carries: its exact dtype tag.
pub(super) const META_DTYPE_ATTRIBUTE: &str = "meta_dtype";
/// The `box/` group attribute recording an undefined cell (absent = defined).
pub(super) const CELL_DEFINED_ATTRIBUTE: &str = "cell_defined";
/// Trajectory-group attribute: the commit marker — frames fully on disk.
pub(super) const NSTEP_ATTRIBUTE: &str = "nstep";
/// Trajectory-group attribute: `step` as `{start, stride}` while it is regular.
pub(super) const STEP_PROGRESSION_ATTRIBUTE: &str = "step_progression";
/// Trajectory-group attribute: `time` as `{start, stride}` while it is regular.
pub(super) const TIME_PROGRESSION_ATTRIBUTE: &str = "time_progression";
/// A block section's mirrored structural shape.
pub(super) const STRUCTURAL_SHAPE_ATTRIBUTE: &str = "structural_shape";
/// Block-section hint: every update so far holds this many rows.
pub(super) const UNIFORM_ROWS_ATTRIBUTE: &str = "uniform_rows";
/// Block-section hint: `step_index[j] == j` for every update so far.
pub(super) const DENSE_UPDATES_ATTRIBUTE: &str = "dense_updates";

/// Children of `trajectory/` a block may not be named after.
pub(super) const RESERVED_BLOCK_NAMES: [&str; 4] = [STEP_ARRAY, TIME_ARRAY, META_GROUP, BOX_GROUP];
/// Children of a block section a column may not be named after.
pub(super) const RESERVED_COLUMN_NAMES: [&str; 3] =
    [OFFSET_ARRAY, STEP_INDEX_ARRAY, VALIDITY_GROUP];

// ---------------------------------------------------------------------------
// Extent policy
// ---------------------------------------------------------------------------

/// The fewest bytes one inner chunk of a block column aims for.
///
/// A chunk is the unit the codec compresses and the shard index points at;
/// below this the per-chunk overhead (a 16 B index entry, a codec header) is
/// no longer negligible against the payload. A frame larger than this is its
/// own chunk.
pub(super) const MIN_CHUNK_BYTES: u64 = 16 * 1024;

/// Bytes one shard file aims for, and therefore the file-count lever.
pub(super) const SHARD_TARGET_BYTES: u64 = 256 * 1024 * 1024;

/// The most inner chunks one shard holds, whatever the byte target says.
///
/// The shard index is 16 B per inner chunk and is rewritten in place on every
/// landing, so this caps that rewrite at 64 KiB.
pub(super) const MAX_CHUNKS_PER_SHARD: u64 = 4096;

/// Rows per inner chunk of the dense per-frame / per-update arrays (`step`,
/// `time`, `meta/*`, `offset`, `step_index`, `box/*`): 8 KiB of `u64`.
pub(super) const DENSE_ROWS_PER_CHUNK: u64 = 1024;

/// Inner chunks per shard of the dense arrays: a 4 KiB shard index, so the
/// in-place index rewrite every landing pays on each of them stays small
/// (a flush-per-frame producer rewrites every dense array's index per frame).
/// 256 × 1024 rows is still 262 144 frames per shard file.
pub(super) const DENSE_CHUNKS_PER_SHARD: u64 = 256;

/// Bytes of frame payload the writer buffers before landing on its own.
pub(super) const FLUSH_TARGET_BYTES: u64 = 4 * 1024 * 1024;

/// The most frames the writer buffers before landing on its own.
pub(super) const MAX_FLUSH_EVERY: u64 = 4096;

/// Bytes assumed for one element of a variable-width ([`DType::String`])
/// column when sizing a growth array.
///
/// [`DType::itemsize`] reports `None` there, but a growth array's extents are
/// frozen at creation, so a number has to be chosen. It moves a chunk boundary
/// and nothing else.
pub(super) const ASSUMED_STRING_ITEMSIZE: u64 = 16;

/// Inner chunks a column reader keeps decoded per column.
///
/// Playback reads the same chunk `frames_per_chunk` times in a row; a small
/// LRU turns those repeats into slices. Two entries cover forward play and a
/// one-step scrub back without holding more than two chunks of a huge column.
pub(super) const CHUNK_CACHE_ENTRIES: usize = 2;
