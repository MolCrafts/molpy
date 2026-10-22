//! The `trajectory/` frame sequence: one on-disk layout, one encoder, one
//! decoder.

use crate::core::Cell;
use std::collections::{BTreeMap, VecDeque};
use std::num::NonZeroU64;
use std::sync::{Arc, Mutex, Once};

use indexmap::IndexMap;
use ndarray::{ArrayD, Axis, Slice};
use serde::{Deserialize, Serialize};
use zarrs::array::codec::array_to_bytes::sharding::{
    ShardingCodecBuilder, ShardingCodecOptions, ShardingIndexLocation, SubchunkWriteOrder,
};
use zarrs::array::codec::bytes_to_bytes::crc32c::Crc32cCodec;
use zarrs::array::data_type::{Float16DataType, Float32DataType};
use zarrs::array::{
    Array, ArrayBuilder, ArraySubset, BytesToBytesCodecTraits, CodecOptions, CodecSpecificOptions,
};
use zarrs::config::{global_config, global_config_mut};
use zarrs::group::{Group, GroupBuilder};
use zarrs::node::{Node, NodeMetadata, NodePath, get_child_nodes};
use zarrs::storage::{
    ListableStorageTraits, ReadableListableStorage, ReadableListableStorageTraits,
    ReadableStorageTraits, ReadableWritableListableStorage, ReadableWritableListableStorageTraits,
    StorageHandle, WritableStorageTraits,
};

use molrs::core::Frame;
use molrs::core::MolRsError;
use molrs::core::SimBox;
use molrs::core::Trajectory;
use molrs::core::{Block, Column, DType};
use molrs::core::{MetaMap, MetaValue};
use molrs::op::F;

use crate::io::reader::TrajectoryReader;

use super::frame_io::{
    BOX_GROUP, GZIP_LEVEL, VALIDITY_GROUP, canonical_width, check_canonical_dtype,
    check_local_references, check_storable_cell, default_precision_compressor, gzip,
    insert_column_into_block, join_path, node_prefix, precision_shuffle, quantized,
    read_column_array, resolve_boundary, stored_dtype, zarr_dtype,
};
use super::record_io::zerr;

// ---------------------------------------------------------------------------
// Layout vocabulary
// ---------------------------------------------------------------------------

/// The record root.
const ROOT_GROUP: &str = "/";
/// The identity document's group.
const META_ROOT_GROUP: &str = "/meta";
/// The record's `system` section, whose block names an aligned trajectory
/// block may not take.
const SYSTEM_GROUP: &str = "/system";
/// The sequence root. Absolute, because every array path is built from it.
const TRAJECTORY_GROUP: &str = "/trajectory";
/// The commit marker: `i64[nstep]`, extended last.
const STEP_ARRAY: &str = "step";
/// Physical time per frame, `f64[nstep]`, only when the run supplies times.
const TIME_ARRAY: &str = "time";
/// The per-step metadata group under `trajectory/`.
const META_GROUP: &str = "meta";
/// A block section's CSR row pointer.
const OFFSET_ARRAY: &str = "offset";
/// A section's sparse update index: the frame ordinals it changed at.
const STEP_INDEX_ARRAY: &str = "step_index";
/// The cell matrices of the `box/` section.
const VECTORS_ARRAY: &str = "vectors";
/// The cell origins of the `box/` section (omitted when all zero).
const ORIGIN_ARRAY: &str = "origin";
/// The per-axis periodic flags of the `box/` section (omitted when all true).
const BOUNDARY_ARRAY: &str = "boundary";
/// The `trajectory/` group attribute the pinned schema lives in.
const SCHEMA_ATTRIBUTE: &str = "sequence_schema";
/// The attribute every per-step meta array carries: its exact dtype tag.
const META_DTYPE_ATTRIBUTE: &str = "meta_dtype";
/// The `box/` group attribute recording an undefined cell (absent = defined).
const CELL_DEFINED_ATTRIBUTE: &str = "cell_defined";
/// Trajectory-group attribute: the commit marker — frames fully on disk.
const NSTEP_ATTRIBUTE: &str = "nstep";
/// Trajectory-group attribute: `step` as `{start, stride}` while it is regular.
const STEP_PROGRESSION_ATTRIBUTE: &str = "step_progression";
/// Trajectory-group attribute: `time` as `{start, stride}` while it is regular.
const TIME_PROGRESSION_ATTRIBUTE: &str = "time_progression";
/// A block section's mirrored structural shape.
const STRUCTURAL_SHAPE_ATTRIBUTE: &str = "structural_shape";
/// Block-section hint: every update so far holds this many rows.
const UNIFORM_ROWS_ATTRIBUTE: &str = "uniform_rows";
/// Block-section hint: `step_index[j] == j` for every update so far.
const DENSE_UPDATES_ATTRIBUTE: &str = "dense_updates";

/// Children of `trajectory/` a block may not be named after.
const RESERVED_BLOCK_NAMES: [&str; 4] = [STEP_ARRAY, TIME_ARRAY, META_GROUP, BOX_GROUP];
/// Children of a block section a column may not be named after.
const RESERVED_COLUMN_NAMES: [&str; 3] = [OFFSET_ARRAY, STEP_INDEX_ARRAY, VALIDITY_GROUP];

// ---------------------------------------------------------------------------
// Extent policy
// ---------------------------------------------------------------------------

/// The fewest bytes one inner chunk of a block column aims for.
///
/// A chunk is the unit the codec compresses and the shard index points at;
/// below this the per-chunk overhead (a 16 B index entry, a codec header) is
/// no longer negligible against the payload. A frame larger than this is its
/// own chunk.
const MIN_CHUNK_BYTES: u64 = 16 * 1024;

/// Bytes one shard file aims for, and therefore the file-count lever.
const SHARD_TARGET_BYTES: u64 = 256 * 1024 * 1024;

/// The most inner chunks one shard holds, whatever the byte target says.
///
/// The shard index is 16 B per inner chunk and is rewritten in place on every
/// landing, so this caps that rewrite at 64 KiB.
const MAX_CHUNKS_PER_SHARD: u64 = 4096;

/// Rows per inner chunk of the dense per-frame / per-update arrays (`step`,
/// `time`, `meta/*`, `offset`, `step_index`, `box/*`): 8 KiB of `u64`.
const DENSE_ROWS_PER_CHUNK: u64 = 1024;

/// Inner chunks per shard of the dense arrays: a 4 KiB shard index, so the
/// in-place index rewrite every landing pays on each of them stays small
/// (a flush-per-frame producer rewrites every dense array's index per frame).
/// 256 × 1024 rows is still 262 144 frames per shard file.
const DENSE_CHUNKS_PER_SHARD: u64 = 256;

/// Bytes of frame payload the writer buffers before landing on its own.
const FLUSH_TARGET_BYTES: u64 = 4 * 1024 * 1024;

/// The most frames the writer buffers before landing on its own.
const MAX_FLUSH_EVERY: u64 = 4096;

/// Bytes assumed for one element of a variable-width ([`DType::String`])
/// column when sizing a growth array.
///
/// [`DType::itemsize`] reports `None` there, but a growth array's extents are
/// frozen at creation, so a number has to be chosen. It moves a chunk boundary
/// and nothing else.
const ASSUMED_STRING_ITEMSIZE: u64 = 16;

/// Inner chunks a column reader keeps decoded per column.
///
/// Playback reads the same chunk `frames_per_chunk` times in a row; a small
/// LRU turns those repeats into slices. Two entries cover forward play and a
/// one-step scrub back without holding more than two chunks of a huge column.
const CHUNK_CACHE_ENTRIES: usize = 2;

/// How the floating-point columns of a sequence are compressed.
///
/// Integer, boolean and string columns, and every index array, always carry
/// `gzip` level 1: they compress by structure. Floating-point coordinates do
/// not — 52 random mantissa bits gzip to about 95 % of their size at a real
/// CPU cost — so their compression is a producer's choice, `None` by default.
///
/// A column with a [declared precision](molrs::core::check_precision) is the
/// exception: its values are rounded onto a binary grid, its pipeline opens
/// with a byte shuffle, and it is compressed whatever this says — `None`
/// selects the reference compressor (`zstd` level 3, or `gzip` level 1 in a
/// build that cannot encode `zstd`), `Gzip` / `Zstd` replace it, and the
/// shuffle stays. Every choice is lossless: the rounding is the writer's,
/// before the pipeline.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum Compression {
    /// Raw little-endian bytes. The default for floating-point columns.
    #[default]
    None,
    /// `gzip` at this level (1–9). Every reader of these stores decodes it.
    Gzip(u32),
    /// `zstd` at this level. Encoding needs a native build (`zarr-codecs`);
    /// every reader decodes it, wasm32 included.
    #[cfg(feature = "zarr-codecs")]
    Zstd(i32),
}

/// The bytes-to-bytes pipeline of one growth array: a compressor, and
/// whether a byte shuffle precedes it (a precision column's pipeline).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
struct Pipeline {
    compression: Compression,
    shuffle: bool,
}

impl Pipeline {
    /// A pipeline with no shuffle.
    const fn plain(compression: Compression) -> Self {
        Self {
            compression,
            shuffle: false,
        }
    }
}

/// The frozen extents of one growth array.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
struct Extents {
    rows_per_chunk: u64,
    chunks_per_shard: u64,
}

impl Extents {
    /// Rows one inner chunk of a block column holds.
    ///
    /// Frame-aligned: the smallest whole number of frames of `frame_rows` rows
    /// whose narrowest column reaches [`MIN_CHUNK_BYTES`]. A block whose
    /// representative frame carries no rows (or is unknown) falls back to the
    /// byte floor alone.
    fn block_rows_per_chunk(frame_rows: u64, min_row_bytes: u64) -> u64 {
        let floor = (MIN_CHUNK_BYTES / min_row_bytes.max(1)).max(1);
        if frame_rows == 0 {
            return floor;
        }
        frame_rows.max(floor).div_ceil(frame_rows) * frame_rows
    }

    /// Inner chunks one shard of an array with `chunk_bytes`-byte chunks holds.
    fn chunks_per_shard_for(chunk_bytes: u64) -> u64 {
        (SHARD_TARGET_BYTES / chunk_bytes.max(1)).clamp(1, MAX_CHUNKS_PER_SHARD)
    }

    /// Extents of one block column: `rows_per_chunk` shared by the block,
    /// `chunks_per_shard` from this column's own width.
    fn for_column(rows_per_chunk: u64, row_bytes: u64, chunks_per_shard: Option<u64>) -> Self {
        Self {
            rows_per_chunk,
            chunks_per_shard: chunks_per_shard
                .unwrap_or_else(|| Self::chunks_per_shard_for(rows_per_chunk * row_bytes)),
        }
    }
}

/// The options every data write of this module carries.
///
/// Built explicitly and threaded into the `_opt` methods on purpose: the
/// non-`_opt` methods construct their own defaults, where
/// `experimental_partial_encoding` is `false` and every landing silently
/// becomes an O(shard) rewrite.
fn partial_encoding_options() -> CodecOptions {
    global_config()
        .codec_options()
        .with_experimental_partial_encoding(true)
}

/// Codec options that pin a shard's subchunk layout to C (row-major) order, so
/// a store's bytes do not depend on thread scheduling.
fn deterministic_shard_layout() -> CodecSpecificOptions {
    CodecSpecificOptions::default().with_option(
        ShardingCodecOptions::default().with_subchunk_write_order(SubchunkWriteOrder::C),
    )
}

/// Switch off zarrs' `_zarrs` provenance attribute on every array this process
/// creates. Done once: the attribute is bundle noise on a contract layout, and
/// the flag is process-wide by the library's design.
fn quiet_zarrs_metadata() {
    static ONCE: Once = Once::new();
    ONCE.call_once(|| {
        global_config_mut().set_include_zarrs_metadata(false);
    });
}

/// The bytes-to-bytes codecs of one inner chunk: the byte shuffle of a
/// precision column, the compressor, then `crc32c` so a torn chunk is a
/// checksum error rather than garbage rows.
///
/// Behind a shuffle, [`Compression::None`] means the reference compressor
/// ([`default_precision_compressor`]): a shuffle exists to feed one.
fn inner_codecs(pipeline: Pipeline) -> Result<Vec<Arc<dyn BytesToBytesCodecTraits>>, MolRsError> {
    let mut codecs: Vec<Arc<dyn BytesToBytesCodecTraits>> = Vec::with_capacity(3);
    if pipeline.shuffle {
        codecs.push(precision_shuffle());
    }
    match pipeline.compression {
        Compression::None if pipeline.shuffle => codecs.push(default_precision_compressor()?),
        Compression::None => {}
        Compression::Gzip(level) => codecs.push(gzip(level)?),
        #[cfg(feature = "zarr-codecs")]
        Compression::Zstd(level) => codecs.push(Arc::new(
            zarrs::array::codec::bytes_to_bytes::zstd::ZstdCodec::new(level, false),
        )),
    }
    codecs.push(Arc::new(Crc32cCodec::new()));
    Ok(codecs)
}

/// Whether a column width is floating point — the widths whose compression is
/// the producer's [`Compression`] choice rather than always-`gzip`.
fn is_float_width(dtype: DType) -> bool {
    matches!(dtype, DType::Float | DType::C64 | DType::C128)
}

/// The tag a column's storage width is written as in the schema attributes.
///
/// The three domain aliases (`Float`/`Int`/`UInt`) are tagged with molrec's
/// concrete-width names (`f64`/`i32`/`u64`) so a written schema validates
/// against molrec's published dtype enum and matches the meta-side tags
/// ([`meta_layout`]); every other width is [`DType::name`] verbatim. The match
/// carries no wildcard, so it stays exhaustive over `DType` inside this crate:
/// a width added to the enum stops the build here until [`dtype_from_tag`] can
/// read it back.
fn dtype_tag(dtype: DType) -> &'static str {
    match dtype {
        DType::Float => "f64",
        DType::Int => "i32",
        DType::Uint => "u64",
        DType::I8
        | DType::I16
        | DType::I64
        | DType::Bool
        | DType::U8
        | DType::U16
        | DType::U32
        | DType::String
        | DType::C64
        | DType::C128 => dtype.name(),
    }
}

/// Every width a column may take, in the order molrec's dtype enum lists them.
const SCHEMA_WIDTHS: [DType; 13] = [
    DType::Float,
    DType::I8,
    DType::I16,
    DType::Int,
    DType::I64,
    DType::U8,
    DType::U16,
    DType::U32,
    DType::Uint,
    DType::Bool,
    DType::String,
    DType::C64,
    DType::C128,
];

/// The column width a schema dtype tag names — the public spelling of the
/// closed dtype set (`f64` … `c128`) for callers that declare a schema from
/// text rather than from a frame.
///
/// # Errors
///
/// A [`MolRsError::Zarr`] naming an unknown tag.
pub fn dtype_from_schema_tag(tag: &str) -> Result<DType, MolRsError> {
    dtype_from_tag(tag)
}

/// The column width a schema dtype tag names.
///
/// Reads the thirteen concrete-width tags [`dtype_tag`] writes.
fn dtype_from_tag(tag: &str) -> Result<DType, MolRsError> {
    SCHEMA_WIDTHS
        .iter()
        .copied()
        .find(|&dtype| dtype_tag(dtype) == tag)
        .ok_or_else(|| MolRsError::zarr(format!("unknown column dtype tag {tag:?}")))
}

/// The column width a stored Zarr data type maps to.
///
/// Narrow float arrays are refused, not promoted: the record has one float
/// (`F = f64`), so a `float16`/`float32` array on disk is an error naming
/// the array and its stored type.
fn dtype_of_stored(name: &str, stored: &zarrs::array::DataType) -> Result<DType, MolRsError> {
    if stored.is::<Float16DataType>() || stored.is::<Float32DataType>() {
        return Err(MolRsError::zarr(format!(
            "{name} is stored as {stored:?}: narrow floats are not read; \
             the record has one float, `F = f64`"
        )));
    }
    SCHEMA_WIDTHS
        .iter()
        .copied()
        .find(|&dtype| zarr_dtype(dtype).0 == *stored)
        .ok_or_else(|| {
            MolRsError::zarr(format!(
                "{name} is stored as {stored:?}, which is no column width molrs reads"
            ))
        })
}

/// The column width and trailing shape a per-step meta tag is stored as.
///
/// `None` for a tag no per-step array can store.
fn meta_layout(tag: &str) -> Option<(DType, Vec<u64>)> {
    let scalar = |dtype| Some((dtype, Vec::new()));
    let vector = |dtype, n: u64| Some((dtype, vec![n]));
    match tag {
        "bool" => scalar(DType::Bool),
        "i32" => scalar(DType::Int),
        "i64" => scalar(DType::I64),
        "u32" => scalar(DType::U32),
        "u64" => scalar(DType::Uint),
        "f64" => scalar(DType::Float),
        // A JSON document per step rides in a string array; the tag says how
        // to read it back.
        "string" | "json" => scalar(DType::String),
        "bool3" => vector(DType::Bool, 3),
        "i32x3" => vector(DType::Int, 3),
        "i64x3" => vector(DType::I64, 3),
        "u32x3" => vector(DType::U32, 3),
        "u64x3" => vector(DType::Uint, 3),
        "f64x3" => vector(DType::Float, 3),
        "f64x6" => vector(DType::Float, 6),
        "f64x9" => vector(DType::Float, 9),
        _ => None,
    }
}

/// Bytes one row of a column occupies, trailing axes included.
fn row_bytes(dtype: DType, trailing: &[u64]) -> u64 {
    let width = dtype
        .itemsize()
        .map_or(ASSUMED_STRING_ITEMSIZE, |width| width as u64);
    trailing
        .iter()
        .fold(width, |acc, &axis| acc.saturating_mul(axis.max(1)))
        .max(1)
}

/// The subset covering `rows` rows from `start` of an array whose trailing
/// axes are `trailing`.
fn rows_subset(start: u64, rows: u64, trailing: &[u64]) -> Result<ArraySubset, MolRsError> {
    let mut origin = Vec::with_capacity(trailing.len() + 1);
    origin.push(start);
    origin.extend(std::iter::repeat_n(0u64, trailing.len()));
    let mut shape = Vec::with_capacity(trailing.len() + 1);
    shape.push(rows);
    shape.extend_from_slice(trailing);
    ArraySubset::new_with_start_shape(origin, shape).map_err(zerr)
}

/// Bitwise column equality — the predicate behind a section's `step_index`.
///
/// Bitwise rather than numeric: a NaN coordinate has to count as *unchanged*,
/// or a section that never moves would earn an entry at every step. Compared
/// element by element on the typed arrays, so a comparison allocates nothing.
fn same_column(left: &Column, right: &Column) -> bool {
    if left.dtype() != right.dtype() || left.shape() != right.shape() {
        return false;
    }
    macro_rules! bits {
        ($a:expr, $b:expr) => {
            $a.iter()
                .zip($b.iter())
                .all(|(x, y)| x.to_bits() == y.to_bits())
        };
    }
    macro_rules! exact {
        ($a:expr, $b:expr) => {
            $a.iter().zip($b.iter()).all(|(x, y)| x == y)
        };
    }
    match (left, right) {
        (Column::Float(a), Column::Float(b)) => bits!(a, b),
        (Column::C64(a), Column::C64(b)) => a
            .iter()
            .zip(b.iter())
            .all(|(x, y)| x.re.to_bits() == y.re.to_bits() && x.im.to_bits() == y.im.to_bits()),
        (Column::C128(a), Column::C128(b)) => a
            .iter()
            .zip(b.iter())
            .all(|(x, y)| x.re.to_bits() == y.re.to_bits() && x.im.to_bits() == y.im.to_bits()),
        (Column::I8(a), Column::I8(b)) => exact!(a, b),
        (Column::I16(a), Column::I16(b)) => exact!(a, b),
        (Column::Int(a), Column::Int(b)) => exact!(a, b),
        (Column::I64(a), Column::I64(b)) => exact!(a, b),
        (Column::U8(a), Column::U8(b)) => exact!(a, b),
        (Column::U16(a), Column::U16(b)) => exact!(a, b),
        (Column::U32(a), Column::U32(b)) => exact!(a, b),
        (Column::Uint(a), Column::Uint(b)) => exact!(a, b),
        (Column::Bool(a), Column::Bool(b)) => exact!(a, b),
        (Column::String(a), Column::String(b)) => exact!(a, b),
        _ => false,
    }
}

/// Whether two blocks carry the same content, column for column.
///
/// The validity masks count as content: two frames whose values agree while
/// one holds nulls under them are different frames, and letting the second
/// earn no update would carry the first frame's mask forward over it.
fn same_block(left: &Block, right: &Block) -> bool {
    left.len() == right.len()
        && left.n_rows() == right.n_rows()
        && left.structural_shape() == right.structural_shape()
        && left.iter().all(|(name, column)| {
            right
                .get(name)
                .is_some_and(|other| same_column(column, other))
                && left.validity(name) == right.validity(name)
        })
}

/// `block` as a writer stores it under `declared`: every column whose
/// precision is declared rounded to its grid ([`quantized`]), carrying the
/// declaration; every other column as presented. Borrowed when the block
/// declares no precision.
fn stored_presentation<'a>(
    declared: &BlockSchema,
    block: &'a Block,
) -> Result<std::borrow::Cow<'a, Block>, MolRsError> {
    let mut stored: Option<Block> = None;
    // A canonical key at another width of its family lands at its declared
    // dtype.
    for column in declared.columns.keys() {
        let Some(values) = block.get(column) else {
            continue;
        };
        if let Some(converted) = canonical_width(column, values)? {
            let target = stored.get_or_insert_with(|| block.clone());
            let mask = target.validity(column).map(<[bool]>::to_vec);
            target.insert(column.as_str(), converted, None)?;
            if let Some(mask) = mask {
                target.set_validity(column, mask)?;
            }
        }
    }
    for (column, schema) in &declared.columns {
        let Some(precision) = schema.precision else {
            continue;
        };
        let Some(values) = block.get(column).and_then(Column::get::<f64>) else {
            continue;
        };
        let rounded = Column::from(quantized(values, precision)?);
        let target = stored.get_or_insert_with(|| block.clone());
        // `Block::insert` drops the mask with the old column; the rows did not
        // move, so it goes back on.
        let mask = target.validity(column).map(<[bool]>::to_vec);
        target.insert(column.as_str(), rounded, None)?;
        if let Some(mask) = mask {
            target.set_validity(column, mask)?;
        }
        target.set_precision(column, precision)?;
    }
    // The pinned row-reference targets travel on the stored block, so the
    // resolved-frame check sees them whether the frame stated them or not.
    if declared
        .targets
        .iter()
        .any(|(column, target)| block.contains_key(column) && block.target(column) != Some(target))
    {
        let target_block = stored.get_or_insert_with(|| block.clone());
        for (column, target) in &declared.targets {
            if target_block.contains_key(column) {
                target_block.set_target(column, target)?;
            }
        }
    }
    Ok(match stored {
        Some(block) => std::borrow::Cow::Owned(block),
        None => std::borrow::Cow::Borrowed(block),
    })
}

/// Check a resolved trajectory frame against the rules that span blocks:
/// every aligned block present there has its target present with the same
/// row count, and every same-frame row reference the schema pins resolves.
/// `resolve(name)` is the block at this ordinal, or `None` where it is
/// absent.
fn check_resolved_frame<'a>(
    schema: &SequenceSchema,
    resolve: impl Fn(&str) -> Option<&'a Block>,
    ordinal: u64,
) -> Result<(), MolRsError> {
    for (aligned, target) in schema.aligned_pairs() {
        let Some(block) = resolve(aligned) else {
            continue;
        };
        let rows = block.n_rows().unwrap_or(0);
        match resolve(target) {
            None => {
                return Err(MolRsError::zarr(format!(
                    "trajectory frame {ordinal}: block {aligned:?} ({rows} rows) is aligned with \
                     {target:?}, which is absent there"
                )));
            }
            Some(with) if with.n_rows().unwrap_or(0) != rows => {
                return Err(MolRsError::zarr(format!(
                    "trajectory frame {ordinal}: block {aligned:?} has {rows} rows but the block \
                     it is aligned with, {target:?}, has {}; restate {aligned:?} when \
                     {target:?} changes its row count",
                    with.n_rows().unwrap_or(0)
                )));
            }
            Some(_) => {}
        }
    }
    let mut resolved = Frame::new();
    for name in schema.blocks.keys() {
        if let Some(block) = resolve(name) {
            resolved.insert(name.clone(), block.clone());
        }
    }
    check_local_references(&resolved, &format!("trajectory frame {ordinal}"))
}

/// Whether two cells are the same cell, bit for bit.
fn same_simbox(left: &SimBox, right: &SimBox) -> bool {
    left.is_cell_defined() == right.is_cell_defined()
        && left.pbc() == right.pbc()
        && left
            .matrix()
            .as_flattened()
            .iter()
            .map(|value| value.to_bits())
            .eq(right
                .matrix()
                .as_flattened()
                .iter()
                .map(|value| value.to_bits()))
        && left
            .origin()
            .iter()
            .map(|value| value.to_bits())
            .eq(right.origin().iter().map(|value| value.to_bits()))
}

/// Origin is optional on disk: all zeros is the default and is not written.
fn origin_is_default(cell: &SimBox) -> bool {
    cell.origin().iter().all(|value| *value == 0.0)
}

/// Boundary is optional on disk: all-periodic is the default and is not written.
fn boundary_is_default(cell: &SimBox) -> bool {
    cell.pbc() == [true, true, true]
}

/// Rows `start..end` of `column`, as a new column.
///
/// The whole column comes back as a cheap Arc clone; any other range is one
/// copy of exactly those rows.
fn column_rows(column: &Column, start: usize, end: usize) -> Column {
    let rows = column.n_rows().unwrap_or(0);
    if start == 0 && end == rows {
        return column.clone();
    }
    macro_rules! slice {
        ($holder:expr) => {
            Column::from(
                $holder
                    .slice_axis(Axis(0), Slice::from(start..end))
                    .to_owned(),
            )
        };
    }
    match column {
        Column::Float(h) => slice!(h),
        Column::I8(h) => slice!(h),
        Column::I16(h) => slice!(h),
        Column::Int(h) => slice!(h),
        Column::I64(h) => slice!(h),
        Column::U8(h) => slice!(h),
        Column::U16(h) => slice!(h),
        Column::U32(h) => slice!(h),
        Column::Uint(h) => slice!(h),
        Column::Bool(h) => slice!(h),
        Column::String(h) => slice!(h),
        Column::C64(h) => slice!(h),
        Column::C128(h) => slice!(h),
    }
}

/// A zero-row column of `dtype` with `trailing` axes — the columns of a block
/// that is present and empty.
fn empty_column(dtype: DType, trailing: &[u64]) -> Result<Column, MolRsError> {
    let mut shape = Vec::with_capacity(trailing.len() + 1);
    shape.push(0usize);
    shape.extend(trailing.iter().map(|&n| n as usize));
    macro_rules! empty {
        ($ty:ty) => {
            Column::from(ArrayD::<$ty>::from_shape_vec(shape, Vec::new()).map_err(zerr)?)
        };
    }
    Ok(match dtype {
        DType::Float => empty!(f64),
        DType::I8 => empty!(i8),
        DType::I16 => empty!(i16),
        DType::Int => empty!(i32),
        DType::I64 => empty!(i64),
        DType::U8 => empty!(u8),
        DType::U16 => empty!(u16),
        DType::U32 => empty!(u32),
        DType::Uint => empty!(u64),
        DType::Bool => empty!(bool),
        DType::String => empty!(String),
        DType::C64 => empty!(num_complex::Complex<f32>),
        DType::C128 => empty!(num_complex::Complex<f64>),
    })
}

// ---------------------------------------------------------------------------
// Schema
// ---------------------------------------------------------------------------

/// One declared column: the width and trailing shape it arrived with,
/// whether its rows may be null, and its declared precision.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
struct ColumnSchema {
    /// [`dtype_tag`] of the column's storage width.
    dtype: String,
    /// Axes after the leading row axis, e.g. `[3]` for xyz.
    #[serde(default)]
    trailing: Vec<u64>,
    /// Whether the column carries a [validity
    /// mask](crate::core::Block::validity), stored as a `bool` array
    /// under the section's `_validity` subgroup. Absent from the serialized
    /// pin while false; a pin without it deserializes as a plain column.
    #[serde(default, skip_serializing_if = "std::ops::Not::not")]
    nullable: bool,
    /// The column's [declared precision](molrs::core::check_precision) — on a
    /// trajectory, stated here and nowhere else. Every presented value is
    /// rounded to its grid before the change detection and the landing.
    /// Absent while undeclared.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    precision: Option<f64>,
}

/// One declared block: its columns, and the structural shape it declares.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
struct BlockSchema {
    columns: IndexMap<String, ColumnSchema>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    structural_shape: Option<Vec<usize>>,
    /// The block's row references (molrec `targets`): column → `<block>`
    /// of the same resolved frame or `/<section>/<block>`. On a trajectory
    /// they are pinned here and nowhere else. Absent while empty.
    #[serde(default, skip_serializing_if = "IndexMap::is_empty")]
    targets: IndexMap<String, String>,
    /// The block this one is [aligned with](SequenceSchema::declare_aligned):
    /// its rows are that block's rows, one for one, at every resolved frame.
    /// Declared only, never derived. Absent while unset.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    aligned_with: Option<String>,
}

/// One declared per-step metadata key.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
struct MetaSchema {
    /// The [`MetaValue::dtype`] tag every step must carry.
    dtype: String,
    /// Value written for a step that omits the key, in its typed JSON form
    /// ([`MetaValue::to_typed_json`], so a NaN fill is `"NaN"`) — `dtype`
    /// beside it says how to read it back. `None` makes the omission an
    /// error — there is no implicit fill.
    ///
    /// A present `null` is kept as `Some(Null)` — not folded into "no fill" —
    /// so a pin whose numeric fill a writer lost to `null` is refused as the
    /// broken value it is rather than read as a run without a fill.
    #[serde(
        default,
        skip_serializing_if = "Option::is_none",
        deserialize_with = "present_value"
    )]
    fill: Option<serde_json::Value>,
}

/// Deserialize a field that is present, `null` included, as `Some`; an absent
/// field takes the `#[serde(default)]` `None`.
fn present_value<'de, D>(deserializer: D) -> Result<Option<serde_json::Value>, D::Error>
where
    D: serde::Deserializer<'de>,
{
    serde_json::Value::deserialize(deserializer).map(Some)
}

/// The blocks, columns, dtypes and trailing shapes a sequence is pinned to.
///
/// Two ways to mint one: **derive** it from representative frames
/// ([`from_frames`](Self::from_frames); one frame is a one-element slice,
/// `std::slice::from_ref(&frame)`),
/// or **declare** it column by column ([`new`](Self::new) then
/// [`declare_block`](Self::declare_block) / [`declare_column`](Self::declare_column)
/// / [`declare_meta`](Self::declare_meta)). A producer that knows its columns
/// declares them; one that has a frame in hand derives.
///
/// A later frame may present a *subset* of the declaration — that is how the
/// layout expresses sparsity — but never anything outside it: a run that
/// decides halfway through to record forces needs a new store.
///
/// The declaration also carries a **representative row count** per block
/// (the frame's row count, or the `rows` given to `declare_block`). It is not
/// part of the pin written to the store: it sizes the frame-aligned inner
/// chunk of the block's columns, nothing else, and a ragged run whose row
/// count wanders away from it still reads back exactly.
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct SequenceSchema {
    blocks: IndexMap<String, BlockSchema>,
    #[serde(default)]
    meta: IndexMap<String, MetaSchema>,
    /// Representative rows per block, for chunk sizing. Never pinned.
    #[serde(skip)]
    rows_hint: BTreeMap<String, u64>,
}

impl PartialEq for SequenceSchema {
    /// The pin — blocks and meta — is the identity; the row hints are sizing
    /// advice and two schemas that differ only there are the same schema.
    fn eq(&self, other: &Self) -> bool {
        self.blocks == other.blocks && self.meta == other.meta
    }
}

fn check_block_name(name: &str) -> Result<(), MolRsError> {
    NodePath::new(&join_path(TRAJECTORY_GROUP, name)).map_err(zerr)?;
    if RESERVED_BLOCK_NAMES.contains(&name) {
        return Err(MolRsError::zarr(format!(
            "{name:?} is a reserved child of the trajectory group; a block cannot take it"
        )));
    }
    Ok(())
}

fn check_column_name(column: &str) -> Result<(), MolRsError> {
    NodePath::new(&join_path("/", column)).map_err(zerr)?;
    if RESERVED_COLUMN_NAMES.contains(&column) {
        return Err(MolRsError::zarr(format!(
            "{column:?} is a reserved child of a block section; a column cannot take it"
        )));
    }
    Ok(())
}

impl SequenceSchema {
    /// Preflight an in-memory trajectory without changing its destination.
    pub(super) fn validate_frames(&self, frames: &[Frame]) -> Result<(), MolRsError> {
        let mut resolved = BTreeMap::new();
        for (ordinal, frame) in frames.iter().enumerate() {
            self.validate_frame(frame)?;
            self.resolve_meta(frame)?;
            for (name, block) in frame.iter() {
                let declared = &self.blocks[name];
                resolved.insert(
                    name.to_owned(),
                    stored_presentation(declared, block)?.into_owned(),
                );
            }
            check_resolved_frame(self, |name| resolved.get(name), ordinal as u64)?;
        }
        Ok(())
    }

    /// Check `frame` against the pinned schema.
    ///
    /// The schema pins a dtype, a trailing shape and a nullability per column.
    /// A column declared nullable carries its [validity
    /// mask](crate::core::Block::validity) into the store — a `bool`
    /// array under the section's reserved `_validity` subgroup, grown row for
    /// row with the values — so a masked column round-trips masked, and a
    /// frame that leaves it whole round-trips whole.
    ///
    /// A mask on a column the pin declares **non-nullable** is refused here
    /// rather than dropped: the pin is a contract, and a writer that silently
    /// forgot which rows hold nothing is the data loss this check exists for.
    fn validate_frame(&self, frame: &Frame) -> Result<(), MolRsError> {
        if let Some(simbox) = &frame.simbox {
            check_storable_cell(simbox)?;
        }
        for (name, block) in frame.iter() {
            let Some(declared) = self.blocks.get(name) else {
                return Err(MolRsError::zarr(format!(
                    "block {name:?} is not declared by this sequence: a schema is pinned at \
                     create and cannot grow mid-run"
                )));
            };
            for (column, values) in block.iter() {
                let Some(pinned) = declared.columns.get(column) else {
                    return Err(MolRsError::zarr(format!(
                        "column {column:?} of block {name:?} is not declared by this sequence: a \
                         schema is pinned at create and cannot grow mid-run"
                    )));
                };
                let dtype = dtype_tag(stored_dtype(column, values.dtype()));
                if pinned.dtype != dtype {
                    return Err(MolRsError::zarr(format!(
                        "column {column:?} of block {name:?} is declared {} but this frame carries \
                         {dtype}",
                        pinned.dtype
                    )));
                }
                if let Some(presented) = block.precision(column)
                    && pinned.precision.map(f64::to_bits) != Some(presented.to_bits())
                {
                    return Err(MolRsError::zarr(format!(
                        "column {column:?} of block {name:?} declares precision {presented} but \
                         this sequence pins {}: a trajectory states a column's precision once, in \
                         its schema",
                        pinned
                            .precision
                            .map_or_else(|| "none".to_string(), |p| p.to_string())
                    )));
                }
                if let Some(presented) = block.target(column)
                    && declared.targets.get(column).map(String::as_str) != Some(presented)
                {
                    return Err(MolRsError::zarr(format!(
                        "column {column:?} of block {name:?} references {presented:?} but this \
                         sequence pins {}: a trajectory states a block's targets once, in its \
                         schema",
                        declared
                            .targets
                            .get(column)
                            .map_or_else(|| "none".to_string(), |t| format!("{t:?}"))
                    )));
                }
                if block.validity(column).is_some() && !pinned.nullable {
                    return Err(MolRsError::zarr(format!(
                        "column {column:?} of block {name:?} carries a validity mask but is \
                         declared non-nullable by this sequence: a schema is pinned at create, \
                         and landing the values without the mask would lose which rows hold \
                         nothing"
                    )));
                }
                let trailing: Vec<u64> = values.shape().iter().skip(1).map(|&n| n as u64).collect();
                if pinned.trailing != trailing {
                    return Err(MolRsError::zarr(format!(
                        "column {column:?} of block {name:?} is declared with trailing shape {:?} \
                         but this frame carries {trailing:?}",
                        pinned.trailing
                    )));
                }
            }
            for column in declared.columns.keys() {
                if !block.contains_key(column) {
                    return Err(MolRsError::zarr(format!(
                        "block {name:?} is present at this step but omits its declared column \
                         {column:?}: sparsity is per block, not per column"
                    )));
                }
            }
            if let Some(shape) = &declared.structural_shape {
                let expected = shape.iter().product::<usize>();
                if block.n_rows() != Some(expected) {
                    return Err(MolRsError::zarr(format!(
                        "block {name:?} declares structural shape {shape:?} ({expected} rows) but \
                         this frame carries {} rows: a shaped block keeps its row count",
                        block.n_rows().unwrap_or(0)
                    )));
                }
            }
        }
        Ok(())
    }

    /// Resolve every declared meta key to the value this step stores.
    fn resolve_meta(&self, frame: &Frame) -> Result<MetaMap, MolRsError> {
        for key in frame.meta.keys() {
            if key == STEP_ARRAY || key == TIME_ARRAY {
                continue;
            }
            if !self.meta.contains_key(key) {
                return Err(MolRsError::zarr(format!(
                    "meta key {key:?} is not declared by this sequence: a schema is pinned at \
                     create and cannot grow mid-run"
                )));
            }
        }
        let mut resolved = MetaMap::new();
        for (key, declared) in &self.meta {
            let value = match frame.meta.get(key) {
                Some(value) if value.dtype() == declared.dtype => value.clone(),
                // A value that arrived at another width — a Python float for a
                // declared `f32`, a JSON list for a declared `f64x3` — is
                // re-read at the declared width, exactly; a value that cannot
                // be is the error.
                Some(value) => MetaValue::from_typed_json(&declared.dtype, &value.to_typed_json())
                    .map_err(|e| {
                        MolRsError::zarr(format!(
                            "meta key {key:?} is declared {} but this frame carries {}: {e}",
                            declared.dtype,
                            value.dtype()
                        ))
                    })?,
                None => {
                    let fill = declared.fill.as_ref().ok_or_else(|| {
                        MolRsError::zarr(format!(
                            "this frame omits the declared meta key {key:?} and no fill value was \
                             declared for it; there is no implicit fill"
                        ))
                    })?;
                    // The pin stores the typed JSON form; the declared tag
                    // says how to read it back.
                    MetaValue::from_typed_json(&declared.dtype, fill)
                        .map_err(|e| MolRsError::zarr(format!("meta key {key:?} fill: {e}")))?
                }
            };
            resolved.insert(key.clone(), value);
        }
        Ok(resolved)
    }

    /// An empty declaration, to be filled with `declare_*`.
    pub fn new() -> Self {
        Self::default()
    }

    /// Mint from the **union** of `frames`.
    ///
    /// This is how a heterogeneous run is expressed: blocks and columns are
    /// unioned across the frames, and each frame later presents whichever
    /// subset it has. A column appearing twice with a conflicting dtype or
    /// trailing shape is an error *here*, at mint, rather than at the append
    /// that would have discovered it. The largest row count a block shows
    /// across the frames becomes its representative row count.
    ///
    /// Nullability is unioned the same way: a column masked in **any** of the
    /// frames is declared nullable, because a schema derived from a run whose
    /// first frame is whole must still describe the frame that holes it.
    ///
    /// # Errors
    ///
    /// Every case is a [`MolRsError::Zarr`] naming the offending block, column
    /// or metadata key:
    ///
    /// - a block named `step`, `time`, `meta` or `box`;
    /// - a column named `offset` or `step_index`;
    /// - the same block declaring two different structural shapes;
    /// - the same column declaring two different dtypes or trailing shapes;
    /// - the same column declaring two different precisions;
    /// - the same `meta` key carrying two different `MetaValue` dtypes.
    pub fn from_frames(frames: &[Frame]) -> Result<Self, MolRsError> {
        let mut schema = Self::new();
        let mut shapes: BTreeMap<String, Option<Vec<usize>>> = BTreeMap::new();

        for frame in frames {
            for (name, block) in frame.iter() {
                let rows = block.n_rows().unwrap_or(0) as u64;
                schema.declare_block(name, Some(rows))?;
                let shape = block.structural_shape().map(<[usize]>::to_vec);
                match shapes.get(name) {
                    None => {
                        shapes.insert(name.to_string(), shape);
                    }
                    Some(previous) if *previous == shape => {}
                    Some(previous) => {
                        return Err(MolRsError::zarr(format!(
                            "sequence schema conflict: block {name:?} declares structural shape \
                             {previous:?} in one frame and {shape:?} in another"
                        )));
                    }
                }
                for (column, values) in block.iter() {
                    let trailing: Vec<u64> =
                        values.shape().iter().skip(1).map(|&n| n as u64).collect();
                    // A canonical key is pinned at its declared dtype; the
                    // writer stores a column of another width of the family
                    // converted ([`canonical_width`]).
                    schema.declare_column(
                        name,
                        column,
                        stored_dtype(column, values.dtype()),
                        &trailing,
                    )?;
                    if block.validity(column).is_some() {
                        schema.declare_nullable(name, column)?;
                    }
                    if let Some(precision) = block.precision(column) {
                        schema.declare_precision(name, column, precision)?;
                    }
                    if let Some(target) = block.target(column) {
                        schema.declare_target(name, column, target)?;
                    }
                }
            }

            for (key, value) in frame.meta.iter() {
                // `step` / `time` are the sequence's own arrays, not per-step
                // meta. A producer that stashed the commit marker on the frame
                // must not mint a duplicate `trajectory/meta/step`.
                if key == STEP_ARRAY || key == TIME_ARRAY {
                    continue;
                }
                schema.declare_meta(key, value.dtype(), None)?;
            }
        }

        for (name, shape) in shapes {
            if let Some(shape) = shape {
                schema.declare_structural_shape(&name, &shape)?;
            }
        }
        Ok(schema)
    }

    /// Declare a block, with the row count a typical frame of it carries.
    ///
    /// Declaring an already-declared block only raises its representative
    /// row count. `rows` sizes the block's frame-aligned inner chunk; `None`
    /// (or `0`) leaves the byte floor to decide.
    ///
    /// # Errors
    ///
    /// A [`MolRsError::Zarr`] when `name` is `step`, `time`, `meta` or `box`.
    pub fn declare_block(&mut self, name: &str, rows: Option<u64>) -> Result<(), MolRsError> {
        check_block_name(name)?;
        self.blocks
            .entry(name.to_string())
            .or_insert_with(|| BlockSchema {
                columns: IndexMap::new(),
                structural_shape: None,
                targets: IndexMap::new(),
                aligned_with: None,
            });
        if let Some(rows) = rows {
            let hint = self.rows_hint.entry(name.to_string()).or_insert(0);
            *hint = (*hint).max(rows);
        }
        Ok(())
    }

    /// Declare a column of `block` — its width and trailing shape.
    ///
    /// Declares the block too when it is new. Declaring the same column twice
    /// with the same width and shape is a no-op.
    ///
    /// # Errors
    ///
    /// A [`MolRsError::Zarr`] when `column` is `offset` or `step_index`, when
    /// `block` takes a reserved name, when a canonical identifier key
    /// (`atomi`, `id`, `type_id`, …) is declared at a width other than `u64`,
    /// or when the column is already declared with another width or trailing
    /// shape — a declaration is a pin, and one column cannot mean two widths
    /// in one sequence.
    pub fn declare_column(
        &mut self,
        block: &str,
        column: &str,
        dtype: DType,
        trailing: &[u64],
    ) -> Result<(), MolRsError> {
        check_column_name(column)?;
        // A canonical key declared at another dtype than the vocabulary's
        // could never be read back: a store holds the declared one.
        check_canonical_dtype(column, dtype)?;
        self.declare_block(block, None)?;
        let declared = ColumnSchema {
            dtype: dtype_tag(dtype).to_string(),
            trailing: trailing.to_vec(),
            nullable: false,
            precision: None,
        };
        let entry = self
            .blocks
            .get_mut(block)
            .expect("declare_block just inserted it");
        match entry.columns.get(column) {
            None => {
                // An aligned pair's column sets are disjoint, so a column
                // added to either one may not name a column of the other.
                if let Some(partner) = self.aligned_partner(block)
                    && self.blocks[&partner].columns.contains_key(column)
                {
                    return Err(MolRsError::zarr(format!(
                        "column {column:?} cannot be declared on block {block:?}: it is aligned \
                         with block {partner:?}, which has a column of that name, and an aligned \
                         pair's columns are disjoint"
                    )));
                }
                let entry = self
                    .blocks
                    .get_mut(block)
                    .expect("declare_block just inserted it");
                entry.columns.insert(column.to_string(), declared);
            }
            Some(existing) if existing.dtype != declared.dtype => {
                return Err(MolRsError::zarr(format!(
                    "sequence schema conflict: column {column:?} of block {block:?} is {} in one \
                     declaration and {} in another",
                    existing.dtype, declared.dtype
                )));
            }
            Some(existing) if existing.trailing != declared.trailing => {
                return Err(MolRsError::zarr(format!(
                    "sequence schema conflict: column {column:?} of block {block:?} has trailing \
                     shape {:?} in one declaration and {:?} in another",
                    existing.trailing, declared.trailing
                )));
            }
            Some(_) => {}
        }
        Ok(())
    }

    /// Record that `column` of `block` may carry nulls.
    ///
    /// A union, never a toggle: a column masked in one frame of a run is
    /// nullable for the whole run, so the flag is set and never cleared.
    ///
    /// [`from_frames`](Self::from_frames) sets this for every column it sees a
    /// mask on, so a schema derived from frames needs no call. A schema
    /// declared by hand does: appending a frame whose column carries a mask to
    /// a column this schema declares non-nullable is refused, and this is the
    /// door that opts the column in.
    ///
    /// # Errors
    ///
    /// A [`MolRsError::Zarr`] when `block` names no declared block, or
    /// `column` no declared column of it.
    pub fn declare_nullable(&mut self, block: &str, column: &str) -> Result<(), MolRsError> {
        self.blocks
            .get_mut(block)
            .and_then(|declared| declared.columns.get_mut(column))
            .map(|declared| declared.nullable = true)
            .ok_or_else(|| {
                MolRsError::zarr(format!(
                    "cannot declare column {column:?} of block {block:?} nullable: it is not declared"
                ))
            })
    }

    /// Declare the [precision](molrs::core::check_precision) of `column` of
    /// `block`: an absolute tolerance in the column's units.
    ///
    /// Every frame's values of the column are rounded to the binary grid the
    /// precision implies before they are compared with the previous update
    /// (a change below half the quantum is no change) and before they land;
    /// the column's pipeline opens with a byte shuffle and a compressor. The
    /// declaration is pinned with the schema and is the only place a
    /// trajectory states it. [`from_frames`](Self::from_frames) declares the
    /// precision every presented column carries
    /// ([`Block::precision`](molrs::core::Block::precision)).
    ///
    /// # Errors
    ///
    /// A [`MolRsError::Zarr`] when the column is not declared, is not `f64`,
    /// is already declared with another precision, or when `precision` is
    /// not finite and within `[2^-1000, 2^1000]`.
    pub fn declare_precision(
        &mut self,
        block: &str,
        column: &str,
        precision: f64,
    ) -> Result<(), MolRsError> {
        let declared = self
            .blocks
            .get_mut(block)
            .and_then(|declared| declared.columns.get_mut(column))
            .ok_or_else(|| {
                MolRsError::zarr(format!(
                    "cannot declare a precision for column {column:?} of block {block:?}: it is \
                     not declared"
                ))
            })?;
        if declared.dtype != dtype_tag(DType::Float) {
            return Err(MolRsError::zarr(format!(
                "column {column:?} of block {block:?} is {}; only an f64 column declares a \
                 precision",
                declared.dtype
            )));
        }
        molrs::core::check_precision(precision)
            .map_err(|e| MolRsError::zarr(format!("column {column:?} of block {block:?}: {e}")))?;
        match declared.precision {
            Some(existing) if existing.to_bits() != precision.to_bits() => {
                Err(MolRsError::zarr(format!(
                    "sequence schema conflict: column {column:?} of block {block:?} declares \
                     precision {existing} and {precision}"
                )))
            }
            _ => {
                declared.precision = Some(precision);
                Ok(())
            }
        }
    }

    /// Declare that `column` of `block` holds row indices into `target`:
    /// `<block>` of the same resolved frame, or `/<section>/<block>` of a
    /// frame-shaped section of the record (molrec "row references").
    ///
    /// Pinned with the schema. The writer refuses a frame whose resolved
    /// blocks break a same-frame reference (target absent while the
    /// referencing block has rows, or a non-null value past its row count),
    /// and the reader refuses a store that does.
    /// [`from_frames`](Self::from_frames) declares every
    /// [`Block::target`](molrs::core::Block::target) it sees.
    ///
    /// # Errors
    ///
    /// A [`MolRsError::Zarr`] when the column is not declared, is not `u64`,
    /// already declares another target, or `target` is malformed or names a
    /// trajectory block absolutely (`/trajectory/…`).
    pub fn declare_target(
        &mut self,
        block: &str,
        column: &str,
        target: &str,
    ) -> Result<(), MolRsError> {
        let declared = self.blocks.get_mut(block).ok_or_else(|| {
            MolRsError::zarr(format!(
                "cannot declare a target for column {column:?} of block {block:?}: the block is \
                 not declared"
            ))
        })?;
        let pinned = declared.columns.get(column).ok_or_else(|| {
            MolRsError::zarr(format!(
                "cannot declare a target for column {column:?} of block {block:?}: it is not \
                 declared"
            ))
        })?;
        if pinned.dtype != dtype_tag(DType::Uint) {
            return Err(MolRsError::zarr(format!(
                "column {column:?} of block {block:?} is {}; a row reference is u64",
                pinned.dtype
            )));
        }
        molrs::core::schema::check_target(target)
            .map_err(|e| MolRsError::zarr(format!("column {column:?} of block {block:?}: {e}")))?;
        match declared.targets.get(column) {
            Some(existing) if existing != target => Err(MolRsError::zarr(format!(
                "sequence schema conflict: column {column:?} of block {block:?} references \
                 {existing:?} and {target:?}"
            ))),
            _ => {
                declared
                    .targets
                    .insert(column.to_string(), target.to_string());
                Ok(())
            }
        }
    }

    /// The declared target of `column` of `block`, or `None`.
    pub fn target(&self, block: &str, column: &str) -> Option<&str> {
        self.blocks
            .get(block)
            .and_then(|declared| declared.targets.get(column))
            .map(String::as_str)
    }

    /// The declared precision of `column` of `block`, or `None` when it
    /// declares none (or is not declared).
    pub fn precision(&self, block: &str, column: &str) -> Option<f64> {
        self.blocks
            .get(block)
            .and_then(|declared| declared.columns.get(column))
            .and_then(|declared| declared.precision)
    }

    /// Declare the structural shape of `block` (a volumetric `[nx][ny][nz]`).
    ///
    /// A block with a structural shape carries exactly `product(shape)` rows
    /// in every update; the writer refuses any other row count.
    ///
    /// # Errors
    ///
    /// A [`MolRsError::Zarr`] when `block` is not declared, or is already
    /// declared with a different shape.
    pub fn declare_structural_shape(
        &mut self,
        block: &str,
        shape: &[usize],
    ) -> Result<(), MolRsError> {
        let entry = self.blocks.get_mut(block).ok_or_else(|| {
            MolRsError::zarr(format!(
                "block {block:?} is not declared; declare it before its structural shape"
            ))
        })?;
        if entry.aligned_with.is_some() {
            return Err(MolRsError::zarr(format!(
                "block {block:?} is aligned with another block and declares no structural shape"
            )));
        }
        match &entry.structural_shape {
            Some(existing) if existing.as_slice() != shape => Err(MolRsError::zarr(format!(
                "sequence schema conflict: block {block:?} declares structural shape {existing:?} \
                 and {shape:?}"
            ))),
            _ => {
                let rows = shape
                    .iter()
                    .try_fold(1_usize, |n, &axis| n.checked_mul(axis))
                    .ok_or_else(|| {
                        MolRsError::zarr(format!(
                            "block {block:?}: structural shape product overflows"
                        ))
                    })? as u64;
                entry.structural_shape = Some(shape.to_vec());
                self.rows_hint.insert(block.to_string(), rows);
                Ok(())
            }
        }
    }

    /// Declare `block` aligned with `target` (molrec `ragged.md`, "Aligned
    /// blocks"): its rows are `target`'s rows, one for one, at every
    /// resolved frame — a sparse companion of a block that changes more
    /// often (atom types beside coordinates under proton hopping).
    ///
    /// Wherever `block` is present (or empty) after carry-forward, `target`
    /// must be present with the same row count; so a frame whose `target`
    /// changes row count restates `block`, and one that keeps it may let
    /// `block` carry forward. `block` may be absent while `target` is
    /// present, never the reverse. The writer refuses a frame that breaks
    /// this; the reader refuses a store that does. A reader hands back two
    /// blocks.
    ///
    /// # Errors
    ///
    /// A [`MolRsError::Zarr`] when either block is not declared, `target` is
    /// `block`, either one is already part of another alignment (no chains),
    /// `block` declares a structural shape, the two share a column name, or
    /// `block` is already aligned with another target.
    pub fn declare_aligned(&mut self, block: &str, target: &str) -> Result<(), MolRsError> {
        let refuse = |why: String| {
            Err(MolRsError::zarr(format!(
                "cannot align block {block:?} with {target:?}: {why}"
            )))
        };
        let (Some(aligned), Some(with)) = (self.blocks.get(block), self.blocks.get(target)) else {
            return refuse("both blocks must be declared".into());
        };
        if block == target {
            return refuse("a block is not aligned with itself".into());
        }
        match &aligned.aligned_with {
            Some(existing) if existing == target => return Ok(()),
            Some(existing) => return refuse(format!("it is already aligned with {existing:?}")),
            None => {}
        }
        if with.aligned_with.is_some() {
            return refuse(format!(
                "{target:?} is itself aligned; alignments do not chain"
            ));
        }
        if self
            .blocks
            .values()
            .any(|other| other.aligned_with.as_deref() == Some(block))
        {
            return refuse(format!(
                "{block:?} is the target of another alignment; alignments do not chain"
            ));
        }
        if aligned.structural_shape.is_some() {
            return refuse("an aligned block declares no structural shape".into());
        }
        if let Some(shared) = aligned
            .columns
            .keys()
            .find(|column| with.columns.contains_key(*column))
        {
            return refuse(format!(
                "both declare column {shared:?}, and an aligned pair's columns are disjoint"
            ));
        }
        self.blocks
            .get_mut(block)
            .expect("checked above")
            .aligned_with = Some(target.to_string());
        Ok(())
    }

    /// The block `block` is aligned with, or `None`.
    pub fn aligned_with(&self, block: &str) -> Option<&str> {
        self.blocks
            .get(block)
            .and_then(|declared| declared.aligned_with.as_deref())
    }

    /// The other half of the aligned pair `block` belongs to, either way
    /// round.
    fn aligned_partner(&self, block: &str) -> Option<String> {
        if let Some(target) = self.aligned_with(block) {
            return Some(target.to_string());
        }
        self.blocks
            .iter()
            .find(|(_, declared)| declared.aligned_with.as_deref() == Some(block))
            .map(|(name, _)| name.clone())
    }

    /// Every aligned pair, as `(aligned block, target)`.
    fn aligned_pairs(&self) -> impl Iterator<Item = (&str, &str)> {
        self.blocks.iter().filter_map(|(name, declared)| {
            declared
                .aligned_with
                .as_deref()
                .map(|target| (name.as_str(), target))
        })
    }

    /// Re-check every pinned alignment against the rules
    /// [`declare_aligned`](Self::declare_aligned) enforces — what a pin read
    /// from a store has to satisfy.
    fn check_alignments(&self) -> Result<(), MolRsError> {
        let mut fresh = self.clone();
        for declared in fresh.blocks.values_mut() {
            declared.aligned_with = None;
        }
        for (block, target) in self.aligned_pairs() {
            fresh.declare_aligned(block, target)?;
        }
        Ok(())
    }

    /// Declare a per-step metadata key by its dtype tag, and optionally the
    /// value written for the steps that omit it.
    ///
    /// The tag is one of the scalar forms `bool`, `i32`, `i64`, `u32`, `u64`,
    /// `f64`, `string`, `json`, the three-component forms `bool3`,
    /// `i32x3`, `i64x3`, `u32x3`, `u64x3`, `f64x3`, or the six- and
    /// nine-component float forms `f64x6`, `f64x9`. Without a `fill`, a frame
    /// that omits the key is an error at append; with one, the omitted steps
    /// store `fill`. There is deliberately no implicit NaN: a NaN nobody asked
    /// for is a measurement nobody made. The fill is recorded in the pinned
    /// schema, so a reopened writer keeps honouring it.
    ///
    /// # Errors
    ///
    /// A [`MolRsError::Zarr`] naming `key` when the tag is unknown, when
    /// `fill`'s dtype is not `dtype`, or when the key is already declared with
    /// a different dtype.
    pub fn declare_meta(
        &mut self,
        key: &str,
        dtype: &str,
        fill: Option<MetaValue>,
    ) -> Result<(), MolRsError> {
        NodePath::new(&join_path(&join_path(TRAJECTORY_GROUP, META_GROUP), key)).map_err(zerr)?;
        if meta_layout(dtype).is_none() {
            return Err(MolRsError::zarr(format!(
                "meta key {key:?} cannot be stored per step: unknown dtype {dtype}"
            )));
        }
        if let Some(fill) = &fill
            && fill.dtype() != dtype
        {
            return Err(MolRsError::zarr(format!(
                "meta key {key:?} is declared {dtype} but its fill is a {}",
                fill.dtype()
            )));
        }
        match self.meta.get(key) {
            Some(existing) if existing.dtype != dtype => {
                return Err(MolRsError::zarr(format!(
                    "sequence schema conflict: meta key {key:?} is {} in one declaration and \
                     {dtype} in another",
                    existing.dtype
                )));
            }
            Some(_) => {}
            None => {
                self.meta.insert(
                    key.to_string(),
                    MetaSchema {
                        dtype: dtype.to_string(),
                        fill: None,
                    },
                );
            }
        }
        if let Some(fill) = fill {
            self.meta
                .get_mut(key)
                .expect("the key was just declared")
                .fill = Some(fill.to_typed_json());
        }
        Ok(())
    }

    /// The declared block names, in declaration order.
    pub fn block_names(&self) -> impl Iterator<Item = &str> {
        self.blocks.keys().map(String::as_str)
    }

    /// The declared column names of `block` in declaration order, or `None`
    /// when it is not declared.
    pub fn column_names(&self, block: &str) -> Option<impl Iterator<Item = &str>> {
        self.blocks
            .get(block)
            .map(|declared| declared.columns.keys().map(String::as_str))
    }

    /// The declared per-step metadata keys with their dtype tags.
    ///
    /// Keys are yielded in declaration order, the order
    /// [`declare_meta`](Self::declare_meta) first inserted each key.
    /// [`from_frames`](Self::from_frames) declares in `frame.meta`'s iteration order, and reading a pinned
    /// sequence inserts into the returned frame in this same order.
    pub fn meta_keys(&self) -> impl Iterator<Item = (&str, &str)> {
        self.meta
            .iter()
            .map(|(key, declared)| (key.as_str(), declared.dtype.as_str()))
    }

    /// The representative row count of `block`, `0` when none was given.
    fn frame_rows(&self, block: &str) -> u64 {
        self.rows_hint.get(block).copied().unwrap_or(0)
    }

    /// Bytes one frame of the representative row counts occupies across every
    /// declared column — the payload estimate the automatic landing cadence
    /// is derived from.
    fn frame_bytes(&self) -> u64 {
        self.blocks
            .iter()
            .map(|(name, block)| {
                let per_row: u64 = block
                    .columns
                    .values()
                    .map(|column| {
                        dtype_from_tag(&column.dtype)
                            .map_or(8, |dtype| row_bytes(dtype, &column.trailing))
                    })
                    .sum();
                per_row.saturating_mul(self.frame_rows(name))
            })
            .sum()
    }
}

/// Read the schema back out of the `trajectory/` group attributes, if one is
/// pinned there.
///
/// A missing pin is `Ok(None)`, not an error, because it is the one fact the
/// two doors answer differently: [`schema_of`] refuses it, and
/// [`MrecReader::open`] derives from the store instead.
fn pinned_schema<S>(store: &Arc<S>) -> Result<Option<SequenceSchema>, MolRsError>
where
    S: ?Sized + ReadableStorageTraits + 'static,
{
    let group = Group::open(store.clone(), TRAJECTORY_GROUP)?;
    let Some(attribute) = group.attributes().get(SCHEMA_ATTRIBUTE) else {
        return Ok(None);
    };
    let schema: SequenceSchema = serde_json::from_value(attribute.clone()).map_err(zerr)?;
    // A fill is a typed JSON value of its key's tag; one that is not (a NaN
    // a writer lost to `null`) is a malformed pin, refused here.
    for (key, declared) in &schema.meta {
        if let Some(fill) = &declared.fill {
            MetaValue::from_typed_json(&declared.dtype, fill).map_err(|e| {
                MolRsError::zarr(format!(
                    "{SCHEMA_ATTRIBUTE}: the fill of meta key {key:?} is not a {}: {e}",
                    declared.dtype
                ))
            })?;
        }
    }
    Ok(Some(schema))
}

/// [`pinned_schema`], requiring the pin — the writer's half of the seam:
/// appending needs the *declared* union and the meta fills, neither of which
/// is recoverable from data that was never written.
fn schema_of<S>(store: &Arc<S>) -> Result<SequenceSchema, MolRsError>
where
    S: ?Sized + ReadableStorageTraits + 'static,
{
    pinned_schema(store)?.ok_or_else(|| {
        MolRsError::zarr(format!(
            "{TRAJECTORY_GROUP} carries no {SCHEMA_ATTRIBUTE} attribute; it is not a molrs frame \
             sequence"
        ))
    })
}

/// The direct child nodes of the group at `path` — one level deep, metadata
/// only. A path holding no children yields an empty vector.
fn children<S>(store: &Arc<S>, path: &str) -> Result<Vec<Node>, MolRsError>
where
    S: ?Sized + ReadableStorageTraits + ListableStorageTraits + 'static,
{
    let path: NodePath = path.try_into().map_err(zerr)?;
    Ok(get_child_nodes(store, &path, false)?)
}

/// The schema of a store that pins none: derived from the layout itself.
///
/// A block is a child *group* of `trajectory/` that is not one of
/// [`RESERVED_BLOCK_NAMES`]; its columns are that group's arrays that are not
/// one of [`RESERVED_COLUMN_NAMES`]; each column's width and trailing shape
/// are its own array metadata. Per-step metadata keys are
/// `trajectory/meta/`'s arrays, each carrying its own
/// [`META_DTYPE_ATTRIBUTE`] tag. Every derived fill is `None`: a fill is what
/// a *writer* chose, and a reader that was never told cannot say more.
fn schema_from_storage<S>(store: &Arc<S>) -> Result<SequenceSchema, MolRsError>
where
    S: ?Sized + ReadableStorageTraits + ListableStorageTraits + 'static,
{
    let mut schema = SequenceSchema::new();
    for section in children(store, TRAJECTORY_GROUP)? {
        if !matches!(section.metadata(), NodeMetadata::Group(_)) {
            continue;
        }
        let name = section.name().as_str().to_string();
        if RESERVED_BLOCK_NAMES.contains(&name.as_str()) {
            continue;
        }
        let path = join_path(TRAJECTORY_GROUP, &name);
        schema.declare_block(&name, None)?;

        for child in children(store, &path)? {
            if !matches!(child.metadata(), NodeMetadata::Array(_)) {
                continue;
            }
            let column = child.name().as_str().to_string();
            if RESERVED_COLUMN_NAMES.contains(&column.as_str()) {
                continue;
            }
            let column_path = join_path(&path, &column);
            let array = Array::open(store.clone(), &column_path)?;
            let dtype = dtype_of_stored(&column_path, array.data_type())?;
            let trailing: Vec<u64> = array.shape().iter().skip(1).copied().collect();
            schema.declare_column(&name, &column, dtype, &trailing)?;
            // A store whose pin was stripped still says which columns are
            // nullable: the mask arrays are on disk.
            let mask_path = join_path(&join_path(&path, VALIDITY_GROUP), &column);
            if array_exists(store, &mask_path)? {
                schema.declare_nullable(&name, &column)?;
            }
        }

        // The section mirrors its structural shape for exactly this reader.
        let structural_shape = Group::open(store.clone(), &path)?
            .attributes()
            .get(STRUCTURAL_SHAPE_ATTRIBUTE)
            .and_then(serde_json::Value::as_array)
            .map(|axes| {
                axes.iter()
                    .filter_map(|axis| axis.as_u64().map(|axis| axis as usize))
                    .collect::<Vec<usize>>()
            })
            .filter(|shape| !shape.is_empty());
        if let Some(shape) = structural_shape {
            schema.declare_structural_shape(&name, &shape)?;
        }
    }

    let meta_path = join_path(TRAJECTORY_GROUP, META_GROUP);
    for child in children(store, &meta_path)? {
        if !matches!(child.metadata(), NodeMetadata::Array(_)) {
            continue;
        }
        let key = child.name().as_str().to_string();
        let key_path = join_path(&meta_path, &key);
        let array = Array::open(store.clone(), &key_path)?;
        let tag = array
            .attributes()
            .get(META_DTYPE_ATTRIBUTE)
            .and_then(serde_json::Value::as_str)
            .ok_or_else(|| {
                MolRsError::zarr(format!(
                    "{key_path} carries no {META_DTYPE_ATTRIBUTE} attribute; its per-step values \
                     cannot be read back"
                ))
            })?;
        schema.declare_meta(&key, tag, None)?;
    }

    Ok(schema)
}

/// Whether an array exists at `path`.
fn array_exists<S>(store: &Arc<S>, path: &str) -> Result<bool, MolRsError>
where
    S: ?Sized + ReadableStorageTraits + 'static,
{
    match Array::open(store.clone(), path) {
        Ok(_) => Ok(true),
        Err(zarrs::array::ArrayCreateError::MissingMetadata) => Ok(false),
        Err(e) => Err(e.into()),
    }
}

/// Whether a group exists at `path`.
fn group_exists<S>(store: &Arc<S>, path: &str) -> Result<bool, MolRsError>
where
    S: ?Sized + ReadableStorageTraits + 'static,
{
    match Group::open(store.clone(), path) {
        Ok(_) => Ok(true),
        Err(zarrs::group::GroupCreateError::MissingMetadata) => Ok(false),
        Err(e) => Err(e.into()),
    }
}

/// Every element of the array at `path`.
fn read_whole<T, S>(store: &Arc<S>, path: &str) -> Result<Vec<T>, MolRsError>
where
    T: zarrs::array::ElementOwned,
    S: ?Sized + ReadableStorageTraits + 'static,
{
    let array = Array::open(store.clone(), path)?;
    let subset = ArraySubset::new_with_shape(array.shape().to_vec());
    array.retrieve_array_subset(&subset).map_err(zerr)
}

/// Rows one inner chunk of `array` holds: the sharding subchunk's leading
/// extent, or the chunk grid's when the array is not sharded.
fn inner_rows_of<S: ?Sized>(array: &Array<S>) -> u64 {
    let metadata = serde_json::to_value(array.metadata()).unwrap_or_default();
    let sharded = metadata["codecs"]
        .as_array()
        .and_then(|codecs| {
            codecs
                .iter()
                .find(|codec| codec["name"] == "sharding_indexed")
        })
        .and_then(|codec| codec["configuration"]["chunk_shape"][0].as_u64());
    sharded
        .or_else(|| metadata["chunk_grid"]["configuration"]["chunk_shape"][0].as_u64())
        .unwrap_or(1)
        .max(1)
}

/// Ensure the record root and its `meta/` group exist, writing `meta` when
/// one is handed in.
fn ensure_root_and_meta(
    store: &ReadableWritableListableStorage,
    meta: Option<&serde_json::Map<String, serde_json::Value>>,
) -> Result<(), MolRsError> {
    if !group_exists(store, ROOT_GROUP)? {
        GroupBuilder::new()
            .build(store.clone(), ROOT_GROUP)?
            .store_metadata()?;
    }
    if meta.is_some() || !group_exists(store, META_ROOT_GROUP)? {
        let attrs = meta.cloned().unwrap_or_default();
        GroupBuilder::new()
            .attributes(attrs)
            .build(store.clone(), META_ROOT_GROUP)?
            .store_metadata()?;
    }
    Ok(())
}

// ---------------------------------------------------------------------------
// Growth arrays
// ---------------------------------------------------------------------------

/// A `zarrs` array whose leading axis grows and whose extents never do.
///
/// `zarrs::Array::set_shape` grows the leading axis and rebuilds the chunk grid
/// from *frozen* chunk metadata, so the inner-chunk and shard extents are
/// decided once, at creation. A landing is three steps in a fixed order —
/// [`reserve`](Self::reserve) the rows, write them, then
/// [`commit_shape`](Self::commit_shape) — so the array's metadata never
/// advertises rows that are not on disk yet.
struct GrowthArray {
    array: Array<dyn ReadableWritableListableStorageTraits>,
    /// Rows landed so far (the leading extent the metadata will publish).
    rows: u64,
}

impl GrowthArray {
    /// Create the array at `path`, zero rows long, extents frozen.
    ///
    /// Every growth array is a `sharding_indexed` array with its index at the
    /// **start** of the shard, so appending a chunk is a tail write plus an
    /// in-place index rewrite. The inner chunk carries the codecs `pipeline`
    /// names ([`inner_codecs`]), ending in `crc32c`.
    fn create(
        store: &ReadableWritableListableStorage,
        path: &str,
        dtype: DType,
        trailing: &[u64],
        extents: Extents,
        attributes: serde_json::Map<String, serde_json::Value>,
        pipeline: Pipeline,
    ) -> Result<Self, MolRsError> {
        quiet_zarrs_metadata();
        let (data_type, fill) = zarr_dtype(dtype);
        let mut shape = Vec::with_capacity(trailing.len() + 1);
        shape.push(0u64);
        shape.extend_from_slice(trailing);
        // A chunk extent must be non-zero on every axis, including a trailing
        // axis that happens to be empty.
        let mut inner: Vec<u64> = shape.iter().map(|&axis| axis.max(1)).collect();
        inner[0] = extents.rows_per_chunk.max(1);
        let mut shard = inner.clone();
        shard[0] = inner[0].saturating_mul(extents.chunks_per_shard.max(1));

        let subchunk: Vec<NonZeroU64> = inner
            .iter()
            .map(|&axis| NonZeroU64::new(axis).expect("chunk extents are floored at one"))
            .collect();
        let mut sharding = ShardingCodecBuilder::new(subchunk, &data_type);
        sharding
            .bytes_to_bytes_codecs(inner_codecs(pipeline)?)
            .index_location(ShardingIndexLocation::Start);

        let mut builder = ArrayBuilder::new(shape, shard, data_type, fill);
        builder.array_to_bytes_codec(sharding.build_arc());
        builder.attributes(attributes);
        let array = builder
            .build(store.clone(), path)?
            .with_codec_specific_options(&deterministic_shard_layout());
        array.store_metadata()?;
        Ok(Self { array, rows: 0 })
    }

    /// Reopen an existing array, adopting the extents it was created with.
    fn open(store: &ReadableWritableListableStorage, path: &str) -> Result<Self, MolRsError> {
        let array = Array::open(store.clone(), path)?
            .with_codec_specific_options(&deterministic_shard_layout());
        let rows = array.shape().first().copied().unwrap_or(0);
        Ok(Self { array, rows })
    }

    /// The array's axes after the leading row axis.
    fn trailing(&self) -> Vec<u64> {
        self.array.shape().iter().skip(1).copied().collect()
    }

    /// Rows one inner chunk holds.
    fn rows_per_chunk(&self) -> u64 {
        inner_rows_of(&self.array)
    }

    /// Grow the leading axis by `count` rows **in memory** and return the
    /// subset they land in. The caller writes the data, then
    /// [`commit_shape`](Self::commit_shape) publishes the new extent.
    fn reserve(&mut self, count: u64) -> Result<ArraySubset, MolRsError> {
        let start = self.rows;
        self.rows += count;
        let mut shape = self.array.shape().to_vec();
        shape[0] = self.rows;
        self.array.set_shape(shape)?;
        rows_subset(start, count, &self.trailing())
    }

    /// Publish the reserved extent: one atomic `zarr.json` replacement.
    fn commit_shape(&self) -> Result<(), MolRsError> {
        self.array.store_metadata()?;
        Ok(())
    }

    /// Roll the leading axis back to `rows` — what a reopen does to whatever a
    /// crash left longer than the commit marker.
    fn truncate_to(&mut self, rows: u64) -> Result<(), MolRsError> {
        if rows >= self.rows {
            return Ok(());
        }
        let mut shape = self.array.shape().to_vec();
        shape[0] = rows;
        self.array.set_shape(shape)?;
        self.array.store_metadata()?;
        self.rows = rows;
        Ok(())
    }

    /// Land `columns` — one per update carrying this block — as one
    /// contiguous row range.
    fn store_columns(
        &self,
        subset: &ArraySubset,
        dtype: DType,
        columns: &[&Column],
        options: &CodecOptions,
    ) -> Result<(), MolRsError> {
        macro_rules! landed {
            ($variant:ident) => {{
                let mut data = Vec::new();
                for column in columns {
                    let Column::$variant(holder) = column else {
                        return Err(MolRsError::zarr(format!(
                            "column changed width mid-buffer: expected {dtype}, found {}",
                            column.dtype()
                        )));
                    };
                    data.extend(holder.as_standard_layout().iter().cloned());
                }
                self.array.store_array_subset_opt(subset, data, options)?;
            }};
        }
        match dtype {
            DType::Float => landed!(Float),
            DType::I8 => landed!(I8),
            DType::I16 => landed!(I16),
            DType::Int => landed!(Int),
            DType::I64 => landed!(I64),
            DType::Bool => landed!(Bool),
            DType::Uint => landed!(Uint),
            DType::U8 => landed!(U8),
            DType::U16 => landed!(U16),
            DType::U32 => landed!(U32),
            DType::String => landed!(String),
            DType::C64 => landed!(C64),
            DType::C128 => landed!(C128),
        }
        Ok(())
    }

    /// Land one typed value per pending frame into a per-step meta array.
    fn store_meta(
        &self,
        subset: &ArraySubset,
        key: &str,
        tag: &str,
        values: &[&MetaValue],
        options: &CodecOptions,
    ) -> Result<(), MolRsError> {
        let mismatch = |value: &MetaValue| {
            MolRsError::zarr(format!(
                "meta key {key:?} is declared {tag} but a frame carries {}",
                value.dtype()
            ))
        };
        macro_rules! scalar {
            ($variant:ident, $ty:ty) => {{
                let mut data: Vec<$ty> = Vec::with_capacity(values.len());
                for value in values.iter().copied() {
                    let MetaValue::$variant(inner) = value else {
                        return Err(mismatch(value));
                    };
                    data.push(inner.clone());
                }
                self.array.store_array_subset_opt(subset, data, options)?;
            }};
        }
        macro_rules! vector {
            ($variant:ident, $ty:ty) => {{
                let mut data: Vec<$ty> = Vec::new();
                for value in values.iter().copied() {
                    let MetaValue::$variant(inner) = value else {
                        return Err(mismatch(value));
                    };
                    data.extend_from_slice(inner);
                }
                self.array.store_array_subset_opt(subset, data, options)?;
            }};
        }
        match tag {
            "bool" => scalar!(Bool, bool),
            "i32" => scalar!(I32, i32),
            "i64" => scalar!(I64, i64),
            "u32" => scalar!(U32, u32),
            "u64" => scalar!(U64, u64),
            "f64" => scalar!(F64, f64),
            "string" => scalar!(String, String),
            "json" => {
                let mut data: Vec<String> = Vec::with_capacity(values.len());
                for value in values.iter().copied() {
                    let MetaValue::Json(inner) = value else {
                        return Err(mismatch(value));
                    };
                    data.push(serde_json::to_string(inner).map_err(zerr)?);
                }
                self.array.store_array_subset_opt(subset, data, options)?;
            }
            "bool3" => vector!(Bool3, bool),
            "i32x3" => vector!(I32x3, i32),
            "i64x3" => vector!(I64x3, i64),
            "u32x3" => vector!(U32x3, u32),
            "u64x3" => vector!(U64x3, u64),
            "f64x3" => vector!(F64x3, f64),
            "f64x6" => vector!(F64x6, f64),
            "f64x9" => vector!(F64x9, f64),
            other => {
                return Err(MolRsError::zarr(format!(
                    "meta key {key:?} declares unknown dtype {other:?}"
                )));
            }
        }
        Ok(())
    }
}

/// Reserve, write, publish — the landing order every plain-valued array uses.
macro_rules! land_values {
    ($array:expr, $count:expr, $values:expr, $options:expr) => {{
        let subset = $array.reserve($count)?;
        $array
            .array
            .store_array_subset_opt(&subset, $values, $options)?;
        $array.commit_shape()?;
    }};
}

/// One value of a per-step meta array, at step `index`.
fn read_meta_value<S>(
    array: &Array<S>,
    key: &str,
    tag: &str,
    index: u64,
) -> Result<MetaValue, MolRsError>
where
    S: ?Sized + ReadableStorageTraits + 'static,
{
    let (_, trailing) = meta_layout(tag).ok_or_else(|| {
        MolRsError::zarr(format!("meta key {key:?} declares unknown dtype {tag:?}"))
    })?;
    let subset = rows_subset(index, 1, &trailing)?;
    let short = |count: usize| {
        MolRsError::zarr(format!(
            "meta key {key:?} is {tag} but step {index} holds {count} value(s)"
        ))
    };
    macro_rules! scalar {
        ($variant:ident, $ty:ty) => {{
            let values: Vec<$ty> = array.retrieve_array_subset(&subset)?;
            let count = values.len();
            MetaValue::$variant(values.into_iter().next().ok_or_else(|| short(count))?)
        }};
    }
    macro_rules! vector {
        ($variant:ident, $ty:ty, $len:expr) => {{
            let values: Vec<$ty> = array.retrieve_array_subset(&subset)?;
            let values: [$ty; $len] = values
                .try_into()
                .map_err(|values: Vec<$ty>| short(values.len()))?;
            MetaValue::$variant(values)
        }};
    }
    Ok(match tag {
        "bool" => scalar!(Bool, bool),
        "i32" => scalar!(I32, i32),
        "i64" => scalar!(I64, i64),
        "u32" => scalar!(U32, u32),
        "u64" => scalar!(U64, u64),
        "f64" => scalar!(F64, f64),
        "string" => scalar!(String, String),
        "json" => {
            let values: Vec<String> = array.retrieve_array_subset(&subset)?;
            let count = values.len();
            let text = values.into_iter().next().ok_or_else(|| short(count))?;
            MetaValue::Json(serde_json::from_str(&text).map_err(zerr)?)
        }
        "bool3" => vector!(Bool3, bool, 3),
        "i32x3" => vector!(I32x3, i32, 3),
        "i64x3" => vector!(I64x3, i64, 3),
        "u32x3" => vector!(U32x3, u32, 3),
        "u64x3" => vector!(U64x3, u64, 3),
        "f64x3" => vector!(F64x3, f64, 3),
        "f64x6" => vector!(F64x6, f64, 6),
        "f64x9" => vector!(F64x9, f64, 9),
        other => {
            return Err(MolRsError::zarr(format!(
                "meta key {key:?} declares unknown dtype {other:?}"
            )));
        }
    })
}

/// A declared block with no rows: every declared column, zero rows long.
fn empty_block(schema: &BlockSchema) -> Result<Block, MolRsError> {
    let mut block = Block::new();
    for (column, declared) in &schema.columns {
        let dtype = dtype_from_tag(&declared.dtype)?;
        insert_column_into_block(&mut block, column, empty_column(dtype, &declared.trailing)?)?;
    }
    Ok(block)
}

/// Apply a declared structural shape to a block whose row count matches it.
fn apply_structural_shape(
    block: &mut Block,
    schema: &BlockSchema,
    path: &str,
) -> Result<(), MolRsError> {
    if let Some(shape) = &schema.structural_shape
        && block.n_rows() == Some(shape.iter().product::<usize>())
    {
        block
            .set_shape(shape)
            .map_err(|e| MolRsError::zarr(format!("block {path:?} shape {shape:?}: {e}")))?;
    }
    Ok(())
}

// ---------------------------------------------------------------------------
// Block hints
// ---------------------------------------------------------------------------

/// Whether every update of a block so far carried the same row count.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum Uniform {
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
struct BlockHints {
    uniform: Uniform,
    dense: bool,
    /// What the group attributes currently say, so a rewrite happens only on
    /// a change.
    stored: (Option<u64>, bool),
}

impl BlockHints {
    fn fresh() -> Self {
        Self {
            uniform: Uniform::Unknown,
            dense: true,
            stored: (None, false),
        }
    }

    /// Rebuild from a section's index arrays — what a reopen does.
    fn from_index(step_index: &[u64], offset: &[u64]) -> Self {
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
    fn observe(&mut self, update: u64, ordinal: u64, rows: u64) {
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
    fn is_regular(&self) -> bool {
        matches!(self.uniform, Uniform::Rows(n) if n > 0) && self.dense
    }

    /// The attributes these hints spell: `(uniform_rows, dense_updates)`.
    fn attributes(&self) -> (Option<u64>, bool) {
        match self.uniform {
            Uniform::Rows(n) if n > 0 => (Some(n), self.dense),
            Uniform::Rows(_) | Uniform::Unknown | Uniform::Broken => (None, false),
        }
    }
}

/// The group attributes of a block section: its mirrored structural shape
/// plus whichever hints currently hold.
fn block_attributes(
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
fn stored_hints<S>(store: &Arc<S>, path: &str) -> Result<(Option<u64>, bool), MolRsError>
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
fn elision_markers<S>(
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
fn store_block_attributes(
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

// ---------------------------------------------------------------------------
// Section arrays
// ---------------------------------------------------------------------------

/// The two test knobs that override the derived extents of every array.
#[derive(Debug, Clone, Copy, Default)]
struct Knobs {
    rows_per_chunk: Option<u64>,
    chunks_per_shard: Option<u64>,
}

impl Knobs {
    /// Extents of a dense array under these knobs.
    fn dense(self) -> Extents {
        Extents {
            rows_per_chunk: self.rows_per_chunk.unwrap_or(DENSE_ROWS_PER_CHUNK),
            chunks_per_shard: self.chunks_per_shard.unwrap_or(DENSE_CHUNKS_PER_SHARD),
        }
    }
}

/// The pipeline every dense array carries.
const DENSE_PIPELINE: Pipeline = Pipeline::plain(Compression::Gzip(GZIP_LEVEL));

/// The arrays of one block section.
///
/// The CSR index — `offset` and `step_index` — is **not** written while the
/// block is regular (every update carries the same non-zero row count at
/// ordinals `0, 1, 2, …`): the block group's `uniform_rows` /
/// `dense_updates` hints let a reader resolve any frame arithmetically, and
/// the two arrays would only restate what the column shapes already say. The
/// first update that breaks either rule materializes both arrays, backfilling
/// the regular history, and withdraws the hints for good.
struct BlockArrays {
    columns: BTreeMap<String, GrowthArray>,
    /// The validity mask of each column pinned nullable: a `bool` array under
    /// the section's `_validity` subgroup, grown in lockstep with the columns
    /// so a mask row sits at the row index of the value it qualifies.
    masks: BTreeMap<String, GrowthArray>,
    /// The CSR index, once the block stopped being regular.
    index: Option<IndexArrays>,
    /// Rows landed across every update.
    total_rows: u64,
    /// Updates landed.
    updates: u64,
    /// Rows one inner chunk of this block's columns holds.
    rows_per_chunk: u64,
    hints: BlockHints,
}

impl BlockArrays {
    /// Land one flag per landed row into every mask array of this section.
    ///
    /// A block that carries no mask for a nullable column lands all-true: the
    /// mask array is dense over the section's rows, which is what keeps a
    /// frame's flags at exactly the row range its values occupy, and what
    /// lets a column be holed in one frame and whole in the next.
    fn land_masks(
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
fn create_mask_arrays(
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
fn open_mask_arrays(
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
struct IndexArrays {
    offset: GrowthArray,
    step_index: GrowthArray,
}

impl IndexArrays {
    /// Create both arrays and backfill the `history` regular updates of
    /// `rows` rows each — the updates the hints describe.
    fn materialize(
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

/// Where the cell of a sequence lives.
///
/// A cell that arrives at ordinal 0 and never changes is recorded as the
/// `box/` group's own attributes — no array, no extra file. The first change
/// migrates it into the `box/` arrays.
enum CellStorage {
    /// One cell, at ordinal 0, in the group attributes.
    Attributes(Box<SimBox>),
    /// The per-update arrays.
    Arrays(Box<BoxArrays>),
}

/// The arrays of the `box/` section.
///
/// `vectors` is the only required array. Origin (default zeros), boundary
/// (default all-periodic) and `step_index` (implied `[0]` for a single
/// update at ordinal 0) are created only when they carry information.
struct BoxArrays {
    step_index: Option<GrowthArray>,
    vectors: GrowthArray,
    origin: Option<GrowthArray>,
    boundary: Option<GrowthArray>,
}

impl BoxArrays {
    fn create(store: &ReadableWritableListableStorage, knobs: Knobs) -> Result<Self, MolRsError> {
        Ok(Self {
            step_index: None,
            vectors: Self::create_optional(store, VECTORS_ARRAY, DType::Float, &[3, 3], knobs)?,
            origin: None,
            boundary: None,
        })
    }

    fn create_optional(
        store: &ReadableWritableListableStorage,
        name: &str,
        dtype: DType,
        trailing: &[u64],
        knobs: Knobs,
    ) -> Result<GrowthArray, MolRsError> {
        let prefix = join_path(TRAJECTORY_GROUP, BOX_GROUP);
        GrowthArray::create(
            store,
            &join_path(&prefix, name),
            dtype,
            trailing,
            knobs.dense(),
            serde_json::Map::new(),
            DENSE_PIPELINE,
        )
    }

    fn ensure_origin(
        &mut self,
        store: &ReadableWritableListableStorage,
        knobs: Knobs,
        previous: u64,
        options: &CodecOptions,
    ) -> Result<&mut GrowthArray, MolRsError> {
        if self.origin.is_none() {
            let mut origin = Self::create_optional(store, ORIGIN_ARRAY, DType::Float, &[3], knobs)?;
            if previous > 0 {
                let fill = vec![0.0 as F; (previous * 3) as usize];
                land_values!(origin, previous, fill, options);
            }
            self.origin = Some(origin);
        }
        Ok(self.origin.as_mut().expect("origin was just created"))
    }

    fn ensure_boundary(
        &mut self,
        store: &ReadableWritableListableStorage,
        knobs: Knobs,
        previous: u64,
        options: &CodecOptions,
    ) -> Result<&mut GrowthArray, MolRsError> {
        if self.boundary.is_none() {
            let mut boundary =
                Self::create_optional(store, BOUNDARY_ARRAY, DType::Bool, &[3], knobs)?;
            if previous > 0 {
                let fill = vec![true; (previous * 3) as usize];
                land_values!(boundary, previous, fill, options);
            }
            self.boundary = Some(boundary);
        }
        Ok(self.boundary.as_mut().expect("boundary was just created"))
    }

    fn ensure_step_index(
        &mut self,
        store: &ReadableWritableListableStorage,
        knobs: Knobs,
        previous: u64,
        options: &CodecOptions,
    ) -> Result<&mut GrowthArray, MolRsError> {
        if self.step_index.is_none() {
            let mut step_index =
                Self::create_optional(store, STEP_INDEX_ARRAY, DType::Uint, &[], knobs)?;
            if previous > 0 {
                // The omitted form is one update at ordinal 0.
                let fill = vec![0u64; previous as usize];
                land_values!(step_index, previous, fill, options);
            }
            self.step_index = Some(step_index);
        }
        Ok(self
            .step_index
            .as_mut()
            .expect("step_index was just created"))
    }

    /// Land `cells` — `(ordinal, cell)` pairs — as updates.
    fn land(
        &mut self,
        store: &ReadableWritableListableStorage,
        knobs: Knobs,
        cells: &[(u64, &SimBox)],
        options: &CodecOptions,
    ) -> Result<(), MolRsError> {
        let mut vectors = Vec::with_capacity(cells.len() * 9);
        let mut origins = Vec::with_capacity(cells.len() * 3);
        let mut boundaries = Vec::with_capacity(cells.len() * 3);
        let mut need_origin = self.origin.is_some();
        let mut need_boundary = self.boundary.is_some();
        for (_, cell) in cells {
            vectors.extend(cell.matrix().as_flattened().iter().copied());
            origins.extend(cell.origin().iter().copied());
            boundaries.extend(cell.pbc().iter().copied());
            need_origin |= !origin_is_default(cell);
            need_boundary |= !boundary_is_default(cell);
        }
        let count = cells.len() as u64;
        let previous = self.vectors.rows;
        let ordinals: Vec<u64> = cells.iter().map(|(ordinal, _)| *ordinal).collect();
        let trivial_index = previous == 0 && matches!(ordinals.as_slice(), [0]);

        land_values!(self.vectors, count, vectors, options);
        if need_origin {
            let origin = self.ensure_origin(store, knobs, previous, options)?;
            land_values!(origin, count, origins, options);
        }
        if need_boundary {
            let boundary = self.ensure_boundary(store, knobs, previous, options)?;
            land_values!(boundary, count, boundaries, options);
        }
        if self.step_index.is_some() || !trivial_index {
            let step_index = self.ensure_step_index(store, knobs, previous, options)?;
            land_values!(step_index, count, ordinals, options);
        }
        Ok(())
    }

    /// Roll every array back to `updates` committed cell updates.
    fn truncate_to(&mut self, updates: u64) -> Result<(), MolRsError> {
        self.vectors.truncate_to(updates)?;
        for array in [&mut self.step_index, &mut self.origin, &mut self.boundary]
            .into_iter()
            .flatten()
        {
            array.truncate_to(updates)?;
        }
        Ok(())
    }
}

/// The `box/` group attributes: `cell_defined` when false, plus the fixed
/// cell itself (`vectors`, and `origin` / `boundary` off their defaults) when
/// the cell lives in the attributes.
fn box_attributes(
    cell_defined: bool,
    fixed: Option<&SimBox>,
) -> serde_json::Map<String, serde_json::Value> {
    let mut attributes = serde_json::Map::new();
    if !cell_defined {
        attributes.insert(CELL_DEFINED_ATTRIBUTE.to_string(), false.into());
    }
    if let Some(cell) = fixed {
        let h = ndarray::arr2(&cell.matrix());
        let rows: Vec<serde_json::Value> = (0..3)
            .map(|i| serde_json::json!([h[[i, 0]], h[[i, 1]], h[[i, 2]]]))
            .collect();
        attributes.insert(VECTORS_ARRAY.to_string(), serde_json::Value::Array(rows));
        if !origin_is_default(cell) {
            let o = cell.origin();
            attributes.insert(
                ORIGIN_ARRAY.to_string(),
                serde_json::json!([o[0], o[1], o[2]]),
            );
        }
        if !boundary_is_default(cell) {
            attributes.insert(BOUNDARY_ARRAY.to_string(), serde_json::json!(cell.pbc()));
        }
    }
    attributes
}

/// The fixed cell recorded in the `box/` group attributes, if any.
fn cell_from_attributes(
    attributes: &serde_json::Map<String, serde_json::Value>,
) -> Result<Option<SimBox>, MolRsError> {
    let Some(vectors) = attributes.get(VECTORS_ARRAY) else {
        return Ok(None);
    };
    let cell_defined = attributes
        .get(CELL_DEFINED_ATTRIBUTE)
        .and_then(serde_json::Value::as_bool)
        .unwrap_or(true);
    // An undefined cell's `vectors` mean nothing: whatever is there is
    // accepted and never parsed or inverted; `Cell::Undefined` carries the
    // identity in its place.
    let flat: Vec<F> = if !cell_defined {
        vec![1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0]
    } else {
        match vectors.as_array() {
            Some(rows) if rows.len() == 3 && rows.iter().all(serde_json::Value::is_array) => rows
                .iter()
                .flat_map(|row| row.as_array().into_iter().flatten())
                .filter_map(serde_json::Value::as_f64)
                .collect(),
            Some(values) => values
                .iter()
                .filter_map(serde_json::Value::as_f64)
                .collect(),
            None => Vec::new(),
        }
    };
    if flat.len() != 9 {
        return Err(MolRsError::zarr(format!(
            "box attribute {VECTORS_ARRAY:?} must hold a 3x3 cell, found {vectors}"
        )));
    }
    let origin: Vec<F> = match attributes.get(ORIGIN_ARRAY).and_then(|v| v.as_array()) {
        Some(values) => values
            .iter()
            .filter_map(serde_json::Value::as_f64)
            .collect(),
        None => vec![0.0, 0.0, 0.0],
    };
    let boundary: Option<Vec<bool>> = attributes
        .get(BOUNDARY_ARRAY)
        .and_then(|v| v.as_array())
        .map(|values| {
            values
                .iter()
                .filter_map(serde_json::Value::as_bool)
                .collect()
        });
    if origin.len() != 3 || boundary.as_ref().is_some_and(|flags| flags.len() != 3) {
        return Err(MolRsError::zarr(
            "box attributes origin / boundary must hold three values each".to_string(),
        ));
    }
    let pbc = resolve_boundary(
        boundary.map(|flags| [flags[0], flags[1], flags[2]]),
        cell_defined,
        "trajectory box attributes",
    )?;
    SimBox::new(
        if cell_defined {
            Cell::Matrix(std::array::from_fn(|r| {
                std::array::from_fn(|c| flat[3 * r + c])
            }))
        } else {
            Cell::Undefined
        },
        [origin[0], origin[1], origin[2]],
        pbc,
    )
    .map(Some)
    .map_err(|e| MolRsError::zarr(format!("box attributes are not a valid cell: {e:?}")))
}

/// Write the `box/` group with these attributes (creating it if needed).
fn store_box_attributes(
    store: &ReadableWritableListableStorage,
    attributes: serde_json::Map<String, serde_json::Value>,
) -> Result<(), MolRsError> {
    let prefix = join_path(TRAJECTORY_GROUP, BOX_GROUP);
    GroupBuilder::new()
        .attributes(attributes)
        .build(store.clone(), &prefix)?
        .store_metadata()?;
    Ok(())
}

// ---------------------------------------------------------------------------
// Dense per-frame series: progression or array
// ---------------------------------------------------------------------------

/// A value type a dense per-frame series (`step`, `time`) holds.
trait Series: Copy + PartialEq + zarrs::array::Element + zarrs::array::ElementOwned {
    const DTYPE: DType;
    /// `start + i * stride`, or `None` when it does not fit the type.
    fn nth(start: Self, stride: Self, i: u64) -> Option<Self>;
    /// The stride from `a` to `b`, or `None` when it does not fit the type.
    fn stride_between(a: Self, b: Self) -> Option<Self>;
    fn to_json(self) -> Option<serde_json::Value>;
    fn from_json(value: &serde_json::Value) -> Option<Self>;
}

impl Series for i64 {
    const DTYPE: DType = DType::I64;
    fn nth(start: Self, stride: Self, i: u64) -> Option<Self> {
        let i = i64::try_from(i).ok()?;
        start.checked_add(stride.checked_mul(i)?)
    }
    fn stride_between(a: Self, b: Self) -> Option<Self> {
        b.checked_sub(a)
    }
    fn to_json(self) -> Option<serde_json::Value> {
        Some(self.into())
    }
    fn from_json(value: &serde_json::Value) -> Option<Self> {
        value.as_i64()
    }
}

impl Series for f64 {
    const DTYPE: DType = DType::Float;
    fn nth(start: Self, stride: Self, i: u64) -> Option<Self> {
        Some(start + (i as f64) * stride)
    }
    fn stride_between(a: Self, b: Self) -> Option<Self> {
        Some(b - a)
    }
    fn to_json(self) -> Option<serde_json::Value> {
        // JSON has no NaN / infinity; such a series goes to an array.
        self.is_finite().then(|| self.into())
    }
    fn from_json(value: &serde_json::Value) -> Option<Self> {
        value.as_f64()
    }
}

/// An arithmetic progression `start + i * stride` recorded as a trajectory
/// group attribute — how `step` and `time` are kept while they are regular.
#[derive(Debug, Clone, Copy, PartialEq)]
struct Progression<T> {
    start: T,
    /// Unknown until the second value.
    stride: Option<T>,
}

impl<T: Series> Progression<T> {
    /// Parse the attribute form `{"start": …, "stride": …}`.
    fn from_attribute(value: &serde_json::Value) -> Option<Self> {
        let start = T::from_json(value.get("start")?)?;
        let stride = value.get("stride").and_then(T::from_json);
        Some(Self { start, stride })
    }

    fn to_attribute(self) -> Option<serde_json::Value> {
        let mut object = serde_json::Map::new();
        object.insert("start".to_string(), self.start.to_json()?);
        if let Some(stride) = self.stride {
            object.insert("stride".to_string(), stride.to_json()?);
        }
        Some(serde_json::Value::Object(object))
    }

    /// The `i`-th value, or `None` past what the progression can say.
    fn nth(&self, i: u64) -> Option<T> {
        if i == 0 {
            return Some(self.start);
        }
        T::nth(self.start, self.stride?, i)
    }

    /// The first `count` values.
    fn values(&self, count: u64) -> Result<Vec<T>, MolRsError> {
        (0..count)
            .map(|i| {
                self.nth(i).ok_or_else(|| {
                    MolRsError::zarr(format!(
                        "progression attribute cannot produce value {i}: stride unknown or overflow"
                    ))
                })
            })
            .collect()
    }
}

/// Where a dense per-frame series lives: in the trajectory group's attribute
/// while it is regular, in its own array once it is not.
enum Track<T: Series> {
    Regular {
        progression: Option<Progression<T>>,
        /// Values landed so far.
        count: u64,
    },
    Array(Box<GrowthArray>),
}

impl<T: Series> Track<T> {
    fn fresh() -> Self {
        Self::Regular {
            progression: None,
            count: 0,
        }
    }

    /// Whether appending `values` keeps the series regular, and the
    /// progression it becomes.
    fn extended(&self, values: &[T]) -> Option<Progression<T>> {
        let Self::Regular { progression, count } = self else {
            return None;
        };
        let mut progression = *progression;
        for (index, &value) in (*count..).zip(values.iter()) {
            progression = match progression {
                None => Some(Progression {
                    start: value,
                    stride: None,
                }),
                Some(Progression {
                    start,
                    stride: None,
                }) => Some(Progression {
                    start,
                    stride: Some(T::stride_between(start, value)?),
                }),
                Some(p) => {
                    if p.nth(index)? != value {
                        return None;
                    }
                    Some(p)
                }
            };
            // The attribute must be able to spell it.
            progression?.to_attribute()?;
        }
        progression
    }

    /// The last value landed.
    fn last(
        &self,
        store: &ReadableWritableListableStorage,
        path: &str,
    ) -> Result<Option<T>, MolRsError> {
        match self {
            Self::Regular { progression, count } => Ok(match (progression, *count) {
                (Some(p), n) if n > 0 => p.nth(n - 1),
                _ => None,
            }),
            Self::Array(array) => {
                if array.rows == 0 {
                    return Ok(None);
                }
                let values: Vec<T> = read_whole(store, path)?;
                Ok(values.get(array.rows as usize - 1).copied())
            }
        }
    }

    /// Land `values`: extend the progression, or materialize the array (backfilling the regular history) and append.
    fn land(
        &mut self,
        store: &ReadableWritableListableStorage,
        path: &str,
        knobs: Knobs,
        values: Vec<T>,
        options: &CodecOptions,
    ) -> Result<(), MolRsError> {
        if let Some(progression) = self.extended(&values) {
            let Self::Regular { count, .. } = self else {
                unreachable!("extended() is Some only for a regular track")
            };
            let count = *count + values.len() as u64;
            *self = Self::Regular {
                progression: Some(progression),
                count,
            };
            return Ok(());
        }
        if let Self::Regular { progression, count } = self {
            let mut array = GrowthArray::create(
                store,
                path,
                T::DTYPE,
                &[],
                knobs.dense(),
                serde_json::Map::new(),
                DENSE_PIPELINE,
            )?;
            if *count > 0 {
                let history = progression
                    .as_ref()
                    .ok_or_else(|| MolRsError::zarr("regular track without a progression"))?
                    .values(*count)?;
                let n = history.len() as u64;
                land_values!(array, n, history, options);
            }
            *self = Self::Array(Box::new(array));
        }
        let Self::Array(array) = self else {
            unreachable!("materialized above")
        };
        let n = values.len() as u64;
        land_values!(array, n, values, options);
        Ok(())
    }

    /// The attribute this track wants on the trajectory group, if any.
    fn attribute(&self) -> Option<serde_json::Value> {
        match self {
            Self::Regular {
                progression: Some(p),
                count,
            } if *count > 0 => p.to_attribute(),
            _ => None,
        }
    }
}

/// Read a dense series back: the progression attribute when present, else
/// the array cut to `nstep`, else `None`.
fn read_track<T: Series, S>(
    store: &Arc<S>,
    attributes: &serde_json::Map<String, serde_json::Value>,
    attribute: &str,
    path: &str,
    nstep: u64,
) -> Result<Option<Vec<T>>, MolRsError>
where
    S: ?Sized + ReadableStorageTraits + 'static,
{
    if let Some(value) = attributes.get(attribute) {
        let progression = Progression::<T>::from_attribute(value).ok_or_else(|| {
            MolRsError::zarr(format!(
                "trajectory attribute {attribute:?} is not a progression"
            ))
        })?;
        return Ok(Some(progression.values(nstep)?));
    }
    if array_exists(store, path)? {
        let mut values: Vec<T> = read_whole(store, path)?;
        values.truncate(nstep as usize);
        return Ok(Some(values));
    }
    Ok(None)
}

/// The commit marker and the regular series, read off the trajectory group.
struct TrajectoryAttributes {
    nstep: Option<u64>,
    all: serde_json::Map<String, serde_json::Value>,
}

fn trajectory_attributes<S>(store: &Arc<S>) -> Result<TrajectoryAttributes, MolRsError>
where
    S: ?Sized + ReadableStorageTraits + 'static,
{
    let group = Group::open(store.clone(), TRAJECTORY_GROUP)?;
    let all = group.attributes().clone();
    let nstep = match all.get(NSTEP_ATTRIBUTE) {
        None => None,
        Some(value) => Some(value.as_u64().ok_or_else(|| {
            MolRsError::zarr(format!(
                "{TRAJECTORY_GROUP}: {NSTEP_ATTRIBUTE} must be a non-negative integer, found \
                 {value}"
            ))
        })?),
    };
    // A progression is only ever written in the same metadata update as the
    // commit marker, so one without it is a malformed store.
    if nstep.is_none() {
        for progression in [STEP_PROGRESSION_ATTRIBUTE, TIME_PROGRESSION_ATTRIBUTE] {
            if all.contains_key(progression) {
                return Err(MolRsError::zarr(format!(
                    "{TRAJECTORY_GROUP} carries {progression} without {NSTEP_ATTRIBUTE}: a \
                     progression is written with the commit marker, never alone"
                )));
            }
        }
    }
    Ok(TrajectoryAttributes { nstep, all })
}

/// The committed frame count: the `nstep` attribute, or — for a store from a
/// writer that kept no marker attribute — the length of the `step` array.
fn committed_frames<S>(store: &Arc<S>, attributes: &TrajectoryAttributes) -> Result<u64, MolRsError>
where
    S: ?Sized + ReadableStorageTraits + 'static,
{
    if let Some(nstep) = attributes.nstep {
        return Ok(nstep);
    }
    let path = join_path(TRAJECTORY_GROUP, STEP_ARRAY);
    if array_exists(store, &path)? {
        return Ok(Array::open(store.clone(), &path)?
            .shape()
            .first()
            .copied()
            .unwrap_or(0));
    }
    Ok(0)
}

/// Every array of one sequence, created at the first append and never
/// re-planned afterwards.
struct SequenceArrays {
    step: Track<i64>,
    time: Option<Track<f64>>,
    meta: IndexMap<String, GrowthArray>,
    blocks: BTreeMap<String, BlockArrays>,
    cell: Option<CellStorage>,
    /// Frames committed — the `nstep` attribute.
    nstep: u64,
}

impl SequenceArrays {
    /// Create every zero-length array the schema declares.
    fn create(
        store: &ReadableWritableListableStorage,
        schema: &SequenceSchema,
        knobs: Knobs,
        compression: Compression,
        with_time: bool,
    ) -> Result<Self, MolRsError> {
        let dense = |store: &ReadableWritableListableStorage,
                     path: &str,
                     dtype: DType,
                     trailing: &[u64],
                     attributes: serde_json::Map<String, serde_json::Value>| {
            GrowthArray::create(
                store,
                path,
                dtype,
                trailing,
                knobs.dense(),
                attributes,
                DENSE_PIPELINE,
            )
        };

        let mut meta = IndexMap::new();
        if !schema.meta.is_empty() {
            GroupBuilder::new()
                .build(store.clone(), &join_path(TRAJECTORY_GROUP, META_GROUP))?
                .store_metadata()?;
        }
        for (key, declared) in &schema.meta {
            let (dtype, trailing) = meta_layout(&declared.dtype).ok_or_else(|| {
                MolRsError::zarr(format!(
                    "meta key {key:?} declares unknown dtype {:?}",
                    declared.dtype
                ))
            })?;
            // The dtype rides with the array, which is what makes exactness
            // free: no convention has to be remembered to read it back.
            let mut attributes = serde_json::Map::new();
            attributes.insert(
                META_DTYPE_ATTRIBUTE.to_string(),
                serde_json::Value::String(declared.dtype.clone()),
            );
            meta.insert(
                key.clone(),
                dense(
                    store,
                    &join_path(&join_path(TRAJECTORY_GROUP, META_GROUP), key),
                    dtype,
                    &trailing,
                    attributes,
                )?,
            );
        }

        let mut blocks = BTreeMap::new();
        for (name, declared) in &schema.blocks {
            let path = join_path(TRAJECTORY_GROUP, name);
            let hints = BlockHints::fresh();
            GroupBuilder::new()
                .attributes(block_attributes(
                    declared.structural_shape.as_deref(),
                    hints.attributes(),
                ))
                .build(store.clone(), &path)?
                .store_metadata()?;

            // Every column of a block shares one rows-per-chunk — a whole
            // number of representative frames, sized so the narrowest column
            // still reaches the byte floor — so a frame's rows stay aligned
            // across the section.
            let widths = declared
                .columns
                .values()
                .map(|column| {
                    dtype_from_tag(&column.dtype).map(|dtype| row_bytes(dtype, &column.trailing))
                })
                .collect::<Result<Vec<u64>, MolRsError>>()?;
            let narrowest = widths.iter().copied().min().unwrap_or(8);
            let rows_per_chunk = knobs.rows_per_chunk.unwrap_or_else(|| {
                Extents::block_rows_per_chunk(schema.frame_rows(name), narrowest)
            });
            let mut columns = BTreeMap::new();
            for ((column, column_schema), width) in declared.columns.iter().zip(widths) {
                let dtype = dtype_from_tag(&column_schema.dtype)?;
                let pipeline = if column_schema.precision.is_some() {
                    Pipeline {
                        compression,
                        shuffle: true,
                    }
                } else if is_float_width(dtype) {
                    Pipeline::plain(compression)
                } else {
                    Pipeline::plain(Compression::Gzip(GZIP_LEVEL))
                };
                columns.insert(
                    column.clone(),
                    GrowthArray::create(
                        store,
                        &join_path(&path, column),
                        dtype,
                        &column_schema.trailing,
                        Extents::for_column(rows_per_chunk, width, knobs.chunks_per_shard),
                        serde_json::Map::new(),
                        pipeline,
                    )?,
                );
            }
            let masks = create_mask_arrays(store, &path, declared, rows_per_chunk, knobs)?;
            blocks.insert(
                name.clone(),
                BlockArrays {
                    columns,
                    masks,
                    index: None,
                    total_rows: 0,
                    updates: 0,
                    rows_per_chunk,
                    hints,
                },
            );
        }

        Ok(Self {
            step: Track::fresh(),
            time: with_time.then(Track::fresh),
            meta,
            blocks,
            cell: None,
            nstep: 0,
        })
    }

    /// Reopen every array the schema declares, validate each against it, and
    /// **roll back** anything a crash left longer than the commit marker.
    ///
    /// The `nstep` attribute is the truth: every per-frame series is cut to
    /// it, every section's index to the updates at ordinals below it, and
    /// every column to the rows those updates own.
    fn open(
        store: &ReadableWritableListableStorage,
        schema: &SequenceSchema,
    ) -> Result<Self, MolRsError> {
        let attributes = trajectory_attributes(store)?;
        let nstep = committed_frames(store, &attributes)?;

        let step_path = join_path(TRAJECTORY_GROUP, STEP_ARRAY);
        let step = if let Some(value) = attributes.all.get(STEP_PROGRESSION_ATTRIBUTE) {
            Track::Regular {
                progression: Progression::<i64>::from_attribute(value),
                count: nstep,
            }
        } else if array_exists(store, &step_path)? {
            let mut array = GrowthArray::open(store, &step_path)?;
            array.truncate_to(nstep)?;
            Track::Array(Box::new(array))
        } else {
            Track::fresh()
        };
        let time_path = join_path(TRAJECTORY_GROUP, TIME_ARRAY);
        let time = if let Some(value) = attributes.all.get(TIME_PROGRESSION_ATTRIBUTE) {
            Some(Track::Regular {
                progression: Progression::<f64>::from_attribute(value),
                count: nstep,
            })
        } else if array_exists(store, &time_path)? {
            let mut array = GrowthArray::open(store, &time_path)?;
            array.truncate_to(nstep)?;
            Some(Track::Array(Box::new(array)))
        } else {
            None
        };

        let mut meta = IndexMap::new();
        for key in schema.meta.keys() {
            let mut array = GrowthArray::open(
                store,
                &join_path(&join_path(TRAJECTORY_GROUP, META_GROUP), key),
            )?;
            array.truncate_to(nstep)?;
            meta.insert(key.clone(), array);
        }

        let mut blocks = BTreeMap::new();
        for (name, declared) in &schema.blocks {
            let path = join_path(TRAJECTORY_GROUP, name);
            let mut columns = BTreeMap::new();
            let mut rows_per_chunk = None;
            for (column, column_schema) in &declared.columns {
                let column_path = join_path(&path, column);
                let opened = GrowthArray::open(store, &column_path)?;
                validate_column(&opened, &column_path, column_schema)?;
                rows_per_chunk.get_or_insert_with(|| opened.rows_per_chunk());
                columns.insert(column.clone(), opened);
            }
            let stored = stored_hints(store, &path)?;

            let (index, updates, total_rows, mut hints) =
                if array_exists(store, &join_path(&path, STEP_INDEX_ARRAY))? {
                    let mut step_index =
                        GrowthArray::open(store, &join_path(&path, STEP_INDEX_ARRAY))?;
                    let mut offset = GrowthArray::open(store, &join_path(&path, OFFSET_ARRAY))?;
                    let ordinals =
                        read_whole::<u64, _>(store, &join_path(&path, STEP_INDEX_ARRAY))?;
                    let updates = ordinals
                        .iter()
                        .take_while(|&&ordinal| ordinal < nstep)
                        .count();
                    step_index.truncate_to(updates as u64)?;
                    offset.truncate_to(if updates == 0 { 0 } else { updates as u64 + 1 })?;
                    let mut offsets = read_whole::<u64, _>(store, &join_path(&path, OFFSET_ARRAY))?;
                    offsets.truncate(if updates == 0 { 0 } else { updates + 1 });
                    let total_rows = offsets.last().copied().unwrap_or(0);
                    (
                        Some(IndexArrays { offset, step_index }),
                        updates as u64,
                        total_rows,
                        BlockHints::from_index(&ordinals[..updates], &offsets),
                    )
                } else {
                    match elision_markers(store, &path, !declared.columns.is_empty())? {
                        Some(rows) if !columns.is_empty() => {
                            let landed = columns.values().map(|c| c.rows).min().unwrap_or(0);
                            let updates = (landed / rows).min(nstep);
                            let mut hints = BlockHints::fresh();
                            for update in 0..updates {
                                hints.observe(update, update, rows);
                            }
                            (None, updates, updates * rows, hints)
                        }
                        // No committed update: whatever the columns hold is
                        // a torn landing.
                        _ => (None, 0, 0, BlockHints::fresh()),
                    }
                };
            for column in columns.values_mut() {
                column.truncate_to(total_rows)?;
            }
            let rows_per_chunk = rows_per_chunk.unwrap_or(1);
            // The extents of a reopened section come off its arrays, so a
            // mask created here takes the block's own rows-per-chunk and
            // derives the rest, exactly as the create path would.
            let mut masks = open_mask_arrays(
                store,
                &path,
                declared,
                total_rows,
                rows_per_chunk,
                Knobs::default(),
            )?;
            for mask in masks.values_mut() {
                mask.truncate_to(total_rows)?;
            }
            hints.stored = stored;
            let wanted = if index.is_some() {
                (None, false)
            } else {
                hints.attributes()
            };
            if wanted != hints.stored {
                store_block_attributes(store, &path, declared.structural_shape.as_deref(), wanted)?;
                hints.stored = wanted;
            }

            blocks.insert(
                name.clone(),
                BlockArrays {
                    columns,
                    masks,
                    index,
                    total_rows,
                    updates,
                    rows_per_chunk,
                    hints,
                },
            );
        }

        let box_prefix = join_path(TRAJECTORY_GROUP, BOX_GROUP);
        let cell = if array_exists(store, &join_path(&box_prefix, VECTORS_ARRAY))? {
            let optional = |name: &str| -> Result<Option<GrowthArray>, MolRsError> {
                let path = join_path(&box_prefix, name);
                if array_exists(store, &path)? {
                    Ok(Some(GrowthArray::open(store, &path)?))
                } else {
                    Ok(None)
                }
            };
            let mut arrays = BoxArrays {
                step_index: optional(STEP_INDEX_ARRAY)?,
                vectors: GrowthArray::open(store, &join_path(&box_prefix, VECTORS_ARRAY))?,
                origin: optional(ORIGIN_ARRAY)?,
                boundary: optional(BOUNDARY_ARRAY)?,
            };
            let committed = match &arrays.step_index {
                Some(_) => read_whole::<u64, _>(store, &join_path(&box_prefix, STEP_INDEX_ARRAY))?
                    .iter()
                    .take_while(|&&ordinal| ordinal < nstep)
                    .count() as u64,
                // The omitted index is one update at ordinal 0.
                None => u64::from(arrays.vectors.rows > 0 && nstep > 0),
            };
            arrays.truncate_to(committed)?;
            Some(CellStorage::Arrays(Box::new(arrays)))
        } else if group_exists(store, &box_prefix)? {
            let group = Group::open(store.clone(), &box_prefix)?;
            match cell_from_attributes(group.attributes())? {
                Some(cell) if nstep > 0 => Some(CellStorage::Attributes(Box::new(cell))),
                _ => None,
            }
        } else {
            None
        };

        Ok(Self {
            step,
            time,
            meta,
            blocks,
            cell,
            nstep,
        })
    }

    /// The trajectory group attributes a commit publishes: the pin, the
    /// marker, and the regular series.
    fn store_marker(
        &self,
        store: &ReadableWritableListableStorage,
        nstep: u64,
    ) -> Result<(), MolRsError> {
        let mut group = Group::open(store.clone(), TRAJECTORY_GROUP)?;
        let attributes = group.attributes_mut();
        attributes.insert(NSTEP_ATTRIBUTE.to_string(), nstep.into());
        match self.step.attribute() {
            Some(value) => attributes.insert(STEP_PROGRESSION_ATTRIBUTE.to_string(), value),
            None => attributes.remove(STEP_PROGRESSION_ATTRIBUTE),
        };
        match self.time.as_ref().and_then(Track::attribute) {
            Some(value) => attributes.insert(TIME_PROGRESSION_ATTRIBUTE.to_string(), value),
            None => attributes.remove(TIME_PROGRESSION_ATTRIBUTE),
        };
        group.store_metadata()?;
        Ok(())
    }
}

/// Check one reopened column array against what the schema pinned for it.
fn validate_column(
    opened: &GrowthArray,
    path: &str,
    schema: &ColumnSchema,
) -> Result<(), MolRsError> {
    let found = dtype_tag(dtype_of_stored(path, opened.array.data_type())?).to_string();
    if found != schema.dtype {
        return Err(MolRsError::zarr(format!(
            "sequence schema mismatch at {path}: dtype expected {}, found {found}",
            schema.dtype
        )));
    }
    let trailing: Vec<u64> = opened.array.shape().iter().skip(1).copied().collect();
    if trailing != schema.trailing {
        return Err(MolRsError::zarr(format!(
            "sequence schema mismatch at {path}: trailing shape expected {:?}, found {trailing:?}",
            schema.trailing
        )));
    }
    Ok(())
}

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

    /// Buffer `frame` at the next step number, with no time.
    ///
    /// Step numbers start at 0 and rise by one — or, after
    /// [`open`](Self::open) has reattached to an existing sequence, continue
    /// from the last step already landed. A run that carries real MD step
    /// numbers or times uses [`append_at`](Self::append_at) instead.
    ///
    /// # Errors
    ///
    /// [`append_at`](Self::append_at)'s.
    pub fn append(&mut self, frame: &Frame) -> Result<(), MolRsError> {
        self.append_at(frame, self.last_step.map_or(0, |step| step + 1), None)
    }

    /// Buffer `frame` at the next step number, carrying a physical time.
    ///
    /// # Errors
    ///
    /// [`append_at`](Self::append_at)'s.
    pub fn append_timed(&mut self, frame: &Frame, time: F) -> Result<(), MolRsError> {
        self.append_at(frame, self.last_step.map_or(0, |step| step + 1), Some(time))
    }

    /// Buffer `frame` at an explicit step number and optional time.
    ///
    /// `step` is the run's own dimensionless step counter. It must be strictly
    /// greater than the previous one. `time` is the frame's physical time in
    /// **femtoseconds (fs)**, stored verbatim, and all-or-nothing across a
    /// run: the first append decides whether the `time` array exists.
    ///
    /// What gets buffered is only what *changed*. A block whose bytes are
    /// identical to its previous update earns no new update; a block the frame
    /// **omits** earns none either and carries forward; a block presented with
    /// zero rows earns a zero-row update (present and empty). The very first
    /// append creates every array the schema declares, freezing the extents.
    /// When the buffer reaches the landing cadence it is committed here.
    ///
    /// # Errors
    ///
    /// Every case is a [`MolRsError::Zarr`] naming what disagreed:
    ///
    /// - `step` is not strictly greater than the previous step;
    /// - `time` is supplied on a run started without times, or omitted on a run
    ///   started with them;
    /// - the frame carries a block, a column, or a `meta` key the pinned schema
    ///   does not declare;
    /// - a column or `meta` key is declared with one dtype and this frame
    ///   carries another, or a column's trailing shape disagrees;
    /// - a block is present but omits one of its declared columns: sparsity is
    ///   per block, not per column;
    /// - a block with a declared structural shape carries another row count;
    /// - the frame omits a declared `meta` key for which no fill was declared.
    ///
    /// Also any storage error raised by an automatic landing.
    pub fn append_at(
        &mut self,
        frame: &Frame,
        step: i64,
        time: Option<F>,
    ) -> Result<(), MolRsError> {
        if let Some(previous) = self.last_step
            && step <= previous
        {
            return Err(MolRsError::zarr(format!(
                "step numbers must increase strictly: {step} follows {previous}"
            )));
        }
        match self.uses_time {
            None => self.uses_time = Some(time.is_some()),
            Some(expected) if expected != time.is_some() => {
                return Err(MolRsError::zarr(format!(
                    "the {TIME_ARRAY:?} array is all-or-nothing across a run: this sequence was \
                     started {}, and this frame supplies {}",
                    if expected {
                        "with times"
                    } else {
                        "without times"
                    },
                    if time.is_some() { "one" } else { "none" }
                )));
            }
            Some(_) => {}
        }

        self.schema.validate_frame(frame)?;
        let meta = self.schema.resolve_meta(frame)?;
        self.ensure_arrays()?;

        // What would be stored, not what was presented: a precision column
        // is rounded first, so a change below half its quantum is no change
        // and earns no update. An omitted block carries forward.
        let mut presented: BTreeMap<String, Block> = BTreeMap::new();
        for (name, declared) in &self.schema.blocks {
            if let Some(block) = frame.get(name) {
                presented.insert(
                    name.clone(),
                    stored_presentation(declared, block)?.into_owned(),
                );
            }
        }
        // The frame as a reader will resolve it, checked before any state
        // moves: a refused frame leaves the writer as it was.
        self.check_resolved(&presented, self.committed + self.pending.len() as u64)?;

        let mut blocks = BTreeMap::new();
        for (name, block) in presented {
            let changed = self
                .landed_blocks
                .get(&name)
                .is_none_or(|landed| !same_block(landed, &block));
            if changed {
                blocks.insert(name.clone(), block.clone());
                self.landed_blocks.insert(name, block);
            }
        }

        let cell = match &frame.simbox {
            Some(simbox) => {
                let changed = self
                    .landed_cell
                    .as_ref()
                    .is_none_or(|landed| !same_simbox(landed, simbox));
                if changed {
                    self.landed_cell = Some(simbox.clone());
                    Some(simbox.clone())
                } else {
                    None
                }
            }
            // The cell carries forward like any other section.
            None => None,
        };

        self.last_step = Some(step);
        self.pending.push(PendingFrame {
            step,
            time,
            blocks,
            cell,
            meta,
        });
        if self.pending.len() as u64 >= self.auto_flush_every {
            self.commit(false)?;
        }
        Ok(())
    }

    /// Check the frame at `ordinal` as a reader resolves it — each declared
    /// block presented now, else carried forward from its latest update —
    /// against the rules that span blocks.
    fn check_resolved(
        &self,
        presented: &BTreeMap<String, Block>,
        ordinal: u64,
    ) -> Result<(), MolRsError> {
        check_resolved_frame(
            &self.schema,
            |name| presented.get(name).or_else(|| self.landed_blocks.get(name)),
            ordinal,
        )
    }

    /// Create every array on the first append, freezing the extents and the
    /// landing cadence.
    fn ensure_arrays(&mut self) -> Result<(), MolRsError> {
        if self.arrays.is_none() {
            self.arrays = Some(SequenceArrays::create(
                &self.store,
                &self.schema,
                self.knobs,
                self.compression,
                self.uses_time == Some(true),
            )?);
            self.auto_flush_every = self.derive_flush_every();
        }
        Ok(())
    }

    /// Commit every buffered frame, durably.
    ///
    /// Every frame appended before this call lands, whether or not it fills a
    /// chunk. `step` is extended **last** and is the commit marker: a reader
    /// sees `nstep` frames only once this call has written it. Unless
    /// [`with_durable(false)`](Self::with_durable) was set, the touched files
    /// are synced before the marker moves and again after it, so a returned
    /// `Ok` means the frames survive a power loss.
    ///
    /// A flush that lands a partially filled trailing inner chunk re-encodes
    /// that chunk and leaves its overwritten copy in the shard as dead bytes —
    /// at most one chunk per column per flush. A flush with nothing buffered
    /// is `Ok(())` and writes nothing.
    ///
    /// # Errors
    ///
    /// A [`MolRsError::Zarr`] wrapping whatever the store or the codec raised.
    /// A flush that fails partway leaves arrays longer than `step`; such a
    /// store reads back as the previous commit, and a reopened writer rolls
    /// the excess back.
    pub fn flush(&mut self) -> Result<(), MolRsError> {
        self.commit(self.durable)
    }

    /// Land every buffered frame; `durable` syncs the touched files around
    /// the commit marker.
    fn commit(&mut self, durable: bool) -> Result<(), MolRsError> {
        if self.pending.is_empty() {
            return Ok(());
        }
        let options = partial_encoding_options();
        let Self {
            store,
            schema,
            knobs,
            arrays,
            committed,
            pending,
            ..
        } = self;
        let Some(arrays) = arrays.as_mut() else {
            return Ok(());
        };
        let base = *committed;

        // 1. Block sections: columns first, then — for a block that is no
        //    longer regular — its own index, then its hints.
        for (name, declared) in &schema.blocks {
            let Some(section) = arrays.blocks.get_mut(name) else {
                continue;
            };
            let updates: Vec<(u64, &Block)> = pending
                .iter()
                .enumerate()
                .filter_map(|(index, frame)| {
                    frame
                        .blocks
                        .get(name)
                        .map(|update| (base + index as u64, update))
                })
                .collect();
            if updates.is_empty() {
                continue;
            }

            let landed: Vec<&Block> = updates.iter().map(|(_, block)| *block).collect();
            let added: u64 = landed
                .iter()
                .map(|block| block.n_rows().unwrap_or(0) as u64)
                .sum();
            if added > 0 {
                for (column, pinned) in &declared.columns {
                    let Some(array) = section.columns.get_mut(column) else {
                        continue;
                    };
                    let values: Vec<&Column> = landed
                        .iter()
                        .map(|block| {
                            block.get(column).ok_or_else(|| {
                                MolRsError::zarr(format!(
                                    "block {name:?} lost its declared column {column:?} between \
                                     append and flush"
                                ))
                            })
                        })
                        .collect::<Result<_, MolRsError>>()?;
                    let subset = array.reserve(added)?;
                    array.store_columns(
                        &subset,
                        dtype_from_tag(&pinned.dtype)?,
                        &values,
                        &options,
                    )?;
                    array.commit_shape()?;
                }
                section.land_masks(&landed, added, &options)?;
            }

            let mut hints = section.hints;
            for (k, (ordinal, block)) in updates.iter().enumerate() {
                hints.observe(
                    section.updates + k as u64,
                    *ordinal,
                    block.n_rows().unwrap_or(0) as u64,
                );
            }
            let regular = hints.is_regular() && !declared.columns.is_empty();

            let mut running = section.total_rows;
            let mut offsets = Vec::with_capacity(updates.len());
            for (_, update) in &updates {
                running += update.n_rows().unwrap_or(0) as u64;
                offsets.push(running);
            }
            let ordinals: Vec<u64> = updates.iter().map(|(ordinal, _)| *ordinal).collect();

            if !regular {
                if section.index.is_none() {
                    let history_rows = match section.hints.uniform {
                        Uniform::Rows(rows) => rows,
                        _ => 0,
                    };
                    section.index = Some(IndexArrays::materialize(
                        store,
                        &join_path(TRAJECTORY_GROUP, name),
                        *knobs,
                        section.updates,
                        history_rows,
                        &options,
                    )?);
                }
                let index = section.index.as_mut().expect("materialized above");
                let count = offsets.len() as u64;
                land_values!(index.offset, count, offsets, &options);
                let count = ordinals.len() as u64;
                land_values!(index.step_index, count, ordinals, &options);
            }

            section.hints = hints;
            section.updates += updates.len() as u64;
            section.total_rows = running;

            // Once the index arrays exist they are authoritative: the hints
            // are withdrawn for good.
            let wanted = if section.index.is_some() {
                (None, false)
            } else {
                section.hints.attributes()
            };
            if wanted != section.hints.stored {
                store_block_attributes(
                    store,
                    &join_path(TRAJECTORY_GROUP, name),
                    declared.structural_shape.as_deref(),
                    wanted,
                )?;
                section.hints.stored = wanted;
            }
        }

        // 2. The cell: attributes while it is one fixed cell from ordinal 0,
        //    arrays from the first change on.
        let cells: Vec<(u64, &SimBox)> = pending
            .iter()
            .enumerate()
            .filter_map(|(index, frame)| {
                frame.cell.as_ref().map(|cell| (base + index as u64, cell))
            })
            .collect();
        if let Some((first_ordinal, first)) = cells.first().copied() {
            match arrays.cell.take() {
                None if cells.len() == 1 && first_ordinal == 0 => {
                    store_box_attributes(
                        store,
                        box_attributes(first.is_cell_defined(), Some(first)),
                    )?;
                    arrays.cell = Some(CellStorage::Attributes(Box::new(first.clone())));
                }
                None => {
                    store_box_attributes(store, box_attributes(first.is_cell_defined(), None))?;
                    let mut section = BoxArrays::create(store, *knobs)?;
                    section.land(store, *knobs, &cells, &options)?;
                    arrays.cell = Some(CellStorage::Arrays(Box::new(section)));
                }
                Some(CellStorage::Attributes(fixed)) => {
                    let mut section = BoxArrays::create(store, *knobs)?;
                    let mut all: Vec<(u64, &SimBox)> = Vec::with_capacity(cells.len() + 1);
                    all.push((0, fixed.as_ref()));
                    all.extend(cells.iter().copied());
                    section.land(store, *knobs, &all, &options)?;
                    // The arrays now carry the cell; the attributes keep only
                    // `cell_defined`.
                    store_box_attributes(store, box_attributes(fixed.is_cell_defined(), None))?;
                    arrays.cell = Some(CellStorage::Arrays(Box::new(section)));
                }
                Some(CellStorage::Arrays(mut section)) => {
                    section.land(store, *knobs, &cells, &options)?;
                    arrays.cell = Some(CellStorage::Arrays(section));
                }
            }
        }

        // 3. Per-step metadata.
        for (key, declared) in &schema.meta {
            let Some(array) = arrays.meta.get_mut(key) else {
                continue;
            };
            let values: Vec<&MetaValue> = pending
                .iter()
                .map(|frame| {
                    frame.meta.get(key).ok_or_else(|| {
                        MolRsError::zarr(format!(
                            "meta key {key:?} was resolved at append but is missing at flush"
                        ))
                    })
                })
                .collect::<Result<_, MolRsError>>()?;
            let subset = array.reserve(values.len() as u64)?;
            array.store_meta(&subset, key, &declared.dtype, &values, &options)?;
            array.commit_shape()?;
        }

        // 4. Time and step: a progression attribute while they are regular,
        //    an array once they are not.
        if let Some(track) = arrays.time.as_mut() {
            let times: Vec<F> = pending
                .iter()
                .map(|frame| frame.time.unwrap_or(0.0))
                .collect();
            track.land(
                store,
                &join_path(TRAJECTORY_GROUP, TIME_ARRAY),
                *knobs,
                times,
                &options,
            )?;
        }
        let steps: Vec<i64> = pending.iter().map(|frame| frame.step).collect();
        arrays.step.land(
            store,
            &join_path(TRAJECTORY_GROUP, STEP_ARRAY),
            *knobs,
            steps,
            &options,
        )?;

        // 5. Everything above is on disk (and, when durable, synced) before
        //    the commit marker moves.
        #[cfg(feature = "filesystem")]
        if durable && let Some(positional) = &self.positional {
            positional.sync_dirty()?;
        }
        #[cfg(not(feature = "filesystem"))]
        let _ = durable;

        // 6. The commit marker, last: the trajectory group's `nstep`
        //    attribute, one atomic metadata replace.
        let nstep = base + pending.len() as u64;
        arrays.store_marker(store, nstep)?;
        arrays.nstep = nstep;

        #[cfg(feature = "filesystem")]
        if durable && let Some(positional) = &self.positional {
            positional.sync_dirty()?;
        }

        *committed += pending.len() as u64;
        pending.clear();
        Ok(())
    }

    /// Commit whatever is still buffered and consume the writer.
    ///
    /// Consuming rather than dropping is the point: this is the only place an
    /// IO error at the end of a run can still be returned to the caller. No
    /// shard is rewritten on close — the store's bytes are exactly what the
    /// landings wrote.
    ///
    /// # Errors
    ///
    /// [`flush`](Self::flush)'s. The writer is consumed either way; whatever
    /// had already been committed stays readable.
    pub fn close(mut self) -> Result<(), MolRsError> {
        self.commit(self.durable)
    }
}

/// The `box/` section's `cell_defined` attribute; absent means a defined cell.
fn cell_defined_of<S>(store: &Arc<S>) -> Result<bool, MolRsError>
where
    S: ?Sized + ReadableStorageTraits + 'static,
{
    let prefix = join_path(TRAJECTORY_GROUP, BOX_GROUP);
    let Ok(group) = Group::open(store.clone(), &prefix) else {
        return Ok(true);
    };
    Ok(group
        .attributes()
        .get(CELL_DEFINED_ATTRIBUTE)
        .and_then(serde_json::Value::as_bool)
        .unwrap_or(true))
}

// ---------------------------------------------------------------------------
// Reader
// ---------------------------------------------------------------------------

/// The index of one block section, as [`MrecReader::open`] cached it.
enum BlockIndex {
    /// The general case: the section's own `step_index` and CSR `offset`.
    Sparse {
        /// Frame ordinals at which the section was updated, ascending.
        step_index: Vec<u64>,
        /// CSR row pointer, one longer than `step_index`.
        offset: Vec<u64>,
    },
    /// The hinted case (`dense_updates` + `uniform_rows`): update `j` sits at
    /// ordinal `j` and owns rows `j*rows..(j+1)*rows`, so nothing was read.
    Regular {
        /// Updates committed.
        updates: u64,
        /// Rows every update carries.
        rows: u64,
    },
}

impl BlockIndex {
    /// `(update, first row, rows)` of the section at frame `index`, or `None`
    /// when the section has not appeared yet.
    fn resolve(&self, index: u64) -> Result<Option<(u64, u64, u64)>, MolRsError> {
        match self {
            Self::Sparse { step_index, offset } => {
                let Some(update) = latest_update(step_index, index) else {
                    return Ok(None);
                };
                let (Some(&start), Some(&end)) = (offset.get(update), offset.get(update + 1))
                else {
                    return Err(MolRsError::zarr(format!(
                        "block section has {} offsets for update {update}",
                        offset.len()
                    )));
                };
                // A hostile or corrupt store can carry a non-monotonic
                // `offset` array; `end - start` would then wrap to a
                // near-`u64::MAX` row count and drive an unbounded allocation.
                let count = end.checked_sub(start).ok_or_else(|| {
                    MolRsError::zarr(format!(
                        "block section update {update} has non-monotonic offsets ({start} > {end})"
                    ))
                })?;
                Ok(Some((update as u64, start, count)))
            }
            Self::Regular { updates, rows } => {
                if *updates == 0 {
                    return Ok(None);
                }
                let update = index.min(updates - 1);
                Ok(Some((update, update * rows, *rows)))
            }
        }
    }
}

/// The index of the `box/` section.
struct BoxIndex {
    step_index: Vec<u64>,
    cell_defined: bool,
}

/// The `box/` section's cells: the fixed cell from the group attributes, or
/// the per-update arrays, with a per-update cache.
struct BoxReader {
    fixed: Option<SimBox>,
    vectors: Option<Array<dyn ReadableListableStorageTraits>>,
    origin: Option<Array<dyn ReadableListableStorageTraits>>,
    boundary: Option<Array<dyn ReadableListableStorageTraits>>,
    cells: BTreeMap<u64, SimBox>,
}

impl BoxReader {
    fn open(store: &ReadableListableStorage) -> Result<Self, MolRsError> {
        let prefix = join_path(TRAJECTORY_GROUP, BOX_GROUP);
        let optional =
            |name: &str| -> Result<Option<Array<dyn ReadableListableStorageTraits>>, MolRsError> {
                let path = join_path(&prefix, name);
                if array_exists(store, &path)? {
                    Ok(Some(Array::open(store.clone(), &path)?))
                } else {
                    Ok(None)
                }
            };
        let vectors = optional(VECTORS_ARRAY)?;
        let fixed = if vectors.is_none() {
            cell_from_attributes(Group::open(store.clone(), &prefix)?.attributes())?
        } else {
            None
        };
        Ok(Self {
            fixed,
            vectors,
            origin: optional(ORIGIN_ARRAY)?,
            boundary: optional(BOUNDARY_ARRAY)?,
            cells: BTreeMap::new(),
        })
    }

    /// Cell update `index`, decoded once.
    fn cell_at(&mut self, index: u64, cell_defined: bool) -> Result<SimBox, MolRsError> {
        if let Some(fixed) = &self.fixed {
            return Ok(fixed.clone());
        }
        if let Some(cell) = self.cells.get(&index) {
            return Ok(cell.clone());
        }
        let vectors = self
            .vectors
            .as_ref()
            .ok_or_else(|| MolRsError::zarr("box section has neither a fixed cell nor arrays"))?;
        // An undefined cell's `vectors` row is never read: its numbers mean
        // nothing, and `Cell::Undefined` carries the identity instead.
        let cell: Vec<F> = if cell_defined {
            vectors.retrieve_array_subset(&rows_subset(index, 1, &[3, 3])?)?
        } else {
            vec![1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0]
        };
        let origin: Vec<F> = match &self.origin {
            Some(array) => array.retrieve_array_subset(&rows_subset(index, 1, &[3])?)?,
            None => vec![0.0, 0.0, 0.0],
        };
        let boundary: Option<Vec<bool>> = match &self.boundary {
            Some(array) => Some(array.retrieve_array_subset(&rows_subset(index, 1, &[3])?)?),
            None => None,
        };
        let flags = boundary.as_ref().map_or(3, Vec::len);
        if cell.len() != 9 || origin.len() != 3 || flags != 3 {
            return Err(MolRsError::zarr(format!(
                "box update {index} is malformed: {} cell values, {} origin values, {flags} \
                 boundary flags",
                cell.len(),
                origin.len(),
            )));
        }
        let pbc = resolve_boundary(
            boundary.map(|flags| [flags[0], flags[1], flags[2]]),
            cell_defined,
            &format!("box update {index}"),
        )?;
        let simbox = SimBox::new(
            if cell_defined {
                Cell::Matrix(std::array::from_fn(|r| {
                    std::array::from_fn(|c| cell[3 * r + c])
                }))
            } else {
                Cell::Undefined
            },
            [origin[0], origin[1], origin[2]],
            pbc,
        )
        .map_err(|e| MolRsError::zarr(format!("box update {index} is not a valid cell: {e:?}")))?;
        if self.cells.len() >= 64 {
            self.cells.clear();
        }
        self.cells.insert(index, simbox.clone());
        Ok(simbox)
    }
}

/// One open column array plus its most recently decoded inner chunks.
struct ColumnReader {
    array: Array<dyn ReadableListableStorageTraits>,
    /// Rows one inner chunk holds.
    chunk_rows: u64,
    /// Decoded whole chunks, most recently used first.
    chunks: VecDeque<(u64, Column)>,
}

impl ColumnReader {
    fn open(store: &ReadableListableStorage, path: &str) -> Result<Self, MolRsError> {
        let array = Array::open(store.clone(), path)?;
        let chunk_rows = inner_rows_of(&array);
        Ok(Self {
            array,
            chunk_rows,
            chunks: VecDeque::with_capacity(CHUNK_CACHE_ENTRIES + 1),
        })
    }

    /// Rows `start..start+rows`, decoding at most the chunks they touch.
    ///
    /// A range inside one chunk comes out of the chunk cache (decoding the
    /// chunk whole on a miss, so the next frame in it is a slice); a range
    /// straddling two chunks — a ragged run — is read directly.
    fn rows(&mut self, start: u64, rows: u64, trailing: &[u64]) -> Result<Column, MolRsError> {
        let first = start / self.chunk_rows;
        let last = (start + rows - 1) / self.chunk_rows;
        if first != last {
            return read_column_array(&self.array, &rows_subset(start, rows, trailing)?);
        }
        let within = (start - first * self.chunk_rows) as usize;
        let chunk = self.chunk(first, trailing)?;
        Ok(column_rows(chunk, within, within + rows as usize))
    }

    /// Inner chunk `index`, decoded whole and kept.
    fn chunk(&mut self, index: u64, trailing: &[u64]) -> Result<&Column, MolRsError> {
        if let Some(position) = self.chunks.iter().position(|(i, _)| *i == index) {
            if position != 0 {
                let hit = self.chunks.remove(position).expect("position is in range");
                self.chunks.push_front(hit);
            }
        } else {
            let total = self.array.shape().first().copied().unwrap_or(0);
            let chunk_start = index * self.chunk_rows;
            let chunk_len = self.chunk_rows.min(total.saturating_sub(chunk_start));
            let column =
                read_column_array(&self.array, &rows_subset(chunk_start, chunk_len, trailing)?)?;
            self.chunks.push_front((index, column));
            self.chunks.truncate(CHUNK_CACHE_ENTRIES);
        }
        Ok(&self
            .chunks
            .front()
            .expect("just inserted or moved to front")
            .1)
    }
}

/// The lazily opened arrays and caches behind a [`MrecReader`].
#[derive(Default)]
struct ReadState {
    columns: BTreeMap<String, ColumnReader>,
    /// Mask readers by array path, `None` for a nullable column the store
    /// carries no mask array for — caching the absence, so a column that was
    /// never holed costs one existence check for the whole run.
    masks: BTreeMap<String, Option<ColumnReader>>,
    metas: IndexMap<String, Array<dyn ReadableListableStorageTraits>>,
    boxes: Option<BoxReader>,
}

/// The lazy store cursor over a frame sequence: index-only open, one frame per
/// read.
///
/// One of the three access forms of a single object — `Trajectory` is the eager
/// in-memory carrier that materializes every frame, [`MrecWriter`] is
/// the streaming producer, and this reads what that writer produced without
/// materializing anything it was not asked for. [`to_trajectory`](Self::to_trajectory)
/// is the named lazy → eager conversion.
///
/// Every read door takes `&self`: the cursor holds no cursor state, only
/// caches (open array handles, the last decoded inner chunk per column, the
/// decoded cells) behind a lock, so a sequence can be shared and read from
/// several places.
///
/// **Error vocabulary.** Opening a record from a [`Source`](crate::io::Source)
/// ([`open`](Self::open)) is a frame-IO door and fails with
/// [`std::io::Error`] like every other. Every read of an opened store yields
/// [`MolRsError`], as the [`MolRsError::Zarr`] variant naming the section or
/// array path at fault. The three [`TrajectoryReader`] methods keep
/// `std::io::Result`: a `MolRsError` becomes an
/// [`InvalidData`](std::io::ErrorKind::InvalidData) error carrying it — the
/// trait is the shape every trajectory backend shares and must not be captured
/// by one backend's error type.
///
/// # One object, three access forms
///
/// A frame sequence is a single thing. Three Rust names reach it, and they are
/// named after the **access form**, never after the content:
///
/// - [`Trajectory`] — the **eager in-memory carrier**. Every frame is
///   materialized in a `Vec<Frame>`; it does not know a store exists.
/// - [`MrecReader`] — the **lazy store cursor**. It opens index-only (each
///   section's `step_index` and `offset`, plus the schema) and reads **one**
///   frame per call. [`MrecReader::to_trajectory`] is the named lazy → eager
///   conversion, not a second representation.
/// - [`MrecWriter`] — the sequence's **streaming producer**. One frame
///   per [`append`](MrecWriter::append); complete inner chunks land
///   on their own, [`flush`](MrecWriter::flush) commits whatever is
///   still buffered.
///
/// The `trajectory/` group on disk is the sequence's serialized form; none of
/// the three names is a second name for it.
///
/// # Error vocabulary
///
/// Every [`MrecReader`] and [`MrecWriter`] door on a store yields
/// [`MolRsError`]; [`MrecReader::open`] of a source is a frame-IO door and
/// yields [`std::io::Error`]. The three [`TrajectoryReader`] methods keep
/// `io::Result` and convert `MolRsError` in at the boundary (an
/// `InvalidData` error carrying it): the trait is the shape every backend
/// shares and must not be captured by one backend's error type.
///
/// # The layout
///
/// ```text
/// /                              root group
/// meta/                          the record's identity document (attributes)
/// trajectory/                    group; attrs: sequence_schema (the pin),
///                                nstep (the commit marker, written last),
///                                step_progression / time_progression
///                                ({start, stride} while the series is regular)
///   step     i64  [nstep]        only once the numbering is not a progression
///   time     f64  [nstep]        only when supplied and not a progression
///   meta/<key>    typed [nstep] (or [nstep][3|6|9]); attr meta_dtype
///   box/                         attrs: cell_defined; vectors / origin /
///                                boundary while the cell is fixed from ordinal 0
///     step_index u64  [n_updates]     the arrays, once the cell changes
///     vectors    f64  [n_updates][3][3]
///     origin     f64  [n_updates][3]
///     boundary   bool [n_updates][3]
///   <block>/                     attrs structural_shape, uniform_rows, dense_updates
///     step_index u64  [n_updates]     absent while the block is regular
///     offset     u64  [n_updates+1]   CSR row pointer, offset[0] = 0
///     <column>        [total_rows][...trailing]
///     _validity/                 group; only for columns declared nullable
///       <column> bool [total_rows]    one flag per row, no trailing axes
/// ```
///
/// The layout is built so the common run — a fixed number of atoms moving
/// every frame, a topology written once, a fixed cell, frames dumped every
/// `k` steps — costs one array per column and nothing else. A **regular**
/// block (a fixed, non-zero row count at ordinals `0, 1, 2, …`) writes no
/// index: its group hints plus any column's length resolve every frame. The
/// first irregular update materializes `offset` and `step_index`, backfilled
/// with the regular history, and withdraws the hints for good.
///
/// `offset` is a **CSR** (compressed sparse row) row pointer, the standard way
/// to store a ragged sequence of row groups in one flat array: entry `j` holds
/// the index of the first row of update `j`, so update `j` owns the half-open
/// row range `offset[j]..offset[j+1]`, and the array is therefore one entry
/// longer than `step_index`.
///
/// # Three states of a block at a frame
///
/// Resolving block `B` at frame `i`: binary-search `B/step_index` for the
/// largest entry `<= i`, giving update `j`; the frame's rows are
/// `offset[j]..offset[j+1]`. The three cases are the layout's whole
/// presence vocabulary:
///
/// - **No entry `<= i`** — the block does not exist at frame `i` (absence).
/// - **An update of zero rows** — the block is present and empty from that
///   frame until its next update. It reads back as an empty [`Block`] with the
///   declared columns.
/// - **A frame that omits a declared block** earns *no* update: the block
///   carries forward from its latest update. That is what makes a constant
///   topology cost one `step_index` entry whether the producer re-presents it
///   every frame (bit-identical content earns no update either) or presents
///   it once.
///
/// Once a block has appeared it never becomes absent again; there is no
/// tombstone. The `box/` section has no `offset` and follows the same
/// carry-forward rule: once a run writes a cell, every later frame resolves to
/// the most recent one.
///
/// [`Trajectory`]: molrs::core::Trajectory
/// [`TrajectoryReader`]: crate::io::reader::TrajectoryReader
pub struct MrecReader {
    /// The read-only view of the store. A read door keeps no write capability,
    /// which is also what lets a packed `.mrec.zip` be opened through it.
    store: ReadableListableStorage,
    schema: SequenceSchema,
    /// Step numbers of the committed frames; its length is `nstep`.
    steps: Vec<i64>,
    times: Option<Vec<F>>,
    blocks: IndexMap<String, BlockIndex>,
    cell: Option<BoxIndex>,
    state: Mutex<ReadState>,
}

impl MrecReader {
    /// Open a sequence for reading, taking only its indices.
    ///
    /// Index-only: the schema attributes plus each section's `step_index` and
    /// `offset` — or, for a section whose hints say every update is regular,
    /// nothing but its array metadata. No frame data is touched until a read
    /// asks for it. The commit marker bounds everything: index entries at or
    /// past `len(step)` (a crash's leftovers) are ignored.
    ///
    /// Any readable, listable store opens: a directory store, an in-memory one,
    /// or the read-only adapter a packed `.mrec.zip` is read through. A conforming
    /// store needs no writer pin — the schema is derived from the store itself
    /// when the attribute is absent. A store created and closed without a
    /// single append opens as a legal, empty sequence.
    ///
    /// # Errors
    ///
    /// Every case is a [`MolRsError::Zarr`] naming the path it failed on:
    ///
    /// - a schema pin is present but is not a schema this build can
    ///   deserialize;
    /// - no pin is present and the derivation from the store fails.
    ///
    /// Also any storage error raised while reading the index arrays.
    pub fn from_storage<S>(store: Arc<S>) -> Result<Self, MolRsError>
    where
        S: ?Sized + ReadableStorageTraits + ListableStorageTraits + 'static,
    {
        // `StorageHandle` is how an unsized store becomes an owned, sized one:
        // `Arc<dyn ReadableWritableListableStorageTraits>` does not coerce to
        // `Arc<dyn ReadableListableStorageTraits>`, so the read-only view is
        // taken here.
        let store: ReadableListableStorage = Arc::new(StorageHandle::new(store));
        let schema = match pinned_schema(&store)? {
            Some(pinned) => pinned,
            None => schema_from_storage(&store)?,
        };
        let attributes = trajectory_attributes(&store)?;
        let nstep = committed_frames(&store, &attributes)?;
        // Created and closed without an append: a legal, empty sequence.
        let steps = read_track::<i64, _>(
            &store,
            &attributes.all,
            STEP_PROGRESSION_ATTRIBUTE,
            &join_path(TRAJECTORY_GROUP, STEP_ARRAY),
            nstep,
        )?
        .unwrap_or_default();
        if steps.len() as u64 != nstep {
            return Err(MolRsError::zarr(format!(
                "trajectory claims {nstep} committed frames but carries {} step numbers",
                steps.len()
            )));
        }
        let times = read_track::<F, _>(
            &store,
            &attributes.all,
            TIME_PROGRESSION_ATTRIBUTE,
            &join_path(TRAJECTORY_GROUP, TIME_ARRAY),
            nstep,
        )?;

        let mut blocks = IndexMap::new();
        for (name, declared) in &schema.blocks {
            let path = join_path(TRAJECTORY_GROUP, name);
            if !group_exists(&store, &path)? {
                // Declared, never appended: the section does not exist yet.
                continue;
            }
            let index_path = join_path(&path, STEP_INDEX_ARRAY);
            let index = if array_exists(&store, &index_path)? {
                let mut step_index = read_whole::<u64, _>(&store, &index_path)?;
                let updates = step_index
                    .iter()
                    .take_while(|&&ordinal| ordinal < nstep)
                    .count();
                step_index.truncate(updates);
                let mut offset = read_whole::<u64, _>(&store, &join_path(&path, OFFSET_ARRAY))?;
                offset.truncate(if updates == 0 { 0 } else { updates + 1 });
                BlockIndex::Sparse { step_index, offset }
            } else {
                match elision_markers(&store, &path, !declared.columns.is_empty())? {
                    Some(rows) => {
                        // Regular: the columns' own length says how many
                        // updates landed; the marker bounds them.
                        let column =
                            declared.columns.keys().next().expect(
                                "elision_markers refuses markers on a block with no columns",
                            );
                        let landed = Array::open(store.clone(), &join_path(&path, column))?
                            .shape()
                            .first()
                            .copied()
                            .unwrap_or(0);
                        BlockIndex::Regular {
                            updates: (landed / rows).min(nstep),
                            rows,
                        }
                    }
                    _ => continue,
                }
            };
            let empty = match &index {
                BlockIndex::Regular { updates, .. } => *updates == 0,
                BlockIndex::Sparse { step_index, .. } => step_index.is_empty(),
            };
            if !empty {
                blocks.insert(name.clone(), index);
            }
        }

        schema.check_alignments()?;
        check_aligned_indexes(&schema, &blocks, nstep)?;
        check_aligned_not_in_system(&store, &schema)?;

        let box_prefix = join_path(TRAJECTORY_GROUP, BOX_GROUP);
        let vectors_path = join_path(&box_prefix, VECTORS_ARRAY);
        let cell = if array_exists(&store, &vectors_path)? {
            let step_path = join_path(&box_prefix, STEP_INDEX_ARRAY);
            let mut step_index = if array_exists(&store, &step_path)? {
                read_whole::<u64, _>(&store, &step_path)?
            } else {
                let n = Array::open(store.clone(), &vectors_path)?
                    .shape()
                    .first()
                    .copied()
                    .unwrap_or(0);
                // The omitted index is one update at ordinal 0.
                if n == 0 { Vec::new() } else { vec![0] }
            };
            let committed = step_index
                .iter()
                .take_while(|&&ordinal| ordinal < nstep)
                .count();
            step_index.truncate(committed);
            if step_index.is_empty() {
                None
            } else {
                Some(BoxIndex {
                    step_index,
                    cell_defined: cell_defined_of(&store)?,
                })
            }
        } else if group_exists(&store, &box_prefix)? {
            let group = Group::open(store.clone(), &box_prefix)?;
            match cell_from_attributes(group.attributes())? {
                Some(cell) if nstep > 0 => Some(BoxIndex {
                    step_index: vec![0],
                    cell_defined: cell.is_cell_defined(),
                }),
                _ => None,
            }
        } else {
            None
        };

        Ok(Self {
            store,
            schema,
            steps,
            times,
            blocks,
            cell,
            state: Mutex::new(ReadState::default()),
        })
    }

    /// Read frame `index`, or `None` when it is past the commit marker.
    ///
    /// The frame is assembled from whichever sections resolve at `index`: a
    /// block whose `step_index` has no entry at or before `index` is left out
    /// (absent); a block whose latest update holds zero rows comes back as an
    /// empty block with its declared columns (present and empty); every other
    /// block carries its latest update's rows. The cell and the per-step
    /// metadata come along the same way.
    ///
    /// # Errors
    ///
    /// A [`MolRsError::Zarr`] naming the section when the store is internally
    /// inconsistent — a non-monotonic `offset`, a block on disk but absent
    /// from the schema, a malformed `box/` update, a `meta` row of the wrong
    /// width — and any storage or codec error raised while reading rows.
    pub fn frame(&self, index: u64) -> Result<Option<Frame>, MolRsError> {
        self.assemble(index, None)
    }

    /// Read frame `index` carrying only the named `(block, column)` pairs.
    ///
    /// A viewer that needs coordinates decodes `x`, `y`, `z` and nothing
    /// else. A block none of whose columns is named is left out; a named
    /// block still resolves to absent / empty / rows exactly as in
    /// [`frame`](Self::frame). The cell and per-step metadata are always
    /// carried — they are cheap.
    ///
    /// # Errors
    ///
    /// [`frame`](Self::frame)'s, plus a [`MolRsError::Zarr`] naming a pair
    /// the schema does not declare.
    pub fn frame_columns(
        &self,
        index: u64,
        columns: &[(&str, &str)],
    ) -> Result<Option<Frame>, MolRsError> {
        let mut wanted: BTreeMap<&str, Vec<&str>> = BTreeMap::new();
        for (block, column) in columns {
            let declared = self.schema.blocks.get(*block).ok_or_else(|| {
                MolRsError::zarr(format!("block {block:?} is not declared by this sequence"))
            })?;
            if !declared.columns.contains_key(*column) {
                return Err(MolRsError::zarr(format!(
                    "column {column:?} of block {block:?} is not declared by this sequence"
                )));
            }
            wanted.entry(block).or_default().push(column);
        }
        self.assemble(index, Some(&wanted))
    }

    fn assemble(
        &self,
        index: u64,
        wanted: Option<&BTreeMap<&str, Vec<&str>>>,
    ) -> Result<Option<Frame>, MolRsError> {
        if index as usize >= self.steps.len() {
            return Ok(None);
        }
        let mut state = self.state.lock().map_err(|_| {
            MolRsError::zarr("frame sequence read cache is poisoned by an earlier panic")
        })?;
        let mut frame = Frame::new();

        for (name, block_index) in &self.blocks {
            let selection = match wanted {
                Some(wanted) => match wanted.get(name.as_str()) {
                    Some(columns) => Some(columns.as_slice()),
                    None => continue,
                },
                None => None,
            };
            let Some((_, start, rows)) = block_index.resolve(index)? else {
                // No entry at or before this step: the section does not exist
                // here.
                continue;
            };
            let schema = self.schema.blocks.get(name).ok_or_else(|| {
                MolRsError::zarr(format!("block {name:?} is on disk but not in the schema"))
            })?;
            let path = join_path(TRAJECTORY_GROUP, name);
            let mut block = Block::new();
            for (column, declared) in &schema.columns {
                if let Some(selection) = selection
                    && !selection.contains(&column.as_str())
                {
                    continue;
                }
                let dtype = dtype_from_tag(&declared.dtype)?;
                let values = if rows == 0 {
                    empty_column(dtype, &declared.trailing)?
                } else {
                    let column_path = join_path(&path, column);
                    if !state.columns.contains_key(&column_path) {
                        state.columns.insert(
                            column_path.clone(),
                            ColumnReader::open(&self.store, &column_path)?,
                        );
                    }
                    state
                        .columns
                        .get_mut(&column_path)
                        .expect("just inserted")
                        .rows(start, rows, &declared.trailing)?
                };
                insert_column_into_block(&mut block, column, values)?;
                // The pinned precision and target travel with the column read
                // back, so a trajectory read and written again declares them.
                if let Some(target) = schema.targets.get(column) {
                    block.set_target(column, target).map_err(|e| {
                        MolRsError::zarr(format!(
                            "target of column {column:?} of block {name:?}: {e}"
                        ))
                    })?;
                }
                if let Some(precision) = declared.precision {
                    block.set_precision(column, precision).map_err(|e| {
                        MolRsError::zarr(format!(
                            "precision of column {column:?} of block {name:?}: {e}"
                        ))
                    })?;
                }
                if declared.nullable
                    && rows > 0
                    && let Some(mask) = self.mask_rows(&mut state, name, column, start, rows)?
                {
                    block.set_validity(column, mask).map_err(|e| {
                        MolRsError::zarr(format!(
                            "validity mask of column {column:?} of block {name:?}: {e}"
                        ))
                    })?;
                }
            }
            if block.is_empty() {
                // A declared block with no (selected) columns is still a block
                // with a row count.
                block.resize(rows as usize)?;
            }
            apply_structural_shape(&mut block, schema, &path)?;
            frame.insert(name.clone(), block);
        }

        if let Some(cell) = &self.cell
            && let Some(update) = latest_update(&cell.step_index, index)
        {
            if state.boxes.is_none() {
                state.boxes = Some(BoxReader::open(&self.store)?);
            }
            let boxes = state.boxes.as_mut().expect("just opened");
            frame.simbox = Some(boxes.cell_at(update as u64, cell.cell_defined)?);
        }

        for (key, declared) in &self.schema.meta {
            let path = join_path(&join_path(TRAJECTORY_GROUP, META_GROUP), key);
            if !state.metas.contains_key(key) {
                state
                    .metas
                    .insert(key.clone(), Array::open(self.store.clone(), &path)?);
            }
            let array = &state.metas[key];
            frame.meta.insert(
                key.clone(),
                read_meta_value(array, key, &declared.dtype, index)?,
            );
        }

        // A whole frame is the resolved frame: its pinned row references
        // must resolve, or the store is refused.
        if wanted.is_none() {
            check_local_references(&frame, &format!("trajectory frame {index}"))?;
        }
        Ok(Some(frame))
    }

    /// The validity flags of `column` over rows `start..start + rows`, or
    /// `None` when the store carries no mask array for it.
    ///
    /// A column pinned nullable whose run never holed it has no mask array;
    /// that absence is cached with the readers, so it costs one lookup rather
    /// than one per frame.
    ///
    /// # Errors
    ///
    /// A [`MolRsError::Zarr`] naming block and column when the mask is not a
    /// `bool` array, or does not cover the rows its column does — a mask that
    /// disagrees with its column is a corrupt store, and padding it would
    /// invent the very answer the mask exists to give.
    fn mask_rows(
        &self,
        state: &mut ReadState,
        block: &str,
        column: &str,
        start: u64,
        rows: u64,
    ) -> Result<Option<Vec<bool>>, MolRsError> {
        let section = join_path(TRAJECTORY_GROUP, block);
        let path = join_path(&join_path(&section, VALIDITY_GROUP), column);
        if !state.masks.contains_key(&path) {
            let reader = match array_exists(&self.store, &path)? {
                true => Some(ColumnReader::open(&self.store, &path)?),
                false => None,
            };
            state.masks.insert(path.clone(), reader);
        }
        let Some(reader) = state.masks.get_mut(&path).expect("just inserted").as_mut() else {
            return Ok(None);
        };
        let flags = reader.rows(start, rows, &[]).map_err(|e| {
            MolRsError::zarr(format!(
                "validity mask of column {column:?} of block {block:?} does not cover rows \
                 {start}..{}: {e}",
                start + rows
            ))
        })?;
        let flags = flags.get::<bool>().ok_or_else(|| {
            MolRsError::zarr(format!(
                "validity mask of column {column:?} of block {block:?} is stored as {}, and a \
                 mask is one bool per row",
                flags.dtype()
            ))
        })?;
        Ok(Some(flags.iter().copied().collect()))
    }

    /// The update of block `name` that frame `index` resolves to, or `None`
    /// when the block is absent there (or not in the store).
    ///
    /// Two consecutive frames resolving to the same update carry the same
    /// rows — a consumer can skip re-uploading a block whose update did not
    /// change without comparing a single value.
    pub fn block_update_at(&self, name: &str, index: u64) -> Result<Option<u64>, MolRsError> {
        if index as usize >= self.steps.len() {
            return Ok(None);
        }
        let Some(block_index) = self.blocks.get(name) else {
            return Ok(None);
        };
        Ok(block_index.resolve(index)?.map(|(update, _, _)| update))
    }

    /// The cell at frame `index`, or `None` when no cell has been written at
    /// or before it (or `index` is past the commit marker).
    ///
    /// # Errors
    ///
    /// A [`MolRsError::Zarr`] when the `box/` update is malformed.
    pub fn box_at(&self, index: u64) -> Result<Option<SimBox>, MolRsError> {
        if index as usize >= self.steps.len() {
            return Ok(None);
        }
        let Some(cell) = &self.cell else {
            return Ok(None);
        };
        let Some(update) = latest_update(&cell.step_index, index) else {
            return Ok(None);
        };
        let mut state = self.state.lock().map_err(|_| {
            MolRsError::zarr("frame sequence read cache is poisoned by an earlier panic")
        })?;
        if state.boxes.is_none() {
            state.boxes = Some(BoxReader::open(&self.store)?);
        }
        let boxes = state.boxes.as_mut().expect("just opened");
        Ok(Some(boxes.cell_at(update as u64, cell.cell_defined)?))
    }

    /// Materialize the whole sequence: the named lazy → eager conversion.
    ///
    /// Reads every committed frame through [`frame`](Self::frame) and hands
    /// back a `Trajectory` carrying those frames, the sequence's step numbers,
    /// and its times when the run wrote any. A run too large for memory is
    /// what [`frame`](Self::frame) is for.
    ///
    /// # Errors
    ///
    /// [`frame`](Self::frame)'s, on the first frame that raises one.
    pub fn to_trajectory(&self) -> Result<Trajectory, MolRsError> {
        let mut frames = Vec::with_capacity(self.steps.len());
        for index in 0..self.steps.len() as u64 {
            let Some(frame) = self.frame(index)? else {
                break;
            };
            frames.push(frame);
        }
        Ok(Trajectory {
            frames,
            step: Some(self.steps.clone()),
            time: self.times.clone(),
        })
    }

    /// The committed frames' step numbers, as [`open`](Self::open) read them.
    pub fn steps(&self) -> &[i64] {
        &self.steps
    }

    /// The committed frames' times, when the run wrote a `time` array.
    pub fn times(&self) -> Option<&[F]> {
        self.times.as_deref()
    }

    /// The sequence schema pinned or derived at [`open`](Self::open).
    pub fn schema(&self) -> &SequenceSchema {
        &self.schema
    }

    /// Whether a block section is present in the store with at least one
    /// committed update, so a reader can decide once — e.g.
    /// `has_block("bonds")` — instead of probing every frame for it.
    pub fn has_block(&self, name: &str) -> bool {
        self.blocks.contains_key(name)
    }

    /// The names of every block section present in the store.
    pub fn block_names(&self) -> impl Iterator<Item = &str> {
        self.blocks.keys().map(String::as_str)
    }
}

/// Refuse a store whose aligned blocks break the rule at some resolved frame,
/// checked from the indexes alone: at every ordinal where either block of a
/// pair has an update, the aligned block, when it resolves, has its target
/// resolving with the same row count.
fn check_aligned_indexes(
    schema: &SequenceSchema,
    blocks: &IndexMap<String, BlockIndex>,
    nstep: u64,
) -> Result<(), MolRsError> {
    let ordinals = |name: &str| -> Vec<u64> {
        match blocks.get(name) {
            None => Vec::new(),
            Some(BlockIndex::Sparse { step_index, .. }) => step_index.clone(),
            Some(BlockIndex::Regular { updates, .. }) => (0..*updates).collect(),
        }
    };
    for (aligned, target) in schema.aligned_pairs() {
        let mut at: Vec<u64> = ordinals(aligned);
        at.extend(ordinals(target));
        at.sort_unstable();
        at.dedup();
        for ordinal in at.into_iter().filter(|&o| o < nstep) {
            let Some((_, _, rows)) = blocks
                .get(aligned)
                .map(|index| index.resolve(ordinal))
                .transpose()?
                .flatten()
            else {
                continue;
            };
            let with = blocks
                .get(target)
                .map(|index| index.resolve(ordinal))
                .transpose()?
                .flatten();
            match with {
                Some((_, _, target_rows)) if target_rows == rows => {}
                Some((_, _, target_rows)) => {
                    return Err(MolRsError::zarr(format!(
                        "trajectory frame {ordinal}: aligned block {aligned:?} resolves to {rows} \
                         rows but its target {target:?} to {target_rows}"
                    )));
                }
                None => {
                    return Err(MolRsError::zarr(format!(
                        "trajectory frame {ordinal}: aligned block {aligned:?} is present but its \
                         target {target:?} is absent"
                    )));
                }
            }
        }
    }
    Ok(())
}

/// Refuse an aligned trajectory block named like a block of the record's
/// `system` section: the name would mean two different row sets.
fn check_aligned_not_in_system<S>(store: &Arc<S>, schema: &SequenceSchema) -> Result<(), MolRsError>
where
    S: ?Sized + ReadableStorageTraits + 'static,
{
    for (aligned, _) in schema.aligned_pairs() {
        if group_exists(store, &join_path(SYSTEM_GROUP, aligned))? {
            return Err(MolRsError::zarr(format!(
                "aligned trajectory block {aligned:?} shares its name with a system block; an \
                 aligned block never does"
            )));
        }
    }
    Ok(())
}

/// Index of the latest entry at or before frame `index`, or `None` when the
/// section had not appeared yet.
///
/// `step_index` is ascending by construction, so this is a binary search: the
/// cost of resolving a section is logarithmic in its *changes*, not in the run
/// length.
fn latest_update(step_index: &[u64], index: u64) -> Option<usize> {
    match step_index.binary_search(&index) {
        Ok(found) => Some(found),
        Err(0) => None,
        Err(after) => Some(after - 1),
    }
}

impl TrajectoryReader for MrecReader {
    /// Already cached: [`MrecReader::open`] is the index-only read, so
    /// there is nothing left for this to build.
    fn build_index(&mut self) -> std::io::Result<()> {
        Ok(())
    }

    /// [`MrecReader::frame`] behind the backend-neutral trait: the same
    /// frame, and the same `Ok(None)` past the commit marker.
    fn read_frame(&mut self, index: usize) -> std::io::Result<Option<Frame>> {
        self.frame(index as u64)
            .map_err(crate::io::mrec::zarr_storage::record_io::io_error)
    }

    /// Committed frames — the length of `trajectory/step` as
    /// [`MrecReader::open`] read it.
    fn len(&mut self) -> std::io::Result<usize> {
        Ok(self.steps.len())
    }
}

/// The whole `trajectory/` sequence contract, pinned before the code exists.
///
/// **Fixtures.** Every test writes a real store under a `tempfile::TempDir`;
/// nothing here is a mock except [`RecordingStorage`], which exists only to
/// observe *write order* and delegates every byte to a real `FilesystemStore`.
/// The stock fixture frame is one `atoms` block carrying one `f64` column
/// [`X`], with non-repeating values, so a mis-sliced CSR row range is a
/// mismatch rather than a coincidence.
///
/// **What is pinned on disk.** The layout of Design § 1 is the contract, so
/// these tests read `trajectory/step`, `trajectory/<block>/offset`,
/// `trajectory/<block>/step_index`, `trajectory/box/step_index` and
/// `trajectory/meta/<key>` directly through `zarrs`, in addition to reading
/// frames back through [`MrecReader`]. A round trip that agreed with itself
/// while writing a private layout would satisfy neither molrec nor molpy.
///
/// **No tolerances anywhere.** Every value assertion is `assert_eq!` on the
/// bits that arrived: the downstream precision study forbids lossy encoding,
/// so a comparison with a tolerance would hide exactly the defect this suite
/// is for.
///
/// **Extents are read from the on-disk array metadata**, because the growth
/// arrays' chunk and shard extents are frozen at the first append and have no
/// other observation point ([`extents`]).
#[cfg(all(test, feature = "filesystem"))]
mod tests {
    use crate::core::Cell;
    use std::collections::BTreeMap;
    use std::path::{Path, PathBuf};
    use std::sync::{Arc, Mutex};

    use molrs::core::Frame;
    use molrs::core::MetaValue;
    use molrs::core::SimBox;
    use molrs::core::{Block, Column, DType};
    use ndarray::ArrayD;
    use tempfile::TempDir;
    use zarrs::array::{Array, ArrayBuilder, ArraySubset, data_type};
    use zarrs::filesystem::FilesystemStore;
    use zarrs::group::{Group, GroupBuilder};
    use zarrs::storage::byte_range::{ByteRange, ByteRangeIterator};
    use zarrs::storage::{
        Bytes, ListableStorageTraits, MaybeBytes, MaybeBytesIterator, OffsetBytesIterator,
        ReadableStorageTraits, ReadableWritableListableStorage, StorageError, StoreKey, StoreKeys,
        StoreKeysPrefixes, StorePrefix, WritableStorageTraits,
    };

    use crate::io::mrec::zarr_storage::positional_write::PositionalWriteStorage;
    use crate::io::reader::TrajectoryReader;

    use super::{MrecReader, MrecWriter, SequenceSchema};

    // -- fixtures -----------------------------------------------------------

    /// The sequence root group: the on-disk name of the whole layout.
    const TRAJ: &str = "/trajectory";
    /// The `trajectory/` group attribute the **reference writer** pins its
    /// declared schema in.
    ///
    /// Spelled out rather than imported from `super`: this is a name on disk
    /// that a foreign writer reads and writes, so a rename of the Rust constant
    /// must break these tests rather than travel silently into them — the same
    /// reading `every_meta_variant_round_trips_bit_exact_with_its_dtype_tag`
    /// takes on `meta_dtype`.
    const SCHEMA_ATTRIBUTE: &str = "sequence_schema";
    /// The block every stock fixture frame carries.
    const ATOMS: &str = "atoms";
    /// The one `f64` column in that block.
    const X: &str = "x";
    /// A second block, used wherever a frame must present a *subset* of the
    /// declared union.
    const BONDS: &str = "bonds";
    /// The `u64` column of [`BONDS`].
    const I: &str = "i";
    /// A column name outside the canonical schema vocabulary
    /// (`core/schema/mod.rs`), so the same name can carry two dtypes.
    ///
    /// [`X`] cannot: `x` is pinned `Float` there, and `Block::insert`
    /// rejects a `u64` under that name before any sequence code runs — the
    /// dtype-conflict fixtures would then be testing `Block`, not the schema.
    /// Same reason `frame_io`'s round-trip tests probe on a name of their own.
    const PROBE: &str = "probe";

    fn storage_in(dir: &TempDir) -> ReadableWritableListableStorage {
        Arc::new(FilesystemStore::new(dir.path()).unwrap())
    }

    fn float_column(values: &[f64]) -> Column {
        Column::from(ArrayD::from_shape_vec(vec![values.len()], values.to_vec()).unwrap())
    }

    fn uint_column(values: &[u64]) -> Column {
        Column::from(ArrayD::from_shape_vec(vec![values.len()], values.to_vec()).unwrap())
    }

    fn block_with(column: &str, col: Column) -> Block {
        let mut block = Block::new();
        block.insert(column, col, None).unwrap();
        block
    }

    /// The stock fixture frame: one `atoms` block, one `f64` column.
    fn atoms_frame(values: &[f64]) -> Frame {
        let mut frame = Frame::new();
        frame.insert(ATOMS, block_with(X, float_column(values)));
        frame
    }

    /// Column [`X`] of the `atoms` block, as it came back from the store.
    fn atoms_x(frame: &Frame) -> Vec<f64> {
        frame
            .get(ATOMS)
            .expect("the frame carries an atoms block")
            .get(X)
            .expect("the atoms block carries column x")
            .get::<f64>()
            .expect("column x arrived as f64")
            .iter()
            .copied()
            .collect()
    }

    /// The ragged golden run: 3, 5 and 4 rows, every value distinct so a row
    /// range taken from the wrong frame cannot match by accident.
    fn ragged_frames() -> Vec<Frame> {
        vec![
            atoms_frame(&[1.0, 2.0, 3.0]),
            atoms_frame(&[10.0, 11.0, 12.0, 13.0, 14.0]),
            atoms_frame(&[20.0, 21.0, 22.0, 23.0]),
        ]
    }

    /// The cell every box fixture carries — default origin and all-periodic
    /// boundary, so those optional arrays are omitted.
    fn fixed_cell() -> SimBox {
        SimBox::new(
            Cell::Matrix([[10.0, 0.0, 0.0], [0.0, 11.0, 0.0], [0.0, 0.0, 12.0]]),
            [0.0, 0.0, 0.0],
            [true, true, true],
        )
        .unwrap()
    }

    /// Mint from the union of `frames`, append them all in order, and close.
    fn write_all(store: &ReadableWritableListableStorage, frames: &[Frame]) {
        let schema = SequenceSchema::from_frames(frames).unwrap();
        let mut writer = MrecWriter::create_in_storage(store.clone(), schema).unwrap();
        for frame in frames {
            writer.append(frame).unwrap();
        }
        writer.close().unwrap();
    }

    /// [`write_all`] with both extent knobs pinned, for the tests that need a
    /// tiny `R` to reach a chunk boundary in a handful of frames.
    fn write_all_with(
        store: &ReadableWritableListableStorage,
        frames: &[Frame],
        rows_per_chunk: u64,
        chunks_per_shard: u64,
    ) {
        let schema = SequenceSchema::from_frames(frames).unwrap();
        let mut writer = MrecWriter::create_in_storage(store.clone(), schema)
            .unwrap()
            .with_rows_per_chunk(rows_per_chunk)
            .unwrap()
            .with_chunks_per_shard(chunks_per_shard)
            .unwrap();
        for frame in frames {
            writer.append(frame).unwrap();
        }
        writer.close().unwrap();
    }

    /// Drop [`SCHEMA_ATTRIBUTE`] from `trajectory/`'s `zarr.json`, leaving every
    /// array, every array attribute and every byte of data untouched.
    ///
    /// That is exactly the store a conforming foreign writer leaves behind:
    /// molrec's `docs/spec/trajectory.md` makes the attribute the **reference
    /// writer's** own reopen pin, and resolution needs only `step_index` /
    /// `offset` plus the array metadata. The removal is asserted rather than
    /// assumed, so a renamed attribute cannot turn either caller into a test of
    /// an unmodified store.
    fn strip_schema_attribute(store: &ReadableWritableListableStorage) {
        let mut group = Group::open(store.clone(), TRAJ).expect("the sequence group exists");
        assert!(
            group.attributes_mut().remove(SCHEMA_ATTRIBUTE).is_some(),
            "the writer must have pinned {SCHEMA_ATTRIBUTE} for this to be a removal"
        );
        group
            .store_metadata()
            .expect("the group metadata rewrites without the attribute");
    }

    fn open_sequence(store: &ReadableWritableListableStorage) -> MrecReader {
        MrecReader::from_storage(store.clone()).unwrap()
    }

    /// `frame(i)`, asserting the frame is present.
    fn frame_at(seq: &mut MrecReader, index: u64) -> Frame {
        seq.frame(index)
            .expect("reading a frame must not error")
            .expect("frame is present")
    }

    /// The committed frame count, taken through the `TrajectoryReader` door —
    /// the surface `FrameIterator` and every generic consumer already use.
    fn committed_len(seq: &mut MrecReader) -> usize {
        TrajectoryReader::len(seq).expect("len must not error")
    }

    fn u64_array(store: &ReadableWritableListableStorage, path: &str) -> Vec<u64> {
        let arr = Array::open(store.clone(), path).expect("array exists");
        let subset = ArraySubset::new_with_shape(arr.shape().to_vec());
        arr.retrieve_array_subset(&subset)
            .expect("array reads back")
    }

    fn i64_array(store: &ReadableWritableListableStorage, path: &str) -> Vec<i64> {
        let arr = Array::open(store.clone(), path).expect("array exists");
        let subset = ArraySubset::new_with_shape(arr.shape().to_vec());
        arr.retrieve_array_subset(&subset)
            .expect("array reads back")
    }

    /// The frozen extents of a growth array: `(shard extent, inner chunk
    /// extent)`, both in rows of the leading axis and whatever the trailing
    /// axes are.
    ///
    /// The array's *shape* is deliberately not returned: it grows with every
    /// append, while the two extents are frozen by the first append and must
    /// never move again (`zarrs::Array::set_shape` rebuilds the grid from the
    /// frozen chunk metadata). Under sharding the array's own chunk extent is
    /// the shard, and the planned chunk is the sharding codec's subchunk —
    /// the same reading `frame_io`'s `a_five_chunk_column_shards_and_round_trips`
    /// takes.
    fn extents(store: &ReadableWritableListableStorage, path: &str) -> (Vec<u64>, Vec<u64>) {
        let arr = Array::open(store.clone(), path).expect("array exists");
        let metadata = serde_json::to_value(arr.metadata()).unwrap();
        let shard: Vec<u64> =
            serde_json::from_value(metadata["chunk_grid"]["configuration"]["chunk_shape"].clone())
                .expect("a regular chunk grid carries a chunk_shape");
        let sharding = metadata["codecs"]
            .as_array()
            .expect("codecs is an array")
            .iter()
            .find(|codec| codec["name"] == "sharding_indexed")
            .expect("a growth array is sharded, so file count follows bytes")
            .clone();
        let inner: Vec<u64> =
            serde_json::from_value(sharding["configuration"]["chunk_shape"].clone())
                .expect("the sharding codec carries its subchunk shape");
        (shard, inner)
    }

    /// Every file under `dir`, keyed by its path relative to `dir`.
    fn file_map(dir: &Path) -> BTreeMap<PathBuf, Vec<u8>> {
        let mut out = BTreeMap::new();
        let mut stack = vec![dir.to_path_buf()];
        while let Some(current) = stack.pop() {
            for entry in std::fs::read_dir(&current).unwrap().flatten() {
                let path = entry.path();
                if path.is_dir() {
                    stack.push(path);
                } else {
                    let key = path.strip_prefix(dir).unwrap().to_path_buf();
                    out.insert(key, std::fs::read(&path).unwrap());
                }
            }
        }
        out
    }

    /// The chunk (shard) files under `dir`, without the `zarr.json` metadata —
    /// the bytes a shard actually costs.
    fn chunk_files(dir: &Path) -> BTreeMap<PathBuf, Vec<u8>> {
        file_map(dir)
            .into_iter()
            .filter(|(path, _)| path.file_name() != Some(std::ffi::OsStr::new("zarr.json")))
            .collect()
    }

    fn chunk_bytes(dir: &Path) -> u64 {
        chunk_files(dir).values().map(|v| v.len() as u64).sum()
    }

    /// Deterministic, poorly compressible `f64` payload in `[1, 2)`: a
    /// splitmix hash supplies 52 random mantissa bits, so gzip lands near
    /// 0.85x and every value is finite and bit-exactly comparable. Same shape
    /// as the `zarrs_pins` payload, for the same reason: a compressible
    /// fixture would make a write-amplification bound pass vacuously.
    fn noise(index: u64) -> f64 {
        let mut x = index.wrapping_add(0x9e37_79b9_7f4a_7c15);
        x = (x ^ (x >> 30)).wrapping_mul(0xbf58_476d_1ce4_e5b9);
        x = (x ^ (x >> 27)).wrapping_mul(0x94d0_49bb_1331_11eb);
        x ^= x >> 31;
        f64::from_bits((0x3ffu64 << 52) | (x & 0x000f_ffff_ffff_ffff))
    }

    // =======================================================================
    // A. Layout primitives — CSR, per-section step_index, derived extents
    // =======================================================================

    /// `offset` is the CSR row pointer of the ragged run and nothing else:
    /// `offset[0] = 0` and the prefix sums of 3, 5 and 4 rows (ac-014).
    ///
    /// The row *counts* are never stored — they are `diff(offset)`, decision 2
    /// — so this array is the only place the raggedness lives.
    #[test]
    fn csr_offset_is_the_prefix_sum_of_a_ragged_run() {
        let dir = TempDir::new().unwrap();
        let store = storage_in(&dir);
        write_all(&store, &ragged_frames());

        assert_eq!(
            u64_array(&store, &format!("{TRAJ}/{ATOMS}/offset")),
            vec![0, 3, 8, 12],
            "offset is the prefix sum of 3, 5 and 4 rows"
        );
    }

    /// `nstep` counts every appended frame, and a regular numbering lives in
    /// the `step_progression` attribute rather than in an array.
    #[test]
    fn step_counts_every_appended_frame() {
        let dir = TempDir::new().unwrap();
        let store = storage_in(&dir);
        write_all(&store, &ragged_frames());
        let group = Group::open(store.clone(), TRAJ).unwrap();
        assert_eq!(group.attributes()["nstep"], 3);
        assert_eq!(
            group.attributes()["step_progression"],
            serde_json::json!({"start": 0, "stride": 1})
        );
        assert!(
            Array::open(store.clone(), &format!("{TRAJ}/step")).is_err(),
            "a regular step series needs no array"
        );
        let seq = open_sequence(&store);
        assert_eq!(seq.steps(), &[0, 1, 2]);
    }

    /// Each frame of the ragged run reads back its own rows, bit exact
    /// (ac-014).
    #[test]
    fn each_frame_of_a_ragged_run_reads_back_its_own_rows() {
        let dir = TempDir::new().unwrap();
        let store = storage_in(&dir);
        let frames = ragged_frames();
        write_all(&store, &frames);

        let mut seq = open_sequence(&store);
        for (index, frame) in frames.iter().enumerate() {
            assert_eq!(
                atoms_x(&frame_at(&mut seq, index as u64)),
                atoms_x(frame),
                "frame {index} must come back bit exact"
            );
        }
    }

    /// Blocks and columns read back in the order the frame carried them —
    /// the pinned schema records declaration order, and the reader walks it.
    #[test]
    fn blocks_and_columns_read_back_in_frame_order() {
        let dir = TempDir::new().unwrap();
        let store = storage_in(&dir);
        let mut zeta = Block::new();
        for column in ["c", "a", "b"] {
            zeta.insert(column, float_column(&[1.0]), None).unwrap();
        }
        let mut frame = Frame::new();
        frame.insert("zeta", zeta);
        frame.insert(ATOMS, block_with(X, float_column(&[2.0])));
        write_all(&store, &[frame]);

        let back = frame_at(&mut open_sequence(&store), 0);
        assert_eq!(back.keys().collect::<Vec<_>>(), ["zeta", ATOMS]);
        assert_eq!(back["zeta"].keys().collect::<Vec<_>>(), ["c", "a", "b"]);
    }

    /// A block with no `step_index` entry `<= i` is **absent** at step `i`,
    /// not present-and-empty (ac-014).
    ///
    /// Absence is how the layout expresses "this section does not exist yet";
    /// a zero-row block would be a different claim (it exists and holds
    /// nothing), and molrec's conformance suite distinguishes the two.
    #[test]
    fn a_block_absent_at_a_step_reads_as_absent_not_empty() {
        let dir = TempDir::new().unwrap();
        let store = storage_in(&dir);
        let mut with_bonds = atoms_frame(&[10.0, 11.0]);
        with_bonds.insert(BONDS, block_with(I, uint_column(&[0, 1])));
        let frames = vec![atoms_frame(&[1.0, 2.0]), with_bonds];
        write_all(&store, &frames);

        let mut seq = open_sequence(&store);
        assert!(
            frame_at(&mut seq, 0).get(BONDS).is_none(),
            "a block whose first step_index entry is 1 does not exist at step 0"
        );
        assert_eq!(
            frame_at(&mut seq, 1)
                .get(BONDS)
                .expect("bonds exists at step 1")
                .n_rows(),
            Some(2),
            "and at step 1 it carries the rows it was appended with"
        );
    }

    /// A fixed cell over a 20-step run writes the cell once, and omits the
    /// optional box arrays that would only restate defaults (decision 7).
    #[test]
    fn a_fixed_cell_omits_default_origin_boundary_and_trivial_step_index() {
        const STEPS: usize = 20;

        let dir = TempDir::new().unwrap();
        let store = storage_in(&dir);
        let frames: Vec<Frame> = (0..STEPS)
            .map(|step| {
                let mut frame = atoms_frame(&[step as f64, step as f64 + 0.5]);
                frame.simbox = Some(fixed_cell());
                frame
            })
            .collect();
        write_all(&store, &frames);

        assert!(
            Array::open(store.clone(), &format!("{TRAJ}/box/step_index")).is_err(),
            "a single update at ordinal 0 does not write step_index"
        );
        assert!(
            Array::open(store.clone(), &format!("{TRAJ}/box/origin")).is_err(),
            "a zero origin is the default and is not written"
        );
        assert!(
            Array::open(store.clone(), &format!("{TRAJ}/box/boundary")).is_err(),
            "all-periodic boundary is the default and is not written"
        );
        let mut seq = open_sequence(&store);
        let cell = frame_at(&mut seq, 19)
            .simbox
            .expect("later frames still carry the cell");
        assert_eq!(cell.origin(), [0.0; 3]);
        assert_eq!(cell.pbc(), [true, true, true]);
    }

    /// An undefined cell's `vectors` are ignored on read, in both box
    /// layouts: zeros in the fixed-cell attributes, and an all-zero (fill
    /// value) `vectors` array, read back as the identity instead of failing
    /// to invert.
    #[test]
    fn an_undefined_cell_reads_whatever_vectors_it_carries() {
        let undefined =
            |x: f64| SimBox::new(Cell::Undefined, [x, 0.0, 0.0], [false, false, false]).unwrap();

        // Fixed cell: `vectors` is a group attribute.
        let dir = TempDir::new().unwrap();
        let store = storage_in(&dir);
        let mut frame = atoms_frame(&[1.0]);
        frame.simbox = Some(undefined(0.0));
        write_all(&store, &[frame]);
        let box_path = format!("{TRAJ}/box");
        let mut attributes = Group::open(store.clone(), &box_path)
            .unwrap()
            .attributes()
            .clone();
        assert_eq!(attributes["cell_defined"], serde_json::json!(false));
        attributes.insert(
            "vectors".into(),
            serde_json::json!([[0.0, 0.0, 0.0], [0.0, 0.0, 0.0], [0.0, 0.0, 0.0]]),
        );
        GroupBuilder::new()
            .attributes(attributes)
            .build(store.clone(), &box_path)
            .unwrap()
            .store_metadata()
            .unwrap();
        let back = open_sequence(&store).box_at(0).unwrap().unwrap();
        assert!(!back.is_cell_defined());
        assert_eq!(
            back.matrix(),
            [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]]
        );

        // Changing origin: the cell lives in arrays. Dropping the `vectors`
        // chunks makes every row the zero fill value.
        let dir = TempDir::new().unwrap();
        let store = storage_in(&dir);
        let mut first = atoms_frame(&[1.0]);
        first.simbox = Some(undefined(0.0));
        let mut second = atoms_frame(&[2.0]);
        second.simbox = Some(undefined(1.0));
        write_all(&store, &[first, second]);
        let chunks = dir.path().join("trajectory/box/vectors/c");
        assert!(chunks.is_dir(), "the cell moved into arrays");
        std::fs::remove_dir_all(chunks).unwrap();
        let seq = open_sequence(&store);
        for (index, x) in [(0, 0.0), (1, 1.0)] {
            let back = seq.box_at(index).unwrap().unwrap();
            assert!(!back.is_cell_defined());
            assert_eq!(
                back.matrix(),
                [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]]
            );
            assert_eq!(back.origin()[0], x);
        }
    }

    /// Rewrite the attributes of the group at `path` through `edit`.
    fn edit_attributes(
        store: &ReadableWritableListableStorage,
        path: &str,
        edit: impl FnOnce(&mut serde_json::Map<String, serde_json::Value>),
    ) {
        let mut group = Group::open(store.clone(), path).unwrap();
        edit(group.attributes_mut());
        group.store_metadata().unwrap();
    }

    /// A regular three-frame run: `atoms` carries both elision markers and
    /// `step` is a progression.
    fn regular_storage(dir: &TempDir) -> ReadableWritableListableStorage {
        let store = storage_in(dir);
        write_all(
            &store,
            &[
                atoms_frame(&[1.0]),
                atoms_frame(&[2.0]),
                atoms_frame(&[3.0]),
            ],
        );
        let attrs = Group::open(store.clone(), &format!("{TRAJ}/{ATOMS}"))
            .unwrap()
            .attributes()
            .clone();
        assert_eq!(attrs["uniform_rows"], serde_json::json!(1));
        assert_eq!(attrs["dense_updates"], serde_json::json!(true));
        store
    }

    #[test]
    fn malformed_elision_markers_are_refused() {
        let section = format!("{TRAJ}/{ATOMS}");
        for edit in [
            |a: &mut serde_json::Map<String, serde_json::Value>| {
                a.remove("dense_updates");
            },
            |a: &mut serde_json::Map<String, serde_json::Value>| {
                a.remove("uniform_rows");
            },
            |a: &mut serde_json::Map<String, serde_json::Value>| {
                a.insert("uniform_rows".into(), serde_json::json!(0));
            },
            |a: &mut serde_json::Map<String, serde_json::Value>| {
                a.insert("dense_updates".into(), serde_json::json!(false));
            },
        ] {
            let dir = TempDir::new().unwrap();
            let store = regular_storage(&dir);
            edit_attributes(&store, &section, edit);
            let err = MrecReader::from_storage(store.clone())
                .err()
                .unwrap()
                .to_string();
            assert!(err.contains(ATOMS), "{err}");
            assert!(MrecWriter::from_storage(store).is_err());
        }
    }

    #[test]
    fn markers_on_a_block_with_no_columns_are_refused_and_a_bare_declared_block_reads_absent() {
        let dir = TempDir::new().unwrap();
        let store = regular_storage(&dir);
        edit_attributes(&store, TRAJ, |a| {
            a[SCHEMA_ATTRIBUTE]["blocks"]["ghost"] = serde_json::json!({"columns": {}});
        });
        GroupBuilder::new()
            .build(store.clone(), &format!("{TRAJ}/ghost"))
            .unwrap()
            .store_metadata()
            .unwrap();
        // Declared, never updated: absent everywhere, and not refused.
        let seq = open_sequence(&store);
        assert!(!seq.has_block("ghost"));
        assert!(!frame_at(&mut open_sequence(&store), 1).contains_key("ghost"));
        drop(seq);

        edit_attributes(&store, &format!("{TRAJ}/ghost"), |a| {
            a.insert("uniform_rows".into(), serde_json::json!(1));
            a.insert("dense_updates".into(), serde_json::json!(true));
        });
        let err = MrecReader::from_storage(store).err().unwrap().to_string();
        assert!(err.contains("ghost") && err.contains("no columns"), "{err}");
    }

    #[test]
    fn a_progression_without_the_commit_marker_is_refused() {
        let dir = TempDir::new().unwrap();
        let store = regular_storage(&dir);
        let attrs = Group::open(store.clone(), TRAJ)
            .unwrap()
            .attributes()
            .clone();
        assert!(attrs.contains_key("step_progression"));
        edit_attributes(&store, TRAJ, |a| {
            a.remove("nstep");
        });
        let err = MrecReader::from_storage(store.clone())
            .err()
            .unwrap()
            .to_string();
        assert!(err.contains("nstep"), "{err}");
        edit_attributes(&store, TRAJ, |a| {
            a.insert("nstep".into(), serde_json::json!(-1));
        });
        assert!(MrecReader::from_storage(store).is_err());
    }

    /// An undefined trajectory cell is periodic on no axis: a writer refuses
    /// one with a periodic flag, an omitted boundary reads all-false, and a
    /// periodic flag in the store is refused.
    #[test]
    fn an_undefined_trajectory_cell_is_periodic_on_no_axis() {
        let undefined = |pbc| SimBox::new(Cell::Undefined, [0.0, 0.0, 0.0], pbc).unwrap();
        let dir = TempDir::new().unwrap();
        let store = storage_in(&dir);
        let mut frame = atoms_frame(&[1.0]);
        frame.simbox = Some(undefined([false, false, true]));
        let mut writer = MrecWriter::create_in_storage(
            store.clone(),
            SequenceSchema::from_frames(std::slice::from_ref(&frame)).unwrap(),
        )
        .unwrap();
        assert!(writer.append(&frame).is_err());

        let dir = TempDir::new().unwrap();
        let store = storage_in(&dir);
        frame.simbox = Some(undefined([false; 3]));
        write_all(&store, &[frame]);
        let box_path = format!("{TRAJ}/box");
        let mut group = Group::open(store.clone(), &box_path).unwrap();
        assert_eq!(
            group.attributes()["boundary"],
            serde_json::json!([false, false, false])
        );
        group.attributes_mut().remove("boundary");
        group.store_metadata().unwrap();
        assert_eq!(
            open_sequence(&store).box_at(0).unwrap().unwrap().pbc(),
            [false; 3]
        );

        let mut group = Group::open(store.clone(), &box_path).unwrap();
        group
            .attributes_mut()
            .insert("boundary".into(), serde_json::json!([true, false, false]));
        group.store_metadata().unwrap();
        let refused = match MrecReader::from_storage(store.clone()) {
            Err(_) => true,
            Ok(seq) => seq.box_at(0).is_err(),
        };
        assert!(
            refused,
            "a periodic flag on an undefined cell must be refused"
        );
    }

    /// A canonical identifier key cannot be declared narrower than `u64`:
    /// the in-memory block widens it, so nothing could be appended, and a
    /// store holding one is refused on read.
    #[test]
    fn a_canonical_identifier_cannot_be_declared_narrow() {
        let mut schema = SequenceSchema::new();
        for dtype in [DType::U8, DType::U16, DType::U32] {
            let err = schema
                .declare_column("bonds", "atomi", dtype, &[])
                .unwrap_err()
                .to_string();
            assert!(err.contains("atomi") && err.contains(dtype.name()), "{err}");
        }
        schema
            .declare_column("bonds", "atomi", DType::Uint, &[])
            .unwrap();
        // A non-canonical name keeps its arrival width.
        schema
            .declare_column("bonds", "my_label", DType::U32, &[])
            .unwrap();
    }

    /// A non-default origin is recorded — as a `box/` attribute while the
    /// cell is fixed, as an array once the cell changes; the still-default
    /// boundary is omitted either way.
    #[test]
    fn a_nonzero_origin_is_written() {
        let cell = |lengths: f64| {
            SimBox::new(
                Cell::Matrix([[lengths, 0.0, 0.0], [0.0, 11.0, 0.0], [0.0, 0.0, 12.0]]),
                [1.0, 2.0, 3.0],
                [true, true, true],
            )
            .unwrap()
        };
        let dir = TempDir::new().unwrap();
        let store = storage_in(&dir);
        let mut frame = atoms_frame(&[1.0]);
        frame.simbox = Some(cell(10.0));
        write_all(&store, &[frame]);
        let group = Group::open(store.clone(), &format!("{TRAJ}/box")).unwrap();
        assert_eq!(
            group.attributes()["origin"],
            serde_json::json!([1.0, 2.0, 3.0])
        );
        assert!(
            group.attributes().get("boundary").is_none(),
            "still-default boundary is omitted"
        );
        assert!(
            Array::open(store.clone(), &format!("{TRAJ}/box/vectors")).is_err(),
            "a fixed cell is attributes, not arrays"
        );
        let seq = open_sequence(&store);
        let back = seq.box_at(0).unwrap().unwrap();
        assert_eq!(back.origin().to_vec(), vec![1.0, 2.0, 3.0]);

        // The first change migrates the cell into arrays.
        let dir = TempDir::new().unwrap();
        let store = storage_in(&dir);
        let mut first = atoms_frame(&[1.0]);
        first.simbox = Some(cell(10.0));
        let mut second = atoms_frame(&[2.0]);
        second.simbox = Some(cell(10.5));
        write_all(&store, &[first, second]);
        let origin: Vec<f64> = {
            let arr = Array::open(store.clone(), &format!("{TRAJ}/box/origin")).unwrap();
            let subset = ArraySubset::new_with_shape(arr.shape().to_vec());
            arr.retrieve_array_subset(&subset).unwrap()
        };
        assert_eq!(origin, vec![1.0, 2.0, 3.0, 1.0, 2.0, 3.0]);
        assert_eq!(
            u64_array(&store, &format!("{TRAJ}/box/step_index")),
            vec![0, 1]
        );
        assert!(Array::open(store.clone(), &format!("{TRAJ}/box/boundary")).is_err());
        let group = Group::open(store.clone(), &format!("{TRAJ}/box")).unwrap();
        assert!(
            group.attributes().get("vectors").is_none(),
            "the attribute form is withdrawn"
        );
        let seq = open_sequence(&store);
        assert_eq!(seq.box_at(1).unwrap().unwrap().matrix()[0][0], 10.5);
        assert_eq!(seq.box_at(0).unwrap().unwrap().matrix()[0][0], 10.0);
    }

    /// A `step` key on frame.meta does not mint `trajectory/meta/step`.
    #[test]
    fn frame_meta_step_is_not_a_per_step_meta_array() {
        let dir = TempDir::new().unwrap();
        let store = storage_in(&dir);
        let mut frame = atoms_frame(&[1.0]);
        frame.meta.insert("step", MetaValue::I64(0));
        write_all(&store, &[frame]);

        assert!(
            Array::open(store.clone(), &format!("{TRAJ}/meta/step")).is_err(),
            "the commit marker is trajectory/step, not meta/step"
        );
        let mut seq = open_sequence(&store);
        assert!(
            frame_at(&mut seq, 0).meta.get("step").is_none(),
            "layout-owned keys are not echoed back as frame meta"
        );
    }

    /// A block that never changes costs one update — and, being regular
    /// (one update at ordinal 0), no index arrays at all: the group hints
    /// carry it and every frame resolves to update 0.
    #[test]
    fn a_constant_block_costs_one_update_and_no_index_arrays() {
        const STEPS: usize = 5;
        let dir = TempDir::new().unwrap();
        let store = storage_in(&dir);
        let frames: Vec<Frame> = (0..STEPS)
            .map(|step| {
                let mut frame = atoms_frame(&[step as f64, step as f64 + 0.5]);
                frame.insert(BONDS, block_with(I, uint_column(&[0, 1])));
                frame
            })
            .collect();
        write_all(&store, &frames);

        let group = Group::open(store.clone(), &format!("{TRAJ}/{BONDS}")).unwrap();
        assert_eq!(group.attributes()["uniform_rows"], 2);
        assert_eq!(group.attributes()["dense_updates"], true);
        for name in ["step_index", "offset"] {
            assert!(
                Array::open(store.clone(), &format!("{TRAJ}/{BONDS}/{name}")).is_err(),
                "a regular block writes no {name} array"
            );
        }
        let rows = Array::open(store.clone(), &format!("{TRAJ}/{BONDS}/{I}"))
            .unwrap()
            .shape()[0];
        assert_eq!(rows, 2, "one update's rows, not one per frame");
        let seq = open_sequence(&store);
        for index in 0..STEPS as u64 {
            assert_eq!(seq.block_update_at(BONDS, index).unwrap(), Some(0));
            assert_eq!(
                seq.frame(index)
                    .unwrap()
                    .unwrap()
                    .get(BONDS)
                    .unwrap()
                    .n_rows(),
                Some(2)
            );
        }
    }

    /// A block that changes every step is regular too — update `i` at ordinal
    /// `i` with a fixed row count — so it costs no index arrays either; a
    /// ragged run is what materializes `step_index` and `offset`.
    #[test]
    fn a_block_changing_every_step_is_regular_and_a_ragged_one_is_indexed() {
        let dir = TempDir::new().unwrap();
        let store = storage_in(&dir);
        let frames: Vec<Frame> = (0..5)
            .map(|i| atoms_frame(&[i as f64, i as f64 + 0.5]))
            .collect();
        write_all(&store, &frames);
        assert!(Array::open(store.clone(), &format!("{TRAJ}/{ATOMS}/step_index")).is_err());
        let seq = open_sequence(&store);
        for index in 0..5u64 {
            assert_eq!(seq.block_update_at(ATOMS, index).unwrap(), Some(index));
        }

        let dir = TempDir::new().unwrap();
        let store = storage_in(&dir);
        write_all(&store, &ragged_frames());
        assert_eq!(
            u64_array(&store, &format!("{TRAJ}/{ATOMS}/step_index")),
            vec![0, 1, 2]
        );
        assert_eq!(
            u64_array(&store, &format!("{TRAJ}/{ATOMS}/offset")),
            vec![0, 3, 8, 12]
        );
    }

    /// A frame whose rows straddle an inner chunk boundary reads back bit
    /// exact (ac-014).
    ///
    /// Frames are **not** rounded up to whole chunks (Design § 1: "a frame may
    /// span chunks"), which is what lets a million-atom frame work at all. With
    /// `R = 4` and three-row frames, frame 1 occupies rows 3..6 and therefore
    /// spans the boundary at row 4.
    #[test]
    fn a_frame_straddling_an_inner_chunk_boundary_reads_back_bit_exact() {
        let dir = TempDir::new().unwrap();
        let store = storage_in(&dir);
        let frames = vec![
            atoms_frame(&[1.0, 2.0, 3.0]),
            atoms_frame(&[4.0, 5.0, 6.0]),
            atoms_frame(&[7.0, 8.0, 9.0]),
        ];
        write_all_with(&store, &frames, 4, 2);

        let mut seq = open_sequence(&store);
        assert_eq!(
            atoms_x(&frame_at(&mut seq, 1)),
            vec![4.0, 5.0, 6.0],
            "the frame at rows 3..6 crosses the chunk boundary at row 4"
        );
    }

    /// The worked example, on disk: a 3000-atom `f64` xyz block derives a
    /// **frame-aligned** inner chunk of exactly one frame and a shard of
    /// 3728 such chunks.
    ///
    /// Hard-coded from the formulas, not from a measurement: row = 3 * 8 =
    /// 24 B; the byte floor is `16 KiB / 24 = 682` rows, which 3000 exceeds,
    /// so `R = 3000` (one frame); chunk = 72 000 B;
    /// `k = clamp(floor(256 MiB / 72 000), 1, 4096) = 3728`; shard =
    /// 3000 * 3728 = 11 184 000 rows.
    #[test]
    fn the_worked_example_derives_one_frame_per_chunk_and_k_3728() {
        const ATOM_COUNT: usize = 3000;
        const ROWS_PER_CHUNK: u64 = 3000;
        const CHUNKS_PER_SHARD: u64 = 3728;
        const ROWS_PER_SHARD: u64 = 11_184_000;

        let dir = TempDir::new().unwrap();
        let store = storage_in(&dir);
        let values: Vec<f64> = (0..ATOM_COUNT * 3).map(|i| i as f64 * 0.5).collect();
        let mut frame = Frame::new();
        frame.insert(
            ATOMS,
            block_with(
                "xyz",
                Column::from(ArrayD::from_shape_vec(vec![ATOM_COUNT, 3], values).unwrap()),
            ),
        );
        write_all(&store, &[frame]);

        let (shard, inner) = extents(&store, &format!("{TRAJ}/{ATOMS}/xyz"));
        assert_eq!(
            inner,
            vec![ROWS_PER_CHUNK, 3],
            "R = one 3000-row frame (the 16 KiB floor is 682 rows), trailing axis whole"
        );
        assert_eq!(
            shard,
            vec![ROWS_PER_SHARD, 3],
            "the shard spans R * k = 11 184 000 rows"
        );
        assert_eq!(
            shard[0] / inner[0],
            CHUNKS_PER_SHARD,
            "k = clamp(floor(256 MiB / 72 000 B chunk), 1, 4096) = 3728"
        );
    }

    /// A small frame is rounded up to the byte floor in whole frames: 100
    /// `f64` rows (800 B) need 21 frames to pass 16 KiB, so `R = 2100`.
    #[test]
    fn a_small_frame_is_chunked_in_whole_frames_up_to_the_byte_floor() {
        let dir = TempDir::new().unwrap();
        let store = storage_in(&dir);
        let values: Vec<f64> = (0..100).map(|i| i as f64).collect();
        write_all(&store, &[atoms_frame(&values)]);
        let (_, inner) = extents(&store, &format!("{TRAJ}/{ATOMS}/{X}"));
        assert_eq!(inner, vec![2100], "ceil(2048 / 100) * 100 = 2100 rows");
    }

    /// The dense per-frame arrays — written once a series stops being
    /// regular — share one small extent whatever the frames look like: 1024
    /// rows per inner chunk, 256 chunks per shard.
    #[test]
    fn dense_arrays_take_the_dense_extents() {
        let dir = TempDir::new().unwrap();
        let store = storage_in(&dir);
        let frames = ragged_frames();
        let mut writer = MrecWriter::create_in_storage(
            store.clone(),
            SequenceSchema::from_frames(&frames).unwrap(),
        )
        .unwrap();
        // Steps 0, 1, 5: not a progression, so `step` becomes an array.
        for (frame, step) in frames.iter().zip([0i64, 1, 5]) {
            writer.append_at(frame, step, None).unwrap();
        }
        writer.close().unwrap();
        for path in [
            format!("{TRAJ}/step"),
            format!("{TRAJ}/{ATOMS}/offset"),
            format!("{TRAJ}/{ATOMS}/step_index"),
        ] {
            let (shard, inner) = extents(&store, &path);
            assert_eq!(inner, vec![1024], "{path}");
            assert_eq!(shard, vec![1024 * 256], "{path}");
        }
        assert_eq!(i64_array(&store, &format!("{TRAJ}/step")), vec![0, 1, 5]);
        let seq = open_sequence(&store);
        assert_eq!(seq.steps(), &[0, 1, 5]);
    }

    // =======================================================================
    // B. Schema — derived union, enforcement at append, reserved names
    // =======================================================================

    /// A union mint over heterogeneous frames writes a store in which every
    /// block **carries forward** from its latest update: a frame reads back
    /// what it presented plus whatever earlier frames left standing.
    ///
    /// `from_frames` unions the blocks, per-section `step_index` records when
    /// each one changed, and omission means "unchanged" — the reading that
    /// makes a topology written once cost one entry.
    #[test]
    fn a_union_mint_carries_each_block_forward_from_its_latest_update() {
        let dir = TempDir::new().unwrap();
        let store = storage_in(&dir);

        let plain = atoms_frame(&[1.0, 2.0]);
        let mut bonded = atoms_frame(&[10.0, 11.0]);
        bonded.insert(BONDS, block_with(I, uint_column(&[0, 1])));
        let mut charged = atoms_frame(&[20.0, 21.0]);
        charged.insert("charges", block_with("q", float_column(&[-0.5, 0.5])));
        let frames = vec![plain, bonded, charged];
        write_all(&store, &frames);

        let mut seq = open_sequence(&store);
        let step0 = frame_at(&mut seq, 0);
        assert_eq!(
            step0.len(),
            1,
            "step 0 presented only atoms; nothing to carry"
        );
        assert_eq!(atoms_x(&step0), vec![1.0, 2.0]);

        let step1 = frame_at(&mut seq, 1);
        assert_eq!(step1.len(), 2, "step 1 presented atoms and bonds");
        assert!(step1.get(BONDS).is_some());
        assert!(step1.get("charges").is_none());

        let step2 = frame_at(&mut seq, 2);
        assert_eq!(
            step2.len(),
            3,
            "step 2 presented atoms and charges and carries bonds forward"
        );
        assert!(step2.get("charges").is_some());
        assert_eq!(
            step2.get(BONDS).expect("bonds carried forward").n_rows(),
            Some(2)
        );
        assert_eq!(
            u64_array(&store, &format!("{TRAJ}/{BONDS}/step_index")),
            vec![1],
            "a block omitted after it appeared earns no update"
        );
    }

    /// The three presence states, on one store: omitted = carried forward,
    /// zero rows = present and empty, no entry yet = absent.
    #[test]
    fn omitted_zero_row_and_unseen_blocks_read_back_as_carried_empty_and_absent() {
        let dir = TempDir::new().unwrap();
        let store = storage_in(&dir);

        let mut with_bonds = atoms_frame(&[1.0]);
        with_bonds.insert(BONDS, block_with(I, uint_column(&[0, 1])));
        let omits_bonds = atoms_frame(&[2.0]);
        let mut empties_bonds = atoms_frame(&[3.0]);
        empties_bonds.insert(BONDS, block_with(I, uint_column(&[])));
        let omits_again = atoms_frame(&[4.0]);
        // Frame 0 has no bonds yet: absent there.
        let frames = vec![
            atoms_frame(&[0.5]),
            with_bonds,
            omits_bonds,
            empties_bonds,
            omits_again,
        ];
        write_all(&store, &frames);

        let mut seq = open_sequence(&store);
        assert!(
            frame_at(&mut seq, 0).get(BONDS).is_none(),
            "no update at or before frame 0: absent"
        );
        assert_eq!(frame_at(&mut seq, 1).get(BONDS).unwrap().n_rows(), Some(2));
        assert_eq!(
            frame_at(&mut seq, 2).get(BONDS).unwrap().n_rows(),
            Some(2),
            "an omitted block carries forward"
        );
        let empty = frame_at(&mut seq, 3);
        let bonds = empty.get(BONDS).expect("a zero-row update is present");
        assert_eq!(bonds.n_rows(), Some(0), "and empty");
        assert!(bonds.get(I).is_some(), "with its declared columns");
        assert_eq!(
            frame_at(&mut seq, 4).get(BONDS).unwrap().n_rows(),
            Some(0),
            "an omission after the empty update carries the empty block forward"
        );
        assert_eq!(
            u64_array(&store, &format!("{TRAJ}/{BONDS}/step_index")),
            vec![1, 3]
        );
        assert_eq!(
            u64_array(&store, &format!("{TRAJ}/{BONDS}/offset")),
            vec![0, 2, 2]
        );
    }

    /// The same column arriving with two dtypes is a mint-time `Err` naming
    /// the column and both values (ac-020).
    ///
    /// Both values are spelled with the schema dtype tag — the same vocabulary
    /// the store is written in, which for the domain aliases is molrec's
    /// concrete-width spelling (`f64`/`u64`), not [`DType`]'s alias name.
    #[test]
    fn from_frames_errs_naming_the_column_and_both_dtypes() {
        let mut as_float = Frame::new();
        as_float.insert(ATOMS, block_with(PROBE, float_column(&[1.0, 2.0])));
        let mut as_uint = Frame::new();
        as_uint.insert(ATOMS, block_with(PROBE, uint_column(&[1, 2])));
        let frames = vec![as_float, as_uint];

        let message = SequenceSchema::from_frames(&frames)
            .expect_err("a conflicting dtype must be rejected at mint")
            .to_string();
        assert!(message.contains(PROBE), "must name the column: {message}");
        assert!(
            message.contains("f64"),
            "must carry the first dtype (schema tag): {message}"
        );
        assert!(
            message.contains("u64"),
            "must carry the second dtype (schema tag): {message}"
        );
    }

    /// The same column arriving with two trailing shapes is a mint-time `Err`
    /// naming the column and both values (ac-020).
    #[test]
    fn from_frames_errs_naming_the_column_and_both_trailing_shapes() {
        let shaped = |trailing: usize| {
            let mut frame = Frame::new();
            frame.insert(
                ATOMS,
                block_with(
                    "pos",
                    Column::from(
                        ArrayD::from_shape_vec(vec![2, trailing], vec![0.5; 2 * trailing]).unwrap(),
                    ),
                ),
            );
            frame
        };
        let frames = vec![shaped(3), shaped(9)];

        let message = SequenceSchema::from_frames(&frames)
            .expect_err("a conflicting trailing shape must be rejected at mint")
            .to_string();
        assert!(message.contains("pos"), "must name the column: {message}");
        assert!(
            message.contains('3'),
            "must carry the first shape: {message}"
        );
        assert!(
            message.contains('9'),
            "must carry the second shape: {message}"
        );
    }

    /// Appending a block the schema never declared is an `Err` naming it
    /// (ac-020).
    #[test]
    fn an_undeclared_block_errs_at_append_naming_it() {
        let dir = TempDir::new().unwrap();
        let store = storage_in(&dir);
        let declared = atoms_frame(&[1.0, 2.0]);
        let schema = SequenceSchema::from_frames(std::slice::from_ref(&declared)).unwrap();
        let mut writer = MrecWriter::create_in_storage(store, schema).unwrap();

        let mut extra = atoms_frame(&[3.0, 4.0]);
        extra.insert(BONDS, block_with(I, uint_column(&[0, 1])));
        let message = writer
            .append(&extra)
            .expect_err("an undeclared block must not be written")
            .to_string();
        assert!(message.contains(BONDS), "must name the block: {message}");
    }

    /// Appending a column the schema never declared is an `Err` naming it
    /// (ac-020) — no mid-run schema extension, decision 6.
    #[test]
    fn an_undeclared_column_errs_at_append_naming_it() {
        let dir = TempDir::new().unwrap();
        let store = storage_in(&dir);
        let schema =
            SequenceSchema::from_frames(std::slice::from_ref(&atoms_frame(&[1.0, 2.0]))).unwrap();
        let mut writer = MrecWriter::create_in_storage(store, schema).unwrap();

        let mut wider = Block::new();
        wider.insert(X, float_column(&[3.0, 4.0]), None).unwrap();
        wider.insert("fx", float_column(&[0.0, 0.0]), None).unwrap();
        let mut frame = Frame::new();
        frame.insert(ATOMS, wider);
        let message = writer
            .append(&frame)
            .expect_err("an undeclared column must not be written")
            .to_string();
        assert!(message.contains("fx"), "must name the column: {message}");
    }

    /// Appending a declared column at a different dtype is an `Err` naming it
    /// (ac-020).
    #[test]
    fn a_changed_dtype_errs_at_append_naming_the_column() {
        let dir = TempDir::new().unwrap();
        let store = storage_in(&dir);
        let mut declared = Frame::new();
        declared.insert(ATOMS, block_with(PROBE, float_column(&[1.0, 2.0])));
        let schema = SequenceSchema::from_frames(std::slice::from_ref(&declared)).unwrap();
        let mut writer = MrecWriter::create_in_storage(store, schema).unwrap();

        let mut frame = Frame::new();
        frame.insert(ATOMS, block_with(PROBE, uint_column(&[3, 4])));
        let message = writer
            .append(&frame)
            .expect_err("a dtype change must not be written")
            .to_string();
        assert!(message.contains(PROBE), "must name the column: {message}");
    }

    /// Appending a declared column at a different trailing shape is an `Err`
    /// naming it (ac-020).
    #[test]
    fn a_changed_trailing_shape_errs_at_append_naming_the_column() {
        let dir = TempDir::new().unwrap();
        let store = storage_in(&dir);
        let mut declared = Frame::new();
        declared.insert(
            ATOMS,
            block_with(
                "pos",
                Column::from(ArrayD::from_shape_vec(vec![2, 3], vec![0.5; 6]).unwrap()),
            ),
        );
        let schema = SequenceSchema::from_frames(std::slice::from_ref(&declared)).unwrap();
        let mut writer = MrecWriter::create_in_storage(store, schema).unwrap();

        let mut widened = Frame::new();
        widened.insert(
            ATOMS,
            block_with(
                "pos",
                Column::from(ArrayD::from_shape_vec(vec![2, 9], vec![0.5; 18]).unwrap()),
            ),
        );
        let message = writer
            .append(&widened)
            .expect_err("a trailing shape change must not be written")
            .to_string();
        assert!(message.contains("pos"), "must name the column: {message}");
    }

    /// A frame presenting a strict subset of the declared union is accepted —
    /// that *is* the sparse expression (ac-020).
    #[test]
    fn a_frame_presenting_a_subset_of_the_declared_blocks_is_accepted() {
        let dir = TempDir::new().unwrap();
        let store = storage_in(&dir);
        let mut bonded = atoms_frame(&[10.0, 11.0]);
        bonded.insert(BONDS, block_with(I, uint_column(&[0, 1])));
        let schema = SequenceSchema::from_frames(&[atoms_frame(&[1.0, 2.0]), bonded]).unwrap();
        let mut writer = MrecWriter::create_in_storage(store, schema).unwrap();

        writer
            .append(&atoms_frame(&[1.0, 2.0]))
            .expect("a frame carrying only atoms is a legal subset of the union");
        writer.close().unwrap();
    }

    /// `trajectory/`'s reserved child names are refused when the schema is
    /// **minted**, not at the first append (ac-020).
    ///
    /// Minting is the only door: a schema that cannot be minted cannot reach
    /// `create`, so this is also why no test hands `create` a reserved-name
    /// schema — the type system makes that unreachable.
    #[test]
    fn a_reserved_block_name_is_rejected_at_mint() {
        for reserved in ["step", "time", "meta", "box"] {
            let mut frame = Frame::new();
            frame.insert(reserved, block_with(X, float_column(&[1.0])));
            let message = SequenceSchema::from_frames(std::slice::from_ref(&frame))
                .expect_err("a reserved block name must be rejected at mint")
                .to_string();
            assert!(
                message.contains(reserved),
                "must name the reserved block {reserved}: {message}"
            );
        }
    }

    /// A block's reserved column names (`offset`, `step_index`) are refused
    /// at mint too (ac-020).
    #[test]
    fn a_reserved_column_name_is_rejected_at_mint() {
        for reserved in ["offset", "step_index"] {
            let mut frame = Frame::new();
            frame.insert(ATOMS, block_with(reserved, uint_column(&[0, 1])));
            let message = SequenceSchema::from_frames(std::slice::from_ref(&frame))
                .expect_err("a reserved column name must be rejected at mint")
                .to_string();
            assert!(
                message.contains(reserved),
                "must name the reserved column {reserved}: {message}"
            );
        }
    }

    // =======================================================================
    // C. Per-step meta — typed, exact, never implicitly filled
    // =======================================================================

    /// One value of every [`MetaValue`] variant, at range extremes where the
    /// variant has any: a per-step meta array that silently narrowed a width
    /// would come back changed.
    fn meta_variants() -> Vec<MetaValue> {
        vec![
            MetaValue::Bool(true),
            MetaValue::I32(i32::MIN),
            MetaValue::I64(i64::MIN),
            MetaValue::U32(u32::MAX),
            MetaValue::U64(u64::MAX),
            MetaValue::F64(f64::MIN_POSITIVE),
            MetaValue::String("gamma-phase".to_string()),
            MetaValue::Bool3([true, false, true]),
            MetaValue::I32x3([i32::MIN, 0, i32::MAX]),
            MetaValue::I64x3([i64::MIN, 0, i64::MAX]),
            MetaValue::U32x3([0, 1, u32::MAX]),
            MetaValue::U64x3([0, 1, u64::MAX]),
            MetaValue::F64x3([1.0, -0.5, f64::MAX]),
            MetaValue::F64x6([1.0, 2.0, 3.0, -4.0, 5.5, 6.25]),
            MetaValue::F64x9([1.0, 2.0, 3.0, -4.0, 5.5, 6.25, 7.0, -8.0, 9.5]),
            MetaValue::Json(serde_json::json!({"basis": "def2-TZVP", "scf": [1, 2, 3]})),
        ]
    }

    /// Every `MetaValue` variant round-trips bit exact through
    /// `trajectory/meta/<key>`, and the array carries its `dtype()` tag in
    /// `meta_dtype` (ac-019).
    ///
    /// The dtype travels with the array rather than with a convention, which
    /// is what makes exactness free (decision 8). Written for two steps so the
    /// array is a real per-step array and not a scalar attribute.
    #[test]
    fn column_dtype_tags_are_molrec_names() {
        use super::{dtype_from_tag, dtype_tag};
        // The three domain aliases are written under molrec's concrete-width
        // spelling so a schema validates against molrec's dtype enum.
        assert_eq!(dtype_tag(DType::Float), "f64");
        assert_eq!(dtype_tag(DType::Int), "i32");
        assert_eq!(dtype_tag(DType::Uint), "u64");
        assert_eq!(dtype_from_tag("f64").unwrap(), DType::Float);
        assert_eq!(dtype_from_tag("i32").unwrap(), DType::Int);
        assert_eq!(dtype_from_tag("u64").unwrap(), DType::Uint);
        // The domain alias names are not tags.
        for alias in ["float", "int", "uint"] {
            assert!(dtype_from_tag(alias).is_err(), "{alias}");
        }
    }

    #[test]
    fn every_meta_variant_round_trips_bit_exact_with_its_dtype_tag() {
        const KEY: &str = "probe";

        for value in meta_variants() {
            let dir = TempDir::new().unwrap();
            let store = storage_in(&dir);
            let frames: Vec<Frame> = [1.0f64, 2.0]
                .iter()
                .map(|v| {
                    let mut frame = atoms_frame(&[*v]);
                    frame.meta.insert(KEY, value.clone());
                    frame
                })
                .collect();
            write_all(&store, &frames);

            let array = Array::open(store.clone(), &format!("{TRAJ}/meta/{KEY}"))
                .expect("a declared meta key is a per-step array");
            assert_eq!(
                array.attributes().get("meta_dtype"),
                Some(&serde_json::json!(value.dtype())),
                "the array must carry the {} tag",
                value.dtype()
            );

            let mut seq = open_sequence(&store);
            for step in 0..2u64 {
                assert_eq!(
                    frame_at(&mut seq, step).meta.get(KEY),
                    Some(&value),
                    "{} must survive step {step} bit exact",
                    value.dtype()
                );
            }
        }
    }

    /// Omitting a declared meta key that has no declared fill is an `Err`
    /// naming the key — there is no implicit NaN (ac-019).
    #[test]
    fn omitting_a_declared_meta_key_without_a_fill_errs_naming_it() {
        const KEY: &str = "temperature";

        let dir = TempDir::new().unwrap();
        let store = storage_in(&dir);
        let mut declaring = atoms_frame(&[1.0]);
        declaring.meta.insert(KEY, MetaValue::F64(300.0));
        let schema = SequenceSchema::from_frames(std::slice::from_ref(&declaring)).unwrap();
        let mut writer = MrecWriter::create_in_storage(store, schema).unwrap();
        writer.append(&declaring).unwrap();

        let message = writer
            .append(&atoms_frame(&[2.0]))
            .expect_err("a declared meta key with no fill may not be silently filled")
            .to_string();
        assert!(message.contains(KEY), "must name the key: {message}");
    }

    /// A meta key declared with an explicit fill writes that fill for the
    /// steps that omit it, and leaves the steps that carry a value alone
    /// (ac-019).
    #[test]
    fn a_declared_fill_is_written_for_an_omitted_meta_key() {
        const KEY: &str = "temperature";
        const FILL: f64 = -1.0;

        let dir = TempDir::new().unwrap();
        let store = storage_in(&dir);
        let mut declaring = atoms_frame(&[1.0]);
        declaring.meta.insert(KEY, MetaValue::F64(300.0));
        let mut schema = SequenceSchema::from_frames(std::slice::from_ref(&declaring)).unwrap();
        schema
            .declare_meta(KEY, "f64", Some(MetaValue::F64(FILL)))
            .unwrap();

        let mut writer = MrecWriter::create_in_storage(store.clone(), schema).unwrap();
        writer.append(&declaring).unwrap();
        writer.append(&atoms_frame(&[2.0])).unwrap();
        writer.close().unwrap();

        let mut seq = open_sequence(&store);
        assert_eq!(
            frame_at(&mut seq, 1).meta.get(KEY),
            Some(&MetaValue::F64(FILL)),
            "the omitting step reads the declared fill, not NaN"
        );
        assert_eq!(
            frame_at(&mut seq, 0).meta.get(KEY),
            Some(&MetaValue::F64(300.0)),
            "and the step that carried a value keeps it"
        );
    }

    /// A NaN fill is pinned as `"NaN"` and lands as NaN; a pin whose fill a
    /// writer lost to `null` is refused, not read as a run without a fill.
    #[test]
    fn a_nan_fill_is_storable_and_a_null_fill_is_refused() {
        const KEY: &str = "temperature";
        let dir = TempDir::new().unwrap();
        let store = storage_in(&dir);
        let mut schema =
            SequenceSchema::from_frames(std::slice::from_ref(&atoms_frame(&[1.0]))).unwrap();
        schema
            .declare_meta(KEY, "f64", Some(MetaValue::F64(f64::NAN)))
            .unwrap();
        schema
            .declare_meta("count", "u64", Some(MetaValue::U64(u64::MAX)))
            .unwrap();
        let mut writer = MrecWriter::create_in_storage(store.clone(), schema).unwrap();
        writer.append(&atoms_frame(&[1.0])).unwrap();
        writer.close().unwrap();

        let pin = Group::open(store.clone(), TRAJ).unwrap().attributes()[SCHEMA_ATTRIBUTE].clone();
        assert_eq!(pin["meta"][KEY]["fill"], serde_json::json!("NaN"));
        assert_eq!(
            pin["meta"]["count"]["fill"],
            serde_json::json!("18446744073709551615")
        );
        let mut seq = open_sequence(&store);
        let frame = frame_at(&mut seq, 0);
        assert!(frame.meta.get(KEY).unwrap().as_f64().unwrap().is_nan());
        assert_eq!(frame.meta.get("count"), Some(&MetaValue::U64(u64::MAX)));

        // Tamper: a `null` fill.
        let mut group = Group::open(store.clone(), TRAJ).unwrap();
        group.attributes_mut()[SCHEMA_ATTRIBUTE]["meta"][KEY]["fill"] = serde_json::Value::Null;
        group.store_metadata().unwrap();
        let err = MrecReader::from_storage(store.clone())
            .err()
            .unwrap()
            .to_string();
        assert!(err.contains(KEY), "{err}");
        assert!(MrecWriter::from_storage(store).is_err());
    }

    /// Declared meta keys read back in `declare_meta` order, not the order the
    /// frame inserted them and not alphabetical order.
    #[test]
    fn declared_meta_keys_read_back_in_declaration_order() {
        let dir = TempDir::new().unwrap();
        let store = storage_in(&dir);
        let mut schema = SequenceSchema::new();
        schema.declare_column(ATOMS, X, DType::Float, &[]).unwrap();
        schema.declare_meta("zeta", "f64", None).unwrap();
        schema.declare_meta("alpha", "f64", None).unwrap();
        schema.declare_meta("mu", "f64", None).unwrap();

        let mut frame = atoms_frame(&[1.0]);
        frame.meta.insert("mu", MetaValue::F64(3.0));
        frame.meta.insert("zeta", MetaValue::F64(1.0));
        frame.meta.insert("alpha", MetaValue::F64(2.0));

        let mut writer = MrecWriter::create_in_storage(store.clone(), schema).unwrap();
        writer.append(&frame).unwrap();
        writer.close().unwrap();

        let seq = MrecReader::from_storage(store).unwrap();
        let back = seq.frame(0).unwrap().expect("frame 0 is present");
        assert_eq!(
            back.meta.keys().map(String::as_str).collect::<Vec<_>>(),
            ["zeta", "alpha", "mu"]
        );
        assert_eq!(back.meta.get("zeta"), Some(&MetaValue::F64(1.0)));
        assert_eq!(back.meta.get("alpha"), Some(&MetaValue::F64(2.0)));
        assert_eq!(back.meta.get("mu"), Some(&MetaValue::F64(3.0)));
    }

    // =======================================================================
    // D. Lifecycle — create, open, commit, reopen, the knob window
    // =======================================================================

    /// `create` on a store that already holds a sequence is an `Err` naming
    /// the path, and the existing store is left byte-unchanged (ac-018).
    ///
    /// Never a silent overwrite: the store on that path is somebody's run.
    #[test]
    fn create_on_an_occupied_store_errs_and_changes_nothing() {
        let dir = TempDir::new().unwrap();
        let store = storage_in(&dir);
        let frames = ragged_frames();
        write_all(&store, &frames);
        let before = file_map(dir.path());

        let schema = SequenceSchema::from_frames(&frames).unwrap();
        let message = MrecWriter::create_in_storage(store, schema)
            .err()
            .expect("create must refuse an occupied store")
            .to_string();

        assert!(
            message.contains("trajectory"),
            "must name the occupied path: {message}"
        );
        assert_eq!(
            file_map(dir.path()),
            before,
            "a refused create must not have touched a byte"
        );
    }

    /// `open` refuses a store whose arrays disagree with the schema it minted,
    /// naming what differed and both values (ac-018).
    ///
    /// The store is corrupted the way a foreign writer would corrupt it — the
    /// declared column `x` is replaced by an `i64` array — because the schema
    /// pinned at `create` is exactly what `open` has to check the arrays
    /// against.
    #[test]
    fn open_errs_naming_what_differed_and_both_values() {
        let dir = TempDir::new().unwrap();
        let store = storage_in(&dir);
        write_all(&store, &ragged_frames());

        let column = format!("{TRAJ}/{ATOMS}/{X}");
        store
            .erase_prefix(&StorePrefix::new(format!("trajectory/{ATOMS}/{X}/")).unwrap())
            .unwrap();
        ArrayBuilder::new(vec![12], vec![12], data_type::int64(), 0i64)
            .build(store.clone(), &column)
            .unwrap()
            .store_metadata()
            .unwrap();

        let message = MrecWriter::from_storage(store)
            .err()
            .expect("open must refuse a store that no longer matches its schema")
            .to_string();
        assert!(
            message.contains("sequence schema mismatch"),
            "the pinned message shape: {message}"
        );
        assert!(
            message.contains(ATOMS),
            "must name what differed: {message}"
        );
        assert!(
            message.contains("expected") && message.contains("found"),
            "must carry both values: {message}"
        );
        assert!(
            message.contains("f64"),
            "must carry the declared dtype (schema tag): {message}"
        );
        assert!(
            message.contains(&DType::I64.to_string()),
            "must carry the dtype found on disk: {message}"
        );
    }

    /// A store that conforms to the layout but carries no [`SCHEMA_ATTRIBUTE`]
    /// still **reads**: the read door derives the schema from the store.
    ///
    /// The attribute is the reference writer's own reopen pin
    /// (molrec `docs/spec/trajectory.md`), not a read requirement — resolving a
    /// frame needs `step_index`, `offset` and the array metadata, all of which
    /// are on disk regardless. Refusing a conforming store for a missing
    /// convenience pin locks every foreign writer out of the format.
    ///
    /// The fixture is written by the real writer and then *stripped*, so the
    /// only difference from a store this suite already reads is the one
    /// attribute. It carries a ragged block column **and** a per-step meta key,
    /// because the derivation has two halves — blocks/columns from the group
    /// tree and array metadata, meta keys from `meta/`'s children and their
    /// `meta_dtype` tags — and one half working is not the claim.
    #[test]
    fn a_conforming_store_without_the_schema_attribute_still_reads() {
        const TEMPERATURE: &str = "temperature";

        let dir = TempDir::new().unwrap();
        let store = storage_in(&dir);
        let frames: Vec<Frame> = ragged_frames()
            .into_iter()
            .enumerate()
            .map(|(step, mut frame)| {
                frame
                    .meta
                    .insert(TEMPERATURE, MetaValue::F64(300.0 + step as f64));
                frame
            })
            .collect();
        write_all(&store, &frames);
        strip_schema_attribute(&store);

        let mut seq = MrecReader::from_storage(store.clone())
            .expect("a conforming store reads without the writer's schema pin");
        assert_eq!(
            committed_len(&mut seq),
            frames.len(),
            "the derived sequence is as long as the run that was written"
        );
        for (index, frame) in frames.iter().enumerate() {
            let read = frame_at(&mut seq, index as u64);
            assert_eq!(
                atoms_x(&read),
                atoms_x(frame),
                "frame {index} must come back bit exact from a derived schema"
            );
            assert_eq!(
                read.meta.get(TEMPERATURE),
                frame.meta.get(TEMPERATURE),
                "the per-step meta key of frame {index} must survive the derivation"
            );
        }
    }

    /// The strict half: [`MrecWriter::open`] on that same stripped
    /// store still errs, naming [`SCHEMA_ATTRIBUTE`].
    ///
    /// A writer needs the *pinned* declaration — the fills for omitted meta keys
    /// and the union a later frame is checked against — and neither can be
    /// derived from data that was never written. Pinned next to the read door so
    /// relaxing the reader cannot quietly relax the writer with it.
    #[test]
    fn a_writer_reopen_still_requires_the_schema_attribute() {
        let dir = TempDir::new().unwrap();
        let store = storage_in(&dir);
        write_all(&store, &ragged_frames());
        strip_schema_attribute(&store);

        let message = MrecWriter::from_storage(store)
            .err()
            .expect("a writer may not reopen a sequence whose schema pin is gone")
            .to_string();
        assert!(
            message.contains(SCHEMA_ATTRIBUTE),
            "must name the missing pin: {message}"
        );
    }

    /// `flush()` is the commit point, and the crash-loss boundary is exactly
    /// "everything appended after it" (ac-015).
    ///
    /// `std::mem::forget` is the crash: the writer never runs another line, so
    /// what the reopened store shows is what a killed process would have left.
    /// It is also why the writer must have no `Drop` — a `Drop` that flushed
    /// would make this test's boundary unobservable, and would swallow IO
    /// errors in the bargain.
    #[test]
    fn a_flush_commits_every_frame_appended_before_it() {
        let dir = TempDir::new().unwrap();
        let store = storage_in(&dir);
        let frames: Vec<Frame> = (0..5)
            .map(|step| atoms_frame(&[step as f64, step as f64 + 0.5]))
            .collect();

        let schema = SequenceSchema::from_frames(&frames).unwrap();
        let mut writer = MrecWriter::create_in_storage(store.clone(), schema).unwrap();
        for frame in &frames[..3] {
            writer.append(frame).unwrap();
        }
        writer.flush().unwrap();
        for frame in &frames[3..] {
            writer.append(frame).unwrap();
        }
        std::mem::forget(writer);

        let mut seq = open_sequence(&store);
        assert_eq!(
            committed_len(&mut seq),
            3,
            "a flush commits every buffered row, ragged tail included"
        );
        for (index, frame) in frames[..3].iter().enumerate() {
            assert_eq!(
                atoms_x(&frame_at(&mut seq, index as u64)),
                atoms_x(frame),
                "committed frame {index} must read bit exact"
            );
        }
        assert!(
            seq.frame(3).unwrap().is_none(),
            "frames appended after the last flush are absent, not partial"
        );
    }

    /// `close(self)` then `open` then appends that carry the run past an inner
    /// chunk boundary: every frame reads bit exact and the extents are the
    /// ones creation froze (ac-016).
    #[test]
    fn a_reopened_writer_appends_across_an_inner_chunk_boundary() {
        let dir = TempDir::new().unwrap();
        let store = storage_in(&dir);
        let frames: Vec<Frame> = (0..4)
            .map(|step| atoms_frame(&[step as f64, step as f64 + 0.25, step as f64 + 0.5]))
            .collect();

        let schema = SequenceSchema::from_frames(&frames).unwrap();
        let mut writer = MrecWriter::create_in_storage(store.clone(), schema)
            .unwrap()
            .with_rows_per_chunk(4)
            .unwrap()
            .with_chunks_per_shard(2)
            .unwrap();
        for frame in &frames[..2] {
            writer.append(frame).unwrap();
        }
        writer.close().unwrap();
        let column = format!("{TRAJ}/{ATOMS}/{X}");
        let frozen = extents(&store, &column);

        let mut writer = MrecWriter::from_storage(store.clone()).unwrap();
        for frame in &frames[2..] {
            writer.append(frame).unwrap();
        }
        writer.close().unwrap();

        let mut seq = open_sequence(&store);
        assert_eq!(committed_len(&mut seq), 4);
        for (index, frame) in frames.iter().enumerate() {
            assert_eq!(
                atoms_x(&frame_at(&mut seq, index as u64)),
                atoms_x(frame),
                "frame {index} must survive the reopen bit exact"
            );
        }
        assert_eq!(
            extents(&store, &column),
            frozen,
            "a reopened writer may not re-plan the extents"
        );
    }

    /// Both knobs are legal after `create` and before the first append, and
    /// the first append freezes exactly what they asked for (decision 13).
    ///
    /// This is the knobs' whole reason to exist: `with_rows_per_chunk`'s
    /// consumer is the lifecycle tests (a boundary in a handful of frames
    /// instead of 512 KiB of them) and `with_chunks_per_shard`'s is the file
    /// count bound.
    #[test]
    fn the_knobs_set_the_extents_the_first_append_freezes() {
        const ROWS_PER_CHUNK: u64 = 4;
        const CHUNKS_PER_SHARD: u64 = 2;

        let dir = TempDir::new().unwrap();
        let store = storage_in(&dir);
        write_all_with(
            &store,
            &[atoms_frame(&[1.0, 2.0, 3.0])],
            ROWS_PER_CHUNK,
            CHUNKS_PER_SHARD,
        );

        assert_eq!(
            extents(&store, &format!("{TRAJ}/{ATOMS}/{X}")),
            (
                vec![ROWS_PER_CHUNK * CHUNKS_PER_SHARD],
                vec![ROWS_PER_CHUNK]
            ),
            "the knobs decide the frozen extents, not the byte target"
        );
    }

    /// A knob after the first append is an `Err` naming the phase: the extents
    /// are frozen and `zarrs` cannot re-plan a grid under live data
    /// (decision 3, decision 13).
    #[test]
    fn a_knob_after_the_first_append_errs_naming_the_phase() {
        let dir = TempDir::new().unwrap();
        let store = storage_in(&dir);
        let schema =
            SequenceSchema::from_frames(std::slice::from_ref(&atoms_frame(&[1.0, 2.0]))).unwrap();
        let mut writer = MrecWriter::create_in_storage(store, schema).unwrap();
        writer.append(&atoms_frame(&[1.0, 2.0])).unwrap();

        let message = writer
            .with_rows_per_chunk(8)
            .err()
            .expect("the knob window closes at the first append")
            .to_string();
        assert!(
            message.contains("append"),
            "must name the phase that closed the window: {message}"
        );
    }

    /// A store created and closed without a single append reopens as an empty
    /// sequence (the knob window's consequence: the arrays are not created
    /// until the first append, so such a store may hold only the group and its
    /// schema attributes).
    #[test]
    fn a_created_but_never_appended_store_reopens_empty() {
        let dir = TempDir::new().unwrap();
        let store = storage_in(&dir);
        let schema =
            SequenceSchema::from_frames(std::slice::from_ref(&atoms_frame(&[1.0]))).unwrap();
        MrecWriter::create_in_storage(store.clone(), schema)
            .unwrap()
            .close()
            .unwrap();

        let mut seq = open_sequence(&store);
        assert_eq!(committed_len(&mut seq), 0, "no append, no committed frame");
        assert!(seq.frame(0).unwrap().is_none());
    }

    // =======================================================================
    // E. Ordering — step is extended last, and that is observable
    // =======================================================================

    /// Rolling the `nstep` marker back to its pre-flush value hides the
    /// uncommitted frames and errors at nothing (ac-017).
    ///
    /// This is the crash window made deterministic: the marker is written
    /// last, so a crash anywhere before that leaves the data arrays long and
    /// the marker short — exactly the state this test builds.
    #[test]
    fn a_marker_rolled_back_hides_the_uncommitted_frames() {
        let dir = TempDir::new().unwrap();
        let store = storage_in(&dir);
        let frames: Vec<Frame> = (0..5)
            .map(|step| atoms_frame(&[step as f64, step as f64 + 0.5]))
            .collect();
        write_all(&store, &frames);

        let mut group = Group::open(store.clone(), TRAJ).unwrap();
        group.attributes_mut().insert("nstep".to_string(), 3.into());
        group.store_metadata().unwrap();

        let mut seq = open_sequence(&store);
        assert_eq!(
            committed_len(&mut seq),
            3,
            "len follows the commit marker, not the data arrays"
        );
        for (index, frame) in frames[..3].iter().enumerate() {
            assert_eq!(
                atoms_x(&frame_at(&mut seq, index as u64)),
                atoms_x(frame),
                "committed frame {index} still reads"
            );
        }
        assert!(
            seq.frame(3).unwrap().is_none(),
            "the uncommitted frames are invisible, and asking for one is not an error"
        );
    }

    /// A `FilesystemStore` that records the order of the keys written through
    /// it. Reads and listings are the wrapped store's; only the write order is
    /// this type's business.
    ///
    /// It records `set` and `set_partial_many` — the two doors a value reaches
    /// the disk through — and deliberately not `erase`, which is not a write
    /// of content.
    #[derive(Debug)]
    struct RecordingStorage {
        inner: FilesystemStore,
        written: Mutex<Vec<String>>,
    }

    impl RecordingStorage {
        fn new(path: &Path) -> Self {
            Self {
                inner: FilesystemStore::new(path).unwrap(),
                written: Mutex::new(Vec::new()),
            }
        }

        fn record(&self, key: &StoreKey) {
            self.written.lock().unwrap().push(key.as_str().to_string());
        }

        fn written_keys(&self) -> Vec<String> {
            self.written.lock().unwrap().clone()
        }

        fn clear(&self) {
            self.written.lock().unwrap().clear();
        }
    }

    impl ReadableStorageTraits for RecordingStorage {
        fn get(&self, key: &StoreKey) -> Result<MaybeBytes, StorageError> {
            self.inner.get(key)
        }

        fn get_partial(
            &self,
            key: &StoreKey,
            byte_range: ByteRange,
        ) -> Result<MaybeBytes, StorageError> {
            self.inner.get_partial(key, byte_range)
        }

        fn get_partial_many<'a>(
            &'a self,
            key: &StoreKey,
            byte_ranges: ByteRangeIterator<'a>,
        ) -> Result<MaybeBytesIterator<'a>, StorageError> {
            self.inner.get_partial_many(key, byte_ranges)
        }

        fn size_key(&self, key: &StoreKey) -> Result<Option<u64>, StorageError> {
            self.inner.size_key(key)
        }

        fn supports_get_partial(&self) -> bool {
            self.inner.supports_get_partial()
        }
    }

    impl ListableStorageTraits for RecordingStorage {
        fn list(&self) -> Result<StoreKeys, StorageError> {
            self.inner.list()
        }

        fn list_prefix(&self, prefix: &StorePrefix) -> Result<StoreKeys, StorageError> {
            self.inner.list_prefix(prefix)
        }

        fn list_dir(&self, prefix: &StorePrefix) -> Result<StoreKeysPrefixes, StorageError> {
            self.inner.list_dir(prefix)
        }

        fn size(&self) -> Result<u64, StorageError> {
            self.inner.size()
        }

        fn size_prefix(&self, prefix: &StorePrefix) -> Result<u64, StorageError> {
            self.inner.size_prefix(prefix)
        }
    }

    impl WritableStorageTraits for RecordingStorage {
        fn set(&self, key: &StoreKey, value: Bytes) -> Result<(), StorageError> {
            self.record(key);
            self.inner.set(key, value)
        }

        fn set_partial_many(
            &self,
            key: &StoreKey,
            offset_values: OffsetBytesIterator,
        ) -> Result<(), StorageError> {
            self.record(key);
            self.inner.set_partial_many(key, offset_values)
        }

        fn erase(&self, key: &StoreKey) -> Result<(), StorageError> {
            self.inner.erase(key)
        }

        fn erase_prefix(&self, prefix: &StorePrefix) -> Result<(), StorageError> {
            self.inner.erase_prefix(prefix)
        }

        fn supports_set_partial(&self) -> bool {
            self.inner.supports_set_partial()
        }
    }

    /// The write order is a property, not a comment: within one `flush()`
    /// the trajectory group's metadata — the `nstep` marker — is written
    /// **after** every other key (ac-017).
    ///
    /// That ordering is what makes a crash between two writes invisible — the
    /// commit marker is the last thing to move — and it is unobservable from
    /// the outside, so the observation is a recording store.
    #[test]
    fn the_marker_is_written_after_every_other_key() {
        let dir = TempDir::new().unwrap();
        let recorder = Arc::new(RecordingStorage::new(dir.path()));
        let store: ReadableWritableListableStorage = recorder.clone();

        let frame = atoms_frame(&[1.0, 2.0]);
        let schema = SequenceSchema::from_frames(std::slice::from_ref(&frame)).unwrap();
        let mut writer = MrecWriter::create_in_storage(store, schema).unwrap();
        writer.append(&frame).unwrap();
        recorder.clear();
        writer.flush().unwrap();

        let log = recorder.written_keys();
        let marker = log
            .iter()
            .rposition(|key| key == "trajectory/zarr.json")
            .unwrap_or_else(|| panic!("a flush must write the marker; wrote {log:?}"));
        assert_eq!(
            marker,
            log.len() - 1,
            "nothing may be written after the commit marker: {log:?}"
        );
        assert!(
            log[..marker]
                .iter()
                .any(|key| key.starts_with(&format!("trajectory/{ATOMS}/"))),
            "the data arrays must be written before the marker: {log:?}"
        );
    }

    // =======================================================================
    // F. Flush cadence — values never depend on it; dead bytes stay bounded
    // =======================================================================

    /// Whatever the flush cadence, the frames read back identical, and the
    /// dead bytes an eager flusher leaves behind are bounded by one
    /// re-encoded trailing chunk per flush — nothing is compacted at close,
    /// because a whole-shard rewrite is the one write that can destroy
    /// committed data on a crash.
    #[test]
    fn flush_cadence_changes_dead_bytes_but_never_what_reads_back() {
        const ROWS_PER_CHUNK: u64 = 64;
        const CHUNKS_PER_SHARD: u64 = 8;
        const FRAMES: usize = 24;
        /// One 64-row `f64` chunk, its `crc32c`, and the shard index slack.
        const CHUNK_BYTES_UPPER_BOUND: u64 = 64 * 8 + 4 + 16;

        // 24 frames of 5 rows = 120 rows: two inner chunks inside one shard.
        let frames: Vec<Frame> = (0..FRAMES)
            .map(|step| {
                let base = step as f64 * 10.0;
                atoms_frame(&[base, base + 1.0, base + 2.0, base + 3.0, base + 4.0])
            })
            .collect();
        let schema = SequenceSchema::from_frames(&frames).unwrap();

        let eager_dir = TempDir::new().unwrap();
        let eager_store = storage_in(&eager_dir);
        let mut eager = MrecWriter::create_in_storage(eager_store.clone(), schema)
            .unwrap()
            .with_rows_per_chunk(ROWS_PER_CHUNK)
            .unwrap()
            .with_chunks_per_shard(CHUNKS_PER_SHARD)
            .unwrap();
        for frame in &frames {
            eager.append(frame).unwrap();
            eager.flush().unwrap();
        }
        eager.close().unwrap();
        let column_dir = PathBuf::from("trajectory").join(ATOMS).join(X);
        let eager_bytes = chunk_bytes(&eager_dir.path().join(&column_dir));

        let lazy_dir = TempDir::new().unwrap();
        let lazy_store = storage_in(&lazy_dir);
        let mut lazy = MrecWriter::create_in_storage(
            lazy_store.clone(),
            SequenceSchema::from_frames(&frames).unwrap(),
        )
        .unwrap()
        .with_rows_per_chunk(ROWS_PER_CHUNK)
        .unwrap()
        .with_chunks_per_shard(CHUNKS_PER_SHARD)
        .unwrap()
        .with_flush_every(FRAMES as u64)
        .unwrap();
        for frame in &frames {
            lazy.append(frame).unwrap();
        }
        lazy.close().unwrap();
        let lazy_bytes = chunk_bytes(&lazy_dir.path().join(&column_dir));

        assert!(
            eager_bytes > lazy_bytes,
            "flushing every frame re-encodes the trailing chunk and leaves dead bytes: \
             {eager_bytes} B vs {lazy_bytes} B"
        );
        assert!(
            eager_bytes <= lazy_bytes + FRAMES as u64 * CHUNK_BYTES_UPPER_BOUND,
            "dead bytes are bounded by one chunk per flush: {eager_bytes} B vs {lazy_bytes} B"
        );

        let mut eager_seq = open_sequence(&eager_store);
        let mut lazy_seq = open_sequence(&lazy_store);
        assert_eq!(committed_len(&mut eager_seq), FRAMES);
        assert_eq!(committed_len(&mut lazy_seq), FRAMES);
        for index in 0..FRAMES as u64 {
            assert_eq!(
                atoms_x(&frame_at(&mut eager_seq, index)),
                atoms_x(&frame_at(&mut lazy_seq, index)),
                "frame {index} must not remember the flush cadence"
            );
        }
    }

    // =======================================================================
    // G. Write amplification — the store half of the append fast path
    // =======================================================================

    /// A steady-state flush writes about one compressed chunk plus one shard
    /// index **at the disk**, and does not grow with the shard file it lands
    /// in (ac-011, store half).
    ///
    /// This is the only thing standing between a 256 MiB shard default and a
    /// silent O(shard) rewrite per flush: `zarrs_filesystem` 0.3.12 reports
    /// `supports_set_partial() == true` while reading, patching and rewriting
    /// the whole value, so the codec-layer measurement in `zarrs_pins` is
    /// structurally blind to it. [`PositionalWriteStorage::bytes_written`] is
    /// the observation that is not.
    #[test]
    fn a_steady_state_flush_writes_a_chunk_not_the_shard_file() {
        /// Rows per frame, and rows per inner chunk: one frame is one chunk,
        /// so one flush is one chunk write.
        const ROWS: u64 = 4096;
        /// 4096 f64 rows = 32 768 B uncompressed.
        const CHUNK_BYTES: u64 = ROWS * 8;
        /// A shard extent of 256 chunks = 1 048 576 rows = 8 MiB, so all 64
        /// flushes land in shard 0 and no seal-on-complete rewrite is in play.
        const CHUNKS_PER_SHARD: u64 = 256;
        /// Two `u64` per inner chunk plus the `crc32c` checksum.
        const INDEX_BYTES: u64 = CHUNKS_PER_SHARD * 16 + 4;
        /// The dense per-step arrays (`step`, the block's `offset` and
        /// `step_index`) and the metadata each flush restates. Their cost is
        /// bounded by the number of arrays, not by the run length; 64 KiB is
        /// a hard ceiling on it, three orders below the shard file the naive
        /// path would have rewritten.
        const AUXILIARY_BYTES: u64 = 64 * 1024;
        const FLUSHES: u64 = 64;

        let dir = TempDir::new().unwrap();
        let positional = Arc::new(PositionalWriteStorage::new(dir.path()).unwrap());
        let store: ReadableWritableListableStorage = positional.clone();

        let frame_of = |step: u64| {
            atoms_frame(
                &(0..ROWS)
                    .map(|row| noise(step * ROWS + row))
                    .collect::<Vec<f64>>(),
            )
        };
        let schema = SequenceSchema::from_frames(std::slice::from_ref(&frame_of(0))).unwrap();
        let mut writer = MrecWriter::create_in_storage(store, schema)
            .unwrap()
            .with_rows_per_chunk(ROWS)
            .unwrap()
            .with_chunks_per_shard(CHUNKS_PER_SHARD)
            .unwrap();

        let mut early = 0u64;
        let mut late = 0u64;
        for step in 0..FLUSHES {
            let before = positional.bytes_written();
            writer.append(&frame_of(step)).unwrap();
            writer.flush().unwrap();
            let delta = positional.bytes_written() - before;
            if step == 3 {
                early = delta;
            }
            if step == FLUSHES - 1 {
                late = delta;
            }
        }
        let file = chunk_bytes(&dir.path().join("trajectory").join(ATOMS).join(X));
        writer.close().unwrap();

        assert!(
            late <= CHUNK_BYTES + INDEX_BYTES + AUXILIARY_BYTES,
            "a flush must cost one chunk plus one shard index, not {late} B"
        );
        assert!(
            late <= 2 * early,
            "the per-flush cost must not grow with the file: {early} B at flush 4, \
             {late} B at flush {FLUSHES}"
        );
        assert!(
            file > 16 * late,
            "the measurement is only meaningful while the file dwarfs the write: \
             {file} B file, {late} B flush"
        );
    }

    // =======================================================================
    // H. File count — bounded by bytes, not by frames
    // =======================================================================

    /// A store's file count follows its **bytes**, not its frame count
    /// (ac-023, decision 13).
    ///
    /// Every number below is hard-coded from the layout rather than measured.
    /// [`MrecWriter::with_chunks_per_shard`] exists for this test: a
    /// 256 MiB shard would swallow the whole fixture, so `k` is pressed down
    /// to 4 and the store really does span several shards. With `R = 8` rows
    /// per inner chunk, one shard spans `R * k = 32` rows, and every array
    /// here is 8 B per row (`f64` x; `i64` step; `u64` offset and
    /// `step_index`), so one shard file holds `S = 256` B.
    ///
    /// Per growth array the file count is `ceil(rows / 32)` shard files plus
    /// one `zarr.json`; the four groups (root, `meta/`, `trajectory/` and
    /// `trajectory/atoms/`) add one `zarr.json` each. A regular run has
    /// exactly one growth array: `x`. `step` is the `step_progression`
    /// attribute and the block's CSR index is implied by its hints, so
    /// neither costs a file. Eight 8-row frames are 64 rows of `x` — two
    /// full shards — 7 files; sixteen frames take `x` to 128 rows, four
    /// shards, 9 files. Doubling the frames adds two files.
    ///
    /// The counted set is **every** file under the store root, `zarr.json`
    /// included — the claim is about what lands on a filesystem, so excluding
    /// the metadata would be excluding the part that scales with arrays.
    #[test]
    fn file_count_scales_with_bytes_not_frames() {
        /// Rows one fixture frame carries, and rows one inner chunk holds:
        /// one frame is exactly one chunk of the `x` column.
        const ROWS_PER_FRAME: u64 = 8;
        const ROWS_PER_CHUNK: u64 = 8;
        /// Pressed far below the derived default so a small store spans
        /// several shards.
        const CHUNKS_PER_SHARD: u64 = 4;
        /// Bytes one row of every array in this store occupies.
        const ROW_BYTES: u64 = 8;
        /// `S`: bytes one shard file holds, `R * k * row`.
        const SHARD_BYTES: u64 = ROWS_PER_CHUNK * CHUNKS_PER_SHARD * ROW_BYTES;
        /// The growth arrays: the block's `x` alone — `step` is a progression
        /// attribute and a regular block writes no index arrays. No `time`, no
        /// `meta`, no `box` — the fixture frames carry none.
        const ARRAYS: u64 = 1;
        /// The root, `meta/`, `trajectory/` and `trajectory/atoms/`.
        const GROUPS: u64 = 4;
        const FRAMES: u64 = 8;

        /// `O(arrays)` of the bound: each array costs its own `zarr.json` and
        /// rounds its last shard up, and each group costs a `zarr.json`.
        const ARRAY_FLOOR: u64 = 2 * ARRAYS + GROUPS;
        /// Rows of `x` at `FRAMES`: 8 * 8.
        const SINGLE_ROWS: u64 = 64;
        /// The same at `2 * FRAMES`.
        const DOUBLE_ROWS: u64 = 128;
        /// 4 group `zarr.json` + 1 array `zarr.json` + 2 `x` shards.
        const SINGLE_FILES: usize = 7;
        /// `x` on four shards.
        const DOUBLE_FILES: usize = 9;

        // Every frame carries distinct values, so every step earns its own
        // `step_index` entry: a repeated block would be stored once and the
        // row counts above would not be the ones on disk.
        let storage_files = |count: u64| -> Vec<PathBuf> {
            let dir = TempDir::new().unwrap();
            let store = storage_in(&dir);
            let frames: Vec<Frame> = (0..count)
                .map(|step| {
                    atoms_frame(
                        &(0..ROWS_PER_FRAME)
                            .map(|row| noise(step * ROWS_PER_FRAME + row))
                            .collect::<Vec<f64>>(),
                    )
                })
                .collect();
            write_all_with(&store, &frames, ROWS_PER_CHUNK, CHUNKS_PER_SHARD);
            file_map(dir.path()).into_keys().collect()
        };

        let single = storage_files(FRAMES);
        let double = storage_files(2 * FRAMES);

        assert_eq!(
            single.len(),
            SINGLE_FILES,
            "{FRAMES} frames must cost {SINGLE_FILES} files: {single:?}"
        );
        assert_eq!(
            double.len(),
            DOUBLE_FILES,
            "{} frames must cost {DOUBLE_FILES} files: {double:?}",
            2 * FRAMES
        );

        let bound = |rows: u64| ((rows * ROW_BYTES) / SHARD_BYTES + ARRAY_FLOOR) as usize;
        assert!(
            single.len() <= bound(SINGLE_ROWS),
            "file count must stay under total_bytes/S + O(arrays): {} > {}",
            single.len(),
            bound(SINGLE_ROWS)
        );
        assert!(
            double.len() <= bound(DOUBLE_ROWS),
            "file count must stay under total_bytes/S + O(arrays): {} > {}",
            double.len(),
            bound(DOUBLE_ROWS)
        );
        assert!(
            double.len() < 2 * single.len(),
            "twice the frames must not be twice the files: {} then {}",
            single.len(),
            double.len()
        );
    }

    // =======================================================================
    // I. Reader — one cursor, two access shapes
    // =======================================================================

    /// `FrameIterator` over a `MrecReader` yields exactly what a `frame(i)`
    /// loop yields (ac-014).
    ///
    /// This is the payoff of dropping the `Reader` supertrait from
    /// `TrajectoryReader`: the existing generic consumers work against a
    /// store-backed sequence with no `BufRead` anywhere.
    #[test]
    fn the_frame_iterator_yields_what_frame_yields() {
        let dir = TempDir::new().unwrap();
        let store = storage_in(&dir);
        let frames = ragged_frames();
        write_all(&store, &frames);

        let mut iterated = open_sequence(&store);
        let streamed: Vec<Vec<f64>> = iterated
            .iter()
            .map(|frame| atoms_x(&frame.unwrap()))
            .collect();

        let mut indexed = open_sequence(&store);
        let random: Vec<Vec<f64>> = (0..frames.len() as u64)
            .map(|index| atoms_x(&frame_at(&mut indexed, index)))
            .collect();

        assert_eq!(streamed, random, "the iterator is the frame(i) loop");
        assert_eq!(
            streamed,
            frames.iter().map(atoms_x).collect::<Vec<Vec<f64>>>(),
            "and both are what was appended"
        );
    }

    /// `to_trajectory()` is the named lazy-to-eager conversion: the same
    /// frames, materialized.
    #[test]
    fn to_trajectory_carries_the_same_frames_as_frame() {
        let dir = TempDir::new().unwrap();
        let store = storage_in(&dir);
        let frames = ragged_frames();
        write_all(&store, &frames);

        let seq = open_sequence(&store);
        let eager = seq.to_trajectory().unwrap();

        assert_eq!(eager.frames.len(), frames.len());
        assert_eq!(
            eager.frames.iter().map(atoms_x).collect::<Vec<Vec<f64>>>(),
            frames.iter().map(atoms_x).collect::<Vec<Vec<f64>>>(),
            "the eager carrier holds the same bits as the lazy cursor"
        );
    }

    // =======================================================================
    // H. The 2026-09 contract — root/meta, cadence, hints, rollback, reads
    // =======================================================================

    /// The streaming writer mints a record, not a bare `trajectory/`: the
    /// root group and a `meta/` group exist before the first append, and the
    /// group carries nothing the producer did not ask for.
    #[test]
    fn create_writes_the_root_and_an_empty_meta_group() {
        let dir = TempDir::new().unwrap();
        let store = storage_in(&dir);
        let schema =
            SequenceSchema::from_frames(std::slice::from_ref(&atoms_frame(&[1.0]))).unwrap();
        let writer = MrecWriter::create_in_storage(store.clone(), schema).unwrap();
        assert!(dir.path().join("zarr.json").is_file(), "root group");
        let meta = Group::open(store.clone(), "/meta").expect("meta group exists");
        assert!(meta.attributes().is_empty(), "{:?}", meta.attributes());
        drop(writer);
    }

    /// `with_meta` writes the identity document the producer hands in.
    #[test]
    fn with_meta_writes_the_identity_document() {
        let dir = TempDir::new().unwrap();
        let store = storage_in(&dir);
        let schema =
            SequenceSchema::from_frames(std::slice::from_ref(&atoms_frame(&[1.0]))).unwrap();
        let mut doc = serde_json::Map::new();
        doc.insert("creator".into(), serde_json::json!({"name": "test"}));
        let writer = MrecWriter::create_in_storage(store.clone(), schema)
            .unwrap()
            .with_meta(&doc)
            .unwrap();
        drop(writer);
        let meta = Group::open(store.clone(), "/meta").unwrap();
        assert_eq!(meta.attributes()["creator"]["name"], "test");
    }

    /// The landing cadence follows the frame size: a 1000-row `f64` frame
    /// (8 KB) lands every 512 frames (4 MiB), rounded to whole chunks of
    /// three frames; `with_flush_every` overrides it.
    #[test]
    fn the_landing_cadence_is_derived_from_the_frame_size() {
        let dir = TempDir::new().unwrap();
        let store = storage_in(&dir);
        let values: Vec<f64> = (0..1000).map(|i| i as f64).collect();
        let frame = atoms_frame(&values);
        let schema = SequenceSchema::from_frames(std::slice::from_ref(&frame)).unwrap();
        let mut writer = MrecWriter::create_in_storage(store.clone(), schema.clone()).unwrap();
        writer.append(&frame).unwrap();
        // 4 MiB / 8000 B = 524, rounded up to a multiple of 3 frames per
        // 2100-row... no: 1000-row frames reach the 16 KiB floor at 3 frames
        // (3000 rows), and ceil(524 / 3) * 3 = 525.
        assert_eq!(writer.flush_every(), 525);
        assert_eq!(writer.committed(), 0, "one frame is below the cadence");

        let dir2 = TempDir::new().unwrap();
        let store2 = storage_in(&dir2);
        let mut writer = MrecWriter::create_in_storage(store2.clone(), schema)
            .unwrap()
            .with_flush_every(2)
            .unwrap();
        for _ in 0..5 {
            writer.append(&frame).unwrap();
        }
        assert_eq!(
            writer.committed(),
            4,
            "two landings of two frames, one buffered"
        );
        writer.close().unwrap();
        let mut seq = open_sequence(&store2);
        assert_eq!(committed_len(&mut seq), 5);
    }

    /// A block whose updates all carry the same row count at ordinals
    /// 0, 1, 2, … advertises `uniform_rows` and `dense_updates` and writes no
    /// index arrays; a reader resolves it arithmetically. The moment a frame
    /// breaks either rule the arrays are materialized and the hints withdrawn.
    #[test]
    fn regular_blocks_carry_hints_and_read_back_through_them() {
        let dir = TempDir::new().unwrap();
        let store = storage_in(&dir);
        let frames: Vec<Frame> = (0..5)
            .map(|i| atoms_frame(&[i as f64, i as f64 + 0.5]))
            .collect();
        write_all(&store, &frames);
        let group = Group::open(store.clone(), &format!("{TRAJ}/{ATOMS}")).unwrap();
        assert_eq!(group.attributes()["uniform_rows"], 2);
        assert_eq!(group.attributes()["dense_updates"], true);

        // There are no index arrays to decode: a reader that trusts the hints
        // resolves arithmetically off the column length alone.
        for name in ["step_index", "offset"] {
            assert!(
                Array::open(store.clone(), &format!("{TRAJ}/{ATOMS}/{name}")).is_err(),
                "a regular block writes no {name} array"
            );
        }
        let mut seq = open_sequence(&store);
        assert_eq!(atoms_x(&frame_at(&mut seq, 3)), vec![3.0, 3.5]);
        assert_eq!(seq.block_update_at(ATOMS, 4).unwrap(), Some(4));

        // A ragged run withdraws `uniform_rows`; a skipped ordinal withdraws
        // `dense_updates`.
        let dir = TempDir::new().unwrap();
        let store = storage_in(&dir);
        let mut frames = ragged_frames();
        frames.push(frames[2].clone()); // identical: no update at ordinal 3
        frames.push(atoms_frame(&[99.0]));
        write_all(&store, &frames);
        let group = Group::open(store.clone(), &format!("{TRAJ}/{ATOMS}")).unwrap();
        assert!(group.attributes().get("uniform_rows").is_none());
        assert!(group.attributes().get("dense_updates").is_none());
        let mut seq = open_sequence(&store);
        assert_eq!(
            atoms_x(&frame_at(&mut seq, 3)),
            vec![20.0, 21.0, 22.0, 23.0]
        );
        assert_eq!(atoms_x(&frame_at(&mut seq, 4)), vec![99.0]);
        assert_eq!(seq.block_update_at(ATOMS, 3).unwrap(), Some(2));
    }

    /// A crash that left every array longer than `step` is rolled back on
    /// reopen, and appending continues from the committed frame with the CSR
    /// pointer intact.
    #[test]
    fn a_reopened_writer_rolls_back_to_the_commit_marker() {
        let dir = TempDir::new().unwrap();
        let store = storage_in(&dir);
        let frames = ragged_frames();
        let mut writer = MrecWriter::create_in_storage(
            store.clone(),
            SequenceSchema::from_frames(&frames).unwrap(),
        )
        .unwrap()
        .with_flush_every(1)
        .unwrap();
        for frame in &frames {
            writer.append(frame).unwrap();
        }
        writer.close().unwrap();

        // Simulate a torn commit: `step` stays at 3 frames while the block's
        // arrays claim a fourth update of 7 rows.
        for (path, extra) in [
            (format!("{TRAJ}/{ATOMS}/{X}"), 7u64),
            (format!("{TRAJ}/{ATOMS}/offset"), 1),
            (format!("{TRAJ}/{ATOMS}/step_index"), 1),
        ] {
            let mut array = Array::open(store.clone(), &path).unwrap();
            let mut shape = array.shape().to_vec();
            shape[0] += extra;
            array.set_shape(shape).unwrap();
            array.store_metadata().unwrap();
        }
        {
            let array = Array::open(store.clone(), &format!("{TRAJ}/{ATOMS}/step_index")).unwrap();
            array
                .store_array_subset(
                    &ArraySubset::new_with_start_shape(vec![3], vec![1]).unwrap(),
                    vec![3u64],
                )
                .unwrap();
        }

        let mut seq = open_sequence(&store);
        assert_eq!(committed_len(&mut seq), 3, "the reader is bounded by step");
        assert_eq!(
            atoms_x(&frame_at(&mut seq, 2)),
            vec![20.0, 21.0, 22.0, 23.0]
        );

        let mut writer = MrecWriter::from_storage(store.clone()).unwrap();
        writer.append(&atoms_frame(&[30.0, 31.0])).unwrap();
        writer.close().unwrap();

        assert_eq!(
            u64_array(&store, &format!("{TRAJ}/{ATOMS}/offset")),
            vec![0, 3, 8, 12, 14],
            "the rolled-back CSR pointer continues from the committed total"
        );
        assert_eq!(
            u64_array(&store, &format!("{TRAJ}/{ATOMS}/step_index")),
            vec![0, 1, 2, 3]
        );
        let mut seq = open_sequence(&store);
        assert_eq!(atoms_x(&frame_at(&mut seq, 3)), vec![30.0, 31.0]);
        assert_eq!(
            atoms_x(&frame_at(&mut seq, 2)),
            vec![20.0, 21.0, 22.0, 23.0]
        );
    }

    /// `frame_columns` decodes only the named columns; the block still
    /// resolves through the same three states.
    #[test]
    fn frame_columns_reads_only_the_named_columns() {
        let dir = TempDir::new().unwrap();
        let store = storage_in(&dir);
        let mut frame = atoms_frame(&[1.0, 2.0]);
        frame
            .get_mut(ATOMS)
            .unwrap()
            .insert("y", float_column(&[10.0, 20.0]), None)
            .unwrap();
        frame.insert(BONDS, block_with(I, uint_column(&[0, 1])));
        write_all(&store, &[frame]);

        let seq = open_sequence(&store);
        let picked = seq
            .frame_columns(0, &[(ATOMS, "y")])
            .unwrap()
            .expect("frame 0 exists");
        let atoms = picked.get(ATOMS).expect("the named block is present");
        assert!(atoms.get("y").is_some());
        assert!(atoms.get(X).is_none(), "an unnamed column is not decoded");
        assert!(picked.get(BONDS).is_none(), "an unnamed block is left out");
        assert_eq!(atoms.n_rows(), Some(2));

        let err = seq
            .frame_columns(0, &[(ATOMS, "nope")])
            .unwrap_err()
            .to_string();
        assert!(err.contains("nope"), "{err}");
    }

    /// `box_at` resolves the cell through the box index and caches it.
    #[test]
    fn box_at_resolves_the_latest_cell() {
        let dir = TempDir::new().unwrap();
        let store = storage_in(&dir);
        let mut with_cell = atoms_frame(&[1.0]);
        with_cell.simbox = Some(fixed_cell());
        write_all(
            &store,
            &[atoms_frame(&[0.0]), with_cell, atoms_frame(&[2.0])],
        );
        let seq = open_sequence(&store);
        assert!(seq.box_at(0).unwrap().is_none(), "no cell yet");
        let cell = seq.box_at(2).unwrap().expect("the cell carries forward");
        assert_eq!(cell.matrix()[1][1], 11.0);
        assert!(seq.box_at(3).unwrap().is_none(), "past the commit marker");
    }

    /// A shaped block keeps its row count: any other count is refused at
    /// append, naming the block and both counts.
    #[test]
    fn a_shaped_block_with_another_row_count_is_refused_at_append() {
        let dir = TempDir::new().unwrap();
        let store = storage_in(&dir);
        let mut grid = Block::new();
        grid.insert("rho", float_column(&[1.0, 2.0, 3.0, 4.0]), None)
            .unwrap();
        grid.set_shape(&[2, 2]).unwrap();
        let mut frame = Frame::new();
        frame.insert("grid", grid);
        let schema = SequenceSchema::from_frames(std::slice::from_ref(&frame)).unwrap();
        let mut writer = MrecWriter::create_in_storage(store, schema).unwrap();
        writer.append(&frame).unwrap();
        let mut wrong = Frame::new();
        wrong.insert("grid", block_with("rho", float_column(&[1.0, 2.0])));
        let err = writer.append(&wrong).unwrap_err().to_string();
        assert!(err.contains("grid") && err.contains("4 rows"), "{err}");
    }

    /// A meta value that arrives at another width is re-read at the declared
    /// width — a JSON list becomes the declared `f64x3` — and one that cannot
    /// be is refused naming both.
    #[test]
    fn a_meta_value_at_another_width_is_read_at_the_declared_one() {
        let dir = TempDir::new().unwrap();
        let store = storage_in(&dir);
        let mut schema =
            SequenceSchema::from_frames(std::slice::from_ref(&atoms_frame(&[1.0]))).unwrap();
        schema.declare_meta("com", "f64x3", None).unwrap();
        schema.declare_meta("count", "i64", None).unwrap();
        let mut writer = MrecWriter::create_in_storage(store.clone(), schema).unwrap();
        let mut frame = atoms_frame(&[1.0]);
        frame
            .meta
            .insert("com", MetaValue::Json(serde_json::json!([1.0, 2.0, 3.0])));
        frame.meta.insert("count", MetaValue::I64(3));
        writer.append(&frame).unwrap();
        let mut wrong = atoms_frame(&[2.0]);
        wrong
            .meta
            .insert("com", MetaValue::Json(serde_json::json!([1.0, 2.0])));
        wrong.meta.insert("count", MetaValue::I64(4));
        let err = writer.append(&wrong).unwrap_err().to_string();
        assert!(err.contains("com") && err.contains("f64x3"), "{err}");
        // Declared keys are walked in declaration order, so a bad `com` masks
        // `count`. This frame's `com` fits; its `count` is an f64 under i64.
        let mut refused = atoms_frame(&[3.0]);
        refused
            .meta
            .insert("com", MetaValue::Json(serde_json::json!([4.0, 5.0, 6.0])));
        refused.meta.insert("count", MetaValue::F64(0.5));
        let err = writer.append(&refused).unwrap_err().to_string();
        assert!(err.contains("count") && err.contains("i64"), "{err}");
        writer.close().unwrap();
        let mut seq = open_sequence(&store);
        let back = frame_at(&mut seq, 0);
        assert_eq!(
            back.meta.get("com"),
            Some(&MetaValue::F64x3([1.0, 2.0, 3.0]))
        );
    }

    /// There are no narrow float tags: a per-step meta key declared with one
    /// is refused, not widened to `f64`. The vector forms share this same
    /// unknown-dtype arm of [`meta_layout`].
    #[test]
    fn declaring_a_narrow_float_meta_tag_is_refused() {
        let mut schema = SequenceSchema::new();
        for tag in ["f16", "f32"] {
            let err = schema
                .declare_meta("scale", tag, None)
                .unwrap_err()
                .to_string();
            assert!(err.contains("unknown dtype"), "accepted `{tag}`: {err}");
        }
    }

    /// A schema declared column by column is the same pin a derived one is.
    #[test]
    fn a_declared_schema_equals_the_derived_one() {
        let mut frame = atoms_frame(&[1.0, 2.0]);
        frame.insert(BONDS, block_with(I, uint_column(&[0])));
        frame.meta.insert("energy", MetaValue::F64(1.5));
        let derived = SequenceSchema::from_frames(std::slice::from_ref(&frame)).unwrap();

        let mut declared = SequenceSchema::new();
        declared
            .declare_column(ATOMS, X, DType::Float, &[])
            .unwrap();
        declared.declare_column(BONDS, I, DType::Uint, &[]).unwrap();
        declared.declare_meta("energy", "f64", None).unwrap();
        assert_eq!(declared, derived);

        // A canonical key is pinned at its vocabulary dtype.
        let err = declared
            .declare_column(ATOMS, X, DType::Int, &[])
            .unwrap_err()
            .to_string();
        assert!(err.contains("canonical") && err.contains("float"), "{err}");
        declared
            .declare_column(ATOMS, PROBE, DType::Float, &[])
            .unwrap();
        let err = declared
            .declare_column(ATOMS, PROBE, DType::Int, &[])
            .unwrap_err()
            .to_string();
        assert!(err.contains("f64") && err.contains("i32"), "{err}");
        let err = declared
            .declare_column("step", "a", DType::Float, &[])
            .unwrap_err()
            .to_string();
        assert!(err.contains("reserved"), "{err}");
        let err = declared
            .declare_meta("bad", "f128", None)
            .unwrap_err()
            .to_string();
        assert!(err.contains("f128"), "{err}");
    }

    /// Every inner chunk ends in `crc32c`, floats stay raw and everything
    /// else carries `gzip` level 1.
    #[test]
    fn inner_codecs_follow_the_width_policy() {
        let dir = TempDir::new().unwrap();
        let store = storage_in(&dir);
        let mut frame = atoms_frame(&[1.0]);
        frame.insert(BONDS, block_with(I, uint_column(&[0])));
        let mut second = atoms_frame(&[2.0]);
        second.insert(BONDS, block_with(I, uint_column(&[0])));
        let mut third = atoms_frame(&[3.0]);
        third.insert(BONDS, block_with(I, uint_column(&[0])));
        let mut writer = MrecWriter::create_in_storage(
            store.clone(),
            SequenceSchema::from_frames(&[frame.clone(), second.clone()]).unwrap(),
        )
        .unwrap();
        // Steps 0, 1, 5: not a progression (two values always are one), so
        // `step` is an array to inspect.
        writer.append_at(&frame, 0, None).unwrap();
        writer.append_at(&second, 1, None).unwrap();
        writer.append_at(&third, 5, None).unwrap();
        writer.close().unwrap();
        let inner_codecs = |path: &str| -> Vec<String> {
            let arr = Array::open(store.clone(), path).unwrap();
            let metadata = serde_json::to_value(arr.metadata()).unwrap();
            let sharding = metadata["codecs"]
                .as_array()
                .unwrap()
                .iter()
                .find(|codec| codec["name"] == "sharding_indexed")
                .unwrap()
                .clone();
            assert_eq!(sharding["configuration"]["index_location"], "start");
            sharding["configuration"]["codecs"]
                .as_array()
                .unwrap()
                .iter()
                .map(|codec| codec["name"].as_str().unwrap().to_string())
                .collect()
        };
        assert_eq!(
            inner_codecs(&format!("{TRAJ}/{ATOMS}/{X}")),
            vec!["bytes", "crc32c"]
        );
        assert_eq!(
            inner_codecs(&format!("{TRAJ}/{BONDS}/{I}")),
            vec!["bytes", "gzip", "crc32c"]
        );
        assert_eq!(
            inner_codecs(&format!("{TRAJ}/step")),
            vec!["bytes", "gzip", "crc32c"]
        );
    }

    // -- nullable columns across a run -------------------------------------

    /// The mask every nullable fixture carries over its three rows: the first
    /// row valid, the last two null. Asymmetric on purpose — a reversed mask
    /// cannot match it by accident.
    const MASK: [bool; 3] = [true, false, false];

    /// A frame whose `atoms` block carries one masked [`PROBE`] column.
    fn masked_frame(values: &[f64], validity: &[bool]) -> Frame {
        let mut block = Block::new();
        block
            .insert(
                PROBE,
                ArrayD::from_shape_vec(vec![values.len()], values.to_vec()).unwrap(),
                Some(validity.to_vec()),
            )
            .unwrap();
        let mut frame = Frame::new();
        frame.insert(ATOMS, block);
        frame
    }

    /// The mask of column [`PROBE`] as that frame came back from the store.
    fn atoms_mask(frame: &Frame) -> Option<Vec<bool>> {
        frame
            .get(ATOMS)
            .expect("the frame carries an atoms block")
            .validity(PROBE)
            .map(<[bool]>::to_vec)
    }

    /// A mask belongs to the frame it was appended with: two frames that both
    /// mask the column must both say so on the way back, not just the one
    /// whose rows happen to sit at the head of the growth array.
    #[test]
    fn every_frame_keeps_the_mask_of_its_masked_column() {
        let dir = TempDir::new().unwrap();
        let store = storage_in(&dir);
        write_all(
            &store,
            &[
                masked_frame(&[1.0, 0.0, 0.0], &MASK),
                masked_frame(&[2.0, 0.0, 0.0], &MASK),
            ],
        );

        let mut seq = open_sequence(&store);
        assert_eq!(atoms_mask(&frame_at(&mut seq, 0)), Some(MASK.to_vec()));
        assert_eq!(atoms_mask(&frame_at(&mut seq, 1)), Some(MASK.to_vec()));
    }

    /// The mask is per frame, not per column of the run: a column that is
    /// whole in one frame and holed in the next keeps both answers, so a
    /// declared-nullable column cannot make every frame look masked.
    #[test]
    fn a_column_masked_in_one_frame_only_keeps_both_answers() {
        let dir = TempDir::new().unwrap();
        let store = storage_in(&dir);
        write_all(
            &store,
            &[
                masked_frame(&[1.0, 2.0, 3.0], &[true, true, true]),
                masked_frame(&[4.0, 0.0, 0.0], &MASK),
            ],
        );

        let mut seq = open_sequence(&store);
        assert_eq!(atoms_mask(&frame_at(&mut seq, 0)), None);
        assert_eq!(atoms_mask(&frame_at(&mut seq, 1)), Some(MASK.to_vec()));
    }

    /// Whether a column may hold nulls is part of the pin, not a per-append
    /// discovery: the declaration derived from a masked frame says so.
    #[test]
    fn from_frames_declares_a_masked_column_nullable() {
        let schema = SequenceSchema::from_frames(&[masked_frame(&[1.0, 0.0, 0.0], &MASK)]).unwrap();
        assert!(schema.blocks[ATOMS].columns[PROBE].nullable);
    }

    /// And a run that never masks the column declares it plain — the flag
    /// states something, rather than being set for every column there is.
    #[test]
    fn from_frames_leaves_an_unmasked_column_not_nullable() {
        let schema = SequenceSchema::from_frames(&[atoms_frame(&[1.0, 2.0, 3.0])]).unwrap();
        assert!(!schema.blocks[ATOMS].columns[X].nullable);
    }

    /// The declaration is the union over the frames it is derived from: one
    /// frame masking the column is enough, otherwise the schema minted from a
    /// run whose first frame is whole could not describe its second.
    #[test]
    fn from_frames_unions_nullable_across_frames() {
        let schema = SequenceSchema::from_frames(&[
            masked_frame(&[1.0, 2.0, 3.0], &[true, true, true]),
            masked_frame(&[4.0, 0.0, 0.0], &MASK),
        ])
        .unwrap();
        assert!(schema.blocks[ATOMS].columns[PROBE].nullable);
    }

    /// The flag survives the pin written to `trajectory/`: a reopened writer
    /// reads its own declaration back out of the store, so a flag that only
    /// existed in memory would be lost at the first reopen.
    #[test]
    fn the_pinned_schema_carries_the_nullable_flag() {
        let dir = TempDir::new().unwrap();
        let store = storage_in(&dir);
        write_all(&store, &[masked_frame(&[1.0, 0.0, 0.0], &MASK)]);

        let group = Group::open(store.clone(), TRAJ).expect("the sequence group exists");
        let pinned: SequenceSchema = serde_json::from_value(
            group
                .attributes()
                .get(SCHEMA_ATTRIBUTE)
                .expect("the writer pins its schema")
                .clone(),
        )
        .expect("the pin deserializes");
        assert!(pinned.blocks[ATOMS].columns[PROBE].nullable);
    }

    /// `nullable` is optional in a pin: a column that omits it is plain.
    #[test]
    fn a_schema_pin_without_nullable_deserializes_as_not_nullable() {
        let pinned =
            r#"{"blocks":{"atoms":{"columns":{"probe":{"dtype":"f64","trailing":[]}}}},"meta":{}}"#;
        let schema: SequenceSchema =
            serde_json::from_str(pinned).expect("a pin without nullable deserializes");
        assert!(!schema.blocks[ATOMS].columns[PROBE].nullable);
    }

    /// A frame whose values repeat while its mask moves is a *different*
    /// frame. Change detection compares the validity masks, so the second
    /// frame earns an update of its own; without the mask in the comparison
    /// it would earn none and read back under the first frame's mask, which
    /// is the silent loss the comparison exists to close.
    #[test]
    fn a_frame_repeating_its_values_under_a_moved_mask_keeps_its_own_mask() {
        /// [`MASK`] with the middle row filled in — same values underneath.
        const MOVED: [bool; 3] = [true, true, false];
        let dir = TempDir::new().unwrap();
        let store = storage_in(&dir);
        write_all(
            &store,
            &[
                masked_frame(&[1.0, 0.0, 0.0], &MASK),
                masked_frame(&[1.0, 0.0, 0.0], &MOVED),
            ],
        );

        let mut seq = open_sequence(&store);
        assert_eq!(atoms_mask(&frame_at(&mut seq, 0)), Some(MASK.to_vec()));
        assert_eq!(atoms_mask(&frame_at(&mut seq, 1)), Some(MOVED.to_vec()));
        assert_eq!(seq.block_update_at(ATOMS, 0).unwrap(), Some(0));
        assert_eq!(
            seq.block_update_at(ATOMS, 1).unwrap(),
            Some(1),
            "a mask that moved is a change, and a change costs an update"
        );
    }

    /// The converse guard: the mask entering the comparison must not make
    /// every masked frame look new. Two frames identical in values *and*
    /// mask still collapse into the one update an unchanging block costs.
    #[test]
    fn two_frames_identical_in_values_and_mask_cost_one_update() {
        let dir = TempDir::new().unwrap();
        let store = storage_in(&dir);
        write_all(
            &store,
            &[
                masked_frame(&[1.0, 0.0, 0.0], &MASK),
                masked_frame(&[1.0, 0.0, 0.0], &MASK),
            ],
        );

        let mut seq = open_sequence(&store);
        assert_eq!(seq.block_update_at(ATOMS, 0).unwrap(), Some(0));
        assert_eq!(
            seq.block_update_at(ATOMS, 1).unwrap(),
            Some(0),
            "an unchanged block, mask included, earns no second update"
        );
        let probe = |frame: &Frame| -> Vec<f64> {
            frame
                .get(ATOMS)
                .expect("the frame carries an atoms block")
                .get(PROBE)
                .expect("the atoms block carries the probe column")
                .get::<f64>()
                .expect("the probe column arrived as f64")
                .iter()
                .copied()
                .collect()
        };
        let first = frame_at(&mut seq, 0);
        let second = frame_at(&mut seq, 1);
        assert_eq!(probe(&first), vec![1.0, 0.0, 0.0]);
        assert_eq!(probe(&second), probe(&first));
        assert_eq!(atoms_mask(&first), Some(MASK.to_vec()));
        assert_eq!(atoms_mask(&second), atoms_mask(&first));
    }

    /// `declare_nullable` is the door a hand-declared schema opts its column
    /// in through: with the call, a masked frame appends and the mask
    /// survives the round trip, exactly as a derived schema's would.
    #[test]
    fn a_hand_declared_nullable_column_accepts_a_masked_frame() {
        let dir = TempDir::new().unwrap();
        let store = storage_in(&dir);
        let mut schema = SequenceSchema::new();
        schema.declare_block(ATOMS, Some(3)).unwrap();
        schema
            .declare_column(ATOMS, PROBE, DType::Float, &[])
            .unwrap();
        schema.declare_nullable(ATOMS, PROBE).unwrap();

        let mut writer = MrecWriter::create_in_storage(store.clone(), schema).unwrap();
        writer
            .append(&masked_frame(&[1.0, 0.0, 0.0], &MASK))
            .unwrap();
        writer.close().unwrap();

        let mut seq = open_sequence(&store);
        assert_eq!(atoms_mask(&frame_at(&mut seq, 0)), Some(MASK.to_vec()));
    }

    /// Without that call the same append is refused rather than landed
    /// mask-less: the pin is a contract, and the error names the block and
    /// the column whose mask has nowhere to go.
    #[test]
    fn a_masked_frame_is_refused_by_a_non_nullable_declaration() {
        let dir = TempDir::new().unwrap();
        let store = storage_in(&dir);
        let mut schema = SequenceSchema::new();
        schema.declare_block(ATOMS, Some(3)).unwrap();
        schema
            .declare_column(ATOMS, PROBE, DType::Float, &[])
            .unwrap();

        let mut writer = MrecWriter::create_in_storage(store.clone(), schema).unwrap();
        let err = writer
            .append(&masked_frame(&[1.0, 0.0, 0.0], &MASK))
            .unwrap_err()
            .to_string();
        assert!(
            err.contains(PROBE) && err.contains(ATOMS) && err.contains("non-nullable"),
            "{err}"
        );
    }

    /// Nullability is a property of a declared column, so declaring it on a
    /// column that was never declared is an error naming the column.
    #[test]
    fn declare_nullable_on_an_undeclared_column_is_refused() {
        let mut schema = SequenceSchema::new();
        schema.declare_block(ATOMS, None).unwrap();
        let err = schema
            .declare_nullable(ATOMS, PROBE)
            .unwrap_err()
            .to_string();
        assert!(err.contains(PROBE) && err.contains(ATOMS), "{err}");
    }

    /// And on a column of a block that was never declared either — the same
    /// refusal, naming both, rather than a silently inserted declaration.
    #[test]
    fn declare_nullable_on_an_undeclared_block_is_refused() {
        let mut schema = SequenceSchema::new();
        let err = schema.declare_nullable(BONDS, I).unwrap_err().to_string();
        assert!(
            err.contains(BONDS) && err.contains(&format!("{I:?}")),
            "{err}"
        );
    }

    // -- declared precision (molrec F1) ------------------------------------

    /// The inner codec names of the sharded growth array at `path`.
    fn inner_codec_names(store: &ReadableWritableListableStorage, path: &str) -> Vec<String> {
        let arr = Array::open(store.clone(), path).unwrap();
        let metadata = serde_json::to_value(arr.metadata()).unwrap();
        metadata["codecs"]
            .as_array()
            .unwrap()
            .iter()
            .find(|codec| codec["name"] == "sharding_indexed")
            .unwrap()["configuration"]["codecs"]
            .as_array()
            .unwrap()
            .iter()
            .map(|codec| codec["name"].as_str().unwrap().to_string())
            .collect()
    }

    /// An atoms frame whose `x` declares precision `p`.
    fn precise_frame(values: &[f64], p: f64) -> Frame {
        let mut frame = atoms_frame(values);
        frame.get_mut(ATOMS).unwrap().set_precision(X, p).unwrap();
        frame
    }

    fn stored(values: &[f64], p: f64) -> Vec<f64> {
        let q = molrs::core::quantum(p).unwrap();
        values
            .iter()
            .map(|&x| molrs::core::quantize(x, q))
            .collect()
    }

    #[test]
    fn a_precision_is_pinned_and_every_frame_reads_back_rounded() {
        let dir = TempDir::new().unwrap();
        let store = storage_in(&dir);
        let p = 1e-3;
        let frames = [
            precise_frame(&[0.123_456, -1.000_49, 2.5], p),
            precise_frame(&[0.2, 7.777_7, -3.0], p),
        ];
        let schema = SequenceSchema::from_frames(&frames).unwrap();
        assert_eq!(schema.precision(ATOMS, X), Some(p));
        write_all(&store, &frames);

        let pin = Group::open(store.clone(), TRAJ).unwrap().attributes()[SCHEMA_ATTRIBUTE].clone();
        assert_eq!(
            pin["blocks"][ATOMS]["columns"][X]["precision"],
            serde_json::json!(p)
        );
        // Declared on the trajectory path in the pin only: no array attribute.
        let array = Array::open(store.clone(), &format!("{TRAJ}/{ATOMS}/{X}")).unwrap();
        assert!(array.attributes().get("precision").is_none());
        assert_eq!(
            inner_codec_names(&store, &format!("{TRAJ}/{ATOMS}/{X}")),
            ["bytes", "numcodecs.shuffle", "zstd", "crc32c"]
        );

        let mut seq = open_sequence(&store);
        for (index, frame) in frames.iter().enumerate() {
            let back = frame_at(&mut seq, index as u64);
            let presented = atoms_x(frame);
            assert_eq!(atoms_x(&back), stored(&presented, p));
            // The pinned precision travels with the column read back.
            assert_eq!(back.get(ATOMS).unwrap().precision(X), Some(p));
        }
    }

    #[test]
    fn a_change_below_half_a_quantum_carries_forward() {
        let dir = TempDir::new().unwrap();
        let store = storage_in(&dir);
        let p = 1e-3;
        let q = molrs::core::quantum(p).unwrap();
        let first = [1.0, 2.0, 3.0];
        let nudged: Vec<f64> = first.iter().map(|x| x + 0.4 * q).collect();
        write_all(
            &store,
            &[precise_frame(&first, p), precise_frame(&nudged, p)],
        );

        let mut seq = open_sequence(&store);
        assert_eq!(seq.block_update_at(ATOMS, 0).unwrap(), Some(0));
        assert_eq!(seq.block_update_at(ATOMS, 1).unwrap(), Some(0));
        assert_eq!(atoms_x(&frame_at(&mut seq, 1)), first);
    }

    #[test]
    fn a_frame_stating_another_precision_than_the_pin_is_refused() {
        let dir = TempDir::new().unwrap();
        let store = storage_in(&dir);
        let schema =
            SequenceSchema::from_frames(std::slice::from_ref(&precise_frame(&[1.0], 1e-3)))
                .unwrap();
        let mut writer = MrecWriter::create_in_storage(store.clone(), schema).unwrap();
        // No precision stated: the pin rounds it.
        writer.append(&atoms_frame(&[1.000_1])).unwrap();
        let err = writer
            .append(&precise_frame(&[1.0], 1e-2))
            .unwrap_err()
            .to_string();
        assert!(err.contains("precision") && err.contains(X), "{err}");

        // A pin without a precision refuses a frame that states one.
        let dir = TempDir::new().unwrap();
        let store = storage_in(&dir);
        let schema =
            SequenceSchema::from_frames(std::slice::from_ref(&atoms_frame(&[1.0]))).unwrap();
        let mut writer = MrecWriter::create_in_storage(store, schema).unwrap();
        assert!(writer.append(&precise_frame(&[1.0], 1e-3)).is_err());
    }

    #[test]
    fn declare_precision_refuses_what_cannot_carry_one() {
        let mut schema = SequenceSchema::new();
        schema.declare_column(ATOMS, X, DType::Float, &[]).unwrap();
        schema.declare_column(BONDS, I, DType::Uint, &[]).unwrap();
        assert!(schema.declare_precision(ATOMS, "nope", 1e-3).is_err());
        assert!(schema.declare_precision(BONDS, I, 1e-3).is_err());
        for bad in [0.0, -1e-3, f64::NAN, f64::INFINITY] {
            assert!(schema.declare_precision(ATOMS, X, bad).is_err(), "{bad}");
        }
        schema.declare_precision(ATOMS, X, 1e-3).unwrap();
        schema.declare_precision(ATOMS, X, 1e-3).unwrap();
        assert!(schema.declare_precision(ATOMS, X, 1e-2).is_err());
        assert_eq!(schema.precision(ATOMS, X), Some(1e-3));

        // Two frames declaring two precisions conflict at mint.
        let err = SequenceSchema::from_frames(&[
            precise_frame(&[1.0], 1e-3),
            precise_frame(&[1.0], 1e-2),
        ])
        .unwrap_err()
        .to_string();
        assert!(err.contains("precision"), "{err}");
    }

    #[test]
    fn the_compression_knob_replaces_the_precision_compressor_and_the_shuffle_stays() {
        let dir = TempDir::new().unwrap();
        let store = storage_in(&dir);
        let frame = precise_frame(&[1.0, 2.0], 1e-3);
        let mut writer = MrecWriter::create_in_storage(
            store.clone(),
            SequenceSchema::from_frames(std::slice::from_ref(&frame)).unwrap(),
        )
        .unwrap()
        .with_compression(super::Compression::Gzip(5))
        .unwrap();
        writer.append(&frame).unwrap();
        writer.close().unwrap();
        assert_eq!(
            inner_codec_names(&store, &format!("{TRAJ}/{ATOMS}/{X}")),
            ["bytes", "numcodecs.shuffle", "gzip", "crc32c"]
        );
    }

    /// The design's density measurement, on a small synthetic run: 3000
    /// atoms uniform in a 40 Å box, 40 frames of a 0.05 Å random walk, one
    /// inner chunk per column per frame. Prints bytes per atom per frame of
    /// the three coordinate columns, lossless and at two precisions: the
    /// shard files as written, and the chunk payload net of the fixed-size
    /// shard index (`16 B × chunks_per_shard + 4` per shard, written whole
    /// at the start of the shard, so a 40-frame run pays it unamortized).
    #[test]
    fn a_declared_precision_cuts_the_coordinate_bytes() {
        const ATOMS_N: usize = 3000;
        const FRAMES: usize = 40;
        let mut state = 0x2545_F491_4F6C_DD1Du64;
        let mut uniform = || {
            state ^= state << 13;
            state ^= state >> 7;
            state ^= state << 17;
            (state >> 11) as f64 / (1u64 << 53) as f64
        };
        let mut xyz: Vec<[f64; 3]> = (0..ATOMS_N)
            .map(|_| [40.0 * uniform(), 40.0 * uniform(), 40.0 * uniform()])
            .collect();
        let mut run = Vec::with_capacity(FRAMES);
        for _ in 0..FRAMES {
            run.push(xyz.clone());
            for atom in &mut xyz {
                for axis in atom.iter_mut() {
                    *axis += 0.05 * (2.0 * uniform() - 1.0);
                }
            }
        }
        let bytes_per_atom_frame = |precision: Option<f64>| -> (f64, f64) {
            let dir = TempDir::new().unwrap();
            let store = storage_in(&dir);
            let frames: Vec<Frame> = run
                .iter()
                .map(|positions| {
                    let mut block = Block::new();
                    for (axis, key) in ["x", "y", "z"].into_iter().enumerate() {
                        let values: Vec<f64> = positions.iter().map(|p| p[axis]).collect();
                        block.insert(key, float_column(&values), None).unwrap();
                        if let Some(p) = precision {
                            block.set_precision(key, p).unwrap();
                        }
                    }
                    let mut frame = Frame::new();
                    frame.insert(ATOMS, block);
                    frame
                })
                .collect();
            write_all(&store, &frames);
            let mut bytes = 0u64;
            let mut index = 0u64;
            for key in ["x", "y", "z"] {
                let files = chunk_files(&dir.path().join("trajectory").join(ATOMS).join(key));
                let (shard, inner) = extents(&store, &format!("{TRAJ}/{ATOMS}/{key}"));
                bytes += files.values().map(|v| v.len() as u64).sum::<u64>();
                index += files.len() as u64 * (16 * (shard[0] / inner[0]) + 4);
            }
            let per = (ATOMS_N * FRAMES) as f64;
            (bytes as f64 / per, (bytes - index) as f64 / per)
        };
        let (lossless, lossless_net) = bytes_per_atom_frame(None);
        let (milli, milli_net) = bytes_per_atom_frame(Some(1e-3));
        let (centi, centi_net) = bytes_per_atom_frame(Some(1e-2));
        eprintln!(
            "B/atom/frame (files / net of shard index): lossless {lossless:.2} / \
             {lossless_net:.2}, p=1e-3 {milli:.2} / {milli_net:.2}, p=1e-2 {centi:.2} / \
             {centi_net:.2}"
        );
        assert!(lossless_net > 23.9, "lossless is raw f64: {lossless_net}");
        assert!(milli_net < 8.0, "p=1e-3: {milli_net}");
        assert!(centi_net < 6.5, "p=1e-2: {centi_net}");
        assert!(milli < lossless / 2.5, "p=1e-3: {milli} vs {lossless}");
        assert!(centi < milli, "p=1e-2: {centi} vs {milli}");
    }

    // -- row references (molrec F4) -----------------------------------------

    /// `atoms` with `n` rows and a `refs` block whose `site` references them.
    fn referencing_frame(n: usize, site: &[u64]) -> Frame {
        let mut frame = atoms_frame(&vec![0.5; n]);
        let mut refs = block_with("site", uint_column(site));
        refs.set_target("site", ATOMS).unwrap();
        frame.insert("refs", refs);
        frame
    }

    #[test]
    fn targets_are_pinned_and_hold_per_resolved_frame() {
        let dir = TempDir::new().unwrap();
        let store = storage_in(&dir);
        let first = referencing_frame(3, &[2]);
        let schema = SequenceSchema::from_frames(std::slice::from_ref(&first)).unwrap();
        assert_eq!(schema.target("refs", "site"), Some(ATOMS));
        let mut writer = MrecWriter::create_in_storage(store.clone(), schema).unwrap();
        writer.append(&first).unwrap();
        // `atoms` shrinks to 2 rows while `refs` carries forward pointing at
        // row 2: the resolved frame breaks the reference.
        let err = writer
            .append(&atoms_frame(&[1.0, 2.0]))
            .unwrap_err()
            .to_string();
        assert!(
            err.contains("site") && err.contains("trajectory frame 1"),
            "{err}"
        );
        // Restating `refs` in range is fine; a refused frame changed nothing.
        let mut second = atoms_frame(&[1.0, 2.0]);
        second.insert("refs", block_with("site", uint_column(&[1])));
        writer.append(&second).unwrap();
        writer.close().unwrap();

        let pin = Group::open(store.clone(), TRAJ).unwrap().attributes()[SCHEMA_ATTRIBUTE].clone();
        assert_eq!(
            pin["blocks"]["refs"]["targets"],
            serde_json::json!({"site": ATOMS})
        );
        let mut seq = open_sequence(&store);
        assert_eq!(committed_len(&mut seq), 2);
        let back = frame_at(&mut seq, 1);
        assert_eq!(back["refs"].target("site"), Some(ATOMS));

        // Tamper the pin to a target the store breaks: the read refuses.
        let mut group = Group::open(store.clone(), TRAJ).unwrap();
        group.attributes_mut()[SCHEMA_ATTRIBUTE]["blocks"]["refs"]["targets"]["site"] =
            serde_json::json!("ghost");
        group.store_metadata().unwrap();
        assert!(open_sequence(&store).frame(0).is_err());
    }

    #[test]
    fn declare_target_refuses_what_cannot_be_a_reference() {
        let mut schema = SequenceSchema::new();
        schema.declare_column(ATOMS, X, DType::Float, &[]).unwrap();
        schema
            .declare_column("refs", "site", DType::Uint, &[])
            .unwrap();
        assert!(schema.declare_target("refs", "nope", ATOMS).is_err());
        assert!(schema.declare_target(ATOMS, X, ATOMS).is_err());
        assert!(
            schema
                .declare_target("refs", "site", "/trajectory/atoms")
                .is_err()
        );
        assert!(schema.declare_target("refs", "site", "a/b").is_err());
        schema
            .declare_target("refs", "site", "/system/atoms")
            .unwrap();
        assert!(schema.declare_target("refs", "site", ATOMS).is_err());
    }

    // -- aligned blocks (molrec F5) -----------------------------------------

    const TYPES: &str = "atom_types";

    fn string_column(values: &[&str]) -> Column {
        Column::from(
            ArrayD::from_shape_vec(
                vec![values.len()],
                values.iter().map(|s| (*s).to_string()).collect(),
            )
            .unwrap(),
        )
    }

    /// `atoms` (x) and, when `types` is given, the aligned `atom_types`.
    fn aligned_frame(x: &[f64], types: Option<&[&str]>) -> Frame {
        let mut frame = atoms_frame(x);
        if let Some(types) = types {
            frame.insert(TYPES, block_with("type", string_column(types)));
        }
        frame
    }

    fn aligned_schema() -> SequenceSchema {
        let mut schema = SequenceSchema::new();
        schema.declare_column(ATOMS, X, DType::Float, &[]).unwrap();
        schema
            .declare_column(TYPES, "type", DType::String, &[])
            .unwrap();
        schema.declare_aligned(TYPES, ATOMS).unwrap();
        schema
    }

    fn write_aligned(frames: &[Frame]) -> (TempDir, ReadableWritableListableStorage) {
        let dir = TempDir::new().unwrap();
        let store = storage_in(&dir);
        let mut writer = MrecWriter::create_in_storage(store.clone(), aligned_schema()).unwrap();
        for frame in frames {
            writer.append(frame).unwrap();
        }
        writer.close().unwrap();
        (dir, store)
    }

    fn types_at(seq: &mut MrecReader, index: u64) -> Option<Vec<String>> {
        frame_at(seq, index).get(TYPES).map(|b| {
            b.get("type")
                .and_then(|c| c.get::<String>())
                .unwrap()
                .iter()
                .cloned()
                .collect()
        })
    }

    #[test]
    fn an_aligned_block_carries_forward_while_its_target_moves() {
        let frames = [
            aligned_frame(&[0.0, 1.0, 2.0], Some(&["A", "B", "C"])),
            aligned_frame(&[0.1, 1.1, 2.1], None),
            aligned_frame(&[0.2, 1.2, 2.2], None),
            aligned_frame(&[0.3, 1.3, 2.3], Some(&["A", "A", "C"])),
        ];
        let (_dir, store) = write_aligned(&frames);
        let pin = Group::open(store.clone(), TRAJ).unwrap().attributes()[SCHEMA_ATTRIBUTE].clone();
        assert_eq!(
            pin["blocks"][TYPES]["aligned_with"],
            serde_json::json!(ATOMS)
        );
        let mut seq = open_sequence(&store);
        assert_eq!(types_at(&mut seq, 2).unwrap(), ["A", "B", "C"]);
        assert_eq!(types_at(&mut seq, 3).unwrap(), ["A", "A", "C"]);
        assert_eq!(seq.block_update_at(TYPES, 2).unwrap(), Some(0));
        assert_eq!(seq.schema().aligned_with(TYPES), Some(ATOMS));
    }

    #[test]
    fn an_aligned_block_is_restated_on_growth_and_may_appear_late_or_empty() {
        let frames = [
            aligned_frame(&[0.0, 1.0, 2.0], None),
            aligned_frame(&[0.0, 1.0, 2.0], Some(&["A", "B", "C"])),
            aligned_frame(&[0.0, 1.0, 2.0, 3.0], Some(&["A", "B", "C", "D"])),
            aligned_frame(&[], Some(&[])),
        ];
        let (_dir, store) = write_aligned(&frames);
        let mut seq = open_sequence(&store);
        assert_eq!(types_at(&mut seq, 0), None);
        assert_eq!(types_at(&mut seq, 2).unwrap().len(), 4);
        assert_eq!(types_at(&mut seq, 3).unwrap().len(), 0);
    }

    #[test]
    fn a_writer_refuses_a_frame_that_breaks_the_alignment() {
        let dir = TempDir::new().unwrap();
        let store = storage_in(&dir);
        let mut writer = MrecWriter::create_in_storage(store.clone(), aligned_schema()).unwrap();
        writer
            .append(&aligned_frame(&[0.0, 1.0, 2.0], Some(&["A", "B", "C"])))
            .unwrap();
        // `atoms` grows and `atom_types` is not restated.
        let err = writer
            .append(&aligned_frame(&[0.0, 1.0, 2.0, 3.0], None))
            .unwrap_err()
            .to_string();
        assert!(
            err.contains(TYPES) && err.contains("trajectory frame 1"),
            "{err}"
        );
        // Restated with another count: refused too.
        assert!(
            writer
                .append(&aligned_frame(&[0.0, 1.0, 2.0, 3.0], Some(&["A"])))
                .is_err()
        );
        // Nothing moved: a good frame still lands at ordinal 1.
        writer
            .append(&aligned_frame(
                &[0.0, 1.0, 2.0, 3.0],
                Some(&["A", "B", "C", "D"]),
            ))
            .unwrap();
        writer.close().unwrap();
        assert_eq!(committed_len(&mut open_sequence(&store)), 2);

        // The aligned block presented before its target exists.
        let dir = TempDir::new().unwrap();
        let store = storage_in(&dir);
        let mut writer = MrecWriter::create_in_storage(store, aligned_schema()).unwrap();
        let mut types_only = Frame::new();
        types_only.insert(TYPES, block_with("type", string_column(&["A"])));
        assert!(writer.append(&types_only).is_err());
    }

    #[test]
    fn a_store_whose_aligned_counts_disagree_is_refused() {
        let frames = [
            aligned_frame(&[0.0, 1.0, 2.0], Some(&["A", "B", "C"])),
            aligned_frame(&[0.0, 1.0], Some(&["A", "B"])),
        ];
        let (_dir, store) = write_aligned(&frames);
        assert!(MrecReader::from_storage(store.clone()).is_ok());
        // Tamper: `atom_types/offset` so update 1 holds 1 row instead of 2.
        let offsets = u64_array(&store, &format!("{TRAJ}/{TYPES}/offset"));
        assert_eq!(offsets, [0, 3, 5]);
        let array = Array::open(store.clone(), &format!("{TRAJ}/{TYPES}/offset")).unwrap();
        array
            .store_array_subset(&ArraySubset::new_with_shape(vec![3]), &[0_u64, 3, 4][..])
            .unwrap();
        array.store_metadata().unwrap();
        let err = MrecReader::from_storage(store).err().unwrap().to_string();
        assert!(err.contains(TYPES), "{err}");
    }

    #[test]
    fn declare_aligned_refuses_what_cannot_align() {
        let mut schema = aligned_schema();
        schema
            .declare_column("other", "flag", DType::Bool, &[])
            .unwrap();
        schema
            .declare_column("cells", "v", DType::Float, &[])
            .unwrap();
        // Undeclared target, self, chain either way.
        assert!(schema.declare_aligned("other", "nope").is_err());
        assert!(schema.declare_aligned("other", "other").is_err());
        assert!(schema.declare_aligned("other", TYPES).is_err());
        assert!(schema.declare_aligned(ATOMS, "other").is_err());
        // Shared column, either at declaration or added afterwards.
        schema.declare_column("more", X, DType::Float, &[]).unwrap();
        assert!(schema.declare_aligned("more", ATOMS).is_err());
        assert!(schema.declare_column(TYPES, X, DType::Float, &[]).is_err());
        assert!(
            schema
                .declare_column(ATOMS, "type", DType::String, &[])
                .is_err()
        );
        // A shaped block cannot align, nor an aligned one take a shape.
        schema.declare_structural_shape("cells", &[1]).unwrap();
        assert!(schema.declare_aligned("cells", ATOMS).is_err());
        assert!(schema.declare_structural_shape(TYPES, &[3]).is_err());
    }

    #[test]
    fn an_overflowing_structural_shape_does_not_change_the_schema() {
        let mut schema = SequenceSchema::new();
        schema.declare_block("cells", Some(6)).unwrap();
        let before = schema.clone();
        assert!(
            schema
                .declare_structural_shape("cells", &[usize::MAX, 2])
                .is_err()
        );
        assert_eq!(schema, before);
        assert_eq!(schema.rows_hint, before.rows_hint);
    }

    #[test]
    fn trajectory_preflight_checks_carry_forward_references() {
        let mut members = Block::new();
        members
            .insert("atom", ndarray::array![1_u64].into_dyn(), None)
            .unwrap();
        members.set_target("atom", "atoms").unwrap();
        let mut atoms = Block::new();
        atoms
            .insert("x", ndarray::array![0.0, 1.0].into_dyn(), None)
            .unwrap();
        let mut first = Frame::new();
        first.insert("atoms", atoms);
        first.insert("members", members);
        let mut second = Frame::new();
        second.insert("atoms", first["atoms"].select_rows(&[0]).unwrap());
        let frames = [first, second];
        let schema = SequenceSchema::from_frames(&frames).unwrap();
        assert!(schema.validate_frames(&frames).is_err());
    }

    #[test]
    fn an_aligned_block_may_not_share_a_name_with_a_system_block() {
        let dir = TempDir::new().unwrap();
        let store = storage_in(&dir);
        for group in ["/", "/system", "/system/atom_types"] {
            GroupBuilder::new()
                .build(store.clone(), group)
                .unwrap()
                .store_metadata()
                .unwrap();
        }
        let err = MrecWriter::create_in_storage(store, aligned_schema())
            .err()
            .unwrap()
            .to_string();
        assert!(err.contains("system"), "{err}");
    }
}
