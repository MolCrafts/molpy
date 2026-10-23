use crate::io::mrec::zarr_storage::frame_io::{
    default_precision_compressor, gzip, precision_shuffle, zarr_dtype,
};
use crate::io::mrec::zarr_storage::record_io::zerr;
use molrs::core::DType;
use molrs::core::MolRsError;
use std::sync::{Arc, Once};
use zarrs::array::codec::array_to_bytes::sharding::{ShardingCodecOptions, SubchunkWriteOrder};
use zarrs::array::codec::bytes_to_bytes::crc32c::Crc32cCodec;
use zarrs::array::data_type::{Float16DataType, Float32DataType};
use zarrs::array::{ArraySubset, BytesToBytesCodecTraits, CodecOptions, CodecSpecificOptions};
use zarrs::config::{global_config, global_config_mut};

use super::layout::{
    ASSUMED_STRING_ITEMSIZE, MAX_CHUNKS_PER_SHARD, MIN_CHUNK_BYTES, SHARD_TARGET_BYTES,
};

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

/// The levels `gzip` accepts.
const GZIP_LEVELS: std::ops::RangeInclusive<i64> = 0..=9;
/// The levels `zstd` accepts (negative levels trade ratio for speed).
const ZSTD_LEVELS: std::ops::RangeInclusive<i64> = -7..=22;

impl std::str::FromStr for Compression {
    type Err = String;

    /// `"none"`, `"gzip"` (level 1), `"gzip:<0-9>"`, `"zstd"` (level 3),
    /// `"zstd:<-7-22>"` — the spelling every binding accepts.
    ///
    /// # Errors
    ///
    /// An unknown compressor, a level that is no integer, or a level outside
    /// the compressor's range, named; `zstd` in a build that cannot encode it.
    fn from_str(spec: &str) -> Result<Self, Self::Err> {
        let (name, raw) = match spec.split_once(':') {
            Some((name, level)) => (name, Some(level)),
            None => (spec, None),
        };
        let level = |default: i64, range: std::ops::RangeInclusive<i64>| -> Result<i64, String> {
            let value = match raw {
                None => return Ok(default),
                Some(text) => text
                    .parse::<i64>()
                    .map_err(|_| format!("compression level {text:?} is not an integer"))?,
            };
            if range.contains(&value) {
                Ok(value)
            } else {
                Err(format!(
                    "{name} level {value} is outside {}..={}",
                    range.start(),
                    range.end()
                ))
            }
        };
        match name {
            "none" if raw.is_none() => Ok(Compression::None),
            "gzip" => Ok(Compression::Gzip(level(1, GZIP_LEVELS)? as u32)),
            #[cfg(feature = "zarr-codecs")]
            "zstd" => Ok(Compression::Zstd(level(3, ZSTD_LEVELS)? as i32)),
            #[cfg(not(feature = "zarr-codecs"))]
            "zstd" => {
                let _ = (level, ZSTD_LEVELS);
                Err("zstd encoding needs a native build (feature zarr-codecs)".to_owned())
            }
            other => Err(format!(
                "unknown compression {other:?}; use \"none\", \"gzip[:level]\" or \"zstd[:level]\""
            )),
        }
    }
}

/// The bytes-to-bytes pipeline of one growth array: a compressor, and
/// whether a byte shuffle precedes it (a precision column's pipeline).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(super) struct Pipeline {
    pub(super) compression: Compression,
    pub(super) shuffle: bool,
}

impl Pipeline {
    /// A pipeline with no shuffle.
    pub(super) const fn plain(compression: Compression) -> Self {
        Self {
            compression,
            shuffle: false,
        }
    }
}

/// The frozen extents of one growth array.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(super) struct Extents {
    pub(super) rows_per_chunk: u64,
    pub(super) chunks_per_shard: u64,
}

impl Extents {
    /// Rows one inner chunk of a block column holds.
    ///
    /// Frame-aligned: the smallest whole number of frames of `frame_rows` rows
    /// whose narrowest column reaches [`MIN_CHUNK_BYTES`]. A block whose
    /// representative frame carries no rows (or is unknown) falls back to the
    /// byte floor alone.
    pub(super) fn block_rows_per_chunk(frame_rows: u64, min_row_bytes: u64) -> u64 {
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
    pub(super) fn for_column(
        rows_per_chunk: u64,
        row_bytes: u64,
        chunks_per_shard: Option<u64>,
    ) -> Self {
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
pub(super) fn partial_encoding_options() -> CodecOptions {
    global_config()
        .codec_options()
        .with_experimental_partial_encoding(true)
}

/// Codec options that pin a shard's subchunk layout to C (row-major) order, so
/// a store's bytes do not depend on thread scheduling.
pub(super) fn deterministic_shard_layout() -> CodecSpecificOptions {
    CodecSpecificOptions::default().with_option(
        ShardingCodecOptions::default().with_subchunk_write_order(SubchunkWriteOrder::C),
    )
}

/// Switch off zarrs' `_zarrs` provenance attribute on every array this process
/// creates. Done once: the attribute is bundle noise on a contract layout, and
/// the flag is process-wide by the library's design.
pub(super) fn quiet_zarrs_metadata() {
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
pub(super) fn inner_codecs(
    pipeline: Pipeline,
) -> Result<Vec<Arc<dyn BytesToBytesCodecTraits>>, MolRsError> {
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
pub(super) fn is_float_width(dtype: DType) -> bool {
    matches!(dtype, DType::Float | DType::C64 | DType::C128)
}

/// The column width a schema dtype tag ([`DType::name`]) names.
pub(super) fn dtype_from_tag(tag: &str) -> Result<DType, MolRsError> {
    DType::from_name(tag)
        .ok_or_else(|| MolRsError::zarr(format!("unknown column dtype tag {tag:?}")))
}

/// The column width a stored Zarr data type maps to.
///
/// Narrow float arrays are refused, not promoted: the record has one float
/// (`F = f64`), so a `float16`/`float32` array on disk is an error naming
/// the array and its stored type.
pub(super) fn dtype_of_stored(
    name: &str,
    stored: &zarrs::array::DataType,
) -> Result<DType, MolRsError> {
    if stored.is::<Float16DataType>() || stored.is::<Float32DataType>() {
        return Err(MolRsError::zarr(format!(
            "{name} is stored as {stored:?}: narrow floats are not read; \
             the record has one float, `F = f64`"
        )));
    }
    DType::ALL
        .into_iter()
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
pub(super) fn meta_layout(tag: &str) -> Option<(DType, Vec<u64>)> {
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
pub(super) fn row_bytes(dtype: DType, trailing: &[u64]) -> u64 {
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
pub(super) fn rows_subset(
    start: u64,
    rows: u64,
    trailing: &[u64],
) -> Result<ArraySubset, MolRsError> {
    let mut origin = Vec::with_capacity(trailing.len() + 1);
    origin.push(start);
    origin.extend(std::iter::repeat_n(0u64, trailing.len()));
    let mut shape = Vec::with_capacity(trailing.len() + 1);
    shape.push(rows);
    shape.extend_from_slice(trailing);
    ArraySubset::new_with_start_shape(origin, shape).map_err(zerr)
}
