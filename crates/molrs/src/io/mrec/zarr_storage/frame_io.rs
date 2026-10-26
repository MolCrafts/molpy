//! Low-level Zarr ↔ Block/Column/SimBox/Frame helpers.
//!
//! These functions are shared by the frame/trajectory Zarr backend. They convert between
//! molrs in-memory types and
//! Zarr V3 arrays/groups, always relative to a caller-supplied path prefix.

#[cfg(feature = "zarr")]
use crate::core::Cell;
use zarrs::array::codec::bytes_to_bytes::crc32c::Crc32cCodec;
use zarrs::array::data_type::{
    BoolDataType, Complex64DataType, Complex128DataType, Float16DataType, Float32DataType,
    Float64DataType, Int8DataType, Int16DataType, Int32DataType, Int64DataType, StringDataType,
    UInt8DataType, UInt16DataType, UInt32DataType, UInt64DataType,
};
use zarrs::array::{Array, ArraySubset};
#[cfg(feature = "zarr")]
use zarrs::array::{
    ArrayBuilder, BytesToBytesCodecTraits,
    codec::{GzipCodec, ShuffleCodec},
    data_type,
};
#[cfg(feature = "zarr")]
use zarrs::group::GroupBuilder;
use zarrs::node::{Node, NodeMetadata, NodePath};
use zarrs::storage::{
    ListableStorageTraits, ReadableStorageTraits, ReadableWritableListableStorage,
};
#[cfg(feature = "zarr")]
use zarrs::storage::{StorePrefix, WritableStorageTraits};

use ndarray::ArrayD;
#[cfg(feature = "zarr")]
use ndarray::ArrayViewD;
use std::sync::Arc;

#[cfg(feature = "zarr")]
use molrs::core::DType;
use molrs::core::Frame;
use molrs::core::MolRsError;
use molrs::core::SimBox;
use molrs::core::{Block, Column};
use molrs::core::{MetaMap, MetaValue};
use molrs::op::F;

/// The attribute of a frame-shaped group that maps every key of its `meta`
/// document to its tag ([`MetaValue::dtype`]). One leading underscore, as
/// `_validity`: binding-owned, legal in every store, and never a meta key.
const META_TYPES_ATTR: &str = "_meta_types";

#[cfg(feature = "zarr")]
use super::chunking::{ChunkPlan, plan};

/// `gzip` level for the fixed-size arrays that compress: integer, boolean and
/// string columns. Level 1 — these compress by structure, not by effort.
///
/// Floating-point columns are stored raw: 52 random mantissa bits gzip to
/// about 95 % of their size at a real CPU cost. No lossy codec is admitted;
/// what makes a float column compress is a [declared
/// precision](molrs::core::check_precision), which rounds the values onto a binary
/// grid *before* they reach the pipeline (see [`frame_codecs`]). Every array
/// carries `crc32c` so a torn chunk is a checksum error rather than garbage
/// rows.
#[cfg(feature = "zarr")]
pub(in crate::io::mrec::zarr_storage) const GZIP_LEVEL: u32 = 1;

/// `zstd` level of a precision column's pipeline (`numcodecs.shuffle`, then
/// `zstd`, then `crc32c`): molrec's reference writer setting.
#[cfg(feature = "zarr-codecs")]
pub(in crate::io::mrec::zarr_storage) const PRECISION_ZSTD_LEVEL: i32 = 3;

/// The array attribute a frame-shaped section's precision column carries its
/// declared precision in. On the trajectory path the precision lives in the
/// pinned `sequence_schema` only.
pub(crate) const PRECISION_ATTRIBUTE: &str = "precision";

/// The byte shuffle every precision column's pipeline opens with: its
/// element size is the width of an `f64`, so the zero low-order mantissa
/// bytes of the rounded values land in runs the compressor removes.
#[cfg(feature = "zarr")]
pub(in crate::io::mrec::zarr_storage) fn precision_shuffle() -> Arc<dyn BytesToBytesCodecTraits> {
    Arc::new(ShuffleCodec::new(std::mem::size_of::<f64>()))
}

/// The compressor of a precision column when the producer chose none:
/// `zstd` level 3 where this build can encode `zstd` (`zarr-codecs`), else
/// `gzip` level 1 — the wasm32 writer's fallback, which every reader decodes.
#[cfg(feature = "zarr")]
pub(in crate::io::mrec::zarr_storage) fn default_precision_compressor()
-> Result<Arc<dyn BytesToBytesCodecTraits>, MolRsError> {
    #[cfg(feature = "zarr-codecs")]
    {
        Ok(Arc::new(
            zarrs::array::codec::bytes_to_bytes::zstd::ZstdCodec::new(PRECISION_ZSTD_LEVEL, false),
        ))
    }
    #[cfg(not(feature = "zarr-codecs"))]
    {
        gzip(GZIP_LEVEL)
    }
}

/// A `gzip` codec at `level`.
#[cfg(feature = "zarr")]
pub(in crate::io::mrec::zarr_storage) fn gzip(
    level: u32,
) -> Result<Arc<dyn BytesToBytesCodecTraits>, MolRsError> {
    Ok(Arc::new(GzipCodec::new(level).map_err(|e| {
        MolRsError::zarr(format!("gzip level {level}: {e}"))
    })?))
}

/// The bytes-to-bytes codecs of one frame-path array of `dtype`, every list
/// closed by `crc32c`:
///
/// | array | codecs |
/// |-------|--------|
/// | `f64` column with a declared precision | `numcodecs.shuffle` (8), `zstd` 3 (`gzip` 1 without `zarr-codecs`) |
/// | any other float column (`f64`, `c64`, `c128`) | none |
/// | everything else | `gzip` 1 |
#[cfg(feature = "zarr")]
pub(in crate::io::mrec::zarr_storage) fn frame_codecs(
    dtype: DType,
    precision: bool,
) -> Result<Vec<Arc<dyn BytesToBytesCodecTraits>>, MolRsError> {
    let mut codecs: Vec<Arc<dyn BytesToBytesCodecTraits>> = Vec::with_capacity(3);
    if precision {
        codecs.push(precision_shuffle());
        codecs.push(default_precision_compressor()?);
    } else if !matches!(dtype, DType::Float | DType::C64 | DType::C128) {
        codecs.push(gzip(GZIP_LEVEL)?);
    }
    codecs.push(Arc::new(Crc32cCodec::new()));
    Ok(codecs)
}

// ---------------------------------------------------------------------------
// Column write
// ---------------------------------------------------------------------------

/// Write one column as the array at `path`.
///
/// `precision` is the column's [declared precision](molrs::core::check_precision):
/// when set, the column must be `f64`, a rounded copy of its values is what
/// lands (`stored(x)`, exactly), the pipeline opens with the byte shuffle and
/// a compressor ([`frame_codecs`]), and the array carries the declaration as
/// its `precision` attribute. The caller's column is not touched.
///
/// # Errors
///
/// A [`MolRsError::Zarr`] naming the array when a precision is declared on a
/// column that is not `f64` or is not admissible; any storage or codec error.
#[cfg(feature = "zarr")]
pub(crate) fn write_column(
    store: &ReadableWritableListableStorage,
    path: &str,
    col: &Column,
    precision: Option<f64>,
) -> Result<(), MolRsError> {
    let shape: Vec<u64> = col.shape().iter().map(|&s| s as u64).collect();
    let chunking = plan(&shape, col.dtype().itemsize());
    let (dt, fill) = dtype_of(col);
    let codecs = frame_codecs(col.dtype(), precision.is_some())?;
    let mut attributes = serde_json::Map::new();
    if let Some(p) = precision {
        let Column::Float(a) = col else {
            return Err(MolRsError::zarr(format!(
                "{path} is {}; only an f64 column declares a precision",
                col.dtype()
            )));
        };
        let rounded = quantized(a, p).map_err(|e| MolRsError::zarr(format!("{path}: {e}")))?;
        attributes.insert(PRECISION_ATTRIBUTE.to_string(), serde_json::Value::from(p));
        return write_typed_array(
            store,
            path,
            rounded.view(),
            dt,
            fill,
            chunking,
            codecs,
            attributes,
        );
    }
    macro_rules! land {
        ($a:expr) => {
            write_typed_array(
                store,
                path,
                $a.view(),
                dt,
                fill,
                chunking,
                codecs,
                attributes,
            )
        };
    }
    match col {
        Column::Float(a) => land!(a),
        Column::I8(a) => land!(a),
        Column::I16(a) => land!(a),
        Column::Int(a) => land!(a),
        Column::I64(a) => land!(a),
        Column::Uint(a) => land!(a),
        Column::U8(a) => land!(a),
        Column::U16(a) => land!(a),
        Column::U32(a) => land!(a),
        Column::Bool(a) => land!(a),
        Column::String(a) => land!(a),
        Column::C64(a) => land!(a),
        Column::C128(a) => land!(a),
    }
}

/// `stored(x)` of every value of `values` under precision `p`: the rounded
/// copy a writer lands for a precision column.
///
/// # Errors
///
/// [`molrs::core::quantum`]'s, for an inadmissible `p`.
pub(in crate::io::mrec::zarr_storage) fn quantized(
    values: &ArrayD<f64>,
    p: f64,
) -> Result<ArrayD<f64>, MolRsError> {
    let q = molrs::core::quantum(p)?;
    let mut rounded = values.as_standard_layout().into_owned();
    rounded.mapv_inplace(|x| molrs::core::quantize(x, q));
    Ok(rounded)
}

/// The Zarr data type and fill value a column of this dtype is stored as.
#[cfg(feature = "zarr")]
fn dtype_of(col: &Column) -> (zarrs::array::DataType, zarrs::array::FillValue) {
    zarr_dtype(col.dtype())
}

/// The Zarr data type and fill value a column of `dtype` is stored as.
///
/// The fill value is always the dtype's zero — [`DType::itemsize`] zero bytes,
/// and no bytes (the empty string) for the variable-width variant. The frame
/// path writes its arrays whole, so no reader there ever observes a fill; the
/// zero is what a sequence's growth array is padded with beyond its live rows.
///
/// Keyed on [`DType`] rather than on a [`Column`] because the sequence writer
/// creates arrays from a *schema*, where no column value exists yet — this is
/// the one dtype table, and [`dtype_of`] is its column-shaped door.
#[cfg(feature = "zarr")]
pub(in crate::io::mrec::zarr_storage) fn zarr_dtype(
    dtype: DType,
) -> (zarrs::array::DataType, zarrs::array::FillValue) {
    let dt = match dtype {
        DType::Float => data_type::float64(),
        DType::I8 => data_type::int8(),
        DType::I16 => data_type::int16(),
        DType::Int => data_type::int32(),
        DType::I64 => data_type::int64(),
        DType::Bool => data_type::bool(),
        DType::Uint => data_type::uint64(),
        DType::U8 => data_type::uint8(),
        DType::U16 => data_type::uint16(),
        DType::U32 => data_type::uint32(),
        DType::String => data_type::string(),
        DType::C64 => data_type::complex64(),
        DType::C128 => data_type::complex128(),
    };
    let fill = zarrs::array::FillValue::new(vec![0u8; dtype.itemsize().unwrap_or(0)]);
    (dt, fill)
}

/// Switch off zarrs' `_zarrs` provenance attribute on every array this process
/// creates. Done once: the attribute is bundle noise on a contract layout, and
/// the flag is process-wide by the library's design, so every array writer
/// calls this before it builds, whichever writes first.
#[cfg(feature = "zarr")]
pub(in crate::io::mrec::zarr_storage) fn quiet_zarrs_metadata() {
    static ONCE: std::sync::Once = std::sync::Once::new();
    ONCE.call_once(|| {
        zarrs::config::global_config_mut().set_include_zarrs_metadata(false);
    });
}

/// The one array writer: every array this backend stores is created and filled
/// here, laid out by `chunking` and encoded by `codecs` (from
/// [`frame_codecs`]), with `attributes` on the array.
///
/// `chunking.chunks` of `None` — a variable-width dtype or an empty leading
/// axis, both of which molrec declines to size — is laid out as
/// one chunk spanning the whole array. `chunking.shards` of `Some` packs those
/// chunks into one shard file per shard extent, with `codecs` on the inner
/// chunks and the shard index at the end (zarrs' default); `None` applies
/// them to the chunks directly.
#[cfg(feature = "zarr")]
#[allow(clippy::too_many_arguments)]
pub(in crate::io::mrec::zarr_storage) fn write_typed_array<T>(
    store: &ReadableWritableListableStorage,
    path: &str,
    a: ArrayViewD<'_, T>,
    dt: zarrs::array::DataType,
    fill: impl Into<zarrs::array::builder::ArrayBuilderFillValue>,
    chunking: ChunkPlan,
    codecs: Vec<Arc<dyn BytesToBytesCodecTraits>>,
    attributes: serde_json::Map<String, serde_json::Value>,
) -> Result<(), MolRsError>
where
    T: zarrs::array::Element + Clone,
{
    let data = a.as_standard_layout();
    let shape: Vec<u64> = data.shape().iter().map(|&s| s as u64).collect();
    // Unplanned, the array is one chunk. A chunk extent is nonzero, so an
    // empty axis (a table of no rows) still gets a chunk of one.
    let chunk = chunking
        .chunks
        .unwrap_or_else(|| shape.iter().map(|&n| n.max(1)).collect());
    // Sharding makes the array's own chunk extent the *shard*, and the planned
    // chunk the subchunk inside it.
    let (extent, subchunk) = match chunking.shards {
        Some(shards) => (shards, Some(chunk)),
        None => (chunk, None),
    };
    quiet_zarrs_metadata();
    let mut builder = ArrayBuilder::new(shape.clone(), extent, dt, fill);
    // Under sharding these codecs encode the subchunks, inside the shard.
    builder.bytes_to_bytes_codecs(codecs);
    if !attributes.is_empty() {
        builder.attributes(attributes);
    }
    if let Some(subchunk) = subchunk {
        builder.subchunk_shape(subchunk);
    }
    let arr = builder.build(store.clone(), path)?;
    arr.store_metadata()?;
    arr.store_array_subset(
        &ArraySubset::new_with_shape(shape),
        data.as_slice().unwrap(),
    )?;
    Ok(())
}

// ---------------------------------------------------------------------------
// Column read
// ---------------------------------------------------------------------------

/// Read the `subset` of the array at `path` back as a [`Column`] — at the
/// width it was stored in. Narrow floats are refused, not widened: the record
/// has one float ([`F`]).
///
/// The column's shape is the subset's, not the array's: a caller reading one
/// frame out of a sequence array passes that frame's subset and gets a column
/// shaped like the frame.
///
/// Generic over the store, and asking for reads only, so the read-write store
/// the record doors hold and the read-only one [`MrecReader`] holds share
/// this one dtype dispatch.
///
/// [`F`]: crate::op::F
/// [`MrecReader`]: super::MrecReader
pub(crate) fn read_column<S>(
    store: &Arc<S>,
    path: &str,
    subset: &ArraySubset,
) -> Result<Column, MolRsError>
where
    S: ?Sized + ReadableStorageTraits + 'static,
{
    let arr = Array::open(store.clone(), path)?;
    read_column_array(&arr, subset)
}

/// [`read_column`] over an already-open array — the form a reader that keeps
/// its array handles across frames uses, so a frame read is one chunk fetch
/// rather than a metadata round trip plus a chunk fetch.
pub(crate) fn read_column_array<S>(
    arr: &Array<S>,
    subset: &ArraySubset,
) -> Result<Column, MolRsError>
where
    S: ?Sized + ReadableStorageTraits + 'static,
{
    let shape: Vec<usize> = subset.shape().iter().map(|&s| s as usize).collect();

    let dt = arr.data_type();

    // Narrow floats are refused, not promoted: the record has one float
    // (`F = f64`), so a `float16`/`float32` array on disk is an error naming
    // the array and its stored type. Same rule as `sequence.rs::dtype_of_stored`.
    if dt.is::<Float16DataType>() || dt.is::<Float32DataType>() {
        Err(MolRsError::zarr(format!(
            "{} is stored as {dt:?}: narrow floats are not read; \
             the record has one float, `F = f64`",
            arr.path()
        )))
    } else if dt.is::<Float64DataType>() {
        let data: Vec<f64> = arr.retrieve_array_subset(subset)?;
        Ok(Column::from(
            ArrayD::from_shape_vec(shape, data).map_err(shape_err)?,
        ))
    } else if dt.is::<Int8DataType>() {
        let data: Vec<i8> = arr.retrieve_array_subset(subset)?;
        Ok(Column::from(
            ArrayD::from_shape_vec(shape, data).map_err(shape_err)?,
        ))
    } else if dt.is::<Int16DataType>() {
        let data: Vec<i16> = arr.retrieve_array_subset(subset)?;
        Ok(Column::from(
            ArrayD::from_shape_vec(shape, data).map_err(shape_err)?,
        ))
    } else if dt.is::<Int32DataType>() {
        let data: Vec<i32> = arr.retrieve_array_subset(subset)?;
        Ok(Column::from(
            ArrayD::from_shape_vec(shape, data).map_err(shape_err)?,
        ))
    } else if dt.is::<Int64DataType>() {
        let data: Vec<i64> = arr.retrieve_array_subset(subset)?;
        Ok(Column::from(
            ArrayD::from_shape_vec(shape, data).map_err(shape_err)?,
        ))
    } else if dt.is::<UInt16DataType>() {
        let data: Vec<u16> = arr.retrieve_array_subset(subset)?;
        Ok(Column::from(
            ArrayD::from_shape_vec(shape, data).map_err(shape_err)?,
        ))
    } else if dt.is::<UInt32DataType>() {
        let data: Vec<u32> = arr.retrieve_array_subset(subset)?;
        Ok(Column::from(
            ArrayD::from_shape_vec(shape, data).map_err(shape_err)?,
        ))
    } else if dt.is::<UInt64DataType>() {
        let data: Vec<u64> = arr.retrieve_array_subset(subset)?;
        Ok(Column::from(
            ArrayD::from_shape_vec(shape, data).map_err(shape_err)?,
        ))
    } else if dt.is::<BoolDataType>() {
        let data: Vec<bool> = arr.retrieve_array_subset(subset)?;
        Ok(Column::from(
            ArrayD::from_shape_vec(shape, data).map_err(shape_err)?,
        ))
    } else if dt.is::<UInt8DataType>() {
        let data: Vec<u8> = arr.retrieve_array_subset(subset)?;
        Ok(Column::from(
            ArrayD::from_shape_vec(shape, data).map_err(shape_err)?,
        ))
    } else if dt.is::<StringDataType>() {
        let data: Vec<String> = arr.retrieve_array_subset(subset)?;
        Ok(Column::from(
            ArrayD::from_shape_vec(shape, data).map_err(shape_err)?,
        ))
    } else if dt.is::<Complex64DataType>() {
        let data: Vec<num_complex::Complex<f32>> = arr.retrieve_array_subset(subset)?;
        Ok(Column::from(
            ArrayD::from_shape_vec(shape, data).map_err(shape_err)?,
        ))
    } else if dt.is::<Complex128DataType>() {
        let data: Vec<num_complex::Complex<f64>> = arr.retrieve_array_subset(subset)?;
        Ok(Column::from(
            ArrayD::from_shape_vec(shape, data).map_err(shape_err)?,
        ))
    } else {
        Err(MolRsError::zarr(format!("unsupported dtype: {:?}", dt)))
    }
}

/// Insert a column read from a store into `block`.
///
/// The store read path is strict where the in-memory API is lenient:
/// [`Block::insert`] admits a canonical key at another width of its
/// family (and widens a narrow unsigned identifier to `u64`), but a store
/// that holds a canonical key at any dtype but its declared one is refused
/// here (see [`check_canonical_dtype`]) rather than silently converted.
pub(crate) fn insert_column_into_block(
    block: &mut Block,
    name: &str,
    col: Column,
) -> Result<(), MolRsError> {
    check_canonical_dtype(name, col.dtype())?;
    // Zero-copy insert: hand the Arc-backed Column directly to the Block.
    block.insert(name, col, None).map_err(MolRsError::Block)
}

/// Refuse a canonical key (every key of the schema tables: `x` `f64`, `ix`
/// `i32`, `id` `u64`, `element` `string`, …) stored at any dtype but the one
/// the vocabulary declares.
///
/// Every molrs writer stores those keys at their declared dtype
/// ([`canonical_width`] converts an in-memory column of another width of the
/// family on the way out), so another stored dtype came from a producer that
/// broke the contract, and converting it on read would hide that.
pub(crate) fn check_canonical_dtype(
    name: &str,
    stored: molrs::core::DType,
) -> Result<(), MolRsError> {
    match molrs::core::schema::column(name) {
        Some(spec) if spec.dtype != stored => Err(MolRsError::zarr(format!(
            "column {name:?} is stored as {}; the canonical key {name:?} is {}, and a store is \
             not converted on read",
            stored.name(),
            spec.dtype.name()
        ))),
        _ => Ok(()),
    }
}

/// The dtype a column under `name` is stored at: the canonical key's
/// declared dtype, or the column's own for any other key.
pub(crate) fn stored_dtype(name: &str, dtype: DType) -> DType {
    molrs::core::schema::column(name).map_or(dtype, |spec| spec.dtype)
}

/// `col` at the dtype a canonical key `name` is stored at, or `None` when it
/// already is (or `name` is not canonical).
///
/// The in-memory block admits a canonical signed key at any signed width
/// (`ix` as `i64`) and a canonical unsigned one at any unsigned width; a
/// store holds exactly the declared dtype. The conversion is exact or
/// refused.
///
/// # Errors
///
/// A [`MolRsError::Zarr`] naming the key when a value does not fit the
/// declared dtype, or the column's family is not the declared one.
pub(crate) fn canonical_width(name: &str, col: &Column) -> Result<Option<Column>, MolRsError> {
    let Some(spec) = molrs::core::schema::column(name) else {
        return Ok(None);
    };
    if spec.dtype == col.dtype() {
        return Ok(None);
    }
    fn narrow<T: Copy + std::fmt::Display, U: TryFrom<T>>(
        name: &str,
        values: &ArrayD<T>,
    ) -> Result<ArrayD<U>, MolRsError> {
        let mut out = Vec::with_capacity(values.len());
        for &v in values.iter() {
            out.push(U::try_from(v).map_err(|_| {
                MolRsError::zarr(format!(
                    "column {name:?}: {v} does not fit the canonical dtype of {name:?}"
                ))
            })?);
        }
        ArrayD::from_shape_vec(values.shape(), out).map_err(shape_err)
    }
    let converted = match (spec.dtype, col) {
        (DType::Int, Column::I8(h)) => Column::from(h.array().mapv(i32::from)),
        (DType::Int, Column::I16(h)) => Column::from(h.array().mapv(i32::from)),
        (DType::Int, Column::I64(h)) => Column::from(narrow::<i64, i32>(name, h.array())?),
        (DType::I64, Column::I8(h)) => Column::from(h.array().mapv(i64::from)),
        (DType::I64, Column::I16(h)) => Column::from(h.array().mapv(i64::from)),
        (DType::I64, Column::Int(h)) => Column::from(h.array().mapv(i64::from)),
        (DType::Uint, Column::U8(h)) => Column::from(h.array().mapv(u64::from)),
        (DType::Uint, Column::U16(h)) => Column::from(h.array().mapv(u64::from)),
        (DType::Uint, Column::U32(h)) => Column::from(h.array().mapv(u64::from)),
        (expected, other) => {
            return Err(MolRsError::zarr(format!(
                "column {name:?} is {}; the canonical key {name:?} is stored as {}",
                other.dtype().name(),
                expected.name()
            )));
        }
    };
    Ok(Some(converted))
}

// ---------------------------------------------------------------------------
// SimBox write / read
// ---------------------------------------------------------------------------

/// The `cell_defined` attribute of a box group: absent means a defined cell
/// (it is only ever written when false).
///
/// # Errors
///
/// A [`MolRsError::Zarr`] naming `what` when the attribute is not a bool.
pub(crate) fn cell_defined_attribute(
    attributes: &serde_json::Map<String, serde_json::Value>,
    what: &str,
) -> Result<bool, MolRsError> {
    match attributes.get("cell_defined") {
        None => Ok(true),
        Some(serde_json::Value::Bool(defined)) => Ok(*defined),
        Some(other) => Err(MolRsError::zarr(format!(
            "{what}: cell_defined must be a bool, found {other}"
        ))),
    }
}

/// The numbers of a JSON box attribute: a flat array, or an array of rows
/// (the cell's `[[..], [..], [..]]`) read row by row.
///
/// # Errors
///
/// A [`MolRsError::Zarr`] naming `what` when any entry is not a number.
pub(crate) fn json_numbers(value: &serde_json::Value, what: &str) -> Result<Vec<F>, MolRsError> {
    let malformed = || {
        MolRsError::zarr(format!(
            "{what}: expected an array of numbers, found {value}"
        ))
    };
    let entries = value.as_array().ok_or_else(malformed)?;
    let mut out = Vec::with_capacity(9);
    for entry in entries {
        match entry {
            serde_json::Value::Array(row) => {
                for number in row {
                    out.push(number.as_f64().ok_or_else(malformed)?);
                }
            }
            number => out.push(number.as_f64().ok_or_else(malformed)?),
        }
    }
    Ok(out)
}

/// The flags of a JSON boundary attribute.
///
/// # Errors
///
/// A [`MolRsError::Zarr`] naming `what` when any entry is not a bool.
pub(crate) fn json_flags(value: &serde_json::Value, what: &str) -> Result<Vec<bool>, MolRsError> {
    let malformed =
        || MolRsError::zarr(format!("{what}: expected an array of bools, found {value}"));
    value
        .as_array()
        .ok_or_else(malformed)?
        .iter()
        .map(|flag| flag.as_bool().ok_or_else(malformed))
        .collect()
}

/// The one decoder of a stored cell, whatever layout it was read from (a
/// frame's box group, a trajectory's fixed-cell attributes, or one per-step
/// box update).
///
/// `h` is the row-major cell matrix and is read only for a defined cell: an
/// undefined cell's vectors mean nothing and [`Cell::Undefined`] carries the
/// identity in their place. An absent `origin` is the coordinate origin; an
/// absent `boundary` is all-periodic for a defined cell and periodic on no
/// axis for an undefined one.
///
/// # Errors
///
/// A [`MolRsError::Zarr`] naming `what` for a missing or mis-sized part, or a
/// cell [`SimBox::new`] refuses (an undefined cell with a periodic flag
/// among them).
pub(crate) fn assemble_cell(
    cell_defined: bool,
    h: Option<&[F]>,
    origin: Option<&[F]>,
    boundary: Option<&[bool]>,
    what: &str,
) -> Result<SimBox, MolRsError> {
    let cell = if cell_defined {
        let h =
            h.ok_or_else(|| MolRsError::zarr(format!("{what}: the cell vectors are missing")))?;
        if h.len() != 9 {
            return Err(MolRsError::zarr(format!(
                "{what}: the cell vectors hold {} values, expected 9",
                h.len()
            )));
        }
        Cell::Matrix(std::array::from_fn(|r| {
            std::array::from_fn(|c| h[3 * r + c])
        }))
    } else {
        Cell::Undefined
    };
    let origin = match origin {
        None => [0.0; 3],
        Some(&[x, y, z]) => [x, y, z],
        Some(other) => {
            return Err(MolRsError::zarr(format!(
                "{what}: the origin holds {} values, expected 3",
                other.len()
            )));
        }
    };
    let pbc = match boundary {
        None => [cell_defined; 3],
        Some(&[a, b, c]) => [a, b, c],
        Some(other) => {
            return Err(MolRsError::zarr(format!(
                "{what}: the boundary holds {} flags, expected 3",
                other.len()
            )));
        }
    };
    SimBox::new(cell, origin, pbc).map_err(|e| MolRsError::zarr(format!("{what}: {e}")))
}

#[cfg(feature = "zarr")]
pub(crate) fn write_simbox(
    store: &ReadableWritableListableStorage,
    prefix: &str,
    simbox: &SimBox,
) -> Result<(), MolRsError> {
    let mut attrs = serde_json::Map::new();
    // `cell_defined` is written only when it is *false*: absent means a
    // defined cell.
    if !simbox.is_cell_defined() {
        attrs.insert("cell_defined".to_string(), serde_json::Value::Bool(false));
    }
    GroupBuilder::new()
        .attributes(attrs)
        .build(store.clone(), prefix)?
        .store_metadata()?;

    // vectors: [3,3] Float64 — lattice vectors are the COLUMNS, per contract.
    // Geometry stays f64 to match the in-memory science contract.
    let h_data: Vec<F> = simbox.matrix().as_flattened().to_vec();
    write_f64_array(store, &format!("{}/vectors", prefix), &[3, 3], &h_data)?;

    // origin: [3] Float64, written only when it carries information (the
    // normative default is the coordinate origin).
    let origin = simbox.origin();
    if origin.iter().any(|&v| v != 0.0) {
        let origin_data: Vec<F> = origin.to_vec();
        write_f64_array(store, &format!("{}/origin", prefix), &[3], &origin_data)?;
    }

    // boundary: [3] bool, the same array form the trajectory path writes;
    // omitted for the all-periodic default. An undefined cell writes its
    // all-false flags explicitly, since the omitted default is periodic.
    let pbc = simbox.pbc();
    if pbc != [true, true, true] {
        let flags = ndarray::ArrayD::from_shape_vec(vec![3], pbc.to_vec()).map_err(shape_err)?;
        write_column(
            store,
            &format!("{}/boundary", prefix),
            &Column::from(flags),
            None,
        )?;
    }

    Ok(())
}

pub(crate) fn read_simbox<S>(store: &Arc<S>, prefix: &str) -> Result<SimBox, MolRsError>
where
    S: ?Sized + ReadableStorageTraits + ListableStorageTraits + 'static,
{
    let group = zarrs::group::Group::open(store.clone(), prefix)?;
    let cell_defined = cell_defined_attribute(group.attributes(), prefix)?;

    // Narrow float arrays are refused; see `read_simbox_float_path`. An
    // undefined cell's `vectors` are not read: whatever a store holds there
    // (zeros included) is never inverted.
    let h = if cell_defined {
        read_simbox_float_path(store, &format!("{prefix}/vectors"))?
    } else {
        None
    };
    // An absent `origin` array is the zero origin: molrec's codec omits it
    // for a cell anchored at the coordinate origin.
    let origin = read_simbox_float_path(store, &format!("{prefix}/origin"))?;
    // An absent `boundary` array is the default `assemble_cell` applies:
    // fully periodic on a defined cell -- the normative default; reading it
    // as vacuum would make one store two physical systems depending on who
    // opened it -- and periodic on no axis on an undefined one.
    let boundary_path = format!("{prefix}/boundary");
    let boundary: Option<Vec<bool>> = match Array::open(store.clone(), &boundary_path) {
        Ok(arr) => {
            Some(arr.retrieve_array_subset(&ArraySubset::new_with_shape(arr.shape().to_vec()))?)
        }
        Err(zarrs::array::ArrayCreateError::MissingMetadata) => None,
        Err(e) => return Err(e.into()),
    };
    assemble_cell(
        cell_defined,
        h.as_deref(),
        origin.as_deref(),
        boundary.as_deref(),
        prefix,
    )
}

/// Read a simbox float array as `Vec<F>` (narrow float arrays are refused —
/// the record has one float), or `None` when the store holds no array at `path`.
///
/// Absence is reported to the caller instead of being raised, because the
/// optional parts of a cell are absent on purpose; every *other* way of failing
/// to open the array is still an error.
fn read_simbox_float_path<S>(store: &Arc<S>, path: &str) -> Result<Option<Vec<F>>, MolRsError>
where
    S: ?Sized + ReadableStorageTraits + 'static,
{
    let arr = match Array::open(store.clone(), path) {
        Ok(arr) => arr,
        Err(zarrs::array::ArrayCreateError::MissingMetadata) => return Ok(None),
        Err(e) => return Err(e.into()),
    };
    let subset = ArraySubset::new_with_shape(arr.shape().to_vec());
    let dt = arr.data_type();
    if dt.is::<Float64DataType>() {
        let data: Vec<f64> = arr.retrieve_array_subset(&subset)?;
        Ok(Some(data))
    } else {
        Err(MolRsError::zarr(format!(
            "simbox array expected float64, got {dt:?}: the record has one float, `F = f64`"
        )))
    }
}

// ---------------------------------------------------------------------------
/// The one reserved child name in a frame group: the cell.
///
/// Rust cannot spell it `box` -- that is a reserved keyword -- but nothing
/// outside Rust source has that problem, so the stored name is `box`.
pub(crate) const BOX_GROUP: &str = "box";

/// The one reserved child name of a *block* group: the subgroup holding the
/// validity masks of that block's nullable columns.
///
/// One leading underscore, not two: Zarr V3 reserves the `__` prefix for node
/// names and `zarrs` enforces it, so `__validity__` is a name no store can
/// carry. A single underscore is legal everywhere and still says, to a reader
/// scanning a block, that this is not a chemistry column.
pub(crate) const VALIDITY_GROUP: &str = "_validity";

/// Check the frame-path layout before a caller replaces any stored data.
pub(in crate::io::mrec::zarr_storage) fn check_storable_frame(
    frame: &Frame,
    prefix: &str,
) -> Result<(), MolRsError> {
    check_local_references(frame, prefix)?;
    if frame.meta.contains_key(META_TYPES_ATTR) {
        return Err(MolRsError::zarr(format!(
            "{META_TYPES_ATTR:?} types a frame group's meta document; a meta key cannot take it"
        )));
    }
    for (name, block) in frame.iter() {
        if name == BOX_GROUP {
            return Err(MolRsError::zarr(format!(
                "{BOX_GROUP:?} names the cell in a frame group; a block cannot take it"
            )));
        }
        // Validate both node names and column declarations before erasing.
        NodePath::new(&join_path(prefix, name)).map_err(|e| MolRsError::zarr(e.to_string()))?;
        check_storable_block(block, &join_path(prefix, name))?;
    }
    Ok(())
}

/// Check the block layout shared by frame sections and force-field tables.
pub(in crate::io::mrec::zarr_storage) fn check_storable_block(
    block: &Block,
    path: &str,
) -> Result<(), MolRsError> {
    if block.contains_key(VALIDITY_GROUP) {
        return Err(MolRsError::zarr(format!(
            "{VALIDITY_GROUP:?} names the validity masks of a block group; a column cannot take it"
        )));
    }
    for (name, column) in block.iter() {
        NodePath::new(&join_path(path, name)).map_err(|e| MolRsError::zarr(e.to_string()))?;
        canonical_width(name, column)?;
    }
    for (name, precision) in block.precisions() {
        if block
            .get(name)
            .is_none_or(|column| column.dtype() != DType::Float)
        {
            return Err(MolRsError::zarr(format!(
                "{path}/{name}: only an f64 column declares a precision"
            )));
        }
        molrs::core::quantum(precision)?;
    }
    Ok(())
}

// Frame (system) write / read — writes all blocks under `{prefix}/`
// ---------------------------------------------------------------------------

#[cfg(feature = "zarr")]
/// Write one [`Frame`] as a Zarr group of blocks.
///
/// # A nullable column carries its mask beside its values
///
/// A block column may carry a [validity
/// mask](crate::core::Block::validity), and the mask is data: without
/// it a row that holds *nothing* reads back as the default filled under it.
/// Each masked column of a block therefore writes one `bool` array, one flag
/// per row, at `<block>/`[`_validity`](VALIDITY_GROUP)`/<column>` — a
/// reserved **subgroup** of the block group, holding the masks of that block
/// and nothing else.
///
/// [`read_frame_group`] skips every non-Array child of a block, so a reader
/// that does not read masks walks past the subgroup instead of taking it for
/// a column; and a block no column of which is masked writes no subgroup at
/// all, and reads back as fully valid.
///
/// `_validity` is reserved among a block group's children the way `box` is
/// among a frame group's: a block carrying a column of that name is refused
/// here rather than silently merged with the masks.
pub(crate) fn write_frame_group(
    store: &ReadableWritableListableStorage,
    prefix: &str,
    frame: &Frame,
) -> Result<(), MolRsError> {
    check_storable_frame(frame, prefix)?;

    // Erase before writing: this group is the target node, so whatever it held
    // before is not part of the frame being written.
    store.erase_prefix(&node_prefix(prefix)?)?;

    // The frame's meta document is this group's attribute map, not a child
    // group: the contract binds document sections as attributes, and a `meta`
    // child would also steal a name from the block namespace.
    //
    // Every value is written in its typed JSON form (a NaN is `"NaN"`, a
    // `u64` past 2^53 a decimal string), and `_meta_types` maps every key to
    // its tag, so a frame reads back with each value at its tag.
    let mut meta_attrs = serde_json::Map::new();
    let mut meta_types = serde_json::Map::new();
    for (k, v) in &frame.meta {
        meta_attrs.insert(k.clone(), v.to_typed_json());
        meta_types.insert(k.clone(), serde_json::Value::from(v.dtype()));
    }
    if !meta_types.is_empty() {
        meta_attrs.insert(
            META_TYPES_ATTR.to_string(),
            serde_json::Value::Object(meta_types),
        );
    }
    GroupBuilder::new()
        .attributes(meta_attrs)
        .build(store.clone(), prefix)?
        .store_metadata()?;

    // The cell. `box` is the one reserved name among a frame group's children.
    if let Some(ref simbox) = frame.simbox {
        write_simbox(store, &format!("{}/{}", prefix, BOX_GROUP), simbox)?;
    }

    // Blocks (atoms, bonds, angles, …). A block with no rows is still a block
    // and still has a count, so it gets a group and an attribute rather than
    // being dropped -- silently losing it would be data loss.
    for (block_name, block) in frame.iter() {
        write_block_group(store, &join_path(prefix, block_name), block)?;
    }

    Ok(())
}

/// Write one [`Block`] as the block group at `group_path`: its `count` (and
/// `structural_shape`, `targets`) attributes, one array per column carrying
/// its declared precision, and the `_validity` masks.
///
/// The one description of a block group every frame-shaped section shares —
/// `frame`, `system` and the `forcefield` style tables
/// ([`crate::io::mrec::zarr_storage::forcefield_io`]).
#[cfg(feature = "zarr")]
pub(crate) fn write_block_group(
    store: &ReadableWritableListableStorage,
    group_path: &str,
    block: &Block,
) -> Result<(), MolRsError> {
    check_storable_block(block, group_path)?;
    let mut block_attrs = serde_json::Map::new();
    block_attrs.insert(
        "count".to_string(),
        serde_json::Value::from(block.n_rows().unwrap_or(0)),
    );
    if let Some(shape) = block.structural_shape() {
        block_attrs.insert(
            "structural_shape".to_string(),
            serde_json::Value::Array(
                shape
                    .iter()
                    .map(|n| serde_json::Value::from(*n as u64))
                    .collect(),
            ),
        );
    }
    let targets: serde_json::Map<String, serde_json::Value> = block
        .targets()
        .map(|(column, target)| (column.to_string(), serde_json::Value::from(target)))
        .collect();
    if !targets.is_empty() {
        block_attrs.insert(
            TARGETS_ATTRIBUTE.to_string(),
            serde_json::Value::Object(targets),
        );
    }
    GroupBuilder::new()
        .attributes(block_attrs)
        .build(store.clone(), group_path)?
        .store_metadata()?;

    for (col_name, col) in block.iter() {
        let arr_path = join_path(group_path, col_name);
        let canonical = canonical_width(col_name, col)?;
        let col = canonical.as_ref().unwrap_or(col);
        write_column(store, &arr_path, col, block.precision(col_name))?;
    }
    write_validity_group(store, group_path, block)
}

/// Write the validity masks of `block` into its reserved `_validity`
/// subgroup, or write nothing at all when no column of it is masked.
///
/// One `bool` array per masked column, named after that column and carrying
/// one flag per row — the same leading row axis as the values it qualifies,
/// and no trailing axes, because a mask marks a *row* null whatever shape the
/// row has.
#[cfg(feature = "zarr")]
fn write_validity_group(
    store: &ReadableWritableListableStorage,
    block_path: &str,
    block: &Block,
) -> Result<(), MolRsError> {
    let masked: Vec<(&str, &[bool])> = block
        .iter()
        .filter_map(|(column, _)| block.validity(column).map(|mask| (column, mask)))
        .collect();
    if masked.is_empty() {
        return Ok(());
    }
    let group_path = format!("{}/{}", block_path, VALIDITY_GROUP);
    GroupBuilder::new()
        .build(store.clone(), &group_path)?
        .store_metadata()?;
    for (column, mask) in masked {
        let flags = ArrayD::from_shape_vec(vec![mask.len()], mask.to_vec()).map_err(shape_err)?;
        write_column(
            store,
            &format!("{}/{}", group_path, column),
            &Column::from(flags),
            None,
        )?;
    }
    Ok(())
}

/// Read one [`Frame`] back from a Zarr group written by [`write_frame_group`].
pub(crate) fn read_frame_group<S>(store: &Arc<S>, prefix: &str) -> Result<Frame, MolRsError>
where
    S: ?Sized + ReadableStorageTraits + ListableStorageTraits + 'static,
{
    let mut frame = Frame::new();

    // Meta lives in this group's own attributes.
    if let Ok(frame_group) = zarrs::group::Group::open(store.clone(), prefix) {
        frame.meta = read_meta_document(frame_group.attributes(), prefix)?;
    }

    // The cell.
    let box_path = format!("{}/{}", prefix, BOX_GROUP);
    if zarrs::group::Group::open(store.clone(), &box_path).is_ok() {
        frame.simbox = Some(read_simbox(store, &box_path)?);
    }

    // Blocks
    let frame_node = Node::open(store, prefix)?;
    for child in frame_node.children() {
        let child_name = child.path().as_str().rsplit('/').next().unwrap_or("");
        if child_name == BOX_GROUP || child_name.is_empty() {
            continue;
        }
        if !matches!(child.metadata(), NodeMetadata::Group(_)) {
            continue;
        }
        let block = read_block_group(store, child.path().as_str(), child_name)?;
        frame.insert(child_name, block);
    }

    check_local_references(&frame, prefix)?;
    Ok(frame)
}

/// Read the block group at `group_path` (named `child_name` in messages)
/// back into a [`Block`]: the inverse of [`write_block_group`].
pub(crate) fn read_block_group<S>(
    store: &Arc<S>,
    group_path: &str,
    child_name: &str,
) -> Result<Block, MolRsError>
where
    S: ?Sized + ReadableStorageTraits + ListableStorageTraits + 'static,
{
    let mut block = Block::new();
    let block_node = Node::open(store, group_path)?;
    for col_child in block_node.children() {
        if !matches!(col_child.metadata(), NodeMetadata::Array(_)) {
            continue;
        }
        let col_path = col_child.path().as_str();
        let col_name = col_path.rsplit('/').next().unwrap_or("");
        // A frame group's column is read whole: the array *is* the column.
        let array = Array::open(store.clone(), col_path)?;
        let whole = ArraySubset::new_with_shape(array.shape().to_vec());
        let col = read_column_array(&array, &whole)?;
        insert_column_into_block(&mut block, col_name, col)?;
        // The declared precision rides on the array. The values are read
        // as stored: a reader neither re-rounds nor checks the grid.
        if let Some(p) = array.attributes().get(PRECISION_ATTRIBUTE) {
            let p = p.as_f64().ok_or_else(|| {
                MolRsError::zarr(format!(
                    "{col_path} carries a {PRECISION_ATTRIBUTE} attribute that is not a \
                     number: {p}"
                ))
            })?;
            block
                .set_precision(col_name, p)
                .map_err(|e| MolRsError::zarr(format!("{col_path}: {e}")))?;
        }
    }
    // `count` is required: a block with no columns still has a row count,
    // and the columns must agree with it.
    let group = zarrs::group::Group::open(store.clone(), group_path)?;
    let attrs = group.attributes();
    let count = match attrs.get("count") {
        Some(count) => count.as_u64().ok_or_else(|| {
            MolRsError::zarr(format!(
                "block {child_name:?}: count must be a non-negative integer, found {count}"
            ))
        })? as usize,
        None => {
            return Err(MolRsError::zarr(format!(
                "block {child_name:?} carries no count attribute; a block group states its \
                 row count"
            )));
        }
    };
    if block.is_empty() {
        block.resize(count);
    } else if block.n_rows() != Some(count) {
        return Err(MolRsError::zarr(format!(
            "row_count_mismatch: block {child_name:?} count={count}, columns have {}",
            block.n_rows().unwrap_or(0)
        )));
    }
    if let Some(shape) = attrs.get("structural_shape").and_then(|v| v.as_array()) {
        let shape: Vec<usize> = shape
            .iter()
            .filter_map(|v| v.as_u64().map(|n| n as usize))
            .collect();
        if !shape.is_empty() {
            block.set_shape(&shape).map_err(|e| {
                MolRsError::zarr(format!(
                    "block {child_name:?} structural_shape {shape:?}: {e}"
                ))
            })?;
        }
    }
    if let Some(targets) = attrs.get(TARGETS_ATTRIBUTE) {
        let targets = targets.as_object().ok_or_else(|| {
            MolRsError::zarr(format!(
                "block {child_name:?}: {TARGETS_ATTRIBUTE} must be an object, found {targets}"
            ))
        })?;
        for (column, target) in targets {
            let target = target.as_str().ok_or_else(|| {
                MolRsError::zarr(format!(
                    "block {child_name:?}: target of {column:?} must be a string, found \
                     {target}"
                ))
            })?;
            block
                .set_target(column, target)
                .map_err(|e| MolRsError::zarr(format!("block {child_name:?} targets: {e}")))?;
        }
    }
    read_validity_group(store, group_path, child_name, &mut block)?;
    Ok(block)
}

/// A frame-shaped group's `meta` document from its attribute map.
///
/// `_meta_types` is taken off the map and never surfaces as a key. A key it
/// types is decoded under its tag and refused in any other form; a key it
/// does not type is inferred ([`MetaValue::from_attr_value`]); a tag whose key is
/// absent is ignored.
///
/// # Errors
///
/// A [`MolRsError::Zarr`] naming the group and key when `_meta_types` is not
/// an object of tag strings, or a typed value is not its tag's form.
pub(crate) fn read_meta_document(
    attrs: &serde_json::Map<String, serde_json::Value>,
    path: &str,
) -> Result<MetaMap, MolRsError> {
    let types = match attrs.get(META_TYPES_ATTR) {
        None => None,
        Some(serde_json::Value::Object(types)) => Some(types),
        Some(other) => {
            return Err(MolRsError::zarr(format!(
                "{path}: {META_TYPES_ATTR} must be an object of tags, found {other}"
            )));
        }
    };
    let mut meta = MetaMap::with_capacity(attrs.len());
    for (key, value) in attrs {
        if key == META_TYPES_ATTR {
            continue;
        }
        let typed = match types.and_then(|types| types.get(key)) {
            None => MetaValue::from_attr_value(value),
            Some(tag) => {
                let tag = tag.as_str().ok_or_else(|| {
                    MolRsError::zarr(format!(
                        "{path}: {META_TYPES_ATTR}[{key:?}] must be a tag string, found {tag}"
                    ))
                })?;
                MetaValue::from_typed_json(tag, value)
                    .map_err(|e| MolRsError::zarr(format!("{path}: meta key {key:?}: {e}")))?
            }
        };
        meta.insert(key.clone(), typed);
    }
    Ok(meta)
}

/// The block-group attribute holding a block's declared row-reference
/// targets (`{column: target}`), written only when the block declares one.
pub(crate) const TARGETS_ATTRIBUTE: &str = "targets";

/// Check every declared row reference of `frame` whose target
/// `rows_of` can resolve: `rows_of(target)` is `None` when the target cannot
/// be checked here (an absolute target, or a section the caller lacks),
/// `Some(None)` when it names a block that does not exist, and
/// `Some(Some(n))` for a block of `n` rows.
///
/// A missing target is refused wherever the referencing block has rows; a
/// non-null value at or past the target's row count is refused. A null row
/// references nothing.
///
/// # Errors
///
/// A [`MolRsError::Zarr`] naming `what`, block, column and target.
pub(crate) fn check_declared_references(
    frame: &Frame,
    what: &str,
    rows_of: &dyn Fn(&str) -> Option<Option<usize>>,
) -> Result<(), MolRsError> {
    for (name, block) in frame.iter() {
        let rows = block.n_rows().unwrap_or(0);
        for (column, target) in block.targets() {
            let Some(target_rows) = rows_of(target) else {
                continue;
            };
            let Some(target_rows) = target_rows else {
                if rows > 0 {
                    return Err(MolRsError::zarr(format!(
                        "{what}: column {column:?} of block {name:?} references {target:?}, \
                         which is not there"
                    )));
                }
                continue;
            };
            let Some(values) = block.get(column).and_then(|c| c.get::<u64>()) else {
                continue;
            };
            let mask = block.validity(column);
            if let Some((row, value)) = values
                .iter()
                .enumerate()
                .find(|&(row, &v)| mask.is_none_or(|m| m[row]) && v as usize >= target_rows)
            {
                return Err(MolRsError::zarr(format!(
                    "{what}: row {row} of column {column:?} of block {name:?} references row \
                     {value} of {target:?}, which has {target_rows} rows"
                )));
            }
        }
    }
    Ok(())
}

/// [`check_declared_references`] for the targets inside `frame` itself.
pub(crate) fn check_local_references(frame: &Frame, what: &str) -> Result<(), MolRsError> {
    check_declared_references(frame, what, &|target| {
        (!target.starts_with('/')).then(|| frame.get(target).map(|b| b.n_rows().unwrap_or(0)))
    })
}

/// Restore the validity masks [`write_validity_group`] wrote for `block`.
///
/// A block group with no `_validity` child is fully valid.
///
/// # Errors
///
/// A [`MolRsError::Zarr`] naming block and column when the mask is not a
/// `bool` array, when it names no column of the block, or when it does not
/// carry exactly one flag per row. A mask that disagrees with its column is a
/// corrupt store: padding or truncating it would invent the very answer the
/// mask exists to give.
fn read_validity_group<S>(
    store: &Arc<S>,
    block_path: &str,
    block_name: &str,
    block: &mut Block,
) -> Result<(), MolRsError>
where
    S: ?Sized + ReadableStorageTraits + ListableStorageTraits + 'static,
{
    let group_path = format!("{}/{}", block_path, VALIDITY_GROUP);
    if zarrs::group::Group::open(store.clone(), &group_path).is_err() {
        return Ok(());
    }
    let rows = block.n_rows().unwrap_or(0);
    for child in Node::open(store, &group_path)?.children() {
        if !matches!(child.metadata(), NodeMetadata::Array(_)) {
            continue;
        }
        let path = child.path().as_str();
        let column = path.rsplit('/').next().unwrap_or("");
        let array = Array::open(store.clone(), path)?;
        let dtype = array.data_type();
        if !dtype.is::<BoolDataType>() {
            return Err(MolRsError::zarr(format!(
                "validity mask of column {column:?} of block {block_name:?} is stored as \
                 {dtype:?}, and a mask is one bool per row"
            )));
        }
        let subset = ArraySubset::new_with_shape(array.shape().to_vec());
        let mask: Vec<bool> = array.retrieve_array_subset(&subset)?;
        if mask.len() != rows {
            return Err(MolRsError::zarr(format!(
                "validity mask of column {column:?} of block {block_name:?} carries {} flags for \
                 {rows} rows",
                mask.len()
            )));
        }
        block.set_validity(column, mask).map_err(|e| {
            MolRsError::zarr(format!(
                "validity mask of column {column:?} of block {block_name:?}: {e}"
            ))
        })?;
    }
    Ok(())
}

// ---------------------------------------------------------------------------
// Primitive array helpers
// ---------------------------------------------------------------------------

/// Write a flat `f64` payload of `shape` — the form the cell's `vectors` and
/// `origin` arrive in, already row-major.
#[cfg(feature = "zarr")]
pub(crate) fn write_f64_array(
    store: &ReadableWritableListableStorage,
    path: &str,
    shape: &[u64],
    data: &[F],
) -> Result<(), MolRsError> {
    let dims: Vec<usize> = shape.iter().map(|&s| s as usize).collect();
    write_typed_array(
        store,
        path,
        ArrayViewD::from_shape(dims, data).map_err(shape_err)?,
        data_type::float64(),
        0.0f64,
        plan(shape, DType::Float.itemsize()),
        frame_codecs(DType::Float, false)?,
        serde_json::Map::new(),
    )
}

fn shape_err(e: impl std::fmt::Display) -> MolRsError {
    MolRsError::zarr(format!("shape error: {}", e))
}

/// The store prefix that holds a node's own metadata key and every descendant,
/// for a node addressed by an absolute Zarr path such as `/frame/atoms`.
///
/// Erasing this prefix is how a writer clears its target before writing: Zarr
/// writes are key-by-key, so a rewrite that only stores the new keys inherits
/// every child of whatever was there before.
#[cfg(feature = "zarr")]
pub(crate) fn node_prefix(path: &str) -> Result<StorePrefix, MolRsError> {
    let trimmed = path.trim_matches('/');
    if trimmed.is_empty() {
        return Ok(StorePrefix::root());
    }
    StorePrefix::new(format!("{trimmed}/"))
        .map_err(|e| MolRsError::zarr(format!("invalid store prefix for {path:?}: {e}")))
}

/// Build a child path from a prefix, avoiding double slashes.
pub(crate) fn join_path(prefix: &str, child: &str) -> String {
    if prefix == "/" {
        format!("/{}", child)
    } else {
        format!("{}/{}", prefix.trim_end_matches('/'), child)
    }
}

#[cfg(all(test, feature = "filesystem"))]
mod tests {
    use super::*;
    use crate::core::Cell;
    use molrs::core::DType;

    use num_complex::Complex;
    use std::ffi::OsStr;
    use std::num::NonZeroU64;
    use std::path::{Path, PathBuf};
    use std::sync::Arc;
    use tempfile::TempDir;
    use zarrs::filesystem::FilesystemStore;

    /// The frame group every round-trip writes into.
    const FRAME: &str = "/frame";
    /// The one block in that frame.
    const BLOCK: &str = "atoms";
    /// A column name outside the schema vocabulary, so no canonical dtype
    /// promotion can mask the arrival width under test.
    const COLUMN: &str = "probe";

    fn storage_in(dir: &TempDir) -> ReadableWritableListableStorage {
        Arc::new(FilesystemStore::new(dir.path()).unwrap())
    }

    /// Write one column as the sole column of a one-block frame and read the
    /// whole frame back, returning the column as it arrived from the store.
    fn round_trip_column(col: Column) -> Column {
        let dir = TempDir::new().unwrap();
        let store = storage_in(&dir);
        let mut block = Block::new();
        block.insert(COLUMN, col, None).unwrap();
        let mut frame = Frame::new();
        frame.insert(BLOCK, block);
        write_frame_group(&store, FRAME, &frame).unwrap();
        read_frame_group(&store, FRAME)
            .unwrap()
            .get(BLOCK)
            .unwrap()
            .get(COLUMN)
            .unwrap()
            .clone()
    }

    // -- the 13-dtype matrix: one test per Column variant -------------------

    #[test]
    fn f64_column_round_trips_at_arrival_width() {
        let values = vec![1.0f64, -2.5, 1.0e-300, f64::MAX];
        let back = round_trip_column(Column::from(
            ArrayD::from_shape_vec(vec![4], values.clone()).unwrap(),
        ));
        assert_eq!(back.dtype(), DType::Float);
        assert_eq!(
            *back.get::<f64>().unwrap(),
            ArrayD::from_shape_vec(vec![4], values).unwrap()
        );
    }

    #[test]
    fn i8_column_round_trips_at_arrival_width() {
        let values = vec![i8::MIN, -1, 0, i8::MAX];
        let back = round_trip_column(Column::from(
            ArrayD::from_shape_vec(vec![4], values.clone()).unwrap(),
        ));
        assert_eq!(back.dtype(), DType::I8);
        assert_eq!(
            *back.get::<i8>().unwrap(),
            ArrayD::from_shape_vec(vec![4], values).unwrap()
        );
    }

    /// A canonical identifier stored narrower than `u64` is refused on read,
    /// not widened — the in-memory insert widens, the store read does not.
    #[test]
    fn a_narrow_canonical_identifier_is_refused_on_read() {
        for (key, column) in [
            (
                "atomi",
                Column::from(ArrayD::from_shape_vec(vec![2], vec![0u32, 1]).unwrap()),
            ),
            (
                "type_id",
                Column::from(ArrayD::from_shape_vec(vec![2], vec![0u8, 1]).unwrap()),
            ),
            (
                "bond_type",
                Column::from(ArrayD::from_shape_vec(vec![2], vec![1u16, 4]).unwrap()),
            ),
        ] {
            let dir = TempDir::new().unwrap();
            let store = storage_in(&dir);
            GroupBuilder::new()
                .build(store.clone(), "/f")
                .unwrap()
                .store_metadata()
                .unwrap();
            let mut count = serde_json::Map::new();
            count.insert("count".into(), serde_json::json!(2));
            GroupBuilder::new()
                .attributes(count)
                .build(store.clone(), "/f/b")
                .unwrap()
                .store_metadata()
                .unwrap();
            write_column(&store, &format!("/f/b/{key}"), &column, None).unwrap();
            let err = read_frame_group(&store, "/f").unwrap_err().to_string();
            assert!(
                err.contains(key) && err.contains(column.dtype().name()),
                "{err}"
            );

            // The same array under a non-canonical name reads at its width.
            write_column(&store, "/f/b/label", &column, None).unwrap();
            std::fs::remove_dir_all(dir.path().join("f/b").join(key)).unwrap();
            let back = read_frame_group(&store, "/f").unwrap();
            assert_eq!(
                back.get("b").unwrap().get("label").unwrap().dtype(),
                column.dtype()
            );
        }

        // In memory the same insert widens to u64.
        let mut block = Block::new();
        block
            .insert(
                "atomi",
                Column::from(ArrayD::from_shape_vec(vec![1], vec![7u32]).unwrap()),
                None,
            )
            .unwrap();
        assert_eq!(block.get("atomi").unwrap().dtype(), DType::Uint);
    }

    #[test]
    fn i16_column_round_trips_at_arrival_width() {
        let values = vec![i16::MIN, -1, 0, i16::MAX];
        let back = round_trip_column(Column::from(
            ArrayD::from_shape_vec(vec![4], values.clone()).unwrap(),
        ));
        assert_eq!(back.dtype(), DType::I16);
        assert_eq!(
            *back.get::<i16>().unwrap(),
            ArrayD::from_shape_vec(vec![4], values).unwrap()
        );
    }

    #[test]
    fn i32_column_round_trips_at_arrival_width() {
        let values = vec![i32::MIN, -1, 0, i32::MAX];
        let back = round_trip_column(Column::from(
            ArrayD::from_shape_vec(vec![4], values.clone()).unwrap(),
        ));
        assert_eq!(back.dtype(), DType::Int);
        assert_eq!(
            *back.get::<i32>().unwrap(),
            ArrayD::from_shape_vec(vec![4], values).unwrap()
        );
    }

    #[test]
    fn i64_column_round_trips_at_arrival_width() {
        let values = vec![i64::MIN, -1, 0, i64::MAX];
        let back = round_trip_column(Column::from(
            ArrayD::from_shape_vec(vec![4], values.clone()).unwrap(),
        ));
        assert_eq!(back.dtype(), DType::I64);
        assert_eq!(
            *back.get::<i64>().unwrap(),
            ArrayD::from_shape_vec(vec![4], values).unwrap()
        );
    }

    #[test]
    fn u8_column_round_trips_at_arrival_width() {
        let values = vec![0u8, 1, 128, u8::MAX];
        let back = round_trip_column(Column::from(
            ArrayD::from_shape_vec(vec![4], values.clone()).unwrap(),
        ));
        assert_eq!(back.dtype(), DType::U8);
        assert_eq!(
            *back.get::<u8>().unwrap(),
            ArrayD::from_shape_vec(vec![4], values).unwrap()
        );
    }

    #[test]
    fn u16_column_round_trips_at_arrival_width() {
        let values = vec![0u16, 1, 40000, u16::MAX];
        let back = round_trip_column(Column::from(
            ArrayD::from_shape_vec(vec![4], values.clone()).unwrap(),
        ));
        assert_eq!(back.dtype(), DType::U16);
        assert_eq!(
            *back.get::<u16>().unwrap(),
            ArrayD::from_shape_vec(vec![4], values).unwrap()
        );
    }

    #[test]
    fn u32_column_round_trips_at_arrival_width() {
        let values = vec![0u32, 1, 3_000_000_000, u32::MAX];
        let back = round_trip_column(Column::from(
            ArrayD::from_shape_vec(vec![4], values.clone()).unwrap(),
        ));
        assert_eq!(back.dtype(), DType::U32);
        assert_eq!(
            *back.get::<u32>().unwrap(),
            ArrayD::from_shape_vec(vec![4], values).unwrap()
        );
    }

    #[test]
    fn u64_column_round_trips_at_arrival_width() {
        let values = vec![0u64, 1, 9_007_199_254_740_993, u64::MAX];
        let back = round_trip_column(Column::from(
            ArrayD::from_shape_vec(vec![4], values.clone()).unwrap(),
        ));
        assert_eq!(back.dtype(), DType::Uint);
        assert_eq!(
            *back.get::<u64>().unwrap(),
            ArrayD::from_shape_vec(vec![4], values).unwrap()
        );
    }

    #[test]
    fn bool_column_round_trips_at_arrival_width() {
        let values = vec![true, false, true, true];
        let back = round_trip_column(Column::from(
            ArrayD::from_shape_vec(vec![4], values.clone()).unwrap(),
        ));
        assert_eq!(back.dtype(), DType::Bool);
        assert_eq!(
            *back.get::<bool>().unwrap(),
            ArrayD::from_shape_vec(vec![4], values).unwrap()
        );
    }

    #[test]
    fn string_column_round_trips_at_arrival_width() {
        let values = vec![
            "atom".to_string(),
            String::new(),
            "Ω".to_string(),
            "x y".to_string(),
        ];
        let back = round_trip_column(Column::from(
            ArrayD::from_shape_vec(vec![4], values.clone()).unwrap(),
        ));
        assert_eq!(back.dtype(), DType::String);
        assert_eq!(
            *back.get::<String>().unwrap(),
            ArrayD::from_shape_vec(vec![4], values).unwrap()
        );
    }

    #[test]
    fn c64_column_round_trips_at_arrival_width() {
        let values = vec![
            Complex::new(1.5f32, -2.5f32),
            Complex::new(0.0f32, 0.0f32),
            Complex::new(-0.125f32, 65536.0f32),
            Complex::new(3.4028235e38f32, 1.1754944e-38f32),
        ];
        let back = round_trip_column(Column::from(
            ArrayD::from_shape_vec(vec![4], values.clone()).unwrap(),
        ));
        assert_eq!(back.dtype(), DType::C64);
        assert_eq!(
            *back.get::<num_complex::Complex<f32>>().unwrap(),
            ArrayD::from_shape_vec(vec![4], values).unwrap()
        );
    }

    #[test]
    fn c128_column_round_trips_at_arrival_width() {
        let values = vec![
            Complex::new(1.5f64, -2.5f64),
            Complex::new(0.0f64, 0.0f64),
            Complex::new(-0.125f64, 65536.0f64),
            Complex::new(f64::MAX, 1.0e-300f64),
        ];
        let back = round_trip_column(Column::from(
            ArrayD::from_shape_vec(vec![4], values.clone()).unwrap(),
        ));
        assert_eq!(back.dtype(), DType::C128);
        assert_eq!(
            *back.get::<num_complex::Complex<f64>>().unwrap(),
            ArrayD::from_shape_vec(vec![4], values).unwrap()
        );
    }

    // -- block-level shape and count ----------------------------------------

    #[test]
    fn structural_shape_round_trips_with_the_block() {
        let dir = TempDir::new().unwrap();
        let store = storage_in(&dir);
        let mut block = Block::new();
        block
            .insert(
                COLUMN,
                Column::from(
                    ArrayD::from_shape_vec(vec![6], vec![0.0, 1.0, 2.0, 3.0, 4.0, 5.0]).unwrap(),
                ),
                None,
            )
            .unwrap();
        block.set_shape(&[3, 2]).unwrap();
        let mut frame = Frame::new();
        frame.insert(BLOCK, block);
        write_frame_group(&store, FRAME, &frame).unwrap();

        let back = read_frame_group(&store, FRAME).unwrap();
        assert_eq!(
            back.get(BLOCK).unwrap().structural_shape(),
            Some(&[3usize, 2usize][..])
        );
    }

    #[test]
    fn structural_shape_is_written_as_a_group_attribute() {
        let dir = TempDir::new().unwrap();
        let store = storage_in(&dir);
        let mut block = Block::new();
        block
            .insert(
                COLUMN,
                Column::from(
                    ArrayD::from_shape_vec(vec![6], vec![0.0, 1.0, 2.0, 3.0, 4.0, 5.0]).unwrap(),
                ),
                None,
            )
            .unwrap();
        block.set_shape(&[3, 2]).unwrap();
        let mut frame = Frame::new();
        frame.insert(BLOCK, block);
        write_frame_group(&store, FRAME, &frame).unwrap();

        let group = zarrs::group::Group::open(store.clone(), &format!("{FRAME}/{BLOCK}")).unwrap();
        assert_eq!(
            group.attributes().get("structural_shape"),
            Some(&serde_json::json!([3, 2]))
        );
    }

    #[test]
    fn column_less_block_keeps_its_row_count() {
        let dir = TempDir::new().unwrap();
        let store = storage_in(&dir);
        let mut block = Block::new();
        block.resize(7);
        let mut frame = Frame::new();
        frame.insert("ghost", block);
        write_frame_group(&store, FRAME, &frame).unwrap();

        let back = read_frame_group(&store, FRAME).unwrap();
        assert_eq!(back.get("ghost").unwrap().n_rows(), Some(7));
    }

    // -- bool is native, not a tagged uint8 (ac-005) -------------------------

    #[test]
    fn bool_column_is_written_as_the_native_zarr_bool_dtype() {
        let dir = TempDir::new().unwrap();
        let store = storage_in(&dir);
        let mut block = Block::new();
        block
            .insert(
                COLUMN,
                Column::from(ArrayD::from_shape_vec(vec![3], vec![true, false, true]).unwrap()),
                None,
            )
            .unwrap();
        let mut frame = Frame::new();
        frame.insert(BLOCK, block);
        write_frame_group(&store, FRAME, &frame).unwrap();

        let arr = Array::open(store.clone(), &format!("{FRAME}/{BLOCK}/{COLUMN}")).unwrap();
        assert!(arr.data_type().is::<BoolDataType>());
    }

    #[test]
    fn bool_column_carries_no_molrs_dtype_attribute() {
        let dir = TempDir::new().unwrap();
        let store = storage_in(&dir);
        let mut block = Block::new();
        block
            .insert(
                COLUMN,
                Column::from(ArrayD::from_shape_vec(vec![3], vec![true, false, true]).unwrap()),
                None,
            )
            .unwrap();
        let mut frame = Frame::new();
        frame.insert(BLOCK, block);
        write_frame_group(&store, FRAME, &frame).unwrap();

        let arr = Array::open(store.clone(), &format!("{FRAME}/{BLOCK}/{COLUMN}")).unwrap();
        assert_eq!(arr.attributes().get("molrs_dtype"), None);
    }

    // -- the cell: boundary / origin defaults (ac-006) -----------------------

    /// The one cell shape every box test writes: an orthogonal 10/11/12 cell.
    fn cell_vectors() -> [F; 9] {
        [10.0, 0.0, 0.0, 0.0, 11.0, 0.0, 0.0, 0.0, 12.0]
    }

    #[test]
    fn absent_boundary_reads_all_periodic() {
        let dir = TempDir::new().unwrap();
        let store = storage_in(&dir);
        // Hand-built: the writer always emits `boundary`, but molrec's own
        // codec omits it for the default case, and that store must read as
        // fully periodic rather than as vacuum.
        GroupBuilder::new()
            .build(store.clone(), "/box")
            .unwrap()
            .store_metadata()
            .unwrap();
        write_f64_array(&store, "/box/vectors", &[3, 3], &cell_vectors()).unwrap();
        write_f64_array(&store, "/box/origin", &[3], &[0.0, 0.0, 0.0]).unwrap();

        let simbox = read_simbox(&store, "/box").unwrap();
        assert_eq!(simbox.pbc().to_vec(), vec![true, true, true]);
    }

    #[test]
    fn explicit_boundary_round_trips_unchanged() {
        let dir = TempDir::new().unwrap();
        let store = storage_in(&dir);
        let simbox = SimBox::new(
            Cell::Matrix([[10.0, 0.0, 0.0], [0.0, 11.0, 0.0], [0.0, 0.0, 12.0]]),
            [0.0, 0.0, 0.0],
            [true, false, true],
        )
        .unwrap();
        write_simbox(&store, "/box", &simbox).unwrap();

        let back = read_simbox(&store, "/box").unwrap();
        assert_eq!(back.pbc().to_vec(), vec![true, false, true]);
    }

    #[test]
    fn defined_cell_stays_defined_across_the_round_trip() {
        let dir = TempDir::new().unwrap();
        let store = storage_in(&dir);
        let simbox = SimBox::new(
            Cell::Matrix([[10.0, 0.0, 0.0], [0.0, 11.0, 0.0], [0.0, 0.0, 12.0]]),
            [0.0, 0.0, 0.0],
            [true, false, true],
        )
        .unwrap();
        write_simbox(&store, "/box", &simbox).unwrap();

        assert!(read_simbox(&store, "/box").unwrap().is_cell_defined());
    }

    #[test]
    fn undefined_cell_round_trips_as_false() {
        let dir = TempDir::new().unwrap();
        let store = storage_in(&dir);
        // A no-cell box carries the identity matrix, per `Cell::Undefined`:
        // geometry ops degrade to no-ops and only the flag says "undefined".
        let simbox = SimBox::new(Cell::Undefined, [0.0, 0.0, 0.0], [false, false, false]).unwrap();
        write_simbox(&store, "/box", &simbox).unwrap();

        // `cell_defined` is emitted in this direction only, so it has to be
        // literally present and literally `false` -- an absent attribute is
        // read back as a *defined* cell, which is the silent loss this pins.
        let group = zarrs::group::Group::open(store.clone(), "/box").unwrap();
        assert_eq!(
            group.attributes().get("cell_defined"),
            Some(&serde_json::json!(false))
        );

        let back = read_simbox(&store, "/box").unwrap();
        assert!(!back.is_cell_defined());
    }

    /// An undefined cell's `vectors` are ignored on read: a zero matrix (or
    /// any other) is accepted, never inverted, and comes back as the identity
    /// — which is also what the writer emits for it.
    #[test]
    fn undefined_cell_accepts_any_vectors_and_writes_the_identity() {
        let dir = TempDir::new().unwrap();
        let store = storage_in(&dir);
        let mut attrs = serde_json::Map::new();
        attrs.insert("cell_defined".into(), false.into());
        GroupBuilder::new()
            .attributes(attrs)
            .build(store.clone(), "/box")
            .unwrap()
            .store_metadata()
            .unwrap();
        write_f64_array(&store, "/box/vectors", &[3, 3], &[0.0; 9]).unwrap();

        let back = read_simbox(&store, "/box").unwrap();
        assert!(!back.is_cell_defined());
        assert_eq!(
            back.matrix(),
            [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]]
        );

        write_simbox(&store, "/rewritten", &back).unwrap();
        let written = read_simbox_float_path(&store, "/rewritten/vectors")
            .unwrap()
            .unwrap();
        assert_eq!(written, vec![1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0]);
    }

    /// An undefined cell is periodic on no axis: its all-false boundary is
    /// written explicitly, an omitted one reads all-false, and a periodic flag
    /// in the store is refused (no `SimBox` can carry one to write).
    #[test]
    fn an_undefined_cell_is_periodic_on_no_axis() {
        let dir = TempDir::new().unwrap();
        let store = storage_in(&dir);
        let cell = |pbc| SimBox::new(Cell::Undefined, [0.0, 0.0, 0.0], pbc);

        write_simbox(&store, "/box", &cell([false; 3]).unwrap()).unwrap();
        assert!(Array::open(store.clone(), "/box/boundary").is_ok());
        assert_eq!(read_simbox(&store, "/box").unwrap().pbc(), [false; 3]);

        // Omitted boundary on an undefined cell: all-false, not periodic.
        std::fs::remove_dir_all(dir.path().join("box/boundary")).unwrap();
        assert_eq!(read_simbox(&store, "/box").unwrap().pbc(), [false; 3]);

        // A periodic flag stored on an undefined cell is malformed.
        write_column(
            &store,
            "/box/boundary",
            &Column::from(ArrayD::from_shape_vec(vec![3], vec![false, true, false]).unwrap()),
            None,
        )
        .unwrap();
        let err = read_simbox(&store, "/box").unwrap_err().to_string();
        assert!(err.contains("undefined cell"), "{err}");
    }

    /// The box group's attributes are read strictly: a `cell_defined` that is
    /// not a bool is malformed, never read as "defined".
    #[test]
    fn a_non_bool_cell_defined_is_refused() {
        let dir = TempDir::new().unwrap();
        let store = storage_in(&dir);
        let mut attrs = serde_json::Map::new();
        attrs.insert("cell_defined".into(), serde_json::json!("no"));
        GroupBuilder::new()
            .attributes(attrs)
            .build(store.clone(), "/box")
            .unwrap()
            .store_metadata()
            .unwrap();
        write_f64_array(&store, "/box/vectors", &[3, 3], &[0.0; 9]).unwrap();
        let err = read_simbox(&store, "/box").unwrap_err().to_string();
        assert!(err.contains("cell_defined must be a bool"), "{err}");
    }

    /// The boundary lives in the `boundary` array only: a group attribute of
    /// that name is not a second spelling of it.
    #[test]
    fn a_boundary_attribute_is_not_read() {
        let dir = TempDir::new().unwrap();
        let store = storage_in(&dir);
        let mut attrs = serde_json::Map::new();
        attrs.insert("boundary".into(), serde_json::json!([false, false, false]));
        GroupBuilder::new()
            .attributes(attrs)
            .build(store.clone(), "/box")
            .unwrap()
            .store_metadata()
            .unwrap();
        let h = [10.0, 0.0, 0.0, 0.0, 10.0, 0.0, 0.0, 0.0, 10.0];
        write_f64_array(&store, "/box/vectors", &[3, 3], &h).unwrap();
        assert_eq!(read_simbox(&store, "/box").unwrap().pbc(), [true; 3]);
    }

    /// A *defined* cell with a singular matrix is still refused.
    #[test]
    fn a_defined_singular_cell_is_refused() {
        let dir = TempDir::new().unwrap();
        let store = storage_in(&dir);
        GroupBuilder::new()
            .build(store.clone(), "/box")
            .unwrap()
            .store_metadata()
            .unwrap();
        write_f64_array(&store, "/box/vectors", &[3, 3], &[0.0; 9]).unwrap();
        assert!(read_simbox(&store, "/box").is_err());
    }

    #[test]
    fn absent_origin_reads_zero_origin() {
        let dir = TempDir::new().unwrap();
        let store = storage_in(&dir);
        let mut attrs = serde_json::Map::new();
        attrs.insert(
            "boundary".to_string(),
            serde_json::json!([true, true, true]),
        );
        GroupBuilder::new()
            .attributes(attrs)
            .build(store.clone(), "/box")
            .unwrap()
            .store_metadata()
            .unwrap();
        write_f64_array(&store, "/box/vectors", &[3, 3], &cell_vectors()).unwrap();

        let simbox = read_simbox(&store, "/box").unwrap();
        assert_eq!(simbox.origin().to_vec(), vec![0.0, 0.0, 0.0]);
    }

    // -- sharding: many inner chunks, one file (ac-010's on-disk half) -------

    /// Every file under `dir`, recursively, that is not a `zarr.json` — the
    /// data files an array actually costs the filesystem.
    fn data_files(dir: &Path) -> Vec<PathBuf> {
        let mut out = Vec::new();
        let mut stack = vec![dir.to_path_buf()];
        while let Some(current) = stack.pop() {
            for entry in std::fs::read_dir(&current).unwrap().flatten() {
                let path = entry.path();
                if path.is_dir() {
                    stack.push(path);
                } else if path.file_name() != Some(OsStr::new("zarr.json")) {
                    out.push(path);
                }
            }
        }
        out.sort();
        out
    }

    /// A fixed-size column past `SHARD_ABOVE` is laid out as **one** shard file
    /// holding all seven inner chunks, and every value survives bit exact.
    ///
    /// This is the on-disk half of the plan goldens: `chunking::plan` deciding
    /// `shards = Some(..)` is only worth anything if `write_typed_array` then
    /// turns that decision into a single file rather than seven.
    #[test]
    fn a_five_chunk_column_shards_and_round_trips() {
        // chunking::plan(&[400_000], Some(8)), hard-coded from its goldens:
        // an 8 B row gives 512 KiB / 8 = 65 536 rows per inner chunk, and
        // ceil(400 000 / 65 536) = 7 chunks clears SHARD_ABOVE = 4, so the
        // shard spans 7 * 65 536 = 458 752 rows.
        const ROWS: usize = 400_000;
        const INNER_CHUNK_ROWS: u64 = 65_536;
        const SHARD_ROWS: u64 = 458_752;

        let dir = TempDir::new().unwrap();
        let store = storage_in(&dir);
        // Index-derived and non-repeating: a constant column would read back
        // correctly even if the shard handed out the wrong chunk.
        let values: Vec<F> = (0..ROWS).map(|i| i as F * 0.5 - 1.0e-3).collect();
        let mut block = Block::new();
        block
            .insert(
                COLUMN,
                Column::from(ArrayD::from_shape_vec(vec![ROWS], values.clone()).unwrap()),
                None,
            )
            .unwrap();
        let mut frame = Frame::new();
        frame.insert(BLOCK, block);
        write_frame_group(&store, FRAME, &frame).unwrap();

        // Under sharding the array's own chunk extent *is* the shard, and the
        // planned chunk is the subchunk inside it.
        let arr = Array::open(store.clone(), &format!("{FRAME}/{BLOCK}/{COLUMN}")).unwrap();
        assert_eq!(
            arr.chunk_shape(&[0]).unwrap(),
            vec![NonZeroU64::new(SHARD_ROWS).unwrap()],
            "the array chunk extent must be the planned shard"
        );
        assert_eq!(arr.chunk_grid_shape(), &[1], "one shard covers the array");
        let metadata = serde_json::to_value(arr.metadata()).unwrap();
        let sharding = metadata["codecs"]
            .as_array()
            .unwrap()
            .iter()
            .find(|codec| codec["name"] == "sharding_indexed")
            .expect("a seven-chunk array must carry the sharding codec");
        assert_eq!(
            sharding["configuration"]["chunk_shape"],
            serde_json::json!([INNER_CHUNK_ROWS]),
            "the inner chunk must be the planned chunk"
        );

        // The whole point of a shard: seven chunks, one file.
        let files = data_files(&dir.path().join("frame").join(BLOCK).join(COLUMN));
        assert_eq!(
            files.len(),
            1,
            "a sharded column is one data file, found {files:?}"
        );

        let back = read_frame_group(&store, FRAME).unwrap();
        assert_eq!(
            *back
                .get(BLOCK)
                .unwrap()
                .get(COLUMN)
                .unwrap()
                .get::<f64>()
                .unwrap(),
            ArrayD::from_shape_vec(vec![ROWS], values).unwrap(),
            "every row must come back bit exact through the shard"
        );
    }

    // -- nullable columns: the mask is data, not a view --------------------

    /// The reserved child name of a block group that holds its masks.
    ///
    /// Spelled out rather than imported from `super`: this is a name on disk
    /// that molrec and molvis read, so a rename of the Rust constant must
    /// break these tests rather than travel silently into them — the same
    /// reading `sequence`'s `SCHEMA_ATTRIBUTE` takes on its own name.
    ///
    /// **Blocked as spelled.** Zarr V3 reserves the `__` prefix for node
    /// names, and `zarrs` enforces it (`NodeName::validate`): every `Array` /
    /// `Group` / `Node` door refuses `/frame/atoms/__validity__` with a
    /// `NodePathError`, so the masks cannot be written under this name at all.
    /// The reserved name has to be one Zarr admits — `validity` or
    /// `_validity` — and this constant is the single place to change.
    const VALIDITY_GROUP: &str = "_validity";

    /// The one mask every nullable fixture carries: three rows, the first
    /// valid and the last two null. Asymmetric on purpose — a reversed or
    /// all-false mask cannot match it by accident.
    const MASK: [bool; 3] = [true, false, false];

    /// Write `block` as the sole block of a frame and read that frame back.
    fn round_trip_block(block: Block) -> Block {
        let dir = TempDir::new().unwrap();
        let store = storage_in(&dir);
        let mut frame = Frame::new();
        frame.insert(BLOCK, block);
        write_frame_group(&store, FRAME, &frame).unwrap();
        read_frame_group(&store, FRAME)
            .unwrap()
            .get(BLOCK)
            .expect("the frame carries its block")
            .clone()
    }

    /// The names of a block group's children, enumerated the way
    /// [`read_frame_group`] enumerates them.
    fn block_child_names(store: &ReadableWritableListableStorage, block: &str) -> Vec<String> {
        Node::open(store, &format!("{FRAME}/{block}"))
            .expect("the block group exists")
            .children()
            .iter()
            .map(|child| {
                child
                    .path()
                    .as_str()
                    .rsplit('/')
                    .next()
                    .unwrap_or("")
                    .to_string()
            })
            .collect()
    }

    /// A three-row `Int` column whose last two rows hold nothing.
    fn masked_int_block() -> Block {
        let mut block = Block::new();
        block
            .insert(
                COLUMN,
                ArrayD::from_shape_vec(vec![3], vec![7i32, 0, 0]).unwrap(),
                Some(MASK.to_vec()),
            )
            .unwrap();
        block
    }

    /// A three-row `F64` column whose last two rows hold nothing.
    fn masked_f64_block() -> Block {
        let mut block = Block::new();
        block
            .insert(
                COLUMN,
                ArrayD::from_shape_vec(vec![3], vec![0.5f64, 0.0, 0.0]).unwrap(),
                Some(MASK.to_vec()),
            )
            .unwrap();
        block
    }

    /// A three-row `Str` column whose last two rows hold nothing.
    fn masked_string_block() -> Block {
        let mut block = Block::new();
        block
            .insert(
                COLUMN,
                ArrayD::from_shape_vec(
                    vec![3],
                    vec!["C".to_string(), String::new(), String::new()],
                )
                .unwrap(),
                Some(MASK.to_vec()),
            )
            .unwrap();
        block
    }

    #[test]
    fn int_column_keeps_its_validity_mask_across_the_frame_round_trip() {
        let back = round_trip_block(masked_int_block());
        assert_eq!(back.validity(COLUMN), Some(&MASK[..]));
    }

    #[test]
    fn int_column_keeps_its_values_beside_its_mask() {
        let back = round_trip_block(masked_int_block());
        assert_eq!(
            *back.get(COLUMN).unwrap().get::<i32>().unwrap(),
            ArrayD::from_shape_vec(vec![3], vec![7i32, 0, 0]).unwrap()
        );
    }

    #[test]
    fn f64_column_keeps_its_validity_mask_across_the_frame_round_trip() {
        let back = round_trip_block(masked_f64_block());
        assert_eq!(back.validity(COLUMN), Some(&MASK[..]));
    }

    #[test]
    fn f64_column_keeps_its_values_beside_its_mask() {
        let back = round_trip_block(masked_f64_block());
        assert_eq!(
            *back.get(COLUMN).unwrap().get::<f64>().unwrap(),
            ArrayD::from_shape_vec(vec![3], vec![0.5f64, 0.0, 0.0]).unwrap()
        );
    }

    #[test]
    fn string_column_keeps_its_validity_mask_across_the_frame_round_trip() {
        let back = round_trip_block(masked_string_block());
        assert_eq!(back.validity(COLUMN), Some(&MASK[..]));
    }

    #[test]
    fn string_column_keeps_its_values_beside_its_mask() {
        let back = round_trip_block(masked_string_block());
        assert_eq!(
            *back.get(COLUMN).unwrap().get::<String>().unwrap(),
            ArrayD::from_shape_vec(vec![3], vec!["C".to_string(), String::new(), String::new()])
                .unwrap()
        );
    }

    /// A column nobody masked stays unmasked: `validity` says `Some` **iff**
    /// a row is null, so a round trip that invented an all-true mask would be
    /// a different block than the one written.
    #[test]
    fn a_fully_valid_column_reads_back_with_no_mask() {
        let mut block = Block::new();
        block
            .insert(
                COLUMN,
                Column::from(ArrayD::from_shape_vec(vec![3], vec![1.0, 2.0, 3.0]).unwrap()),
                None,
            )
            .unwrap();
        let back = round_trip_block(block);
        assert_eq!(back.validity(COLUMN), None);
    }

    /// No mask, no subgroup: a frame with no masked column writes no
    /// validity child at all.
    #[test]
    fn a_frame_with_no_masked_column_writes_no_validity_child() {
        let dir = TempDir::new().unwrap();
        let store = storage_in(&dir);
        let mut block = Block::new();
        block
            .insert(
                COLUMN,
                Column::from(ArrayD::from_shape_vec(vec![3], vec![1.0, 2.0, 3.0]).unwrap()),
                None,
            )
            .unwrap();
        let mut frame = Frame::new();
        frame.insert(BLOCK, block);
        write_frame_group(&store, FRAME, &frame).unwrap();

        assert_eq!(block_child_names(&store, BLOCK), vec![COLUMN.to_string()]);
    }

    /// The mask lives in a reserved **subgroup**, not in a sibling array: a
    /// reader that does not read masks skips non-Array children of a block
    /// group, so it ignores the subgroup instead of taking it for a column.
    #[test]
    fn a_masked_column_writes_its_mask_into_a_validity_subgroup() {
        let dir = TempDir::new().unwrap();
        let store = storage_in(&dir);
        let mut frame = Frame::new();
        frame.insert(BLOCK, masked_int_block());
        write_frame_group(&store, FRAME, &frame).unwrap();

        let child = Node::open(&store, &format!("{FRAME}/{BLOCK}/{VALIDITY_GROUP}"))
            .expect("the mask subgroup exists");
        assert!(
            matches!(child.metadata(), NodeMetadata::Group(_)),
            "{VALIDITY_GROUP} must be a group, so a reader that skips groups skips it"
        );
    }

    /// Inside that subgroup the mask is a boolean array under the column's
    /// own name — one mask per masked column, addressable without an index.
    #[test]
    fn a_mask_is_stored_as_a_bool_array_named_after_its_column() {
        let dir = TempDir::new().unwrap();
        let store = storage_in(&dir);
        let mut frame = Frame::new();
        frame.insert(BLOCK, masked_int_block());
        write_frame_group(&store, FRAME, &frame).unwrap();

        let path = format!("{FRAME}/{BLOCK}/{VALIDITY_GROUP}/{COLUMN}");
        let arr = Array::open(store.clone(), &path).expect("the mask array exists");
        assert!(arr.data_type().is::<BoolDataType>());
        let subset = ArraySubset::new_with_shape(arr.shape().to_vec());
        let stored: Vec<bool> = arr.retrieve_array_subset(&subset).unwrap();
        assert_eq!(stored, MASK.to_vec());
    }

    /// The mask subgroup's name is reserved among a block group's children
    /// exactly as `box` is among a frame group's: a column of that name would
    /// collide with the masks, so the write is refused rather than silently
    /// reshaped.
    ///
    /// Passes today only because Zarr refuses the `__` prefix outright (see
    /// [`VALIDITY_GROUP`]) — it becomes a real assertion the moment the
    /// reserved name is one Zarr admits.
    #[test]
    fn a_block_column_named_validity_is_refused_on_write() {
        let dir = TempDir::new().unwrap();
        let store = storage_in(&dir);
        let mut block = Block::new();
        block
            .insert(
                VALIDITY_GROUP,
                Column::from(ArrayD::from_shape_vec(vec![3], vec![1.0, 2.0, 3.0]).unwrap()),
                None,
            )
            .unwrap();
        let mut frame = Frame::new();
        frame.insert(BLOCK, block);

        let err = write_frame_group(&store, FRAME, &frame)
            .expect_err("a column named __validity__ collides with the mask subgroup")
            .to_string();
        assert!(err.contains(VALIDITY_GROUP), "{err}");
    }

    /// A mask that does not cover its column's rows is a corrupt store, not a
    /// mask to pad or truncate — and the error has to say which column of
    /// which block, because that is all the operator can act on.
    #[test]
    fn a_mask_of_the_wrong_length_is_a_read_error_naming_block_and_column() {
        let dir = TempDir::new().unwrap();
        let store = storage_in(&dir);
        let mut frame = Frame::new();
        frame.insert(BLOCK, masked_int_block());
        write_frame_group(&store, FRAME, &frame).unwrap();

        // Two flags for a three-row column: the one damage no reader can
        // resolve on its own.
        let path = format!("{FRAME}/{BLOCK}/{VALIDITY_GROUP}/{COLUMN}");
        store.erase_prefix(&node_prefix(&path).unwrap()).unwrap();
        write_column(
            &store,
            &path,
            &Column::from(ArrayD::from_shape_vec(vec![2], vec![true, false]).unwrap()),
            None,
        )
        .unwrap();

        let err = read_frame_group(&store, FRAME)
            .expect_err("a mask that does not cover its column must not read")
            .to_string();
        assert!(err.contains(BLOCK) && err.contains(COLUMN), "{err}");
    }

    #[test]
    fn frame_group_attributes_round_trip_meta_keys_in_insertion_order() {
        let dir = TempDir::new().unwrap();
        let store = storage_in(&dir);
        let mut frame = Frame::new();
        frame.meta.insert("z", "Z");
        frame.meta.insert("a", "A");
        frame.meta.insert("m", "M");
        write_frame_group(&store, FRAME, &frame).unwrap();

        let group = zarrs::group::Group::open(store.clone(), FRAME).unwrap();
        assert_eq!(
            group
                .attributes()
                .keys()
                .map(String::as_str)
                .collect::<Vec<_>>(),
            vec!["z", "a", "m", META_TYPES_ATTR]
        );

        let back = read_frame_group(&store, FRAME).unwrap();
        assert_eq!(
            back.meta.keys().map(String::as_str).collect::<Vec<_>>(),
            vec!["z", "a", "m"]
        );
    }

    // -- declared precision (molrec F1) ------------------------------------

    /// Write one `f64` column of `values` declaring precision `p`, and read
    /// the frame back.
    fn round_trip_precise(values: &[f64], p: f64) -> (Block, Vec<String>) {
        let dir = TempDir::new().unwrap();
        let store = storage_in(&dir);
        let mut block = Block::new();
        block
            .insert(
                COLUMN,
                Column::from(ArrayD::from_shape_vec(vec![values.len()], values.to_vec()).unwrap()),
                None,
            )
            .unwrap();
        block.set_precision(COLUMN, p).unwrap();
        let mut frame = Frame::new();
        frame.insert(BLOCK, block);
        write_frame_group(&store, FRAME, &frame).unwrap();

        let array = Array::open(store.clone(), &format!("{FRAME}/{BLOCK}/{COLUMN}")).unwrap();
        let metadata = serde_json::to_value(array.metadata()).unwrap();
        let mut codecs: Vec<String> = Vec::new();
        let mut collect = |list: &serde_json::Value| {
            for codec in list.as_array().unwrap() {
                // A codec without configuration may be spelled by name alone.
                let name = codec.as_str().or_else(|| codec["name"].as_str()).unwrap();
                codecs.push(name.to_string());
            }
        };
        match metadata["codecs"]
            .as_array()
            .unwrap()
            .iter()
            .find(|codec| codec["name"] == "sharding_indexed")
        {
            Some(sharding) => collect(&sharding["configuration"]["codecs"]),
            None => collect(&metadata["codecs"]),
        }
        let back = read_frame_group(&store, FRAME).unwrap();
        (back.get(BLOCK).unwrap().clone(), codecs)
    }

    fn values_of(block: &Block) -> Vec<f64> {
        block
            .get(COLUMN)
            .unwrap()
            .get::<f64>()
            .unwrap()
            .iter()
            .copied()
            .collect()
    }

    #[test]
    fn a_precision_column_lands_rounded_shuffled_and_zstd_compressed() {
        let p = 1e-3;
        let q = molrs::core::quantum(p).unwrap();
        let values = [0.123_456_7, -12.345_678, 39.999_9, 1.0e-7];
        let (back, codecs) = round_trip_precise(&values, p);
        let expected: Vec<f64> = values
            .iter()
            .map(|&x| molrs::core::quantize(x, q))
            .collect();
        assert_eq!(values_of(&back), expected);
        assert_eq!(back.precision(COLUMN), Some(p));
        assert_eq!(codecs, ["bytes", "numcodecs.shuffle", "zstd", "crc32c"]);
        for (x, stored) in values.iter().zip(&expected) {
            assert!((x - stored).abs() <= q / 2.0);
        }
    }

    #[test]
    fn precision_edge_values_land_as_the_spec_says() {
        let p = 1e-3;
        let q = molrs::core::quantum(p).unwrap();
        let huge = 2f64.powi(52) * q * 1.5;
        let values = [
            f64::NAN,
            f64::INFINITY,
            f64::NEG_INFINITY,
            -0.0,
            -0.1 * q,
            huge,
            2.5 * q,
            3.5 * q,
        ];
        let (back, _) = round_trip_precise(&values, p);
        let back = values_of(&back);
        assert!(back[0].is_nan());
        assert_eq!(back[1], f64::INFINITY);
        assert_eq!(back[2], f64::NEG_INFINITY);
        assert!(back[3] == 0.0 && back[3].is_sign_negative());
        assert!(back[4] == 0.0 && back[4].is_sign_negative());
        assert_eq!(back[5].to_bits(), huge.to_bits());
        assert_eq!(back[6], 2.0 * q);
        assert_eq!(back[7], 4.0 * q);
    }

    #[test]
    fn a_column_without_a_precision_stays_raw_and_declares_none() {
        let back = round_trip_column(Column::from(
            ArrayD::from_shape_vec(vec![2], vec![0.123_456_7, 2.0]).unwrap(),
        ));
        assert_eq!(
            back.get::<f64>().unwrap().as_slice().unwrap(),
            &[0.123_456_7, 2.0]
        );
    }

    #[test]
    fn a_precision_on_a_non_f64_column_is_refused_at_write() {
        let dir = TempDir::new().unwrap();
        let store = storage_in(&dir);
        let column = Column::from(ArrayD::from_shape_vec(vec![1], vec![3_i64]).unwrap());
        let err = write_column(&store, "/f/b/n", &column, Some(1e-3))
            .unwrap_err()
            .to_string();
        assert!(err.contains("f64"), "{err}");
        for bad in [0.0, -1.0, f64::INFINITY] {
            let float = Column::from(ArrayD::from_shape_vec(vec![1], vec![1.0]).unwrap());
            assert!(write_column(&store, "/f/b/x", &float, Some(bad)).is_err());
        }
    }

    #[test]
    fn a_malformed_precision_attribute_is_refused_on_read() {
        let dir = TempDir::new().unwrap();
        let store = storage_in(&dir);
        let mut frame = Frame::new();
        let mut block = Block::new();
        block
            .insert(
                COLUMN,
                Column::from(ArrayD::from_shape_vec(vec![1], vec![3_i64]).unwrap()),
                None,
            )
            .unwrap();
        frame.insert(BLOCK, block);
        write_frame_group(&store, FRAME, &frame).unwrap();
        // Tamper: an i64 array claiming a precision.
        let path = format!("{FRAME}/{BLOCK}/{COLUMN}");
        let mut array = Array::open(store.clone(), &path).unwrap();
        array
            .attributes_mut()
            .insert(PRECISION_ATTRIBUTE.to_string(), serde_json::json!(1e-3));
        array.store_metadata().unwrap();
        let err = read_frame_group(&store, FRAME).unwrap_err().to_string();
        assert!(err.contains(COLUMN) && err.contains("f64"), "{err}");
    }

    // -- typed frame meta (molrec F3) ---------------------------------------

    /// One value of every tag, with the edges the typed JSON forms exist for.
    fn every_tag() -> Vec<(&'static str, MetaValue)> {
        vec![
            ("b", MetaValue::Bool(true)),
            ("i32", MetaValue::I32(-7)),
            ("i64", MetaValue::I64(i64::MIN)),
            ("u32", MetaValue::U32(u32::MAX)),
            ("u64", MetaValue::U64(u64::MAX)),
            ("u64_53", MetaValue::U64((1 << 53) + 1)),
            ("f64", MetaValue::F64(1.0)),
            ("inf", MetaValue::F64(f64::INFINITY)),
            ("ninf", MetaValue::F64(f64::NEG_INFINITY)),
            ("s", MetaValue::String("NaN".into())),
            ("b3", MetaValue::Bool3([true, false, true])),
            ("i32x3", MetaValue::I32x3([1, -2, 3])),
            ("i64x3", MetaValue::I64x3([i64::MIN, 0, i64::MAX])),
            ("u32x3", MetaValue::U32x3([0, 1, 2])),
            ("u64x3", MetaValue::U64x3([0, 1, u64::MAX])),
            ("f64x3", MetaValue::F64x3([1.0, f64::INFINITY, 2.5])),
            ("f64x6", MetaValue::F64x6([1.0, 2.0, 3.0, 4.0, 5.0, 6.0])),
            ("f64x9", MetaValue::F64x9([0.5; 9])),
            (
                "doc",
                MetaValue::Json(serde_json::json!({"k": [1, 2], "n": null})),
            ),
        ]
    }

    fn meta_round_trip(frame: &Frame) -> Frame {
        let dir = TempDir::new().unwrap();
        let store = storage_in(&dir);
        write_frame_group(&store, FRAME, frame).unwrap();
        read_frame_group(&store, FRAME).unwrap()
    }

    #[test]
    fn every_meta_tag_round_trips_at_its_tag() {
        let mut frame = Frame::new();
        for (key, value) in every_tag() {
            frame.meta.insert(key, value);
        }
        frame.meta.insert("nan", f64::NAN);
        let back = meta_round_trip(&frame);
        for (key, value) in every_tag() {
            assert_eq!(back.meta.get(key), Some(&value), "{key}");
        }
        assert!(back.meta.get("nan").unwrap().as_f64().unwrap().is_nan());
        assert!(!back.meta.contains_key(META_TYPES_ATTR));
    }

    #[test]
    fn meta_is_stored_in_typed_json_with_a_tag_per_key() {
        let dir = TempDir::new().unwrap();
        let store = storage_in(&dir);
        let mut frame = Frame::new();
        frame.meta.insert("t", f64::NAN);
        frame.meta.insert("n", u64::MAX);
        frame.meta.insert("k", MetaValue::I32(3));
        write_frame_group(&store, FRAME, &frame).unwrap();
        let attrs = zarrs::group::Group::open(store.clone(), FRAME)
            .unwrap()
            .attributes()
            .clone();
        assert_eq!(attrs["t"], serde_json::json!("NaN"));
        assert_eq!(attrs["n"], serde_json::json!("18446744073709551615"));
        assert_eq!(
            attrs[META_TYPES_ATTR],
            serde_json::json!({"t": "f64", "n": "u64", "k": "i32"})
        );
        // An empty document writes no `_meta_types`.
        write_frame_group(&store, FRAME, &Frame::new()).unwrap();
        let attrs = zarrs::group::Group::open(store, FRAME)
            .unwrap()
            .attributes()
            .clone();
        assert!(attrs.is_empty());
    }

    /// Write `frame`, then let `tamper` edit the group attributes.
    fn tampered(
        frame: &Frame,
        tamper: impl FnOnce(&mut serde_json::Map<String, serde_json::Value>),
    ) -> Result<Frame, MolRsError> {
        let dir = TempDir::new().unwrap();
        let store = storage_in(&dir);
        write_frame_group(&store, FRAME, frame).unwrap();
        let mut group = zarrs::group::Group::open(store.clone(), FRAME).unwrap();
        tamper(group.attributes_mut());
        group.store_metadata().unwrap();
        read_frame_group(&store, FRAME)
    }

    fn one_key(key: &str, value: impl Into<MetaValue>) -> Frame {
        let mut frame = Frame::new();
        frame.meta.insert(key, value);
        frame
    }

    #[test]
    fn untyped_meta_is_inferred_and_a_stale_tag_is_ignored() {
        let back = tampered(&one_key("x", MetaValue::I32(4)), |attrs| {
            attrs.remove(META_TYPES_ATTR);
            attrs.insert("big".into(), serde_json::json!(u64::MAX));
            attrs.insert("f".into(), serde_json::json!(2.5));
            attrs.insert("s".into(), serde_json::json!("NaN"));
        })
        .unwrap();
        assert_eq!(back.meta.get("x"), Some(&MetaValue::I64(4)));
        assert_eq!(back.meta.get("big"), Some(&MetaValue::U64(u64::MAX)));
        assert_eq!(back.meta.get("f"), Some(&MetaValue::F64(2.5)));
        assert_eq!(back.meta.get("s"), Some(&MetaValue::String("NaN".into())));

        let back = tampered(&one_key("x", 1.0), |attrs| {
            attrs[META_TYPES_ATTR]["gone"] = serde_json::json!("i32");
        })
        .unwrap();
        assert_eq!(back.meta.get("x"), Some(&MetaValue::F64(1.0)));
        assert!(!back.meta.contains_key("gone"));
    }

    #[test]
    fn a_value_off_its_tag_is_refused() {
        for (value, tag) in [
            (serde_json::json!(1.5), "i32"),
            (serde_json::json!(1_i64 << 40), "i32"),
            (serde_json::Value::Null, "f64"),
            (serde_json::json!([1, 2]), "f64x3"),
        ] {
            let err = tampered(&one_key("x", 1.0), |attrs| {
                attrs.insert("x".into(), value.clone());
                attrs[META_TYPES_ATTR]["x"] = serde_json::json!(tag);
            })
            .unwrap_err()
            .to_string();
            assert!(err.contains("\"x\""), "{err}");
        }
        let err = tampered(&one_key("x", 1.0), |attrs| {
            attrs.insert(META_TYPES_ATTR.into(), serde_json::json!(["f64"]));
        })
        .unwrap_err();
        assert!(err.to_string().contains(META_TYPES_ATTR), "{err}");
    }

    #[test]
    fn a_meta_key_named_like_the_tag_map_is_refused_at_write() {
        let dir = TempDir::new().unwrap();
        let store = storage_in(&dir);
        let err = write_frame_group(&store, FRAME, &one_key(META_TYPES_ATTR, 1.0))
            .unwrap_err()
            .to_string();
        assert!(err.contains(META_TYPES_ATTR), "{err}");
    }

    #[test]
    fn a_block_group_without_its_count_is_refused() {
        let dir = TempDir::new().unwrap();
        let store = storage_in(&dir);
        let mut frame = Frame::new();
        let mut block = Block::new();
        block
            .insert(
                COLUMN,
                Column::from(ArrayD::from_shape_vec(vec![2], vec![1.0, 2.0]).unwrap()),
                None,
            )
            .unwrap();
        frame.insert(BLOCK, block);
        write_frame_group(&store, FRAME, &frame).unwrap();
        let mut group =
            zarrs::group::Group::open(store.clone(), &format!("{FRAME}/{BLOCK}")).unwrap();
        group.attributes_mut().remove("count");
        group.store_metadata().unwrap();
        let err = read_frame_group(&store, FRAME).unwrap_err().to_string();
        assert!(err.contains("count") && err.contains(BLOCK), "{err}");
        group
            .attributes_mut()
            .insert("count".into(), serde_json::json!(-1));
        group.store_metadata().unwrap();
        assert!(read_frame_group(&store, FRAME).is_err());
    }

    /// Every canonical key is held to its declared dtype on read, not only the
    /// u64 identifiers; a writer converts a column of another width of the
    /// family on the way out.
    #[test]
    fn every_canonical_key_is_held_to_its_dtype_on_read_and_converted_on_write() {
        let dir = TempDir::new().unwrap();
        let store = storage_in(&dir);
        let mut block = Block::new();
        block
            .insert(
                "ix",
                Column::from(ArrayD::from_shape_vec(vec![2], vec![-1_i64, 3]).unwrap()),
                None,
            )
            .unwrap();
        let mut frame = Frame::new();
        frame.insert(BLOCK, block);
        write_frame_group(&store, FRAME, &frame).unwrap();
        let back = read_frame_group(&store, FRAME).unwrap();
        let ix = back.get(BLOCK).unwrap().get("ix").unwrap();
        assert_eq!(ix.dtype(), DType::Int);
        assert_eq!(ix.get::<i32>().unwrap().as_slice().unwrap(), &[-1, 3]);

        // An i64 `ix` that does not fit i32 is refused at write.
        let mut block = Block::new();
        block
            .insert(
                "ix",
                Column::from(ArrayD::from_shape_vec(vec![1], vec![1_i64 << 40]).unwrap()),
                None,
            )
            .unwrap();
        let mut frame = Frame::new();
        frame.insert(BLOCK, block);
        assert!(write_frame_group(&store, FRAME, &frame).is_err());

        // A store holding `ix` as i64 (a foreign writer) is refused on read.
        write_column(
            &store,
            &format!("{FRAME}/{BLOCK}/ix"),
            &Column::from(ArrayD::from_shape_vec(vec![2], vec![-1_i64, 3]).unwrap()),
            None,
        )
        .unwrap();
        let mut group =
            zarrs::group::Group::open(store.clone(), &format!("{FRAME}/{BLOCK}")).unwrap();
        group
            .attributes_mut()
            .insert("count".into(), serde_json::json!(2));
        group.store_metadata().unwrap();
        let err = read_frame_group(&store, FRAME).unwrap_err().to_string();
        assert!(err.contains("\"ix\"") && err.contains("i32"), "{err}");
    }

    // -- row references and topology conventions (molrec F4) ---------------

    fn uints(values: &[u64]) -> Column {
        Column::from(ArrayD::from_shape_vec(vec![values.len()], values.to_vec()).unwrap())
    }

    fn strings(values: &[&str]) -> Column {
        Column::from(
            ArrayD::from_shape_vec(
                vec![values.len()],
                values.iter().map(|s| (*s).to_string()).collect(),
            )
            .unwrap(),
        )
    }

    fn atoms_of(n: usize) -> Block {
        let mut atoms = Block::new();
        atoms
            .insert("x", Column::from(ArrayD::from_elem(vec![n], 0.5)), None)
            .unwrap();
        atoms
    }

    #[test]
    fn declared_targets_round_trip_as_the_block_attribute() {
        let dir = TempDir::new().unwrap();
        let store = storage_in(&dir);
        let mut frame = Frame::new();
        frame.insert("atoms", atoms_of(3));
        let mut members = Block::new();
        members.insert("ibead", uints(&[0, 2]), None).unwrap();
        members.insert("atom", uints(&[10, 11]), None).unwrap();
        members.set_target("ibead", "atoms").unwrap();
        members.set_target("atom", "/frame/atoms").unwrap();
        frame.insert("members", members);
        write_frame_group(&store, FRAME, &frame).unwrap();

        let attrs = zarrs::group::Group::open(store.clone(), &format!("{FRAME}/members"))
            .unwrap()
            .attributes()
            .clone();
        assert_eq!(
            attrs[TARGETS_ATTRIBUTE],
            serde_json::json!({"ibead": "atoms", "atom": "/frame/atoms"})
        );
        let back = read_frame_group(&store, FRAME).unwrap();
        assert_eq!(back["members"].target("ibead"), Some("atoms"));
        assert_eq!(back["members"].target("atom"), Some("/frame/atoms"));
    }

    #[test]
    fn a_broken_same_frame_reference_is_refused_both_ways() {
        let dir = TempDir::new().unwrap();
        let store = storage_in(&dir);
        // Out of range: ibead 5 with 3 atoms.
        let mut frame = Frame::new();
        frame.insert("atoms", atoms_of(3));
        let mut members = Block::new();
        members.insert("ibead", uints(&[5]), None).unwrap();
        members.set_target("ibead", "atoms").unwrap();
        frame.insert("members", members);
        assert!(write_frame_group(&store, FRAME, &frame).is_err());

        // Missing target block.
        let mut frame = Frame::new();
        let mut refs = Block::new();
        refs.insert("site", uints(&[0]), None).unwrap();
        refs.set_target("site", "sites").unwrap();
        frame.insert("refs", refs);
        let err = write_frame_group(&store, FRAME, &frame)
            .unwrap_err()
            .to_string();
        assert!(err.contains("sites"), "{err}");

        // A store that breaks it (tampered) is refused on read.
        let mut frame = Frame::new();
        frame.insert("atoms", atoms_of(3));
        let mut refs = Block::new();
        refs.insert("site", uints(&[2]), None).unwrap();
        frame.insert("refs", refs);
        write_frame_group(&store, FRAME, &frame).unwrap();
        let mut group = zarrs::group::Group::open(store.clone(), &format!("{FRAME}/refs")).unwrap();
        group.attributes_mut().insert(
            TARGETS_ATTRIBUTE.into(),
            serde_json::json!({"site": "nope"}),
        );
        group.store_metadata().unwrap();
        assert!(read_frame_group(&store, FRAME).is_err());

        // A target on a non-u64 column is refused on read.
        group
            .attributes_mut()
            .insert(TARGETS_ATTRIBUTE.into(), serde_json::json!({"x": "atoms"}));
        group.store_metadata().unwrap();
        assert!(read_frame_group(&store, FRAME).is_err());
    }

    /// Every new canonical atom column and relation block at its canonical
    /// dtype round-trips, a `virtual_sites` row with a null `atoml` included.
    #[test]
    fn the_canonical_topology_round_trips() {
        let dir = TempDir::new().unwrap();
        let store = storage_in(&dir);
        let mut atoms = atoms_of(3);
        for key in ["fx", "fy", "fz", "occupancy", "b_factor"] {
            atoms
                .insert(key, Column::from(ArrayD::from_elem(vec![3], 0.25)), None)
                .unwrap();
        }
        atoms
            .insert(
                "formal_charge",
                Column::from(ArrayD::from_shape_vec(vec![3], vec![-1_i64, 0, 1]).unwrap()),
                None,
            )
            .unwrap();
        atoms.insert("atom_map", uints(&[1, 0, 2]), None).unwrap();
        for key in ["chain", "icode", "altloc"] {
            atoms.insert(key, strings(&["A", "", "B"]), None).unwrap();
        }
        let mut frame = Frame::new();
        frame.insert("atoms", atoms);
        let mut sites = Block::new();
        for (key, v) in [("atomi", 0), ("atomj", 1), ("atomk", 2)] {
            sites.insert(key, uints(&[v]), None).unwrap();
        }
        sites
            .insert(
                "atoml",
                ndarray::Array1::from_vec(vec![0_u64]).into_dyn(),
                Some(vec![false]),
            )
            .unwrap();
        frame.insert("virtual_sites", sites);
        for name in ["constraints", "drudes"] {
            let mut pair = Block::new();
            pair.insert("atomi", uints(&[0]), None).unwrap();
            pair.insert("atomj", uints(&[1]), None).unwrap();
            pair.insert("style", strings(&["harmonic"]), None).unwrap();
            frame.insert(name, pair);
        }
        assert!(
            molrs::core::schema::Validator::canonical()
                .check(&frame)
                .is_empty()
        );
        write_frame_group(&store, FRAME, &frame).unwrap();
        let back = read_frame_group(&store, FRAME).unwrap();
        let atoms = &back["atoms"];
        assert_eq!(atoms.dtype("formal_charge"), Some(DType::I64));
        assert_eq!(atoms.dtype("atom_map"), Some(DType::Uint));
        assert_eq!(atoms.dtype("chain"), Some(DType::String));
        assert_eq!(back["virtual_sites"].validity("atoml"), Some(&[false][..]));
        assert_eq!(back["drudes"].n_rows(), Some(1));
    }

    /// A `cmaps` block — five endpoints, `atomi` through `atomm`, a `type`
    /// and a `style` — round-trips at its canonical dtypes.
    #[test]
    fn a_cmaps_block_round_trips() {
        let dir = TempDir::new().unwrap();
        let store = storage_in(&dir);
        let mut frame = Frame::new();
        frame.insert("atoms", atoms_of(6));
        let mut cmaps = Block::new();
        for (key, rows) in [
            ("atomi", [0, 1]),
            ("atomj", [1, 2]),
            ("atomk", [2, 3]),
            ("atoml", [3, 4]),
            ("atomm", [4, 5]),
        ] {
            cmaps.insert(key, uints(&rows), None).unwrap();
        }
        cmaps.insert("type", strings(&["c1", "c2"]), None).unwrap();
        cmaps
            .insert("style", strings(&["charmm", "charmm"]), None)
            .unwrap();
        frame.insert("cmaps", cmaps);
        assert!(
            molrs::core::schema::Validator::canonical()
                .check(&frame)
                .is_empty()
        );
        write_frame_group(&store, FRAME, &frame).unwrap();
        let back = read_frame_group(&store, FRAME).unwrap();
        let cmaps = &back["cmaps"];
        assert_eq!(cmaps.dtype("atomm"), Some(DType::Uint));
        assert_eq!(
            cmaps
                .get("atomm")
                .and_then(Column::get::<u64>)
                .unwrap()
                .iter()
                .copied()
                .collect::<Vec<_>>(),
            [4, 5]
        );
        assert_eq!(
            cmaps.get("type").and_then(Column::get::<String>).unwrap()[[1]],
            "c2"
        );
    }
}
