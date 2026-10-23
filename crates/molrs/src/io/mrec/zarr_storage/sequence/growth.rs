use crate::io::mrec::zarr_storage::frame_io::{insert_column_into_block, zarr_dtype};
use crate::io::mrec::zarr_storage::record_io::zerr;
use molrs::core::MetaValue;
use molrs::core::MolRsError;
use molrs::core::{Block, Column, DType};
use std::num::NonZeroU64;
use zarrs::array::codec::array_to_bytes::sharding::{ShardingCodecBuilder, ShardingIndexLocation};
use zarrs::array::{Array, ArrayBuilder, ArraySubset, CodecOptions};
use zarrs::storage::{
    ReadableStorageTraits, ReadableWritableListableStorage, ReadableWritableListableStorageTraits,
};

use super::codec::{
    Extents, Pipeline, deterministic_shard_layout, dtype_from_tag, inner_codecs, meta_layout,
    quiet_zarrs_metadata, rows_subset,
};
use super::frame_content::empty_column;
use super::schema::BlockSchema;
use super::store_access::inner_rows_of;

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
pub(super) struct GrowthArray {
    pub(super) array: Array<dyn ReadableWritableListableStorageTraits>,
    /// Rows landed so far (the leading extent the metadata will publish).
    pub(super) rows: u64,
}

impl GrowthArray {
    /// Create the array at `path`, zero rows long, extents frozen.
    ///
    /// Every growth array is a `sharding_indexed` array with its index at the
    /// **start** of the shard, so appending a chunk is a tail write plus an
    /// in-place index rewrite. The inner chunk carries the codecs `pipeline`
    /// names ([`inner_codecs`]), ending in `crc32c`.
    pub(super) fn create(
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
    pub(super) fn open(
        store: &ReadableWritableListableStorage,
        path: &str,
    ) -> Result<Self, MolRsError> {
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
    pub(super) fn rows_per_chunk(&self) -> u64 {
        inner_rows_of(&self.array)
    }

    /// Grow the leading axis by `count` rows **in memory** and return the
    /// subset they land in. The caller writes the data, then
    /// [`commit_shape`](Self::commit_shape) publishes the new extent.
    pub(super) fn reserve(&mut self, count: u64) -> Result<ArraySubset, MolRsError> {
        let start = self.rows;
        self.rows += count;
        let mut shape = self.array.shape().to_vec();
        shape[0] = self.rows;
        self.array.set_shape(shape)?;
        rows_subset(start, count, &self.trailing())
    }

    /// Publish the reserved extent: one atomic `zarr.json` replacement.
    pub(super) fn commit_shape(&self) -> Result<(), MolRsError> {
        self.array.store_metadata()?;
        Ok(())
    }

    /// Roll the leading axis back to `rows` — what a reopen does to whatever a
    /// crash left longer than the commit marker.
    pub(super) fn truncate_to(&mut self, rows: u64) -> Result<(), MolRsError> {
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
    pub(super) fn store_columns(
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
    pub(super) fn store_meta(
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
pub(super) use land_values;

/// One value of a per-step meta array, at step `index`.
pub(super) fn read_meta_value<S>(
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
pub(super) fn empty_block(schema: &BlockSchema) -> Result<Block, MolRsError> {
    let mut block = Block::new();
    for (column, declared) in &schema.columns {
        let dtype = dtype_from_tag(&declared.dtype)?;
        insert_column_into_block(&mut block, column, empty_column(dtype, &declared.trailing)?)?;
    }
    Ok(block)
}

/// Apply a declared structural shape to a block whose row count matches it.
pub(super) fn apply_structural_shape(
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
