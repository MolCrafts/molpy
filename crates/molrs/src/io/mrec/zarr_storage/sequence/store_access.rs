use crate::io::mrec::zarr_storage::frame_io::{VALIDITY_GROUP, join_path};
use crate::io::mrec::zarr_storage::record_io::zerr;
use molrs::core::MetaValue;
use molrs::core::MolRsError;
use std::sync::Arc;
use zarrs::array::{Array, ArraySubset};
use zarrs::group::{Group, GroupBuilder};
use zarrs::node::{Node, NodeMetadata, NodePath, get_child_nodes};
use zarrs::storage::{
    ListableStorageTraits, ReadableStorageTraits, ReadableWritableListableStorage,
};

use super::codec::dtype_of_stored;
use super::layout::{
    META_DTYPE_ATTRIBUTE, META_GROUP, META_ROOT_GROUP, RESERVED_BLOCK_NAMES, RESERVED_COLUMN_NAMES,
    ROOT_GROUP, SCHEMA_ATTRIBUTE, STRUCTURAL_SHAPE_ATTRIBUTE, TRAJECTORY_GROUP,
};
use super::schema::SequenceSchema;

/// Read the schema back out of the `trajectory/` group attributes, if one is
/// pinned there.
///
/// A missing pin is `Ok(None)`, not an error, because it is the one fact the
/// two doors answer differently: [`schema_of`] refuses it, and
/// [`MrecReader::open`] derives from the store instead.
pub(super) fn pinned_schema<S>(store: &Arc<S>) -> Result<Option<SequenceSchema>, MolRsError>
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
pub(super) fn schema_of<S>(store: &Arc<S>) -> Result<SequenceSchema, MolRsError>
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
pub(super) fn schema_from_storage<S>(store: &Arc<S>) -> Result<SequenceSchema, MolRsError>
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
pub(super) fn array_exists<S>(store: &Arc<S>, path: &str) -> Result<bool, MolRsError>
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
pub(super) fn group_exists<S>(store: &Arc<S>, path: &str) -> Result<bool, MolRsError>
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
pub(super) fn read_whole<T, S>(store: &Arc<S>, path: &str) -> Result<Vec<T>, MolRsError>
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
pub(super) fn inner_rows_of<S: ?Sized>(array: &Array<S>) -> u64 {
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
pub(super) fn ensure_root_and_meta(
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
