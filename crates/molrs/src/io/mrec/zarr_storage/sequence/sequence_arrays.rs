use crate::io::mrec::zarr_storage::frame_io::{BOX_GROUP, GZIP_LEVEL, join_path};
use indexmap::IndexMap;
use molrs::core::DType;
use molrs::core::MolRsError;
use std::collections::BTreeMap;
use zarrs::group::{Group, GroupBuilder};
use zarrs::storage::ReadableWritableListableStorage;

use super::block_arrays::{
    BlockArrays, DENSE_PIPELINE, IndexArrays, Knobs, create_mask_arrays, open_mask_arrays,
};
use super::block_hints::{
    BlockHints, block_attributes, elision_markers, store_block_attributes, stored_hints,
};
use super::box_arrays::{BoxArrays, CellStorage, cell_from_attributes};
use super::codec::{
    Compression, Extents, Pipeline, dtype_from_tag, dtype_of_stored, is_float_width, meta_layout,
    row_bytes,
};
use super::growth::GrowthArray;
use super::layout::{
    BOUNDARY_ARRAY, META_DTYPE_ATTRIBUTE, META_GROUP, NSTEP_ATTRIBUTE, OFFSET_ARRAY, ORIGIN_ARRAY,
    STEP_ARRAY, STEP_INDEX_ARRAY, STEP_PROGRESSION_ATTRIBUTE, TIME_ARRAY,
    TIME_PROGRESSION_ATTRIBUTE, TRAJECTORY_GROUP, VECTORS_ARRAY,
};
use super::schema::{ColumnSchema, SequenceSchema};
use super::series::{Progression, Track, committed_frames, trajectory_attributes};
use super::store_access::{array_exists, group_exists, read_whole};

/// Every array of one sequence, created at the first append and never
/// re-planned afterwards.
pub(super) struct SequenceArrays {
    pub(super) step: Track<i64>,
    pub(super) time: Option<Track<f64>>,
    pub(super) meta: IndexMap<String, GrowthArray>,
    pub(super) blocks: BTreeMap<String, BlockArrays>,
    pub(super) cell: Option<CellStorage>,
    /// Frames committed — the `nstep` attribute.
    pub(super) nstep: u64,
}

impl SequenceArrays {
    /// Create every zero-length array the schema declares.
    pub(super) fn create(
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
    pub(super) fn open(
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
    pub(super) fn store_marker(
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
    let found = dtype_of_stored(path, opened.array.data_type())?
        .name()
        .to_string();
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
