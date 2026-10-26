use crate::io::mrec::zarr_storage::frame_io::{
    BOX_GROUP, assemble_cell, cell_defined_attribute, join_path, json_flags, json_numbers,
};
use molrs::core::DType;
use molrs::core::MolRsError;
use molrs::core::SimBox;
use molrs::op::F;
use std::sync::Arc;
use zarrs::array::CodecOptions;
use zarrs::group::{Group, GroupBuilder};
use zarrs::storage::{ReadableStorageTraits, ReadableWritableListableStorage};

use super::block_arrays::{DENSE_PIPELINE, Knobs};
use super::frame_content::{boundary_is_default, origin_is_default};
use super::growth::{GrowthArray, land_values};
use super::layout::{
    BOUNDARY_ARRAY, CELL_DEFINED_ATTRIBUTE, ORIGIN_ARRAY, STEP_INDEX_ARRAY, TRAJECTORY_GROUP,
    VECTORS_ARRAY,
};

/// Where the cell of a sequence lives.
///
/// A cell that arrives at ordinal 0 and never changes is recorded as the
/// `box/` group's own attributes — no array, no extra file. The first change
/// migrates it into the `box/` arrays.
pub(super) enum CellStorage {
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
pub(super) struct BoxArrays {
    pub(super) step_index: Option<GrowthArray>,
    pub(super) vectors: GrowthArray,
    pub(super) origin: Option<GrowthArray>,
    pub(super) boundary: Option<GrowthArray>,
}

impl BoxArrays {
    pub(super) fn create(
        store: &ReadableWritableListableStorage,
        knobs: Knobs,
    ) -> Result<Self, MolRsError> {
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
    pub(super) fn land(
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
    pub(super) fn truncate_to(&mut self, updates: u64) -> Result<(), MolRsError> {
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
pub(super) fn box_attributes(
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
pub(super) fn cell_from_attributes(
    attributes: &serde_json::Map<String, serde_json::Value>,
) -> Result<Option<SimBox>, MolRsError> {
    const WHAT: &str = "trajectory box attributes";
    let Some(vectors) = attributes.get(VECTORS_ARRAY) else {
        return Ok(None);
    };
    let cell_defined = cell_defined_attribute(attributes, WHAT)?;
    // An undefined cell's `vectors` mean nothing and are never parsed.
    let h = if cell_defined {
        Some(json_numbers(vectors, WHAT)?)
    } else {
        None
    };
    let origin = attributes
        .get(ORIGIN_ARRAY)
        .map(|v| json_numbers(v, WHAT))
        .transpose()?;
    let boundary = attributes
        .get(BOUNDARY_ARRAY)
        .map(|v| json_flags(v, WHAT))
        .transpose()?;
    assemble_cell(
        cell_defined,
        h.as_deref(),
        origin.as_deref(),
        boundary.as_deref(),
        WHAT,
    )
    .map(Some)
}

/// Write the `box/` group with these attributes (creating it if needed).
pub(super) fn store_box_attributes(
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

/// The `box/` section's `cell_defined` attribute; absent means a defined cell.
/// A trajectory with no `box/` group has no cell to describe, so it reads as
/// the default; any other failure to open the group is an error.
pub(super) fn cell_defined_of<S>(store: &Arc<S>) -> Result<bool, MolRsError>
where
    S: ?Sized + ReadableStorageTraits + 'static,
{
    let prefix = join_path(TRAJECTORY_GROUP, BOX_GROUP);
    match Group::open(store.clone(), &prefix) {
        Ok(group) => cell_defined_attribute(group.attributes(), &prefix),
        Err(zarrs::group::GroupCreateError::MissingMetadata) => Ok(true),
        Err(e) => Err(e.into()),
    }
}
