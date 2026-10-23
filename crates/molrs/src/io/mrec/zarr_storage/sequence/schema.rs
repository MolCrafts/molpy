mod declarations;

use crate::io::mrec::zarr_storage::frame_io::{join_path, stored_dtype};
use crate::io::mrec::zarr_storage::record_io::zerr;
use indexmap::IndexMap;
use molrs::core::Frame;
use molrs::core::MolRsError;
use molrs::core::{MetaMap, MetaValue};
use serde::{Deserialize, Serialize};
use std::collections::BTreeMap;
use zarrs::node::NodePath;

use super::frame_content::{check_resolved_frame, stored_presentation};
use super::layout::{
    RESERVED_BLOCK_NAMES, RESERVED_COLUMN_NAMES, STEP_ARRAY, TIME_ARRAY, TRAJECTORY_GROUP,
};

// ---------------------------------------------------------------------------
// Schema
// ---------------------------------------------------------------------------

/// One declared column: the width and trailing shape it arrived with,
/// whether its rows may be null, and its declared precision.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub(super) struct ColumnSchema {
    /// [`DType::name`] of the column's storage width.
    pub(super) dtype: String,
    /// Axes after the leading row axis, e.g. `[3]` for xyz.
    #[serde(default)]
    pub(super) trailing: Vec<u64>,
    /// Whether the column carries a [validity
    /// mask](crate::core::Block::validity), stored as a `bool` array
    /// under the section's `_validity` subgroup. Absent from the serialized
    /// pin while false; a pin without it deserializes as a plain column.
    #[serde(default, skip_serializing_if = "std::ops::Not::not")]
    pub(super) nullable: bool,
    /// The column's [declared precision](molrs::core::check_precision) — on a
    /// trajectory, stated here and nowhere else. Every presented value is
    /// rounded to its grid before the change detection and the landing.
    /// Absent while undeclared.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub(super) precision: Option<f64>,
}

/// One declared block: its columns, and the structural shape it declares.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub(super) struct BlockSchema {
    pub(super) columns: IndexMap<String, ColumnSchema>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub(super) structural_shape: Option<Vec<usize>>,
    /// The block's row references (molrec `targets`): column → `<block>`
    /// of the same resolved frame or `/<section>/<block>`. On a trajectory
    /// they are pinned here and nowhere else. Absent while empty.
    #[serde(default, skip_serializing_if = "IndexMap::is_empty")]
    pub(super) targets: IndexMap<String, String>,
    /// The block this one is [aligned with](SequenceSchema::declare_aligned):
    /// its rows are that block's rows, one for one, at every resolved frame.
    /// Declared only, never derived. Absent while unset.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    aligned_with: Option<String>,
}

/// One declared per-step metadata key.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub(super) struct MetaSchema {
    /// The [`MetaValue::dtype`] tag every step must carry.
    pub(super) dtype: String,
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
    pub(super) fill: Option<serde_json::Value>,
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
    pub(super) blocks: IndexMap<String, BlockSchema>,
    #[serde(default)]
    pub(super) meta: IndexMap<String, MetaSchema>,
    /// Representative rows per block, for chunk sizing. Never pinned.
    #[serde(skip)]
    pub(super) rows_hint: BTreeMap<String, u64>,
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
    pub(in crate::io::mrec::zarr_storage) fn validate_frames(
        &self,
        frames: &[Frame],
    ) -> Result<(), MolRsError> {
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
    pub(super) fn validate_frame(&self, frame: &Frame) -> Result<(), MolRsError> {
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
                let dtype = stored_dtype(column, values.dtype()).name();
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
    pub(super) fn resolve_meta(&self, frame: &Frame) -> Result<MetaMap, MolRsError> {
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
}
