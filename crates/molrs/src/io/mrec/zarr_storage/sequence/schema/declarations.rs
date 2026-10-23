use crate::io::mrec::zarr_storage::frame_io::{check_canonical_dtype, join_path};
use crate::io::mrec::zarr_storage::record_io::zerr;
use indexmap::IndexMap;
use molrs::core::DType;
use molrs::core::MetaValue;
use molrs::core::MolRsError;
use zarrs::node::NodePath;

use super::super::codec::{dtype_from_tag, meta_layout, row_bytes};
use super::super::layout::{META_GROUP, TRAJECTORY_GROUP};
use super::{
    BlockSchema, ColumnSchema, MetaSchema, SequenceSchema, check_block_name, check_column_name,
};

impl SequenceSchema {
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
            dtype: dtype.name().to_string(),
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
        if declared.dtype != DType::Float.name() {
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
        if pinned.dtype != DType::Uint.name() {
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
    pub(in crate::io::mrec::zarr_storage::sequence) fn aligned_pairs(
        &self,
    ) -> impl Iterator<Item = (&str, &str)> {
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
    pub(in crate::io::mrec::zarr_storage::sequence) fn check_alignments(
        &self,
    ) -> Result<(), MolRsError> {
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
    pub(in crate::io::mrec::zarr_storage::sequence) fn frame_rows(&self, block: &str) -> u64 {
        self.rows_hint.get(block).copied().unwrap_or(0)
    }

    /// Bytes one frame of the representative row counts occupies across every
    /// declared column — the payload estimate the automatic landing cadence
    /// is derived from.
    pub(in crate::io::mrec::zarr_storage::sequence) fn frame_bytes(&self) -> u64 {
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
