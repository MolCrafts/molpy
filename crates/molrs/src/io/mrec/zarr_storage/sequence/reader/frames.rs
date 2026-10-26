use crate::io::mrec::zarr_storage::frame_io::{
    VALIDITY_GROUP, check_local_references, insert_column_into_block, join_path,
};
use crate::io::reader::TrajectoryReader;
use molrs::core::Block;
use molrs::core::Frame;
use molrs::core::MolRsError;
use molrs::core::SimBox;
use molrs::core::Trajectory;
use molrs::op::F;
use std::collections::BTreeMap;
use zarrs::array::Array;

use super::super::codec::dtype_from_tag;
use super::super::frame_content::empty_column;
use super::super::growth::{apply_structural_shape, read_meta_value};
use super::super::layout::{META_GROUP, TRAJECTORY_GROUP};
use super::super::schema::SequenceSchema;
use super::super::store_access::array_exists;
use super::{BoxReader, ColumnReader, MrecReader, ReadState, latest_update};

impl MrecReader {
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
                block.resize(rows as usize);
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
