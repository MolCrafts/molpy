use crate::io::mrec::zarr_storage::frame_io::join_path;
use molrs::core::Frame;
use molrs::core::MetaValue;
use molrs::core::MolRsError;
use molrs::core::SimBox;
use molrs::core::{Block, Column};
use molrs::op::F;
use std::collections::BTreeMap;

use super::super::block_arrays::IndexArrays;
use super::super::block_hints::{Uniform, store_block_attributes};
use super::super::box_arrays::{BoxArrays, CellStorage, box_attributes, store_box_attributes};
use super::super::codec::{dtype_from_tag, partial_encoding_options};
use super::super::frame_content::{
    check_resolved_frame, same_block, same_simbox, stored_presentation,
};
use super::super::growth::land_values;
use super::super::layout::{STEP_ARRAY, TIME_ARRAY, TRAJECTORY_GROUP};
use super::super::sequence_arrays::SequenceArrays;
use super::{MrecWriter, PendingFrame};

impl MrecWriter {
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
