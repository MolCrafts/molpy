mod frames;

use crate::io::mrec::zarr_storage::frame_io::{
    BOX_GROUP, assemble_cell, join_path, read_column_array,
};
use indexmap::IndexMap;
use molrs::core::Column;
use molrs::core::MolRsError;
use molrs::core::SimBox;
use molrs::op::F;
use std::collections::{BTreeMap, VecDeque};
use std::sync::{Arc, Mutex};
use zarrs::array::Array;
use zarrs::group::Group;
use zarrs::storage::{
    ListableStorageTraits, ReadableListableStorage, ReadableListableStorageTraits,
    ReadableStorageTraits, StorageHandle,
};

use super::block_hints::elision_markers;
use super::box_arrays::{cell_defined_of, cell_from_attributes};
use super::codec::rows_subset;
use super::frame_content::column_rows;
use super::layout::{
    BOUNDARY_ARRAY, CHUNK_CACHE_ENTRIES, OFFSET_ARRAY, ORIGIN_ARRAY, STEP_ARRAY, STEP_INDEX_ARRAY,
    STEP_PROGRESSION_ATTRIBUTE, SYSTEM_GROUP, TIME_ARRAY, TIME_PROGRESSION_ATTRIBUTE,
    TRAJECTORY_GROUP, VECTORS_ARRAY,
};
use super::schema::SequenceSchema;
use super::series::{committed_frames, read_track, trajectory_attributes};
use super::store_access::{
    array_exists, group_exists, inner_rows_of, pinned_schema, read_whole, schema_from_storage,
};

// ---------------------------------------------------------------------------
// Reader
// ---------------------------------------------------------------------------

/// The index of one block section, as [`MrecReader::open`] cached it.
enum BlockIndex {
    /// The general case: the section's own `step_index` and CSR `offset`.
    Sparse {
        /// Frame ordinals at which the section was updated, ascending.
        step_index: Vec<u64>,
        /// CSR row pointer, one longer than `step_index`.
        offset: Vec<u64>,
    },
    /// The hinted case (`dense_updates` + `uniform_rows`): update `j` sits at
    /// ordinal `j` and owns rows `j*rows..(j+1)*rows`, so nothing was read.
    Regular {
        /// Updates committed.
        updates: u64,
        /// Rows every update carries.
        rows: u64,
    },
}

impl BlockIndex {
    /// `(update, first row, rows)` of the section at frame `index`, or `None`
    /// when the section has not appeared yet.
    fn resolve(&self, index: u64) -> Result<Option<(u64, u64, u64)>, MolRsError> {
        match self {
            Self::Sparse { step_index, offset } => {
                let Some(update) = latest_update(step_index, index) else {
                    return Ok(None);
                };
                let (Some(&start), Some(&end)) = (offset.get(update), offset.get(update + 1))
                else {
                    return Err(MolRsError::zarr(format!(
                        "block section has {} offsets for update {update}",
                        offset.len()
                    )));
                };
                // A hostile or corrupt store can carry a non-monotonic
                // `offset` array; `end - start` would then wrap to a
                // near-`u64::MAX` row count and drive an unbounded allocation.
                let count = end.checked_sub(start).ok_or_else(|| {
                    MolRsError::zarr(format!(
                        "block section update {update} has non-monotonic offsets ({start} > {end})"
                    ))
                })?;
                Ok(Some((update as u64, start, count)))
            }
            Self::Regular { updates, rows } => {
                if *updates == 0 {
                    return Ok(None);
                }
                let update = index.min(updates - 1);
                Ok(Some((update, update * rows, *rows)))
            }
        }
    }
}

/// The index of the `box/` section.
struct BoxIndex {
    step_index: Vec<u64>,
    cell_defined: bool,
}

/// The `box/` section's cells: the fixed cell from the group attributes, or
/// the per-update arrays, with a per-update cache.
pub(super) struct BoxReader {
    fixed: Option<SimBox>,
    vectors: Option<Array<dyn ReadableListableStorageTraits>>,
    origin: Option<Array<dyn ReadableListableStorageTraits>>,
    boundary: Option<Array<dyn ReadableListableStorageTraits>>,
    cells: BTreeMap<u64, SimBox>,
}

impl BoxReader {
    pub(super) fn open(store: &ReadableListableStorage) -> Result<Self, MolRsError> {
        let prefix = join_path(TRAJECTORY_GROUP, BOX_GROUP);
        let optional =
            |name: &str| -> Result<Option<Array<dyn ReadableListableStorageTraits>>, MolRsError> {
                let path = join_path(&prefix, name);
                if array_exists(store, &path)? {
                    Ok(Some(Array::open(store.clone(), &path)?))
                } else {
                    Ok(None)
                }
            };
        let vectors = optional(VECTORS_ARRAY)?;
        let fixed = if vectors.is_none() {
            cell_from_attributes(Group::open(store.clone(), &prefix)?.attributes())?
        } else {
            None
        };
        Ok(Self {
            fixed,
            vectors,
            origin: optional(ORIGIN_ARRAY)?,
            boundary: optional(BOUNDARY_ARRAY)?,
            cells: BTreeMap::new(),
        })
    }

    /// Cell update `index`, decoded once.
    pub(super) fn cell_at(&mut self, index: u64, cell_defined: bool) -> Result<SimBox, MolRsError> {
        if let Some(fixed) = &self.fixed {
            return Ok(fixed.clone());
        }
        if let Some(cell) = self.cells.get(&index) {
            return Ok(cell.clone());
        }
        let vectors = self
            .vectors
            .as_ref()
            .ok_or_else(|| MolRsError::zarr("box section has neither a fixed cell nor arrays"))?;
        // An undefined cell's `vectors` row is never read: its numbers mean
        // nothing, and `Cell::Undefined` carries the identity instead.
        let h: Option<Vec<F>> = if cell_defined {
            Some(vectors.retrieve_array_subset(&rows_subset(index, 1, &[3, 3])?)?)
        } else {
            None
        };
        let origin: Option<Vec<F>> = match &self.origin {
            Some(array) => Some(array.retrieve_array_subset(&rows_subset(index, 1, &[3])?)?),
            None => None,
        };
        let boundary: Option<Vec<bool>> = match &self.boundary {
            Some(array) => Some(array.retrieve_array_subset(&rows_subset(index, 1, &[3])?)?),
            None => None,
        };
        let simbox = assemble_cell(
            cell_defined,
            h.as_deref(),
            origin.as_deref(),
            boundary.as_deref(),
            &format!("box update {index}"),
        )?;
        if self.cells.len() >= 64 {
            self.cells.clear();
        }
        self.cells.insert(index, simbox.clone());
        Ok(simbox)
    }
}

/// One open column array plus its most recently decoded inner chunks.
struct ColumnReader {
    array: Array<dyn ReadableListableStorageTraits>,
    /// Rows one inner chunk holds.
    chunk_rows: u64,
    /// Decoded whole chunks, most recently used first.
    chunks: VecDeque<(u64, Column)>,
}

impl ColumnReader {
    fn open(store: &ReadableListableStorage, path: &str) -> Result<Self, MolRsError> {
        let array = Array::open(store.clone(), path)?;
        let chunk_rows = inner_rows_of(&array);
        Ok(Self {
            array,
            chunk_rows,
            chunks: VecDeque::with_capacity(CHUNK_CACHE_ENTRIES + 1),
        })
    }

    /// Rows `start..start+rows`, decoding at most the chunks they touch.
    ///
    /// A range inside one chunk comes out of the chunk cache (decoding the
    /// chunk whole on a miss, so the next frame in it is a slice); a range
    /// straddling two chunks — a ragged run — is read directly.
    fn rows(&mut self, start: u64, rows: u64, trailing: &[u64]) -> Result<Column, MolRsError> {
        let first = start / self.chunk_rows;
        let last = (start + rows - 1) / self.chunk_rows;
        if first != last {
            return read_column_array(&self.array, &rows_subset(start, rows, trailing)?);
        }
        let within = (start - first * self.chunk_rows) as usize;
        let chunk = self.chunk(first, trailing)?;
        Ok(column_rows(chunk, within, within + rows as usize))
    }

    /// Inner chunk `index`, decoded whole and kept.
    fn chunk(&mut self, index: u64, trailing: &[u64]) -> Result<&Column, MolRsError> {
        if let Some(position) = self.chunks.iter().position(|(i, _)| *i == index) {
            if position != 0 {
                let hit = self.chunks.remove(position).expect("position is in range");
                self.chunks.push_front(hit);
            }
        } else {
            let total = self.array.shape().first().copied().unwrap_or(0);
            let chunk_start = index * self.chunk_rows;
            let chunk_len = self.chunk_rows.min(total.saturating_sub(chunk_start));
            let column =
                read_column_array(&self.array, &rows_subset(chunk_start, chunk_len, trailing)?)?;
            self.chunks.push_front((index, column));
            self.chunks.truncate(CHUNK_CACHE_ENTRIES);
        }
        Ok(&self
            .chunks
            .front()
            .expect("just inserted or moved to front")
            .1)
    }
}

/// The lazily opened arrays and caches behind a [`MrecReader`].
#[derive(Default)]
struct ReadState {
    columns: BTreeMap<String, ColumnReader>,
    /// Mask readers by array path, `None` for a nullable column the store
    /// carries no mask array for — caching the absence, so a column that was
    /// never holed costs one existence check for the whole run.
    masks: BTreeMap<String, Option<ColumnReader>>,
    metas: IndexMap<String, Array<dyn ReadableListableStorageTraits>>,
    boxes: Option<BoxReader>,
}

/// The lazy store cursor over a frame sequence: index-only open, one frame per
/// read.
///
/// One of the three access forms of a single object — `Trajectory` is the eager
/// in-memory carrier that materializes every frame, [`MrecWriter`] is
/// the streaming producer, and this reads what that writer produced without
/// materializing anything it was not asked for. [`to_trajectory`](Self::to_trajectory)
/// is the named lazy → eager conversion.
///
/// Every read door takes `&self`: the cursor holds no cursor state, only
/// caches (open array handles, the last decoded inner chunk per column, the
/// decoded cells) behind a lock, so a sequence can be shared and read from
/// several places.
///
/// **Error vocabulary.** Opening a record from a [`Source`](crate::io::Source)
/// ([`open`](Self::open)) is a frame-IO door and fails with
/// [`std::io::Error`] like every other. Every read of an opened store yields
/// [`MolRsError`], as the [`MolRsError::Zarr`] variant naming the section or
/// array path at fault. The three [`TrajectoryReader`] methods keep
/// `std::io::Result`: a `MolRsError` becomes an
/// [`InvalidData`](std::io::ErrorKind::InvalidData) error carrying it — the
/// trait is the shape every trajectory backend shares and must not be captured
/// by one backend's error type.
///
/// # One object, three access forms
///
/// A frame sequence is a single thing. Three Rust names reach it, and they are
/// named after the **access form**, never after the content:
///
/// - [`Trajectory`] — the **eager in-memory carrier**. Every frame is
///   materialized in a `Vec<Frame>`; it does not know a store exists.
/// - [`MrecReader`] — the **lazy store cursor**. It opens index-only (each
///   section's `step_index` and `offset`, plus the schema) and reads **one**
///   frame per call. [`MrecReader::to_trajectory`] is the named lazy → eager
///   conversion, not a second representation.
/// - [`MrecWriter`] — the sequence's **streaming producer**. One frame
///   per [`append`](super::MrecWriter::append); complete inner chunks land
///   on their own, [`flush`](super::MrecWriter::flush) commits whatever is
///   still buffered.
///
/// The `trajectory/` group on disk is the sequence's serialized form; none of
/// the three names is a second name for it.
///
/// # Error vocabulary
///
/// Every [`MrecReader`] and [`MrecWriter`] door on a store yields
/// [`MolRsError`]; [`MrecReader::open`] of a source is a frame-IO door and
/// yields [`std::io::Error`]. The three [`TrajectoryReader`] methods keep
/// `io::Result` and convert `MolRsError` in at the boundary (an
/// `InvalidData` error carrying it): the trait is the shape every backend
/// shares and must not be captured by one backend's error type.
///
/// # The layout
///
/// ```text
/// /                              root group
/// meta/                          the record's identity document (attributes)
/// trajectory/                    group; attrs: sequence_schema (the pin),
///                                nstep (the commit marker, written last),
///                                step_progression / time_progression
///                                ({start, stride} while the series is regular)
///   step     i64  [nstep]        only once the numbering is not a progression
///   time     f64  [nstep]        only when supplied and not a progression
///   meta/<key>    typed [nstep] (or [nstep][3|6|9]); attr meta_dtype
///   box/                         attrs: cell_defined; vectors / origin /
///                                boundary while the cell is fixed from ordinal 0
///     step_index u64  [n_updates]     the arrays, once the cell changes
///     vectors    f64  [n_updates][3][3]
///     origin     f64  [n_updates][3]
///     boundary   bool [n_updates][3]
///   <block>/                     attrs structural_shape, uniform_rows, dense_updates
///     step_index u64  [n_updates]     absent while the block is regular
///     offset     u64  [n_updates+1]   CSR row pointer, offset[0] = 0
///     <column>        [total_rows][...trailing]
///     _validity/                 group; only for columns declared nullable
///       <column> bool [total_rows]    one flag per row, no trailing axes
/// ```
///
/// The layout is built so the common run — a fixed number of atoms moving
/// every frame, a topology written once, a fixed cell, frames dumped every
/// `k` steps — costs one array per column and nothing else. A **regular**
/// block (a fixed, non-zero row count at ordinals `0, 1, 2, …`) writes no
/// index: its group hints plus any column's length resolve every frame. The
/// first irregular update materializes `offset` and `step_index`, backfilled
/// with the regular history, and withdraws the hints for good.
///
/// `offset` is a **CSR** (compressed sparse row) row pointer, the standard way
/// to store a ragged sequence of row groups in one flat array: entry `j` holds
/// the index of the first row of update `j`, so update `j` owns the half-open
/// row range `offset[j]..offset[j+1]`, and the array is therefore one entry
/// longer than `step_index`.
///
/// # Three states of a block at a frame
///
/// Resolving block `B` at frame `i`: binary-search `B/step_index` for the
/// largest entry `<= i`, giving update `j`; the frame's rows are
/// `offset[j]..offset[j+1]`. The three cases are the layout's whole
/// presence vocabulary:
///
/// - **No entry `<= i`** — the block does not exist at frame `i` (absence).
/// - **An update of zero rows** — the block is present and empty from that
///   frame until its next update. It reads back as an empty [`Block`] with the
///   declared columns.
/// - **A frame that omits a declared block** earns *no* update: the block
///   carries forward from its latest update. That is what makes a constant
///   topology cost one `step_index` entry whether the producer re-presents it
///   every frame (bit-identical content earns no update either) or presents
///   it once.
///
/// Once a block has appeared it never becomes absent again; there is no
/// tombstone. The `box/` section has no `offset` and follows the same
/// carry-forward rule: once a run writes a cell, every later frame resolves to
/// the most recent one.
///
/// [`Trajectory`]: molrs::core::Trajectory
/// [`MrecWriter`]: super::MrecWriter
/// [`Block`]: molrs::core::Block
/// [`TrajectoryReader`]: crate::io::reader::TrajectoryReader
pub struct MrecReader {
    /// The read-only view of the store. A read door keeps no write capability,
    /// which is also what lets a packed `.mrec.zip` be opened through it.
    store: ReadableListableStorage,
    schema: SequenceSchema,
    /// Step numbers of the committed frames; its length is `nstep`.
    steps: Vec<i64>,
    times: Option<Vec<F>>,
    blocks: IndexMap<String, BlockIndex>,
    cell: Option<BoxIndex>,
    state: Mutex<ReadState>,
}

impl MrecReader {
    /// Open a sequence for reading, taking only its indices.
    ///
    /// Index-only: the schema attributes plus each section's `step_index` and
    /// `offset` — or, for a section whose hints say every update is regular,
    /// nothing but its array metadata. No frame data is touched until a read
    /// asks for it. The commit marker bounds everything: index entries at or
    /// past `len(step)` (a crash's leftovers) are ignored.
    ///
    /// Any readable, listable store opens: a directory store, an in-memory one,
    /// or the read-only adapter a packed `.mrec.zip` is read through. A conforming
    /// store needs no writer pin — the schema is derived from the store itself
    /// when the attribute is absent. A store created and closed without a
    /// single append opens as a legal, empty sequence.
    ///
    /// # Errors
    ///
    /// Every case is a [`MolRsError::Zarr`] naming the path it failed on:
    ///
    /// - a schema pin is present but is not a schema this build can
    ///   deserialize;
    /// - no pin is present and the derivation from the store fails.
    ///
    /// Also any storage error raised while reading the index arrays.
    pub fn from_storage<S>(store: Arc<S>) -> Result<Self, MolRsError>
    where
        S: ?Sized + ReadableStorageTraits + ListableStorageTraits + 'static,
    {
        // `StorageHandle` is how an unsized store becomes an owned, sized one:
        // `Arc<dyn ReadableWritableListableStorageTraits>` does not coerce to
        // `Arc<dyn ReadableListableStorageTraits>`, so the read-only view is
        // taken here.
        let store: ReadableListableStorage = Arc::new(StorageHandle::new(store));
        let schema = match pinned_schema(&store)? {
            Some(pinned) => pinned,
            None => schema_from_storage(&store)?,
        };
        let attributes = trajectory_attributes(&store)?;
        let nstep = committed_frames(&store, &attributes)?;
        // Created and closed without an append: a legal, empty sequence.
        let steps = read_track::<i64, _>(
            &store,
            &attributes.all,
            STEP_PROGRESSION_ATTRIBUTE,
            &join_path(TRAJECTORY_GROUP, STEP_ARRAY),
            nstep,
        )?
        .unwrap_or_default();
        if steps.len() as u64 != nstep {
            return Err(MolRsError::zarr(format!(
                "trajectory claims {nstep} committed frames but carries {} step numbers",
                steps.len()
            )));
        }
        let times = read_track::<F, _>(
            &store,
            &attributes.all,
            TIME_PROGRESSION_ATTRIBUTE,
            &join_path(TRAJECTORY_GROUP, TIME_ARRAY),
            nstep,
        )?;

        let mut blocks = IndexMap::new();
        for (name, declared) in &schema.blocks {
            let path = join_path(TRAJECTORY_GROUP, name);
            if !group_exists(&store, &path)? {
                // Declared, never appended: the section does not exist yet.
                continue;
            }
            let index_path = join_path(&path, STEP_INDEX_ARRAY);
            let index = if array_exists(&store, &index_path)? {
                let mut step_index = read_whole::<u64, _>(&store, &index_path)?;
                let updates = step_index
                    .iter()
                    .take_while(|&&ordinal| ordinal < nstep)
                    .count();
                step_index.truncate(updates);
                let mut offset = read_whole::<u64, _>(&store, &join_path(&path, OFFSET_ARRAY))?;
                offset.truncate(if updates == 0 { 0 } else { updates + 1 });
                BlockIndex::Sparse { step_index, offset }
            } else {
                match elision_markers(&store, &path, !declared.columns.is_empty())? {
                    Some(rows) => {
                        // Regular: the columns' own length says how many
                        // updates landed; the marker bounds them.
                        let column =
                            declared.columns.keys().next().expect(
                                "elision_markers refuses markers on a block with no columns",
                            );
                        let landed = Array::open(store.clone(), &join_path(&path, column))?
                            .shape()
                            .first()
                            .copied()
                            .unwrap_or(0);
                        BlockIndex::Regular {
                            updates: (landed / rows).min(nstep),
                            rows,
                        }
                    }
                    _ => continue,
                }
            };
            let empty = match &index {
                BlockIndex::Regular { updates, .. } => *updates == 0,
                BlockIndex::Sparse { step_index, .. } => step_index.is_empty(),
            };
            if !empty {
                blocks.insert(name.clone(), index);
            }
        }

        schema.check_alignments()?;
        check_aligned_indexes(&schema, &blocks, nstep)?;
        check_aligned_not_in_system(&store, &schema)?;

        let box_prefix = join_path(TRAJECTORY_GROUP, BOX_GROUP);
        let vectors_path = join_path(&box_prefix, VECTORS_ARRAY);
        let cell = if array_exists(&store, &vectors_path)? {
            let step_path = join_path(&box_prefix, STEP_INDEX_ARRAY);
            let mut step_index = if array_exists(&store, &step_path)? {
                read_whole::<u64, _>(&store, &step_path)?
            } else {
                let n = Array::open(store.clone(), &vectors_path)?
                    .shape()
                    .first()
                    .copied()
                    .unwrap_or(0);
                // The omitted index is one update at ordinal 0.
                if n == 0 { Vec::new() } else { vec![0] }
            };
            let committed = step_index
                .iter()
                .take_while(|&&ordinal| ordinal < nstep)
                .count();
            step_index.truncate(committed);
            if step_index.is_empty() {
                None
            } else {
                Some(BoxIndex {
                    step_index,
                    cell_defined: cell_defined_of(&store)?,
                })
            }
        } else if group_exists(&store, &box_prefix)? {
            let group = Group::open(store.clone(), &box_prefix)?;
            match cell_from_attributes(group.attributes())? {
                Some(cell) if nstep > 0 => Some(BoxIndex {
                    step_index: vec![0],
                    cell_defined: cell.is_cell_defined(),
                }),
                _ => None,
            }
        } else {
            None
        };

        Ok(Self {
            store,
            schema,
            steps,
            times,
            blocks,
            cell,
            state: Mutex::new(ReadState::default()),
        })
    }
}

/// Refuse a store whose aligned blocks break the rule at some resolved frame,
/// checked from the indexes alone: at every ordinal where either block of a
/// pair has an update, the aligned block, when it resolves, has its target
/// resolving with the same row count.
fn check_aligned_indexes(
    schema: &SequenceSchema,
    blocks: &IndexMap<String, BlockIndex>,
    nstep: u64,
) -> Result<(), MolRsError> {
    let ordinals = |name: &str| -> Vec<u64> {
        match blocks.get(name) {
            None => Vec::new(),
            Some(BlockIndex::Sparse { step_index, .. }) => step_index.clone(),
            Some(BlockIndex::Regular { updates, .. }) => (0..*updates).collect(),
        }
    };
    for (aligned, target) in schema.aligned_pairs() {
        let mut at: Vec<u64> = ordinals(aligned);
        at.extend(ordinals(target));
        at.sort_unstable();
        at.dedup();
        for ordinal in at.into_iter().filter(|&o| o < nstep) {
            let Some((_, _, rows)) = blocks
                .get(aligned)
                .map(|index| index.resolve(ordinal))
                .transpose()?
                .flatten()
            else {
                continue;
            };
            let with = blocks
                .get(target)
                .map(|index| index.resolve(ordinal))
                .transpose()?
                .flatten();
            match with {
                Some((_, _, target_rows)) if target_rows == rows => {}
                Some((_, _, target_rows)) => {
                    return Err(MolRsError::zarr(format!(
                        "trajectory frame {ordinal}: aligned block {aligned:?} resolves to {rows} \
                         rows but its target {target:?} to {target_rows}"
                    )));
                }
                None => {
                    return Err(MolRsError::zarr(format!(
                        "trajectory frame {ordinal}: aligned block {aligned:?} is present but its \
                         target {target:?} is absent"
                    )));
                }
            }
        }
    }
    Ok(())
}

/// Refuse an aligned trajectory block named like a block of the record's
/// `system` section: the name would mean two different row sets.
pub(super) fn check_aligned_not_in_system<S>(
    store: &Arc<S>,
    schema: &SequenceSchema,
) -> Result<(), MolRsError>
where
    S: ?Sized + ReadableStorageTraits + 'static,
{
    for (aligned, _) in schema.aligned_pairs() {
        if group_exists(store, &join_path(SYSTEM_GROUP, aligned))? {
            return Err(MolRsError::zarr(format!(
                "aligned trajectory block {aligned:?} shares its name with a system block; an \
                 aligned block never does"
            )));
        }
    }
    Ok(())
}

/// Index of the latest entry at or before frame `index`, or `None` when the
/// section had not appeared yet.
///
/// `step_index` is ascending by construction, so this is a binary search: the
/// cost of resolving a section is logarithmic in its *changes*, not in the run
/// length.
fn latest_update(step_index: &[u64], index: u64) -> Option<usize> {
    match step_index.binary_search(&index) {
        Ok(found) => Some(found),
        Err(0) => None,
        Err(after) => Some(after - 1),
    }
}
