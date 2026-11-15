use super::*;
use std::sync::atomic::{AtomicU64, Ordering};

// =======================================================================
// E. Ordering — step is extended last, and that is observable
// =======================================================================

/// Rolling the `nstep` marker back to its pre-flush value hides the
/// uncommitted frames and errors at nothing (ac-017).
///
/// This is the crash window made deterministic: the marker is written
/// last, so a crash anywhere before that leaves the data arrays long and
/// the marker short — exactly the state this test builds.
#[test]
fn a_marker_rolled_back_hides_the_uncommitted_frames() {
    let dir = TempDir::new().unwrap();
    let store = storage_in(&dir);
    let frames: Vec<Frame> = (0..5)
        .map(|step| atoms_frame(&[step as f64, step as f64 + 0.5]))
        .collect();
    write_all(&store, &frames);

    let mut group = Group::open(store.clone(), TRAJ).unwrap();
    group.attributes_mut().insert("nstep".to_string(), 3.into());
    group.store_metadata().unwrap();

    let mut seq = open_sequence(&store);
    assert_eq!(
        committed_len(&mut seq),
        3,
        "len follows the commit marker, not the data arrays"
    );
    for (index, frame) in frames[..3].iter().enumerate() {
        assert_eq!(
            atoms_x(&frame_at(&mut seq, index as u64)),
            atoms_x(frame),
            "committed frame {index} still reads"
        );
    }
    assert!(
        seq.frame(3).unwrap().is_none(),
        "the uncommitted frames are invisible, and asking for one is not an error"
    );
}

/// The writer's own store ([`PositionalWriteStorage`]) with the order of the
/// keys written through it recorded. Reads and listings are the wrapped
/// store's; only the write order is this type's business.
///
/// It records `set` and `set_partial_many` — the two doors a value reaches
/// the disk through — and deliberately not `erase`, which is not a write
/// of content.
///
/// It can also stop: after [`fail_after`](Self::fail_after) the store lets
/// that many more content writes through and refuses every one after, which
/// is a writer killed between two writes made deterministic.
#[derive(Debug)]
struct RecordingStorage {
    inner: PositionalWriteStorage,
    written: Mutex<Vec<String>>,
    /// Content writes still allowed; [`UNLIMITED`] never runs out.
    budget: AtomicU64,
}

/// The budget of a [`RecordingStorage`] that never refuses a write.
const UNLIMITED: u64 = u64::MAX;

impl RecordingStorage {
    fn new(path: &Path) -> Self {
        Self {
            inner: PositionalWriteStorage::new(path).unwrap(),
            written: Mutex::new(Vec::new()),
            budget: AtomicU64::new(UNLIMITED),
        }
    }

    /// Let `writes` more content writes through, then refuse the rest.
    fn fail_after(&self, writes: u64) {
        self.budget.store(writes, Ordering::SeqCst);
    }

    fn record(&self, key: &StoreKey) -> Result<(), StorageError> {
        let left = self.budget.load(Ordering::SeqCst);
        if left == 0 {
            return Err(StorageError::IOError(Arc::new(std::io::Error::other(
                format!("injected failure writing {key}"),
            ))));
        }
        if left != UNLIMITED {
            self.budget.store(left - 1, Ordering::SeqCst);
        }
        self.written.lock().unwrap().push(key.as_str().to_string());
        Ok(())
    }

    fn written_keys(&self) -> Vec<String> {
        self.written.lock().unwrap().clone()
    }

    fn clear(&self) {
        self.written.lock().unwrap().clear();
    }
}

impl ReadableStorageTraits for RecordingStorage {
    fn get(&self, key: &StoreKey) -> Result<MaybeBytes, StorageError> {
        self.inner.get(key)
    }

    fn get_partial(
        &self,
        key: &StoreKey,
        byte_range: ByteRange,
    ) -> Result<MaybeBytes, StorageError> {
        self.inner.get_partial(key, byte_range)
    }

    fn get_partial_many<'a>(
        &'a self,
        key: &StoreKey,
        byte_ranges: ByteRangeIterator<'a>,
    ) -> Result<MaybeBytesIterator<'a>, StorageError> {
        self.inner.get_partial_many(key, byte_ranges)
    }

    fn size_key(&self, key: &StoreKey) -> Result<Option<u64>, StorageError> {
        self.inner.size_key(key)
    }

    fn supports_get_partial(&self) -> bool {
        self.inner.supports_get_partial()
    }
}

impl ListableStorageTraits for RecordingStorage {
    fn list(&self) -> Result<StoreKeys, StorageError> {
        self.inner.list()
    }

    fn list_prefix(&self, prefix: &StorePrefix) -> Result<StoreKeys, StorageError> {
        self.inner.list_prefix(prefix)
    }

    fn list_dir(&self, prefix: &StorePrefix) -> Result<StoreKeysPrefixes, StorageError> {
        self.inner.list_dir(prefix)
    }

    fn size(&self) -> Result<u64, StorageError> {
        self.inner.size()
    }

    fn size_prefix(&self, prefix: &StorePrefix) -> Result<u64, StorageError> {
        self.inner.size_prefix(prefix)
    }
}

impl WritableStorageTraits for RecordingStorage {
    fn set(&self, key: &StoreKey, value: Bytes) -> Result<(), StorageError> {
        self.record(key)?;
        self.inner.set(key, value)
    }

    fn set_partial_many(
        &self,
        key: &StoreKey,
        offset_values: OffsetBytesIterator,
    ) -> Result<(), StorageError> {
        self.record(key)?;
        self.inner.set_partial_many(key, offset_values)
    }

    fn erase(&self, key: &StoreKey) -> Result<(), StorageError> {
        self.inner.erase(key)
    }

    fn erase_prefix(&self, prefix: &StorePrefix) -> Result<(), StorageError> {
        self.inner.erase_prefix(prefix)
    }

    fn supports_set_partial(&self) -> bool {
        self.inner.supports_set_partial()
    }
}

/// The write order is a property, not a comment: within one `flush()`
/// the trajectory group's metadata — the `nstep` marker — is written
/// **after** every other key (ac-017).
///
/// That ordering is what makes a crash between two writes invisible — the
/// commit marker is the last thing to move — and it is unobservable from
/// the outside, so the observation is a recording store.
#[test]
fn the_marker_is_written_after_every_other_key() {
    let dir = TempDir::new().unwrap();
    let recorder = Arc::new(RecordingStorage::new(dir.path()));
    let store: ReadableWritableListableStorage = recorder.clone();

    let frame = atoms_frame(&[1.0, 2.0]);
    let schema = SequenceSchema::from_frames(std::slice::from_ref(&frame)).unwrap();
    let mut writer = MrecWriter::create_in_storage(store, schema).unwrap();
    writer.append(&frame).unwrap();
    recorder.clear();
    writer.flush().unwrap();

    let log = recorder.written_keys();
    let marker = log
        .iter()
        .rposition(|key| key == "trajectory/zarr.json")
        .unwrap_or_else(|| panic!("a flush must write the marker; wrote {log:?}"));
    assert_eq!(
        marker,
        log.len() - 1,
        "nothing may be written after the commit marker: {log:?}"
    );
    assert!(
        log[..marker]
            .iter()
            .any(|key| key.starts_with(&format!("trajectory/{ATOMS}/"))),
        "the data arrays must be written before the marker: {log:?}"
    );
}

// -----------------------------------------------------------------------
// A flush stopped after any one write reads as the last commit
// -----------------------------------------------------------------------

/// A frame with the stock `atoms` block and a `bonds` block of `bonds`.
fn bonded_frame(x: f64, bonds: &[u64]) -> Frame {
    let mut frame = atoms_frame(&[x, x + 0.5]);
    frame.insert(BONDS, block_with(I, uint_column(bonds)));
    frame
}

/// A cubic cell of edge `edge`, default origin and boundary.
fn cubic_cell(edge: f64) -> SimBox {
    SimBox::new(
        Cell::Matrix([[edge, 0.0, 0.0], [0.0, edge, 0.0], [0.0, 0.0, edge]]),
        [0.0, 0.0, 0.0],
        [true, true, true],
    )
    .unwrap()
}

/// Everything a reader resolves at ordinal `index`, in one comparable value.
fn resolved(seq: &mut MrecReader, index: u64) -> String {
    let frame = frame_at(seq, index);
    let bonds: Option<Vec<u64>> = frame.get(BONDS).map(|block| {
        block
            .get(I)
            .expect("bonds carries column i")
            .get::<u64>()
            .expect("column i arrived as u64")
            .iter()
            .copied()
            .collect()
    });
    let cell = seq
        .box_at(index)
        .expect("resolving the cell must not error")
        .map(|cell| (cell.matrix(), cell.origin().to_vec(), cell.pbc().to_vec()));
    format!(
        "x={:?} bonds={bonds:?} cell={cell:?} step={:?}",
        atoms_x(&frame),
        frame.meta.get("step")
    )
}

/// Append `committed` and flush, then append `flushed` and flush again with
/// the store allowed `budget` more writes. Returns the outcome of the second
/// flush and how many writes it made.
fn flush_twice(
    dir: &Path,
    schema: &SequenceSchema,
    committed: &[(Frame, i64)],
    flushed: &[(Frame, i64)],
    budget: u64,
) -> (Result<(), molrs::core::MolRsError>, usize) {
    let recorder = Arc::new(RecordingStorage::new(dir));
    let store: ReadableWritableListableStorage = recorder.clone();
    let mut writer = MrecWriter::create_in_storage(store, schema.clone()).unwrap();
    for (frame, step) in committed {
        writer.append_at(frame, *step, None).unwrap();
    }
    writer.flush().unwrap();
    for (frame, step) in flushed {
        writer.append_at(frame, *step, None).unwrap();
    }
    recorder.clear();
    recorder.fail_after(budget);
    let outcome = writer.flush();
    let writes = recorder.written_keys().len();
    if outcome.is_err() {
        // A killed process runs no destructor: whatever the store would
        // tidy up on the way out stays undone, as it would on a crash.
        std::mem::forget(recorder);
    }
    (outcome, writes)
}

/// The property of molrec's `docs/spec/chunking.md` (materialization order):
/// a flush stopped after any one of its writes leaves a store that reads as
/// the commit before it — every committed frame, with the content it had.
///
/// `committed` is flushed cleanly; the flush of `flushed` is then cut after
/// 0, 1, 2, … writes, and each wreck is read through a plain store.
fn every_cut_reads_as_the_last_commit(committed: &[(Frame, i64)], flushed: &[(Frame, i64)]) {
    let all: Vec<Frame> = committed
        .iter()
        .chain(flushed)
        .map(|(frame, _)| frame.clone())
        .collect();
    let schema = SequenceSchema::from_frames(&all).unwrap();

    let whole = TempDir::new().unwrap();
    let (outcome, writes) = flush_twice(whole.path(), &schema, committed, flushed, UNLIMITED);
    outcome.expect("an uninterrupted flush succeeds");
    assert!(
        writes > 1,
        "the flush under test writes more than the marker"
    );

    let reference = TempDir::new().unwrap();
    let (outcome, _) = flush_twice(reference.path(), &schema, committed, &[], UNLIMITED);
    outcome.unwrap();
    let mut expected_seq = open_sequence(&storage_in(&reference));
    let expected: Vec<String> = (0..committed.len() as u64)
        .map(|index| resolved(&mut expected_seq, index))
        .collect();

    for cut in 0..writes as u64 {
        let dir = TempDir::new().unwrap();
        let (outcome, made) = flush_twice(dir.path(), &schema, committed, flushed, cut);
        assert!(outcome.is_err(), "a flush cut after {cut} writes must fail");
        let mut seq = MrecReader::from_storage(storage_in(&dir)).unwrap_or_else(|error| {
            panic!("cut after {cut} of {writes} writes ({made} landed): unreadable: {error}")
        });
        assert_eq!(
            committed_len(&mut seq),
            committed.len(),
            "cut after {cut} of {writes} writes: the commit marker did not move"
        );
        for (index, want) in expected.iter().enumerate() {
            assert_eq!(
                &resolved(&mut seq, index as u64),
                want,
                "cut after {cut} of {writes} writes: frame {index} changed"
            );
        }
    }
}

/// A block stated once and then changed: the flush that changes it replaces
/// the elision markers with `offset` / `step_index`.
#[test]
fn a_cut_index_materialization_reads_as_the_last_commit() {
    let committed: Vec<(Frame, i64)> = (0..3)
        .map(|step| (bonded_frame(step as f64, &[0, 1]), step))
        .collect();
    let flushed = vec![(bonded_frame(3.0, &[1, 0, 1]), 3)];
    every_cut_reads_as_the_last_commit(&committed, &flushed);
}

/// A fixed cell (the `box/` attributes) that changes: the flush that changes
/// it creates the `box/` arrays.
#[test]
fn a_cut_cell_materialization_reads_as_the_last_commit() {
    let framed = |step: i64, edge: f64| {
        let mut frame = atoms_frame(&[step as f64, 0.5]);
        frame.simbox = Some(cubic_cell(edge));
        (frame, step)
    };
    let committed: Vec<(Frame, i64)> = (0..3).map(|step| framed(step, 10.0)).collect();
    every_cut_reads_as_the_last_commit(&committed, &[framed(3, 10.5)]);
}

/// A run with no cell that gains one: the first cell arrives as arrays.
#[test]
fn a_cut_first_cell_reads_as_no_cell() {
    let committed: Vec<(Frame, i64)> = (0..2)
        .map(|step| (atoms_frame(&[step as f64, 0.5]), step))
        .collect();
    let mut late = atoms_frame(&[2.0, 0.5]);
    late.simbox = Some(cubic_cell(10.0));
    every_cut_reads_as_the_last_commit(&committed, &[(late, 2)]);
}

/// A step numbering that stops being a progression: the flush that breaks it
/// creates the `step` array.
#[test]
fn a_cut_step_materialization_reads_as_the_last_commit() {
    let committed: Vec<(Frame, i64)> = (0..3)
        .map(|step| (atoms_frame(&[step as f64, 0.5]), step))
        .collect();
    let flushed = vec![(atoms_frame(&[9.0, 0.5]), 10)];
    every_cut_reads_as_the_last_commit(&committed, &flushed);
}
