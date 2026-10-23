use super::*;

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

/// A `FilesystemStore` that records the order of the keys written through
/// it. Reads and listings are the wrapped store's; only the write order is
/// this type's business.
///
/// It records `set` and `set_partial_many` — the two doors a value reaches
/// the disk through — and deliberately not `erase`, which is not a write
/// of content.
#[derive(Debug)]
struct RecordingStorage {
    inner: FilesystemStore,
    written: Mutex<Vec<String>>,
}

impl RecordingStorage {
    fn new(path: &Path) -> Self {
        Self {
            inner: FilesystemStore::new(path).unwrap(),
            written: Mutex::new(Vec::new()),
        }
    }

    fn record(&self, key: &StoreKey) {
        self.written.lock().unwrap().push(key.as_str().to_string());
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
        self.record(key);
        self.inner.set(key, value)
    }

    fn set_partial_many(
        &self,
        key: &StoreKey,
        offset_values: OffsetBytesIterator,
    ) -> Result<(), StorageError> {
        self.record(key);
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
