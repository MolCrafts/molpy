use super::*;

// =======================================================================
// F. Flush cadence — values never depend on it; dead bytes stay bounded
// =======================================================================

/// Whatever the flush cadence, the frames read back identical, and the
/// dead bytes an eager flusher leaves behind are bounded by one
/// re-encoded trailing chunk per flush — nothing is compacted at close,
/// because a whole-shard rewrite is the one write that can destroy
/// committed data on a crash.
#[test]
fn flush_cadence_changes_dead_bytes_but_never_what_reads_back() {
    const ROWS_PER_CHUNK: u64 = 64;
    const CHUNKS_PER_SHARD: u64 = 8;
    const FRAMES: usize = 24;
    /// One 64-row `f64` chunk, its `crc32c`, and the shard index slack.
    const CHUNK_BYTES_UPPER_BOUND: u64 = 64 * 8 + 4 + 16;

    // 24 frames of 5 rows = 120 rows: two inner chunks inside one shard.
    let frames: Vec<Frame> = (0..FRAMES)
        .map(|step| {
            let base = step as f64 * 10.0;
            atoms_frame(&[base, base + 1.0, base + 2.0, base + 3.0, base + 4.0])
        })
        .collect();
    let schema = SequenceSchema::from_frames(&frames).unwrap();

    let eager_dir = TempDir::new().unwrap();
    let eager_store = storage_in(&eager_dir);
    let mut eager = MrecWriter::create_in_storage(eager_store.clone(), schema)
        .unwrap()
        .with_rows_per_chunk(ROWS_PER_CHUNK)
        .unwrap()
        .with_chunks_per_shard(CHUNKS_PER_SHARD)
        .unwrap();
    for frame in &frames {
        eager.append(frame).unwrap();
        eager.flush().unwrap();
    }
    eager.close().unwrap();
    let column_dir = PathBuf::from("trajectory").join(ATOMS).join(X);
    let eager_bytes = chunk_bytes(&eager_dir.path().join(&column_dir));

    let lazy_dir = TempDir::new().unwrap();
    let lazy_store = storage_in(&lazy_dir);
    let mut lazy = MrecWriter::create_in_storage(
        lazy_store.clone(),
        SequenceSchema::from_frames(&frames).unwrap(),
    )
    .unwrap()
    .with_rows_per_chunk(ROWS_PER_CHUNK)
    .unwrap()
    .with_chunks_per_shard(CHUNKS_PER_SHARD)
    .unwrap()
    .with_flush_every(FRAMES as u64)
    .unwrap();
    for frame in &frames {
        lazy.append(frame).unwrap();
    }
    lazy.close().unwrap();
    let lazy_bytes = chunk_bytes(&lazy_dir.path().join(&column_dir));

    assert!(
        eager_bytes > lazy_bytes,
        "flushing every frame re-encodes the trailing chunk and leaves dead bytes: \
         {eager_bytes} B vs {lazy_bytes} B"
    );
    assert!(
        eager_bytes <= lazy_bytes + FRAMES as u64 * CHUNK_BYTES_UPPER_BOUND,
        "dead bytes are bounded by one chunk per flush: {eager_bytes} B vs {lazy_bytes} B"
    );

    let mut eager_seq = open_sequence(&eager_store);
    let mut lazy_seq = open_sequence(&lazy_store);
    assert_eq!(committed_len(&mut eager_seq), FRAMES);
    assert_eq!(committed_len(&mut lazy_seq), FRAMES);
    for index in 0..FRAMES as u64 {
        assert_eq!(
            atoms_x(&frame_at(&mut eager_seq, index)),
            atoms_x(&frame_at(&mut lazy_seq, index)),
            "frame {index} must not remember the flush cadence"
        );
    }
}

// =======================================================================
// G. Write amplification — the store half of the append fast path
// =======================================================================

/// A steady-state flush writes about one compressed chunk plus one shard
/// index **at the disk**, and does not grow with the shard file it lands
/// in (ac-011, store half).
///
/// This is the only thing standing between a 256 MiB shard default and a
/// silent O(shard) rewrite per flush: `zarrs_filesystem` 0.3.12 reports
/// `supports_set_partial() == true` while reading, patching and rewriting
/// the whole value, so the codec-layer measurement in `zarrs_pins` is
/// structurally blind to it. [`PositionalWriteStorage::bytes_written`] is
/// the observation that is not.
#[test]
fn a_steady_state_flush_writes_a_chunk_not_the_shard_file() {
    /// Rows per frame, and rows per inner chunk: one frame is one chunk,
    /// so one flush is one chunk write.
    const ROWS: u64 = 4096;
    /// 4096 f64 rows = 32 768 B uncompressed.
    const CHUNK_BYTES: u64 = ROWS * 8;
    /// A shard extent of 256 chunks = 1 048 576 rows = 8 MiB, so all 64
    /// flushes land in shard 0 and no seal-on-complete rewrite is in play.
    const CHUNKS_PER_SHARD: u64 = 256;
    /// Two `u64` per inner chunk plus the `crc32c` checksum.
    const INDEX_BYTES: u64 = CHUNKS_PER_SHARD * 16 + 4;
    /// The dense per-step arrays (`step`, the block's `offset` and
    /// `step_index`) and the metadata each flush restates. Their cost is
    /// bounded by the number of arrays, not by the run length; 64 KiB is
    /// a hard ceiling on it, three orders below the shard file the naive
    /// path would have rewritten.
    const AUXILIARY_BYTES: u64 = 64 * 1024;
    const FLUSHES: u64 = 64;

    let dir = TempDir::new().unwrap();
    let positional = Arc::new(PositionalWriteStorage::new(dir.path()).unwrap());
    let store: ReadableWritableListableStorage = positional.clone();

    let frame_of = |step: u64| {
        atoms_frame(
            &(0..ROWS)
                .map(|row| noise(step * ROWS + row))
                .collect::<Vec<f64>>(),
        )
    };
    let schema = SequenceSchema::from_frames(std::slice::from_ref(&frame_of(0))).unwrap();
    let mut writer = MrecWriter::create_in_storage(store, schema)
        .unwrap()
        .with_rows_per_chunk(ROWS)
        .unwrap()
        .with_chunks_per_shard(CHUNKS_PER_SHARD)
        .unwrap();

    let mut early = 0u64;
    let mut late = 0u64;
    for step in 0..FLUSHES {
        let before = positional.bytes_written();
        writer.append(&frame_of(step)).unwrap();
        writer.flush().unwrap();
        let delta = positional.bytes_written() - before;
        if step == 3 {
            early = delta;
        }
        if step == FLUSHES - 1 {
            late = delta;
        }
    }
    let file = chunk_bytes(&dir.path().join("trajectory").join(ATOMS).join(X));
    writer.close().unwrap();

    assert!(
        late <= CHUNK_BYTES + INDEX_BYTES + AUXILIARY_BYTES,
        "a flush must cost one chunk plus one shard index, not {late} B"
    );
    assert!(
        late <= 2 * early,
        "the per-flush cost must not grow with the file: {early} B at flush 4, \
         {late} B at flush {FLUSHES}"
    );
    assert!(
        file > 16 * late,
        "the measurement is only meaningful while the file dwarfs the write: \
         {file} B file, {late} B flush"
    );
}

// =======================================================================
// H. File count — bounded by bytes, not by frames
// =======================================================================

/// A store's file count follows its **bytes**, not its frame count
/// (ac-023, decision 13).
///
/// Every number below is hard-coded from the layout rather than measured.
/// [`MrecWriter::with_chunks_per_shard`] exists for this test: a
/// 256 MiB shard would swallow the whole fixture, so `k` is pressed down
/// to 4 and the store really does span several shards. With `R = 8` rows
/// per inner chunk, one shard spans `R * k = 32` rows, and every array
/// here is 8 B per row (`f64` x; `i64` step; `u64` offset and
/// `step_index`), so one shard file holds `S = 256` B.
///
/// Per growth array the file count is `ceil(rows / 32)` shard files plus
/// one `zarr.json`; the four groups (root, `meta/`, `trajectory/` and
/// `trajectory/atoms/`) add one `zarr.json` each. A regular run has
/// exactly one growth array: `x`. `step` is the `step_progression`
/// attribute and the block's CSR index is implied by its hints, so
/// neither costs a file. Eight 8-row frames are 64 rows of `x` — two
/// full shards — 7 files; sixteen frames take `x` to 128 rows, four
/// shards, 9 files. Doubling the frames adds two files.
///
/// The counted set is **every** file under the store root, `zarr.json`
/// included — the claim is about what lands on a filesystem, so excluding
/// the metadata would be excluding the part that scales with arrays.
#[test]
fn file_count_scales_with_bytes_not_frames() {
    /// Rows one fixture frame carries, and rows one inner chunk holds:
    /// one frame is exactly one chunk of the `x` column.
    const ROWS_PER_FRAME: u64 = 8;
    const ROWS_PER_CHUNK: u64 = 8;
    /// Pressed far below the derived default so a small store spans
    /// several shards.
    const CHUNKS_PER_SHARD: u64 = 4;
    /// Bytes one row of every array in this store occupies.
    const ROW_BYTES: u64 = 8;
    /// `S`: bytes one shard file holds, `R * k * row`.
    const SHARD_BYTES: u64 = ROWS_PER_CHUNK * CHUNKS_PER_SHARD * ROW_BYTES;
    /// The growth arrays: the block's `x` alone — `step` is a progression
    /// attribute and a regular block writes no index arrays. No `time`, no
    /// `meta`, no `box` — the fixture frames carry none.
    const ARRAYS: u64 = 1;
    /// The root, `meta/`, `trajectory/` and `trajectory/atoms/`.
    const GROUPS: u64 = 4;
    const FRAMES: u64 = 8;

    /// `O(arrays)` of the bound: each array costs its own `zarr.json` and
    /// rounds its last shard up, and each group costs a `zarr.json`.
    const ARRAY_FLOOR: u64 = 2 * ARRAYS + GROUPS;
    /// Rows of `x` at `FRAMES`: 8 * 8.
    const SINGLE_ROWS: u64 = 64;
    /// The same at `2 * FRAMES`.
    const DOUBLE_ROWS: u64 = 128;
    /// 4 group `zarr.json` + 1 array `zarr.json` + 2 `x` shards.
    const SINGLE_FILES: usize = 7;
    /// `x` on four shards.
    const DOUBLE_FILES: usize = 9;

    // Every frame carries distinct values, so every step earns its own
    // `step_index` entry: a repeated block would be stored once and the
    // row counts above would not be the ones on disk.
    let storage_files = |count: u64| -> Vec<PathBuf> {
        let dir = TempDir::new().unwrap();
        let store = storage_in(&dir);
        let frames: Vec<Frame> = (0..count)
            .map(|step| {
                atoms_frame(
                    &(0..ROWS_PER_FRAME)
                        .map(|row| noise(step * ROWS_PER_FRAME + row))
                        .collect::<Vec<f64>>(),
                )
            })
            .collect();
        write_all_with(&store, &frames, ROWS_PER_CHUNK, CHUNKS_PER_SHARD);
        file_map(dir.path()).into_keys().collect()
    };

    let single = storage_files(FRAMES);
    let double = storage_files(2 * FRAMES);

    assert_eq!(
        single.len(),
        SINGLE_FILES,
        "{FRAMES} frames must cost {SINGLE_FILES} files: {single:?}"
    );
    assert_eq!(
        double.len(),
        DOUBLE_FILES,
        "{} frames must cost {DOUBLE_FILES} files: {double:?}",
        2 * FRAMES
    );

    let bound = |rows: u64| ((rows * ROW_BYTES) / SHARD_BYTES + ARRAY_FLOOR) as usize;
    assert!(
        single.len() <= bound(SINGLE_ROWS),
        "file count must stay under total_bytes/S + O(arrays): {} > {}",
        single.len(),
        bound(SINGLE_ROWS)
    );
    assert!(
        double.len() <= bound(DOUBLE_ROWS),
        "file count must stay under total_bytes/S + O(arrays): {} > {}",
        double.len(),
        bound(DOUBLE_ROWS)
    );
    assert!(
        double.len() < 2 * single.len(),
        "twice the frames must not be twice the files: {} then {}",
        single.len(),
        double.len()
    );
}
