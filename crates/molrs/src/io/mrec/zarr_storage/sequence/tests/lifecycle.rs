use super::*;

// =======================================================================
// D. Lifecycle — create, open, commit, reopen, the knob window
// =======================================================================

/// `create` on a store that already holds a sequence is an `Err` naming
/// the path, and the existing store is left byte-unchanged (ac-018).
///
/// Never a silent overwrite: the store on that path is somebody's run.
#[test]
fn create_on_an_occupied_store_errs_and_changes_nothing() {
    let dir = TempDir::new().unwrap();
    let store = storage_in(&dir);
    let frames = ragged_frames();
    write_all(&store, &frames);
    let before = file_map(dir.path());

    let schema = SequenceSchema::from_frames(&frames).unwrap();
    let message = MrecWriter::create_in_storage(store, schema)
        .err()
        .expect("create must refuse an occupied store")
        .to_string();

    assert!(
        message.contains("trajectory"),
        "must name the occupied path: {message}"
    );
    assert_eq!(
        file_map(dir.path()),
        before,
        "a refused create must not have touched a byte"
    );
}

/// `open` refuses a store whose arrays disagree with the schema it minted,
/// naming what differed and both values (ac-018).
///
/// The store is corrupted the way a foreign writer would corrupt it — the
/// declared column `x` is replaced by an `i64` array — because the schema
/// pinned at `create` is exactly what `open` has to check the arrays
/// against.
#[test]
fn open_errs_naming_what_differed_and_both_values() {
    let dir = TempDir::new().unwrap();
    let store = storage_in(&dir);
    write_all(&store, &ragged_frames());

    let column = format!("{TRAJ}/{ATOMS}/{X}");
    store
        .erase_prefix(&StorePrefix::new(format!("trajectory/{ATOMS}/{X}/")).unwrap())
        .unwrap();
    ArrayBuilder::new(vec![12], vec![12], data_type::int64(), 0i64)
        .build(store.clone(), &column)
        .unwrap()
        .store_metadata()
        .unwrap();

    let message = MrecWriter::from_storage(store)
        .err()
        .expect("open must refuse a store that no longer matches its schema")
        .to_string();
    assert!(
        message.contains("sequence schema mismatch"),
        "the pinned message shape: {message}"
    );
    assert!(
        message.contains(ATOMS),
        "must name what differed: {message}"
    );
    assert!(
        message.contains("expected") && message.contains("found"),
        "must carry both values: {message}"
    );
    assert!(
        message.contains("f64"),
        "must carry the declared dtype (schema tag): {message}"
    );
    assert!(
        message.contains(&DType::I64.to_string()),
        "must carry the dtype found on disk: {message}"
    );
}

/// A store that conforms to the layout but carries no [`SCHEMA_ATTRIBUTE`]
/// still **reads**: the read door derives the schema from the store.
///
/// The attribute is the reference writer's own reopen pin
/// (molrec `docs/spec/trajectory.md`), not a read requirement — resolving a
/// frame needs `step_index`, `offset` and the array metadata, all of which
/// are on disk regardless. Refusing a conforming store for a missing
/// convenience pin locks every foreign writer out of the format.
///
/// The fixture is written by the real writer and then *stripped*, so the
/// only difference from a store this suite already reads is the one
/// attribute. It carries a ragged block column **and** a per-step meta key,
/// because the derivation has two halves — blocks/columns from the group
/// tree and array metadata, meta keys from `meta/`'s children and their
/// `meta_dtype` tags — and one half working is not the claim.
#[test]
fn a_conforming_store_without_the_schema_attribute_still_reads() {
    const TEMPERATURE: &str = "temperature";

    let dir = TempDir::new().unwrap();
    let store = storage_in(&dir);
    let frames: Vec<Frame> = ragged_frames()
        .into_iter()
        .enumerate()
        .map(|(step, mut frame)| {
            frame
                .meta
                .insert(TEMPERATURE, MetaValue::F64(300.0 + step as f64));
            frame
        })
        .collect();
    write_all(&store, &frames);
    strip_schema_attribute(&store);

    let mut seq = MrecReader::from_storage(store.clone())
        .expect("a conforming store reads without the writer's schema pin");
    assert_eq!(
        committed_len(&mut seq),
        frames.len(),
        "the derived sequence is as long as the run that was written"
    );
    for (index, frame) in frames.iter().enumerate() {
        let read = frame_at(&mut seq, index as u64);
        assert_eq!(
            atoms_x(&read),
            atoms_x(frame),
            "frame {index} must come back bit exact from a derived schema"
        );
        assert_eq!(
            read.meta.get(TEMPERATURE),
            frame.meta.get(TEMPERATURE),
            "the per-step meta key of frame {index} must survive the derivation"
        );
    }
}

/// The strict half: [`MrecWriter::open`] on that same stripped
/// store still errs, naming [`SCHEMA_ATTRIBUTE`].
///
/// A writer needs the *pinned* declaration — the fills for omitted meta keys
/// and the union a later frame is checked against — and neither can be
/// derived from data that was never written. Pinned next to the read door so
/// relaxing the reader cannot quietly relax the writer with it.
#[test]
fn a_writer_reopen_still_requires_the_schema_attribute() {
    let dir = TempDir::new().unwrap();
    let store = storage_in(&dir);
    write_all(&store, &ragged_frames());
    strip_schema_attribute(&store);

    let message = MrecWriter::from_storage(store)
        .err()
        .expect("a writer may not reopen a sequence whose schema pin is gone")
        .to_string();
    assert!(
        message.contains(SCHEMA_ATTRIBUTE),
        "must name the missing pin: {message}"
    );
}

/// `flush()` is the commit point, and the crash-loss boundary is exactly
/// "everything appended after it" (ac-015).
///
/// `std::mem::forget` is the crash: the writer never runs another line, so
/// what the reopened store shows is what a killed process would have left.
/// It is also why the writer must have no `Drop` — a `Drop` that flushed
/// would make this test's boundary unobservable, and would swallow IO
/// errors in the bargain.
#[test]
fn a_flush_commits_every_frame_appended_before_it() {
    let dir = TempDir::new().unwrap();
    let store = storage_in(&dir);
    let frames: Vec<Frame> = (0..5)
        .map(|step| atoms_frame(&[step as f64, step as f64 + 0.5]))
        .collect();

    let schema = SequenceSchema::from_frames(&frames).unwrap();
    let mut writer = MrecWriter::create_in_storage(store.clone(), schema).unwrap();
    for frame in &frames[..3] {
        writer.append(frame).unwrap();
    }
    writer.flush().unwrap();
    for frame in &frames[3..] {
        writer.append(frame).unwrap();
    }
    std::mem::forget(writer);

    let mut seq = open_sequence(&store);
    assert_eq!(
        committed_len(&mut seq),
        3,
        "a flush commits every buffered row, ragged tail included"
    );
    for (index, frame) in frames[..3].iter().enumerate() {
        assert_eq!(
            atoms_x(&frame_at(&mut seq, index as u64)),
            atoms_x(frame),
            "committed frame {index} must read bit exact"
        );
    }
    assert!(
        seq.frame(3).unwrap().is_none(),
        "frames appended after the last flush are absent, not partial"
    );
}

/// `close(self)` then `open` then appends that carry the run past an inner
/// chunk boundary: every frame reads bit exact and the extents are the
/// ones creation froze (ac-016).
#[test]
fn a_reopened_writer_appends_across_an_inner_chunk_boundary() {
    let dir = TempDir::new().unwrap();
    let store = storage_in(&dir);
    let frames: Vec<Frame> = (0..4)
        .map(|step| atoms_frame(&[step as f64, step as f64 + 0.25, step as f64 + 0.5]))
        .collect();

    let schema = SequenceSchema::from_frames(&frames).unwrap();
    let mut writer = MrecWriter::create_in_storage(store.clone(), schema)
        .unwrap()
        .with_rows_per_chunk(4)
        .unwrap()
        .with_chunks_per_shard(2)
        .unwrap();
    for frame in &frames[..2] {
        writer.append(frame).unwrap();
    }
    writer.close().unwrap();
    let column = format!("{TRAJ}/{ATOMS}/{X}");
    let frozen = extents(&store, &column);

    let mut writer = MrecWriter::from_storage(store.clone()).unwrap();
    for frame in &frames[2..] {
        writer.append(frame).unwrap();
    }
    writer.close().unwrap();

    let mut seq = open_sequence(&store);
    assert_eq!(committed_len(&mut seq), 4);
    for (index, frame) in frames.iter().enumerate() {
        assert_eq!(
            atoms_x(&frame_at(&mut seq, index as u64)),
            atoms_x(frame),
            "frame {index} must survive the reopen bit exact"
        );
    }
    assert_eq!(
        extents(&store, &column),
        frozen,
        "a reopened writer may not re-plan the extents"
    );
}

/// Both knobs are legal after `create` and before the first append, and
/// the first append freezes exactly what they asked for (decision 13).
///
/// This is the knobs' whole reason to exist: `with_rows_per_chunk`'s
/// consumer is the lifecycle tests (a boundary in a handful of frames
/// instead of 512 KiB of them) and `with_chunks_per_shard`'s is the file
/// count bound.
#[test]
fn the_knobs_set_the_extents_the_first_append_freezes() {
    const ROWS_PER_CHUNK: u64 = 4;
    const CHUNKS_PER_SHARD: u64 = 2;

    let dir = TempDir::new().unwrap();
    let store = storage_in(&dir);
    write_all_with(
        &store,
        &[atoms_frame(&[1.0, 2.0, 3.0])],
        ROWS_PER_CHUNK,
        CHUNKS_PER_SHARD,
    );

    assert_eq!(
        extents(&store, &format!("{TRAJ}/{ATOMS}/{X}")),
        (
            vec![ROWS_PER_CHUNK * CHUNKS_PER_SHARD],
            vec![ROWS_PER_CHUNK]
        ),
        "the knobs decide the frozen extents, not the byte target"
    );
}

/// A knob after the first append is an `Err` naming the phase: the extents
/// are frozen and `zarrs` cannot re-plan a grid under live data
/// (decision 3, decision 13).
#[test]
fn a_knob_after_the_first_append_errs_naming_the_phase() {
    let dir = TempDir::new().unwrap();
    let store = storage_in(&dir);
    let schema =
        SequenceSchema::from_frames(std::slice::from_ref(&atoms_frame(&[1.0, 2.0]))).unwrap();
    let mut writer = MrecWriter::create_in_storage(store, schema).unwrap();
    writer.append(&atoms_frame(&[1.0, 2.0])).unwrap();

    let message = writer
        .with_rows_per_chunk(8)
        .err()
        .expect("the knob window closes at the first append")
        .to_string();
    assert!(
        message.contains("append"),
        "must name the phase that closed the window: {message}"
    );
}

/// A store created and closed without a single append reopens as an empty
/// sequence (the knob window's consequence: the arrays are not created
/// until the first append, so such a store may hold only the group and its
/// schema attributes).
#[test]
fn a_created_but_never_appended_store_reopens_empty() {
    let dir = TempDir::new().unwrap();
    let store = storage_in(&dir);
    let schema = SequenceSchema::from_frames(std::slice::from_ref(&atoms_frame(&[1.0]))).unwrap();
    MrecWriter::create_in_storage(store.clone(), schema)
        .unwrap()
        .close()
        .unwrap();

    let mut seq = open_sequence(&store);
    assert_eq!(committed_len(&mut seq), 0, "no append, no committed frame");
    assert!(seq.frame(0).unwrap().is_none());
}
