use super::*;

// =======================================================================
// H. The 2026-09 contract — root/meta, cadence, hints, rollback, reads
// =======================================================================

/// The streaming writer mints a record, not a bare `trajectory/`: the
/// root group and a `meta/` group exist before the first append, and the
/// group carries nothing the producer did not ask for.
#[test]
fn create_writes_the_root_and_an_empty_meta_group() {
    let dir = TempDir::new().unwrap();
    let store = storage_in(&dir);
    let schema = SequenceSchema::from_frames(std::slice::from_ref(&atoms_frame(&[1.0]))).unwrap();
    let writer = MrecWriter::create_in_storage(store.clone(), schema).unwrap();
    assert!(dir.path().join("zarr.json").is_file(), "root group");
    let meta = Group::open(store.clone(), "/meta").expect("meta group exists");
    assert!(meta.attributes().is_empty(), "{:?}", meta.attributes());
    drop(writer);
}

/// `with_meta` writes the identity document the producer hands in.
#[test]
fn with_meta_writes_the_identity_document() {
    let dir = TempDir::new().unwrap();
    let store = storage_in(&dir);
    let schema = SequenceSchema::from_frames(std::slice::from_ref(&atoms_frame(&[1.0]))).unwrap();
    let mut doc = serde_json::Map::new();
    doc.insert("creator".into(), serde_json::json!({"name": "test"}));
    let writer = MrecWriter::create_in_storage(store.clone(), schema)
        .unwrap()
        .with_meta(&doc)
        .unwrap();
    drop(writer);
    let meta = Group::open(store.clone(), "/meta").unwrap();
    assert_eq!(meta.attributes()["creator"]["name"], "test");
}

/// The landing cadence follows the frame size: a 1000-row `f64` frame
/// (8 KB) lands every 512 frames (4 MiB), rounded to whole chunks of
/// three frames; `with_flush_every` overrides it.
#[test]
fn the_landing_cadence_is_derived_from_the_frame_size() {
    let dir = TempDir::new().unwrap();
    let store = storage_in(&dir);
    let values: Vec<f64> = (0..1000).map(|i| i as f64).collect();
    let frame = atoms_frame(&values);
    let schema = SequenceSchema::from_frames(std::slice::from_ref(&frame)).unwrap();
    let mut writer = MrecWriter::create_in_storage(store.clone(), schema.clone()).unwrap();
    writer.append(&frame).unwrap();
    // 4 MiB / 8000 B = 524, rounded up to a multiple of 3 frames per
    // 2100-row... no: 1000-row frames reach the 16 KiB floor at 3 frames
    // (3000 rows), and ceil(524 / 3) * 3 = 525.
    assert_eq!(writer.flush_every(), 525);
    assert_eq!(writer.committed(), 0, "one frame is below the cadence");

    let dir2 = TempDir::new().unwrap();
    let store2 = storage_in(&dir2);
    let mut writer = MrecWriter::create_in_storage(store2.clone(), schema)
        .unwrap()
        .with_flush_every(2)
        .unwrap();
    for _ in 0..5 {
        writer.append(&frame).unwrap();
    }
    assert_eq!(
        writer.committed(),
        4,
        "two landings of two frames, one buffered"
    );
    writer.close().unwrap();
    let mut seq = open_sequence(&store2);
    assert_eq!(committed_len(&mut seq), 5);
}

/// A block whose updates all carry the same row count at ordinals
/// 0, 1, 2, … advertises `uniform_rows` and `dense_updates` and writes no
/// index arrays; a reader resolves it arithmetically. The moment a frame
/// breaks either rule the arrays are materialized and the hints withdrawn.
#[test]
fn regular_blocks_carry_hints_and_read_back_through_them() {
    let dir = TempDir::new().unwrap();
    let store = storage_in(&dir);
    let frames: Vec<Frame> = (0..5)
        .map(|i| atoms_frame(&[i as f64, i as f64 + 0.5]))
        .collect();
    write_all(&store, &frames);
    let group = Group::open(store.clone(), &format!("{TRAJ}/{ATOMS}")).unwrap();
    assert_eq!(group.attributes()["uniform_rows"], 2);
    assert_eq!(group.attributes()["dense_updates"], true);

    // There are no index arrays to decode: a reader that trusts the hints
    // resolves arithmetically off the column length alone.
    for name in ["step_index", "offset"] {
        assert!(
            Array::open(store.clone(), &format!("{TRAJ}/{ATOMS}/{name}")).is_err(),
            "a regular block writes no {name} array"
        );
    }
    let mut seq = open_sequence(&store);
    assert_eq!(atoms_x(&frame_at(&mut seq, 3)), vec![3.0, 3.5]);
    assert_eq!(seq.block_update_at(ATOMS, 4).unwrap(), Some(4));

    // A ragged run withdraws `uniform_rows`; a skipped ordinal withdraws
    // `dense_updates`.
    let dir = TempDir::new().unwrap();
    let store = storage_in(&dir);
    let mut frames = ragged_frames();
    frames.push(frames[2].clone()); // identical: no update at ordinal 3
    frames.push(atoms_frame(&[99.0]));
    write_all(&store, &frames);
    let group = Group::open(store.clone(), &format!("{TRAJ}/{ATOMS}")).unwrap();
    assert!(group.attributes().get("uniform_rows").is_none());
    assert!(group.attributes().get("dense_updates").is_none());
    let mut seq = open_sequence(&store);
    assert_eq!(
        atoms_x(&frame_at(&mut seq, 3)),
        vec![20.0, 21.0, 22.0, 23.0]
    );
    assert_eq!(atoms_x(&frame_at(&mut seq, 4)), vec![99.0]);
    assert_eq!(seq.block_update_at(ATOMS, 3).unwrap(), Some(2));
}

/// A crash that left every array longer than `step` is rolled back on
/// reopen, and appending continues from the committed frame with the CSR
/// pointer intact.
#[test]
fn a_reopened_writer_rolls_back_to_the_commit_marker() {
    let dir = TempDir::new().unwrap();
    let store = storage_in(&dir);
    let frames = ragged_frames();
    let mut writer =
        MrecWriter::create_in_storage(store.clone(), SequenceSchema::from_frames(&frames).unwrap())
            .unwrap()
            .with_flush_every(1)
            .unwrap();
    for frame in &frames {
        writer.append(frame).unwrap();
    }
    writer.close().unwrap();

    // Simulate a torn commit: `step` stays at 3 frames while the block's
    // arrays claim a fourth update of 7 rows.
    for (path, extra) in [
        (format!("{TRAJ}/{ATOMS}/{X}"), 7u64),
        (format!("{TRAJ}/{ATOMS}/offset"), 1),
        (format!("{TRAJ}/{ATOMS}/step_index"), 1),
    ] {
        let mut array = Array::open(store.clone(), &path).unwrap();
        let mut shape = array.shape().to_vec();
        shape[0] += extra;
        array.set_shape(shape).unwrap();
        array.store_metadata().unwrap();
    }
    {
        let array = Array::open(store.clone(), &format!("{TRAJ}/{ATOMS}/step_index")).unwrap();
        array
            .store_array_subset(
                &ArraySubset::new_with_start_shape(vec![3], vec![1]).unwrap(),
                vec![3u64],
            )
            .unwrap();
    }

    let mut seq = open_sequence(&store);
    assert_eq!(committed_len(&mut seq), 3, "the reader is bounded by step");
    assert_eq!(
        atoms_x(&frame_at(&mut seq, 2)),
        vec![20.0, 21.0, 22.0, 23.0]
    );

    let mut writer = MrecWriter::from_storage(store.clone()).unwrap();
    writer.append(&atoms_frame(&[30.0, 31.0])).unwrap();
    writer.close().unwrap();

    assert_eq!(
        u64_array(&store, &format!("{TRAJ}/{ATOMS}/offset")),
        vec![0, 3, 8, 12, 14],
        "the rolled-back CSR pointer continues from the committed total"
    );
    assert_eq!(
        u64_array(&store, &format!("{TRAJ}/{ATOMS}/step_index")),
        vec![0, 1, 2, 3]
    );
    let mut seq = open_sequence(&store);
    assert_eq!(atoms_x(&frame_at(&mut seq, 3)), vec![30.0, 31.0]);
    assert_eq!(
        atoms_x(&frame_at(&mut seq, 2)),
        vec![20.0, 21.0, 22.0, 23.0]
    );
}

/// `frame_columns` decodes only the named columns; the block still
/// resolves through the same three states.
#[test]
fn frame_columns_reads_only_the_named_columns() {
    let dir = TempDir::new().unwrap();
    let store = storage_in(&dir);
    let mut frame = atoms_frame(&[1.0, 2.0]);
    frame
        .get_mut(ATOMS)
        .unwrap()
        .insert("y", float_column(&[10.0, 20.0]), None)
        .unwrap();
    frame.insert(BONDS, block_with(I, uint_column(&[0, 1])));
    write_all(&store, &[frame]);

    let seq = open_sequence(&store);
    let picked = seq
        .frame_columns(0, &[(ATOMS, "y")])
        .unwrap()
        .expect("frame 0 exists");
    let atoms = picked.get(ATOMS).expect("the named block is present");
    assert!(atoms.get("y").is_some());
    assert!(atoms.get(X).is_none(), "an unnamed column is not decoded");
    assert!(picked.get(BONDS).is_none(), "an unnamed block is left out");
    assert_eq!(atoms.n_rows(), Some(2));

    let err = seq
        .frame_columns(0, &[(ATOMS, "nope")])
        .unwrap_err()
        .to_string();
    assert!(err.contains("nope"), "{err}");
}

/// `box_at` resolves the cell through the box index and caches it.
#[test]
fn box_at_resolves_the_latest_cell() {
    let dir = TempDir::new().unwrap();
    let store = storage_in(&dir);
    let mut with_cell = atoms_frame(&[1.0]);
    with_cell.simbox = Some(fixed_cell());
    write_all(
        &store,
        &[atoms_frame(&[0.0]), with_cell, atoms_frame(&[2.0])],
    );
    let seq = open_sequence(&store);
    assert!(seq.box_at(0).unwrap().is_none(), "no cell yet");
    let cell = seq.box_at(2).unwrap().expect("the cell carries forward");
    assert_eq!(cell.matrix()[1][1], 11.0);
    assert!(seq.box_at(3).unwrap().is_none(), "past the commit marker");
}

/// A shaped block keeps its row count: any other count is refused at
/// append, naming the block and both counts.
#[test]
fn a_shaped_block_with_another_row_count_is_refused_at_append() {
    let dir = TempDir::new().unwrap();
    let store = storage_in(&dir);
    let mut grid = Block::new();
    grid.insert("rho", float_column(&[1.0, 2.0, 3.0, 4.0]), None)
        .unwrap();
    grid.set_shape(&[2, 2]).unwrap();
    let mut frame = Frame::new();
    frame.insert("grid", grid);
    let schema = SequenceSchema::from_frames(std::slice::from_ref(&frame)).unwrap();
    let mut writer = MrecWriter::create_in_storage(store, schema).unwrap();
    writer.append(&frame).unwrap();
    let mut wrong = Frame::new();
    wrong.insert("grid", block_with("rho", float_column(&[1.0, 2.0])));
    let err = writer.append(&wrong).unwrap_err().to_string();
    assert!(err.contains("grid") && err.contains("4 rows"), "{err}");
}

/// A meta value that arrives at another width is re-read at the declared
/// width — a JSON list becomes the declared `f64x3` — and one that cannot
/// be is refused naming both.
#[test]
fn a_meta_value_at_another_width_is_read_at_the_declared_one() {
    let dir = TempDir::new().unwrap();
    let store = storage_in(&dir);
    let mut schema =
        SequenceSchema::from_frames(std::slice::from_ref(&atoms_frame(&[1.0]))).unwrap();
    schema.declare_meta("com", "f64x3", None).unwrap();
    schema.declare_meta("count", "i64", None).unwrap();
    let mut writer = MrecWriter::create_in_storage(store.clone(), schema).unwrap();
    let mut frame = atoms_frame(&[1.0]);
    frame
        .meta
        .insert("com", MetaValue::Json(serde_json::json!([1.0, 2.0, 3.0])));
    frame.meta.insert("count", MetaValue::I64(3));
    writer.append(&frame).unwrap();
    let mut wrong = atoms_frame(&[2.0]);
    wrong
        .meta
        .insert("com", MetaValue::Json(serde_json::json!([1.0, 2.0])));
    wrong.meta.insert("count", MetaValue::I64(4));
    let err = writer.append(&wrong).unwrap_err().to_string();
    assert!(err.contains("com") && err.contains("f64x3"), "{err}");
    // Declared keys are walked in declaration order, so a bad `com` masks
    // `count`. This frame's `com` fits; its `count` is an f64 under i64.
    let mut refused = atoms_frame(&[3.0]);
    refused
        .meta
        .insert("com", MetaValue::Json(serde_json::json!([4.0, 5.0, 6.0])));
    refused.meta.insert("count", MetaValue::F64(0.5));
    let err = writer.append(&refused).unwrap_err().to_string();
    assert!(err.contains("count") && err.contains("i64"), "{err}");
    writer.close().unwrap();
    let mut seq = open_sequence(&store);
    let back = frame_at(&mut seq, 0);
    assert_eq!(
        back.meta.get("com"),
        Some(&MetaValue::F64x3([1.0, 2.0, 3.0]))
    );
}

/// There are no narrow float tags: a per-step meta key declared with one
/// is refused, not widened to `f64`. The vector forms share this same
/// unknown-dtype arm of [`meta_layout`].
#[test]
fn declaring_a_narrow_float_meta_tag_is_refused() {
    let mut schema = SequenceSchema::new();
    for tag in ["f16", "f32"] {
        let err = schema
            .declare_meta("scale", tag, None)
            .unwrap_err()
            .to_string();
        assert!(err.contains("unknown dtype"), "accepted `{tag}`: {err}");
    }
}

/// A schema declared column by column is the same pin a derived one is.
#[test]
fn a_declared_schema_equals_the_derived_one() {
    let mut frame = atoms_frame(&[1.0, 2.0]);
    frame.insert(BONDS, block_with(I, uint_column(&[0])));
    frame.meta.insert("energy", MetaValue::F64(1.5));
    let derived = SequenceSchema::from_frames(std::slice::from_ref(&frame)).unwrap();

    let mut declared = SequenceSchema::new();
    declared
        .declare_column(ATOMS, X, DType::Float, &[])
        .unwrap();
    declared.declare_column(BONDS, I, DType::Uint, &[]).unwrap();
    declared.declare_meta("energy", "f64", None).unwrap();
    assert_eq!(declared, derived);

    // A canonical key is pinned at its vocabulary dtype.
    let err = declared
        .declare_column(ATOMS, X, DType::Int, &[])
        .unwrap_err()
        .to_string();
    assert!(err.contains("canonical") && err.contains("f64"), "{err}");
    declared
        .declare_column(ATOMS, PROBE, DType::Float, &[])
        .unwrap();
    let err = declared
        .declare_column(ATOMS, PROBE, DType::Int, &[])
        .unwrap_err()
        .to_string();
    assert!(err.contains("f64") && err.contains("i32"), "{err}");
    let err = declared
        .declare_column("step", "a", DType::Float, &[])
        .unwrap_err()
        .to_string();
    assert!(err.contains("reserved"), "{err}");
    let err = declared
        .declare_meta("bad", "f128", None)
        .unwrap_err()
        .to_string();
    assert!(err.contains("f128"), "{err}");
}

/// Every inner chunk ends in `crc32c`, floats stay raw and everything
/// else carries `gzip` level 1.
#[test]
fn inner_codecs_follow_the_width_policy() {
    let dir = TempDir::new().unwrap();
    let store = storage_in(&dir);
    let mut frame = atoms_frame(&[1.0]);
    frame.insert(BONDS, block_with(I, uint_column(&[0])));
    let mut second = atoms_frame(&[2.0]);
    second.insert(BONDS, block_with(I, uint_column(&[0])));
    let mut third = atoms_frame(&[3.0]);
    third.insert(BONDS, block_with(I, uint_column(&[0])));
    let mut writer = MrecWriter::create_in_storage(
        store.clone(),
        SequenceSchema::from_frames(&[frame.clone(), second.clone()]).unwrap(),
    )
    .unwrap();
    // Steps 0, 1, 5: not a progression (two values always are one), so
    // `step` is an array to inspect.
    writer.append_at(&frame, 0, None).unwrap();
    writer.append_at(&second, 1, None).unwrap();
    writer.append_at(&third, 5, None).unwrap();
    writer.close().unwrap();
    let inner_codecs = |path: &str| -> Vec<String> {
        let arr = Array::open(store.clone(), path).unwrap();
        let metadata = serde_json::to_value(arr.metadata()).unwrap();
        let sharding = metadata["codecs"]
            .as_array()
            .unwrap()
            .iter()
            .find(|codec| codec["name"] == "sharding_indexed")
            .unwrap()
            .clone();
        assert_eq!(sharding["configuration"]["index_location"], "start");
        sharding["configuration"]["codecs"]
            .as_array()
            .unwrap()
            .iter()
            .map(|codec| codec["name"].as_str().unwrap().to_string())
            .collect()
    };
    assert_eq!(
        inner_codecs(&format!("{TRAJ}/{ATOMS}/{X}")),
        vec!["bytes", "crc32c"]
    );
    assert_eq!(
        inner_codecs(&format!("{TRAJ}/{BONDS}/{I}")),
        vec!["bytes", "gzip", "crc32c"]
    );
    assert_eq!(
        inner_codecs(&format!("{TRAJ}/step")),
        vec!["bytes", "gzip", "crc32c"]
    );
}
