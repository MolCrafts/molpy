use super::*;

// -- declared precision (molrec F1) ------------------------------------

/// The inner codec names of the sharded growth array at `path`.
fn inner_codec_names(store: &ReadableWritableListableStorage, path: &str) -> Vec<String> {
    let arr = Array::open(store.clone(), path).unwrap();
    let metadata = serde_json::to_value(arr.metadata()).unwrap();
    metadata["codecs"]
        .as_array()
        .unwrap()
        .iter()
        .find(|codec| codec["name"] == "sharding_indexed")
        .unwrap()["configuration"]["codecs"]
        .as_array()
        .unwrap()
        .iter()
        .map(|codec| codec["name"].as_str().unwrap().to_string())
        .collect()
}

/// An atoms frame whose `x` declares precision `p`.
fn precise_frame(values: &[f64], p: f64) -> Frame {
    let mut frame = atoms_frame(values);
    frame.get_mut(ATOMS).unwrap().set_precision(X, p).unwrap();
    frame
}

fn stored(values: &[f64], p: f64) -> Vec<f64> {
    let q = molrs::core::quantum(p).unwrap();
    values
        .iter()
        .map(|&x| molrs::core::quantize(x, q))
        .collect()
}

#[test]
fn a_precision_is_pinned_and_every_frame_reads_back_rounded() {
    let dir = TempDir::new().unwrap();
    let store = storage_in(&dir);
    let p = 1e-3;
    let frames = [
        precise_frame(&[0.123_456, -1.000_49, 2.5], p),
        precise_frame(&[0.2, 7.777_7, -3.0], p),
    ];
    let schema = SequenceSchema::from_frames(&frames).unwrap();
    assert_eq!(schema.precision(ATOMS, X), Some(p));
    write_all(&store, &frames);

    let pin = Group::open(store.clone(), TRAJ).unwrap().attributes()[SCHEMA_ATTRIBUTE].clone();
    assert_eq!(
        pin["blocks"][ATOMS]["columns"][X]["precision"],
        serde_json::json!(p)
    );
    // Declared on the trajectory path in the pin only: no array attribute.
    let array = Array::open(store.clone(), &format!("{TRAJ}/{ATOMS}/{X}")).unwrap();
    assert!(array.attributes().get("precision").is_none());
    assert_eq!(
        inner_codec_names(&store, &format!("{TRAJ}/{ATOMS}/{X}")),
        ["bytes", "numcodecs.shuffle", "zstd", "crc32c"]
    );

    let mut seq = open_sequence(&store);
    for (index, frame) in frames.iter().enumerate() {
        let back = frame_at(&mut seq, index as u64);
        let presented = atoms_x(frame);
        assert_eq!(atoms_x(&back), stored(&presented, p));
        // The pinned precision travels with the column read back.
        assert_eq!(back.get(ATOMS).unwrap().precision(X), Some(p));
    }
}

#[test]
fn a_change_below_half_a_quantum_carries_forward() {
    let dir = TempDir::new().unwrap();
    let store = storage_in(&dir);
    let p = 1e-3;
    let q = molrs::core::quantum(p).unwrap();
    let first = [1.0, 2.0, 3.0];
    let nudged: Vec<f64> = first.iter().map(|x| x + 0.4 * q).collect();
    write_all(
        &store,
        &[precise_frame(&first, p), precise_frame(&nudged, p)],
    );

    let mut seq = open_sequence(&store);
    assert_eq!(seq.block_update_at(ATOMS, 0).unwrap(), Some(0));
    assert_eq!(seq.block_update_at(ATOMS, 1).unwrap(), Some(0));
    assert_eq!(atoms_x(&frame_at(&mut seq, 1)), first);
}

#[test]
fn a_frame_stating_another_precision_than_the_pin_is_refused() {
    let dir = TempDir::new().unwrap();
    let store = storage_in(&dir);
    let schema =
        SequenceSchema::from_frames(std::slice::from_ref(&precise_frame(&[1.0], 1e-3))).unwrap();
    let mut writer = MrecWriter::create_in_storage(store.clone(), schema).unwrap();
    // No precision stated: the pin rounds it.
    writer.append(&atoms_frame(&[1.000_1])).unwrap();
    let err = writer
        .append(&precise_frame(&[1.0], 1e-2))
        .unwrap_err()
        .to_string();
    assert!(err.contains("precision") && err.contains(X), "{err}");

    // A pin without a precision refuses a frame that states one.
    let dir = TempDir::new().unwrap();
    let store = storage_in(&dir);
    let schema = SequenceSchema::from_frames(std::slice::from_ref(&atoms_frame(&[1.0]))).unwrap();
    let mut writer = MrecWriter::create_in_storage(store, schema).unwrap();
    assert!(writer.append(&precise_frame(&[1.0], 1e-3)).is_err());
}

#[test]
fn declare_precision_refuses_what_cannot_carry_one() {
    let mut schema = SequenceSchema::new();
    schema.declare_column(ATOMS, X, DType::Float, &[]).unwrap();
    schema.declare_column(BONDS, I, DType::Uint, &[]).unwrap();
    assert!(schema.declare_precision(ATOMS, "nope", 1e-3).is_err());
    assert!(schema.declare_precision(BONDS, I, 1e-3).is_err());
    for bad in [0.0, -1e-3, f64::NAN, f64::INFINITY] {
        assert!(schema.declare_precision(ATOMS, X, bad).is_err(), "{bad}");
    }
    schema.declare_precision(ATOMS, X, 1e-3).unwrap();
    schema.declare_precision(ATOMS, X, 1e-3).unwrap();
    assert!(schema.declare_precision(ATOMS, X, 1e-2).is_err());
    assert_eq!(schema.precision(ATOMS, X), Some(1e-3));

    // Two frames declaring two precisions conflict at mint.
    let err =
        SequenceSchema::from_frames(&[precise_frame(&[1.0], 1e-3), precise_frame(&[1.0], 1e-2)])
            .unwrap_err()
            .to_string();
    assert!(err.contains("precision"), "{err}");
}

#[test]
fn the_compression_knob_replaces_the_precision_compressor_and_the_shuffle_stays() {
    let dir = TempDir::new().unwrap();
    let store = storage_in(&dir);
    let frame = precise_frame(&[1.0, 2.0], 1e-3);
    let mut writer = MrecWriter::create_in_storage(
        store.clone(),
        SequenceSchema::from_frames(std::slice::from_ref(&frame)).unwrap(),
    )
    .unwrap()
    .with_compression(super::Compression::Gzip(5))
    .unwrap();
    writer.append(&frame).unwrap();
    writer.close().unwrap();
    assert_eq!(
        inner_codec_names(&store, &format!("{TRAJ}/{ATOMS}/{X}")),
        ["bytes", "numcodecs.shuffle", "gzip", "crc32c"]
    );
}

/// The design's density measurement, on a small synthetic run: 3000
/// atoms uniform in a 40 Å box, 40 frames of a 0.05 Å random walk, one
/// inner chunk per column per frame. Prints bytes per atom per frame of
/// the three coordinate columns, lossless and at two precisions: the
/// shard files as written, and the chunk payload net of the fixed-size
/// shard index (`16 B × chunks_per_shard + 4` per shard, written whole
/// at the start of the shard, so a 40-frame run pays it unamortized).
#[test]
fn a_declared_precision_cuts_the_coordinate_bytes() {
    const ATOMS_N: usize = 3000;
    const FRAMES: usize = 40;
    let mut state = 0x2545_F491_4F6C_DD1Du64;
    let mut uniform = || {
        state ^= state << 13;
        state ^= state >> 7;
        state ^= state << 17;
        (state >> 11) as f64 / (1u64 << 53) as f64
    };
    let mut xyz: Vec<[f64; 3]> = (0..ATOMS_N)
        .map(|_| [40.0 * uniform(), 40.0 * uniform(), 40.0 * uniform()])
        .collect();
    let mut run = Vec::with_capacity(FRAMES);
    for _ in 0..FRAMES {
        run.push(xyz.clone());
        for atom in &mut xyz {
            for axis in atom.iter_mut() {
                *axis += 0.05 * (2.0 * uniform() - 1.0);
            }
        }
    }
    let bytes_per_atom_frame = |precision: Option<f64>| -> (f64, f64) {
        let dir = TempDir::new().unwrap();
        let store = storage_in(&dir);
        let frames: Vec<Frame> = run
            .iter()
            .map(|positions| {
                let mut block = Block::new();
                for (axis, key) in ["x", "y", "z"].into_iter().enumerate() {
                    let values: Vec<f64> = positions.iter().map(|p| p[axis]).collect();
                    block.insert(key, float_column(&values), None).unwrap();
                    if let Some(p) = precision {
                        block.set_precision(key, p).unwrap();
                    }
                }
                let mut frame = Frame::new();
                frame.insert(ATOMS, block);
                frame
            })
            .collect();
        write_all(&store, &frames);
        let mut bytes = 0u64;
        let mut index = 0u64;
        for key in ["x", "y", "z"] {
            let files = chunk_files(&dir.path().join("trajectory").join(ATOMS).join(key));
            let (shard, inner) = extents(&store, &format!("{TRAJ}/{ATOMS}/{key}"));
            bytes += files.values().map(|v| v.len() as u64).sum::<u64>();
            index += files.len() as u64 * (16 * (shard[0] / inner[0]) + 4);
        }
        let per = (ATOMS_N * FRAMES) as f64;
        (bytes as f64 / per, (bytes - index) as f64 / per)
    };
    let (lossless, lossless_net) = bytes_per_atom_frame(None);
    let (milli, milli_net) = bytes_per_atom_frame(Some(1e-3));
    let (centi, centi_net) = bytes_per_atom_frame(Some(1e-2));
    eprintln!(
        "B/atom/frame (files / net of shard index): lossless {lossless:.2} / \
         {lossless_net:.2}, p=1e-3 {milli:.2} / {milli_net:.2}, p=1e-2 {centi:.2} / \
         {centi_net:.2}"
    );
    assert!(lossless_net > 23.9, "lossless is raw f64: {lossless_net}");
    assert!(milli_net < 8.0, "p=1e-3: {milli_net}");
    assert!(centi_net < 6.5, "p=1e-2: {centi_net}");
    assert!(milli < lossless / 2.5, "p=1e-3: {milli} vs {lossless}");
    assert!(centi < milli, "p=1e-2: {centi} vs {milli}");
}
