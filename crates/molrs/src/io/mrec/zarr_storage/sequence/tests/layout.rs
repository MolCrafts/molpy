use super::*;

// =======================================================================
// A. Layout primitives — CSR, per-section step_index, derived extents
// =======================================================================

/// `offset` is the CSR row pointer of the ragged run and nothing else:
/// `offset[0] = 0` and the prefix sums of 3, 5 and 4 rows (ac-014).
///
/// The row *counts* are never stored — they are `diff(offset)`, decision 2
/// — so this array is the only place the raggedness lives.
#[test]
fn csr_offset_is_the_prefix_sum_of_a_ragged_run() {
    let dir = TempDir::new().unwrap();
    let store = storage_in(&dir);
    write_all(&store, &ragged_frames());

    assert_eq!(
        u64_array(&store, &format!("{TRAJ}/{ATOMS}/offset")),
        vec![0, 3, 8, 12],
        "offset is the prefix sum of 3, 5 and 4 rows"
    );
}

/// `nstep` counts every appended frame, and a regular numbering lives in
/// the `step_progression` attribute rather than in an array.
#[test]
fn step_counts_every_appended_frame() {
    let dir = TempDir::new().unwrap();
    let store = storage_in(&dir);
    write_all(&store, &ragged_frames());
    let group = Group::open(store.clone(), TRAJ).unwrap();
    assert_eq!(group.attributes()["nstep"], 3);
    assert_eq!(
        group.attributes()["step_progression"],
        serde_json::json!({"start": 0, "stride": 1})
    );
    assert!(
        Array::open(store.clone(), &format!("{TRAJ}/step")).is_err(),
        "a regular step series needs no array"
    );
    let seq = open_sequence(&store);
    assert_eq!(seq.steps(), &[0, 1, 2]);
}

/// Each frame of the ragged run reads back its own rows, bit exact
/// (ac-014).
#[test]
fn each_frame_of_a_ragged_run_reads_back_its_own_rows() {
    let dir = TempDir::new().unwrap();
    let store = storage_in(&dir);
    let frames = ragged_frames();
    write_all(&store, &frames);

    let mut seq = open_sequence(&store);
    for (index, frame) in frames.iter().enumerate() {
        assert_eq!(
            atoms_x(&frame_at(&mut seq, index as u64)),
            atoms_x(frame),
            "frame {index} must come back bit exact"
        );
    }
}

/// Blocks and columns read back in the order the frame carried them —
/// the pinned schema records declaration order, and the reader walks it.
#[test]
fn blocks_and_columns_read_back_in_frame_order() {
    let dir = TempDir::new().unwrap();
    let store = storage_in(&dir);
    let mut zeta = Block::new();
    for column in ["c", "a", "b"] {
        zeta.insert(column, float_column(&[1.0]), None).unwrap();
    }
    let mut frame = Frame::new();
    frame.insert("zeta", zeta);
    frame.insert(ATOMS, block_with(X, float_column(&[2.0])));
    write_all(&store, &[frame]);

    let back = frame_at(&mut open_sequence(&store), 0);
    assert_eq!(back.keys().collect::<Vec<_>>(), ["zeta", ATOMS]);
    assert_eq!(back["zeta"].keys().collect::<Vec<_>>(), ["c", "a", "b"]);
}

/// A block with no `step_index` entry `<= i` is **absent** at step `i`,
/// not present-and-empty (ac-014).
///
/// Absence is how the layout expresses "this section does not exist yet";
/// a zero-row block would be a different claim (it exists and holds
/// nothing), and molrec's conformance suite distinguishes the two.
#[test]
fn a_block_absent_at_a_step_reads_as_absent_not_empty() {
    let dir = TempDir::new().unwrap();
    let store = storage_in(&dir);
    let mut with_bonds = atoms_frame(&[10.0, 11.0]);
    with_bonds.insert(BONDS, block_with(I, uint_column(&[0, 1])));
    let frames = vec![atoms_frame(&[1.0, 2.0]), with_bonds];
    write_all(&store, &frames);

    let mut seq = open_sequence(&store);
    assert!(
        frame_at(&mut seq, 0).get(BONDS).is_none(),
        "a block whose first step_index entry is 1 does not exist at step 0"
    );
    assert_eq!(
        frame_at(&mut seq, 1)
            .get(BONDS)
            .expect("bonds exists at step 1")
            .n_rows(),
        Some(2),
        "and at step 1 it carries the rows it was appended with"
    );
}

/// A fixed cell over a 20-step run writes the cell once, and omits the
/// optional box arrays that would only restate defaults (decision 7).
#[test]
fn a_fixed_cell_omits_default_origin_boundary_and_trivial_step_index() {
    const STEPS: usize = 20;

    let dir = TempDir::new().unwrap();
    let store = storage_in(&dir);
    let frames: Vec<Frame> = (0..STEPS)
        .map(|step| {
            let mut frame = atoms_frame(&[step as f64, step as f64 + 0.5]);
            frame.simbox = Some(fixed_cell());
            frame
        })
        .collect();
    write_all(&store, &frames);

    assert!(
        Array::open(store.clone(), &format!("{TRAJ}/box/step_index")).is_err(),
        "a single update at ordinal 0 does not write step_index"
    );
    assert!(
        Array::open(store.clone(), &format!("{TRAJ}/box/origin")).is_err(),
        "a zero origin is the default and is not written"
    );
    assert!(
        Array::open(store.clone(), &format!("{TRAJ}/box/boundary")).is_err(),
        "all-periodic boundary is the default and is not written"
    );
    let mut seq = open_sequence(&store);
    let cell = frame_at(&mut seq, 19)
        .simbox
        .expect("later frames still carry the cell");
    assert_eq!(cell.origin(), [0.0; 3]);
    assert_eq!(cell.pbc(), [true, true, true]);
}

/// An undefined cell's `vectors` are ignored on read, in both box
/// layouts: zeros in the fixed-cell attributes, and an all-zero (fill
/// value) `vectors` array, read back as the identity instead of failing
/// to invert.
#[test]
fn an_undefined_cell_reads_whatever_vectors_it_carries() {
    let undefined =
        |x: f64| SimBox::new(Cell::Undefined, [x, 0.0, 0.0], [false, false, false]).unwrap();

    // Fixed cell: `vectors` is a group attribute.
    let dir = TempDir::new().unwrap();
    let store = storage_in(&dir);
    let mut frame = atoms_frame(&[1.0]);
    frame.simbox = Some(undefined(0.0));
    write_all(&store, &[frame]);
    let box_path = format!("{TRAJ}/box");
    let mut attributes = Group::open(store.clone(), &box_path)
        .unwrap()
        .attributes()
        .clone();
    assert_eq!(attributes["cell_defined"], serde_json::json!(false));
    attributes.insert(
        "vectors".into(),
        serde_json::json!([[0.0, 0.0, 0.0], [0.0, 0.0, 0.0], [0.0, 0.0, 0.0]]),
    );
    GroupBuilder::new()
        .attributes(attributes)
        .build(store.clone(), &box_path)
        .unwrap()
        .store_metadata()
        .unwrap();
    let back = open_sequence(&store).box_at(0).unwrap().unwrap();
    assert!(!back.is_cell_defined());
    assert_eq!(
        back.matrix(),
        [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]]
    );

    // Changing origin: the cell lives in arrays. Dropping the `vectors`
    // chunks makes every row the zero fill value.
    let dir = TempDir::new().unwrap();
    let store = storage_in(&dir);
    let mut first = atoms_frame(&[1.0]);
    first.simbox = Some(undefined(0.0));
    let mut second = atoms_frame(&[2.0]);
    second.simbox = Some(undefined(1.0));
    write_all(&store, &[first, second]);
    let chunks = dir.path().join("trajectory/box/vectors/c");
    assert!(chunks.is_dir(), "the cell moved into arrays");
    std::fs::remove_dir_all(chunks).unwrap();
    let seq = open_sequence(&store);
    for (index, x) in [(0, 0.0), (1, 1.0)] {
        let back = seq.box_at(index).unwrap().unwrap();
        assert!(!back.is_cell_defined());
        assert_eq!(
            back.matrix(),
            [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]]
        );
        assert_eq!(back.origin()[0], x);
    }
}

/// Rewrite the attributes of the group at `path` through `edit`.
fn edit_attributes(
    store: &ReadableWritableListableStorage,
    path: &str,
    edit: impl FnOnce(&mut serde_json::Map<String, serde_json::Value>),
) {
    let mut group = Group::open(store.clone(), path).unwrap();
    edit(group.attributes_mut());
    group.store_metadata().unwrap();
}

/// A regular three-frame run: `atoms` carries both elision markers and
/// `step` is a progression.
fn regular_storage(dir: &TempDir) -> ReadableWritableListableStorage {
    let store = storage_in(dir);
    write_all(
        &store,
        &[
            atoms_frame(&[1.0]),
            atoms_frame(&[2.0]),
            atoms_frame(&[3.0]),
        ],
    );
    let attrs = Group::open(store.clone(), &format!("{TRAJ}/{ATOMS}"))
        .unwrap()
        .attributes()
        .clone();
    assert_eq!(attrs["uniform_rows"], serde_json::json!(1));
    assert_eq!(attrs["dense_updates"], serde_json::json!(true));
    store
}

#[test]
fn malformed_elision_markers_are_refused() {
    let section = format!("{TRAJ}/{ATOMS}");
    for edit in [
        |a: &mut serde_json::Map<String, serde_json::Value>| {
            a.remove("dense_updates");
        },
        |a: &mut serde_json::Map<String, serde_json::Value>| {
            a.remove("uniform_rows");
        },
        |a: &mut serde_json::Map<String, serde_json::Value>| {
            a.insert("uniform_rows".into(), serde_json::json!(0));
        },
        |a: &mut serde_json::Map<String, serde_json::Value>| {
            a.insert("dense_updates".into(), serde_json::json!(false));
        },
    ] {
        let dir = TempDir::new().unwrap();
        let store = regular_storage(&dir);
        edit_attributes(&store, &section, edit);
        let err = MrecReader::from_storage(store.clone())
            .err()
            .unwrap()
            .to_string();
        assert!(err.contains(ATOMS), "{err}");
        assert!(MrecWriter::from_storage(store).is_err());
    }
}

#[test]
fn markers_on_a_block_with_no_columns_are_refused_and_a_bare_declared_block_reads_absent() {
    let dir = TempDir::new().unwrap();
    let store = regular_storage(&dir);
    edit_attributes(&store, TRAJ, |a| {
        a[SCHEMA_ATTRIBUTE]["blocks"]["ghost"] = serde_json::json!({"columns": {}});
    });
    GroupBuilder::new()
        .build(store.clone(), &format!("{TRAJ}/ghost"))
        .unwrap()
        .store_metadata()
        .unwrap();
    // Declared, never updated: absent everywhere, and not refused.
    let seq = open_sequence(&store);
    assert!(!seq.has_block("ghost"));
    assert!(!frame_at(&mut open_sequence(&store), 1).contains_key("ghost"));
    drop(seq);

    edit_attributes(&store, &format!("{TRAJ}/ghost"), |a| {
        a.insert("uniform_rows".into(), serde_json::json!(1));
        a.insert("dense_updates".into(), serde_json::json!(true));
    });
    let err = MrecReader::from_storage(store).err().unwrap().to_string();
    assert!(err.contains("ghost") && err.contains("no columns"), "{err}");
}

#[test]
fn a_progression_without_the_commit_marker_is_refused() {
    let dir = TempDir::new().unwrap();
    let store = regular_storage(&dir);
    let attrs = Group::open(store.clone(), TRAJ)
        .unwrap()
        .attributes()
        .clone();
    assert!(attrs.contains_key("step_progression"));
    edit_attributes(&store, TRAJ, |a| {
        a.remove("nstep");
    });
    let err = MrecReader::from_storage(store.clone())
        .err()
        .unwrap()
        .to_string();
    assert!(err.contains("nstep"), "{err}");
    edit_attributes(&store, TRAJ, |a| {
        a.insert("nstep".into(), serde_json::json!(-1));
    });
    assert!(MrecReader::from_storage(store).is_err());
}

/// An undefined trajectory cell is periodic on no axis: an omitted
/// boundary reads all-false, and a periodic flag in the store is refused
/// (no `SimBox` can carry one to write).
#[test]
fn an_undefined_trajectory_cell_is_periodic_on_no_axis() {
    let undefined = |pbc| SimBox::new(Cell::Undefined, [0.0, 0.0, 0.0], pbc).unwrap();
    let mut frame = atoms_frame(&[1.0]);
    let dir = TempDir::new().unwrap();
    let store = storage_in(&dir);
    frame.simbox = Some(undefined([false; 3]));
    write_all(&store, &[frame]);
    let box_path = format!("{TRAJ}/box");
    let mut group = Group::open(store.clone(), &box_path).unwrap();
    assert_eq!(
        group.attributes()["boundary"],
        serde_json::json!([false, false, false])
    );
    group.attributes_mut().remove("boundary");
    group.store_metadata().unwrap();
    assert_eq!(
        open_sequence(&store).box_at(0).unwrap().unwrap().pbc(),
        [false; 3]
    );

    let mut group = Group::open(store.clone(), &box_path).unwrap();
    group
        .attributes_mut()
        .insert("boundary".into(), serde_json::json!([true, false, false]));
    group.store_metadata().unwrap();
    let refused = match MrecReader::from_storage(store.clone()) {
        Err(_) => true,
        Ok(seq) => seq.box_at(0).is_err(),
    };
    assert!(
        refused,
        "a periodic flag on an undefined cell must be refused"
    );
}

/// A canonical identifier key cannot be declared narrower than `u64`:
/// the in-memory block widens it, so nothing could be appended, and a
/// store holding one is refused on read.
#[test]
fn a_canonical_identifier_cannot_be_declared_narrow() {
    let mut schema = SequenceSchema::new();
    for dtype in [DType::U8, DType::U16, DType::U32] {
        let err = schema
            .declare_column("bonds", "atomi", dtype, &[])
            .unwrap_err()
            .to_string();
        assert!(err.contains("atomi") && err.contains(dtype.name()), "{err}");
    }
    schema
        .declare_column("bonds", "atomi", DType::Uint, &[])
        .unwrap();
    // A non-canonical name keeps its arrival width.
    schema
        .declare_column("bonds", "my_label", DType::U32, &[])
        .unwrap();
}

/// A non-default origin is recorded — as a `box/` attribute while the
/// cell is fixed, as an array once the cell changes; the still-default
/// boundary is omitted either way.
#[test]
fn a_nonzero_origin_is_written() {
    let cell = |lengths: f64| {
        SimBox::new(
            Cell::Matrix([[lengths, 0.0, 0.0], [0.0, 11.0, 0.0], [0.0, 0.0, 12.0]]),
            [1.0, 2.0, 3.0],
            [true, true, true],
        )
        .unwrap()
    };
    let dir = TempDir::new().unwrap();
    let store = storage_in(&dir);
    let mut frame = atoms_frame(&[1.0]);
    frame.simbox = Some(cell(10.0));
    write_all(&store, &[frame]);
    let group = Group::open(store.clone(), &format!("{TRAJ}/box")).unwrap();
    assert_eq!(
        group.attributes()["origin"],
        serde_json::json!([1.0, 2.0, 3.0])
    );
    assert!(
        group.attributes().get("boundary").is_none(),
        "still-default boundary is omitted"
    );
    assert!(
        Array::open(store.clone(), &format!("{TRAJ}/box/vectors")).is_err(),
        "a fixed cell is attributes, not arrays"
    );
    let seq = open_sequence(&store);
    let back = seq.box_at(0).unwrap().unwrap();
    assert_eq!(back.origin().to_vec(), vec![1.0, 2.0, 3.0]);

    // The first change migrates the cell into arrays.
    let dir = TempDir::new().unwrap();
    let store = storage_in(&dir);
    let mut first = atoms_frame(&[1.0]);
    first.simbox = Some(cell(10.0));
    let mut second = atoms_frame(&[2.0]);
    second.simbox = Some(cell(10.5));
    write_all(&store, &[first, second]);
    let origin: Vec<f64> = {
        let arr = Array::open(store.clone(), &format!("{TRAJ}/box/origin")).unwrap();
        let subset = ArraySubset::new_with_shape(arr.shape().to_vec());
        arr.retrieve_array_subset(&subset).unwrap()
    };
    assert_eq!(origin, vec![1.0, 2.0, 3.0, 1.0, 2.0, 3.0]);
    assert_eq!(
        u64_array(&store, &format!("{TRAJ}/box/step_index")),
        vec![0, 1]
    );
    assert!(Array::open(store.clone(), &format!("{TRAJ}/box/boundary")).is_err());
    let group = Group::open(store.clone(), &format!("{TRAJ}/box")).unwrap();
    assert!(
        group.attributes().get("vectors").is_none(),
        "the attribute form is withdrawn"
    );
    let seq = open_sequence(&store);
    assert_eq!(seq.box_at(1).unwrap().unwrap().matrix()[0][0], 10.5);
    assert_eq!(seq.box_at(0).unwrap().unwrap().matrix()[0][0], 10.0);
}

/// A `step` key on frame.meta does not mint `trajectory/meta/step`.
#[test]
fn frame_meta_step_is_not_a_per_step_meta_array() {
    let dir = TempDir::new().unwrap();
    let store = storage_in(&dir);
    let mut frame = atoms_frame(&[1.0]);
    frame.meta.insert("step", MetaValue::I64(0));
    write_all(&store, &[frame]);

    assert!(
        Array::open(store.clone(), &format!("{TRAJ}/meta/step")).is_err(),
        "the commit marker is trajectory/step, not meta/step"
    );
    let mut seq = open_sequence(&store);
    assert!(
        frame_at(&mut seq, 0).meta.get("step").is_none(),
        "layout-owned keys are not echoed back as frame meta"
    );
}

/// A block that never changes costs one update — and, being regular
/// (one update at ordinal 0), no index arrays at all: the group hints
/// carry it and every frame resolves to update 0.
#[test]
fn a_constant_block_costs_one_update_and_no_index_arrays() {
    const STEPS: usize = 5;
    let dir = TempDir::new().unwrap();
    let store = storage_in(&dir);
    let frames: Vec<Frame> = (0..STEPS)
        .map(|step| {
            let mut frame = atoms_frame(&[step as f64, step as f64 + 0.5]);
            frame.insert(BONDS, block_with(I, uint_column(&[0, 1])));
            frame
        })
        .collect();
    write_all(&store, &frames);

    let group = Group::open(store.clone(), &format!("{TRAJ}/{BONDS}")).unwrap();
    assert_eq!(group.attributes()["uniform_rows"], 2);
    assert_eq!(group.attributes()["dense_updates"], true);
    for name in ["step_index", "offset"] {
        assert!(
            Array::open(store.clone(), &format!("{TRAJ}/{BONDS}/{name}")).is_err(),
            "a regular block writes no {name} array"
        );
    }
    let rows = Array::open(store.clone(), &format!("{TRAJ}/{BONDS}/{I}"))
        .unwrap()
        .shape()[0];
    assert_eq!(rows, 2, "one update's rows, not one per frame");
    let seq = open_sequence(&store);
    for index in 0..STEPS as u64 {
        assert_eq!(seq.block_update_at(BONDS, index).unwrap(), Some(0));
        assert_eq!(
            seq.frame(index)
                .unwrap()
                .unwrap()
                .get(BONDS)
                .unwrap()
                .n_rows(),
            Some(2)
        );
    }
}

/// A block that changes every step is regular too — update `i` at ordinal
/// `i` with a fixed row count — so it costs no index arrays either; a
/// ragged run is what materializes `step_index` and `offset`.
#[test]
fn a_block_changing_every_step_is_regular_and_a_ragged_one_is_indexed() {
    let dir = TempDir::new().unwrap();
    let store = storage_in(&dir);
    let frames: Vec<Frame> = (0..5)
        .map(|i| atoms_frame(&[i as f64, i as f64 + 0.5]))
        .collect();
    write_all(&store, &frames);
    assert!(Array::open(store.clone(), &format!("{TRAJ}/{ATOMS}/step_index")).is_err());
    let seq = open_sequence(&store);
    for index in 0..5u64 {
        assert_eq!(seq.block_update_at(ATOMS, index).unwrap(), Some(index));
    }

    let dir = TempDir::new().unwrap();
    let store = storage_in(&dir);
    write_all(&store, &ragged_frames());
    assert_eq!(
        u64_array(&store, &format!("{TRAJ}/{ATOMS}/step_index")),
        vec![0, 1, 2]
    );
    assert_eq!(
        u64_array(&store, &format!("{TRAJ}/{ATOMS}/offset")),
        vec![0, 3, 8, 12]
    );
}

/// A frame whose rows straddle an inner chunk boundary reads back bit
/// exact (ac-014).
///
/// Frames are **not** rounded up to whole chunks (Design § 1: "a frame may
/// span chunks"), which is what lets a million-atom frame work at all. With
/// `R = 4` and three-row frames, frame 1 occupies rows 3..6 and therefore
/// spans the boundary at row 4.
#[test]
fn a_frame_straddling_an_inner_chunk_boundary_reads_back_bit_exact() {
    let dir = TempDir::new().unwrap();
    let store = storage_in(&dir);
    let frames = vec![
        atoms_frame(&[1.0, 2.0, 3.0]),
        atoms_frame(&[4.0, 5.0, 6.0]),
        atoms_frame(&[7.0, 8.0, 9.0]),
    ];
    write_all_with(&store, &frames, 4, 2);

    let mut seq = open_sequence(&store);
    assert_eq!(
        atoms_x(&frame_at(&mut seq, 1)),
        vec![4.0, 5.0, 6.0],
        "the frame at rows 3..6 crosses the chunk boundary at row 4"
    );
}

/// The worked example, on disk: a 3000-atom `f64` xyz block derives a
/// **frame-aligned** inner chunk of exactly one frame and a shard of
/// 3728 such chunks.
///
/// Hard-coded from the formulas, not from a measurement: row = 3 * 8 =
/// 24 B; the byte floor is `16 KiB / 24 = 682` rows, which 3000 exceeds,
/// so `R = 3000` (one frame); chunk = 72 000 B;
/// `k = clamp(floor(256 MiB / 72 000), 1, 4096) = 3728`; shard =
/// 3000 * 3728 = 11 184 000 rows.
#[test]
fn the_worked_example_derives_one_frame_per_chunk_and_k_3728() {
    const ATOM_COUNT: usize = 3000;
    const ROWS_PER_CHUNK: u64 = 3000;
    const CHUNKS_PER_SHARD: u64 = 3728;
    const ROWS_PER_SHARD: u64 = 11_184_000;

    let dir = TempDir::new().unwrap();
    let store = storage_in(&dir);
    let values: Vec<f64> = (0..ATOM_COUNT * 3).map(|i| i as f64 * 0.5).collect();
    let mut frame = Frame::new();
    frame.insert(
        ATOMS,
        block_with(
            "xyz",
            Column::from(ArrayD::from_shape_vec(vec![ATOM_COUNT, 3], values).unwrap()),
        ),
    );
    write_all(&store, &[frame]);

    let (shard, inner) = extents(&store, &format!("{TRAJ}/{ATOMS}/xyz"));
    assert_eq!(
        inner,
        vec![ROWS_PER_CHUNK, 3],
        "R = one 3000-row frame (the 16 KiB floor is 682 rows), trailing axis whole"
    );
    assert_eq!(
        shard,
        vec![ROWS_PER_SHARD, 3],
        "the shard spans R * k = 11 184 000 rows"
    );
    assert_eq!(
        shard[0] / inner[0],
        CHUNKS_PER_SHARD,
        "k = clamp(floor(256 MiB / 72 000 B chunk), 1, 4096) = 3728"
    );
}

/// A small frame is rounded up to the byte floor in whole frames: 100
/// `f64` rows (800 B) need 21 frames to pass 16 KiB, so `R = 2100`.
#[test]
fn a_small_frame_is_chunked_in_whole_frames_up_to_the_byte_floor() {
    let dir = TempDir::new().unwrap();
    let store = storage_in(&dir);
    let values: Vec<f64> = (0..100).map(|i| i as f64).collect();
    write_all(&store, &[atoms_frame(&values)]);
    let (_, inner) = extents(&store, &format!("{TRAJ}/{ATOMS}/{X}"));
    assert_eq!(inner, vec![2100], "ceil(2048 / 100) * 100 = 2100 rows");
}

/// The dense per-frame arrays — written once a series stops being
/// regular — share one small extent whatever the frames look like: 1024
/// rows per inner chunk, 256 chunks per shard.
#[test]
fn dense_arrays_take_the_dense_extents() {
    let dir = TempDir::new().unwrap();
    let store = storage_in(&dir);
    let frames = ragged_frames();
    let mut writer =
        MrecWriter::create_in_storage(store.clone(), SequenceSchema::from_frames(&frames).unwrap())
            .unwrap();
    // Steps 0, 1, 5: not a progression, so `step` becomes an array.
    for (frame, step) in frames.iter().zip([0i64, 1, 5]) {
        writer.append_at(frame, step, None).unwrap();
    }
    writer.close().unwrap();
    for path in [
        format!("{TRAJ}/step"),
        format!("{TRAJ}/{ATOMS}/offset"),
        format!("{TRAJ}/{ATOMS}/step_index"),
    ] {
        let (shard, inner) = extents(&store, &path);
        assert_eq!(inner, vec![1024], "{path}");
        assert_eq!(shard, vec![1024 * 256], "{path}");
    }
    assert_eq!(i64_array(&store, &format!("{TRAJ}/step")), vec![0, 1, 5]);
    let seq = open_sequence(&store);
    assert_eq!(seq.steps(), &[0, 1, 5]);
}
