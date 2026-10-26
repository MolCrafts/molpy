use super::*;

// -- nullable columns across a run -------------------------------------

/// The mask every nullable fixture carries over its three rows: the first
/// row valid, the last two null. Asymmetric on purpose — a reversed mask
/// cannot match it by accident.
const MASK: [bool; 3] = [true, false, false];

/// A frame whose `atoms` block carries one masked [`PROBE`] column.
fn masked_frame(values: &[f64], validity: &[bool]) -> Frame {
    let mut block = Block::new();
    block
        .insert(
            PROBE,
            ArrayD::from_shape_vec(vec![values.len()], values.to_vec()).unwrap(),
            Some(validity.to_vec()),
        )
        .unwrap();
    let mut frame = Frame::new();
    frame.insert(ATOMS, block);
    frame
}

/// The mask of column [`PROBE`] as that frame came back from the store.
fn atoms_mask(frame: &Frame) -> Option<Vec<bool>> {
    frame
        .get(ATOMS)
        .expect("the frame carries an atoms block")
        .validity(PROBE)
        .map(<[bool]>::to_vec)
}

/// A mask belongs to the frame it was appended with: two frames that both
/// mask the column must both say so on the way back, not just the one
/// whose rows happen to sit at the head of the growth array.
#[test]
fn every_frame_keeps_the_mask_of_its_masked_column() {
    let dir = TempDir::new().unwrap();
    let store = storage_in(&dir);
    write_all(
        &store,
        &[
            masked_frame(&[1.0, 0.0, 0.0], &MASK),
            masked_frame(&[2.0, 0.0, 0.0], &MASK),
        ],
    );

    let mut seq = open_sequence(&store);
    assert_eq!(atoms_mask(&frame_at(&mut seq, 0)), Some(MASK.to_vec()));
    assert_eq!(atoms_mask(&frame_at(&mut seq, 1)), Some(MASK.to_vec()));
}

/// The mask is per frame, not per column of the run: a column that is
/// whole in one frame and holed in the next keeps both answers, so a
/// declared-nullable column cannot make every frame look masked.
#[test]
fn a_column_masked_in_one_frame_only_keeps_both_answers() {
    let dir = TempDir::new().unwrap();
    let store = storage_in(&dir);
    write_all(
        &store,
        &[
            masked_frame(&[1.0, 2.0, 3.0], &[true, true, true]),
            masked_frame(&[4.0, 0.0, 0.0], &MASK),
        ],
    );

    let mut seq = open_sequence(&store);
    assert_eq!(atoms_mask(&frame_at(&mut seq, 0)), None);
    assert_eq!(atoms_mask(&frame_at(&mut seq, 1)), Some(MASK.to_vec()));
}

/// Whether a column may hold nulls is part of the pin, not a per-append
/// discovery: the declaration derived from a masked frame says so.
#[test]
fn from_frames_declares_a_masked_column_nullable() {
    let schema = SequenceSchema::from_frames(&[masked_frame(&[1.0, 0.0, 0.0], &MASK)]).unwrap();
    assert!(schema.blocks[ATOMS].columns[PROBE].nullable);
}

/// And a run that never masks the column declares it plain — the flag
/// states something, rather than being set for every column there is.
#[test]
fn from_frames_leaves_an_unmasked_column_not_nullable() {
    let schema = SequenceSchema::from_frames(&[atoms_frame(&[1.0, 2.0, 3.0])]).unwrap();
    assert!(!schema.blocks[ATOMS].columns[X].nullable);
}

/// The declaration is the union over the frames it is derived from: one
/// frame masking the column is enough, otherwise the schema minted from a
/// run whose first frame is whole could not describe its second.
#[test]
fn from_frames_unions_nullable_across_frames() {
    let schema = SequenceSchema::from_frames(&[
        masked_frame(&[1.0, 2.0, 3.0], &[true, true, true]),
        masked_frame(&[4.0, 0.0, 0.0], &MASK),
    ])
    .unwrap();
    assert!(schema.blocks[ATOMS].columns[PROBE].nullable);
}

/// The flag survives the pin written to `trajectory/`: a reopened writer
/// reads its own declaration back out of the store, so a flag that only
/// existed in memory would be lost at the first reopen.
#[test]
fn the_pinned_schema_carries_the_nullable_flag() {
    let dir = TempDir::new().unwrap();
    let store = storage_in(&dir);
    write_all(&store, &[masked_frame(&[1.0, 0.0, 0.0], &MASK)]);

    let group = Group::open(store.clone(), TRAJ).expect("the sequence group exists");
    let pinned: SequenceSchema = serde_json::from_value(
        group
            .attributes()
            .get(SCHEMA_ATTRIBUTE)
            .expect("the writer pins its schema")
            .clone(),
    )
    .expect("the pin deserializes");
    assert!(pinned.blocks[ATOMS].columns[PROBE].nullable);
}

/// `nullable` is optional in a pin: a column that omits it is plain.
#[test]
fn a_schema_pin_without_nullable_deserializes_as_not_nullable() {
    let pinned =
        r#"{"blocks":{"atoms":{"columns":{"probe":{"dtype":"f64","trailing":[]}}}},"meta":{}}"#;
    let schema: SequenceSchema =
        serde_json::from_str(pinned).expect("a pin without nullable deserializes");
    assert!(!schema.blocks[ATOMS].columns[PROBE].nullable);
}

/// A frame whose values repeat while its mask moves is a *different*
/// frame. Change detection compares the validity masks, so the second
/// frame earns an update of its own; without the mask in the comparison
/// it would earn none and read back under the first frame's mask, which
/// is the silent loss the comparison exists to close.
#[test]
fn a_frame_repeating_its_values_under_a_moved_mask_keeps_its_own_mask() {
    /// [`MASK`] with the middle row filled in — same values underneath.
    const MOVED: [bool; 3] = [true, true, false];
    let dir = TempDir::new().unwrap();
    let store = storage_in(&dir);
    write_all(
        &store,
        &[
            masked_frame(&[1.0, 0.0, 0.0], &MASK),
            masked_frame(&[1.0, 0.0, 0.0], &MOVED),
        ],
    );

    let mut seq = open_sequence(&store);
    assert_eq!(atoms_mask(&frame_at(&mut seq, 0)), Some(MASK.to_vec()));
    assert_eq!(atoms_mask(&frame_at(&mut seq, 1)), Some(MOVED.to_vec()));
    assert_eq!(seq.block_update_at(ATOMS, 0).unwrap(), Some(0));
    assert_eq!(
        seq.block_update_at(ATOMS, 1).unwrap(),
        Some(1),
        "a mask that moved is a change, and a change costs an update"
    );
}

/// The converse guard: the mask entering the comparison must not make
/// every masked frame look new. Two frames identical in values *and*
/// mask still collapse into the one update an unchanging block costs.
#[test]
fn two_frames_identical_in_values_and_mask_cost_one_update() {
    let dir = TempDir::new().unwrap();
    let store = storage_in(&dir);
    write_all(
        &store,
        &[
            masked_frame(&[1.0, 0.0, 0.0], &MASK),
            masked_frame(&[1.0, 0.0, 0.0], &MASK),
        ],
    );

    let mut seq = open_sequence(&store);
    assert_eq!(seq.block_update_at(ATOMS, 0).unwrap(), Some(0));
    assert_eq!(
        seq.block_update_at(ATOMS, 1).unwrap(),
        Some(0),
        "an unchanged block, mask included, earns no second update"
    );
    let probe = |frame: &Frame| -> Vec<f64> {
        frame
            .get(ATOMS)
            .expect("the frame carries an atoms block")
            .get(PROBE)
            .expect("the atoms block carries the probe column")
            .get::<f64>()
            .expect("the probe column arrived as f64")
            .iter()
            .copied()
            .collect()
    };
    let first = frame_at(&mut seq, 0);
    let second = frame_at(&mut seq, 1);
    assert_eq!(probe(&first), vec![1.0, 0.0, 0.0]);
    assert_eq!(probe(&second), probe(&first));
    assert_eq!(atoms_mask(&first), Some(MASK.to_vec()));
    assert_eq!(atoms_mask(&second), atoms_mask(&first));
}

/// `declare_nullable` is the door a hand-declared schema opts its column
/// in through: with the call, a masked frame appends and the mask
/// survives the round trip, exactly as a derived schema's would.
#[test]
fn a_hand_declared_nullable_column_accepts_a_masked_frame() {
    let dir = TempDir::new().unwrap();
    let store = storage_in(&dir);
    let mut schema = SequenceSchema::new();
    schema.declare_block(ATOMS, Some(3)).unwrap();
    schema
        .declare_column(ATOMS, PROBE, DType::Float, &[])
        .unwrap();
    schema.declare_nullable(ATOMS, PROBE).unwrap();

    let mut writer = MrecWriter::create_in_storage(store.clone(), schema).unwrap();
    writer
        .append(&masked_frame(&[1.0, 0.0, 0.0], &MASK))
        .unwrap();
    writer.close().unwrap();

    let mut seq = open_sequence(&store);
    assert_eq!(atoms_mask(&frame_at(&mut seq, 0)), Some(MASK.to_vec()));
}

/// Without that call the same append is refused rather than landed
/// mask-less: the pin is a contract, and the error names the block and
/// the column whose mask has nowhere to go.
#[test]
fn a_masked_frame_is_refused_by_a_non_nullable_declaration() {
    let dir = TempDir::new().unwrap();
    let store = storage_in(&dir);
    let mut schema = SequenceSchema::new();
    schema.declare_block(ATOMS, Some(3)).unwrap();
    schema
        .declare_column(ATOMS, PROBE, DType::Float, &[])
        .unwrap();

    let mut writer = MrecWriter::create_in_storage(store.clone(), schema).unwrap();
    let err = writer
        .append(&masked_frame(&[1.0, 0.0, 0.0], &MASK))
        .unwrap_err()
        .to_string();
    assert!(
        err.contains(PROBE) && err.contains(ATOMS) && err.contains("non-nullable"),
        "{err}"
    );
}

/// Nullability is a property of a declared column, so declaring it on a
/// column that was never declared is an error naming the column.
#[test]
fn declare_nullable_on_an_undeclared_column_is_refused() {
    let mut schema = SequenceSchema::new();
    schema.declare_block(ATOMS, None).unwrap();
    let err = schema
        .declare_nullable(ATOMS, PROBE)
        .unwrap_err()
        .to_string();
    assert!(err.contains(PROBE) && err.contains(ATOMS), "{err}");
}

/// And on a column of a block that was never declared either — the same
/// refusal, naming both, rather than a silently inserted declaration.
#[test]
fn declare_nullable_on_an_undeclared_block_is_refused() {
    let mut schema = SequenceSchema::new();
    let err = schema.declare_nullable(BONDS, I).unwrap_err().to_string();
    assert!(
        err.contains(BONDS) && err.contains(&format!("{I:?}")),
        "{err}"
    );
}
