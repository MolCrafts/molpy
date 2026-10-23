use super::*;

// =======================================================================
// B. Schema — derived union, enforcement at append, reserved names
// =======================================================================

/// A union mint over heterogeneous frames writes a store in which every
/// block **carries forward** from its latest update: a frame reads back
/// what it presented plus whatever earlier frames left standing.
///
/// `from_frames` unions the blocks, per-section `step_index` records when
/// each one changed, and omission means "unchanged" — the reading that
/// makes a topology written once cost one entry.
#[test]
fn a_union_mint_carries_each_block_forward_from_its_latest_update() {
    let dir = TempDir::new().unwrap();
    let store = storage_in(&dir);

    let plain = atoms_frame(&[1.0, 2.0]);
    let mut bonded = atoms_frame(&[10.0, 11.0]);
    bonded.insert(BONDS, block_with(I, uint_column(&[0, 1])));
    let mut charged = atoms_frame(&[20.0, 21.0]);
    charged.insert("charges", block_with("q", float_column(&[-0.5, 0.5])));
    let frames = vec![plain, bonded, charged];
    write_all(&store, &frames);

    let mut seq = open_sequence(&store);
    let step0 = frame_at(&mut seq, 0);
    assert_eq!(
        step0.len(),
        1,
        "step 0 presented only atoms; nothing to carry"
    );
    assert_eq!(atoms_x(&step0), vec![1.0, 2.0]);

    let step1 = frame_at(&mut seq, 1);
    assert_eq!(step1.len(), 2, "step 1 presented atoms and bonds");
    assert!(step1.get(BONDS).is_some());
    assert!(step1.get("charges").is_none());

    let step2 = frame_at(&mut seq, 2);
    assert_eq!(
        step2.len(),
        3,
        "step 2 presented atoms and charges and carries bonds forward"
    );
    assert!(step2.get("charges").is_some());
    assert_eq!(
        step2.get(BONDS).expect("bonds carried forward").n_rows(),
        Some(2)
    );
    assert_eq!(
        u64_array(&store, &format!("{TRAJ}/{BONDS}/step_index")),
        vec![1],
        "a block omitted after it appeared earns no update"
    );
}

/// The three presence states, on one store: omitted = carried forward,
/// zero rows = present and empty, no entry yet = absent.
#[test]
fn omitted_zero_row_and_unseen_blocks_read_back_as_carried_empty_and_absent() {
    let dir = TempDir::new().unwrap();
    let store = storage_in(&dir);

    let mut with_bonds = atoms_frame(&[1.0]);
    with_bonds.insert(BONDS, block_with(I, uint_column(&[0, 1])));
    let omits_bonds = atoms_frame(&[2.0]);
    let mut empties_bonds = atoms_frame(&[3.0]);
    empties_bonds.insert(BONDS, block_with(I, uint_column(&[])));
    let omits_again = atoms_frame(&[4.0]);
    // Frame 0 has no bonds yet: absent there.
    let frames = vec![
        atoms_frame(&[0.5]),
        with_bonds,
        omits_bonds,
        empties_bonds,
        omits_again,
    ];
    write_all(&store, &frames);

    let mut seq = open_sequence(&store);
    assert!(
        frame_at(&mut seq, 0).get(BONDS).is_none(),
        "no update at or before frame 0: absent"
    );
    assert_eq!(frame_at(&mut seq, 1).get(BONDS).unwrap().n_rows(), Some(2));
    assert_eq!(
        frame_at(&mut seq, 2).get(BONDS).unwrap().n_rows(),
        Some(2),
        "an omitted block carries forward"
    );
    let empty = frame_at(&mut seq, 3);
    let bonds = empty.get(BONDS).expect("a zero-row update is present");
    assert_eq!(bonds.n_rows(), Some(0), "and empty");
    assert!(bonds.get(I).is_some(), "with its declared columns");
    assert_eq!(
        frame_at(&mut seq, 4).get(BONDS).unwrap().n_rows(),
        Some(0),
        "an omission after the empty update carries the empty block forward"
    );
    assert_eq!(
        u64_array(&store, &format!("{TRAJ}/{BONDS}/step_index")),
        vec![1, 3]
    );
    assert_eq!(
        u64_array(&store, &format!("{TRAJ}/{BONDS}/offset")),
        vec![0, 2, 2]
    );
}

/// The same column arriving with two dtypes is a mint-time `Err` naming
/// the column and both values (ac-020).
///
/// Both values are spelled with the schema dtype tag — the same vocabulary
/// the store is written in, which for the domain aliases is molrec's
/// concrete-width spelling (`f64`/`u64`), not [`DType`]'s alias name.
#[test]
fn from_frames_errs_naming_the_column_and_both_dtypes() {
    let mut as_float = Frame::new();
    as_float.insert(ATOMS, block_with(PROBE, float_column(&[1.0, 2.0])));
    let mut as_uint = Frame::new();
    as_uint.insert(ATOMS, block_with(PROBE, uint_column(&[1, 2])));
    let frames = vec![as_float, as_uint];

    let message = SequenceSchema::from_frames(&frames)
        .expect_err("a conflicting dtype must be rejected at mint")
        .to_string();
    assert!(message.contains(PROBE), "must name the column: {message}");
    assert!(
        message.contains("f64"),
        "must carry the first dtype (schema tag): {message}"
    );
    assert!(
        message.contains("u64"),
        "must carry the second dtype (schema tag): {message}"
    );
}

/// The same column arriving with two trailing shapes is a mint-time `Err`
/// naming the column and both values (ac-020).
#[test]
fn from_frames_errs_naming_the_column_and_both_trailing_shapes() {
    let shaped = |trailing: usize| {
        let mut frame = Frame::new();
        frame.insert(
            ATOMS,
            block_with(
                "pos",
                Column::from(
                    ArrayD::from_shape_vec(vec![2, trailing], vec![0.5; 2 * trailing]).unwrap(),
                ),
            ),
        );
        frame
    };
    let frames = vec![shaped(3), shaped(9)];

    let message = SequenceSchema::from_frames(&frames)
        .expect_err("a conflicting trailing shape must be rejected at mint")
        .to_string();
    assert!(message.contains("pos"), "must name the column: {message}");
    assert!(
        message.contains('3'),
        "must carry the first shape: {message}"
    );
    assert!(
        message.contains('9'),
        "must carry the second shape: {message}"
    );
}

/// Appending a block the schema never declared is an `Err` naming it
/// (ac-020).
#[test]
fn an_undeclared_block_errs_at_append_naming_it() {
    let dir = TempDir::new().unwrap();
    let store = storage_in(&dir);
    let declared = atoms_frame(&[1.0, 2.0]);
    let schema = SequenceSchema::from_frames(std::slice::from_ref(&declared)).unwrap();
    let mut writer = MrecWriter::create_in_storage(store, schema).unwrap();

    let mut extra = atoms_frame(&[3.0, 4.0]);
    extra.insert(BONDS, block_with(I, uint_column(&[0, 1])));
    let message = writer
        .append(&extra)
        .expect_err("an undeclared block must not be written")
        .to_string();
    assert!(message.contains(BONDS), "must name the block: {message}");
}

/// Appending a column the schema never declared is an `Err` naming it
/// (ac-020) — no mid-run schema extension, decision 6.
#[test]
fn an_undeclared_column_errs_at_append_naming_it() {
    let dir = TempDir::new().unwrap();
    let store = storage_in(&dir);
    let schema =
        SequenceSchema::from_frames(std::slice::from_ref(&atoms_frame(&[1.0, 2.0]))).unwrap();
    let mut writer = MrecWriter::create_in_storage(store, schema).unwrap();

    let mut wider = Block::new();
    wider.insert(X, float_column(&[3.0, 4.0]), None).unwrap();
    wider.insert("fx", float_column(&[0.0, 0.0]), None).unwrap();
    let mut frame = Frame::new();
    frame.insert(ATOMS, wider);
    let message = writer
        .append(&frame)
        .expect_err("an undeclared column must not be written")
        .to_string();
    assert!(message.contains("fx"), "must name the column: {message}");
}

/// Appending a declared column at a different dtype is an `Err` naming it
/// (ac-020).
#[test]
fn a_changed_dtype_errs_at_append_naming_the_column() {
    let dir = TempDir::new().unwrap();
    let store = storage_in(&dir);
    let mut declared = Frame::new();
    declared.insert(ATOMS, block_with(PROBE, float_column(&[1.0, 2.0])));
    let schema = SequenceSchema::from_frames(std::slice::from_ref(&declared)).unwrap();
    let mut writer = MrecWriter::create_in_storage(store, schema).unwrap();

    let mut frame = Frame::new();
    frame.insert(ATOMS, block_with(PROBE, uint_column(&[3, 4])));
    let message = writer
        .append(&frame)
        .expect_err("a dtype change must not be written")
        .to_string();
    assert!(message.contains(PROBE), "must name the column: {message}");
}

/// Appending a declared column at a different trailing shape is an `Err`
/// naming it (ac-020).
#[test]
fn a_changed_trailing_shape_errs_at_append_naming_the_column() {
    let dir = TempDir::new().unwrap();
    let store = storage_in(&dir);
    let mut declared = Frame::new();
    declared.insert(
        ATOMS,
        block_with(
            "pos",
            Column::from(ArrayD::from_shape_vec(vec![2, 3], vec![0.5; 6]).unwrap()),
        ),
    );
    let schema = SequenceSchema::from_frames(std::slice::from_ref(&declared)).unwrap();
    let mut writer = MrecWriter::create_in_storage(store, schema).unwrap();

    let mut widened = Frame::new();
    widened.insert(
        ATOMS,
        block_with(
            "pos",
            Column::from(ArrayD::from_shape_vec(vec![2, 9], vec![0.5; 18]).unwrap()),
        ),
    );
    let message = writer
        .append(&widened)
        .expect_err("a trailing shape change must not be written")
        .to_string();
    assert!(message.contains("pos"), "must name the column: {message}");
}

/// A frame presenting a strict subset of the declared union is accepted —
/// that *is* the sparse expression (ac-020).
#[test]
fn a_frame_presenting_a_subset_of_the_declared_blocks_is_accepted() {
    let dir = TempDir::new().unwrap();
    let store = storage_in(&dir);
    let mut bonded = atoms_frame(&[10.0, 11.0]);
    bonded.insert(BONDS, block_with(I, uint_column(&[0, 1])));
    let schema = SequenceSchema::from_frames(&[atoms_frame(&[1.0, 2.0]), bonded]).unwrap();
    let mut writer = MrecWriter::create_in_storage(store, schema).unwrap();

    writer
        .append(&atoms_frame(&[1.0, 2.0]))
        .expect("a frame carrying only atoms is a legal subset of the union");
    writer.close().unwrap();
}

/// `trajectory/`'s reserved child names are refused when the schema is
/// **minted**, not at the first append (ac-020).
///
/// Minting is the only door: a schema that cannot be minted cannot reach
/// `create`, so this is also why no test hands `create` a reserved-name
/// schema — the type system makes that unreachable.
#[test]
fn a_reserved_block_name_is_rejected_at_mint() {
    for reserved in ["step", "time", "meta", "box"] {
        let mut frame = Frame::new();
        frame.insert(reserved, block_with(X, float_column(&[1.0])));
        let message = SequenceSchema::from_frames(std::slice::from_ref(&frame))
            .expect_err("a reserved block name must be rejected at mint")
            .to_string();
        assert!(
            message.contains(reserved),
            "must name the reserved block {reserved}: {message}"
        );
    }
}

/// A block's reserved column names (`offset`, `step_index`) are refused
/// at mint too (ac-020).
#[test]
fn a_reserved_column_name_is_rejected_at_mint() {
    for reserved in ["offset", "step_index"] {
        let mut frame = Frame::new();
        frame.insert(ATOMS, block_with(reserved, uint_column(&[0, 1])));
        let message = SequenceSchema::from_frames(std::slice::from_ref(&frame))
            .expect_err("a reserved column name must be rejected at mint")
            .to_string();
        assert!(
            message.contains(reserved),
            "must name the reserved column {reserved}: {message}"
        );
    }
}
