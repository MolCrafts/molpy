use super::*;

// -- row references (molrec F4) -----------------------------------------

/// `atoms` with `n` rows and a `refs` block whose `site` references them.
fn referencing_frame(n: usize, site: &[u64]) -> Frame {
    let mut frame = atoms_frame(&vec![0.5; n]);
    let mut refs = block_with("site", uint_column(site));
    refs.set_target("site", ATOMS).unwrap();
    frame.insert("refs", refs);
    frame
}

#[test]
fn targets_are_pinned_and_hold_per_resolved_frame() {
    let dir = TempDir::new().unwrap();
    let store = storage_in(&dir);
    let first = referencing_frame(3, &[2]);
    let schema = SequenceSchema::from_frames(std::slice::from_ref(&first)).unwrap();
    assert_eq!(schema.target("refs", "site"), Some(ATOMS));
    let mut writer = MrecWriter::create_in_storage(store.clone(), schema).unwrap();
    writer.append(&first).unwrap();
    // `atoms` shrinks to 2 rows while `refs` carries forward pointing at
    // row 2: the resolved frame breaks the reference.
    let err = writer
        .append(&atoms_frame(&[1.0, 2.0]))
        .unwrap_err()
        .to_string();
    assert!(
        err.contains("site") && err.contains("trajectory frame 1"),
        "{err}"
    );
    // Restating `refs` in range is fine; a refused frame changed nothing.
    let mut second = atoms_frame(&[1.0, 2.0]);
    second.insert("refs", block_with("site", uint_column(&[1])));
    writer.append(&second).unwrap();
    writer.close().unwrap();

    let pin = Group::open(store.clone(), TRAJ).unwrap().attributes()[SCHEMA_ATTRIBUTE].clone();
    assert_eq!(
        pin["blocks"]["refs"]["targets"],
        serde_json::json!({"site": ATOMS})
    );
    let mut seq = open_sequence(&store);
    assert_eq!(committed_len(&mut seq), 2);
    let back = frame_at(&mut seq, 1);
    assert_eq!(back["refs"].target("site"), Some(ATOMS));

    // Tamper the pin to a target the store breaks: the read refuses.
    let mut group = Group::open(store.clone(), TRAJ).unwrap();
    group.attributes_mut()[SCHEMA_ATTRIBUTE]["blocks"]["refs"]["targets"]["site"] =
        serde_json::json!("ghost");
    group.store_metadata().unwrap();
    assert!(open_sequence(&store).frame(0).is_err());
}

#[test]
fn declare_target_refuses_what_cannot_be_a_reference() {
    let mut schema = SequenceSchema::new();
    schema.declare_column(ATOMS, X, DType::Float, &[]).unwrap();
    schema
        .declare_column("refs", "site", DType::Uint, &[])
        .unwrap();
    assert!(schema.declare_target("refs", "nope", ATOMS).is_err());
    assert!(schema.declare_target(ATOMS, X, ATOMS).is_err());
    assert!(
        schema
            .declare_target("refs", "site", "/trajectory/atoms")
            .is_err()
    );
    assert!(schema.declare_target("refs", "site", "a/b").is_err());
    schema
        .declare_target("refs", "site", "/system/atoms")
        .unwrap();
    assert!(schema.declare_target("refs", "site", ATOMS).is_err());
}
