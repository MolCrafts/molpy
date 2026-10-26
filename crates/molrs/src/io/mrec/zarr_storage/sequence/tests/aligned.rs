use super::*;

// -- aligned blocks (molrec F5) -----------------------------------------

const TYPES: &str = "atom_types";

fn string_column(values: &[&str]) -> Column {
    Column::from(
        ArrayD::from_shape_vec(
            vec![values.len()],
            values.iter().map(|s| (*s).to_string()).collect(),
        )
        .unwrap(),
    )
}

/// `atoms` (x) and, when `types` is given, the aligned `atom_types`.
fn aligned_frame(x: &[f64], types: Option<&[&str]>) -> Frame {
    let mut frame = atoms_frame(x);
    if let Some(types) = types {
        frame.insert(TYPES, block_with("type", string_column(types)));
    }
    frame
}

fn aligned_schema() -> SequenceSchema {
    let mut schema = SequenceSchema::new();
    schema.declare_column(ATOMS, X, DType::Float, &[]).unwrap();
    schema
        .declare_column(TYPES, "type", DType::String, &[])
        .unwrap();
    schema.declare_aligned(TYPES, ATOMS).unwrap();
    schema
}

fn write_aligned(frames: &[Frame]) -> (TempDir, ReadableWritableListableStorage) {
    let dir = TempDir::new().unwrap();
    let store = storage_in(&dir);
    let mut writer = MrecWriter::create_in_storage(store.clone(), aligned_schema()).unwrap();
    for frame in frames {
        writer.append(frame).unwrap();
    }
    writer.close().unwrap();
    (dir, store)
}

fn types_at(seq: &mut MrecReader, index: u64) -> Option<Vec<String>> {
    frame_at(seq, index).get(TYPES).map(|b| {
        b.get("type")
            .and_then(|c| c.get::<String>())
            .unwrap()
            .iter()
            .cloned()
            .collect()
    })
}

#[test]
fn an_aligned_block_carries_forward_while_its_target_moves() {
    let frames = [
        aligned_frame(&[0.0, 1.0, 2.0], Some(&["A", "B", "C"])),
        aligned_frame(&[0.1, 1.1, 2.1], None),
        aligned_frame(&[0.2, 1.2, 2.2], None),
        aligned_frame(&[0.3, 1.3, 2.3], Some(&["A", "A", "C"])),
    ];
    let (_dir, store) = write_aligned(&frames);
    let pin = Group::open(store.clone(), TRAJ).unwrap().attributes()[SCHEMA_ATTRIBUTE].clone();
    assert_eq!(
        pin["blocks"][TYPES]["aligned_with"],
        serde_json::json!(ATOMS)
    );
    let mut seq = open_sequence(&store);
    assert_eq!(types_at(&mut seq, 2).unwrap(), ["A", "B", "C"]);
    assert_eq!(types_at(&mut seq, 3).unwrap(), ["A", "A", "C"]);
    assert_eq!(seq.block_update_at(TYPES, 2).unwrap(), Some(0));
    assert_eq!(seq.schema().aligned_with(TYPES), Some(ATOMS));
}

#[test]
fn an_aligned_block_is_restated_on_growth_and_may_appear_late_or_empty() {
    let frames = [
        aligned_frame(&[0.0, 1.0, 2.0], None),
        aligned_frame(&[0.0, 1.0, 2.0], Some(&["A", "B", "C"])),
        aligned_frame(&[0.0, 1.0, 2.0, 3.0], Some(&["A", "B", "C", "D"])),
        aligned_frame(&[], Some(&[])),
    ];
    let (_dir, store) = write_aligned(&frames);
    let mut seq = open_sequence(&store);
    assert_eq!(types_at(&mut seq, 0), None);
    assert_eq!(types_at(&mut seq, 2).unwrap().len(), 4);
    assert_eq!(types_at(&mut seq, 3).unwrap().len(), 0);
}

#[test]
fn a_writer_refuses_a_frame_that_breaks_the_alignment() {
    let dir = TempDir::new().unwrap();
    let store = storage_in(&dir);
    let mut writer = MrecWriter::create_in_storage(store.clone(), aligned_schema()).unwrap();
    writer
        .append(&aligned_frame(&[0.0, 1.0, 2.0], Some(&["A", "B", "C"])))
        .unwrap();
    // `atoms` grows and `atom_types` is not restated.
    let err = writer
        .append(&aligned_frame(&[0.0, 1.0, 2.0, 3.0], None))
        .unwrap_err()
        .to_string();
    assert!(
        err.contains(TYPES) && err.contains("trajectory frame 1"),
        "{err}"
    );
    // Restated with another count: refused too.
    assert!(
        writer
            .append(&aligned_frame(&[0.0, 1.0, 2.0, 3.0], Some(&["A"])))
            .is_err()
    );
    // Nothing moved: a good frame still lands at ordinal 1.
    writer
        .append(&aligned_frame(
            &[0.0, 1.0, 2.0, 3.0],
            Some(&["A", "B", "C", "D"]),
        ))
        .unwrap();
    writer.close().unwrap();
    assert_eq!(committed_len(&mut open_sequence(&store)), 2);

    // The aligned block presented before its target exists.
    let dir = TempDir::new().unwrap();
    let store = storage_in(&dir);
    let mut writer = MrecWriter::create_in_storage(store, aligned_schema()).unwrap();
    let mut types_only = Frame::new();
    types_only.insert(TYPES, block_with("type", string_column(&["A"])));
    assert!(writer.append(&types_only).is_err());
}

#[test]
fn a_store_whose_aligned_counts_disagree_is_refused() {
    let frames = [
        aligned_frame(&[0.0, 1.0, 2.0], Some(&["A", "B", "C"])),
        aligned_frame(&[0.0, 1.0], Some(&["A", "B"])),
    ];
    let (_dir, store) = write_aligned(&frames);
    assert!(MrecReader::from_storage(store.clone()).is_ok());
    // Tamper: `atom_types/offset` so update 1 holds 1 row instead of 2.
    let offsets = u64_array(&store, &format!("{TRAJ}/{TYPES}/offset"));
    assert_eq!(offsets, [0, 3, 5]);
    let array = Array::open(store.clone(), &format!("{TRAJ}/{TYPES}/offset")).unwrap();
    array
        .store_array_subset(&ArraySubset::new_with_shape(vec![3]), &[0_u64, 3, 4][..])
        .unwrap();
    array.store_metadata().unwrap();
    let err = MrecReader::from_storage(store).err().unwrap().to_string();
    assert!(err.contains(TYPES), "{err}");
}

#[test]
fn declare_aligned_refuses_what_cannot_align() {
    let mut schema = aligned_schema();
    schema
        .declare_column("other", "flag", DType::Bool, &[])
        .unwrap();
    schema
        .declare_column("cells", "v", DType::Float, &[])
        .unwrap();
    // Undeclared target, self, chain either way.
    assert!(schema.declare_aligned("other", "nope").is_err());
    assert!(schema.declare_aligned("other", "other").is_err());
    assert!(schema.declare_aligned("other", TYPES).is_err());
    assert!(schema.declare_aligned(ATOMS, "other").is_err());
    // Shared column, either at declaration or added afterwards.
    schema.declare_column("more", X, DType::Float, &[]).unwrap();
    assert!(schema.declare_aligned("more", ATOMS).is_err());
    assert!(schema.declare_column(TYPES, X, DType::Float, &[]).is_err());
    assert!(
        schema
            .declare_column(ATOMS, "type", DType::String, &[])
            .is_err()
    );
    // A shaped block cannot align, nor an aligned one take a shape.
    schema.declare_structural_shape("cells", &[1]).unwrap();
    assert!(schema.declare_aligned("cells", ATOMS).is_err());
    assert!(schema.declare_structural_shape(TYPES, &[3]).is_err());
}

#[test]
fn an_overflowing_structural_shape_does_not_change_the_schema() {
    let mut schema = SequenceSchema::new();
    schema.declare_block("cells", Some(6)).unwrap();
    let before = schema.clone();
    assert!(
        schema
            .declare_structural_shape("cells", &[usize::MAX, 2])
            .is_err()
    );
    assert_eq!(schema, before);
    assert_eq!(schema.rows_hint, before.rows_hint);
}

#[test]
fn trajectory_preflight_checks_carry_forward_references() {
    let mut members = Block::new();
    members
        .insert("atom", ndarray::array![1_u64].into_dyn(), None)
        .unwrap();
    members.set_target("atom", "atoms").unwrap();
    let mut atoms = Block::new();
    atoms
        .insert("x", ndarray::array![0.0, 1.0].into_dyn(), None)
        .unwrap();
    let mut first = Frame::new();
    first.insert("atoms", atoms);
    first.insert("members", members);
    let mut second = Frame::new();
    second.insert("atoms", first["atoms"].select_rows(&[0]).unwrap());
    let frames = [first, second];
    let schema = SequenceSchema::from_frames(&frames).unwrap();
    assert!(schema.validate_frames(&frames).is_err());
}

#[test]
fn an_aligned_block_may_not_share_a_name_with_a_system_block() {
    let dir = TempDir::new().unwrap();
    let store = storage_in(&dir);
    for group in ["/", "/system", "/system/atom_types"] {
        GroupBuilder::new()
            .build(store.clone(), group)
            .unwrap()
            .store_metadata()
            .unwrap();
    }
    let err = MrecWriter::create_in_storage(store, aligned_schema())
        .err()
        .unwrap()
        .to_string();
    assert!(err.contains("system"), "{err}");
}
