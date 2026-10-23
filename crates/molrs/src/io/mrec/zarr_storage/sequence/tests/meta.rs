use super::*;

// =======================================================================
// C. Per-step meta — typed, exact, never implicitly filled
// =======================================================================

/// One value of every [`MetaValue`] variant, at range extremes where the
/// variant has any: a per-step meta array that silently narrowed a width
/// would come back changed.
fn meta_variants() -> Vec<MetaValue> {
    vec![
        MetaValue::Bool(true),
        MetaValue::I32(i32::MIN),
        MetaValue::I64(i64::MIN),
        MetaValue::U32(u32::MAX),
        MetaValue::U64(u64::MAX),
        MetaValue::F64(f64::MIN_POSITIVE),
        MetaValue::String("gamma-phase".to_string()),
        MetaValue::Bool3([true, false, true]),
        MetaValue::I32x3([i32::MIN, 0, i32::MAX]),
        MetaValue::I64x3([i64::MIN, 0, i64::MAX]),
        MetaValue::U32x3([0, 1, u32::MAX]),
        MetaValue::U64x3([0, 1, u64::MAX]),
        MetaValue::F64x3([1.0, -0.5, f64::MAX]),
        MetaValue::F64x6([1.0, 2.0, 3.0, -4.0, 5.5, 6.25]),
        MetaValue::F64x9([1.0, 2.0, 3.0, -4.0, 5.5, 6.25, 7.0, -8.0, 9.5]),
        MetaValue::Json(serde_json::json!({"basis": "def2-TZVP", "scf": [1, 2, 3]})),
    ]
}

/// Every `MetaValue` variant round-trips bit exact through
/// `trajectory/meta/<key>`, and the array carries its `dtype()` tag in
/// `meta_dtype` (ac-019).
///
/// The dtype travels with the array rather than with a convention, which
/// is what makes exactness free (decision 8). Written for two steps so the
/// array is a real per-step array and not a scalar attribute.
#[test]
fn column_dtype_tags_are_molrec_names() {
    use super::dtype_from_tag;
    // The three domain aliases are written under molrec's concrete-width
    // spelling so a schema validates against molrec's dtype enum.
    assert_eq!(DType::Float.name(), "f64");
    assert_eq!(DType::Int.name(), "i32");
    assert_eq!(DType::Uint.name(), "u64");
    assert_eq!(dtype_from_tag("f64").unwrap(), DType::Float);
    assert_eq!(dtype_from_tag("i32").unwrap(), DType::Int);
    assert_eq!(dtype_from_tag("u64").unwrap(), DType::Uint);
    // The domain alias names are not tags.
    for alias in ["float", "int", "uint"] {
        assert!(dtype_from_tag(alias).is_err(), "{alias}");
    }
}

#[test]
fn every_meta_variant_round_trips_bit_exact_with_its_dtype_tag() {
    const KEY: &str = "probe";

    for value in meta_variants() {
        let dir = TempDir::new().unwrap();
        let store = storage_in(&dir);
        let frames: Vec<Frame> = [1.0f64, 2.0]
            .iter()
            .map(|v| {
                let mut frame = atoms_frame(&[*v]);
                frame.meta.insert(KEY, value.clone());
                frame
            })
            .collect();
        write_all(&store, &frames);

        let array = Array::open(store.clone(), &format!("{TRAJ}/meta/{KEY}"))
            .expect("a declared meta key is a per-step array");
        assert_eq!(
            array.attributes().get("meta_dtype"),
            Some(&serde_json::json!(value.dtype())),
            "the array must carry the {} tag",
            value.dtype()
        );

        let mut seq = open_sequence(&store);
        for step in 0..2u64 {
            assert_eq!(
                frame_at(&mut seq, step).meta.get(KEY),
                Some(&value),
                "{} must survive step {step} bit exact",
                value.dtype()
            );
        }
    }
}

/// Omitting a declared meta key that has no declared fill is an `Err`
/// naming the key — there is no implicit NaN (ac-019).
#[test]
fn omitting_a_declared_meta_key_without_a_fill_errs_naming_it() {
    const KEY: &str = "temperature";

    let dir = TempDir::new().unwrap();
    let store = storage_in(&dir);
    let mut declaring = atoms_frame(&[1.0]);
    declaring.meta.insert(KEY, MetaValue::F64(300.0));
    let schema = SequenceSchema::from_frames(std::slice::from_ref(&declaring)).unwrap();
    let mut writer = MrecWriter::create_in_storage(store, schema).unwrap();
    writer.append(&declaring).unwrap();

    let message = writer
        .append(&atoms_frame(&[2.0]))
        .expect_err("a declared meta key with no fill may not be silently filled")
        .to_string();
    assert!(message.contains(KEY), "must name the key: {message}");
}

/// A meta key declared with an explicit fill writes that fill for the
/// steps that omit it, and leaves the steps that carry a value alone
/// (ac-019).
#[test]
fn a_declared_fill_is_written_for_an_omitted_meta_key() {
    const KEY: &str = "temperature";
    const FILL: f64 = -1.0;

    let dir = TempDir::new().unwrap();
    let store = storage_in(&dir);
    let mut declaring = atoms_frame(&[1.0]);
    declaring.meta.insert(KEY, MetaValue::F64(300.0));
    let mut schema = SequenceSchema::from_frames(std::slice::from_ref(&declaring)).unwrap();
    schema
        .declare_meta(KEY, "f64", Some(MetaValue::F64(FILL)))
        .unwrap();

    let mut writer = MrecWriter::create_in_storage(store.clone(), schema).unwrap();
    writer.append(&declaring).unwrap();
    writer.append(&atoms_frame(&[2.0])).unwrap();
    writer.close().unwrap();

    let mut seq = open_sequence(&store);
    assert_eq!(
        frame_at(&mut seq, 1).meta.get(KEY),
        Some(&MetaValue::F64(FILL)),
        "the omitting step reads the declared fill, not NaN"
    );
    assert_eq!(
        frame_at(&mut seq, 0).meta.get(KEY),
        Some(&MetaValue::F64(300.0)),
        "and the step that carried a value keeps it"
    );
}

/// A NaN fill is pinned as `"NaN"` and lands as NaN; a pin whose fill a
/// writer lost to `null` is refused, not read as a run without a fill.
#[test]
fn a_nan_fill_is_storable_and_a_null_fill_is_refused() {
    const KEY: &str = "temperature";
    let dir = TempDir::new().unwrap();
    let store = storage_in(&dir);
    let mut schema =
        SequenceSchema::from_frames(std::slice::from_ref(&atoms_frame(&[1.0]))).unwrap();
    schema
        .declare_meta(KEY, "f64", Some(MetaValue::F64(f64::NAN)))
        .unwrap();
    schema
        .declare_meta("count", "u64", Some(MetaValue::U64(u64::MAX)))
        .unwrap();
    let mut writer = MrecWriter::create_in_storage(store.clone(), schema).unwrap();
    writer.append(&atoms_frame(&[1.0])).unwrap();
    writer.close().unwrap();

    let pin = Group::open(store.clone(), TRAJ).unwrap().attributes()[SCHEMA_ATTRIBUTE].clone();
    assert_eq!(pin["meta"][KEY]["fill"], serde_json::json!("NaN"));
    assert_eq!(
        pin["meta"]["count"]["fill"],
        serde_json::json!("18446744073709551615")
    );
    let mut seq = open_sequence(&store);
    let frame = frame_at(&mut seq, 0);
    assert!(frame.meta.get(KEY).unwrap().as_f64().unwrap().is_nan());
    assert_eq!(frame.meta.get("count"), Some(&MetaValue::U64(u64::MAX)));

    // Tamper: a `null` fill.
    let mut group = Group::open(store.clone(), TRAJ).unwrap();
    group.attributes_mut()[SCHEMA_ATTRIBUTE]["meta"][KEY]["fill"] = serde_json::Value::Null;
    group.store_metadata().unwrap();
    let err = MrecReader::from_storage(store.clone())
        .err()
        .unwrap()
        .to_string();
    assert!(err.contains(KEY), "{err}");
    assert!(MrecWriter::from_storage(store).is_err());
}

/// Declared meta keys read back in `declare_meta` order, not the order the
/// frame inserted them and not alphabetical order.
#[test]
fn declared_meta_keys_read_back_in_declaration_order() {
    let dir = TempDir::new().unwrap();
    let store = storage_in(&dir);
    let mut schema = SequenceSchema::new();
    schema.declare_column(ATOMS, X, DType::Float, &[]).unwrap();
    schema.declare_meta("zeta", "f64", None).unwrap();
    schema.declare_meta("alpha", "f64", None).unwrap();
    schema.declare_meta("mu", "f64", None).unwrap();

    let mut frame = atoms_frame(&[1.0]);
    frame.meta.insert("mu", MetaValue::F64(3.0));
    frame.meta.insert("zeta", MetaValue::F64(1.0));
    frame.meta.insert("alpha", MetaValue::F64(2.0));

    let mut writer = MrecWriter::create_in_storage(store.clone(), schema).unwrap();
    writer.append(&frame).unwrap();
    writer.close().unwrap();

    let seq = MrecReader::from_storage(store).unwrap();
    let back = seq.frame(0).unwrap().expect("frame 0 is present");
    assert_eq!(
        back.meta.keys().map(String::as_str).collect::<Vec<_>>(),
        ["zeta", "alpha", "mu"]
    );
    assert_eq!(back.meta.get("zeta"), Some(&MetaValue::F64(1.0)));
    assert_eq!(back.meta.get("alpha"), Some(&MetaValue::F64(2.0)));
    assert_eq!(back.meta.get("mu"), Some(&MetaValue::F64(3.0)));
}
