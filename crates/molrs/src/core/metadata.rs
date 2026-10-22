//! Typed frame metadata: [`MetaMap`] of [`MetaValue`]s.

use indexmap::IndexMap;

/// Exact metadata value stored on a frame.
///
/// Every value carries an exact scalar or fixed-vector dtype; metadata is never
/// routed through a string representation. [`MetaMap`] is the only
/// frame-metadata container.
#[derive(Clone, Debug, PartialEq)]
pub enum MetaValue {
    Bool(bool),
    I32(i32),
    I64(i64),
    U32(u32),
    U64(u64),
    F64(f64),
    String(String),
    Bool3([bool; 3]),
    I32x3([i32; 3]),
    I64x3([i64; 3]),
    U32x3([u32; 3]),
    U64x3([u64; 3]),
    F64x3([f64; 3]),
    /// Symmetric stress tensor in `(xx, yy, zz, xy, xz, yz)` order.
    F64x6([f64; 6]),
    /// Row-major 3x3 tensor.
    F64x9([f64; 9]),
    /// Nested JSON document value. Frame group attributes are a document,
    /// not a typed scalar map: objects, arrays, and nulls land here.
    Json(serde_json::Value),
}

impl MetaValue {
    /// Stable dtype tag used by every serialized representation and binding.
    pub const fn dtype(&self) -> &'static str {
        match self {
            Self::Bool(_) => "bool",
            Self::I32(_) => "i32",
            Self::I64(_) => "i64",
            Self::U32(_) => "u32",
            Self::U64(_) => "u64",
            Self::F64(_) => "f64",
            Self::String(_) => "string",
            Self::Bool3(_) => "bool3",
            Self::I32x3(_) => "i32x3",
            Self::I64x3(_) => "i64x3",
            Self::U32x3(_) => "u32x3",
            Self::U64x3(_) => "u64x3",
            Self::F64x3(_) => "f64x3",
            Self::F64x6(_) => "f64x6",
            Self::F64x9(_) => "f64x9",
            Self::Json(_) => "json",
        }
    }

    pub const fn as_bool(&self) -> Option<bool> {
        if let Self::Bool(value) = self {
            Some(*value)
        } else {
            None
        }
    }
    pub const fn as_i32(&self) -> Option<i32> {
        if let Self::I32(value) = self {
            Some(*value)
        } else {
            None
        }
    }
    pub const fn as_i64(&self) -> Option<i64> {
        if let Self::I64(value) = self {
            Some(*value)
        } else {
            None
        }
    }
    pub const fn as_u32(&self) -> Option<u32> {
        if let Self::U32(value) = self {
            Some(*value)
        } else {
            None
        }
    }
    pub const fn as_u64(&self) -> Option<u64> {
        if let Self::U64(value) = self {
            Some(*value)
        } else {
            None
        }
    }
    pub const fn as_f64(&self) -> Option<f64> {
        if let Self::F64(value) = self {
            Some(*value)
        } else {
            None
        }
    }
    pub fn as_str(&self) -> Option<&str> {
        if let Self::String(value) = self {
            Some(value)
        } else {
            None
        }
    }

    /// The value in its typed JSON form: NaN and
    /// ±∞ as `"NaN"` / `"Infinity"` / `"-Infinity"`, integers beyond ±2⁵³
    /// as decimal strings, vectors as arrays of element forms, `json`
    /// verbatim. [`Self::dtype`] says how to read it back
    /// ([`Self::from_typed_json`]).
    ///
    /// This is the one encoder of a typed meta value: a frame group's
    /// attributes (beside `_meta_types`), a `sequence_schema` fill, and the
    /// payload of the `{dtype, value}` envelope all use it.
    pub fn to_typed_json(&self) -> serde_json::Value {
        use crate::core::typed_json::{encode_f64, encode_i64, encode_u64};
        use serde_json::Value;
        fn array<T: Copy>(values: &[T], encode: impl Fn(T) -> Value) -> Value {
            Value::Array(values.iter().map(|&v| encode(v)).collect())
        }
        match self {
            Self::Bool(v) => Value::Bool(*v),
            Self::I32(v) => Value::from(*v),
            Self::I64(v) => encode_i64(*v),
            Self::U32(v) => Value::from(*v),
            Self::U64(v) => encode_u64(*v),
            Self::F64(v) => encode_f64(*v),
            Self::String(v) => Value::String(v.clone()),
            Self::Bool3(v) => array(v, Value::Bool),
            Self::I32x3(v) => array(v, Value::from),
            Self::I64x3(v) => array(v, encode_i64),
            Self::U32x3(v) => array(v, Value::from),
            Self::U64x3(v) => array(v, encode_u64),
            Self::F64x3(v) => array(v, encode_f64),
            Self::F64x6(v) => array(v, encode_f64),
            Self::F64x9(v) => array(v, encode_f64),
            Self::Json(v) => v.clone(),
        }
    }

    /// Decode `value` as a value of tag `dtype`, exactly: the inverse of
    /// [`Self::to_typed_json`].
    ///
    /// Accepts the typed JSON forms and an exact JSON integer beyond 2⁵³;
    /// refuses every other form — `null` where a number is declared, a real
    /// where an integer is, an `i32` / `u32` out of range, a vector of the
    /// wrong length.
    ///
    /// # Errors
    ///
    /// A message naming the tag and what was found.
    pub fn from_typed_json(dtype: &str, value: &serde_json::Value) -> Result<Self, String> {
        use crate::core::typed_json::{
            decode_bool, decode_f64, decode_i64, decode_signed, decode_string, decode_u64,
            decode_unsigned,
        };
        fn array<T, const N: usize>(
            value: &serde_json::Value,
            dtype: &str,
            decode: impl Fn(&serde_json::Value) -> Result<T, String>,
        ) -> Result<[T; N], String> {
            let items = value
                .as_array()
                .ok_or_else(|| format!("expects an array of {N}, found {value}"))?;
            let decoded = items
                .iter()
                .map(decode)
                .collect::<Result<Vec<T>, String>>()?;
            decoded
                .try_into()
                .map_err(|v: Vec<T>| format!("expects {N} values, got {}", v.len()))
                .map_err(|e| format!("{dtype} {e}"))
        }
        let typed = match dtype {
            "bool" => decode_bool(value).map(Self::Bool),
            "i32" => decode_signed(value, "i32").map(Self::I32),
            "i64" => decode_i64(value).map(Self::I64),
            "u32" => decode_unsigned(value, "u32").map(Self::U32),
            "u64" => decode_u64(value).map(Self::U64),
            "f64" => decode_f64(value).map(Self::F64),
            "string" => decode_string(value).map(Self::String),
            "bool3" => array(value, dtype, decode_bool).map(Self::Bool3),
            "i32x3" => array(value, dtype, |v| decode_signed(v, "i32")).map(Self::I32x3),
            "i64x3" => array(value, dtype, decode_i64).map(Self::I64x3),
            "u32x3" => array(value, dtype, |v| decode_unsigned(v, "u32")).map(Self::U32x3),
            "u64x3" => array(value, dtype, decode_u64).map(Self::U64x3),
            "f64x3" => array(value, dtype, decode_f64).map(Self::F64x3),
            "f64x6" => array(value, dtype, decode_f64).map(Self::F64x6),
            "f64x9" => array(value, dtype, decode_f64).map(Self::F64x9),
            "json" => Ok(Self::Json(value.clone())),
            other => return Err(format!("unknown metadata dtype `{other}`")),
        };
        typed.map_err(|e| format!("metadata `{dtype}`: {e}"))
    }

    /// The typed `{dtype, value}` envelope (the stream wire form and the
    /// `serde` form), its payload in the typed JSON form.
    pub fn to_json_value(&self) -> serde_json::Value {
        serde_json::json!({ "dtype": self.dtype(), "value": self.to_typed_json() })
    }

    /// Decode an untagged document value by inference — the reading of a key
    /// a frame group's `_meta_types` does not type (one written by a tool that
    /// writes plain JSON).
    ///
    /// JSON `true`/`false` → `bool`; an integer in `[−2⁶³, 2⁶³)` → `i64`, in
    /// `[2⁶³, 2⁶⁴)` → `u64`; any other number → `f64`; a string → `string`
    /// (a `"NaN"` stays a string: only a tag makes it a float); an array,
    /// object or `null` → `json`.
    ///
    /// The typed `{dtype, value}` envelope is [`Self::from_json_value`]'s
    /// alone: a document value shaped like it is ordinary user JSON.
    pub fn from_attr_value(value: &serde_json::Value) -> Self {
        match value {
            serde_json::Value::Bool(v) => Self::Bool(*v),
            serde_json::Value::Number(n) if n.is_i64() => Self::I64(n.as_i64().unwrap()),
            serde_json::Value::Number(n) if n.is_u64() => Self::U64(n.as_u64().unwrap()),
            serde_json::Value::Number(n) => Self::F64(n.as_f64().unwrap_or(f64::NAN)),
            serde_json::Value::String(v) => Self::String(v.clone()),
            other => Self::Json(other.clone()),
        }
    }

    /// Decode the exact JSON object emitted by [`Self::to_json_value`].
    ///
    /// # Errors
    ///
    /// A message when `value` is not exactly `{dtype, value}`, or the payload
    /// is not its tag's typed JSON form ([`Self::from_typed_json`]).
    pub fn from_json_value(value: &serde_json::Value) -> Result<Self, String> {
        let object = value
            .as_object()
            .ok_or_else(|| "typed metadata must be an object".to_string())?;
        if object.len() != 2 || !object.contains_key("dtype") || !object.contains_key("value") {
            return Err("typed metadata object must contain exactly `dtype` and `value`".into());
        }
        let dtype = object["dtype"]
            .as_str()
            .ok_or_else(|| "metadata dtype must be a string".to_string())?;
        Self::from_typed_json(dtype, &object["value"])
    }
}

macro_rules! impl_from_meta {
    ($ty:ty, $variant:ident) => {
        impl From<$ty> for MetaValue {
            fn from(value: $ty) -> Self {
                Self::$variant(value)
            }
        }
    };
}
impl_from_meta!(bool, Bool);
impl_from_meta!(i32, I32);
impl_from_meta!(i64, I64);
impl_from_meta!(u32, U32);
impl_from_meta!(u64, U64);
impl_from_meta!(f64, F64);
impl_from_meta!(String, String);
impl_from_meta!([bool; 3], Bool3);
impl_from_meta!([i32; 3], I32x3);
impl_from_meta!([i64; 3], I64x3);
impl_from_meta!([u32; 3], U32x3);
impl_from_meta!([u64; 3], U64x3);
impl_from_meta!([f64; 3], F64x3);
impl_from_meta!([f64; 6], F64x6);
impl_from_meta!([f64; 9], F64x9);

impl From<&str> for MetaValue {
    fn from(value: &str) -> Self {
        Self::String(value.to_owned())
    }
}

/// The unique metadata map used by owned and borrowed frames.
///
/// Iteration is insertion-ordered. [`Self::remove`] drops one key and keeps
/// the remaining keys in their original relative order. Inserting a key that
/// is already present updates its value and leaves it where it was.
///
/// # Examples
///
/// ```
/// use molrs::core::MetaMap;
///
/// let mut meta = MetaMap::new();
/// meta.insert("z", "Z");
/// meta.insert("a", "A");
/// meta.insert("m", "M");
/// meta.remove("a");
/// assert_eq!(
///     meta.keys().map(String::as_str).collect::<Vec<_>>(),
///     ["z", "m"]
/// );
/// ```
#[derive(Clone, Debug, Default, PartialEq)]
pub struct MetaMap(IndexMap<String, MetaValue>);

/// Borrowed iterator over a [`MetaMap`], in insertion order.
pub struct MetaIter<'a>(indexmap::map::Iter<'a, String, MetaValue>);

impl<'a> Iterator for MetaIter<'a> {
    type Item = (&'a String, &'a MetaValue);

    fn next(&mut self) -> Option<Self::Item> {
        self.0.next()
    }

    fn size_hint(&self) -> (usize, Option<usize>) {
        self.0.size_hint()
    }
}

impl ExactSizeIterator for MetaIter<'_> {
    fn len(&self) -> usize {
        self.0.len()
    }
}

impl DoubleEndedIterator for MetaIter<'_> {
    fn next_back(&mut self) -> Option<Self::Item> {
        self.0.next_back()
    }
}

impl MetaMap {
    pub fn new() -> Self {
        Self::default()
    }
    pub fn with_capacity(capacity: usize) -> Self {
        Self(IndexMap::with_capacity(capacity))
    }
    pub fn len(&self) -> usize {
        self.0.len()
    }
    pub fn is_empty(&self) -> bool {
        self.0.is_empty()
    }
    pub fn clear(&mut self) {
        self.0.clear();
    }
    pub fn contains_key(&self, key: &str) -> bool {
        self.0.contains_key(key)
    }
    pub fn get(&self, key: &str) -> Option<&MetaValue> {
        self.0.get(key)
    }
    pub fn get_mut(&mut self, key: &str) -> Option<&mut MetaValue> {
        self.0.get_mut(key)
    }
    pub fn insert(
        &mut self,
        key: impl Into<String>,
        value: impl Into<MetaValue>,
    ) -> Option<MetaValue> {
        self.0.insert(key.into(), value.into())
    }
    /// Removes `key`, shifting later keys down so their relative order holds.
    pub fn remove(&mut self, key: &str) -> Option<MetaValue> {
        self.0.shift_remove(key)
    }
    pub fn iter(&self) -> MetaIter<'_> {
        MetaIter(self.0.iter())
    }
    pub fn keys(&self) -> impl Iterator<Item = &String> {
        self.0.keys()
    }
    pub fn values(&self) -> impl Iterator<Item = &MetaValue> {
        self.0.values()
    }
    pub fn extend(&mut self, values: impl IntoIterator<Item = (String, MetaValue)>) {
        self.0.extend(values);
    }
}

impl<'a> IntoIterator for &'a MetaMap {
    type Item = (&'a String, &'a MetaValue);
    type IntoIter = MetaIter<'a>;
    fn into_iter(self) -> Self::IntoIter {
        self.iter()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn json_roundtrip_preserves_every_dtype() {
        let values = [
            MetaValue::Bool(true),
            MetaValue::I32(-3),
            MetaValue::I64(i64::MIN + 7),
            MetaValue::U32(9),
            MetaValue::U64(u64::MAX - 7),
            MetaValue::F64(-2.5),
            MetaValue::String("x".into()),
            MetaValue::F64x3([1.0, 2.0, 3.0]),
            MetaValue::F64x6([1.0, 2.0, 3.0, 4.0, 5.0, 6.0]),
            MetaValue::F64x9([1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0]),
        ];
        for value in values {
            assert_eq!(
                MetaValue::from_json_value(&value.to_json_value()).unwrap(),
                value
            );
        }
    }

    #[test]
    fn typed_json_round_trips_every_tag_and_the_edges() {
        let values = [
            MetaValue::Bool(true),
            MetaValue::I32(i32::MIN),
            MetaValue::I64(i64::MIN),
            MetaValue::U32(u32::MAX),
            MetaValue::U64(u64::MAX),
            MetaValue::U64((1 << 53) + 1),
            MetaValue::F64(f64::INFINITY),
            MetaValue::F64(f64::NEG_INFINITY),
            MetaValue::F64(1.0),
            MetaValue::String("NaN".into()),
            MetaValue::Bool3([true, false, true]),
            MetaValue::I32x3([-1, 0, 1]),
            MetaValue::I64x3([i64::MIN, 0, i64::MAX]),
            MetaValue::U32x3([0, 1, u32::MAX]),
            MetaValue::U64x3([0, 1, u64::MAX]),
            MetaValue::F64x3([1.0, f64::INFINITY, -0.5]),
            MetaValue::F64x6([1.0; 6]),
            MetaValue::F64x9([2.0; 9]),
            MetaValue::Json(serde_json::json!({"a": [1, null]})),
        ];
        for value in values {
            let back = MetaValue::from_typed_json(value.dtype(), &value.to_typed_json()).unwrap();
            assert_eq!(back, value);
        }
        let nan = MetaValue::from_typed_json("f64", &MetaValue::F64(f64::NAN).to_typed_json());
        assert!(nan.unwrap().as_f64().unwrap().is_nan());
        assert_eq!(
            MetaValue::F64(f64::NAN).to_typed_json(),
            serde_json::json!("NaN")
        );
        assert_eq!(
            MetaValue::U64(u64::MAX).to_typed_json(),
            serde_json::json!("18446744073709551615")
        );
    }

    #[test]
    fn typed_json_refuses_other_forms() {
        for (tag, raw) in [
            ("i32", serde_json::json!(1.5)),
            ("i32", serde_json::json!(1_i64 << 40)),
            ("u64", serde_json::json!(-1)),
            ("f64", serde_json::Value::Null),
            ("f64", serde_json::json!("nan")),
            ("bool", serde_json::json!(1)),
            ("f64x3", serde_json::json!([1.0, 2.0])),
            ("i64x3", serde_json::json!([1, 2, 3.5])),
            ("f128", serde_json::json!(1.0)),
        ] {
            assert!(
                MetaValue::from_typed_json(tag, &raw).is_err(),
                "{tag} {raw}"
            );
        }
    }

    #[test]
    fn the_envelope_carries_nan_and_wide_integers() {
        let nan = MetaValue::from_json_value(&MetaValue::F64(f64::NAN).to_json_value()).unwrap();
        assert!(nan.as_f64().unwrap().is_nan());
        let wide = MetaValue::from_json_value(&serde_json::json!({
            "dtype": "u64", "value": "18446744073709551615"
        }))
        .unwrap();
        assert_eq!(wide, MetaValue::U64(u64::MAX));
    }

    #[test]
    fn untyped_json_is_rejected() {
        assert!(MetaValue::from_json_value(&serde_json::json!("plain")).is_err());
    }

    #[test]
    fn envelope_shaped_user_json_survives_verbatim() {
        // A document key whose value happens to look like the MessagePack
        // envelope is still ordinary user JSON on a Zarr attribute.
        let raw = serde_json::json!({"dtype": "f64", "value": 1.5});
        assert_eq!(
            MetaValue::from_attr_value(&raw),
            MetaValue::Json(serde_json::json!({"dtype": "f64", "value": 1.5}))
        );
    }

    #[test]
    fn unknown_dtype_object_survives_verbatim() {
        let raw = serde_json::json!({"dtype": "not-a-dtype", "value": 1});
        assert_eq!(
            MetaValue::from_attr_value(&raw),
            MetaValue::Json(serde_json::json!({"dtype": "not-a-dtype", "value": 1}))
        );
    }

    #[test]
    fn serde_path_still_decodes_the_typed_envelope() {
        // The path split: `from_json_value` owns the envelope, so
        // `from_attr_value` does not have to speculate about one.
        assert_eq!(
            MetaValue::from_json_value(&serde_json::json!({"dtype": "f64", "value": 1.5})).unwrap(),
            MetaValue::F64(1.5)
        );
    }

    #[test]
    fn narrow_float_tags_are_refused() {
        // The narrow vector tags fall through this same unknown-dtype arm.
        for tag in ["f16", "f32"] {
            let raw = serde_json::json!({"dtype": tag, "value": 1.5});
            assert!(
                MetaValue::from_json_value(&raw).is_err(),
                "accepted `{tag}`"
            );
        }
    }

    #[test]
    fn keys_iter_and_values_follow_insertion_order() {
        let mut meta = MetaMap::new();
        meta.insert("z", "Z");
        meta.insert("a", "A");
        meta.insert("m", "M");

        assert_eq!(
            meta.keys().map(String::as_str).collect::<Vec<_>>(),
            vec!["z", "a", "m"]
        );
        assert_eq!(
            meta.iter().map(|(k, _)| k.as_str()).collect::<Vec<_>>(),
            vec!["z", "a", "m"]
        );
        assert_eq!(
            meta.values()
                .map(|v| v.as_str().unwrap())
                .collect::<Vec<_>>(),
            vec!["Z", "A", "M"]
        );
    }

    #[test]
    fn remove_of_the_second_key_keeps_the_order_of_the_survivors() {
        let mut meta = MetaMap::new();
        meta.insert("z", "Z");
        meta.insert("a", "A");
        meta.insert("m", "M");
        meta.insert("q", "Q");

        let removed = meta.remove("a");
        assert_eq!(removed.as_ref().and_then(MetaValue::as_str), Some("A"));
        // Four keys, second removed: shift_remove leaves [z, m, q].
        // swap_remove would park the last key in the hole: [z, q, m].
        assert_eq!(
            meta.keys().map(String::as_str).collect::<Vec<_>>(),
            vec!["z", "m", "q"]
        );
    }

    #[test]
    fn reinserting_a_key_keeps_its_position_and_updates_the_value() {
        let mut meta = MetaMap::new();
        meta.insert("z", "Z");
        meta.insert("a", "A");
        meta.insert("m", "M");

        let replaced = meta.insert("a", "A2");
        assert_eq!(replaced.as_ref().and_then(MetaValue::as_str), Some("A"));
        assert_eq!(meta.get("a").and_then(MetaValue::as_str), Some("A2"));
        assert_eq!(
            meta.keys().map(String::as_str).collect::<Vec<_>>(),
            vec!["z", "a", "m"]
        );
    }

    #[test]
    fn clear_then_insert_restarts_insertion_order() {
        let mut meta = MetaMap::new();
        meta.insert("z", "Z");
        meta.insert("a", "A");
        meta.insert("m", "M");
        meta.clear();
        meta.insert("m", "M");
        meta.insert("z", "Z");
        meta.insert("a", "A");
        meta.insert("q", "Q");

        assert_eq!(
            meta.keys().map(String::as_str).collect::<Vec<_>>(),
            vec!["m", "z", "a", "q"]
        );
    }
}
