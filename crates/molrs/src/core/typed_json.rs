//! Typed JSON values — the one encoding of a value whose dtype is known.
//!
//! JSON has no NaN, no infinity and no complex number, and a reader whose
//! numbers are binary64 (JavaScript, a wasm viewer) silently rounds an
//! integer beyond 2⁵³. Wherever a JSON value has a declared dtype — a
//! per-step `fill` in a `sequence_schema`, a value of a frame's typed `meta`
//! (`_meta_types`) — molrec fixes one form per element dtype
//! (`conventions.md`, "Typed JSON values"):
//!
//! | element | JSON form |
//! |---------|-----------|
//! | `f64` | a number when finite; `"NaN"`, `"Infinity"`, `"-Infinity"` otherwise |
//! | `c64`, `c128` | `[re, im]`, each part an `f64` as above |
//! | an integer | a number when `|v| ≤ 2⁵³`; its decimal string beyond |
//! | `bool`, `string` | itself |
//!
//! A reader accepts exactly these forms, plus an exact JSON integer beyond
//! 2⁵³ (what a writer with 64-bit integers may emit), and refuses everything
//! else: `null` where a number is declared is a broken value, not a NaN, and
//! a real where an integer is declared is not rounded.

use serde_json::Value;

/// The largest integer magnitude every JSON reader holds exactly: 2⁵³.
pub const SAFE_INTEGER: u64 = 1 << 53;

/// The JSON form of an `f64`.
pub fn encode_f64(value: f64) -> Value {
    if value.is_nan() {
        Value::String("NaN".into())
    } else if value == f64::INFINITY {
        Value::String("Infinity".into())
    } else if value == f64::NEG_INFINITY {
        Value::String("-Infinity".into())
    } else {
        serde_json::Number::from_f64(value)
            .map(Value::Number)
            .expect("a finite f64 is a JSON number")
    }
}

/// Decode an `f64`: a JSON number, or one of the three non-finite strings.
///
/// # Errors
///
/// A message naming the value for any other form (`null` included).
pub fn decode_f64(value: &Value) -> Result<f64, String> {
    match value {
        Value::Number(n) => n
            .as_f64()
            .ok_or_else(|| format!("{n} is not representable as f64")),
        Value::String(s) => match s.as_str() {
            "NaN" => Ok(f64::NAN),
            "Infinity" => Ok(f64::INFINITY),
            "-Infinity" => Ok(f64::NEG_INFINITY),
            other => Err(format!(
                "f64 is a JSON number or \"NaN\"/\"Infinity\"/\"-Infinity\", found {other:?}"
            )),
        },
        other => Err(format!("f64 is a JSON number, found {other}")),
    }
}

/// The JSON form of a signed integer.
pub fn encode_i64(value: i64) -> Value {
    if value.unsigned_abs() > SAFE_INTEGER {
        Value::String(value.to_string())
    } else {
        Value::from(value)
    }
}

/// The JSON form of an unsigned integer.
pub fn encode_u64(value: u64) -> Value {
    if value > SAFE_INTEGER {
        Value::String(value.to_string())
    } else {
        Value::from(value)
    }
}

/// Whether `text` is a plain decimal integer: an optional `-`, then digits.
fn is_decimal(text: &str) -> bool {
    let digits = text.strip_prefix('-').unwrap_or(text);
    !digits.is_empty() && digits.bytes().all(|b| b.is_ascii_digit())
}

/// Decode a signed integer: a JSON integer or its decimal string.
///
/// # Errors
///
/// A message for a real, a bool, `null`, any other string, or a value
/// outside `i64`.
pub fn decode_i64(value: &Value) -> Result<i64, String> {
    match value {
        Value::Number(n) if n.is_i64() => Ok(n.as_i64().expect("checked")),
        Value::Number(n) if n.is_u64() => Err(format!("{n} is out of range for i64")),
        Value::String(s) if is_decimal(s) => s
            .parse::<i64>()
            .map_err(|_| format!("{s} is out of range for i64")),
        other => Err(format!("expected an integer, found {other}")),
    }
}

/// Decode an unsigned integer: a JSON integer or its decimal string.
///
/// # Errors
///
/// A message for a real, a bool, `null`, a negative, any other string, or a
/// value beyond `u64`.
pub fn decode_u64(value: &Value) -> Result<u64, String> {
    match value {
        Value::Number(n) if n.is_u64() => Ok(n.as_u64().expect("checked")),
        Value::Number(n) if n.is_i64() => Err(format!("{n} is out of range for u64")),
        Value::String(s) if is_decimal(s) => s
            .parse::<u64>()
            .map_err(|_| format!("{s} is out of range for u64")),
        other => Err(format!("expected an integer, found {other}")),
    }
}

/// Decode a signed integer held to `T`'s range (`i32`, …).
///
/// # Errors
///
/// [`decode_i64`]'s, plus a value outside `T`.
pub fn decode_signed<T: TryFrom<i64>>(value: &Value, dtype: &str) -> Result<T, String> {
    let wide = decode_i64(value)?;
    T::try_from(wide).map_err(|_| format!("{wide} is out of range for {dtype}"))
}

/// Decode an unsigned integer held to `T`'s range (`u32`, …).
///
/// # Errors
///
/// [`decode_u64`]'s, plus a value outside `T`.
pub fn decode_unsigned<T: TryFrom<u64>>(value: &Value, dtype: &str) -> Result<T, String> {
    let wide = decode_u64(value)?;
    T::try_from(wide).map_err(|_| format!("{wide} is out of range for {dtype}"))
}

/// Decode a bool.
///
/// # Errors
///
/// A message for any other JSON value.
pub fn decode_bool(value: &Value) -> Result<bool, String> {
    value
        .as_bool()
        .ok_or_else(|| format!("expected a bool, found {value}"))
}

/// Decode a string.
///
/// # Errors
///
/// A message for any other JSON value.
pub fn decode_string(value: &Value) -> Result<String, String> {
    value
        .as_str()
        .map(str::to_owned)
        .ok_or_else(|| format!("expected a string, found {value}"))
}

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::json;

    #[test]
    fn non_finite_floats_are_the_three_strings_and_read_back() {
        assert_eq!(encode_f64(f64::NAN), json!("NaN"));
        assert_eq!(encode_f64(f64::INFINITY), json!("Infinity"));
        assert_eq!(encode_f64(f64::NEG_INFINITY), json!("-Infinity"));
        assert_eq!(encode_f64(1.5), json!(1.5));
        assert!(decode_f64(&json!("NaN")).unwrap().is_nan());
        assert_eq!(decode_f64(&json!("Infinity")).unwrap(), f64::INFINITY);
        assert_eq!(decode_f64(&json!("-Infinity")).unwrap(), f64::NEG_INFINITY);
        assert_eq!(decode_f64(&json!(3)).unwrap(), 3.0);
        assert!(decode_f64(&Value::Null).is_err());
        assert!(decode_f64(&json!("nan")).is_err());
        assert!(decode_f64(&json!("1.5")).is_err());
    }

    #[test]
    fn integers_beyond_two_to_the_53_are_decimal_strings() {
        assert_eq!(encode_u64(SAFE_INTEGER), json!(SAFE_INTEGER));
        assert_eq!(encode_u64(SAFE_INTEGER + 1), json!("9007199254740993"));
        assert_eq!(encode_u64(u64::MAX), json!("18446744073709551615"));
        assert_eq!(encode_i64(i64::MIN), json!("-9223372036854775808"));
        assert_eq!(encode_i64(-5), json!(-5));
        for v in [0, 1, SAFE_INTEGER, SAFE_INTEGER + 1, u64::MAX] {
            assert_eq!(decode_u64(&encode_u64(v)).unwrap(), v);
        }
        for v in [i64::MIN, -1, 0, i64::MAX] {
            assert_eq!(decode_i64(&encode_i64(v)).unwrap(), v);
        }
        // An exact JSON integer beyond 2^53 is accepted too.
        assert_eq!(decode_u64(&json!(u64::MAX)).unwrap(), u64::MAX);
    }

    #[test]
    fn integers_refuse_reals_signs_ranges_and_junk() {
        assert!(decode_i64(&json!(1.5)).is_err());
        assert!(decode_i64(&json!(1.0)).is_err());
        assert!(decode_u64(&json!(-1)).is_err());
        assert!(decode_u64(&json!("-1")).is_err());
        assert!(decode_i64(&json!("+1")).is_err());
        assert!(decode_i64(&json!("1e3")).is_err());
        assert!(decode_i64(&json!(true)).is_err());
        assert!(decode_i64(&Value::Null).is_err());
        assert!(decode_i64(&json!(u64::MAX)).is_err());
        assert!(decode_signed::<i32>(&json!(1_i64 << 40), "i32").is_err());
        assert_eq!(decode_signed::<i32>(&json!(-7), "i32").unwrap(), -7);
        assert!(decode_unsigned::<u32>(&json!(1_u64 << 33), "u32").is_err());
    }
}
